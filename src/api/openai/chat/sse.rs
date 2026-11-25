// SPDX-License-Identifier: MIT OR Apache-2.0
//! Streaming (SSE) chat completions.

use super::*;

/// Create streaming chat completion.
///
/// Returns a full [`Response`] rather than a bare stream so the enqueue
/// attempt — which can fail with a real HTTP status (503 when the queue is
/// full) before any SSE frame is sent — can report that status instead of
/// the 200 an already-constructed `Sse` response is stuck with.
pub(super) fn create_chat_stream(
    state: AppState,
    request: ChatCompletionRequest,
    priority: crate::engine::Priority,
) -> Response {
    // The SAME prompt construction as the non-streaming path. These two sites
    // drifting apart is not hypothetical: this one was the second of the two
    // `role: content` joins, and it is the one a fix aimed at
    // `create_chat_completion` leaves behind — the endpoint looks correct,
    // returns 200, and prompts a chat model like a base model on every
    // `"stream": true` request.
    let prompt = build_prompt(&state, &request.messages);

    let max_new_tokens = request.max_tokens.unwrap_or(100);
    let model = request.model.clone();

    // Cloned out of `state` BEFORE `state.inference_tx` is moved below, and
    // before the token stream — which is `'static` and outlives this function —
    // captures them. The streaming path observing its own completions is the
    // point: it is the path a fix aimed at `create_chat_completion` leaves
    // behind, and a monitor fed by only one of the two never fills its window
    // on a server whose clients stream.
    let eos_monitor = state.eos_monitor.clone();
    let model_entry = state.model();
    let resolved_by = model_entry.chat_template.as_ref().map(|t| t.resolved_by());
    // Copied out before `state.inference_tx` is moved below, same reason as
    // `eos_monitor`/`resolved_by` above. Sourcery review of #155: the
    // non-streaming timeout had no streaming counterpart at all -- a
    // stalled runner left an SSE client waiting forever, the exact hang
    // this PR otherwise fixes.
    let timeout_ms = state.config.request_timeout_ms;
    let include_usage = request
        .stream_options
        .as_ref()
        .is_some_and(|o| o.include_usage);
    // Same reason as the other clones above.
    let queue_depth = state.queue_depth.clone();

    // If a model runner is available, create a streaming job
    if let Some(tx) = model_entry.inference_tx {
        let (stream_tx, stream_rx) = tokio::sync::mpsc::unbounded_channel();

        let job = crate::engine::InferenceJob {
            id: uuid::Uuid::new_v4().to_string(),
            prompt: prompt.text,
            max_new_tokens,
            temperature: request.temperature as f64,
            seed: request.seed.map(|s| s as u64),
            extras: crate::engine::model_runner::SamplingExtras {
                logprobs: requested_logprobs(&request),
                ..sampling_extras(
                    request.top_p,
                    request.presence_penalty,
                    request.frequency_penalty,
                    &request.extra_sampling,
                )
            },
            // Same value the non-streaming path sends, because it comes from
            // the same `build_prompt` call. The two paths carrying separate
            // answers to this is the shape of the defect this file already has
            // history with.
            add_special_tokens: prompt.add_special_tokens,
            priority,
            response_mode: crate::engine::model_runner::ResponseMode::Streaming(stream_tx),
        };

        // Counts this job from here until `StreamState` (below, which holds
        // this guard) is dropped -- whether the stream finishes normally,
        // the client disconnects early, or `try_send` below fails and the
        // job never really entered the queue (same reasoning as
        // `run_inference_once`'s guard).
        let queue_depth_guard = crate::engine::model_runner::QueueDepthGuard::new(queue_depth);

        // Send the job to the model runner thread
        use std::sync::mpsc::TrySendError;
        match tx.try_send(job) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(serde_json::json!({
                        "error": {
                            "message": "inference queue is full; try again shortly",
                            "type": "server_busy"
                        }
                    })),
                )
                    .into_response();
            }
            Err(TrySendError::Disconnected(_)) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": {
                            "message": "Failed to enqueue inference job",
                            "type": "internal_error"
                        }
                    })),
                )
                    .into_response();
            }
        }

        // Create stream from the receiver using futures combinators
        let created = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let chat_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());

        use crate::engine::model_runner::StreamItem;

        // Per-chunk idle timeout: each step waits at most `timeout_ms` for
        // the *next* item, not once per whole response -- a stream that is
        // actively producing tokens, just slowly, is not what this guards
        // against; a runner that has stalled entirely is.
        //
        // Built with `stream::unfold` driving `tokio::time::timeout`
        // directly over `recv()`, rather than `tokio_stream::StreamExt::
        // timeout()` composed with `.scan()`: that combinator pairing,
        // once wrapped in `Sse::new(...)` and driven through axum's SSE
        // body, silently stopped firing its timeout (observed: the 20ms
        // idle timeout in a test never elapsed until the producer thread's
        // own unrelated 5s sleep woke the task some other way) -- isolated
        // with a battery of throwaway diagnostic tests that polled the
        // same combinator directly (always correct) versus through
        // `Sse`+`to_bytes` (reproduced the stall, `scan` specifically; a
        // `.take()/.map()` stand-in through the same `Sse` wrapper was
        // fine). `unfold` sidesteps whatever that interaction was by never
        // composing a tokio_stream `Timeout` with `scan` in the first
        // place -- each step is one plain `tokio::time::timeout` future.
        struct StreamState {
            rx: tokio::sync::mpsc::UnboundedReceiver<anyhow::Result<StreamItem>>,
            is_first: bool,
            done: bool,
            // The `include_usage` chunk, held back one step so it follows the
            // `finish_reason` chunk.
            pending_usage: Option<serde_json::Value>,
            // The logprob entry that arrived ahead of its token, attached to that token's chunk.
            pending_logprob: Option<crate::engine::model_runner::TokenLogprob>,
            // Held only so its `Drop` runs when THIS struct is dropped --
            // i.e. when the stream itself ends, whether that's normal
            // completion or the client disconnecting early -- not when
            // `create_chat_stream` (which returns almost immediately after
            // building the response) returns.
            _queue_depth_guard: crate::engine::model_runner::QueueDepthGuard,
        }
        let state = StreamState {
            rx: stream_rx,
            is_first: true,
            done: false,
            pending_usage: None,
            pending_logprob: None,
            _queue_depth_guard: queue_depth_guard,
        };
        let token_stream = stream::unfold(state, move |mut st| {
            let chat_id = chat_id.clone();
            let model = model.clone();
            let eos_monitor = eos_monitor.clone();
            async move {
                if let Some(usage) = st.pending_usage.take() {
                    st.done = true;
                    return Some((sse_event(Event::default().data(usage.to_string())), st));
                }
                if st.done {
                    return None;
                }
                let first = st.is_first;
                st.is_first = false;
                let mut received = tokio::time::timeout(
                    std::time::Duration::from_millis(timeout_ms),
                    st.rx.recv(),
                )
                .await;
                // A logprob entry arrives just ahead of the token it describes; it rides on that token's chunk,
                // so it is held and the token is read next (under the same idle timeout).
                if let Ok(Some(Ok(StreamItem::Logprob(lp)))) = received {
                    st.pending_logprob = Some(lp);
                    received = tokio::time::timeout(
                        std::time::Duration::from_millis(timeout_ms),
                        st.rx.recv(),
                    )
                    .await;
                }
                let event = match received {
                    Err(_elapsed) => {
                        st.done = true;
                        sse_event(
                            Event::default().data(
                                serde_json::json!({
                                    "error": {
                                        "message": "inference request timed out",
                                        "type": "timeout_error"
                                    }
                                })
                                .to_string(),
                            ),
                        )
                    }
                    Ok(None) => return None,
                    // Handled above, where it is held for the token that follows; a second one in a row cannot
                    // happen, and its entry is dropped rather than sent as a chunk of its own.
                    Ok(Some(Ok(StreamItem::Logprob(_)))) => {
                        st.is_first = first;
                        return Some((sse_event(Event::default().comment("")), st));
                    }
                    Ok(Some(Ok(StreamItem::Token(token_text)))) => {
                        let mut choice = serde_json::json!({
                            "index": 0,
                            "delta": {
                                "role": if first { Some("assistant") } else { None },
                                "content": token_text
                            },
                            "finish_reason": None::<String>
                        });
                        if let Some(lp) = st.pending_logprob.take() {
                            choice["logprobs"] = super::super::logprobs::chat_json(&[lp]);
                        }
                        sse_event(
                            Event::default().data(
                                serde_json::json!({
                                    "id": chat_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model,
                                    "choices": [choice]
                                })
                                .to_string(),
                            ),
                        )
                    }
                    Ok(Some(Ok(StreamItem::Done {
                        finish_reason,
                        prompt_tokens,
                        completion_tokens,
                    }))) => {
                        // The stream's one observation of how generation
                        // ended. It is the same value the frame below
                        // reports to the client, read here rather than
                        // re-derived.
                        record_completion(&eos_monitor, resolved_by, finish_reason);
                        if include_usage {
                            st.pending_usage = Some(usage_chunk(
                                &chat_id,
                                created,
                                &model,
                                prompt_tokens,
                                completion_tokens,
                            ));
                        } else {
                            st.done = true;
                        }
                        sse_event(
                            Event::default().data(
                                serde_json::json!({
                                    "id": chat_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {},
                                        "finish_reason": finish_reason.as_str()
                                    }]
                                })
                                .to_string(),
                            ),
                        )
                    }
                    Ok(Some(Err(e))) => {
                        st.done = true;
                        sse_event(
                            Event::default().data(
                                serde_json::json!({
                                    "error": {
                                        "message": e.to_string(),
                                        "type": "inference_error"
                                    }
                                })
                                .to_string(),
                            ),
                        )
                    }
                };
                Some((event, st))
            }
        });

        // OpenAI's real streaming contract ends with a literal `data: [DONE]`
        // line, not just the last content/done frame -- appended after
        // whatever terminal event the stream above produced (`Done` or
        // `Err`), so every exit path gets it exactly once.
        let token_stream = token_stream.chain(stream::once(async {
            sse_event(Event::default().data("[DONE]"))
        }));

        return Sse::new(token_stream.boxed()).into_response();
    }

    // Fallback: no model runner available - return mock stream
    let model_name = request.model.clone();
    let chunks = vec![
        "No ",
        "model ",
        "runner ",
        "available. ",
        "Please ",
        "configure ",
        "model ",
        "path.",
    ];

    let stream = stream::iter(chunks.into_iter().enumerate().map(move |(i, chunk)| {
        let data = serde_json::json!({
            "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            "object": "chat.completion.chunk",
            "created": SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            "model": model_name.clone(),
            "choices": [{
                "index": 0,
                "delta": {
                    "content": chunk
                },
                "finish_reason": if i == 7 { Some("stop") } else { None }
            }]
        });

        sse_event(Event::default().data(data.to_string()))
    }))
    .chain(stream::once(async {
        sse_event(Event::default().data("[DONE]"))
    }))
    .boxed();

    Sse::new(stream).into_response()
}
