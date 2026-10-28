// SPDX-License-Identifier: MIT OR Apache-2.0
//! Lightbulb CLI - Chat with models running on Lightbulb server
//!
//! Usage:
//!   lightbulb-cli --api-key <key> --url <base-url> [--stream]

use clap::Parser;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::io::{self, Write};

/// Lightbulb CLI for interacting with the API server
#[derive(Parser, Debug)]
#[command(name = "lightbulb-cli")]
#[command(about = "Chat with models running on Lightbulb", long_about = None)]
struct Args {
    /// API key for authentication
    #[arg(long)]
    api_key: Option<String>,

    /// Base URL of the Lightbulb server
    #[arg(long, default_value = "http://localhost:8080")]
    url: String,

    /// Enable streaming mode
    #[arg(long, short = 's')]
    stream: bool,

    /// Model to use
    #[arg(long, short = 'm', default_value = "default")]
    model: String,

    /// System prompt
    #[arg(long)]
    system: Option<String>,

    /// Temperature (0.0 - 2.0)
    #[arg(long, default_value = "0.7")]
    temperature: f32,

    /// Maximum tokens to generate
    #[arg(long, default_value = "512")]
    max_tokens: u32,
}

#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    stream: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct Message {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    id: String,
    choices: Vec<Choice>,
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: Message,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct StreamChunk {
    id: String,
    choices: Vec<StreamChoice>,
}

#[derive(Debug, Deserialize)]
struct StreamChoice {
    delta: Delta,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Delta {
    #[serde(default)]
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(Debug, Deserialize)]
struct ErrorDetail {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Get API key from arg or environment
    let api_key = args
        .api_key
        .or_else(|| std::env::var("LIGHTBULB_API_KEY").ok())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "API key required. Provide via --api-key or LIGHTBULB_API_KEY environment variable"
            )
        })?;

    println!("🔦 Lightbulb CLI");
    println!("Connected to: {}", args.url);
    println!(
        "Streaming: {}",
        if args.stream { "enabled" } else { "disabled" }
    );
    println!();

    let client = reqwest::Client::new();
    let mut conversation: Vec<Message> = Vec::new();

    // Add system message if provided
    if let Some(system_prompt) = &args.system {
        conversation.push(Message {
            role: "system".to_string(),
            content: system_prompt.clone(),
        });
    }

    // Interactive chat loop
    loop {
        print!("You: ");
        io::stdout().flush()?;

        let mut user_input = String::new();
        io::stdin().read_line(&mut user_input)?;
        let user_input = user_input.trim();

        if user_input.is_empty() {
            continue;
        }

        // Exit commands
        if user_input.eq_ignore_ascii_case("exit")
            || user_input.eq_ignore_ascii_case("quit")
            || user_input.eq_ignore_ascii_case("q")
        {
            println!("Goodbye! 👋");
            break;
        }

        // Clear conversation
        if user_input.eq_ignore_ascii_case("clear") {
            conversation.clear();
            println!("Conversation cleared.");
            continue;
        }

        // Add user message to conversation
        conversation.push(Message {
            role: "user".to_string(),
            content: user_input.to_string(),
        });

        // Create request
        let request = ChatCompletionRequest {
            model: args.model.clone(),
            messages: conversation.clone(),
            temperature: Some(args.temperature),
            max_tokens: Some(args.max_tokens),
            stream: args.stream,
        };

        print!("Assistant: ");
        io::stdout().flush()?;

        if args.stream {
            // Streaming mode
            match stream_chat(&client, &args.url, &api_key, &request).await {
                Ok(content) => {
                    println!(); // New line after streaming
                    conversation.push(Message {
                        role: "assistant".to_string(),
                        content,
                    });
                }
                Err(e) => {
                    eprintln!("\n❌ Error: {}", e);
                    conversation.pop(); // Remove the user message
                }
            }
        } else {
            // Non-streaming mode
            match non_stream_chat(&client, &args.url, &api_key, &request).await {
                Ok((content, usage)) => {
                    println!("{}", content);
                    if let Some(usage) = usage {
                        println!(
                            "\n[Tokens: {} prompt + {} completion = {} total]",
                            usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
                        );
                    }
                    conversation.push(Message {
                        role: "assistant".to_string(),
                        content,
                    });
                }
                Err(e) => {
                    eprintln!("❌ Error: {}", e);
                    conversation.pop(); // Remove the user message
                }
            }
        }

        println!();
    }

    Ok(())
}

async fn non_stream_chat(
    client: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    request: &ChatCompletionRequest,
) -> anyhow::Result<(String, Option<Usage>)> {
    let url = format!("{}/v1/chat/completions", base_url);

    let response = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await?;

    let status = response.status();
    let body = response.text().await?;

    if !status.is_success() {
        // Try to parse error response
        if let Ok(error_resp) = serde_json::from_str::<ErrorResponse>(&body) {
            anyhow::bail!(
                "{} ({})",
                error_resp.error.message,
                error_resp.error.error_type
            );
        } else {
            anyhow::bail!("HTTP {}: {}", status, body);
        }
    }

    let completion: ChatCompletionResponse = serde_json::from_str(&body)?;

    let content = completion
        .choices
        .first()
        .ok_or_else(|| anyhow::anyhow!("No choices in response"))?
        .message
        .content
        .clone();

    Ok((content, completion.usage))
}

async fn stream_chat(
    client: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    request: &ChatCompletionRequest,
) -> anyhow::Result<String> {
    let url = format!("{}/v1/chat/completions", base_url);

    let response = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await?;

    let status = response.status();

    if !status.is_success() {
        let body = response.text().await?;
        if let Ok(error_resp) = serde_json::from_str::<ErrorResponse>(&body) {
            anyhow::bail!(
                "{} ({})",
                error_resp.error.message,
                error_resp.error.error_type
            );
        } else {
            anyhow::bail!("HTTP {}: {}", status, body);
        }
    }

    let mut stream = response.bytes_stream();
    let mut full_content = String::new();
    let mut buffer = String::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        let text = String::from_utf8_lossy(&chunk);

        // SSE format: "data: {...}\n\n"
        buffer.push_str(&text);

        // Process complete SSE messages
        while let Some(pos) = buffer.find("\n\n") {
            let message = buffer[..pos].to_string();
            buffer.drain(..pos + 2);

            if message.starts_with("data: ") {
                let json_str = &message[6..]; // Skip "data: "

                if json_str.trim() == "[DONE]" {
                    break;
                }

                if let Ok(chunk_data) = serde_json::from_str::<StreamChunk>(json_str) {
                    if let Some(choice) = chunk_data.choices.first() {
                        if let Some(content) = &choice.delta.content {
                            print!("{}", content);
                            io::stdout().flush()?;
                            full_content.push_str(content);
                        }
                    }
                }
            }
        }
    }

    Ok(full_content)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Json, Router, response::sse::Event, routing::post};
    use std::convert::Infallible;

    /// Spawn a real local server on an OS-assigned port and return its base
    /// URL. `non_stream_chat`/`stream_chat` are plain HTTP clients taking a
    /// base URL -- no mocking library needed, axum is already a dependency.
    async fn spawn_server(router: Router) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("binding a loopback port");
        let addr = listener.local_addr().expect("local_addr");
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        format!("http://{addr}")
    }

    fn sample_request() -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "test-model".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: "hi".to_string(),
            }],
            temperature: Some(0.7),
            max_tokens: Some(16),
            stream: false,
        }
    }

    #[tokio::test]
    async fn non_stream_chat_returns_the_assistant_content_and_usage() {
        let router = Router::new().route(
            "/v1/chat/completions",
            post(|| async {
                Json(serde_json::json!({
                    "id": "chatcmpl-1",
                    "choices": [{
                        "message": {"role": "assistant", "content": "Paris."},
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
                }))
            }),
        );
        let base_url = spawn_server(router).await;
        let client = reqwest::Client::new();

        let (content, usage) = non_stream_chat(&client, &base_url, "test-key", &sample_request())
            .await
            .expect("the mock server's response must parse");

        assert_eq!(content, "Paris.");
        let usage = usage.expect("usage must be present");
        assert_eq!(usage.prompt_tokens, 5);
        assert_eq!(usage.completion_tokens, 2);
        assert_eq!(usage.total_tokens, 7);
    }

    /// The error-response path: a non-2xx with the server's own
    /// `{"error": {"message", "type"}}` shape must surface that message,
    /// not a generic "HTTP 429" string.
    #[tokio::test]
    async fn non_stream_chat_surfaces_the_servers_error_message() {
        let router = Router::new().route(
            "/v1/chat/completions",
            post(|| async {
                (
                    axum::http::StatusCode::TOO_MANY_REQUESTS,
                    Json(serde_json::json!({
                        "error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}
                    })),
                )
            }),
        );
        let base_url = spawn_server(router).await;
        let client = reqwest::Client::new();

        let err = non_stream_chat(&client, &base_url, "test-key", &sample_request())
            .await
            .expect_err("a 429 must be an Err, not a parsed response");

        assert!(
            err.to_string().contains("Rate limit exceeded"),
            "error must carry the server's own message, got: {err}"
        );
    }

    #[tokio::test]
    async fn stream_chat_concatenates_token_deltas_and_stops_at_the_done_sentinel() {
        let router = Router::new().route(
            "/v1/chat/completions",
            post(|| async {
                let frames = [
                    serde_json::json!({
                        "id": "chatcmpl-1",
                        "choices": [{"delta": {"content": "Par"}, "finish_reason": null}]
                    }),
                    serde_json::json!({
                        "id": "chatcmpl-1",
                        "choices": [{"delta": {"content": "is."}, "finish_reason": null}]
                    }),
                ];
                let events = frames
                    .into_iter()
                    .map(|f| Ok::<_, Infallible>(Event::default().data(f.to_string())))
                    .chain(std::iter::once(Ok(Event::default().data("[DONE]"))));
                axum::response::Sse::new(futures_util::stream::iter(events))
            }),
        );
        let base_url = spawn_server(router).await;
        let client = reqwest::Client::new();

        let content = stream_chat(&client, &base_url, "test-key", &sample_request())
            .await
            .expect("the mock SSE stream must parse");

        assert_eq!(content, "Paris.");
    }

    #[tokio::test]
    async fn stream_chat_surfaces_the_servers_error_message() {
        let router = Router::new().route(
            "/v1/chat/completions",
            post(|| async {
                (
                    axum::http::StatusCode::SERVICE_UNAVAILABLE,
                    Json(serde_json::json!({
                        "error": {"message": "inference queue is full", "type": "server_busy"}
                    })),
                )
            }),
        );
        let base_url = spawn_server(router).await;
        let client = reqwest::Client::new();

        let err = stream_chat(&client, &base_url, "test-key", &sample_request())
            .await
            .expect_err("a 503 must be an Err");

        assert!(
            err.to_string().contains("inference queue is full"),
            "error must carry the server's own message, got: {err}"
        );
    }
}
