//! Determine a checkpoint's chat template by generating under each candidate.
//!
//! Tier 4 of `api::chat_template`'s resolution order, and the only tier that is
//! an operator action rather than a startup step: it costs real generations and
//! its conclusion is **persisted** at the highest-authority tier, so a wrong
//! answer that nobody looked at is worse than no answer at all. See the design
//! spec's §4.
//!
//! # Why a sibling binary and not a `lightbulb-cli` subcommand
//!
//! `src/bin/lightbulb-cli.rs` contains zero `lightbulb::` references — it is a
//! pure HTTP client that talks to a running server. The probe needs in-process
//! inference, so putting it there would put the engine's link requirements
//! (CUDA among them) onto a tool whose whole job is to talk to a server that
//! already has them. `src/main.rs` already links the library, so this binary
//! has exactly the server's dependency footprint and adds no new link risk.
//!
//! # One generation per candidate, not N trials
//!
//! An earlier design generated `--trials 8` times per candidate and reported an
//! EOS *rate*. That cannot vary on the shipped default build: the only readers
//! of `temperature` are the `fuel-engine` path's, and the default backend
//! decodes greedily (`ParallelModelManager`, `logits_slice.argmax(0)`), so
//! eight trials of one prompt are the same generation eight times and every row
//! could only ever read `0/8` or `8/8`.
//!
//! Greedy decoding is the feature here, not an obstacle. One generation per
//! candidate is deterministic and reproducible, so the table below reports a
//! fact rather than a sample. The confirmation gate stays, because the risk it
//! guards was never sampling noise — it is that a probe over-fits its one
//! prompt, which is just as true at N=1.
//!
//! # The automatic rule is inert; `--accept` is the path that works
//!
//! `select_probe_winner` acts only on a board where **exactly one** candidate
//! stopped on EOS. That has now been measured to carry no information, on two
//! checkpoints picked to be as unalike as the question allows.
//!
//! TinyLlama-1.1B-Chat, whose turn markers are ordinary text, on `CONVERSATION`:
//!
//! ```text
//! zephyr      yes   14   Japan. Tokyo is the capital and largest city of Japan.</s>
//! chatml      yes   10   Tokyo, the capital of Japan.</s>
//! llama2      yes    3   Tokyo.</s>
//! ```
//!
//! SmolLM2-360M-Instruct, whose `<|im_start|>`/`<|im_end|>` are genuine added
//! tokens and which ships its own ChatML template — so its ground truth is known
//! independently of anything measured here:
//!
//! ```text
//! zephyr      yes    8   The capital of Japan is Tokyo.<|im_end|>
//! chatml      yes    4   Tokyo.<|im_end|>
//! llama2      yes   27   Nativescriptscript
//! ```
//!
//! All three candidates fired on both, so the probe refused both as ties. The
//! EOS column is constant across a board whose ground truth is not: the flag
//! does not discriminate. **The output column does, immediately** — on SmolLM2
//! the ground-truth-correct `chatml` gave the crispest correct answer and
//! `llama2` produced obvious garbage.
//!
//! Token count is not the missing rule and must not become one: SmolLM2's
//! shortest answer is the correct candidate and TinyLlama's shortest is the
//! wrong one, so a length heuristic would be right once and confidently wrong
//! once — which is worse than refusing, because it persists at
//! `Resolution::Probe`.
//!
//! So `--accept <candidate>` hands the reading of that column to the operator.
//! The automatic rule stays, unchanged, for when `--accept` is absent: it is
//! safe-but-inert — its refusals mean it cannot pick wrong — and deleting a
//! tested safety rule to buy nothing is not a simplification. All the
//! generations still run under `--accept`, because the `evidence` the sidecar
//! records has to describe a board this run actually observed, not one the
//! operator remembers.
//!
//! # It must not overwrite an answer better than the one it produces
//!
//! Measured 2026-08-13 on SmolLM2-360M-Instruct, which ships its own
//! `tokenizer_config.json` and therefore already resolved at
//! `Resolution::TokenizerConfig`:
//!
//! ```text
//! lightbulb-probe <dir> --accept chatml --yes
//! → wrote lightbulb-chat-template.json … will now resolve at Resolution::Probe
//! → exit 0
//! ```
//!
//! The probe replaced the checkpoint's own authoritative template with a
//! strictly worse approximation and reported success. Diffing the two
//! afterwards: SmolLM2's template injects a default system message ("You are a
//! helpful AI assistant named SmolLM, trained by Hugging Face") when the caller
//! supplies none; `registry::CHATML` does not. `resolve` reads the sidecar
//! FIRST, so the server would have prompted this model differently — and worse —
//! than before the probe ran, indefinitely. Nothing warned.
//!
//! `probe_override_check` is that refusal, and the whole table of what each
//! current tier earns is on it: refuse over the checkpoint's own template or an
//! existing sidecar, warn over a guess, proceed over nothing. `--force` turns
//! the two refusals into warnings and does nothing else — in particular it
//! bypasses none of the refusals numbered here and does not imply `--yes`.
//!
//! # Every measurement constant is fixed, deliberately
//!
//! The prompt, the token budget, the temperature, the `add_special_tokens`
//! flag, the runner's batch size and context length are all pinned to what
//! `tests/chat_template_e2e.rs` uses, so a probe result is comparable with what
//! the server will then do with the template it writes. Making any of them a
//! flag would make two probe runs incomparable without saying so.

use std::io::Write as _;
use std::path::Path;

use anyhow::{anyhow, bail};
use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use lightbulb::api::chat_template::{
    ChatTemplate, ProbeOverride, ProbeRow, Resolution, Sidecar, fingerprint, first_line_of,
    format_probe_report, probe_override_check, registry, resolve, select_named_candidate,
    select_probe_winner, sidecar_path, special_tokens, write_sidecar,
};
use lightbulb::contracts::validation::RawMessage;
use lightbulb::engine::ModelRunner;
use lightbulb::engine::model_runner::{
    CompletionResult, FinishReason, InferenceJob, InferenceRequestSender, ResponseMode,
};

/// The one conversation every candidate is measured on.
///
/// **Multi-turn, and that is the whole point.** The first live run used a
/// single user message ("Name the capital of France.") and all three candidates
/// fired EOS — zephyr and chatml byte-identically — so the probe refused as a
/// tie and measured nothing. A well-behaved chat model answers a one-turn
/// factual question correctly under *any* plausible formatting, so EOS carries
/// almost no signal there.
///
/// A wrong template's characteristic damage is to **turn boundaries**: it drops
/// a role marker, merges two turns, or attributes the assistant's reply to the
/// user. A one-message conversation has no boundaries to damage. Three turns
/// have two, and the model's own prior answer sits inside the prompt where a
/// mangled template will run it together with the new question.
///
/// The first turn is the prompt `tests/chat_template_e2e.rs` sends, so the
/// opening of a probe row is still comparable with an e2e result.
const CONVERSATION: [(&str, &str); 3] = [
    ("user", "Name the capital of France."),
    ("assistant", "Paris."),
    ("user", "And of Japan?"),
];

/// The generation budget, per candidate.
///
/// The EOS rate is a function of the budget: too small and a correct template
/// looks like a failure because it simply had not finished. 48 is ~6× the
/// 8-token answer TinyLlama gives to `CONVERSATION`'s first turn under its own
/// template.
const MAX_NEW_TOKENS: usize = 48;

#[derive(Parser, Debug)]
#[command(name = "lightbulb-probe")]
#[command(about = "Determine a checkpoint's chat template by generating under each candidate")]
struct Args {
    /// Checkpoint directory, or a single .gguf file.
    model_dir: std::path::PathBuf,
    /// Write the sidecar without the interactive confirmation. Does NOT bypass
    /// any refusal below — a flag that suppresses a refusal is a flag that
    /// writes a wrong answer unattended.
    #[arg(long)]
    yes: bool,
    /// Record this candidate, overriding the EOS rule. The operator reads the
    /// table and names the winner; see the module docs for why that column and
    /// not the EOS one. Every generation still runs, and every refusal other
    /// than the EOS rule still applies.
    #[arg(long, value_name = "CANDIDATE")]
    accept: Option<String>,
    /// Probe a checkpoint that ALREADY resolves at a tier the probe cannot
    /// improve on — `Resolution::TokenizerConfig` (the checkpoint's own
    /// chat_template) or an existing sidecar. Narrowly that, and nothing else:
    /// it does not bypass the nonexistent-path, empty-EOS, unfingerprintable or
    /// selection refusals, it does not imply `--yes`, and it is a no-op on a
    /// checkpoint whose current answer is a guess (those warn either way). See
    /// `probe_override_check` for the measurement this exists because of.
    #[arg(long)]
    force: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Before anything that might log. Without a global subscriber every
    // `tracing::warn!` in the library is a no-op — including the one in
    // `special_tokens` saying a checkpoint declares no BOS/EOS. `ModelRunner`'s
    // own `println!`/`eprintln!` would still appear, so the output would look
    // healthy while the single warning that invalidates the measurement was
    // silently dropped.
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,lightbulb=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let args = Args::parse();
    probe(
        &args.model_dir,
        args.yes,
        args.accept.as_deref(),
        args.force,
    )
    .await
}

async fn probe(
    model_dir: &Path,
    assume_yes: bool,
    accept: Option<&str>,
    force: bool,
) -> anyhow::Result<()> {
    // Refusal -1, FIRST. Without this the most common operator error — a typo
    // in the path — gets the one diagnosis guaranteed to be wrong.
    // `special_tokens` swallows all three of its file reads, so a nonexistent
    // directory returns `bos: "", eos: ""` and falls into the next refusal,
    // telling the operator to add a `tokenizer_config.json` to a directory that
    // does not exist. `ModelRunner::start` has no existence check either
    // (`src/api/mod.rs` guards with `model_path.exists()` before calling it;
    // nothing else does), so without this the failure arrives from the model
    // loader and reads as a runner fault rather than a bad argument.
    if !model_dir.exists() {
        bail!("{} does not exist", model_dir.display());
    }

    // Refusal 0: an empty EOS makes the measurement meaningless, so do not
    // spend three model runs discovering it.
    let tokens = special_tokens(model_dir);
    if tokens.eos.is_empty() {
        bail!(
            "{} declares no EOS token, so this probe cannot measure anything: ZEPHYR and LLAMA2 \
             both interpolate `eos_token` and would render with no end-of-turn marker at all, \
             while CHATML — which references neither — would be unaffected. The comparison would \
             be rigged, not close. Add a tokenizer_config.json with eos_token, or a config.json \
             with eos_token_id WHOSE ID APPEARS IN tokenizer.json's added_tokens. Nothing was \
             written.",
            model_dir.display()
        );
    }

    // Refusal 0.5: do not persist an answer over one that is already better
    // than anything this probe can produce. Measured on SmolLM2-360M-Instruct,
    // which ships its own template and got it replaced by `registry::CHATML` —
    // a strictly worse approximation — with an exit 0 and no warning; the whole
    // account is on `probe_override_check`.
    //
    // Numbered 0.5 rather than renumbering the refusals below it: 1 and 2 are
    // `select_probe_winner`'s, named as such in its docs and in the tests.
    //
    // HERE, before `ModelRunner::start`, for refusal 0's reason — the verdict
    // reads files and nothing else, so an operator who pointed the probe at the
    // wrong checkpoint learns it in milliseconds rather than after three
    // generations. `--force` turns the refusal into a warning and changes
    // nothing else: every refusal above and below still applies to it, and it
    // does not imply `--yes`.
    match probe_override_check(resolve(model_dir).resolved_by, force) {
        ProbeOverride::Refuse(why) => bail!("{}: {why}", model_dir.display()),
        // `tracing::warn!`, so it carries the WARN level and the timestamp that
        // make it legible in a log the operator scrolls back through after the
        // generations have printed their own noise.
        ProbeOverride::Warn(why) => tracing::warn!("{}: {why}", model_dir.display()),
        ProbeOverride::Proceed => {}
    }

    let messages: Vec<RawMessage> = CONVERSATION
        .iter()
        .map(|(role, content)| RawMessage {
            role: role.to_string(),
            content: content.to_string(),
        })
        .collect();

    // One runner, reused across candidates: loading the checkpoint three times
    // would triple the wall clock and change nothing about the result. The
    // arguments match every test call site exactly (`dtype: None` parses to F32
    // as well, but gratuitously differently).
    let tx = ModelRunner::start(model_dir, 1, 512, Some("f32".to_string()))?;

    let mut rows: Vec<ProbeRow> = Vec::new();
    for (candidate, source) in registry::candidates() {
        let template = ChatTemplate {
            source: source.to_string(),
            resolved_by: Resolution::Probe,
        };

        // A candidate whose render FAILS is not a candidate that lost. Record
        // why, so the operator sees the parse error rather than a silent zero.
        let prompt = match template.render(&messages, &tokens.bos, &tokens.eos) {
            Ok(p) => p,
            Err(e) => {
                rows.push(ProbeRow {
                    candidate,
                    source,
                    stopped_on_eos: false,
                    tokens_generated: 0,
                    first_line: first_line_of(&format!("render failed: {e:#}")),
                });
                continue;
            }
        };

        // `?`, not a losing row. An `Err` here means the RUNNER failed, not the
        // template: `ModelRunner::start` returns `Ok(tx)` even when the model
        // fails to load and then answers every job with
        // `Err("model load failed: …")`, so this is exactly where a broken
        // checkpoint announces itself. Mapping it to `stopped_on_eos: false`
        // would burn three model runs to conclude "no template matched" about a
        // model that never loaded, discarding a precise diagnosis the runner
        // already handed us.
        let result = generate(&tx, prompt).await?;
        rows.push(ProbeRow {
            candidate,
            source,
            stopped_on_eos: result.finish_reason == FinishReason::Stop,
            tokens_generated: result.completion_tokens,
            first_line: first_line_of(&result.text),
        });
    }

    // Printed before any refusal below, so an operator always gets the board
    // even when the probe declines to act on it.
    print!("\n{}", format_probe_report(&rows));
    std::io::stdout().flush()?;

    // Both selection rules live in the library, with their refusal messages,
    // because a `match` in `src/bin/*.rs` is unreachable from any test — see
    // those functions and `tests/chat_template_render.rs`.
    //
    // `--accept` is checked HERE, after every generation, and is deliberately
    // not pre-validated against `registry::candidates()` up at argument
    // parsing. A typo would then be caught in two places by two copies of one
    // rule, which is the mechanism by which the two drift; and the operator
    // has to see the board anyway before their name means anything.
    let winner: &ProbeRow = match accept {
        Some(name) => select_named_candidate(&rows, name)?,
        None => select_probe_winner(&rows)?,
    };

    // Refusal 3. `fingerprint` is `Option`, and it fails closed on purpose —
    // reaching for `unwrap_or_default()` here is the bug that made every
    // unidentifiable checkpoint hash to one constant, so a foreign sidecar was
    // served at `Resolution::Sidecar` carrying evidence claiming it had been
    // measured on this model.
    let Some(fp) = fingerprint(model_dir) else {
        bail!(
            "cannot fingerprint {} — a directory checkpoint needs a readable config.json; a .gguf \
             file needs to be readable itself — so a sidecar written here could not later be \
             checked against the checkpoint it describes. Nothing was written.",
            model_dir.display()
        );
    };

    // `--yes` skips only this. Every refusal above still applies to it.
    let target = sidecar_path(model_dir);
    if !assume_yes && !confirm(&target, winner.candidate)? {
        println!("Nothing was written.");
        return Ok(());
    }

    let sc = Sidecar {
        // `winner.source`, carried on the row — never re-looked-up from
        // `registry::candidates()` by name.
        template: winner.source.to_string(),
        resolved_by: Resolution::Probe,
        // One prose line, matching the shape every existing sidecar fixture
        // uses — not the whole table with its header stuffed into a JSON
        // string.
        //
        // The two paths say different things on purpose. This file is read at
        // `Resolution::Probe`, above every other tier, on every later server
        // start, and "a measurement selected this" and "a human judged this"
        // are different epistemic claims: the first is reproducible from the
        // checkpoint, the second is not. The operator line therefore carries
        // the whole board it was judged from — including each candidate's
        // output, which is where the information actually was — so the next
        // reader can audit the call instead of taking it on faith. Still one
        // line: `first_line_of` already guarantees every cell is.
        evidence: match accept {
            // `winner.candidate`, not the raw `--accept` argument, for the same
            // reason `winner.source` is used above: what is recorded is the row
            // that was selected, never a string that merely ought to match it.
            Some(_) => format!(
                "probe: operator selected `{}` with --accept after reviewing this board: {}",
                winner.candidate,
                rows.iter()
                    .map(|r| format!(
                        "{} (EOS {}, {} tokens) {:?}",
                        r.candidate,
                        if r.stopped_on_eos { "yes" } else { "no" },
                        r.tokens_generated,
                        r.first_line
                    ))
                    .collect::<Vec<_>>()
                    .join("; ")
            ),
            None => format!(
                "probe: {} stopped on EOS in {} tokens; {} did not",
                winner.candidate,
                winner.tokens_generated,
                rows.iter()
                    .filter(|r| !r.stopped_on_eos)
                    .map(|r| r.candidate)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        },
        resolved_at: chrono::Utc::now().to_rfc3339(),
        model_fingerprint: fp,
    };
    // Propagated, never logged-and-ignored: a failed write must not report
    // success, or the next server start silently resolves at a lower tier while
    // the operator believes the question is settled.
    write_sidecar(model_dir, &sc)?;
    println!(
        "wrote {} — {} will now resolve at Resolution::Probe",
        target.display(),
        model_dir.display()
    );
    Ok(())
}

/// Ask before writing.
///
/// **EOF and an empty line are NO.** Run from a script with no stdin,
/// `read_line` returns `Ok(0)` and leaves the buffer empty; an implementation
/// that treats either as "yes" writes the sidecar unattended, which is the
/// exact thing `--yes` exists to make deliberate.
fn confirm(target: &Path, candidate: &str) -> anyhow::Result<bool> {
    print!(
        "\nWrite {}, recording `{}` as this checkpoint's chat template? [y/N] ",
        target.display(),
        candidate
    );
    std::io::stdout().flush()?;

    let mut line = String::new();
    if std::io::stdin().read_line(&mut line)? == 0 {
        println!();
        return Ok(false);
    }
    let answer = line.trim().to_ascii_lowercase();
    Ok(answer == "y" || answer == "yes")
}

/// One complete-mode generation.
///
/// A re-implementation of `openai/chat.rs`'s `run_inference_once`, which is
/// `pub(crate)` and therefore unreachable from here: `src/bin/*.rs` is a
/// separate crate. This is the one place the chat-template work knowingly
/// accepts a second copy, because the crate boundary leaves no alternative.
async fn generate(tx: &InferenceRequestSender, prompt: String) -> anyhow::Result<CompletionResult> {
    let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
    let job = InferenceJob {
        id: uuid::Uuid::new_v4().to_string(),
        prompt,
        max_new_tokens: MAX_NEW_TOKENS,
        // Greedy. Immaterial on the default backend, which argmaxes regardless,
        // but it is what makes the `fuel-engine` build measure the same thing.
        temperature: 0.0,
        // What the server passes for every `ResolvedTemplate::render` output,
        // and matching it is the entire reason a probe result is comparable
        // with what the server will then do. NOT justified by "the template
        // already emits BOS" — none of these three candidates interpolates
        // `bos_token`, so these prompts carry no BOS at all. That is a
        // difference from the general templated case, not a reason to flip the
        // flag: `true` would add a BOS the template did not ask for.
        add_special_tokens: false,
        response_mode: ResponseMode::Complete(resp_tx),
    };
    tx.send(job)
        .map_err(|e| anyhow!("failed to enqueue inference job: {e}"))?;
    // Two `Result` layers. The outer is the channel's — the runner thread died.
    // The inner is the runner's own verdict and is returned as-is; see the call
    // site for why it aborts the probe rather than scoring a row.
    resp_rx
        .await
        .map_err(|e| anyhow!("inference runner dropped: {e}"))?
}
