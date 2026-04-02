// hitl_hook.rs - lightweight example that simulates uncertainty and optionally invokes the Clarify CLI
use rand::Rng;
use std::env;
use std::process::Command;

fn main() {
    println!("HITL hook example: simulate uncertainty and optionally invoke Clarify CLI");
    let mut rng = rand::thread_rng();

    for step in 1..=5 {
        let uncertainty: f64 = rng.gen();
        println!("step {}: uncertainty={:.2}", step, uncertainty);
        if uncertainty > 0.85 {
            println!("  uncertainty high -> would request human clarification");
            if env::var("INVOKE_CLI").unwrap_or_default() == "1" {
                // Try to run the Clarify CLI; this is optional and may fail if CLI isn't built.
                let prompt = format!(
                    "Uncertain decision at step {} (u={:.2}). Clarify?",
                    step, uncertainty
                );
                println!("  invoking clarify CLI with prompt: {}", prompt);
                let status = Command::new("cargo")
                    .args([
                        "run",
                        "-p",
                        "dynaniml-cli",
                        "--",
                        "clarify",
                        "--prompt",
                        &prompt,
                    ])
                    .status();
                match status {
                    Ok(s) => println!("  clarify returned: {}", s),
                    Err(e) => println!("  failed to invoke clarify CLI: {}", e),
                }
            } else {
                println!("  INVOKE_CLI not set; printing sample command:");
                println!("  cargo run -p dynaniml-cli -- clarify --prompt \"Uncertain decision\"");
            }
        }
    }
}
