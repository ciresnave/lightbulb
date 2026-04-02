use rand::Rng;
use std::env;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn now_ts() -> String {
    let n = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    format!("{}.{}", n.as_secs(), n.subsec_millis())
}

fn maybe_invoke_clarify(prompt: &str) {
    if env::var("INVOKE_CLI").unwrap_or_default() == "1" {
        // Try to invoke `dynaniml clarify --prompt "..."` if available on PATH
        let status = Command::new("dynaniml")
            .arg("clarify")
            .arg("--prompt")
            .arg(prompt)
            .status();

        match status {
            Ok(s) => println!("Clarify CLI exited with: {}", s),
            Err(e) => eprintln!("Failed to invoke clarify CLI: {}", e),
        }
    } else {
        println!("INVOKE_CLI not set — skipping external clarification.");
    }
}

fn main() {
    println!("hitl_hook example — simulate uncertainty and optional HITL CLI invocation");

    let mut rng = rand::thread_rng();

    for step in 1..=5 {
        let uncertainty: f64 = rng.gen(); // 0.0 .. 1.0
        println!("t={} uncertainty={:.3}", step, uncertainty);

        if uncertainty > 0.8 {
            let prompt = format!(
                "High uncertainty at {}: {:.3} — please clarify.",
                now_ts(),
                uncertainty
            );
            println!("Triggering HITL: {}", prompt);
            maybe_invoke_clarify(&prompt);
        }
    }

    println!("hitl_hook finished");
}
