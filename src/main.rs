use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(
    name = "lightbulb",
    version,
    about = "Lightbulb: Candle-based ML runner"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Minimal hello-world generation to validate wiring
    HelloGenerate {
        /// Prompt text
        #[arg(short, long, default_value = "Hello from Lightbulb!")]
        prompt: String,
    },
    /// Offline CPU-only tokenization using a local tokenizer.json
    Tokenize {
        /// Path to tokenizer.json
        #[arg(long, default_value = "tokenizer.json")]
        tokenizer: String,
        /// Text to encode
        #[arg(short, long, default_value = "Hello from Lightbulb!")]
        prompt: String,
    },
    /// Run local LLaMA generation using a folder with config.json/tokenizer.json/*.safetensors
    LocalLlamaGen {
        /// Path to local model directory
        #[arg(long, default_value = "model/")]
        model_dir: String,
        /// Prompt text
        #[arg(short, long, default_value = "Hello from Lightbulb!")]
        prompt: String,
        /// Number of tokens to generate
        #[arg(long, default_value_t = 64)]
        sample_len: usize,
        /// Temperature (<=0 for ArgMax)
        #[arg(long, default_value_t = 0.7)]
        temperature: f64,
        /// Top-p nucleus sampling (optional)
        #[arg(long)]
        top_p: Option<f64>,
        /// RNG seed
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Same as LocalLlamaGen but routed through the minimal Scheduler
    LocalLlamaSched {
        /// Path to local model directory
        #[arg(long, default_value = "model/")]
        model_dir: String,
        /// Prompt text
        #[arg(short, long, default_value = "Hello from Lightbulb!")]
        prompt: String,
        /// Number of tokens to generate
        #[arg(long, default_value_t = 32)]
        sample_len: usize,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    match cli.command {
        Commands::HelloGenerate { prompt } => {
            lightbulb::hello_generate(&prompt).await?;
        }
        Commands::Tokenize { tokenizer, prompt } => {
            let (ids, text) = lightbulb::encode_decode_with_tokenizer(&tokenizer, &prompt)?;
            println!("[tokenize] ids={ids:?} decoded={text}");
        }
        Commands::LocalLlamaGen {
            model_dir,
            prompt,
            sample_len,
            temperature,
            top_p,
            seed,
        } => {
            let out = lightbulb::local_llama_generate(
                &model_dir,
                &prompt,
                sample_len,
                temperature,
                top_p,
                seed,
            )?;
            println!("{prompt}{out}");
        }
        Commands::LocalLlamaSched {
            model_dir,
            prompt,
            sample_len,
        } => {
            let sched = lightbulb::engine::Scheduler::new();
            let req = lightbulb::engine::Request {
                id: "r1".into(),
                prompt: prompt.clone(),
                max_new_tokens: sample_len,
            };
            let out = sched.run_single(&req, |p, n| {
                // Delegate to the same local generation path with deterministic defaults.
                lightbulb::local_llama_generate(&model_dir, p, n, 0.7, None, 42)
            })?;
            println!("{prompt}{out}");
        }
    }
    Ok(())
}
