# Lightbulb CLI

Interactive command-line chat client for Lightbulb API server.

## Installation

Build the CLI:

```bash
cargo build --release --bin lightbulb-cli
```

The binary will be at `target/release/lightbulb-cli.exe` (Windows) or `target/release/lightbulb-cli` (Linux/Mac).

## Usage

### Basic usage

```bash
lightbulb-cli --api-key lb-your-api-key-here
```

Or set the API key as an environment variable:

```bash
# Windows (PowerShell)
$env:LIGHTBULB_API_KEY="lb-your-api-key-here"
lightbulb-cli

# Linux/Mac
export LIGHTBULB_API_KEY="lb-your-api-key-here"
lightbulb-cli
```

### Options

- `--api-key <KEY>` - API key for authentication (or use `LIGHTBULB_API_KEY` env var)
- `--url <URL>` - Base URL of the Lightbulb server (default: `http://localhost:8080`)
- `--stream` / `-s` - Enable streaming mode for real-time token-by-token responses
- `--model <MODEL>` / `-m` - Model to use (default: `default`)
- `--system <PROMPT>` - System prompt to set context
- `--temperature <TEMP>` - Temperature for sampling (default: 0.7, range: 0.0-2.0)
- `--max-tokens <NUM>` - Maximum tokens to generate (default: 512)

### Examples

**Non-streaming chat:**
```bash
lightbulb-cli --api-key lb-abc123
```

**Streaming chat:**
```bash
lightbulb-cli --api-key lb-abc123 --stream
```

**With system prompt:**
```bash
lightbulb-cli --api-key lb-abc123 --system "You are a helpful coding assistant"
```

**Custom server URL:**
```bash
lightbulb-cli --api-key lb-abc123 --url http://my-server:8080
```

**All options:**
```bash
lightbulb-cli \
  --api-key lb-abc123 \
  --url http://localhost:8080 \
  --stream \
  --model gpt-3.5-turbo \
  --system "You are a helpful assistant" \
  --temperature 0.8 \
  --max-tokens 1024
```

## Interactive Commands

While in the chat session:

- Type your message and press Enter to send
- `exit`, `quit`, or `q` - Exit the CLI
- `clear` - Clear conversation history

## Getting an API Key

To create an API key, use the admin endpoint (requires admin access):

```bash
curl -X POST http://localhost:8080/v1/lightbulb/admin/api-keys \
  -H "Authorization: Bearer YOUR-ADMIN-KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "role": "user",
    "expires_in_seconds": 2592000,
    "description": "My CLI key"
  }'
```

The response will contain your API key (shown only once):

```json
{
  "api_key": "lb-a1b2c3d4...",
  "key_id": "uuid-here",
  "role": "user",
  "expires_at": "2024-12-01T00:00:00Z"
}
```

Save the `api_key` value and use it with the CLI.

## Example Session

```
🔦 Lightbulb CLI
Connected to: http://localhost:8080
Streaming: enabled

You: Hello! What's the capital of France?
Assistant: The capital of France is Paris. It's one of the most famous and visited cities in the world, known for landmarks like the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral.

You: Tell me more about the Eiffel Tower
Assistant: The Eiffel Tower was built in 1889 for the World's Fair and was designed by engineer Gustave Eiffel. Standing at 324 meters (1,063 feet) tall, it was the world's tallest man-made structure until 1930...

You: exit
Goodbye! 👋
```
