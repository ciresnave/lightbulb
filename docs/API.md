# Lightbulb API Documentation

## Overview

Lightbulb provides an OpenAI-compatible REST API with additional extensions for advanced features like knowledge base operations, reasoning controls, and state management.

## Base URL

```
http://localhost:8080
```

## Authentication

All API requests require a Bearer token:

```
Authorization: Bearer YOUR_API_KEY
```

## OpenAI-Compatible Endpoints

### Chat Completions

Create a chat completion response.

**Endpoint:** `POST /v1/chat/completions`

**Request Body:**

```json
{
  "model": "lightbulb-7b",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 100,
  "stream": false,
  "lightbulb": {
    "reasoning_budget": {
      "max_chains": 5,
      "max_steps": 10
    },
    "use_knowledge_base": true,
    "metadata": {
      "priority": "high",
      "tags": ["research"]
    }
  }
}
```

**Response:**

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "lightbulb-7b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 15,
    "total_tokens": 25
  }
}
```

### Completions

Create a text completion.

**Endpoint:** `POST /v1/completions`

**Request Body:**

```json
{
  "model": "lightbulb-7b",
  "prompt": "Once upon a time",
  "max_tokens": 50,
  "temperature": 0.8
}
```

### List Models

List available models.

**Endpoint:** `GET /v1/models`

**Response:**

```json
{
  "object": "list",
  "data": [
    {
      "id": "lightbulb-7b",
      "object": "model",
      "created": 1234567890,
      "owned_by": "lightbulb"
    }
  ]
}
```

## Lightbulb Extensions

### Knowledge Base

#### Query Knowledge Base

Search the knowledge base for relevant facts.

**Endpoint:** `POST /v1/lightbulb/knowledge/query`

**Request:**

```json
{
  "query": "What is machine learning?",
  "max_results": 5,
  "category": "technology"
}
```

**Response:**

```json
{
  "facts": [
    {
      "content": "Machine learning is...",
      "category": "technology",
      "confidence": 0.95
    }
  ],
  "relevance_scores": [0.95, 0.87, 0.82]
}
```

#### Add Fact

Add a fact to the knowledge base.

**Endpoint:** `POST /v1/lightbulb/knowledge/add`

**Request:**

```json
{
  "content": "Python is a programming language",
  "category": "programming",
  "confidence": 1.0
}
```

#### Validate Consistency

Check knowledge base consistency.

**Endpoint:** `POST /v1/lightbulb/knowledge/validate`

**Request:**

```json
{
  "fact_ids": ["fact-123", "fact-456"]
}
```

**Response:**

```json
{
  "is_consistent": true,
  "conflicts": []
}
```

### Reasoning Controls

#### Set Reasoning Budget

Configure reasoning parameters.

**Endpoint:** `POST /v1/lightbulb/reasoning/budget`

**Request:**

```json
{
  "max_chains": 10,
  "max_steps": 20,
  "max_tokens": 2000
}
```

#### Check Convergence

Check if reasoning has converged.

**Endpoint:** `GET /v1/lightbulb/reasoning/convergence`

**Response:**

```json
{
  "has_converged": true,
  "iterations": 5,
  "confidence": 0.92
}
```

#### Reasoning Statistics

Get reasoning statistics.

**Endpoint:** `GET /v1/lightbulb/reasoning/stats`

**Response:**

```json
{
  "total_chains": 150,
  "total_steps": 500,
  "average_chain_length": 3.33,
  "convergence_rate": 0.85
}
```

### State Management

#### Save State

Save current inference state.

**Endpoint:** `POST /v1/lightbulb/state/save`

**Request:**

```json
{
  "state_name": "checkpoint-1",
  "description": "After processing 100 samples"
}
```

#### Restore State

Restore a saved state.

**Endpoint:** `POST /v1/lightbulb/state/restore`

**Request:**

```json
{
  "state_id": "state-123"
}
```

#### Create Branch

Create a state branch for experimentation.

**Endpoint:** `POST /v1/lightbulb/state/branch`

**Request:**

```json
{
  "branch_name": "experiment-1",
  "from_state_id": "state-123"
}
```

#### List States

List all saved states.

**Endpoint:** `GET /v1/lightbulb/state/list`

**Response:**

```json
{
  "states": [
    {
      "state_id": "state-123",
      "name": "checkpoint-1",
      "description": "After 100 samples",
      "created_at": "2025-10-30T12:00:00Z",
      "branch": "main"
    }
  ]
}
```

## Admin API

### Cache Statistics

Get cache statistics.

**Endpoint:** `GET /v1/lightbulb/admin/cache/stats`

**Response:**

```json
{
  "total_entries": 1000,
  "total_size_bytes": 104857600,
  "hit_rate": 0.85,
  "eviction_count": 50,
  "layers": [
    {
      "layer_idx": 0,
      "entries": 100,
      "size_bytes": 10485760,
      "compression_ratio": 0.5
    }
  ]
}
```

### Clear Cache

Clear cache entries.

**Endpoint:** `POST /v1/lightbulb/admin/cache/clear`

**Request:**

```json
{
  "layer_indices": [0, 1, 2],
  "older_than_seconds": 3600
}
```

### Scheduler Queue

View scheduler queue.

**Endpoint:** `GET /v1/lightbulb/admin/scheduler/queue`

**Response:**

```json
{
  "pending_requests": 5,
  "running_requests": 3,
  "queue": [
    {
      "request_id": "req-123",
      "priority": "high",
      "queued_at": "2025-10-30T12:00:00Z",
      "estimated_tokens": 500
    }
  ]
}
```

### Scheduler Statistics

Get scheduler statistics.

**Endpoint:** `GET /v1/lightbulb/admin/scheduler/stats`

**Response:**

```json
{
  "total_requests": 1000,
  "completed_requests": 950,
  "failed_requests": 10,
  "average_latency_ms": 150.5,
  "throughput_tokens_per_sec": 1000.0,
  "memory_usage_bytes": 1073741824
}
```

### System Metrics

Get system metrics.

**Endpoint:** `GET /v1/lightbulb/admin/metrics`

**Response:**

```json
{
  "cpu_usage_percent": 45.5,
  "memory_total_bytes": 17179869184,
  "memory_used_bytes": 8589934592,
  "gpu_metrics": [
    {
      "gpu_id": 0,
      "utilization_percent": 85.0,
      "memory_total_bytes": 17179869184,
      "memory_used_bytes": 12884901888,
      "temperature_celsius": 72.0
    }
  ]
}
```

## Rate Limiting

Rate limits are enforced per API key:

- Default: 60 requests per minute
- Admin endpoints: 120 requests per minute

When rate limited, the API returns HTTP 429 with:

```json
{
  "error": {
    "message": "Rate limit exceeded",
    "type": "rate_limit_error",
    "code": "too_many_requests"
  }
}
```

## Error Responses

All errors follow OpenAI's format:

```json
{
  "error": {
    "message": "Invalid request",
    "type": "invalid_request_error",
    "code": "invalid_parameter"
  }
}
```

**Error Types:**

- `authentication_error` - Invalid or missing API key
- `permission_error` - Insufficient permissions
- `invalid_request_error` - Malformed request
- `rate_limit_error` - Rate limit exceeded
- `server_error` - Internal server error

## Streaming

For streaming responses, set `"stream": true` in the request. The API returns Server-Sent Events:

```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"lightbulb-7b","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"lightbulb-7b","choices":[{"index":0,"delta":{"content":" there"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"lightbulb-7b","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":"stop"}]}

data: [DONE]
```

## Database Setup

The API requires PostgreSQL:

```bash
docker run -d \
  --name lightbulb-postgres \
  -e POSTGRES_USER=lightbulb \
  -e POSTGRES_PASSWORD=lightbulb \
  -e POSTGRES_DB=lightbulb \
  -p 5432:5432 \
  postgres:16-alpine
```

Migrations run automatically on server start.

## Configuration

Configure via `ApiConfig`:

```rust
let config = ApiConfig {
    database_url: "postgresql://lightbulb:lightbulb@localhost:5432/lightbulb".to_string(),
    bind_address: "0.0.0.0:8080".to_string(),
    enable_openai_api: true,
    enable_admin_api: true,
    enable_lightbulb_extensions: true,
    jwt_secret: "your-secret-here".to_string(),
    rate_limit_per_minute: 60,
    enable_audit_log: true,
};
```
