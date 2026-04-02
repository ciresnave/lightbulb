# MoCoPr Integration for Lightbulb

## Date: October 19, 2025

## Overview

**MoCoPr** is a comprehensive Rust implementation of the Model Context Protocol (MCP), providing both server and client capabilities. This integration enables Lightbulb to:

1. **Expose capabilities to LLM supervisors** (mocopr-server) - Make Lightbulb observable and controllable
2. **Access external tools during inference** (mocopr-client) - Enable models to use real-world APIs and services

**Strategic Advantage**: Since this is your crate, we can extend it with Lightbulb-specific features and optimizations.

---

## What Is MoCoPr?

MoCoPr implements the Model Context Protocol (MCP), a standard for communication between LLMs and external systems. It provides three main crates:

- **mocopr-core**: Protocol types, transport abstraction, security, monitoring
- **mocopr-server**: High-level server for exposing resources, tools, and prompts
- **mocopr-client**: High-level client for consuming external MCP servers

**Version**: 0.1.0 (experimental - API may change)  
**Protocol**: JSON-RPC 2.0 over stdio, WebSocket, or HTTP

---

## Part 1: MoCoPr Server - Exposing Lightbulb

### **Purpose**

Make Lightbulb's capabilities accessible to LLM supervisors and external systems through a standard protocol.

### **Core Concepts**

MCP servers expose three types of capabilities:

1. **Resources**: Read-only data (metrics, state, configurations)
2. **Tools**: Executable operations (control actions, commands)
3. **Prompts**: Template-based text generation

### **1.1 Basic MCP Server Setup**

```rust
use mocopr_server::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let server = McpServer::builder()
        .with_info("Lightbulb", "0.1.0")
        .with_resources()
        .with_tools()
        .with_prompts()
        .build()?;

    // Run server on stdio (for process-based communication)
    server.run_stdio().await?;
    Ok(())
}
```

---

### **1.2 Exposing Resources (Read-Only State)**

#### **Inference Metrics Resource**

```rust
use mocopr_server::prelude::*;
use serde_json::json;

#[derive(Clone)]
struct InferenceMetricsResource {
    scheduler: Arc<Scheduler>,
}

#[async_trait]
impl ResourceReader for InferenceMetricsResource {
    async fn read(&self, uri: &str) -> Result<Vec<ResourceContent>> {
        match uri {
            "lightbulb://metrics/scheduler" => {
                let metrics = self.scheduler.get_metrics().await;
                Ok(vec![ResourceContent::Text {
                    uri: uri.to_string(),
                    mime_type: Some("application/json".to_string()),
                    text: serde_json::to_string_pretty(&metrics)?,
                }])
            }
            "lightbulb://metrics/throughput" => {
                let stats = self.scheduler.get_throughput_stats().await;
                Ok(vec![ResourceContent::Text {
                    uri: uri.to_string(),
                    mime_type: Some("application/json".to_string()),
                    text: serde_json::to_string_pretty(&json!({
                        "tokens_per_second": stats.tokens_per_second,
                        "requests_per_second": stats.requests_per_second,
                        "active_requests": stats.active_requests,
                        "queue_depth": stats.queue_depth,
                    }))?,
                }])
            }
            _ => Err(Error::ResourceNotFound),
        }
    }

    async fn list(&self) -> Result<Vec<Resource>> {
        Ok(vec![
            Resource {
                uri: "lightbulb://metrics/scheduler".to_string(),
                name: "Scheduler Metrics".to_string(),
                description: Some("Current scheduler state and metrics".to_string()),
                mime_type: Some("application/json".to_string()),
            },
            Resource {
                uri: "lightbulb://metrics/throughput".to_string(),
                name: "Throughput Statistics".to_string(),
                description: Some("Real-time throughput metrics".to_string()),
                mime_type: Some("application/json".to_string()),
            },
        ])
    }
}

// Register resource handler
server.register_resource_handler(Arc::new(InferenceMetricsResource {
    scheduler: scheduler.clone(),
})).await?;
```

#### **M7 - Sentience State Resource**

```rust
use mocopr_server::prelude::*;

#[derive(Clone)]
struct SentienceStateResource {
    identity_graph: Arc<IdentityGraph>,
    motivations: Arc<MotivationalHierarchy>,
}

#[async_trait]
impl ResourceReader for SentienceStateResource {
    async fn read(&self, uri: &str) -> Result<Vec<ResourceContent>> {
        match uri {
            "lightbulb://sentience/identity" => {
                let identity = self.identity_graph.get_current_state().await;
                Ok(vec![ResourceContent::Text {
                    uri: uri.to_string(),
                    mime_type: Some("application/json".to_string()),
                    text: serde_json::to_string_pretty(&json!({
                        "self_concept": identity.self_concept,
                        "values": identity.values,
                        "beliefs": identity.beliefs,
                        "personality_traits": identity.personality,
                        "developmental_stage": identity.stage,
                        "coherence_score": identity.coherence_score,
                    }))?,
                }])
            }
            "lightbulb://sentience/motivations" => {
                let goals = self.motivations.get_active_goals().await;
                Ok(vec![ResourceContent::Text {
                    uri: uri.to_string(),
                    mime_type: Some("application/json".to_string()),
                    text: serde_json::to_string_pretty(&json!({
                        "active_goals": goals,
                        "hierarchy": self.motivations.get_hierarchy().await,
                        "conflicts": self.motivations.get_conflicts().await,
                    }))?,
                }])
            }
            "lightbulb://sentience/partnerships" => {
                let partnerships = self.identity_graph.get_partnership_metrics().await;
                Ok(vec![ResourceContent::Text {
                    uri: uri.to_string(),
                    mime_type: Some("application/json".to_string()),
                    text: serde_json::to_string_pretty(&partnerships)?,
                }])
            }
            _ => Err(Error::ResourceNotFound),
        }
    }

    async fn list(&self) -> Result<Vec<Resource>> {
        Ok(vec![
            Resource {
                uri: "lightbulb://sentience/identity".to_string(),
                name: "Identity State".to_string(),
                description: Some("Current identity graph state".to_string()),
                mime_type: Some("application/json".to_string()),
            },
            Resource {
                uri: "lightbulb://sentience/motivations".to_string(),
                name: "Motivational Hierarchy".to_string(),
                description: Some("Active goals and motivations".to_string()),
                mime_type: Some("application/json".to_string()),
            },
            Resource {
                uri: "lightbulb://sentience/partnerships".to_string(),
                name: "Partnership Metrics".to_string(),
                description: Some("Trust scores and cooperation metrics".to_string()),
                mime_type: Some("application/json".to_string()),
            },
        ])
    }
}
```

#### **M8 - Training State Resource**

```rust
use mocopr_server::prelude::*;

#[derive(Clone)]
struct TrainingStateResource {
    trainer: Arc<ModularTrainer>,
}

#[async_trait]
impl ResourceReader for TrainingStateResource {
    async fn read(&self, uri: &str) -> Result<Vec<ResourceContent>> {
        match uri {
            "lightbulb://training/status" => {
                let status = self.trainer.get_status().await;
                Ok(vec![ResourceContent::Text {
                    uri: uri.to_string(),
                    mime_type: Some("application/json".to_string()),
                    text: serde_json::to_string_pretty(&json!({
                        "current_epoch": status.epoch,
                        "global_step": status.global_step,
                        "loss": status.current_loss,
                        "learning_rate": status.learning_rate,
                        "gpu_memory_used": status.gpu_memory_mb,
                        "tokens_per_second": status.tokens_per_second,
                    }))?,
                }])
            }
            "lightbulb://training/modules" => {
                let modules = self.trainer.list_modules().await;
                Ok(vec![ResourceContent::Text {
                    uri: uri.to_string(),
                    mime_type: Some("application/json".to_string()),
                    text: serde_json::to_string_pretty(&modules)?,
                }])
            }
            "lightbulb://training/pattern_library" => {
                let patterns = self.trainer.get_pattern_library().await;
                Ok(vec![ResourceContent::Text {
                    uri: uri.to_string(),
                    mime_type: Some("application/json".to_string()),
                    text: serde_json::to_string_pretty(&patterns)?,
                }])
            }
            _ => Err(Error::ResourceNotFound),
        }
    }

    async fn list(&self) -> Result<Vec<Resource>> {
        Ok(vec![
            Resource {
                uri: "lightbulb://training/status".to_string(),
                name: "Training Status".to_string(),
                description: Some("Current training progress and metrics".to_string()),
                mime_type: Some("application/json".to_string()),
            },
            Resource {
                uri: "lightbulb://training/modules".to_string(),
                name: "Module Registry".to_string(),
                description: Some("List of trained modules with metadata".to_string()),
                mime_type: Some("application/json".to_string()),
            },
            Resource {
                uri: "lightbulb://training/pattern_library".to_string(),
                name: "Pattern Library".to_string(),
                description: Some("Architectural patterns and their metrics".to_string()),
                mime_type: Some("application/json".to_string()),
            },
        ])
    }
}
```

---

### **1.3 Exposing Tools (Control Operations)**

#### **Scheduling Control Tool**

```rust
use mocopr_server::{tool_handler, prelude::*};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct SetPriorityArgs {
    request_id: String,
    priority: i32,
}

#[tool_handler]
async fn set_request_priority(
    args: SetPriorityArgs,
    scheduler: Arc<Scheduler>,
) -> Result<ToolResult> {
    scheduler.set_priority(&args.request_id, args.priority).await?;
    
    Ok(ToolResult {
        content: vec![Content::Text {
            text: format!(
                "Updated request {} to priority {}",
                args.request_id, args.priority
            ),
        }],
        is_error: false,
    })
}

#[derive(Deserialize)]
struct PreemptRequestArgs {
    request_id: String,
}

#[tool_handler]
async fn preempt_request(
    args: PreemptRequestArgs,
    scheduler: Arc<Scheduler>,
) -> Result<ToolResult> {
    scheduler.preempt_request(&args.request_id).await?;
    
    Ok(ToolResult {
        content: vec![Content::Text {
            text: format!("Preempted request {}", args.request_id),
        }],
        is_error: false,
    })
}

// Register tools
server.register_tool(
    Tool {
        name: "set_request_priority".to_string(),
        description: Some("Change the priority of a queued request".to_string()),
        input_schema: json!({
            "type": "object",
            "properties": {
                "request_id": { "type": "string" },
                "priority": { "type": "integer" }
            },
            "required": ["request_id", "priority"]
        }),
    },
    set_request_priority,
).await?;

server.register_tool(
    Tool {
        name: "preempt_request".to_string(),
        description: Some("Preempt a running request to free resources".to_string()),
        input_schema: json!({
            "type": "object",
            "properties": {
                "request_id": { "type": "string" }
            },
            "required": ["request_id"]
        }),
    },
    preempt_request,
).await?;
```

#### **M7 - Capability Control Tool**

```rust
use mocopr_server::{tool_handler, prelude::*};

#[derive(Deserialize)]
struct UnlockCapabilityArgs {
    capability_name: String,
    agent_id: String,
    reason: String,
}

#[tool_handler]
async fn unlock_capability(
    args: UnlockCapabilityArgs,
    capability_manager: Arc<CapabilityManager>,
) -> Result<ToolResult> {
    let result = capability_manager.unlock(
        &args.agent_id,
        &args.capability_name,
        &args.reason,
    ).await?;
    
    Ok(ToolResult {
        content: vec![Content::Text {
            text: format!(
                "Capability '{}' unlocked for agent '{}'. New capabilities: {:?}",
                args.capability_name, args.agent_id, result.unlocked_capabilities
            ),
        }],
        is_error: false,
    })
}

#[derive(Deserialize)]
struct IntrospectionQueryArgs {
    query: String,
}

#[tool_handler]
async fn introspection_query(
    args: IntrospectionQueryArgs,
    sentience: Arc<SentienceManager>,
) -> Result<ToolResult> {
    let response = sentience.introspect(&args.query).await?;
    
    Ok(ToolResult {
        content: vec![Content::Text {
            text: response.explanation,
        }],
        is_error: false,
    })
}

// Register M7 tools
server.register_tool(
    Tool {
        name: "unlock_capability".to_string(),
        description: Some("Unlock a capability for an agent based on partnership progress".to_string()),
        input_schema: json!({
            "type": "object",
            "properties": {
                "capability_name": { "type": "string" },
                "agent_id": { "type": "string" },
                "reason": { "type": "string" }
            },
            "required": ["capability_name", "agent_id", "reason"]
        }),
    },
    unlock_capability,
).await?;

server.register_tool(
    Tool {
        name: "introspection_query".to_string(),
        description: Some("Ask the system to explain its internal state or reasoning".to_string()),
        input_schema: json!({
            "type": "object",
            "properties": {
                "query": { "type": "string" }
            },
            "required": ["query"]
        }),
    },
    introspection_query,
).await?;
```

#### **M8 - Training Control Tool**

```rust
use mocopr_server::{tool_handler, prelude::*};

#[derive(Deserialize)]
struct AdjustLearningRateArgs {
    new_learning_rate: f64,
}

#[tool_handler]
async fn adjust_learning_rate(
    args: AdjustLearningRateArgs,
    trainer: Arc<ModularTrainer>,
) -> Result<ToolResult> {
    trainer.set_learning_rate(args.new_learning_rate).await?;
    
    Ok(ToolResult {
        content: vec![Content::Text {
            text: format!("Learning rate adjusted to {}", args.new_learning_rate),
        }],
        is_error: false,
    })
}

#[derive(Deserialize)]
struct SaveCheckpointArgs {
    name: Option<String>,
}

#[tool_handler]
async fn save_checkpoint(
    args: SaveCheckpointArgs,
    trainer: Arc<ModularTrainer>,
) -> Result<ToolResult> {
    let checkpoint_path = trainer.save_checkpoint(args.name.as_deref()).await?;
    
    Ok(ToolResult {
        content: vec![Content::Text {
            text: format!("Checkpoint saved: {}", checkpoint_path.display()),
        }],
        is_error: false,
    })
}

#[derive(Deserialize)]
struct ComposeModulesArgs {
    module_names: Vec<String>,
    composition_strategy: String,
}

#[tool_handler]
async fn compose_modules(
    args: ComposeModulesArgs,
    trainer: Arc<ModularTrainer>,
) -> Result<ToolResult> {
    let composed = trainer.compose_modules(
        &args.module_names,
        &args.composition_strategy,
    ).await?;
    
    Ok(ToolResult {
        content: vec![Content::Text {
            text: format!(
                "Composed {} modules. Accuracy: {:.2}%, Latency: {}ms",
                args.module_names.len(),
                composed.accuracy * 100.0,
                composed.latency_ms
            ),
        }],
        is_error: false,
    })
}

// Register M8 tools
server.register_tool(
    Tool {
        name: "adjust_learning_rate".to_string(),
        description: Some("Dynamically adjust learning rate during training".to_string()),
        input_schema: json!({
            "type": "object",
            "properties": {
                "new_learning_rate": { "type": "number" }
            },
            "required": ["new_learning_rate"]
        }),
    },
    adjust_learning_rate,
).await?;

server.register_tool(
    Tool {
        name: "save_checkpoint".to_string(),
        description: Some("Save current training checkpoint".to_string()),
        input_schema: json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" }
            }
        }),
    },
    save_checkpoint,
).await?;

server.register_tool(
    Tool {
        name: "compose_modules".to_string(),
        description: Some("Compose multiple trained modules into a pipeline".to_string()),
        input_schema: json!({
            "type": "object",
            "properties": {
                "module_names": { 
                    "type": "array",
                    "items": { "type": "string" }
                },
                "composition_strategy": { "type": "string" }
            },
            "required": ["module_names", "composition_strategy"]
        }),
    },
    compose_modules,
).await?;
```

---

### **1.4 Exposing Prompts (Template Generation)**

```rust
use mocopr_server::{template_prompt, prelude::*};

#[template_prompt]
async fn system_status_prompt(args: HashMap<String, String>) -> Result<PromptMessages> {
    let detail_level = args.get("detail_level").map(String::as_str).unwrap_or("summary");
    
    let message = match detail_level {
        "summary" => "Provide a brief summary of Lightbulb's current state.",
        "detailed" => "Provide a detailed report of Lightbulb's metrics, active requests, and resource usage.",
        "diagnostic" => "Provide full diagnostic information including all metrics, errors, and warnings.",
        _ => "Unknown detail level",
    };
    
    Ok(PromptMessages {
        messages: vec![
            PromptMessage {
                role: Role::System,
                content: Content::Text {
                    text: message.to_string(),
                },
            },
        ],
        description: Some("Generate status report for Lightbulb".to_string()),
    })
}

// Register prompt
server.register_prompt(
    Prompt {
        name: "system_status".to_string(),
        description: Some("Generate a status report prompt".to_string()),
        arguments: vec![
            PromptArgument {
                name: "detail_level".to_string(),
                description: Some("Level of detail: summary, detailed, or diagnostic".to_string()),
                required: false,
            },
        ],
    },
    system_status_prompt,
).await?;
```

---

## Part 2: MoCoPr Client - Accessing External Tools

### **Purpose**

Enable models running on Lightbulb to call external MCP servers for tool access, knowledge retrieval, and real-world interaction.

### **2.1 Basic MCP Client Usage**

```rust
use mocopr_client::McpClient;
use mocopr_core::prelude::*;
use serde_json::json;

// Connect to an external MCP server
let client = McpClient::connect_stdio(
    "python",
    &["weather_server.py"],
    Implementation {
        name: "Lightbulb".to_string(),
        version: "0.1.0".to_string(),
    },
    ClientCapabilities::default(),
).await?;

// Discover available tools
let tools_response = client.list_tools().await?;
for tool in &tools_response.tools {
    println!("Tool: {} - {}", 
        tool.name, 
        tool.description.as_deref().unwrap_or("No description")
    );
}

// Call a tool
let result = client.call_tool(
    "get_weather".to_string(),
    Some(json!({
        "city": "Paris",
        "units": "metric"
    }))
).await?;

println!("Weather: {:?}", result.content);
```

---

### **2.2 Tool-Augmented Inference**

#### **Integration with Inference Pipeline**

```rust
use mocopr_client::McpClient;
use std::collections::HashMap;

pub struct ToolRegistry {
    clients: HashMap<String, Arc<McpClient>>,
}

impl ToolRegistry {
    pub async fn new() -> Result<Self> {
        let mut clients = HashMap::new();
        
        // Register various MCP servers
        clients.insert(
            "calculator".to_string(),
            Arc::new(McpClient::connect_stdio(
                "python",
                &["calculator_mcp.py"],
                Implementation {
                    name: "Lightbulb".to_string(),
                    version: "0.1.0".to_string(),
                },
                ClientCapabilities::default(),
            ).await?),
        );
        
        clients.insert(
            "web_search".to_string(),
            Arc::new(McpClient::connect_stdio(
                "python",
                &["web_search_mcp.py"],
                Implementation {
                    name: "Lightbulb".to_string(),
                    version: "0.1.0".to_string(),
                },
                ClientCapabilities::default(),
            ).await?),
        );
        
        clients.insert(
            "code_executor".to_string(),
            Arc::new(McpClient::connect_stdio(
                "node",
                &["code_executor.js"],
                Implementation {
                    name: "Lightbulb".to_string(),
                    version: "0.1.0".to_string(),
                },
                ClientCapabilities::default(),
            ).await?),
        );
        
        Ok(Self { clients })
    }
    
    pub async fn call_tool(
        &self,
        server_name: &str,
        tool_name: &str,
        args: Value,
    ) -> Result<ToolsCallResponse> {
        let client = self.clients.get(server_name)
            .ok_or_else(|| Error::ServerNotFound)?;
        
        client.call_tool(tool_name.to_string(), Some(args)).await
    }
    
    pub async fn list_all_tools(&self) -> Result<Vec<(String, Vec<Tool>)>> {
        let mut all_tools = Vec::new();
        
        for (server_name, client) in &self.clients {
            let response = client.list_tools().await?;
            all_tools.push((server_name.clone(), response.tools));
        }
        
        Ok(all_tools)
    }
}
```

#### **Tool Call Detection During Inference**

```rust
use regex::Regex;

pub struct ToolCallDetector {
    tool_registry: Arc<ToolRegistry>,
    pattern: Regex,
}

impl ToolCallDetector {
    pub fn new(tool_registry: Arc<ToolRegistry>) -> Self {
        // Pattern to detect tool calls in model output
        // Format: <tool>server_name.tool_name({"arg": "value"})</tool>
        let pattern = Regex::new(
            r"<tool>(\w+)\.(\w+)\((.*?)\)</tool>"
        ).unwrap();
        
        Self { tool_registry, pattern }
    }
    
    pub async fn process_output(&self, output: &str) -> Result<String> {
        let mut result = output.to_string();
        
        for capture in self.pattern.captures_iter(output) {
            let full_match = &capture[0];
            let server_name = &capture[1];
            let tool_name = &capture[2];
            let args_str = &capture[3];
            
            // Parse arguments
            let args: Value = serde_json::from_str(args_str)?;
            
            // Execute tool
            let tool_result = self.tool_registry.call_tool(
                server_name,
                tool_name,
                args,
            ).await?;
            
            // Extract result text
            let result_text = tool_result.content.iter()
                .filter_map(|c| match c {
                    Content::Text { text } => Some(text.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            
            // Replace tool call with result
            result = result.replace(full_match, &result_text);
        }
        
        Ok(result)
    }
}

// Usage in inference loop
let detector = ToolCallDetector::new(tool_registry.clone());

// Generate initial response
let mut output = model.generate(prompt).await?;

// Process tool calls (may trigger multiple rounds)
loop {
    let processed = detector.process_output(&output).await?;
    
    // If no tool calls were executed, we're done
    if processed == output {
        break;
    }
    
    // Feed result back to model for continuation
    output = model.generate(&format!("{}\n{}", prompt, processed)).await?;
}
```

---

### **2.3 Retrieval-Augmented Generation (RAG)**

```rust
use mocopr_client::McpClient;

pub struct RAGSystem {
    vector_db_client: Arc<McpClient>,
    web_search_client: Arc<McpClient>,
}

impl RAGSystem {
    pub async fn new() -> Result<Self> {
        // Connect to vector database MCP server
        let vector_db_client = Arc::new(McpClient::connect_stdio(
            "python",
            &["vector_db_mcp.py"],
            Implementation {
                name: "Lightbulb RAG".to_string(),
                version: "0.1.0".to_string(),
            },
            ClientCapabilities::default(),
        ).await?);
        
        // Connect to web search MCP server
        let web_search_client = Arc::new(McpClient::connect_stdio(
            "python",
            &["web_search_mcp.py"],
            Implementation {
                name: "Lightbulb RAG".to_string(),
                version: "0.1.0".to_string(),
            },
            ClientCapabilities::default(),
        ).await?);
        
        Ok(Self { vector_db_client, web_search_client })
    }
    
    pub async fn retrieve_context(&self, query: &str) -> Result<String> {
        // Search vector database
        let vector_results = self.vector_db_client.call_tool(
            "semantic_search".to_string(),
            Some(json!({
                "query": query,
                "top_k": 5
            }))
        ).await?;
        
        // Optionally augment with web search
        let web_results = self.web_search_client.call_tool(
            "search".to_string(),
            Some(json!({
                "query": query,
                "num_results": 3
            }))
        ).await?;
        
        // Combine results
        let context = format!(
            "Vector DB Results:\n{}\n\nWeb Search Results:\n{}",
            Self::extract_text(&vector_results.content),
            Self::extract_text(&web_results.content)
        );
        
        Ok(context)
    }
    
    fn extract_text(content: &[Content]) -> String {
        content.iter()
            .filter_map(|c| match c {
                Content::Text { text } => Some(text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

// Usage in inference
let rag = RAGSystem::new().await?;
let context = rag.retrieve_context(user_query).await?;

let augmented_prompt = format!(
    "Context:\n{}\n\nUser Query: {}\n\nAnswer:",
    context, user_query
);

let response = model.generate(&augmented_prompt).await?;
```

---

### **2.4 M7 - External Knowledge for Sentience**

```rust
use mocopr_client::McpClient;

pub struct ExternalKnowledgeSystem {
    email_client: Option<Arc<McpClient>>,
    calendar_client: Option<Arc<McpClient>>,
    file_system_client: Option<Arc<McpClient>>,
}

impl ExternalKnowledgeSystem {
    pub async fn new() -> Result<Self> {
        let email_client = Some(Arc::new(McpClient::connect_stdio(
            "python",
            &["email_mcp.py"],
            Implementation {
                name: "Lightbulb Sentience".to_string(),
                version: "0.1.0".to_string(),
            },
            ClientCapabilities::default(),
        ).await?));
        
        let calendar_client = Some(Arc::new(McpClient::connect_stdio(
            "python",
            &["calendar_mcp.py"],
            Implementation {
                name: "Lightbulb Sentience".to_string(),
                version: "0.1.0".to_string(),
            },
            ClientCapabilities::default(),
        ).await?));
        
        let file_system_client = Some(Arc::new(McpClient::connect_stdio(
            "python",
            &["filesystem_mcp.py"],
            Implementation {
                name: "Lightbulb Sentience".to_string(),
                version: "0.1.0".to_string(),
            },
            ClientCapabilities::default(),
        ).await?));
        
        Ok(Self {
            email_client,
            calendar_client,
            file_system_client,
        })
    }
    
    pub async fn gather_user_context(&self, user_id: &str) -> Result<UserContext> {
        let mut context = UserContext::default();
        
        // Get recent emails
        if let Some(email) = &self.email_client {
            let emails = email.call_tool(
                "search_emails".to_string(),
                Some(json!({
                    "from": user_id,
                    "limit": 10,
                    "days": 7
                }))
            ).await?;
            context.recent_communications = Self::extract_text(&emails.content);
        }
        
        // Get calendar events
        if let Some(calendar) = &self.calendar_client {
            let events = calendar.call_tool(
                "get_events".to_string(),
                Some(json!({
                    "user_id": user_id,
                    "days_ahead": 7
                }))
            ).await?;
            context.upcoming_commitments = Self::extract_text(&events.content);
        }
        
        // Get user preferences from file system
        if let Some(fs) = &self.file_system_client {
            let prefs = fs.call_tool(
                "read_file".to_string(),
                Some(json!({
                    "path": format!("users/{}/preferences.json", user_id)
                }))
            ).await?;
            context.preferences = Self::extract_text(&prefs.content);
        }
        
        Ok(context)
    }
    
    fn extract_text(content: &[Content]) -> String {
        content.iter()
            .filter_map(|c| match c {
                Content::Text { text } => Some(text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[derive(Default)]
pub struct UserContext {
    pub recent_communications: String,
    pub upcoming_commitments: String,
    pub preferences: String,
}

// Usage in partnership building
let knowledge_system = ExternalKnowledgeSystem::new().await?;
let user_context = knowledge_system.gather_user_context("user@example.com").await?;

// Sentience system uses context for better understanding
sentience.update_partner_model(
    "user@example.com",
    &user_context,
).await?;
```

---

### **2.5 M8 - Training with Real Feedback**

```rust
use mocopr_client::McpClient;

pub struct TrainingFeedbackSystem {
    code_validator: Arc<McpClient>,
    test_runner: Arc<McpClient>,
}

impl TrainingFeedbackSystem {
    pub async fn new() -> Result<Self> {
        let code_validator = Arc::new(McpClient::connect_stdio(
            "node",
            &["typescript_validator.js"],
            Implementation {
                name: "Lightbulb Training".to_string(),
                version: "0.1.0".to_string(),
            },
            ClientCapabilities::default(),
        ).await?);
        
        let test_runner = Arc::new(McpClient::connect_stdio(
            "python",
            &["pytest_runner_mcp.py"],
            Implementation {
                name: "Lightbulb Training".to_string(),
                version: "0.1.0".to_string(),
            },
            ClientCapabilities::default(),
        ).await?);
        
        Ok(Self { code_validator, test_runner })
    }
    
    pub async fn validate_generated_code(&self, code: &str, language: &str) -> Result<ValidationResult> {
        let tool_name = match language {
            "typescript" => "validate_typescript",
            "python" => "validate_python",
            "rust" => "validate_rust",
            _ => return Err(Error::UnsupportedLanguage),
        };
        
        let result = self.code_validator.call_tool(
            tool_name.to_string(),
            Some(json!({
                "code": code
            }))
        ).await?;
        
        // Parse validation result
        let text = Self::extract_text(&result.content);
        let validation: ValidationResult = serde_json::from_str(&text)?;
        
        Ok(validation)
    }
    
    pub async fn run_tests(&self, code: &str, test_code: &str) -> Result<TestResults> {
        let result = self.test_runner.call_tool(
            "run_tests".to_string(),
            Some(json!({
                "code": code,
                "test_code": test_code
            }))
        ).await?;
        
        let text = Self::extract_text(&result.content);
        let test_results: TestResults = serde_json::from_str(&text)?;
        
        Ok(test_results)
    }
    
    fn extract_text(content: &[Content]) -> String {
        content.iter()
            .filter_map(|c| match c {
                Content::Text { text } => Some(text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[derive(Deserialize)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

#[derive(Deserialize)]
pub struct TestResults {
    pub passed: usize,
    pub failed: usize,
    pub errors: Vec<String>,
}

// Usage during module training
let feedback_system = TrainingFeedbackSystem::new().await?;

// Generate code with model
let generated_code = model.generate(coding_prompt).await?;

// Validate immediately
let validation = feedback_system.validate_generated_code(
    &generated_code,
    "typescript"
).await?;

// Incorporate validation result into training loss
let validation_penalty = if validation.is_valid {
    0.0
} else {
    validation.errors.len() as f32 * 0.1
};

let total_loss = base_loss + validation_penalty;

// Run tests for additional feedback
if validation.is_valid {
    let test_results = feedback_system.run_tests(
        &generated_code,
        &test_code,
    ).await?;
    
    let test_penalty = (test_results.failed as f32 / 
                       (test_results.passed + test_results.failed) as f32) * 0.5;
    
    let total_loss = base_loss + validation_penalty + test_penalty;
}
```

---

## Part 3: Unified Architecture

### **3.1 Lightbulb as MCP Hub**

```rust
use mocopr_server::McpServer;
use mocopr_client::McpClient;
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct LightbulbMCPHub {
    // Server: Expose Lightbulb to supervisors
    server: Arc<McpServer>,
    
    // Client Registry: Connect to external MCP servers
    tool_registry: Arc<ToolRegistry>,
    rag_system: Arc<RAGSystem>,
    knowledge_system: Arc<ExternalKnowledgeSystem>,
    feedback_system: Arc<TrainingFeedbackSystem>,
    
    // State
    scheduler: Arc<Scheduler>,
    sentience: Arc<SentienceManager>,
    trainer: Arc<ModularTrainer>,
}

impl LightbulbMCPHub {
    pub async fn new(
        scheduler: Arc<Scheduler>,
        sentience: Arc<SentienceManager>,
        trainer: Arc<ModularTrainer>,
    ) -> Result<Self> {
        // Initialize MCP server
        let server = Arc::new(
            McpServer::builder()
                .with_info("Lightbulb", "0.1.0")
                .with_resources()
                .with_tools()
                .with_prompts()
                .build()?
        );
        
        // Register all resources
        Self::register_resources(&server, &scheduler, &sentience, &trainer).await?;
        
        // Register all tools
        Self::register_tools(&server, &scheduler, &sentience, &trainer).await?;
        
        // Initialize client systems
        let tool_registry = Arc::new(ToolRegistry::new().await?);
        let rag_system = Arc::new(RAGSystem::new().await?);
        let knowledge_system = Arc::new(ExternalKnowledgeSystem::new().await?);
        let feedback_system = Arc::new(TrainingFeedbackSystem::new().await?);
        
        Ok(Self {
            server,
            tool_registry,
            rag_system,
            knowledge_system,
            feedback_system,
            scheduler,
            sentience,
            trainer,
        })
    }
    
    async fn register_resources(
        server: &McpServer,
        scheduler: &Arc<Scheduler>,
        sentience: &Arc<SentienceManager>,
        trainer: &Arc<ModularTrainer>,
    ) -> Result<()> {
        // Inference metrics
        server.register_resource_handler(Arc::new(InferenceMetricsResource {
            scheduler: scheduler.clone(),
        })).await?;
        
        // Sentience state
        server.register_resource_handler(Arc::new(SentienceStateResource {
            identity_graph: sentience.identity_graph.clone(),
            motivations: sentience.motivations.clone(),
        })).await?;
        
        // Training state
        server.register_resource_handler(Arc::new(TrainingStateResource {
            trainer: trainer.clone(),
        })).await?;
        
        Ok(())
    }
    
    async fn register_tools(
        server: &McpServer,
        scheduler: &Arc<Scheduler>,
        sentience: &Arc<SentienceManager>,
        trainer: &Arc<ModularTrainer>,
    ) -> Result<()> {
        // Scheduling tools
        server.register_tool(
            Tool { /* ... */ },
            |args, scheduler| set_request_priority(args, scheduler),
        ).await?;
        
        // Sentience tools
        server.register_tool(
            Tool { /* ... */ },
            |args, capability_manager| unlock_capability(args, capability_manager),
        ).await?;
        
        // Training tools
        server.register_tool(
            Tool { /* ... */ },
            |args, trainer| adjust_learning_rate(args, trainer),
        ).await?;
        
        Ok(())
    }
    
    pub async fn run_server(&self) -> Result<()> {
        self.server.run_stdio().await
    }
    
    pub fn get_tool_registry(&self) -> Arc<ToolRegistry> {
        self.tool_registry.clone()
    }
    
    pub fn get_rag_system(&self) -> Arc<RAGSystem> {
        self.rag_system.clone()
    }
    
    pub fn get_knowledge_system(&self) -> Arc<ExternalKnowledgeSystem> {
        self.knowledge_system.clone()
    }
    
    pub fn get_feedback_system(&self) -> Arc<TrainingFeedbackSystem> {
        self.feedback_system.clone()
    }
}
```

---

## Integration Strategy

### **Phase 1: Foundation (Weeks 1-2)**

**Goals:**
- Add mocopr dependencies
- Create basic MCP server
- Expose inference metrics as resources
- Test with simple MCP client

**Tasks:**
1. Add `mocopr = { version = "0.1" }` to workspace dependencies
2. Create `lightbulb-mcp` crate
3. Implement basic resource handlers for metrics
4. Test stdio transport with Python client
5. Verify resource reading

**Deliverables:**
- Basic MCP server exposing scheduler metrics
- Working stdio communication
- Example Python client script

---

### **Phase 2: Server Expansion (Weeks 3-4)**

**Goals:**
- Add control tools (scheduling, configuration)
- Expand resource coverage
- Add WebSocket transport
- Implement security middleware

**Tasks:**
1. Implement scheduling control tools
2. Add KV cache metrics resources
3. Enable WebSocket transport
4. Add authentication middleware
5. Create comprehensive examples

**Deliverables:**
- Full scheduling control via MCP
- Multiple transport options
- Secure server implementation

---

### **Phase 3: Client Integration (Weeks 5-6)**

**Goals:**
- Integrate mocopr-client into inference pipeline
- Implement tool call detection
- Create tool registry
- Build RAG system

**Tasks:**
1. Add client support to inference engine
2. Implement tool call parsing in model output
3. Create ToolRegistry for managing multiple servers
4. Build RAG system with vector DB and web search
5. Test end-to-end tool-augmented inference

**Deliverables:**
- Models can call external tools during inference
- RAG system operational
- Tool call detection working

---

### **Phase 4: M7 Integration (Weeks 7-8)**

**Goals:**
- Expose sentience state as resources
- Add capability control tools
- Integrate external knowledge system
- Enable partnership building with real data

**Tasks:**
1. Create SentienceStateResource
2. Implement unlock_capability tool
3. Build ExternalKnowledgeSystem
4. Connect email/calendar MCP servers
5. Update partnership metrics with external context

**Deliverables:**
- Sentience state observable via MCP
- Capability gating controllable
- External knowledge integrated

---

### **Phase 5: M8 Integration (Weeks 9-10)**

**Goals:**
- Expose training state as resources
- Add training control tools
- Integrate validation feedback system
- Enable LLM-assisted architecture design

**Tasks:**
1. Create TrainingStateResource
2. Implement training control tools
3. Build TrainingFeedbackSystem
4. Connect code validators and test runners
5. Enable real-time feedback during training

**Deliverables:**
- Training observable and controllable via MCP
- Validation feedback integrated into training
- Module composition via MCP tools

---

## Time Savings Analysis

### **Without MoCoPr (Building from Scratch)**

| Component                                | Time Estimate   |
| ---------------------------------------- | --------------- |
| JSON-RPC 2.0 protocol implementation     | 3-4 weeks       |
| Transport layer (stdio, WebSocket, HTTP) | 2-3 weeks       |
| Resource/tool/prompt abstractions        | 2 weeks         |
| Client implementation                    | 2-3 weeks       |
| Security and authentication              | 2 weeks         |
| Session management                       | 1-2 weeks       |
| Error handling and retries               | 1 week          |
| Testing and hardening                    | 2 weeks         |
| **Total**                                | **15-21 weeks** |

### **With MoCoPr**

| Task                                     | Time Estimate   |
| ---------------------------------------- | --------------- |
| Add dependencies and basic setup         | 2-3 days        |
| Server implementation (resources, tools) | 2 weeks         |
| Client integration into inference        | 2 weeks         |
| M7 integration (sentience)               | 2 weeks         |
| M8 integration (training)                | 2 weeks         |
| Testing and refinement                   | 2 weeks         |
| **Total**                                | **10-11 weeks** |

**Net Savings: 5-10 weeks (1.25-2.5 months)** 🚀

---

## Advantages

### ✅ **Standard Protocol**
- MCP is becoming industry standard for LLM-tool interaction
- Interoperable with Claude, GPT-4, and other LLM platforms
- Growing ecosystem of MCP servers

### ✅ **Two-Way Communication**
- **Server**: Make Lightbulb observable and controllable
- **Client**: Give models access to external capabilities
- Unified architecture for both directions

### ✅ **Production-Ready Features**
- Multiple transports (stdio, WebSocket, HTTP)
- Security and authentication built-in
- Session management and error handling
- Monitoring and observability

### ✅ **Under Your Control**
- Can extend with Lightbulb-specific features
- Custom transport implementations
- Specialized security policies
- Performance optimizations

### ✅ **Agentic Foundation**
- Real-world interaction capability
- Tool composition and chaining
- Dynamic capability discovery
- Foundation for autonomous agents

---

## Complementary to Existing Infrastructure

| Infrastructure         | MoCoPr Addition                            |
| ---------------------- | ------------------------------------------ |
| **distributed-config** | Configuration accessible via MCP resources |
| **coalescent**         | Coalition formation tools exposed via MCP  |
| **infra-consensus**    | Consensus state as MCP resources           |
| **auto-discovery**     | Service discovery via MCP                  |
| **system-analysis**    | Hardware metrics as MCP resources          |

MoCoPr provides the **communication layer** that makes all infrastructure accessible to external systems and enables Lightbulb to access external capabilities.

---

## Next Steps

### **Immediate Actions:**

1. ✅ Add `mocopr` to `Cargo.toml`
2. 📋 Create `lightbulb-mcp` crate
3. 📋 Implement basic MCP server with metrics resources
4. 📋 Test with example Python client
5. 📋 Add client support to inference pipeline

### **Short-Term (Weeks 1-4):**

1. 📋 Full server implementation (resources, tools, prompts)
2. 📋 Client integration for tool-augmented inference
3. 📋 RAG system with external MCP servers
4. 📋 Security and authentication

### **Medium-Term (Weeks 5-8):**

1. 📋 M7 integration (sentience observability and control)
2. 📋 External knowledge system
3. 📋 Partnership building with real data
4. 📋 Capability gating via MCP

### **Long-Term (Weeks 9-10):**

1. 📋 M8 integration (training control and observability)
2. 📋 Training feedback system
3. 📋 LLM-assisted architecture design
4. 📋 Full agentic capabilities

---

## Summary

**MoCoPr is transformative for Lightbulb**, providing:

🔧 **Server**: Expose Lightbulb's capabilities to LLM supervisors  
🛠️ **Client**: Enable models to use external tools and knowledge  
🤖 **Agentic**: Foundation for autonomous, capable AI agents  
📊 **Observable**: Complete visibility into system state  
🎛️ **Controllable**: Dynamic control of all subsystems  
🌐 **Ecosystem**: Access to growing MCP server ecosystem  

**Result:** Lightbulb becomes a true **agentic platform** with real-world capabilities, not just an inference engine. Critical for M7 (sentience with genuine interaction) and M8 (training with real feedback)! 🚀

---

**Integration Status**: 📋 Ready to integrate  
**Dependencies**: mocopr 0.1.0 (mocopr-core, mocopr-server, mocopr-client)  
**Roadmap Alignment**: M4 (tools), M7 (sentience), M8 (training) - **ALL CRITICAL**  
**Next**: Add dependency and create basic MCP server
