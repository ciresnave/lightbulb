# Distributed-Config Integration for Lightbulb

## Date: October 19, 2025

## Overview

**distributed-config** (0.1.0) is a robust configuration management library for distributed Rust applications. It provides unified configuration management with support for dynamic updates, distributed synchronization, and hierarchical organization - exactly what Lightbulb needs for managing complex federated ML workloads.

**Strategic Advantage**: Since this is your crate, we can extend it with Lightbulb-specific features as needed.

---

## What Distributed-Config Provides

### **1. Hierarchical Configuration**

```rust
use distributed_config::ConfigManager;

let mut config = ConfigManager::new();

// Hierarchical organization
config.set("server.host", "127.0.0.1").await?;
config.set("server.port", 8080).await?;
config.set("model.default", "mistral-7b").await?;
config.set("model.quantization", "4bit").await?;
```

**Lightbulb Use Cases:**
- Model configuration hierarchies
- Multi-tier scheduling policies
- Agent capability configurations
- Training hyperparameter trees

---

### **2. Multiple Configuration Sources**

```rust
use distributed_config::sources::{FileSource, EnvSource, RemoteSource};

// Priority-based source merging
config.add_source(FileSource::new().add_file("base.yaml", None), 10);    // Base config
config.add_source(EnvSource::new().with_prefix("LIGHTBULB_"), 20);       // Env overrides
config.add_source(RemoteSource::new("http://config-server/api"), 30);    // Remote overrides

config.initialize().await?;
```

**Sources Available:**
- **FileSource**: YAML, JSON, TOML files
- **EnvSource**: Environment variables with prefix filtering
- **RemoteSource**: HTTP/HTTPS configuration endpoints
- **CompositeSource**: Combine multiple sources

**Lightbulb Use Cases:**
- Base configuration from files
- Environment-specific overrides (dev/staging/prod)
- Federated configuration from remote nodes
- Multi-source policy aggregation

---

### **3. Dynamic Updates with Notifications**

```rust
use distributed_config::ConfigWatcher;

// Watch for configuration changes
let watcher = config.watch("model.default").await?;

tokio::spawn(async move {
    while let Some(change) = watcher.recv().await {
        println!("Model changed: {} -> {}", change.old_value, change.new_value);
        // Hot-reload model without restart
        reload_model(&change.new_value).await?;
    }
});
```

**Lightbulb Use Cases:**
- Hot-reload models without restart
- Dynamic scheduling policy updates
- Real-time capability flag changes
- Training hyperparameter adjustments mid-training

---

### **4. Schema Validation**

```rust
use distributed_config::SchemaValidator;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
struct ModelConfig {
    name: String,
    #[serde(default = "default_batch_size")]
    batch_size: usize,
    quantization: Option<String>,
}

// Validate configuration against schema
let validator = SchemaValidator::from_type::<ModelConfig>()?;
config.set_validator("model", validator).await?;

// Rejected if invalid
config.set("model.batch_size", "invalid").await?; // Error!
```

**Lightbulb Use Cases:**
- Prevent invalid model configurations
- Type-safe scheduling policies
- Validated training hyperparameters
- Schema evolution with versioning

---

### **5. Distributed Synchronization**

```rust
// Node 1: Update configuration
config.set("feature_flags.new_algorithm", true).await?;
config.sync().await?; // Broadcast to all nodes

// Node 2: Automatically receives update
let enabled = config.get::<bool>("feature_flags.new_algorithm").await?;
```

**Lightbulb Use Cases:**
- Synchronized feature flags across federated nodes
- Coordinated policy updates
- Cluster-wide configuration changes
- Emergency config rollbacks

---

### **6. Feature Flags**

```rust
// Built-in feature flag support
config.set_feature_flag("use_flash_attention", true).await?;
config.set_feature_flag("experimental_moe", false).await?;

// Conditional execution
if config.is_feature_enabled("use_flash_attention").await? {
    use_flash_attention_path();
} else {
    use_standard_attention();
}
```

**Lightbulb Use Cases:**
- Gradual rollout of new features
- A/B testing different algorithms
- Experimental feature toggles
- Capability gating (M7)

---

### **7. Versioning and Rollback**

```rust
// Configuration history tracking
let history = config.get_history("model.default").await?;

for entry in history {
    println!("Changed at {}: {}", entry.timestamp, entry.value);
}

// Rollback to previous version
config.rollback("model.default", 1).await?; // Roll back 1 version
```

**Lightbulb Use Cases:**
- Rollback bad configuration changes
- Audit configuration evolution
- Track policy modification history
- Debug configuration-related issues

---

## Roadmap Integration Points

### **M4 — Advanced Scheduling (0.5)**

**Current Roadmap Features:**
- Memory-aware, priority scheduler
- Metadata-driven scheduling enhancements
- Multi-stage inference pipeline orchestration

**Distributed-Config Integration:**

```rust
// Hierarchical scheduling configuration
#[derive(Deserialize, Serialize)]
struct SchedulingConfig {
    priority_classes: Vec<PriorityClass>,
    preemption_enabled: bool,
    batch_sizing: BatchSizingPolicy,
    kv_orchestration: KVOrchestrationConfig,
}

let mut config = ConfigManager::new();
config.add_source(FileSource::new().add_file("scheduling.yaml", None), 10);
config.initialize().await?;

let sched_config = config.get::<SchedulingConfig>("scheduling").await?;

// Watch for dynamic policy updates
let watcher = config.watch("scheduling.preemption_enabled").await?;
tokio::spawn(async move {
    while let Some(change) = watcher.recv().await {
        scheduler.update_preemption_policy(change.new_value).await?;
    }
});
```

**Benefits:**
- Dynamic scheduling policy updates without restart
- Hierarchical policy organization
- Environment-specific scheduling behavior
- Real-time tuning based on workload

---

### **M5 — Frontier Options (0.6)**

**Current Roadmap Features:**
- Three-tier dynamic memory management
- KV cache optimization
- Reasoning efficiency controls

**Distributed-Config Integration:**

```rust
// Memory tier configuration
#[derive(Deserialize, Serialize)]
struct MemoryConfig {
    tiers: Vec<MemoryTier>,
    eviction_policy: EvictionPolicy,
    admission_control: AdmissionConfig,
}

#[derive(Deserialize, Serialize)]
struct MemoryTier {
    name: String,        // "hot", "warm", "cold"
    size_mb: usize,
    medium: String,      // "vram", "ram", "ssd"
}

// Load and watch memory configuration
let mem_config = config.get::<MemoryConfig>("memory").await?;

// Hot-reload tier sizes
config.watch("memory.tiers").await?;
```

**Benefits:**
- Dynamic memory tier sizing
- Runtime eviction policy changes
- Adaptive admission control
- Per-workload memory budgets

---

### **M7 — Sentience Infrastructure (0.8+)**

**Current Roadmap Features:**
- Capability gating and resource control
- Partnership quality metrics
- Developmental stage progression
- Feature flags for capability unlocking

**Distributed-Config Integration:**

#### **1. Capability Gating**

```rust
// Progressive capability unlocking
#[derive(Deserialize, Serialize)]
struct CapabilityGates {
    basic_inference: CapabilityGate,
    knowledge_edit: CapabilityGate,
    system_config: CapabilityGate,
}

#[derive(Deserialize, Serialize)]
struct CapabilityGate {
    enabled: bool,
    trust_threshold: f64,
    cooperation_score_required: f64,
}

let gates = config.get::<CapabilityGates>("capabilities").await?;

// Dynamic capability unlocking based on trust
async fn check_capability_access(
    agent: &Agent,
    capability: &str,
    config: &ConfigManager
) -> bool {
    let gate_path = format!("capabilities.{}", capability);
    let gate = config.get::<CapabilityGate>(&gate_path).await.ok()?;
    
    gate.enabled && 
    agent.trust_score() >= gate.trust_threshold &&
    agent.cooperation_score() >= gate.cooperation_score_required
}
```

#### **2. Developmental Stages**

```rust
// Track developmental progression
#[derive(Deserialize, Serialize)]
struct DevelopmentalStages {
    current_stage: String,           // "infant", "child", "adolescent", "mature"
    progression_metrics: HashMap<String, f64>,
    unlocked_capabilities: Vec<String>,
}

// Automatically progress stages based on metrics
let stage_watcher = config.watch("development.progression_metrics").await?;
tokio::spawn(async move {
    while let Some(change) = stage_watcher.recv().await {
        if should_progress_stage(&change.new_value) {
            config.set("development.current_stage", next_stage()).await?;
            config.sync().await?; // Broadcast to all nodes
        }
    }
});
```

#### **3. Feature Flags for Sentience**

```rust
// Gradual sentience feature rollout
config.set_feature_flag("identity_graph_enabled", true).await?;
config.set_feature_flag("autonomous_rewards", false).await?; // Not ready yet
config.set_feature_flag("core_mind_separation", true).await?;

// Conditional sentience features
if config.is_feature_enabled("identity_graph_enabled").await? {
    identity_graph.track_self_concept(action).await?;
}
```

**Benefits:**
- Safe progressive capability unlocking
- Auditable capability evolution
- Distributed capability synchronization
- Emergency capability revocation

---

### **M8 — Modular Training Infrastructure (0.9+)**

**Current Roadmap Features:**
- Task decomposition framework
- Hardware-aware training optimization
- Training monitoring and telemetry
- Progressive fine-tuning

**Distributed-Config Integration:**

#### **1. Training Hyperparameters**

```rust
#[derive(Deserialize, Serialize)]
struct TrainingConfig {
    learning_rate: f64,
    batch_size: usize,
    epochs: usize,
    quantization: QuantizationConfig,
    checkpointing: CheckpointConfig,
}

// Load training config
let train_config = config.get::<TrainingConfig>("training").await?;

// Dynamic hyperparameter adjustment mid-training
let lr_watcher = config.watch("training.learning_rate").await?;
tokio::spawn(async move {
    while let Some(change) = lr_watcher.recv().await {
        trainer.set_learning_rate(change.new_value).await?;
        log::info!("Learning rate adjusted: {}", change.new_value);
    }
});
```

#### **2. Distributed Training Coordination**

```rust
// Coordinator node: Update training config
config.set("training.current_epoch", 5).await?;
config.set("training.global_step", 10000).await?;
config.sync().await?; // Broadcast to all training nodes

// Worker nodes: Monitor training state
let epoch_watcher = config.watch("training.current_epoch").await?;
let step_watcher = config.watch("training.global_step").await?;

// Coordinated checkpointing across nodes
if config.get::<usize>("training.global_step").await? % 1000 == 0 {
    worker.save_checkpoint().await?;
}
```

#### **3. Module Composition Configuration**

```rust
#[derive(Deserialize, Serialize)]
struct ModuleComposition {
    modules: Vec<ModuleConfig>,
    shims: Vec<ShimConfig>,
    routing: RoutingConfig,
}

#[derive(Deserialize, Serialize)]
struct ModuleConfig {
    name: String,
    path: PathBuf,
    frozen: bool,
    trust_score: f64,
}

// Dynamic module composition
let composition = config.get::<ModuleComposition>("composition").await?;

// Hot-swap modules during development
config.watch("composition.modules").await?;
```

**Benefits:**
- Dynamic hyperparameter tuning
- Distributed training coordination
- Module composition versioning
- Training state synchronization

---

### **Cross-Cutting Benefits**

#### **1. Environment-Specific Configurations**

```rust
// base.yaml (development)
server:
  host: "localhost"
  port: 8080
  debug: true
model:
  default: "tinyllama-1b"

// production.yaml (overrides)
server:
  host: "0.0.0.0"
  port: 443
  debug: false
model:
  default: "llama-3.2-11b"

// Load with environment-aware merging
config.add_source(FileSource::new().add_file("base.yaml", None), 10);
if env::var("ENVIRONMENT")? == "production" {
    config.add_source(FileSource::new().add_file("production.yaml", None), 20);
}
```

#### **2. Remote Configuration Management**

```rust
// Central configuration server for federated cluster
let remote = RemoteSource::new("https://config.lightbulb.ai/api")
    .with_auth_token(&token)
    .with_polling_interval(Duration::from_secs(30));

config.add_source(remote, 30); // Highest priority

// Nodes automatically receive cluster-wide updates
```

#### **3. Configuration History and Debugging**

```rust
// Debug why a model changed
let history = config.get_history("model.default").await?;

for entry in history.iter().rev().take(10) {
    println!("{}: {} (source: {})", 
        entry.timestamp, 
        entry.value,
        entry.source
    );
}

// Rollback problematic change
if model_not_working() {
    config.rollback("model.default", 1).await?;
}
```

---

## Integration Strategy

### **Phase 1: Foundation (Week 1)**

**Add Dependency:**
```toml
[workspace.dependencies]
distributed-config = { version = "0.1" }
```

**Create Base Configuration Structure:**
```yaml
# config/base.yaml
server:
  host: "127.0.0.1"
  port: 8080
  
model:
  default: "mistral-7b"
  quantization: "4bit"
  context_window: 4096
  
scheduling:
  priority_classes:
    - name: "high"
      timeout_ms: 1000
    - name: "normal"  
      timeout_ms: 5000
  preemption_enabled: true
  
memory:
  tiers:
    - name: "hot"
      size_mb: 8192
      medium: "vram"
    - name: "warm"
      size_mb: 16384
      medium: "ram"
    - name: "cold"
      size_mb: 102400
      medium: "ssd"
```

---

### **Phase 2: M4 Integration (Week 2-3)**

**Tasks:**
1. Integrate config manager into scheduler
2. Implement dynamic policy updates
3. Add configuration watchers for scheduling
4. Create hierarchical scheduling config

**Example:**
```rust
pub struct LightbulbScheduler {
    config: ConfigManager,
    // ... other fields
}

impl LightbulbScheduler {
    pub async fn new() -> Result<Self> {
        let mut config = ConfigManager::new();
        config.add_source(FileSource::new().add_file("config/base.yaml", None), 10);
        config.add_source(EnvSource::new().with_prefix("LIGHTBULB_"), 20);
        config.initialize().await?;
        
        // Watch for policy changes
        let preemption_watcher = config.watch("scheduling.preemption_enabled").await?;
        // ... spawn watcher tasks
        
        Ok(Self { config })
    }
}
```

---

### **Phase 3: M7 Integration (Week 4-5)**

**Tasks:**
1. Implement capability gating with config
2. Add developmental stage tracking
3. Create feature flag system
4. Implement distributed sync for sentience state

**Example:**
```rust
pub struct SentienceManager {
    config: ConfigManager,
    identity_graph: IdentityGraph,
}

impl SentienceManager {
    pub async fn check_capability(&self, agent: &Agent, capability: &str) -> bool {
        let gate_path = format!("capabilities.{}", capability);
        let gate: CapabilityGate = self.config.get(&gate_path).await.ok()?;
        
        gate.enabled && 
        agent.trust_score() >= gate.trust_threshold
    }
    
    pub async fn progress_development_stage(&mut self) -> Result<()> {
        let current = self.config.get::<String>("development.current_stage").await?;
        let next = calculate_next_stage(&current, &self.identity_graph)?;
        
        self.config.set("development.current_stage", next).await?;
        self.config.sync().await?; // Broadcast to cluster
        Ok(())
    }
}
```

---

### **Phase 4: M8 Integration (Week 6)

**

**Tasks:**
1. Training hyperparameter management
2. Distributed training coordination
3. Module composition configuration
4. Training state synchronization

---

## Extensibility Opportunities

Since distributed-config is your crate, we can extend it:

### **1. Lightbulb-Specific Validation**

```rust
// Add custom validators
pub struct ModelConfigValidator;

impl ConfigValidator for ModelConfigValidator {
    fn validate(&self, value: &ConfigValue) -> Result<()> {
        // Validate model exists
        // Check memory requirements
        // Verify quantization compatibility
    }
}
```

### **2. ML-Specific Configuration Types**

```rust
// Add ML config value types
pub enum MLConfigValue {
    ModelPath(PathBuf),
    Quantization(QuantizationType),
    SchedulingPolicy(Box<dyn SchedulingPolicy>),
    MemoryTier(MemoryTierConfig),
}
```

### **3. Federation-Aware Sync**

```rust
// Privacy-tiered configuration sync
pub enum ConfigSyncTier {
    Local,        // Never synced
    Trusted,      // Synced to trusted nodes
    Public,       // Synced to all nodes
}
```

---

## Time Savings Analysis

### **Without Distributed-Config (Building from Scratch)**

| Component                  | Time Estimate   |
| -------------------------- | --------------- |
| Hierarchical config system | 2 weeks         |
| Multi-source merging       | 1-2 weeks       |
| Dynamic updates & watchers | 2 weeks         |
| Schema validation          | 1-2 weeks       |
| Distributed sync           | 3-4 weeks       |
| Feature flags              | 1 week          |
| Versioning & rollback      | 2 weeks         |
| Testing & hardening        | 2 weeks         |
| **Total**                  | **14-19 weeks** |

### **With Distributed-Config**

| Component                     | Time Estimate |
| ----------------------------- | ------------- |
| Add dependency & setup        | 1 day         |
| Create config schemas         | 2-3 days      |
| Integrate with scheduler (M4) | 1 week        |
| Integrate with sentience (M7) | 1 week        |
| Integrate with training (M8)  | 1 week        |
| Testing & documentation       | 1 week        |
| **Total**                     | **4-5 weeks** |

**Net Savings: 10-14 weeks (2.5-3.5 months)** 🚀

---

## Advantages

### ✅ **Production-Ready Features**
- Hierarchical organization
- Multi-source merging with priorities
- Dynamic updates without restart
- Schema validation
- Distributed synchronization
- Built-in versioning

### ✅ **Perfect for Distributed ML**
- Federated configuration sync
- Dynamic hyperparameter tuning
- Cluster-wide policy updates
- Environment-specific configs

### ✅ **Under Your Control**
- Can extend with ML-specific types
- Add Lightbulb-specific validators
- Custom sync strategies
- Federation-aware features

### ✅ **Clean API Design**
- Intuitive hierarchical paths
- Type-safe configuration access
- Async-first with Tokio
- Well-documented

---

## Complementary to Existing Infrastructure

| Infrastructure         | Focus                    | Distributed-Config Addition      |
| ---------------------- | ------------------------ | -------------------------------- |
| **infra-storage**      | Data persistence         | Configuration persistence        |
| **infra-consensus**    | Raft consensus           | Configuration consensus          |
| **auto-discovery**     | Service discovery        | Remote config discovery          |
| **coalescent**         | Agent coordination       | Agent configuration              |
| **system-analysis**    | Hardware detection       | Hardware-based config selection  |
| **distributed-config** | Configuration management | **Unified config layer for all** |

Distributed-config provides the **configuration management layer** that ties all infrastructure together with consistent, validated, dynamically-updatable settings.

---

## Next Steps

### **Immediate Actions:**

1. ✅ Add `distributed-config` to `Cargo.toml`
2. 📋 Create base configuration files (base.yaml, development.yaml, production.yaml)
3. 📋 Integrate ConfigManager into Lightbulb core
4. 📋 Create configuration schemas for major subsystems
5. 📋 Implement configuration watchers for dynamic updates

### **M4 Integration (Future):**

1. 📋 Scheduling policy configuration
2. 📋 Memory tier configuration
3. 📋 Dynamic policy updates via watchers
4. 📋 Environment-specific scheduling behavior

### **M7 Integration (Future):**

1. 📋 Capability gating configuration
2. 📋 Developmental stage tracking
3. 📋 Feature flag system
4. 📋 Distributed sentience state sync

### **M8 Integration (Future):**

1. 📋 Training hyperparameter management
2. 📋 Distributed training coordination
3. 📋 Module composition configuration
4. 📋 Training state synchronization

---

## Summary

**Distributed-config is a perfect fit for Lightbulb**, providing:

⚙️ **Configuration Management** for all subsystems  
🔄 **Dynamic Updates** without restart  
🌐 **Distributed Sync** for federated clusters  
🎛️ **Feature Flags** for safe rollouts  
📜 **Versioning** for audit and rollback  
✅ **Schema Validation** for type safety  

**Strategic Value:**
- **10-14 weeks of development time saved**
- **Production-ready configuration system**
- **Foundation for M4, M7, M8 features**
- **Extensible with Lightbulb-specific types**

**Result:** Distributed-config provides the unified configuration management layer that enables safe, dynamic, validated configuration across Lightbulb's distributed infrastructure! 🚀

---

**Integration Status**: 📋 Ready to integrate  
**Dependencies**: distributed-config 0.1.0  
**Roadmap Alignment**: M4 (critical), M7 (critical), M8 (critical)  
**Next**: Add dependency and create base configuration structure
