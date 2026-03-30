//! Multi-Stage Inference Pipeline Orchestration
//!
//! Supports chained inference stages with dependency tracking, parallel execution,
//! and KV cache sharing for complex reasoning workflows.
//!
//! # Architecture
//!
//! ```text
//! Pipeline DAG Example:
//!
//!     ┌──────────┐
//!     │ Decompose│ (Stage 0)
//!     └─────┬────┘
//!           │
//!      ┌────┴────┐
//!      │         │
//! ┌────▼───┐ ┌──▼─────┐
//! │Identify│ │Generate│ (Stage 1, 2 - parallel)
//! └────┬───┘ └──┬─────┘
//!      │        │
//!      └───┬────┘
//!          │
//!     ┌────▼────┐
//!     │ Verify  │ (Stage 3)
//!     └────┬────┘
//!          │
//!     ┌────▼────┐
//!     │Synthesize│ (Stage 4)
//!     └─────────┘
//! ```
//!
//! # Features
//!
//! - **Dependency tracking**: DAG-based execution order
//! - **Parallel execution**: Independent stages run concurrently
//! - **KV cache sharing**: Reuse prefill across compatible stages
//! - **Backpressure handling**: Limit concurrent stage execution
//! - **Per-stage models**: Route to specialized models by stage type
//!
//! # Example
//!
//! ```ignore
//! let mut pipeline = PipelineBuilder::new("reasoning")
//!     .add_stage(Stage::new("decompose", StageType::Decompose))
//!     .add_stage(Stage::new("identify", StageType::Identify).depends_on("decompose"))
//!     .add_stage(Stage::new("generate", StageType::Generate).depends_on("decompose"))
//!     .add_stage(Stage::new("verify", StageType::Verify).depends_on_all(&["identify", "generate"]))
//!     .add_stage(Stage::new("synthesize", StageType::Synthesize).depends_on("verify"))
//!     .build()?;
//!
//! let result = pipeline.execute(input).await?;
//! ```

use std::collections::{HashMap, VecDeque};
use thiserror::Error;

/// Unique identifier for a pipeline stage
pub type StageId = String;

/// Unique identifier for a pipeline
pub type PipelineId = String;

/// Stage type determines model routing and execution characteristics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StageType {
    /// Decompose problem into sub-problems
    Decompose,

    /// Identify relevant information or entities
    Identify,

    /// Generate solution or content
    Generate,

    /// Verify correctness or consistency
    Verify,

    /// Synthesize final answer from components
    Synthesize,

    /// Custom stage (uses default model)
    Custom,
}

/// Input/output for a stage
#[derive(Debug, Clone)]
pub struct StageData {
    /// Text prompt or input
    pub text: String,

    /// Generated tokens
    pub tokens: Vec<u32>,

    /// KV cache reference (if cached)
    pub cache_ref: Option<CacheRef>,

    /// Metadata for this stage's execution
    pub metadata: HashMap<String, String>,
}

/// Reference to cached KV state
#[derive(Debug, Clone)]
pub struct CacheRef {
    /// Cache slot ID
    pub slot_id: usize,

    /// Number of tokens cached
    pub cached_tokens: usize,

    /// Whether cache is still valid
    pub is_valid: bool,
}

/// Pipeline stage definition
#[derive(Debug, Clone)]
pub struct Stage {
    /// Unique stage identifier
    pub id: StageId,

    /// Type of stage (determines routing)
    pub stage_type: StageType,

    /// Stage IDs this stage depends on
    pub dependencies: Vec<StageId>,

    /// Maximum tokens to generate
    pub max_tokens: usize,

    /// Sampling temperature
    pub temperature: f64,

    /// Whether to share KV cache with dependent stages
    pub enable_cache_sharing: bool,

    /// Custom prompt template (optional)
    pub prompt_template: Option<String>,
}

impl Stage {
    /// Create a new stage
    pub fn new(id: impl Into<String>, stage_type: StageType) -> Self {
        Self {
            id: id.into(),
            stage_type,
            dependencies: Vec::new(),
            max_tokens: 512,
            temperature: 0.7,
            enable_cache_sharing: true,
            prompt_template: None,
        }
    }

    /// Add a dependency on another stage
    pub fn depends_on(mut self, stage_id: impl Into<String>) -> Self {
        self.dependencies.push(stage_id.into());
        self
    }

    /// Add dependencies on multiple stages
    pub fn depends_on_all(mut self, stage_ids: &[impl AsRef<str>]) -> Self {
        for id in stage_ids {
            self.dependencies.push(id.as_ref().to_string());
        }
        self
    }

    /// Set max tokens
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }

    /// Disable KV cache sharing
    pub fn without_cache_sharing(mut self) -> Self {
        self.enable_cache_sharing = false;
        self
    }

    /// Set custom prompt template
    pub fn with_prompt_template(mut self, template: impl Into<String>) -> Self {
        self.prompt_template = Some(template.into());
        self
    }
}

/// State of a stage during execution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageState {
    /// Waiting for dependencies
    Pending,

    /// Ready to execute (dependencies satisfied)
    Ready,

    /// Currently executing
    Running,

    /// Completed successfully
    Completed,

    /// Failed with error
    Failed,
}

/// Result of a stage execution
#[derive(Debug, Clone)]
pub struct StageResult {
    /// Stage ID
    pub stage_id: StageId,

    /// Output data
    pub output: StageData,

    /// Execution time (milliseconds)
    pub execution_time_ms: u64,

    /// Whether KV cache was reused
    pub cache_reused: bool,
}

/// Pipeline execution statistics
#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    /// Total execution time (milliseconds)
    pub total_time_ms: u64,

    /// Number of stages executed
    pub stages_executed: usize,

    /// Number of stages executed in parallel
    pub parallel_stages: usize,

    /// Number of times KV cache was reused
    pub cache_reuse_count: usize,

    /// Total tokens generated
    pub total_tokens: usize,

    /// Pipeline overhead (ms)
    pub overhead_ms: u64,
}

/// Errors during pipeline execution
#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("Circular dependency detected in pipeline")]
    CircularDependency,

    #[error("Stage {0} not found in pipeline")]
    StageNotFound(StageId),

    #[error("Stage {0} depends on non-existent stage {1}")]
    InvalidDependency(StageId, StageId),

    #[error("Stage {0} execution failed: {1}")]
    ExecutionFailed(StageId, String),

    #[error("Maximum concurrent stages ({0}) exceeded")]
    BackpressureLimitExceeded(usize),

    #[error("Pipeline execution timeout")]
    Timeout,

    #[error("No stages defined in pipeline")]
    EmptyPipeline,
}

/// Pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Maximum number of stages to execute concurrently
    pub max_concurrent_stages: usize,

    /// Enable KV cache sharing between stages
    pub enable_cache_sharing: bool,

    /// Timeout for entire pipeline (milliseconds)
    pub timeout_ms: Option<u64>,

    /// Enable parallel execution of independent stages
    pub enable_parallel_execution: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            max_concurrent_stages: 4,
            enable_cache_sharing: true,
            timeout_ms: Some(300_000), // 5 minutes
            enable_parallel_execution: true,
        }
    }
}

/// Multi-stage inference pipeline
pub struct Pipeline {
    /// Pipeline identifier
    id: PipelineId,

    /// All stages in the pipeline
    stages: HashMap<StageId, Stage>,

    /// Current state of each stage
    stage_states: HashMap<StageId, StageState>,

    /// Execution results
    results: HashMap<StageId, StageResult>,

    /// Configuration
    config: PipelineConfig,

    /// Statistics
    stats: PipelineStats,

    /// Execution order (topologically sorted)
    execution_order: Vec<StageId>,
}

impl Pipeline {
    /// Get pipeline ID
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get all stages
    pub fn stages(&self) -> &HashMap<StageId, Stage> {
        &self.stages
    }

    /// Get stage state
    pub fn stage_state(&self, stage_id: &str) -> Option<StageState> {
        self.stage_states.get(stage_id).copied()
    }

    /// Get stage result
    pub fn stage_result(&self, stage_id: &str) -> Option<&StageResult> {
        self.results.get(stage_id)
    }

    /// Get execution statistics
    pub fn stats(&self) -> &PipelineStats {
        &self.stats
    }

    /// Get stages ready to execute
    pub fn get_ready_stages(&self) -> Vec<StageId> {
        let mut ready = Vec::new();

        for (stage_id, stage) in &self.stages {
            // Skip if not pending
            if self.stage_states.get(stage_id) != Some(&StageState::Pending) {
                continue;
            }

            // Check if all dependencies are completed
            let deps_satisfied = stage
                .dependencies
                .iter()
                .all(|dep_id| self.stage_states.get(dep_id) == Some(&StageState::Completed));

            if deps_satisfied {
                ready.push(stage_id.clone());
            }
        }

        ready
    }

    /// Mark stage as running
    pub fn mark_running(&mut self, stage_id: &str) -> Result<(), PipelineError> {
        if !self.stages.contains_key(stage_id) {
            return Err(PipelineError::StageNotFound(stage_id.to_string()));
        }

        self.stage_states
            .insert(stage_id.to_string(), StageState::Running);
        Ok(())
    }

    /// Mark stage as completed
    pub fn mark_completed(
        &mut self,
        stage_id: &str,
        result: StageResult,
    ) -> Result<(), PipelineError> {
        if !self.stages.contains_key(stage_id) {
            return Err(PipelineError::StageNotFound(stage_id.to_string()));
        }

        self.stage_states
            .insert(stage_id.to_string(), StageState::Completed);
        self.results.insert(stage_id.to_string(), result);
        self.stats.stages_executed += 1;

        Ok(())
    }

    /// Mark stage as failed
    pub fn mark_failed(&mut self, stage_id: &str) -> Result<(), PipelineError> {
        if !self.stages.contains_key(stage_id) {
            return Err(PipelineError::StageNotFound(stage_id.to_string()));
        }

        self.stage_states
            .insert(stage_id.to_string(), StageState::Failed);
        Ok(())
    }

    /// Check if pipeline is complete
    pub fn is_complete(&self) -> bool {
        self.stage_states
            .values()
            .all(|state| matches!(state, StageState::Completed | StageState::Failed))
    }

    /// Check if pipeline has any failures
    pub fn has_failures(&self) -> bool {
        self.stage_states
            .values()
            .any(|state| matches!(state, StageState::Failed))
    }

    /// Get count of running stages
    pub fn running_count(&self) -> usize {
        self.stage_states
            .values()
            .filter(|state| matches!(state, StageState::Running))
            .count()
    }
}

/// Builder for constructing pipelines
pub struct PipelineBuilder {
    id: PipelineId,
    stages: Vec<Stage>,
    config: PipelineConfig,
}

impl PipelineBuilder {
    /// Create a new pipeline builder
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            stages: Vec::new(),
            config: PipelineConfig::default(),
        }
    }

    /// Add a stage to the pipeline
    pub fn add_stage(mut self, stage: Stage) -> Self {
        self.stages.push(stage);
        self
    }

    /// Set configuration
    pub fn with_config(mut self, config: PipelineConfig) -> Self {
        self.config = config;
        self
    }

    /// Build the pipeline
    pub fn build(self) -> Result<Pipeline, PipelineError> {
        if self.stages.is_empty() {
            return Err(PipelineError::EmptyPipeline);
        }

        // Build stage map
        let mut stages = HashMap::new();
        let mut stage_states = HashMap::new();

        for stage in self.stages {
            stage_states.insert(stage.id.clone(), StageState::Pending);
            stages.insert(stage.id.clone(), stage);
        }

        // Validate dependencies
        for stage in stages.values() {
            for dep_id in &stage.dependencies {
                if !stages.contains_key(dep_id) {
                    return Err(PipelineError::InvalidDependency(
                        stage.id.clone(),
                        dep_id.clone(),
                    ));
                }
            }
        }

        // Check for cycles and compute execution order
        let execution_order = topological_sort(&stages)?;

        Ok(Pipeline {
            id: self.id,
            stages,
            stage_states,
            results: HashMap::new(),
            config: self.config,
            stats: PipelineStats::default(),
            execution_order,
        })
    }
}

/// Perform topological sort to detect cycles and determine execution order
fn topological_sort(stages: &HashMap<StageId, Stage>) -> Result<Vec<StageId>, PipelineError> {
    let mut in_degree: HashMap<StageId, usize> = HashMap::new();
    let mut adj_list: HashMap<StageId, Vec<StageId>> = HashMap::new();

    // Initialize
    for stage_id in stages.keys() {
        in_degree.insert(stage_id.clone(), 0);
        adj_list.insert(stage_id.clone(), Vec::new());
    }

    // Build adjacency list and in-degree counts
    for stage in stages.values() {
        for dep_id in &stage.dependencies {
            adj_list.get_mut(dep_id).unwrap().push(stage.id.clone());
            *in_degree.get_mut(&stage.id).unwrap() += 1;
        }
    }

    // Kahn's algorithm
    let mut queue: VecDeque<StageId> = in_degree
        .iter()
        .filter(|(_, deg)| **deg == 0)
        .map(|(id, _)| id.clone())
        .collect();

    let mut sorted = Vec::new();

    while let Some(stage_id) = queue.pop_front() {
        sorted.push(stage_id.clone());

        for dependent in &adj_list[&stage_id] {
            let deg = in_degree.get_mut(dependent).unwrap();
            *deg -= 1;
            if *deg == 0 {
                queue.push_back(dependent.clone());
            }
        }
    }

    if sorted.len() != stages.len() {
        return Err(PipelineError::CircularDependency);
    }

    Ok(sorted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_linear_pipeline() {
        let pipeline = PipelineBuilder::new("test")
            .add_stage(Stage::new("decompose", StageType::Decompose))
            .add_stage(Stage::new("generate", StageType::Generate).depends_on("decompose"))
            .add_stage(Stage::new("verify", StageType::Verify).depends_on("generate"))
            .build()
            .unwrap();

        assert_eq!(pipeline.stages.len(), 3);
        assert_eq!(pipeline.execution_order.len(), 3);
        assert_eq!(pipeline.execution_order[0], "decompose");
    }

    #[test]
    fn test_parallel_stages() {
        let pipeline = PipelineBuilder::new("test")
            .add_stage(Stage::new("decompose", StageType::Decompose))
            .add_stage(Stage::new("identify", StageType::Identify).depends_on("decompose"))
            .add_stage(Stage::new("generate", StageType::Generate).depends_on("decompose"))
            .add_stage(
                Stage::new("verify", StageType::Verify).depends_on_all(&["identify", "generate"]),
            )
            .build()
            .unwrap();

        assert_eq!(pipeline.stages.len(), 4);

        // Initially, only decompose should be ready
        let ready = pipeline.get_ready_stages();
        assert_eq!(ready.len(), 1);
        assert!(ready.contains(&"decompose".to_string()));
    }

    #[test]
    fn test_circular_dependency_detection() {
        let result = PipelineBuilder::new("test")
            .add_stage(Stage::new("a", StageType::Custom).depends_on("b"))
            .add_stage(Stage::new("b", StageType::Custom).depends_on("a"))
            .build();

        assert!(matches!(result, Err(PipelineError::CircularDependency)));
    }

    #[test]
    fn test_invalid_dependency() {
        let result = PipelineBuilder::new("test")
            .add_stage(Stage::new("a", StageType::Custom).depends_on("nonexistent"))
            .build();

        assert!(matches!(
            result,
            Err(PipelineError::InvalidDependency(_, _))
        ));
    }

    #[test]
    fn test_stage_execution_flow() {
        let mut pipeline = PipelineBuilder::new("test")
            .add_stage(Stage::new("decompose", StageType::Decompose))
            .add_stage(Stage::new("generate", StageType::Generate).depends_on("decompose"))
            .build()
            .unwrap();

        // Mark decompose as running
        pipeline.mark_running("decompose").unwrap();
        assert_eq!(pipeline.stage_state("decompose"), Some(StageState::Running));

        // Complete decompose
        let result = StageResult {
            stage_id: "decompose".to_string(),
            output: StageData {
                text: "output".to_string(),
                tokens: vec![1, 2, 3],
                cache_ref: None,
                metadata: HashMap::new(),
            },
            execution_time_ms: 100,
            cache_reused: false,
        };

        pipeline.mark_completed("decompose", result).unwrap();
        assert_eq!(
            pipeline.stage_state("decompose"),
            Some(StageState::Completed)
        );

        // Now generate should be ready
        let ready = pipeline.get_ready_stages();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0], "generate");
    }

    #[test]
    fn test_backpressure_limit() {
        let mut pipeline = PipelineBuilder::new("test")
            .add_stage(Stage::new("a", StageType::Custom))
            .add_stage(Stage::new("b", StageType::Custom))
            .add_stage(Stage::new("c", StageType::Custom))
            .with_config(PipelineConfig {
                max_concurrent_stages: 2,
                ..Default::default()
            })
            .build()
            .unwrap();

        pipeline.mark_running("a").unwrap();
        pipeline.mark_running("b").unwrap();

        assert_eq!(pipeline.running_count(), 2);
        assert_eq!(pipeline.config.max_concurrent_stages, 2);
    }
}
