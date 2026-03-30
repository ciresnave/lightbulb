// M4.A: Metadata-Driven Scheduling
//
// Provides intelligent request routing based on query metadata.
// Uses M4.D QueryAnalyzer to automatically tag requests and route to
// specialized pipelines or models.
//
// Architecture:
// - RequestMetadata with priority, tags, context hints, ethical flags
// - RoutingPolicy for tag-based model/pipeline selection
// - ConstraintValidator for type checking and validation
// - MetadataScheduler for orchestrating routing decisions
//
// Key Features:
// - Automatic tagging from query analysis (intent → tags)
// - Tag-based routing (reasoning/factual/creative → specialized pipelines)
// - Priority scheduling (high priority requests jump queue)
// - Constraint satisfaction hooks (validation before execution)
// - Episode metadata logging for RL
//
// Performance Targets:
// - Tag-based routing: >90% accuracy
// - Constraint validation: <1ms overhead
// - Metadata logging: No p95 latency impact
//
// Integration:
// - Uses M4.D QueryAnalyzer for automatic tagging
// - Routes to appropriate inference pipelines
// - Logs metadata for training/debugging

use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::engine::query_analysis::{AnalyzedQuery, QueryIntent};

/// Request priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Priority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

impl fmt::Display for Priority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Priority::Low => write!(f, "low"),
            Priority::Normal => write!(f, "normal"),
            Priority::High => write!(f, "high"),
            Priority::Critical => write!(f, "critical"),
        }
    }
}

/// Request tags for routing
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RequestTag {
    /// Requires logical reasoning
    Reasoning,

    /// Factual information retrieval
    Factual,

    /// Creative generation
    Creative,

    /// Code generation or analysis
    Code,

    /// Mathematical computation
    Math,

    /// Comparison or analysis
    Comparison,

    /// Troubleshooting or debugging
    Troubleshooting,

    /// Procedural instructions
    Procedural,

    /// Custom tag
    Custom(String),
}

impl fmt::Display for RequestTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RequestTag::Reasoning => write!(f, "reasoning"),
            RequestTag::Factual => write!(f, "factual"),
            RequestTag::Creative => write!(f, "creative"),
            RequestTag::Code => write!(f, "code"),
            RequestTag::Math => write!(f, "math"),
            RequestTag::Comparison => write!(f, "comparison"),
            RequestTag::Troubleshooting => write!(f, "troubleshooting"),
            RequestTag::Procedural => write!(f, "procedural"),
            RequestTag::Custom(s) => write!(f, "custom:{}", s),
        }
    }
}

impl RequestTag {
    /// Convert QueryIntent to appropriate tags
    pub fn from_intent(intent: QueryIntent) -> Vec<Self> {
        match intent {
            QueryIntent::Definition => vec![RequestTag::Factual],
            QueryIntent::Procedure => vec![RequestTag::Procedural, RequestTag::Factual],
            QueryIntent::Comparison => vec![RequestTag::Comparison, RequestTag::Reasoning],
            QueryIntent::Troubleshooting => {
                vec![RequestTag::Troubleshooting, RequestTag::Reasoning]
            }
            QueryIntent::Explanation => vec![RequestTag::Reasoning, RequestTag::Factual],
            QueryIntent::Analysis => vec![RequestTag::Reasoning, RequestTag::Comparison],
            QueryIntent::Synthesis => vec![RequestTag::Reasoning, RequestTag::Creative],
            QueryIntent::Factual => vec![RequestTag::Factual],
            QueryIntent::Unknown => vec![],
        }
    }
}

/// Ethical flags for content filtering
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EthicalFlag {
    /// Contains potentially harmful content
    Harmful,

    /// Contains hate speech
    HateSpeech,

    /// Contains violent content
    Violence,

    /// Contains explicit sexual content
    Sexual,

    /// Contains personal information
    PersonalInfo,

    /// Misinformation risk
    Misinformation,

    /// Safe content
    Safe,
}

/// Request metadata for routing and logging
#[derive(Debug, Clone)]
pub struct RequestMetadata {
    /// Request identifier
    pub request_id: String,

    /// Priority level
    pub priority: Priority,

    /// Tags for routing
    pub tags: HashSet<RequestTag>,

    /// Context hints for the model
    pub context_hints: Vec<String>,

    /// Ethical flags
    pub ethical_flags: HashSet<EthicalFlag>,

    /// Custom metadata
    pub custom: HashMap<String, String>,

    /// Episode ID for RL training
    pub episode_id: Option<String>,
}

impl RequestMetadata {
    pub fn new(request_id: String) -> Self {
        Self {
            request_id,
            priority: Priority::Normal,
            tags: HashSet::new(),
            context_hints: Vec::new(),
            ethical_flags: HashSet::new(),
            custom: HashMap::new(),
            episode_id: None,
        }
    }

    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    pub fn add_tag(mut self, tag: RequestTag) -> Self {
        self.tags.insert(tag);
        self
    }

    pub fn add_context_hint(mut self, hint: String) -> Self {
        self.context_hints.push(hint);
        self
    }

    pub fn add_ethical_flag(mut self, flag: EthicalFlag) -> Self {
        self.ethical_flags.insert(flag);
        self
    }

    pub fn with_episode_id(mut self, episode_id: String) -> Self {
        self.episode_id = Some(episode_id);
        self
    }

    /// Auto-populate from analyzed query
    pub fn from_query(request_id: String, query: &AnalyzedQuery) -> Self {
        let mut metadata = Self::new(request_id);

        // Add tags from intent
        for tag in RequestTag::from_intent(query.intent) {
            metadata.tags.insert(tag);
        }

        // Add context hints from entities
        for entity in &query.entities {
            metadata.context_hints.push(entity.text.clone());
        }

        // Add code tag if code-related entities detected
        for entity in &query.entities {
            if entity.text.to_lowercase().contains("rust")
                || entity.text.to_lowercase().contains("python")
                || entity.text.to_lowercase().contains("code")
            {
                metadata.tags.insert(RequestTag::Code);
                break;
            }
        }

        metadata
    }
}

/// Pipeline or model identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PipelineId(pub String);

impl fmt::Display for PipelineId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Routing decision
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    /// Selected pipeline/model
    pub pipeline_id: PipelineId,

    /// Confidence in this routing (0.0 to 1.0)
    pub confidence: f64,

    /// Reason for this routing
    pub reason: String,

    /// Tags that influenced this decision
    pub matched_tags: HashSet<RequestTag>,
}

impl RoutingDecision {
    pub fn new(pipeline_id: PipelineId, confidence: f64, reason: String) -> Self {
        Self {
            pipeline_id,
            confidence,
            reason,
            matched_tags: HashSet::new(),
        }
    }

    pub fn with_matched_tags(mut self, tags: HashSet<RequestTag>) -> Self {
        self.matched_tags = tags;
        self
    }
}

/// Routing policy for tag-based selection
#[derive(Debug, Clone)]
pub struct RoutingPolicy {
    /// Policy name
    pub name: String,

    /// Tag patterns and their target pipelines
    rules: Vec<RoutingRule>,

    /// Default pipeline if no rules match
    default_pipeline: PipelineId,
}

#[derive(Debug, Clone)]
struct RoutingRule {
    /// Required tags (all must be present)
    required_tags: HashSet<RequestTag>,

    /// Optional tags (any can be present for bonus)
    optional_tags: HashSet<RequestTag>,

    /// Target pipeline
    target: PipelineId,

    /// Priority of this rule (higher = checked first)
    priority: u32,
}

impl RoutingPolicy {
    pub fn new(name: String, default_pipeline: PipelineId) -> Self {
        Self {
            name,
            rules: Vec::new(),
            default_pipeline,
        }
    }

    /// Add a routing rule
    pub fn add_rule(
        mut self,
        required_tags: Vec<RequestTag>,
        target: PipelineId,
        priority: u32,
    ) -> Self {
        self.rules.push(RoutingRule {
            required_tags: required_tags.into_iter().collect(),
            optional_tags: HashSet::new(),
            target,
            priority,
        });

        // Keep rules sorted by priority
        self.rules.sort_by(|a, b| b.priority.cmp(&a.priority));

        self
    }

    /// Route based on metadata
    pub fn route(&self, metadata: &RequestMetadata) -> RoutingDecision {
        // Check rules in priority order
        for rule in &self.rules {
            if rule.required_tags.is_subset(&metadata.tags) {
                let matched_tags = rule.required_tags.clone();
                let confidence = self.calculate_confidence(&matched_tags, &metadata.tags);

                return RoutingDecision::new(
                    rule.target.clone(),
                    confidence,
                    format!("Matched rule with tags: {:?}", matched_tags),
                )
                .with_matched_tags(matched_tags);
            }
        }

        // No rule matched, use default
        RoutingDecision::new(
            self.default_pipeline.clone(),
            0.5,
            "No specific rule matched, using default pipeline".to_string(),
        )
    }

    fn calculate_confidence(
        &self,
        matched_tags: &HashSet<RequestTag>,
        all_tags: &HashSet<RequestTag>,
    ) -> f64 {
        if all_tags.is_empty() {
            return 0.5;
        }

        // Confidence based on how many of the request's tags were matched
        let match_ratio = matched_tags.len() as f64 / all_tags.len() as f64;
        0.5 + (match_ratio * 0.5) // Range: 0.5 to 1.0
    }
}

/// Constraint validation errors
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintError {
    /// Type mismatch
    TypeMismatch(String, String), // expected, actual

    /// Value out of range
    OutOfRange(String),

    /// Required field missing
    MissingField(String),

    /// Schema validation failed
    SchemaViolation(String),

    /// Ethical constraint violated
    EthicalViolation(String),
}

impl fmt::Display for ConstraintError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConstraintError::TypeMismatch(expected, actual) => {
                write!(f, "Type mismatch: expected {}, got {}", expected, actual)
            }
            ConstraintError::OutOfRange(msg) => write!(f, "Out of range: {}", msg),
            ConstraintError::MissingField(field) => write!(f, "Missing required field: {}", field),
            ConstraintError::SchemaViolation(msg) => write!(f, "Schema violation: {}", msg),
            ConstraintError::EthicalViolation(msg) => write!(f, "Ethical violation: {}", msg),
        }
    }
}

impl std::error::Error for ConstraintError {}

/// Constraint validator
pub struct ConstraintValidator {
    /// Whether to enforce ethical constraints
    enforce_ethical: bool,

    /// Blocked tags for ethical filtering
    blocked_tags: HashSet<RequestTag>,
}

impl ConstraintValidator {
    pub fn new() -> Self {
        Self {
            enforce_ethical: true,
            blocked_tags: HashSet::new(),
        }
    }

    pub fn with_ethical_enforcement(mut self, enforce: bool) -> Self {
        self.enforce_ethical = enforce;
        self
    }

    pub fn block_tag(mut self, tag: RequestTag) -> Self {
        self.blocked_tags.insert(tag);
        self
    }

    /// Validate request metadata
    pub fn validate(&self, metadata: &RequestMetadata) -> Result<(), ConstraintError> {
        // Check for blocked tags
        for tag in &metadata.tags {
            if self.blocked_tags.contains(tag) {
                return Err(ConstraintError::SchemaViolation(format!(
                    "Tag {} is blocked",
                    tag
                )));
            }
        }

        // Check ethical constraints
        if self.enforce_ethical {
            for flag in &metadata.ethical_flags {
                match flag {
                    EthicalFlag::Harmful | EthicalFlag::HateSpeech | EthicalFlag::Violence => {
                        return Err(ConstraintError::EthicalViolation(format!(
                            "Request contains {:?} content",
                            flag
                        )));
                    }
                    _ => {}
                }
            }
        }

        Ok(())
    }
}

impl Default for ConstraintValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Metadata-driven scheduler
pub struct MetadataScheduler {
    /// Routing policy
    policy: RoutingPolicy,

    /// Constraint validator
    validator: ConstraintValidator,

    /// Episode metadata logging enabled
    log_episodes: bool,
}

impl MetadataScheduler {
    pub fn new(policy: RoutingPolicy, validator: ConstraintValidator) -> Self {
        Self {
            policy,
            validator,
            log_episodes: false,
        }
    }

    pub fn with_episode_logging(mut self, enabled: bool) -> Self {
        self.log_episodes = enabled;
        self
    }

    /// Schedule a request
    pub fn schedule(&self, metadata: &RequestMetadata) -> Result<RoutingDecision, ConstraintError> {
        // Validate constraints
        self.validator.validate(metadata)?;

        // Route based on metadata
        let decision = self.policy.route(metadata);

        // Log episode metadata if enabled
        if self.log_episodes {
            if let Some(episode_id) = &metadata.episode_id {
                // In production, this would log to a persistent store
                tracing::debug!(
                    episode_id = %episode_id,
                    request_id = %metadata.request_id,
                    pipeline = %decision.pipeline_id,
                    tags = ?metadata.tags,
                    "Episode metadata logged"
                );
            }
        }

        Ok(decision)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::query_analysis::QueryAnalyzer;

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Critical > Priority::High);
        assert!(Priority::High > Priority::Normal);
        assert!(Priority::Normal > Priority::Low);
    }

    #[test]
    fn test_priority_display() {
        assert_eq!(Priority::Low.to_string(), "low");
        assert_eq!(Priority::Normal.to_string(), "normal");
        assert_eq!(Priority::High.to_string(), "high");
        assert_eq!(Priority::Critical.to_string(), "critical");
    }

    #[test]
    fn test_request_tag_from_intent() {
        let tags = RequestTag::from_intent(QueryIntent::Definition);
        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0], RequestTag::Factual);

        let tags = RequestTag::from_intent(QueryIntent::Comparison);
        assert_eq!(tags.len(), 2);
        assert!(tags.contains(&RequestTag::Comparison));
        assert!(tags.contains(&RequestTag::Reasoning));
    }

    #[test]
    fn test_request_metadata_creation() {
        let metadata = RequestMetadata::new("req1".to_string())
            .with_priority(Priority::High)
            .add_tag(RequestTag::Reasoning)
            .add_context_hint("context1".to_string());

        assert_eq!(metadata.request_id, "req1");
        assert_eq!(metadata.priority, Priority::High);
        assert!(metadata.tags.contains(&RequestTag::Reasoning));
        assert_eq!(metadata.context_hints.len(), 1);
    }

    #[test]
    fn test_request_metadata_from_query() {
        let analyzer = QueryAnalyzer::new();
        let query = analyzer.analyze("What is Rust?").unwrap();

        let metadata = RequestMetadata::from_query("req1".to_string(), &query);

        assert_eq!(metadata.request_id, "req1");
        assert!(metadata.tags.contains(&RequestTag::Factual));
        assert!(metadata.tags.contains(&RequestTag::Code));
    }

    #[test]
    fn test_routing_policy_basic() {
        let policy =
            RoutingPolicy::new("test_policy".to_string(), PipelineId("default".to_string()))
                .add_rule(
                    vec![RequestTag::Reasoning],
                    PipelineId("reasoning_pipeline".to_string()),
                    100,
                );

        let metadata = RequestMetadata::new("req1".to_string()).add_tag(RequestTag::Reasoning);

        let decision = policy.route(&metadata);
        assert_eq!(decision.pipeline_id.0, "reasoning_pipeline");
        assert!(decision.confidence > 0.5);
    }

    #[test]
    fn test_routing_policy_default() {
        let policy =
            RoutingPolicy::new("test_policy".to_string(), PipelineId("default".to_string()));

        let metadata = RequestMetadata::new("req1".to_string()).add_tag(RequestTag::Factual);

        let decision = policy.route(&metadata);
        assert_eq!(decision.pipeline_id.0, "default");
        assert_eq!(decision.confidence, 0.5);
    }

    #[test]
    fn test_routing_policy_priority() {
        let policy =
            RoutingPolicy::new("test_policy".to_string(), PipelineId("default".to_string()))
                .add_rule(
                    vec![RequestTag::Reasoning],
                    PipelineId("reasoning".to_string()),
                    100,
                )
                .add_rule(
                    vec![RequestTag::Reasoning, RequestTag::Code],
                    PipelineId("code_reasoning".to_string()),
                    200, // Higher priority
                );

        let metadata = RequestMetadata::new("req1".to_string())
            .add_tag(RequestTag::Reasoning)
            .add_tag(RequestTag::Code);

        let decision = policy.route(&metadata);
        assert_eq!(decision.pipeline_id.0, "code_reasoning");
    }

    #[test]
    fn test_constraint_validator_pass() {
        let validator = ConstraintValidator::new();
        let metadata = RequestMetadata::new("req1".to_string())
            .add_tag(RequestTag::Factual)
            .add_ethical_flag(EthicalFlag::Safe);

        assert!(validator.validate(&metadata).is_ok());
    }

    #[test]
    fn test_constraint_validator_ethical_violation() {
        let validator = ConstraintValidator::new();
        let metadata =
            RequestMetadata::new("req1".to_string()).add_ethical_flag(EthicalFlag::Harmful);

        let result = validator.validate(&metadata);
        assert!(result.is_err());

        match result.unwrap_err() {
            ConstraintError::EthicalViolation(_) => {}
            _ => panic!("Expected EthicalViolation"),
        }
    }

    #[test]
    fn test_constraint_validator_blocked_tag() {
        let validator =
            ConstraintValidator::new().block_tag(RequestTag::Custom("blocked".to_string()));

        let metadata = RequestMetadata::new("req1".to_string())
            .add_tag(RequestTag::Custom("blocked".to_string()));

        let result = validator.validate(&metadata);
        assert!(result.is_err());
    }

    #[test]
    fn test_metadata_scheduler_basic() {
        let policy = RoutingPolicy::new("test".to_string(), PipelineId("default".to_string()))
            .add_rule(
                vec![RequestTag::Reasoning],
                PipelineId("reasoning".to_string()),
                100,
            );

        let validator = ConstraintValidator::new();
        let scheduler = MetadataScheduler::new(policy, validator);

        let metadata = RequestMetadata::new("req1".to_string())
            .add_tag(RequestTag::Reasoning)
            .add_ethical_flag(EthicalFlag::Safe);

        let decision = scheduler.schedule(&metadata).unwrap();
        assert_eq!(decision.pipeline_id.0, "reasoning");
    }

    #[test]
    fn test_metadata_scheduler_validation_failure() {
        let policy = RoutingPolicy::new("test".to_string(), PipelineId("default".to_string()));

        let validator = ConstraintValidator::new();
        let scheduler = MetadataScheduler::new(policy, validator);

        let metadata =
            RequestMetadata::new("req1".to_string()).add_ethical_flag(EthicalFlag::Harmful);

        let result = scheduler.schedule(&metadata);
        assert!(result.is_err());
    }

    #[test]
    fn test_metadata_scheduler_with_episode() {
        let policy = RoutingPolicy::new("test".to_string(), PipelineId("default".to_string()));

        let validator = ConstraintValidator::new();
        let scheduler = MetadataScheduler::new(policy, validator).with_episode_logging(true);

        let metadata = RequestMetadata::new("req1".to_string())
            .with_episode_id("episode123".to_string())
            .add_ethical_flag(EthicalFlag::Safe);

        let decision = scheduler.schedule(&metadata).unwrap();
        assert_eq!(decision.pipeline_id.0, "default");
    }

    #[test]
    fn test_routing_decision_confidence() {
        let decision = RoutingDecision::new(
            PipelineId("test".to_string()),
            0.85,
            "High confidence match".to_string(),
        );

        assert_eq!(decision.confidence, 0.85);
        assert!(decision.reason.contains("High confidence"));
    }

    #[test]
    fn test_request_metadata_custom_fields() {
        let mut metadata = RequestMetadata::new("req1".to_string());
        metadata
            .custom
            .insert("key1".to_string(), "value1".to_string());
        metadata
            .custom
            .insert("key2".to_string(), "value2".to_string());

        assert_eq!(metadata.custom.get("key1"), Some(&"value1".to_string()));
        assert_eq!(metadata.custom.len(), 2);
    }
}
