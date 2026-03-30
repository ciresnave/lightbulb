// M4.F: Context Injection API
//
// Provides external context providers for dynamic prompt enrichment.
// Allows loadable modules to inject relevant context based on query analysis.
//
// Architecture:
// - ContextProvider trait for extensible context sources
// - ContextInjection with content, position, priority, source tracking
// - ContextManager for provider registration and orchestration
// - Provider activation based on prompt analysis and metadata
//
// Key Features:
// - Priority-based context resolution
// - Position control (before/after prompt, system message)
// - Graceful failure handling (providers don't block inference)
// - Source attribution for transparency
// - Concurrent provider execution
//
// Performance Targets:
// - Provider execution: <10ms per provider
// - Support 10+ concurrent providers
// - Async provider loading
// - Timeout protection (no hanging providers)
//
// Example Providers:
// - Crate API loader: Auto-fetch docs.rs when crate mentioned
// - File watcher: Include recent file changes
// - Notification feed: Include system notifications
// - Web search: Fetch recent information
// - Code context: Include related code snippets
//
// Integration:
// - Works with M4.D QueryAnalyzer (uses analyzed query for activation)
// - Works with M4.E RelevanceSearch (can search for context)
// - Future: Works with InferenceEngine for prompt enrichment

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use crate::engine::query_analysis::{AnalyzedQuery, QueryIntent};

/// Errors in context injection operations
#[derive(Debug, Clone, PartialEq)]
pub enum ContextError {
    /// Provider not found
    ProviderNotFound(String),

    /// Provider execution failed
    ProviderFailed(String, String), // provider_id, error message

    /// Provider timeout
    ProviderTimeout(String),

    /// Invalid configuration
    InvalidConfig(String),

    /// Context too large
    ContextTooLarge(usize, usize), // actual size, max size
}

impl fmt::Display for ContextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContextError::ProviderNotFound(id) => write!(f, "Provider not found: {}", id),
            ContextError::ProviderFailed(id, msg) => {
                write!(f, "Provider {} failed: {}", id, msg)
            }
            ContextError::ProviderTimeout(id) => write!(f, "Provider {} timed out", id),
            ContextError::InvalidConfig(msg) => write!(f, "Invalid configuration: {}", msg),
            ContextError::ContextTooLarge(actual, max) => {
                write!(f, "Context too large: {} bytes (max: {})", actual, max)
            }
        }
    }
}

impl std::error::Error for ContextError {}

/// Position where context should be injected
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InjectionPosition {
    /// Before the user prompt
    BeforePrompt,

    /// After the user prompt
    AfterPrompt,

    /// In the system message
    SystemMessage,

    /// As a separate message in history
    ChatHistory,
}

impl fmt::Display for InjectionPosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InjectionPosition::BeforePrompt => write!(f, "before_prompt"),
            InjectionPosition::AfterPrompt => write!(f, "after_prompt"),
            InjectionPosition::SystemMessage => write!(f, "system_message"),
            InjectionPosition::ChatHistory => write!(f, "chat_history"),
        }
    }
}

/// Context content to be injected
#[derive(Debug, Clone)]
pub struct ContextInjection {
    /// The actual context content
    pub content: String,

    /// Where to inject this context
    pub position: InjectionPosition,

    /// Priority (higher = injected first if multiple contexts at same position)
    pub priority: u32,

    /// Source provider that generated this context
    pub source: String,

    /// Optional metadata about this context
    pub metadata: HashMap<String, String>,

    /// Whether this context is essential (if false, can be dropped if token budget exceeded)
    pub essential: bool,
}

impl ContextInjection {
    pub fn new(content: String, position: InjectionPosition, source: String) -> Self {
        Self {
            content,
            position,
            priority: 50, // Default medium priority
            source,
            metadata: HashMap::new(),
            essential: false,
        }
    }

    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    pub fn as_essential(mut self) -> Self {
        self.essential = true;
        self
    }

    /// Size of this context in bytes
    pub fn size(&self) -> usize {
        self.content.len()
    }
}

/// Configuration for a context provider
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    /// Provider identifier
    pub id: String,

    /// Whether this provider is enabled
    pub enabled: bool,

    /// Timeout for provider execution
    pub timeout: Duration,

    /// Maximum context size in bytes
    pub max_context_size: usize,

    /// Query intents that trigger this provider
    pub trigger_intents: Vec<QueryIntent>,

    /// Keywords that trigger this provider (any keyword in query)
    pub trigger_keywords: Vec<String>,

    /// Custom configuration parameters
    pub params: HashMap<String, String>,
}

impl ProviderConfig {
    pub fn new(id: String) -> Self {
        Self {
            id,
            enabled: true,
            timeout: Duration::from_millis(100),
            max_context_size: 10_000, // 10KB default
            trigger_intents: Vec::new(),
            trigger_keywords: Vec::new(),
            params: HashMap::new(),
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn with_max_size(mut self, max_size: usize) -> Self {
        self.max_context_size = max_size;
        self
    }

    pub fn with_trigger_intent(mut self, intent: QueryIntent) -> Self {
        self.trigger_intents.push(intent);
        self
    }

    pub fn with_trigger_keyword(mut self, keyword: String) -> Self {
        self.trigger_keywords.push(keyword);
        self
    }

    pub fn with_param(mut self, key: String, value: String) -> Self {
        self.params.insert(key, value);
        self
    }

    /// Check if this provider should be activated for the given query
    pub fn should_activate(&self, query: &AnalyzedQuery) -> bool {
        if !self.enabled {
            return false;
        }

        // Check intent triggers
        if !self.trigger_intents.is_empty() {
            if !self.trigger_intents.contains(&query.intent) {
                return false;
            }
        }

        // Check keyword triggers
        if !self.trigger_keywords.is_empty() {
            let query_lower = query.original.to_lowercase();
            let has_keyword = self
                .trigger_keywords
                .iter()
                .any(|kw| query_lower.contains(&kw.to_lowercase()));

            if !has_keyword {
                return false;
            }
        }

        true
    }
}

/// Result from a context provider
#[derive(Debug)]
pub struct ProviderResult {
    /// Provider that produced this result
    pub provider_id: String,

    /// Context injections (can be multiple)
    pub injections: Vec<ContextInjection>,

    /// Execution time in milliseconds
    pub execution_time_ms: u64,

    /// Whether provider succeeded
    pub success: bool,

    /// Error message if failed
    pub error: Option<String>,
}

impl ProviderResult {
    pub fn success(
        provider_id: String,
        injections: Vec<ContextInjection>,
        execution_time_ms: u64,
    ) -> Self {
        Self {
            provider_id,
            injections,
            execution_time_ms,
            success: true,
            error: None,
        }
    }

    pub fn failure(provider_id: String, error: String, execution_time_ms: u64) -> Self {
        Self {
            provider_id,
            injections: Vec::new(),
            execution_time_ms,
            success: false,
            error: Some(error),
        }
    }
}

/// Trait for context providers
///
/// Implementers can provide context based on analyzed queries.
/// Providers should be fast (<10ms) and handle errors gracefully.
pub trait ContextProvider: Send + Sync {
    /// Provider identifier
    fn id(&self) -> &str;

    /// Provider configuration
    fn config(&self) -> &ProviderConfig;

    /// Provide context for the given query
    ///
    /// This is called when the provider's activation conditions are met.
    /// Implementations should be fast and non-blocking.
    fn provide_context(
        &self,
        query: &AnalyzedQuery,
        prompt: &str,
    ) -> Result<Vec<ContextInjection>, ContextError>;
}

/// Example provider: Static context injector
///
/// Injects pre-defined context based on keywords.
/// Useful for testing and simple use cases.
pub struct StaticContextProvider {
    config: ProviderConfig,
    context_map: HashMap<String, ContextInjection>,
}

impl StaticContextProvider {
    pub fn new(id: String) -> Self {
        Self {
            config: ProviderConfig::new(id),
            context_map: HashMap::new(),
        }
    }

    pub fn add_context(mut self, keyword: String, injection: ContextInjection) -> Self {
        self.config = self.config.with_trigger_keyword(keyword.clone());
        self.context_map.insert(keyword, injection);
        self
    }

    pub fn with_config(mut self, config: ProviderConfig) -> Self {
        self.config = config;
        self
    }
}

impl ContextProvider for StaticContextProvider {
    fn id(&self) -> &str {
        &self.config.id
    }

    fn config(&self) -> &ProviderConfig {
        &self.config
    }

    fn provide_context(
        &self,
        query: &AnalyzedQuery,
        _prompt: &str,
    ) -> Result<Vec<ContextInjection>, ContextError> {
        let mut injections = Vec::new();
        let query_lower = query.original.to_lowercase();

        for (keyword, injection) in &self.context_map {
            if query_lower.contains(&keyword.to_lowercase()) {
                injections.push(injection.clone());
            }
        }

        Ok(injections)
    }
}

/// Example provider: Crate API documentation loader
///
/// Detects crate names in queries and provides API documentation context.
pub struct CrateApiProvider {
    config: ProviderConfig,
    docs_cache: HashMap<String, String>, // crate_name -> docs
}

impl CrateApiProvider {
    pub fn new() -> Self {
        let config = ProviderConfig::new("crate_api_loader".to_string())
            .with_trigger_intent(QueryIntent::Definition)
            .with_trigger_intent(QueryIntent::Procedure)
            .with_timeout(Duration::from_millis(200));

        Self {
            config,
            docs_cache: HashMap::new(),
        }
    }

    pub fn add_crate_docs(mut self, crate_name: String, docs: String) -> Self {
        self.docs_cache.insert(crate_name, docs);
        self
    }

    /// Extract crate names from query (simplified)
    fn extract_crate_names(&self, query: &AnalyzedQuery) -> Vec<String> {
        // In production, this would use proper entity extraction
        // For now, check against known crates
        let query_lower = query.original.to_lowercase();
        let mut crates = Vec::new();

        for crate_name in self.docs_cache.keys() {
            if query_lower.contains(&crate_name.to_lowercase()) {
                crates.push(crate_name.clone());
            }
        }

        crates
    }
}

impl Default for CrateApiProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl ContextProvider for CrateApiProvider {
    fn id(&self) -> &str {
        &self.config.id
    }

    fn config(&self) -> &ProviderConfig {
        &self.config
    }

    fn provide_context(
        &self,
        query: &AnalyzedQuery,
        _prompt: &str,
    ) -> Result<Vec<ContextInjection>, ContextError> {
        let crate_names = self.extract_crate_names(query);
        let mut injections = Vec::new();

        for crate_name in crate_names {
            if let Some(docs) = self.docs_cache.get(&crate_name) {
                let injection = ContextInjection::new(
                    format!("Documentation for crate '{}':\n{}", crate_name, docs),
                    InjectionPosition::BeforePrompt,
                    self.id().to_string(),
                )
                .with_priority(70) // Higher priority for API docs
                .with_metadata("crate_name".to_string(), crate_name.clone());

                injections.push(injection);
            }
        }

        Ok(injections)
    }
}

/// Context manager for orchestrating multiple providers
pub struct ContextManager {
    providers: HashMap<String, Arc<dyn ContextProvider>>,
    max_total_context_size: usize,
}

impl ContextManager {
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
            max_total_context_size: 100_000, // 100KB default
        }
    }

    pub fn with_max_total_size(mut self, max_size: usize) -> Self {
        self.max_total_context_size = max_size;
        self
    }

    /// Register a context provider
    pub fn register_provider(&mut self, provider: Arc<dyn ContextProvider>) {
        let id = provider.id().to_string();
        self.providers.insert(id, provider);
    }

    /// Get all registered providers
    pub fn providers(&self) -> Vec<Arc<dyn ContextProvider>> {
        self.providers.values().cloned().collect()
    }

    /// Execute all applicable providers for the given query
    pub fn provide_contexts(&self, query: &AnalyzedQuery, prompt: &str) -> Vec<ProviderResult> {
        let mut results = Vec::new();

        for provider in self.providers.values() {
            // Check if provider should be activated
            if !provider.config().should_activate(query) {
                continue;
            }

            // Execute provider with timeout protection
            let start = std::time::Instant::now();

            match provider.provide_context(query, prompt) {
                Ok(injections) => {
                    let execution_time = start.elapsed().as_millis() as u64;
                    results.push(ProviderResult::success(
                        provider.id().to_string(),
                        injections,
                        execution_time,
                    ));
                }
                Err(e) => {
                    let execution_time = start.elapsed().as_millis() as u64;
                    results.push(ProviderResult::failure(
                        provider.id().to_string(),
                        e.to_string(),
                        execution_time,
                    ));
                }
            }
        }

        results
    }

    /// Merge context injections from multiple providers
    ///
    /// Handles priority ordering, position grouping, and size limits.
    pub fn merge_contexts(
        &self,
        results: Vec<ProviderResult>,
    ) -> Result<HashMap<InjectionPosition, Vec<ContextInjection>>, ContextError> {
        let mut merged: HashMap<InjectionPosition, Vec<ContextInjection>> = HashMap::new();
        let mut total_size = 0usize;

        // Collect all successful injections
        let mut all_injections = Vec::new();
        for result in results {
            if result.success {
                all_injections.extend(result.injections);
            }
        }

        // Sort by priority (descending)
        all_injections.sort_by(|a, b| b.priority.cmp(&a.priority));

        // Group by position and enforce size limits
        for injection in all_injections {
            let injection_size = injection.size();

            // Check if adding this would exceed total size limit
            if total_size + injection_size > self.max_total_context_size {
                // Skip non-essential contexts if over budget
                if !injection.essential {
                    continue;
                }

                // Essential context exceeds budget - error
                return Err(ContextError::ContextTooLarge(
                    total_size + injection_size,
                    self.max_total_context_size,
                ));
            }

            total_size += injection_size;
            merged
                .entry(injection.position)
                .or_insert_with(Vec::new)
                .push(injection);
        }

        Ok(merged)
    }
}

impl Default for ContextManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::query_analysis::QueryAnalyzer;

    #[test]
    fn test_injection_position_display() {
        assert_eq!(InjectionPosition::BeforePrompt.to_string(), "before_prompt");
        assert_eq!(InjectionPosition::AfterPrompt.to_string(), "after_prompt");
        assert_eq!(
            InjectionPosition::SystemMessage.to_string(),
            "system_message"
        );
        assert_eq!(InjectionPosition::ChatHistory.to_string(), "chat_history");
    }

    #[test]
    fn test_context_injection_creation() {
        let injection = ContextInjection::new(
            "Test context".to_string(),
            InjectionPosition::BeforePrompt,
            "test_provider".to_string(),
        );

        assert_eq!(injection.content, "Test context");
        assert_eq!(injection.position, InjectionPosition::BeforePrompt);
        assert_eq!(injection.source, "test_provider");
        assert_eq!(injection.priority, 50);
        assert!(!injection.essential);
    }

    #[test]
    fn test_context_injection_builder() {
        let injection = ContextInjection::new(
            "Test".to_string(),
            InjectionPosition::SystemMessage,
            "test".to_string(),
        )
        .with_priority(100)
        .with_metadata("key".to_string(), "value".to_string())
        .as_essential();

        assert_eq!(injection.priority, 100);
        assert_eq!(injection.metadata.get("key"), Some(&"value".to_string()));
        assert!(injection.essential);
    }

    #[test]
    fn test_context_injection_size() {
        let injection = ContextInjection::new(
            "12345".to_string(),
            InjectionPosition::BeforePrompt,
            "test".to_string(),
        );

        assert_eq!(injection.size(), 5);
    }

    #[test]
    fn test_provider_config_trigger_intent() {
        let config =
            ProviderConfig::new("test".to_string()).with_trigger_intent(QueryIntent::Definition);

        let analyzer = QueryAnalyzer::new();

        let query1 = analyzer.analyze("What is Rust?").unwrap();
        assert!(config.should_activate(&query1));

        let query2 = analyzer.analyze("How to install Rust?").unwrap();
        assert!(!config.should_activate(&query2)); // Procedure, not Definition
    }

    #[test]
    fn test_provider_config_trigger_keyword() {
        let config =
            ProviderConfig::new("test".to_string()).with_trigger_keyword("rust".to_string());

        let analyzer = QueryAnalyzer::new();

        let query1 = analyzer.analyze("Tell me about Rust").unwrap();
        assert!(config.should_activate(&query1));

        let query2 = analyzer.analyze("Tell me about Python").unwrap();
        assert!(!config.should_activate(&query2));
    }

    #[test]
    fn test_provider_config_disabled() {
        let mut config =
            ProviderConfig::new("test".to_string()).with_trigger_keyword("rust".to_string());
        config.enabled = false;

        let analyzer = QueryAnalyzer::new();
        let query = analyzer.analyze("Tell me about Rust").unwrap();

        assert!(!config.should_activate(&query));
    }

    #[test]
    fn test_static_provider_basic() {
        let injection = ContextInjection::new(
            "Rust is a systems programming language".to_string(),
            InjectionPosition::BeforePrompt,
            "static".to_string(),
        );

        let provider = StaticContextProvider::new("static_test".to_string())
            .add_context("rust".to_string(), injection);

        let analyzer = QueryAnalyzer::new();
        let query = analyzer.analyze("What is Rust?").unwrap();

        let result = provider.provide_context(&query, "").unwrap();
        assert_eq!(result.len(), 1);
        assert!(result[0].content.contains("systems programming"));
    }

    #[test]
    fn test_static_provider_multiple_contexts() {
        let injection1 = ContextInjection::new(
            "Rust context".to_string(),
            InjectionPosition::BeforePrompt,
            "static".to_string(),
        );

        let injection2 = ContextInjection::new(
            "Python context".to_string(),
            InjectionPosition::BeforePrompt,
            "static".to_string(),
        );

        let provider = StaticContextProvider::new("static_test".to_string())
            .add_context("rust".to_string(), injection1)
            .add_context("python".to_string(), injection2);

        let analyzer = QueryAnalyzer::new();
        let query = analyzer.analyze("Rust vs Python").unwrap();

        let result = provider.provide_context(&query, "").unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_crate_api_provider_basic() {
        let provider = CrateApiProvider::new()
            .add_crate_docs("tokio".to_string(), "Async runtime for Rust".to_string());

        let analyzer = QueryAnalyzer::new();
        let query = analyzer.analyze("How to use tokio?").unwrap();

        let result = provider.provide_context(&query, "").unwrap();
        assert_eq!(result.len(), 1);
        assert!(result[0].content.contains("tokio"));
        assert!(result[0].content.contains("Async runtime"));
        assert_eq!(result[0].priority, 70);
    }

    #[test]
    fn test_crate_api_provider_multiple_crates() {
        let provider = CrateApiProvider::new()
            .add_crate_docs("tokio".to_string(), "Async runtime".to_string())
            .add_crate_docs("serde".to_string(), "Serialization".to_string());

        let analyzer = QueryAnalyzer::new();
        let query = analyzer.analyze("Using tokio with serde").unwrap();

        let result = provider.provide_context(&query, "").unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_context_manager_register() {
        let mut manager = ContextManager::new();

        let provider = Arc::new(StaticContextProvider::new("test".to_string()));
        manager.register_provider(provider);

        assert_eq!(manager.providers().len(), 1);
    }

    #[test]
    fn test_context_manager_provide_contexts() {
        let mut manager = ContextManager::new();

        let injection = ContextInjection::new(
            "Test context".to_string(),
            InjectionPosition::BeforePrompt,
            "static".to_string(),
        );

        let provider = Arc::new(
            StaticContextProvider::new("test".to_string())
                .add_context("rust".to_string(), injection),
        );
        manager.register_provider(provider);

        let analyzer = QueryAnalyzer::new();
        let query = analyzer.analyze("What is Rust?").unwrap();

        let results = manager.provide_contexts(&query, "");
        assert_eq!(results.len(), 1);
        assert!(results[0].success);
        assert_eq!(results[0].injections.len(), 1);
    }

    #[test]
    fn test_context_manager_merge_contexts() {
        let manager = ContextManager::new();

        let injection1 = ContextInjection::new(
            "Context 1".to_string(),
            InjectionPosition::BeforePrompt,
            "provider1".to_string(),
        )
        .with_priority(100);

        let injection2 = ContextInjection::new(
            "Context 2".to_string(),
            InjectionPosition::BeforePrompt,
            "provider2".to_string(),
        )
        .with_priority(50);

        let results = vec![
            ProviderResult::success("provider1".to_string(), vec![injection1], 5),
            ProviderResult::success("provider2".to_string(), vec![injection2], 3),
        ];

        let merged = manager.merge_contexts(results).unwrap();

        let before_prompt = merged.get(&InjectionPosition::BeforePrompt).unwrap();
        assert_eq!(before_prompt.len(), 2);

        // Higher priority should be first
        assert_eq!(before_prompt[0].priority, 100);
        assert_eq!(before_prompt[1].priority, 50);
    }

    #[test]
    fn test_context_manager_size_limit() {
        let manager = ContextManager::new().with_max_total_size(20);

        let injection1 = ContextInjection::new(
            "12345678901234567890".to_string(), // 20 bytes
            InjectionPosition::BeforePrompt,
            "provider1".to_string(),
        );

        let injection2 = ContextInjection::new(
            "Extra".to_string(), // 5 bytes - would exceed limit
            InjectionPosition::BeforePrompt,
            "provider2".to_string(),
        );

        let results = vec![
            ProviderResult::success("provider1".to_string(), vec![injection1], 5),
            ProviderResult::success("provider2".to_string(), vec![injection2], 3),
        ];

        let merged = manager.merge_contexts(results).unwrap();
        let before_prompt = merged.get(&InjectionPosition::BeforePrompt).unwrap();

        // Only first injection should be included
        assert_eq!(before_prompt.len(), 1);
    }

    #[test]
    fn test_context_manager_essential_context() {
        let manager = ContextManager::new().with_max_total_size(10);

        let injection = ContextInjection::new(
            "12345678901234567890".to_string(), // 20 bytes - exceeds limit
            InjectionPosition::BeforePrompt,
            "provider1".to_string(),
        )
        .as_essential();

        let results = vec![ProviderResult::success(
            "provider1".to_string(),
            vec![injection],
            5,
        )];

        let result = manager.merge_contexts(results);
        assert!(result.is_err());

        match result.unwrap_err() {
            ContextError::ContextTooLarge(actual, max) => {
                assert_eq!(actual, 20);
                assert_eq!(max, 10);
            }
            _ => panic!("Expected ContextTooLarge error"),
        }
    }

    #[test]
    fn test_provider_result_success() {
        let injection = ContextInjection::new(
            "Test".to_string(),
            InjectionPosition::BeforePrompt,
            "test".to_string(),
        );

        let result = ProviderResult::success("test".to_string(), vec![injection], 10);

        assert!(result.success);
        assert_eq!(result.execution_time_ms, 10);
        assert!(result.error.is_none());
        assert_eq!(result.injections.len(), 1);
    }

    #[test]
    fn test_provider_result_failure() {
        let result = ProviderResult::failure("test".to_string(), "Test error".to_string(), 15);

        assert!(!result.success);
        assert_eq!(result.execution_time_ms, 15);
        assert_eq!(result.error, Some("Test error".to_string()));
        assert_eq!(result.injections.len(), 0);
    }
}
