// M5.D: Adaptive Context Selection
//
// Dynamic provider selection based on query analysis. Routes to the best
// context providers based on intent, entities, query complexity, and provider
// performance tracking.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::engine::context_injection::{ContextProvider, ProviderConfig, ProviderResult};
use crate::engine::query_analysis::AnalyzedQuery;

/// Provider selection strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionStrategy {
    /// Select all matching providers
    All,

    /// Select top N providers by confidence
    TopN(usize),

    /// Select providers until confidence threshold
    ThresholdBased,

    /// Performance-weighted selection
    PerformanceWeighted,
}

/// Configuration for adaptive selection
#[derive(Debug, Clone)]
pub struct SelectionConfig {
    pub strategy: SelectionStrategy,
    pub confidence_threshold: f32, // 0.0 to 1.0
    pub max_providers: usize,
    pub enable_performance_tracking: bool,
    pub enable_fallbacks: bool,
}

impl Default for SelectionConfig {
    fn default() -> Self {
        Self {
            strategy: SelectionStrategy::TopN(3),
            confidence_threshold: 0.6,
            max_providers: 5,
            enable_performance_tracking: true,
            enable_fallbacks: true,
        }
    }
}

/// Provider performance metrics
#[derive(Debug, Clone)]
pub struct ProviderMetrics {
    pub total_calls: usize,
    pub successful_calls: usize,
    pub failed_calls: usize,
    pub avg_latency_ms: f32,
    pub success_rate: f32,
}

impl ProviderMetrics {
    pub fn new() -> Self {
        Self {
            total_calls: 0,
            successful_calls: 0,
            failed_calls: 0,
            avg_latency_ms: 0.0,
            success_rate: 1.0,
        }
    }

    pub fn update(&mut self, success: bool, latency_ms: f32) {
        self.total_calls += 1;
        if success {
            self.successful_calls += 1;
        } else {
            self.failed_calls += 1;
        }

        // Exponential moving average for latency
        if self.total_calls == 1 {
            self.avg_latency_ms = latency_ms;
        } else {
            self.avg_latency_ms = 0.9 * self.avg_latency_ms + 0.1 * latency_ms;
        }

        self.success_rate = self.successful_calls as f32 / self.total_calls as f32;
    }
}

/// Provider with metadata
pub struct RegisteredProvider {
    pub provider: Arc<dyn ContextProvider>,
    pub config: ProviderConfig,
    pub confidence_scorer: Box<dyn Fn(&AnalyzedQuery) -> f32 + Send + Sync>,
    pub fallback_providers: Vec<String>,
}

/// Adaptive provider selector
pub struct ProviderSelector {
    config: SelectionConfig,
    providers: HashMap<String, RegisteredProvider>,
    metrics: Arc<Mutex<HashMap<String, ProviderMetrics>>>,
}

impl ProviderSelector {
    pub fn new(config: SelectionConfig) -> Self {
        Self {
            config,
            providers: HashMap::new(),
            metrics: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Register a provider with confidence scoring function
    pub fn register_provider<F>(
        &mut self,
        provider: Arc<dyn ContextProvider>,
        config: ProviderConfig,
        confidence_scorer: F,
    ) where
        F: Fn(&AnalyzedQuery) -> f32 + Send + Sync + 'static,
    {
        let provider_id = config.id.clone();

        self.providers.insert(
            provider_id.clone(),
            RegisteredProvider {
                provider,
                config,
                confidence_scorer: Box::new(confidence_scorer),
                fallback_providers: Vec::new(),
            },
        );

        self.metrics
            .lock()
            .unwrap()
            .insert(provider_id, ProviderMetrics::new());
    }

    /// Set fallback providers for a provider
    pub fn set_fallbacks(&mut self, provider_id: &str, fallbacks: Vec<String>) {
        if let Some(provider) = self.providers.get_mut(provider_id) {
            provider.fallback_providers = fallbacks;
        }
    }

    /// Select providers for a query
    pub fn select_providers(
        &self,
        query: &AnalyzedQuery,
        query_text: &str,
    ) -> Vec<(String, Arc<dyn ContextProvider>, f32)> {
        // Score all providers
        let mut scored: Vec<(String, Arc<dyn ContextProvider>, f32)> = self
            .providers
            .iter()
            .filter(|(_, reg)| self.matches_intent(reg, query))
            .filter_map(|(id, reg)| {
                let base_confidence = (reg.confidence_scorer)(query);
                let adjusted = if self.config.enable_performance_tracking {
                    self.adjust_for_performance(id, base_confidence)
                } else {
                    base_confidence
                };

                if adjusted >= self.config.confidence_threshold {
                    Some((id.clone(), reg.provider.clone(), adjusted))
                } else {
                    None
                }
            })
            .collect();

        // Sort by confidence (descending)
        scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

        // Apply selection strategy
        match self.config.strategy {
            SelectionStrategy::All => {
                scored.truncate(self.config.max_providers);
                scored
            }
            SelectionStrategy::TopN(n) => {
                scored.truncate(n.min(self.config.max_providers));
                scored
            }
            SelectionStrategy::ThresholdBased => {
                scored.retain(|(_, _, conf)| *conf >= self.config.confidence_threshold);
                scored.truncate(self.config.max_providers);
                scored
            }
            SelectionStrategy::PerformanceWeighted => {
                // Already adjusted for performance
                scored.truncate(self.config.max_providers);
                scored
            }
        }
    }

    /// Execute selected providers with fallback support
    pub fn execute_with_fallbacks(
        &self,
        query: &AnalyzedQuery,
        query_text: &str,
    ) -> Vec<ProviderResult> {
        let selected = self.select_providers(query, query_text);
        let mut results = Vec::new();

        for (provider_id, provider, _confidence) in selected {
            let start = std::time::Instant::now();

            match provider.provide_context(query, query_text) {
                Ok(contexts) => {
                    let latency = start.elapsed().as_millis() as f32;
                    if self.config.enable_performance_tracking {
                        self.update_metrics(&provider_id, true, latency);
                    }

                    results.push(ProviderResult {
                        provider_id: provider_id.clone(),
                        injections: contexts,
                        execution_time_ms: latency as u64,
                        success: true,
                        error: None,
                    });
                }
                Err(e) => {
                    let latency = start.elapsed().as_millis() as f32;
                    if self.config.enable_performance_tracking {
                        self.update_metrics(&provider_id, false, latency);
                    }

                    // Try fallbacks if enabled
                    if self.config.enable_fallbacks {
                        if let Some(fallback_result) =
                            self.try_fallbacks(&provider_id, query, query_text)
                        {
                            results.push(fallback_result);
                            continue;
                        }
                    }

                    results.push(ProviderResult {
                        provider_id: provider_id.clone(),
                        injections: Vec::new(),
                        execution_time_ms: latency as u64,
                        success: false,
                        error: Some(e.to_string()),
                    });
                }
            }
        }

        results
    }

    /// Check if provider matches query intent
    fn matches_intent(&self, provider: &RegisteredProvider, query: &AnalyzedQuery) -> bool {
        if provider.config.trigger_intents.is_empty() {
            return true; // No intent filter
        }

        provider.config.trigger_intents.contains(&query.intent)
    }

    /// Adjust confidence based on performance history
    fn adjust_for_performance(&self, provider_id: &str, base_confidence: f32) -> f32 {
        let metrics = self.metrics.lock().unwrap();

        if let Some(m) = metrics.get(provider_id) {
            // Penalty for low success rate
            let reliability_factor = m.success_rate;

            // Penalty for high latency (assume target is 100ms)
            let latency_factor = if m.avg_latency_ms > 100.0 {
                100.0 / m.avg_latency_ms
            } else {
                1.0
            };

            base_confidence * reliability_factor * latency_factor
        } else {
            base_confidence
        }
    }

    /// Update performance metrics
    fn update_metrics(&self, provider_id: &str, success: bool, latency_ms: f32) {
        let mut metrics = self.metrics.lock().unwrap();
        if let Some(m) = metrics.get_mut(provider_id) {
            m.update(success, latency_ms);
        }
    }

    /// Try fallback providers
    fn try_fallbacks(
        &self,
        failed_provider_id: &str,
        query: &AnalyzedQuery,
        query_text: &str,
    ) -> Option<ProviderResult> {
        let fallbacks = self
            .providers
            .get(failed_provider_id)?
            .fallback_providers
            .clone();

        for fallback_id in fallbacks {
            if let Some(registered) = self.providers.get(&fallback_id) {
                let start = std::time::Instant::now();

                if let Ok(contexts) = registered.provider.provide_context(query, query_text) {
                    let latency = start.elapsed().as_millis() as f32;
                    self.update_metrics(&fallback_id, true, latency);

                    return Some(ProviderResult {
                        provider_id: format!("{}_fallback", fallback_id),
                        injections: contexts,
                        execution_time_ms: latency as u64,
                        success: true,
                        error: None,
                    });
                }
            }
        }

        None
    }

    /// Get metrics for a provider
    pub fn get_metrics(&self, provider_id: &str) -> Option<ProviderMetrics> {
        self.metrics.lock().unwrap().get(provider_id).cloned()
    }

    /// Get all metrics
    pub fn get_all_metrics(&self) -> HashMap<String, ProviderMetrics> {
        self.metrics.lock().unwrap().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::context_injection::{ContextError, ContextInjection, InjectionPosition};
    use crate::engine::query_analysis::{QueryAnalyzer, QueryIntent};

    struct MockProvider {
        id: String,
        config: ProviderConfig,
        should_fail: bool,
    }

    impl MockProvider {
        fn new(id: &str, should_fail: bool) -> Self {
            Self {
                id: id.to_string(),
                config: ProviderConfig::new(id.to_string()),
                should_fail,
            }
        }
    }

    impl ContextProvider for MockProvider {
        fn id(&self) -> &str {
            &self.id
        }

        fn config(&self) -> &ProviderConfig {
            &self.config
        }

        fn provide_context(
            &self,
            _query: &AnalyzedQuery,
            _query_text: &str,
        ) -> Result<Vec<ContextInjection>, ContextError> {
            if self.should_fail {
                Err(ContextError::ProviderFailed(
                    self.id.clone(),
                    "Mock failure".to_string(),
                ))
            } else {
                Ok(vec![ContextInjection::new(
                    format!("Context from {}", self.id),
                    InjectionPosition::SystemMessage,
                    self.id.clone(),
                )])
            }
        }
    }

    #[test]
    fn test_selection_config_default() {
        let config = SelectionConfig::default();
        assert_eq!(config.strategy, SelectionStrategy::TopN(3));
        assert_eq!(config.confidence_threshold, 0.6);
        assert!(config.enable_performance_tracking);
    }

    #[test]
    fn test_provider_metrics_update() {
        let mut metrics = ProviderMetrics::new();

        metrics.update(true, 50.0);
        assert_eq!(metrics.successful_calls, 1);
        assert_eq!(metrics.total_calls, 1);
        assert_eq!(metrics.success_rate, 1.0);

        metrics.update(false, 100.0);
        assert_eq!(metrics.failed_calls, 1);
        assert_eq!(metrics.total_calls, 2);
        assert_eq!(metrics.success_rate, 0.5);
    }

    #[test]
    fn test_register_provider() {
        let mut selector = ProviderSelector::new(SelectionConfig::default());

        let provider = Arc::new(MockProvider::new("test", false));

        let config = ProviderConfig::new("test".to_string());
        selector.register_provider(provider, config, |_| 0.8);

        assert_eq!(selector.providers.len(), 1);
        assert!(selector.metrics.lock().unwrap().contains_key("test"));
    }

    #[test]
    fn test_select_providers_by_confidence() {
        let mut selector = ProviderSelector::new(SelectionConfig {
            strategy: SelectionStrategy::TopN(2),
            confidence_threshold: 0.5,
            ..Default::default()
        });

        // Register providers with different confidence scores
        for i in 0..3 {
            let id = format!("provider{}", i);
            let provider = Arc::new(MockProvider::new(&id, false));
            let confidence = 0.3 + (i as f32 * 0.2); // 0.3, 0.5, 0.7
            selector.register_provider(provider, ProviderConfig::new(id), move |_| confidence);
        }

        let analyzer = QueryAnalyzer::new();
        let query = analyzer.analyze("test query").unwrap();
        let selected = selector.select_providers(&query, "test query");

        // Should select top 2 providers above threshold (0.7 and 0.5)
        assert_eq!(selected.len(), 2);
        assert!(selected[0].2 > selected[1].2); // Sorted by confidence
    }

    #[test]
    fn test_execute_with_fallbacks() {
        let mut selector = ProviderSelector::new(SelectionConfig::default());

        // Primary provider that fails
        let primary = Arc::new(MockProvider::new("primary", true));
        selector.register_provider(primary, ProviderConfig::new("primary".to_string()), |_| 0.9);

        // Fallback provider that succeeds
        let fallback = Arc::new(MockProvider::new("fallback", false));
        selector.register_provider(
            fallback,
            ProviderConfig::new("fallback".to_string()),
            |_| 0.8,
        );

        // Set fallback chain
        selector.set_fallbacks("primary", vec!["fallback".to_string()]);

        let analyzer = QueryAnalyzer::new();
        let query = analyzer.analyze("test").unwrap();
        let results = selector.execute_with_fallbacks(&query, "test");

        // Should have results from fallback
        assert!(!results.is_empty());
        let has_fallback = results.iter().any(|r| r.provider_id.contains("fallback"));
        assert!(has_fallback);
    }

    #[test]
    fn test_performance_tracking() {
        let mut selector = ProviderSelector::new(SelectionConfig {
            enable_performance_tracking: true,
            ..Default::default()
        });

        let provider = Arc::new(MockProvider::new("tracked", false));
        selector.register_provider(provider, ProviderConfig::new("tracked".to_string()), |_| {
            0.9
        });

        let analyzer = QueryAnalyzer::new();
        let query = analyzer.analyze("test").unwrap();

        // Execute multiple times
        for _ in 0..3 {
            selector.execute_with_fallbacks(&query, "test");
        }

        let metrics = selector.get_metrics("tracked").unwrap();
        assert_eq!(metrics.total_calls, 3);
        assert_eq!(metrics.successful_calls, 3);
        assert_eq!(metrics.success_rate, 1.0);
    }

    #[test]
    fn test_intent_filtering() {
        let mut selector = ProviderSelector::new(SelectionConfig::default());

        let provider = Arc::new(MockProvider::new("code_only", false));

        let mut config = ProviderConfig::new("code_only".to_string());
        config.trigger_intents = vec![QueryIntent::Explanation];

        selector.register_provider(provider, config, |_| 0.9);

        let analyzer = QueryAnalyzer::new();

        // Explanation query should match
        let query1 = analyzer.analyze("explain this code").unwrap();
        let selected1 = selector.select_providers(&query1, "explain this code");
        assert!(!selected1.is_empty());

        // General query should not match
        let query2 = analyzer.analyze("hello world").unwrap();
        let selected2 = selector.select_providers(&query2, "hello world");
        assert!(selected2.is_empty() || selected2[0].0 != "code_only");
    }

    #[test]
    fn test_get_all_metrics() {
        let mut selector = ProviderSelector::new(SelectionConfig::default());

        let p1 = Arc::new(MockProvider::new("p1", false));
        let p2 = Arc::new(MockProvider::new("p2", false));

        selector.register_provider(p1, ProviderConfig::new("p1".to_string()), |_| 0.8);
        selector.register_provider(p2, ProviderConfig::new("p2".to_string()), |_| 0.7);

        let all_metrics = selector.get_all_metrics();
        assert_eq!(all_metrics.len(), 2);
        assert!(all_metrics.contains_key("p1"));
        assert!(all_metrics.contains_key("p2"));
    }
}
