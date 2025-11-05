//! Performance optimizations for M4 components.
//!
//! This module provides performance enhancements:
//! - Async context provider execution (parallel provider calls)
//! - LRU cache with TTL for context results
//! - Lazy loading for heavy components (embedding models)
//! - Connection pooling for external services
//!
//! Performance targets:
//! - Provider execution: <10ms overhead for async coordination
//! - Cache hit rate: >80% for repeated queries
//! - Memory efficiency: <100MB for cache + lazy models

use crate::engine::context_injection::{ContextInjection, ContextManager, ContextProvider};
use crate::engine::query_analysis::AnalyzedQuery;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

// ============================================================================
// Context Result Cache with LRU + TTL
// ============================================================================

/// Cache key based on query fingerprint.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct QueryFingerprint {
    query_text: String,
    intent: String,
    entities: Vec<String>,
}

impl QueryFingerprint {
    fn from_query(query_text: &str, analyzed: &AnalyzedQuery) -> Self {
        let mut entities: Vec<String> = analyzed.entities.iter().map(|e| e.text.clone()).collect();
        entities.sort();

        Self {
            query_text: query_text.to_lowercase(),
            intent: format!("{:?}", analyzed.intent),
            entities,
        }
    }
}

/// Cached context result with metadata.
#[derive(Debug, Clone)]
struct CachedContextResult {
    contexts: Vec<ContextInjection>,
    created_at: SystemTime,
    hit_count: usize,
    last_accessed: SystemTime,
}

impl CachedContextResult {
    fn is_expired(&self, ttl: Duration) -> bool {
        SystemTime::now()
            .duration_since(self.created_at)
            .map(|age| age > ttl)
            .unwrap_or(true)
    }

    fn record_hit(&mut self) {
        self.hit_count += 1;
        self.last_accessed = SystemTime::now();
    }
}

/// LRU cache configuration.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of cached entries.
    pub max_size: usize,
    /// Time-to-live for cache entries.
    pub ttl: Duration,
    /// Whether cache is enabled.
    pub enabled: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_size: 1000,
            ttl: Duration::from_secs(300), // 5 minutes
            enabled: true,
        }
    }
}

/// LRU cache for context provider results.
pub struct ContextCache {
    config: CacheConfig,
    cache: Arc<Mutex<HashMap<QueryFingerprint, CachedContextResult>>>,
}

impl ContextCache {
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config,
            cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Get cached contexts if available and not expired.
    pub fn get(&self, query_text: &str, analyzed: &AnalyzedQuery) -> Option<Vec<ContextInjection>> {
        if !self.config.enabled {
            return None;
        }

        let key = QueryFingerprint::from_query(query_text, analyzed);
        let mut cache = self.cache.lock().unwrap();

        if let Some(entry) = cache.get_mut(&key) {
            if !entry.is_expired(self.config.ttl) {
                entry.record_hit();
                return Some(entry.contexts.clone());
            } else {
                // Remove expired entry
                cache.remove(&key);
            }
        }

        None
    }

    /// Store contexts in cache.
    pub fn put(&self, query_text: &str, analyzed: &AnalyzedQuery, contexts: Vec<ContextInjection>) {
        if !self.config.enabled {
            return;
        }

        let key = QueryFingerprint::from_query(query_text, analyzed);
        let mut cache = self.cache.lock().unwrap();

        // Enforce size limit (LRU eviction)
        if cache.len() >= self.config.max_size {
            // Find least recently used entry
            if let Some(lru_key) = cache
                .iter()
                .min_by_key(|(_, v)| v.last_accessed)
                .map(|(k, _)| k.clone())
            {
                cache.remove(&lru_key);
            }
        }

        cache.insert(
            key,
            CachedContextResult {
                contexts,
                created_at: SystemTime::now(),
                hit_count: 0,
                last_accessed: SystemTime::now(),
            },
        );
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        let cache = self.cache.lock().unwrap();
        let total = cache.len();
        let expired = cache
            .values()
            .filter(|v| v.is_expired(self.config.ttl))
            .count();
        let total_hits: usize = cache.values().map(|v| v.hit_count).sum();

        CacheStats {
            total_entries: total,
            expired_entries: expired,
            total_hits,
        }
    }

    /// Clear all cache entries.
    pub fn clear(&self) {
        self.cache.lock().unwrap().clear();
    }
}

/// Cache statistics.
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub total_entries: usize,
    pub expired_entries: usize,
    pub total_hits: usize,
}

// ============================================================================
// Async Context Manager (Parallel Provider Execution)
// ============================================================================

/// Async-capable context manager for parallel provider execution.
///
/// Note: This is a synchronous wrapper that simulates async behavior.
/// In production, use tokio::spawn for true async execution.
pub struct AsyncContextManager {
    inner: ContextManager,
    cache: ContextCache,
    parallel_execution: bool,
}

impl AsyncContextManager {
    pub fn new(cache_config: CacheConfig) -> Self {
        Self {
            inner: ContextManager::new(),
            cache: ContextCache::new(cache_config),
            parallel_execution: true,
        }
    }

    pub fn with_parallel(mut self, enabled: bool) -> Self {
        self.parallel_execution = enabled;
        self
    }

    pub fn register_provider(&mut self, provider: Arc<dyn ContextProvider>) {
        self.inner.register_provider(provider);
    }

    /// Provide contexts with caching and parallel execution.
    ///
    /// In production, this would use tokio::spawn to execute providers
    /// in parallel. Current implementation: sequential with cache.
    pub fn provide_contexts_cached(
        &self,
        query: &AnalyzedQuery,
        query_text: &str,
    ) -> Vec<ContextInjection> {
        // Check cache first
        if let Some(cached) = self.cache.get(query_text, query) {
            return cached;
        }

        // Execute providers (would be parallel with tokio)
        let results = self.inner.provide_contexts(query, query_text);

        // Extract successful contexts
        let contexts: Vec<ContextInjection> = results
            .into_iter()
            .filter(|r| r.success)
            .flat_map(|r| r.injections)
            .collect();

        // Cache results
        self.cache.put(query_text, query, contexts.clone());

        contexts
    }

    /// Get cache statistics.
    pub fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }

    /// Clear cache.
    pub fn clear_cache(&self) {
        self.cache.clear();
    }
}

// ============================================================================
// Lazy Loading for Heavy Components
// ============================================================================

/// Lazy-loaded embedding model.
///
/// Delays model initialization until first use.
pub struct LazyEmbeddingModel {
    model: Arc<Mutex<Option<Box<dyn EmbeddingModel>>>>,
    config: EmbeddingModelConfig,
}

/// Trait for embedding models.
pub trait EmbeddingModel: Send + Sync {
    fn embed(&self, text: &str) -> Result<Vec<f32>, String>;
    fn dimension(&self) -> usize;
}

/// Configuration for embedding model.
#[derive(Debug, Clone)]
pub struct EmbeddingModelConfig {
    pub model_name: String,
    pub dimension: usize,
}

impl Default for EmbeddingModelConfig {
    fn default() -> Self {
        Self {
            model_name: "all-MiniLM-L6-v2".to_string(),
            dimension: 384,
        }
    }
}

/// Placeholder embedding model (replace with real implementation).
struct PlaceholderEmbeddingModel {
    config: EmbeddingModelConfig,
}

impl EmbeddingModel for PlaceholderEmbeddingModel {
    fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let seed = hasher.finish();

        let mut vec = Vec::with_capacity(self.config.dimension);
        let mut rng_state = seed;
        for _ in 0..self.config.dimension {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let value = ((rng_state >> 32) as f32) / (u32::MAX as f32);
            vec.push(value * 2.0 - 1.0);
        }

        // Normalize
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut vec {
                *v /= norm;
            }
        }

        Ok(vec)
    }

    fn dimension(&self) -> usize {
        self.config.dimension
    }
}

impl LazyEmbeddingModel {
    pub fn new(config: EmbeddingModelConfig) -> Self {
        Self {
            model: Arc::new(Mutex::new(None)),
            config,
        }
    }

    /// Get or initialize the model.
    fn get_or_init(&self) -> Result<Arc<Mutex<Option<Box<dyn EmbeddingModel>>>>, String> {
        let mut model_guard = self.model.lock().unwrap();

        if model_guard.is_none() {
            // Initialize model on first use
            let new_model = Box::new(PlaceholderEmbeddingModel {
                config: self.config.clone(),
            }) as Box<dyn EmbeddingModel>;
            *model_guard = Some(new_model);
        }

        Ok(Arc::clone(&self.model))
    }

    /// Embed text using lazy-loaded model.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
        let model_arc = self.get_or_init()?;
        let model_guard = model_arc.lock().unwrap();

        if let Some(ref model) = *model_guard {
            model.embed(text)
        } else {
            Err("Model not initialized".to_string())
        }
    }

    /// Check if model is loaded.
    pub fn is_loaded(&self) -> bool {
        self.model.lock().unwrap().is_some()
    }
}

// ============================================================================
// Connection Pool (for external services)
// ============================================================================

/// Simple connection pool for external API clients.
#[derive(Clone)]
pub struct ConnectionPool<T: Clone> {
    connections: Arc<Mutex<Vec<T>>>,
    max_size: usize,
}

impl<T: Clone> ConnectionPool<T> {
    pub fn new(max_size: usize) -> Self {
        Self {
            connections: Arc::new(Mutex::new(Vec::new())),
            max_size,
        }
    }

    /// Get a connection from pool or create new one.
    pub fn get<F>(&self, create: F) -> Result<T, String>
    where
        F: FnOnce() -> Result<T, String>,
    {
        let mut pool = self.connections.lock().unwrap();

        if let Some(conn) = pool.pop() {
            Ok(conn)
        } else if pool.len() < self.max_size {
            create()
        } else {
            Err("Connection pool exhausted".to_string())
        }
    }

    /// Return connection to pool.
    pub fn return_connection(&self, conn: T) {
        let mut pool = self.connections.lock().unwrap();
        if pool.len() < self.max_size {
            pool.push(conn);
        }
    }

    /// Get pool statistics.
    pub fn stats(&self) -> PoolStats {
        let pool = self.connections.lock().unwrap();
        PoolStats {
            available: pool.len(),
            max_size: self.max_size,
        }
    }
}

/// Connection pool statistics.
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub available: usize,
    pub max_size: usize,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ContextError;
    use crate::engine::context_injection::{InjectionPosition, ProviderConfig};
    use crate::engine::query_analysis::{QueryAnalyzer, QueryIntent};
    use std::thread;

    // Mock provider for testing
    struct MockProvider {
        config: ProviderConfig,
        delay_ms: u64,
    }

    impl MockProvider {
        fn new(id: &str, delay_ms: u64) -> Self {
            Self {
                config: ProviderConfig::new(id.to_string()),
                delay_ms,
            }
        }
    }

    impl ContextProvider for MockProvider {
        fn id(&self) -> &str {
            &self.config.id
        }

        fn config(&self) -> &ProviderConfig {
            &self.config
        }

        fn provide_context(
            &self,
            _query: &AnalyzedQuery,
            _query_text: &str,
        ) -> Result<Vec<ContextInjection>, ContextError> {
            // Simulate work
            thread::sleep(Duration::from_millis(self.delay_ms));

            let injection = ContextInjection {
                content: format!("Context from {}", self.id()),
                source: self.id().to_string(),
                priority: 50,
                position: InjectionPosition::BeforePrompt,
                metadata: HashMap::new(),
                essential: false,
            };

            Ok(vec![injection])
        }
    }

    #[test]
    fn test_query_fingerprint() {
        let analyzer = QueryAnalyzer::new();
        let query = analyzer.analyze("What is Rust?").unwrap();

        let fp1 = QueryFingerprint::from_query("What is Rust?", &query);
        let fp2 = QueryFingerprint::from_query("what is rust?", &query); // Case insensitive

        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_context_cache_basic() {
        let cache = ContextCache::new(CacheConfig::default());
        let analyzer = QueryAnalyzer::new();
        let query = analyzer.analyze("test query").unwrap();

        // Miss on first access
        assert!(cache.get("test query", &query).is_none());

        // Store in cache
        let contexts = vec![ContextInjection {
            content: "test content".to_string(),
            source: "test".to_string(),
            priority: 50,
            position: InjectionPosition::BeforePrompt,
            metadata: HashMap::new(),
            essential: false,
        }];
        cache.put("test query", &query, contexts.clone());

        // Hit on second access
        let cached = cache.get("test query", &query);
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().len(), 1);

        // Check stats
        let stats = cache.stats();
        assert_eq!(stats.total_entries, 1);
        assert_eq!(stats.total_hits, 1);
    }

    #[test]
    fn test_context_cache_expiration() {
        let config = CacheConfig {
            max_size: 10,
            ttl: Duration::from_millis(100), // Short TTL for testing
            enabled: true,
        };
        let cache = ContextCache::new(config);
        let analyzer = QueryAnalyzer::new();
        let query = analyzer.analyze("test").unwrap();

        let contexts = vec![ContextInjection {
            content: "test".to_string(),
            source: "test".to_string(),
            priority: 50,
            position: InjectionPosition::BeforePrompt,
            metadata: HashMap::new(),
            essential: false,
        }];

        cache.put("test", &query, contexts);

        // Should hit immediately
        assert!(cache.get("test", &query).is_some());

        // Wait for expiration
        thread::sleep(Duration::from_millis(150));

        // Should miss after expiration
        assert!(cache.get("test", &query).is_none());
    }

    #[test]
    fn test_context_cache_lru_eviction() {
        let config = CacheConfig {
            max_size: 2, // Small cache for testing
            ttl: Duration::from_secs(300),
            enabled: true,
        };
        let cache = ContextCache::new(config);
        let analyzer = QueryAnalyzer::new();

        let contexts = vec![ContextInjection {
            content: "test".to_string(),
            source: "test".to_string(),
            priority: 50,
            position: InjectionPosition::BeforePrompt,
            metadata: HashMap::new(),
            essential: false,
        }];

        // Fill cache
        let q1 = analyzer.analyze("query1").unwrap();
        cache.put("query1", &q1, contexts.clone());

        let q2 = analyzer.analyze("query2").unwrap();
        cache.put("query2", &q2, contexts.clone());

        // Both should be cached
        assert!(cache.get("query1", &q1).is_some());
        assert!(cache.get("query2", &q2).is_some());

        // Add third entry (should evict LRU)
        let q3 = analyzer.analyze("query3").unwrap();
        cache.put("query3", &q3, contexts.clone());

        // query1 should be evicted (least recently accessed)
        let stats = cache.stats();
        assert_eq!(stats.total_entries, 2);
    }

    #[test]
    fn test_async_context_manager() {
        let mut manager = AsyncContextManager::new(CacheConfig::default());

        let provider = Arc::new(MockProvider::new("test_provider", 10));
        manager.register_provider(provider);

        let analyzer = QueryAnalyzer::new();
        let query = analyzer.analyze("test query").unwrap();

        // First call - should execute provider
        let start = std::time::Instant::now();
        let contexts = manager.provide_contexts_cached(&query, "test query");
        let first_duration = start.elapsed();

        assert_eq!(contexts.len(), 1);
        assert_eq!(contexts[0].source, "test_provider");

        // Second call - should hit cache (faster)
        let start = std::time::Instant::now();
        let cached_contexts = manager.provide_contexts_cached(&query, "test query");
        let cached_duration = start.elapsed();

        assert_eq!(cached_contexts.len(), 1);
        assert!(cached_duration < first_duration);

        // Check cache stats
        let stats = manager.cache_stats();
        assert_eq!(stats.total_hits, 1);
    }

    #[test]
    fn test_lazy_embedding_model() {
        let model = LazyEmbeddingModel::new(EmbeddingModelConfig::default());

        // Model not loaded initially
        assert!(!model.is_loaded());

        // First embed triggers loading
        let embedding = model.embed("test text").unwrap();
        assert_eq!(embedding.len(), 384);

        // Model now loaded
        assert!(model.is_loaded());

        // Second embed uses loaded model
        let embedding2 = model.embed("test text").unwrap();
        assert_eq!(embedding, embedding2);
    }

    #[test]
    fn test_connection_pool() {
        let pool: ConnectionPool<String> = ConnectionPool::new(2);

        // Get connections
        let conn1 = pool.get(|| Ok("conn1".to_string())).unwrap();
        let conn2 = pool.get(|| Ok("conn2".to_string())).unwrap();

        // Return to pool
        pool.return_connection(conn1.clone());
        pool.return_connection(conn2.clone());

        // Check stats
        let stats = pool.stats();
        assert_eq!(stats.available, 2);
        assert_eq!(stats.max_size, 2);

        // Reuse from pool
        let reused = pool.get(|| Ok("new_conn".to_string())).unwrap();
        assert!(reused == "conn1" || reused == "conn2");
    }

    #[test]
    fn test_cache_disabled() {
        let config = CacheConfig {
            max_size: 100,
            ttl: Duration::from_secs(300),
            enabled: false, // Disabled
        };
        let cache = ContextCache::new(config);
        let analyzer = QueryAnalyzer::new();
        let query = analyzer.analyze("test").unwrap();

        let contexts = vec![ContextInjection {
            content: "test".to_string(),
            source: "test".to_string(),
            priority: 50,
            position: InjectionPosition::BeforePrompt,
            metadata: HashMap::new(),
            essential: false,
        }];

        cache.put("test", &query, contexts);

        // Should always miss when disabled
        assert!(cache.get("test", &query).is_none());

        let stats = cache.stats();
        assert_eq!(stats.total_entries, 0);
    }
}
