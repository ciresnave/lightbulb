// M5.A: Streaming Context Injection
//
// Real-time context updates during generation. Allows providers to push context
// incrementally as tokens are generated, enabling dynamic context adaptation.
//
// Use Cases:
// - Code completion: Inject relevant imports as code is generated
// - Web search: Stream search results as query is refined
// - File watching: Push file changes as they happen
// - Chat history: Inject relevant past messages based on current topic
//
// Architecture:
// - StreamingContextProvider trait for push-based context
// - ContextStream for managing streaming updates
// - Token-triggered context injection
// - Buffer management and backpressure handling

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use crate::engine::context_injection::{ContextInjection, InjectionPosition};
use crate::engine::query_analysis::AnalyzedQuery;

/// Streaming context provider that can push updates during generation
pub trait StreamingContextProvider: Send + Sync {
    /// Provider identifier
    fn id(&self) -> &str;

    /// Start streaming context for a query
    fn start_stream(&self, query: &AnalyzedQuery) -> Result<(), String>;

    /// Called when a token is generated
    fn on_token(&self, token: &str, position: usize) -> Option<Vec<ContextInjection>>;

    /// Stop streaming and cleanup
    fn stop_stream(&self);

    /// Get buffered contexts if any
    fn poll_contexts(&self) -> Vec<ContextInjection>;
}

/// Configuration for streaming context
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Maximum buffer size (number of pending injections)
    pub max_buffer_size: usize,

    /// Minimum tokens between injections
    pub min_token_interval: usize,

    /// Maximum latency for context injection (ms)
    pub max_latency_ms: u64,

    /// Whether to enable backpressure (pause streaming if buffer full)
    pub enable_backpressure: bool,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            max_buffer_size: 100,
            min_token_interval: 10,
            max_latency_ms: 50,
            enable_backpressure: true,
        }
    }
}

/// Manages streaming context from multiple providers
pub struct ContextStream {
    config: StreamConfig,
    providers: Vec<Arc<dyn StreamingContextProvider>>,
    buffer: Arc<Mutex<VecDeque<ContextInjection>>>,
    last_injection_position: Arc<Mutex<Option<usize>>>,
    active: Arc<Mutex<bool>>,
}

impl ContextStream {
    pub fn new(config: StreamConfig) -> Self {
        Self {
            config,
            providers: Vec::new(),
            buffer: Arc::new(Mutex::new(VecDeque::new())),
            last_injection_position: Arc::new(Mutex::new(None)),
            active: Arc::new(Mutex::new(false)),
        }
    }

    /// Register a streaming provider
    pub fn register_provider(&mut self, provider: Arc<dyn StreamingContextProvider>) {
        self.providers.push(provider);
    }

    /// Start streaming for a query
    pub fn start(&self, query: &AnalyzedQuery) -> Result<(), String> {
        let mut active = self.active.lock().unwrap();
        *active = true;

        for provider in &self.providers {
            provider.start_stream(query)?;
        }

        Ok(())
    }

    /// Process a generated token
    pub fn on_token(&self, token: &str, position: usize) -> Vec<ContextInjection> {
        let active = self.active.lock().unwrap();
        if !*active {
            return Vec::new();
        }
        drop(active);

        let last_pos = *self.last_injection_position.lock().unwrap();
        // Check interval if we've had a previous injection
        if let Some(last) = last_pos {
            if position - last < self.config.min_token_interval {
                return Vec::new();
            }
        }

        let mut new_contexts = Vec::new();

        // Collect contexts from providers
        for provider in &self.providers {
            if let Some(contexts) = provider.on_token(token, position) {
                new_contexts.extend(contexts);
            }
        }

        // Add to buffer
        let mut buffer = self.buffer.lock().unwrap();
        for context in &new_contexts {
            if buffer.len() >= self.config.max_buffer_size {
                if self.config.enable_backpressure {
                    break; // Don't add more if buffer full
                } else {
                    buffer.pop_front(); // Drop oldest
                }
            }
            buffer.push_back(context.clone());
        }

        // Update last injection position
        if !new_contexts.is_empty() {
            *self.last_injection_position.lock().unwrap() = Some(position);
        }

        new_contexts
    }

    /// Get all buffered contexts
    pub fn drain_buffer(&self) -> Vec<ContextInjection> {
        let mut buffer = self.buffer.lock().unwrap();
        buffer.drain(..).collect()
    }

    /// Stop streaming
    pub fn stop(&self) {
        let mut active = self.active.lock().unwrap();
        *active = false;

        for provider in &self.providers {
            provider.stop_stream();
        }
    }

    /// Get buffer stats
    pub fn buffer_stats(&self) -> (usize, usize) {
        let buffer = self.buffer.lock().unwrap();
        (buffer.len(), self.config.max_buffer_size)
    }
}

/// Example: Code completion streaming provider
pub struct CodeCompletionStreamProvider {
    id: String,
    active: Arc<Mutex<bool>>,
    seen_imports: Arc<Mutex<HashMap<String, bool>>>,
}

impl CodeCompletionStreamProvider {
    pub fn new(id: String) -> Self {
        Self {
            id,
            active: Arc::new(Mutex::new(false)),
            seen_imports: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl StreamingContextProvider for CodeCompletionStreamProvider {
    fn id(&self) -> &str {
        &self.id
    }

    fn start_stream(&self, _query: &AnalyzedQuery) -> Result<(), String> {
        let mut active = self.active.lock().unwrap();
        *active = true;
        Ok(())
    }

    fn on_token(&self, token: &str, _position: usize) -> Option<Vec<ContextInjection>> {
        let active = self.active.lock().unwrap();
        if !*active {
            return None;
        }
        drop(active);

        // Detect if token looks like a type/function that needs an import
        if token.chars().next()?.is_uppercase() && token.len() > 2 {
            let mut seen = self.seen_imports.lock().unwrap();
            if !seen.contains_key(token) {
                seen.insert(token.to_string(), true);

                let import_suggestion = format!("// Consider importing: use crate::{};\n", token);
                let injection = ContextInjection::new(
                    import_suggestion,
                    InjectionPosition::BeforePrompt,
                    self.id.clone(),
                )
                .with_priority(70);

                return Some(vec![injection]);
            }
        }

        None
    }

    fn stop_stream(&self) {
        let mut active = self.active.lock().unwrap();
        *active = false;
        let mut seen = self.seen_imports.lock().unwrap();
        seen.clear();
    }

    fn poll_contexts(&self) -> Vec<ContextInjection> {
        Vec::new()
    }
}

/// Example: Real-time web search streaming provider
pub struct WebSearchStreamProvider {
    id: String,
    active: Arc<Mutex<bool>>,
    search_buffer: Arc<Mutex<Vec<String>>>,
}

impl WebSearchStreamProvider {
    pub fn new(id: String) -> Self {
        Self {
            id,
            active: Arc::new(Mutex::new(false)),
            search_buffer: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Simulate receiving search results (in real impl, this would be async)
    pub fn push_search_result(&self, result: String) {
        let mut buffer = self.search_buffer.lock().unwrap();
        buffer.push(result);
    }
}

impl StreamingContextProvider for WebSearchStreamProvider {
    fn id(&self) -> &str {
        &self.id
    }

    fn start_stream(&self, _query: &AnalyzedQuery) -> Result<(), String> {
        let mut active = self.active.lock().unwrap();
        *active = true;
        Ok(())
    }

    fn on_token(&self, _token: &str, _position: usize) -> Option<Vec<ContextInjection>> {
        None // Web search pushes via poll_contexts instead
    }

    fn stop_stream(&self) {
        let mut active = self.active.lock().unwrap();
        *active = false;
        let mut buffer = self.search_buffer.lock().unwrap();
        buffer.clear();
    }

    fn poll_contexts(&self) -> Vec<ContextInjection> {
        let active = self.active.lock().unwrap();
        if !*active {
            return Vec::new();
        }
        drop(active);

        let mut buffer = self.search_buffer.lock().unwrap();
        let results: Vec<_> = buffer.drain(..).collect();

        results
            .into_iter()
            .map(|result| {
                ContextInjection::new(
                    format!("Search result: {}\n", result),
                    InjectionPosition::SystemMessage,
                    self.id.clone(),
                )
                .with_priority(60)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::query_analysis::{QueryAnalyzer, QueryIntent};

    #[test]
    fn test_stream_config_default() {
        let config = StreamConfig::default();
        assert_eq!(config.max_buffer_size, 100);
        assert_eq!(config.min_token_interval, 10);
        assert!(config.enable_backpressure);
    }

    #[test]
    fn test_context_stream_creation() {
        let stream = ContextStream::new(StreamConfig::default());
        let (size, max) = stream.buffer_stats();
        assert_eq!(size, 0);
        assert_eq!(max, 100);
    }

    #[test]
    fn test_code_completion_provider() {
        let provider = CodeCompletionStreamProvider::new("code_completion".to_string());
        let analyzer = QueryAnalyzer::new();
        let query = analyzer.analyze("Write a HashMap").unwrap();

        provider.start_stream(&query).unwrap();

        // First occurrence of "HashMap" should trigger suggestion
        let contexts = provider.on_token("HashMap", 0);
        assert!(contexts.is_some());
        assert_eq!(contexts.as_ref().unwrap().len(), 1);

        // Second occurrence should not (already seen)
        let contexts2 = provider.on_token("HashMap", 10);
        assert!(contexts2.is_none());

        provider.stop_stream();
    }

    #[test]
    fn test_web_search_provider() {
        let provider = WebSearchStreamProvider::new("web_search".to_string());
        let analyzer = QueryAnalyzer::new();
        let query = analyzer.analyze("latest news").unwrap();

        provider.start_stream(&query).unwrap();

        // Push some search results
        provider.push_search_result("Result 1".to_string());
        provider.push_search_result("Result 2".to_string());

        // Poll should return buffered results
        let contexts = provider.poll_contexts();
        assert_eq!(contexts.len(), 2);
        assert!(contexts[0].content.contains("Result 1"));

        provider.stop_stream();
    }

    #[test]
    fn test_stream_token_interval() {
        let mut stream = ContextStream::new(StreamConfig {
            min_token_interval: 5,
            ..Default::default()
        });

        let provider = Arc::new(CodeCompletionStreamProvider::new("test".to_string()));
        stream.register_provider(provider);

        let analyzer = QueryAnalyzer::new();
        let query = analyzer.analyze("test").unwrap();
        stream.start(&query).unwrap();

        // Token at position 0 should work
        let result1 = stream.on_token("HashMap", 0);
        assert!(!result1.is_empty());

        // Token at position 3 should be skipped (< interval)
        let result2 = stream.on_token("Vector", 3);
        assert!(result2.is_empty());

        // Token at position 10 should work (>= interval)
        let result3 = stream.on_token("String", 10);
        assert!(!result3.is_empty());

        stream.stop();
    }

    #[test]
    fn test_stream_buffer_backpressure() {
        let mut stream = ContextStream::new(StreamConfig {
            max_buffer_size: 2,
            min_token_interval: 0,
            enable_backpressure: true,
            ..Default::default()
        });

        let provider = Arc::new(CodeCompletionStreamProvider::new("test".to_string()));
        stream.register_provider(provider);

        let analyzer = QueryAnalyzer::new();
        let query = analyzer.analyze("test").unwrap();
        stream.start(&query).unwrap();

        // Fill buffer
        stream.on_token("First", 0);
        stream.on_token("Second", 10);
        stream.on_token("Third", 20); // Should not add due to backpressure

        let (size, _) = stream.buffer_stats();
        assert_eq!(size, 2); // Should stop at max

        stream.stop();
    }

    #[test]
    fn test_stream_buffer_no_backpressure() {
        let mut stream = ContextStream::new(StreamConfig {
            max_buffer_size: 2,
            min_token_interval: 0,
            enable_backpressure: false,
            ..Default::default()
        });

        let provider = Arc::new(CodeCompletionStreamProvider::new("test".to_string()));
        stream.register_provider(provider);

        let analyzer = QueryAnalyzer::new();
        let query = analyzer.analyze("test").unwrap();
        stream.start(&query).unwrap();

        // Fill buffer beyond limit
        stream.on_token("First", 0);
        stream.on_token("Second", 10);
        stream.on_token("Third", 20); // Should evict "First"

        let buffered = stream.drain_buffer();
        assert_eq!(buffered.len(), 2);
        // Should have Second and Third (First was evicted)

        stream.stop();
    }

    #[test]
    fn test_stream_drain_buffer() {
        let mut stream = ContextStream::new(StreamConfig::default());
        let provider = Arc::new(CodeCompletionStreamProvider::new("test".to_string()));
        stream.register_provider(provider);

        let analyzer = QueryAnalyzer::new();
        let query = analyzer.analyze("test").unwrap();
        stream.start(&query).unwrap();

        stream.on_token("HashMap", 0);
        stream.on_token("Vector", 20);

        let buffered = stream.drain_buffer();
        assert_eq!(buffered.len(), 2);

        // Buffer should be empty after drain
        let (size, _) = stream.buffer_stats();
        assert_eq!(size, 0);

        stream.stop();
    }
}
