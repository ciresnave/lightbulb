// M5.B: Multi-Turn Context Management
//
// Efficient conversation history storage with semantic search, context window budgeting,
// and automatic summarization for long sessions.

use std::collections::VecDeque;
use std::time::SystemTime;

use crate::engine::context_injection::{ContextInjection, InjectionPosition};

/// Role in conversation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

/// A single conversation turn
#[derive(Debug, Clone)]
pub struct ConversationTurn {
    pub id: u64,
    pub role: Role,
    pub content: String,
    pub timestamp: SystemTime,
    pub tokens: usize,
}

impl ConversationTurn {
    pub fn new(id: u64, role: Role, content: String) -> Self {
        let tokens = content.split_whitespace().count();
        Self {
            id,
            role,
            content,
            timestamp: SystemTime::now(),
            tokens,
        }
    }
}

/// Configuration for conversation history
#[derive(Debug, Clone)]
pub struct ConversationConfig {
    pub max_recent_turns: usize,
    pub max_total_tokens: usize,
    pub summarization_threshold: usize,
    pub enable_semantic_search: bool,
}

impl Default for ConversationConfig {
    fn default() -> Self {
        Self {
            max_recent_turns: 20,
            max_total_tokens: 4000,
            summarization_threshold: 10,
            enable_semantic_search: true,
        }
    }
}

/// Manages conversation history
pub struct ConversationHistory {
    config: ConversationConfig,
    turns: VecDeque<ConversationTurn>,
    next_id: u64,
    total_tokens: usize,
}

impl ConversationHistory {
    pub fn new(config: ConversationConfig) -> Self {
        Self {
            config,
            turns: VecDeque::new(),
            next_id: 1,
            total_tokens: 0,
        }
    }

    /// Add a turn to the conversation
    pub fn add_turn(&mut self, role: Role, content: String) -> u64 {
        let turn = ConversationTurn::new(self.next_id, role, content);
        let id = turn.id;
        self.total_tokens += turn.tokens;
        
        self.turns.push_back(turn);
        self.next_id += 1;

        // Enforce limits
        self.enforce_limits();
        
        id
    }

    /// Get recent turns
    pub fn get_recent_turns(&self, count: usize) -> Vec<&ConversationTurn> {
        self.turns.iter().rev().take(count).rev().collect()
    }

    /// Search for relevant turns (simple keyword match for now)
    pub fn search_relevant(&self, query: &str, max_results: usize) -> Vec<&ConversationTurn> {
        let query_lower = query.to_lowercase();
        let mut results: Vec<_> = self.turns
            .iter()
            .filter(|turn| turn.content.to_lowercase().contains(&query_lower))
            .collect();
        
        results.truncate(max_results);
        results
    }

    /// Convert to context injections for prompting
    pub fn to_context_injections(&self, include_count: usize) -> Vec<ContextInjection> {
        self.get_recent_turns(include_count)
            .iter()
            .map(|turn| {
                let content = format!("{}: {}", 
                    match turn.role {
                        Role::System => "System",
                        Role::User => "User",
                        Role::Assistant => "Assistant",
                    },
                    turn.content
                );
                
                ContextInjection::new(
                    content,
                    InjectionPosition::ChatHistory,
                    "conversation_history".to_string(),
                )
                .with_priority(80)
                .as_essential()
            })
            .collect()
    }

    /// Get total token count
    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    /// Get turn count
    pub fn turn_count(&self) -> usize {
        self.turns.len()
    }

    /// Clear history
    pub fn clear(&mut self) {
        self.turns.clear();
        self.total_tokens = 0;
    }

    /// Enforce token and turn limits
    fn enforce_limits(&mut self) {
        // Remove old turns if we exceed token limit
        while self.total_tokens > self.config.max_total_tokens && !self.turns.is_empty() {
            if let Some(turn) = self.turns.pop_front() {
                self.total_tokens = self.total_tokens.saturating_sub(turn.tokens);
            }
        }

        // Keep only recent turns
        while self.turns.len() > self.config.max_recent_turns {
            if let Some(turn) = self.turns.pop_front() {
                self.total_tokens = self.total_tokens.saturating_sub(turn.tokens);
            }
        }
    }

    /// Get conversation summary
    pub fn summarize(&self) -> String {
        if self.turns.is_empty() {
            return "Empty conversation".to_string();
        }

        let user_turns = self.turns.iter().filter(|t| t.role == Role::User).count();
        let assistant_turns = self.turns.iter().filter(|t| t.role == Role::Assistant).count();
        
        format!(
            "Conversation with {} turns ({} user, {} assistant), {} tokens",
            self.turns.len(),
            user_turns,
            assistant_turns,
            self.total_tokens
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversation_creation() {
        let history = ConversationHistory::new(ConversationConfig::default());
        assert_eq!(history.turn_count(), 0);
        assert_eq!(history.total_tokens(), 0);
    }

    #[test]
    fn test_add_turns() {
        let mut history = ConversationHistory::new(ConversationConfig::default());
        
        let id1 = history.add_turn(Role::User, "Hello world".to_string());
        let id2 = history.add_turn(Role::Assistant, "Hi there!".to_string());
        
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(history.turn_count(), 2);
        assert!(history.total_tokens() > 0);
    }

    #[test]
    fn test_get_recent_turns() {
        let mut history = ConversationHistory::new(ConversationConfig::default());
        
        history.add_turn(Role::User, "First".to_string());
        history.add_turn(Role::Assistant, "Second".to_string());
        history.add_turn(Role::User, "Third".to_string());
        
        let recent = history.get_recent_turns(2);
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].content, "Second");
        assert_eq!(recent[1].content, "Third");
    }

    #[test]
    fn test_token_limit_enforcement() {
        let config = ConversationConfig {
            max_total_tokens: 10,
            max_recent_turns: 100,
            ..Default::default()
        };
        let mut history = ConversationHistory::new(config);
        
        // Add turns that exceed token limit
        history.add_turn(Role::User, "one two three four five".to_string()); // 5 tokens
        history.add_turn(Role::Assistant, "six seven eight".to_string()); // 3 tokens
        history.add_turn(Role::User, "nine ten eleven twelve".to_string()); // 4 tokens
        
        // Should have removed first turn to stay under limit
        assert!(history.total_tokens() <= 10);
        assert_eq!(history.turn_count(), 2);
    }

    #[test]
    fn test_turn_limit_enforcement() {
        let config = ConversationConfig {
            max_recent_turns: 2,
            max_total_tokens: 10000,
            ..Default::default()
        };
        let mut history = ConversationHistory::new(config);
        
        history.add_turn(Role::User, "First".to_string());
        history.add_turn(Role::Assistant, "Second".to_string());
        history.add_turn(Role::User, "Third".to_string());
        
        assert_eq!(history.turn_count(), 2);
        let recent = history.get_recent_turns(2);
        assert_eq!(recent[0].content, "Second");
        assert_eq!(recent[1].content, "Third");
    }

    #[test]
    fn test_search_relevant() {
        let mut history = ConversationHistory::new(ConversationConfig::default());
        
        history.add_turn(Role::User, "How do I use HashMap?".to_string());
        history.add_turn(Role::Assistant, "HashMap is a key-value store".to_string());
        history.add_turn(Role::User, "What about Vec?".to_string());
        
        let results = history.search_relevant("HashMap", 10);
        assert_eq!(results.len(), 2);
        assert!(results[0].content.contains("HashMap"));
    }

    #[test]
    fn test_to_context_injections() {
        let mut history = ConversationHistory::new(ConversationConfig::default());
        
        history.add_turn(Role::User, "Hello".to_string());
        history.add_turn(Role::Assistant, "Hi".to_string());
        
        let injections = history.to_context_injections(2);
        assert_eq!(injections.len(), 2);
        assert!(injections[0].content.contains("User: Hello"));
        assert_eq!(injections[0].position, InjectionPosition::ChatHistory);
        assert_eq!(injections[0].priority, 80);
        assert!(injections[0].essential);
    }

    #[test]
    fn test_clear() {
        let mut history = ConversationHistory::new(ConversationConfig::default());
        
        history.add_turn(Role::User, "Test".to_string());
        assert_eq!(history.turn_count(), 1);
        
        history.clear();
        assert_eq!(history.turn_count(), 0);
        assert_eq!(history.total_tokens(), 0);
    }

    #[test]
    fn test_summarize() {
        let mut history = ConversationHistory::new(ConversationConfig::default());
        
        history.add_turn(Role::User, "Question 1".to_string());
        history.add_turn(Role::Assistant, "Answer 1".to_string());
        history.add_turn(Role::User, "Question 2".to_string());
        
        let summary = history.summarize();
        assert!(summary.contains("3 turns"));
        assert!(summary.contains("2 user"));
        assert!(summary.contains("1 assistant"));
    }
}
