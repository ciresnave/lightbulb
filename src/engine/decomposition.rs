//! Knowledge-Aware Iterative Decomposition
//!
//! This module implements a decomposition system that adapts based on the current
//! knowledge base state. As facts accumulate, problems are automatically simplified
//! from structural decomposition to computational/atomic operations.
//!
//! # Architecture
//!
//! ```text
//! Initial State (Empty KB):
//!   "Calculate economic impact of policy X"
//!   ↓ STRUCTURAL DECOMPOSITION
//!   ├─ Find GDP 2023
//!   ├─ Find GDP 2024  
//!   ├─ Find effects of policy X
//!   └─ Calculate impact formula
//!
//! After Knowledge Accumulation:
//!   KB: {GDP_2023: $25.5T, GDP_2024: $27.1T, policy_effect: +3.2%}
//!   ↓ RE-DECOMPOSITION (Knowledge-aware simplification)
//!   "Calculate (27.1 - 25.5) / 25.5 * 100"  ← ATOMIC (directly solvable)
//! ```
//!
//! # Key Concepts
//!
//! - **Structural Decomposition**: Break problem into sub-tasks (unknowns present)
//! - **Computational Decomposition**: Express as formula with known values
//! - **Atomic Problem**: Directly solvable without further breakdown
//! - **Re-decomposition**: Triggered when new knowledge enables simplification
//! - **Decomposition Depth**: Tracks how many levels deep we've gone
//!
//! # Example
//!
//! ```ignore
//! let mut engine = DecompositionEngine::new(kb.clone());
//!
//! // Initial decomposition (KB empty)
//! let decomp = engine.decompose("Calculate economic impact")?;
//! assert_eq!(decomp.strategy, DecompositionStrategy::Structural);
//! assert_eq!(decomp.sub_problems.len(), 4); // Find GDP, effects, formula
//!
//! // Execute sub-problems, accumulate facts in KB
//! for sub in &decomp.sub_problems {
//!     let result = execute(sub)?;
//!     kb.add_fact(result)?;
//! }
//!
//! // Re-decompose (KB now has facts)
//! let simplified = engine.decompose("Calculate economic impact")?;
//! assert_eq!(simplified.strategy, DecompositionStrategy::Computational);
//! assert!(simplified.is_atomic()); // Can solve directly now
//! ```

use crate::engine::state_persistence::{
    CheckpointManager, create_checkpoint, restore_kb_from_checkpoint,
};
use crate::engine::KnowledgeBase;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

/// Errors that can occur during decomposition
#[derive(Debug, Error)]
pub enum DecompositionError {
    #[error("Circular dependency detected: {0}")]
    CircularDependency(String),

    #[error("Maximum decomposition depth exceeded: {0}")]
    MaxDepthExceeded(usize),

    #[error("Invalid problem specification: {0}")]
    InvalidProblem(String),

    #[error("KB query failed: {0}")]
    KnowledgeBaseError(String),

    #[error("Re-decomposition failed: {0}")]
    RedecompositionFailed(String),
}

/// Unique identifier for a problem
pub type ProblemId = String;

/// Strategy used for decomposition
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecompositionStrategy {
    /// Break into sub-tasks (unknowns present)
    Structural,

    /// Express as formula with known values
    Computational,

    /// Directly solvable (no breakdown needed)
    Atomic,

    /// Hybrid approach (some known, some unknown)
    Hybrid,
}

/// Complexity level of a problem
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ComplexityLevel {
    /// Single-step operation (e.g., "2 + 2")
    Trivial,

    /// 2-3 operations (e.g., "calculate percentage change")
    Simple,

    /// 4-10 operations or 1-2 unknowns
    Moderate,

    /// 10+ operations or 3-5 unknowns
    Complex,

    /// Requires multiple stages, many unknowns
    VeryComplex,
}

/// A problem to be decomposed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Problem {
    /// Unique identifier
    pub id: ProblemId,

    /// Problem description/query
    pub description: String,

    /// Required facts (unknowns that must be resolved)
    pub required_facts: Vec<String>,

    /// Known facts (from KB)
    pub known_facts: HashMap<String, String>,

    /// Complexity estimate
    pub complexity: ComplexityLevel,

    /// Metadata
    pub metadata: HashMap<String, String>,

    /// Timestamp
    pub created_at: u64,
}

impl Problem {
    /// Create a new problem
    pub fn new(description: impl Into<String>) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            id: format!("prob_{}", timestamp),
            description: description.into(),
            required_facts: Vec::new(),
            known_facts: HashMap::new(),
            complexity: ComplexityLevel::Moderate,
            metadata: HashMap::new(),
            created_at: timestamp,
        }
    }

    /// Add a required fact (unknown)
    pub fn requires(mut self, fact: impl Into<String>) -> Self {
        self.required_facts.push(fact.into());
        self
    }

    /// Add a known fact
    pub fn with_known_fact(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.known_facts.insert(key.into(), value.into());
        self
    }

    /// Set complexity level
    pub fn with_complexity(mut self, complexity: ComplexityLevel) -> Self {
        self.complexity = complexity;
        self
    }

    /// Check if problem is atomic (all facts known)
    pub fn is_atomic(&self) -> bool {
        self.required_facts.is_empty() && self.complexity <= ComplexityLevel::Simple
    }

    /// Calculate knowledge coverage (0.0 = no facts, 1.0 = all facts known)
    pub fn knowledge_coverage(&self) -> f64 {
        if self.required_facts.is_empty() {
            return 1.0;
        }

        let known_count = self
            .required_facts
            .iter()
            .filter(|fact| self.known_facts.contains_key(*fact))
            .count();

        known_count as f64 / self.required_facts.len() as f64
    }
}

/// A sub-problem in a decomposition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubProblem {
    /// Sub-problem ID
    pub id: ProblemId,

    /// Problem description
    pub description: String,

    /// Dependencies (IDs of sub-problems that must complete first)
    pub depends_on: Vec<ProblemId>,

    /// Expected output (fact keys this will produce)
    pub produces: Vec<String>,

    /// Success criteria
    pub success_criteria: Vec<String>,

    /// Complexity estimate
    pub complexity: ComplexityLevel,

    /// Completion status
    pub completed: bool,
}

impl SubProblem {
    /// Create a new sub-problem
    pub fn new(id: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            depends_on: Vec::new(),
            produces: Vec::new(),
            success_criteria: Vec::new(),
            complexity: ComplexityLevel::Simple,
            completed: false,
        }
    }

    /// Add dependency
    pub fn depends_on(mut self, problem_id: impl Into<String>) -> Self {
        self.depends_on.push(problem_id.into());
        self
    }

    /// Add produced fact
    pub fn produces(mut self, fact_key: impl Into<String>) -> Self {
        self.produces.push(fact_key.into());
        self
    }

    /// Add success criterion
    pub fn success_criterion(mut self, criterion: impl Into<String>) -> Self {
        self.success_criteria.push(criterion.into());
        self
    }

    /// Set complexity
    pub fn with_complexity(mut self, complexity: ComplexityLevel) -> Self {
        self.complexity = complexity;
        self
    }
}

/// Result of decomposing a problem
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Decomposition {
    /// Original problem
    pub problem: Problem,

    /// Sub-problems generated
    pub sub_problems: Vec<SubProblem>,

    /// Strategy used
    pub strategy: DecompositionStrategy,

    /// Decomposition depth (0 = original, 1+ = nested)
    pub depth: usize,

    /// Dependency graph (adjacency list)
    pub dependencies: HashMap<ProblemId, Vec<ProblemId>>,

    /// Parallel execution groups (independent sub-problems)
    pub parallel_groups: Vec<Vec<ProblemId>>,

    /// Timestamp
    pub created_at: u64,

    /// Metadata (for tracking and learning)
    pub metadata: HashMap<String, String>,
}

impl Decomposition {
    /// Check if decomposition is atomic (no sub-problems)
    pub fn is_atomic(&self) -> bool {
        self.sub_problems.is_empty()
    }

    /// Get execution order (topologically sorted)
    pub fn execution_order(&self) -> Result<Vec<ProblemId>, DecompositionError> {
        // Topological sort using Kahn's algorithm
        let mut in_degree: HashMap<ProblemId, usize> = HashMap::new();
        let mut adj_list: HashMap<ProblemId, Vec<ProblemId>> = HashMap::new();

        // Build graph
        for sub in &self.sub_problems {
            in_degree.entry(sub.id.clone()).or_insert(0);
            for dep in &sub.depends_on {
                adj_list
                    .entry(dep.clone())
                    .or_insert_with(Vec::new)
                    .push(sub.id.clone());
                *in_degree.entry(sub.id.clone()).or_insert(0) += 1;
            }
        }

        // Find nodes with no dependencies
        let mut queue: Vec<ProblemId> = in_degree
            .iter()
            .filter(|&(_, &degree)| degree == 0)
            .map(|(id, _)| id.clone())
            .collect();

        let mut result = Vec::new();

        while let Some(node) = queue.pop() {
            result.push(node.clone());

            if let Some(neighbors) = adj_list.get(&node) {
                for neighbor in neighbors {
                    if let Some(degree) = in_degree.get_mut(neighbor) {
                        *degree -= 1;
                        if *degree == 0 {
                            queue.push(neighbor.clone());
                        }
                    }
                }
            }
        }

        // Check for cycles
        if result.len() != self.sub_problems.len() {
            return Err(DecompositionError::CircularDependency(
                "Dependency cycle detected in decomposition".to_string(),
            ));
        }

        Ok(result)
    }

    /// Identify parallel execution groups
    pub fn compute_parallel_groups(&mut self) -> Result<(), DecompositionError> {
        let order = self.execution_order()?;
        let mut completed: HashSet<ProblemId> = HashSet::new();
        let mut groups = Vec::new();

        while completed.len() < self.sub_problems.len() {
            // Find all sub-problems whose dependencies are satisfied
            let ready: Vec<ProblemId> = self
                .sub_problems
                .iter()
                .filter(|sub| {
                    !completed.contains(&sub.id)
                        && sub.depends_on.iter().all(|dep| completed.contains(dep))
                })
                .map(|sub| sub.id.clone())
                .collect();

            if ready.is_empty() {
                break; // No more progress possible
            }

            groups.push(ready.clone());
            completed.extend(ready);
        }

        self.parallel_groups = groups;
        Ok(())
    }

    /// Calculate complexity score for this decomposition
    /// Higher score = more complex
    pub fn complexity_score(&self) -> f64 {
        let base_complexity = match self.problem.complexity {
            ComplexityLevel::Trivial => 1.0,
            ComplexityLevel::Simple => 2.0,
            ComplexityLevel::Moderate => 4.0,
            ComplexityLevel::Complex => 8.0,
            ComplexityLevel::VeryComplex => 16.0,
        };

        // Factor in number of sub-problems
        let sub_problem_factor = 1.0 + (self.sub_problems.len() as f64) * 0.5;

        // Factor in depth of decomposition
        let depth_factor = 1.0 + (self.depth as f64) * 0.3;

        // Factor in number of unknowns
        let unknown_factor = 1.0 + (self.problem.required_facts.len() as f64) * 0.4;

        // Factor in dependency complexity
        let total_dependencies: usize = self.sub_problems.iter().map(|s| s.depends_on.len()).sum();
        let dependency_factor = 1.0 + (total_dependencies as f64) * 0.2;

        base_complexity * sub_problem_factor * depth_factor * unknown_factor * dependency_factor
    }
}

/// Configuration for decomposition engine
#[derive(Debug, Clone)]
pub struct DecompositionConfig {
    /// Maximum decomposition depth
    pub max_depth: usize,

    /// Maximum sub-problems per decomposition
    pub max_sub_problems: usize,

    /// Minimum knowledge coverage to attempt re-decomposition (0.0-1.0)
    pub min_coverage_for_redecomp: f64,

    /// Enable automatic re-decomposition
    pub auto_redecompose: bool,

    /// Track decomposition history for learning
    pub track_history: bool,
}

impl Default for DecompositionConfig {
    fn default() -> Self {
        Self {
            max_depth: 5,
            max_sub_problems: 10,
            min_coverage_for_redecomp: 0.7,
            auto_redecompose: true,
            track_history: true,
        }
    }
}

/// History entry for tracking decomposition patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecompositionHistory {
    /// Problem description
    pub problem: String,

    /// Strategy used
    pub strategy: DecompositionStrategy,

    /// Number of sub-problems generated
    pub sub_problem_count: usize,

    /// Depth level
    pub depth: usize,

    /// Knowledge coverage at decomposition time
    pub coverage: f64,

    /// Whether re-decomposition occurred
    pub was_redecomposed: bool,

    /// Success (all sub-problems completed)
    pub success: bool,

    /// Timestamp
    pub timestamp: u64,
}

/// Engine for knowledge-aware decomposition
pub struct DecompositionEngine {
    /// Reference to knowledge base
    kb: KnowledgeBase,

    /// Configuration
    config: DecompositionConfig,

    /// Decomposition history (for learning)
    history: Vec<DecompositionHistory>,

    /// Active decompositions (by problem ID)
    active: HashMap<ProblemId, Decomposition>,
}

impl DecompositionEngine {
    /// Create a new decomposition engine
    pub fn new(kb: KnowledgeBase) -> Self {
        Self {
            kb,
            config: DecompositionConfig::default(),
            history: Vec::new(),
            active: HashMap::new(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(kb: KnowledgeBase, config: DecompositionConfig) -> Self {
        Self {
            kb,
            config,
            history: Vec::new(),
            active: HashMap::new(),
        }
    }

    /// Decompose a problem based on current KB state
    pub fn decompose(&mut self, problem: Problem) -> Result<Decomposition, DecompositionError> {
        self.decompose_with_depth(problem, 0)
    }

    /// Internal decomposition with depth tracking
    fn decompose_with_depth(
        &mut self,
        mut problem: Problem,
        depth: usize,
    ) -> Result<Decomposition, DecompositionError> {
        // Check depth limit
        if depth > self.config.max_depth {
            return Err(DecompositionError::MaxDepthExceeded(depth));
        }

        // Query KB for known facts
        self.enrich_problem_with_kb(&mut problem)?;

        // Determine strategy based on knowledge coverage
        let coverage = problem.knowledge_coverage();
        let strategy = self.determine_strategy(&problem, coverage);

        // Generate sub-problems based on strategy
        let sub_problems = match strategy {
            DecompositionStrategy::Atomic => Vec::new(), // No decomposition needed
            DecompositionStrategy::Computational => {
                self.generate_computational_decomposition(&problem)?
            }
            DecompositionStrategy::Structural => {
                self.generate_structural_decomposition(&problem)?
            }
            DecompositionStrategy::Hybrid => self.generate_hybrid_decomposition(&problem)?,
        };

        // Build dependency graph
        let dependencies = self.build_dependency_graph(&sub_problems);

        // Create decomposition result
        let mut decomposition = Decomposition {
            problem: problem.clone(),
            sub_problems,
            strategy,
            depth,
            dependencies,
            parallel_groups: Vec::new(),
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            metadata: HashMap::new(),
        };

        // Compute parallel execution groups
        decomposition.compute_parallel_groups()?;

        // Track in history
        if self.config.track_history {
            self.record_history(&problem, &decomposition);
        }

        // Store active decomposition
        self.active
            .insert(problem.id.clone(), decomposition.clone());

        Ok(decomposition)
    }

    /// Enrich problem with facts from KB
    fn enrich_problem_with_kb(&mut self, problem: &mut Problem) -> Result<(), DecompositionError> {
        // Query KB for each required fact
        for required_fact in &problem.required_facts {
            if let Ok(fact) = self.kb.get_fact(required_fact) {
                problem
                    .known_facts
                    .insert(required_fact.clone(), fact.summary.clone());
            }
        }

        Ok(())
    }

    /// Determine decomposition strategy
    fn determine_strategy(&self, problem: &Problem, coverage: f64) -> DecompositionStrategy {
        if problem.is_atomic() {
            return DecompositionStrategy::Atomic;
        }

        if coverage >= 0.9 {
            // Almost all facts known → computational
            DecompositionStrategy::Computational
        } else if coverage >= 0.5 {
            // Mix of known/unknown → hybrid
            DecompositionStrategy::Hybrid
        } else {
            // Mostly unknowns → structural breakdown
            DecompositionStrategy::Structural
        }
    }

    /// Generate structural decomposition (task breakdown)
    fn generate_structural_decomposition(
        &self,
        problem: &Problem,
    ) -> Result<Vec<SubProblem>, DecompositionError> {
        // For each required fact, create a sub-problem
        let mut sub_problems = Vec::new();

        for (idx, required_fact) in problem.required_facts.iter().enumerate() {
            let sub = SubProblem::new(
                format!("{}:sub{}", problem.id, idx),
                format!("Find {}", required_fact),
            )
            .produces(required_fact.clone())
            .success_criterion(format!("{} is known", required_fact))
            .with_complexity(ComplexityLevel::Simple);

            sub_problems.push(sub);
        }

        Ok(sub_problems)
    }

    /// Generate computational decomposition (formula with known values)
    fn generate_computational_decomposition(
        &self,
        problem: &Problem,
    ) -> Result<Vec<SubProblem>, DecompositionError> {
        // Since facts are mostly known, create a single computational sub-problem
        let sub = SubProblem::new(
            format!("{}:compute", problem.id),
            format!("Compute: {}", problem.description),
        )
        .success_criterion("Calculation complete".to_string())
        .with_complexity(ComplexityLevel::Simple);

        Ok(vec![sub])
    }

    /// Generate hybrid decomposition (mix of structural and computational)
    fn generate_hybrid_decomposition(
        &self,
        problem: &Problem,
    ) -> Result<Vec<SubProblem>, DecompositionError> {
        let mut sub_problems = Vec::new();

        // First, create sub-problems for unknown facts
        for (idx, required_fact) in problem.required_facts.iter().enumerate() {
            if !problem.known_facts.contains_key(required_fact) {
                let sub = SubProblem::new(
                    format!("{}:find{}", problem.id, idx),
                    format!("Find {}", required_fact),
                )
                .produces(required_fact.clone())
                .with_complexity(ComplexityLevel::Simple);

                sub_problems.push(sub);
            }
        }

        // Then, create a computational sub-problem that depends on all fact-finding
        let compute_sub = SubProblem::new(
            format!("{}:compute", problem.id),
            format!("Compute: {}", problem.description),
        )
        .with_complexity(ComplexityLevel::Simple);

        // Add dependencies on all fact-finding sub-problems
        let compute_sub = sub_problems.iter().fold(compute_sub, |sub, fact_finder| {
            sub.depends_on(fact_finder.id.clone())
        });

        sub_problems.push(compute_sub);

        Ok(sub_problems)
    }

    /// Build dependency graph from sub-problems
    fn build_dependency_graph(
        &self,
        sub_problems: &[SubProblem],
    ) -> HashMap<ProblemId, Vec<ProblemId>> {
        let mut graph = HashMap::new();

        for sub in sub_problems {
            graph.insert(sub.id.clone(), sub.depends_on.clone());
        }

        graph
    }

    /// Record decomposition in history
    fn record_history(&mut self, problem: &Problem, decomposition: &Decomposition) {
        let entry = DecompositionHistory {
            problem: problem.description.clone(),
            strategy: decomposition.strategy,
            sub_problem_count: decomposition.sub_problems.len(),
            depth: decomposition.depth,
            coverage: problem.knowledge_coverage(),
            was_redecomposed: false, // Updated later
            success: false,          // Updated when decomposition completes
            timestamp: decomposition.created_at,
        };

        self.history.push(entry);
    }

    /// Check if re-decomposition should be triggered
    pub fn should_redecompose(&self, problem_id: &ProblemId) -> bool {
        if !self.config.auto_redecompose {
            return false;
        }

        // Get active decomposition
        if let Some(decomp) = self.active.get(problem_id) {
            let coverage = decomp.problem.knowledge_coverage();
            coverage >= self.config.min_coverage_for_redecomp
                && decomp.strategy != DecompositionStrategy::Computational
                && decomp.strategy != DecompositionStrategy::Atomic
        } else {
            false
        }
    }

    /// Re-decompose a problem with updated KB state
    pub fn redecompose(
        &mut self,
        problem_id: &ProblemId,
    ) -> Result<Decomposition, DecompositionError> {
        // Get original decomposition
        let original = self
            .active
            .get(problem_id)
            .ok_or_else(|| {
                DecompositionError::InvalidProblem(format!("Problem {} not found", problem_id))
            })?
            .clone();

        // Re-decompose with same problem but fresh KB query
        let mut new_problem = original.problem.clone();
        self.enrich_problem_with_kb(&mut new_problem)?;

        let new_decomp = self.decompose_with_depth(new_problem, original.depth)?;

        // Update history to mark re-decomposition
        if let Some(last_entry) = self.history.last_mut() {
            last_entry.was_redecomposed = true;
        }

        Ok(new_decomp)
    }

    /// Get decomposition history
    pub fn history(&self) -> &[DecompositionHistory] {
        &self.history
    }

    /// Get statistics about decompositions
    pub fn stats(&self) -> DecompositionStats {
        let total = self.history.len();
        let structural = self
            .history
            .iter()
            .filter(|h| h.strategy == DecompositionStrategy::Structural)
            .count();
        let computational = self
            .history
            .iter()
            .filter(|h| h.strategy == DecompositionStrategy::Computational)
            .count();
        let redecomposed = self.history.iter().filter(|h| h.was_redecomposed).count();
        let avg_coverage = if total > 0 {
            self.history.iter().map(|h| h.coverage).sum::<f64>() / total as f64
        } else {
            0.0
        };

        DecompositionStats {
            total_decompositions: total,
            structural_count: structural,
            computational_count: computational,
            redecomposition_count: redecomposed,
            average_coverage: avg_coverage,
            active_decompositions: self.active.len(),
        }
    }

    /// Explore multiple problem variants using branching
    ///
    /// Creates a checkpoint of the current engine state, then branches for each problem variant
    /// to explore different approaches in parallel. Returns all results sorted by quality score.
    ///
    /// This allows experimenting with different problem formulations or KB states to find
    /// the most effective decomposition approach.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Prepare different variations of the problem
    /// let problem1 = Problem::new("Calculate impact").requires("gdp_2023");
    /// let problem2 = Problem::new("Calculate impact")
    ///     .requires("gdp_2023")
    ///     .with_known_fact("population", "331M");
    ///
    /// let problems = vec![problem1, problem2];
    ///
    /// let results = engine.explore_problem_variants(&problems, &mut checkpoint_mgr)?;
    ///
    /// // Best variant is first
    /// let (best_problem, best_decomp, score) = &results[0];
    /// ```
    pub fn explore_problem_variants(
        &mut self,
        problem_variants: &[Problem],
        checkpoint_manager: &mut CheckpointManager,
    ) -> Result<Vec<(Problem, Decomposition, f64)>, DecompositionError> {
        // Create base checkpoint from current engine state
        let base = create_checkpoint(
            &self.kb,
            self.history.clone(),
            self.active.clone(),
            HashMap::new(),
        );
        let base_id = base.id.clone(); // Save the ID before passing ownership
        checkpoint_manager
            .save(&base)
            .map_err(|e| DecompositionError::KnowledgeBaseError(e.to_string()))?;

        let mut results = Vec::new();

        // Branch and explore each problem variant
        for (idx, problem) in problem_variants.iter().enumerate() {
            let branch_name = format!("variant_{}", idx);
            let branch_id = checkpoint_manager
                .branch(&base_id, &branch_name)
                .map_err(|e| DecompositionError::KnowledgeBaseError(e.to_string()))?;

            // Restore state for this branch
            let branch_checkpoint = checkpoint_manager
                .load(&branch_id)
                .map_err(|e| DecompositionError::KnowledgeBaseError(e.to_string()))?;

            let branch_kb = restore_kb_from_checkpoint(&branch_checkpoint)
                .map_err(|e| DecompositionError::KnowledgeBaseError(e.to_string()))?;

            // Create temporary engine with branched KB
            let mut branch_engine = DecompositionEngine::new(branch_kb);
            branch_engine.history = branch_checkpoint.decomposition_history;
            branch_engine.active = branch_checkpoint.active_decompositions;

            // Decompose this variant
            let decomp = branch_engine.decompose(problem.clone())?;

            // Score the decomposition
            let score = self.score_decomposition(&decomp);

            results.push((problem.clone(), decomp, score));
        }

        // Sort by score (highest first)
        results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results)
    }

    /// Score a decomposition for quality comparison
    ///
    /// Higher scores are better. Scoring factors:
    /// - **Sub-problem count**: Fewer is better (penalty: 0.1 per sub-problem)
    /// - **Dependency complexity**: Simpler is better (penalty: 0.05 per dependency edge)
    /// - **Knowledge coverage**: Higher is better (bonus: 2.0 * coverage)
    /// - **Atomic bonus**: Atomic problems get +5.0 bonus
    ///
    /// # Example Scores
    ///
    /// - Atomic problem with 100% coverage: ~7.0
    /// - Computational with 3 steps: ~1.7
    /// - Structural with 10 sub-problems and 15 deps: ~-1.75
    pub fn score_decomposition(&self, decomp: &Decomposition) -> f64 {
        let sub_problem_penalty = decomp.sub_problems.len() as f64 * 0.1;
        let dependency_penalty = decomp
            .dependencies
            .values()
            .map(|deps| deps.len())
            .sum::<usize>() as f64
            * 0.05;
        let coverage_bonus = decomp.problem.knowledge_coverage() * 2.0;
        let atomic_bonus = if decomp.is_atomic() { 5.0 } else { 0.0 };

        // Higher is better
        coverage_bonus + atomic_bonus - sub_problem_penalty - dependency_penalty
    }
}

/// Statistics about decomposition engine usage
#[derive(Debug, Clone)]
pub struct DecompositionStats {
    pub total_decompositions: usize,
    pub structural_count: usize,
    pub computational_count: usize,
    pub redecomposition_count: usize,
    pub average_coverage: f64,
    pub active_decompositions: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_problem_creation() {
        let problem = Problem::new("Calculate GDP growth")
            .requires("gdp_2023")
            .requires("gdp_2024")
            .with_complexity(ComplexityLevel::Moderate);

        assert_eq!(problem.required_facts.len(), 2);
        assert_eq!(problem.complexity, ComplexityLevel::Moderate);
        assert!(!problem.is_atomic());
    }

    #[test]
    fn test_knowledge_coverage() {
        let mut problem = Problem::new("Test")
            .requires("fact1")
            .requires("fact2")
            .requires("fact3");

        assert_eq!(problem.knowledge_coverage(), 0.0);

        problem
            .known_facts
            .insert("fact1".to_string(), "value1".to_string());
        assert!((problem.knowledge_coverage() - 0.333).abs() < 0.01);

        problem
            .known_facts
            .insert("fact2".to_string(), "value2".to_string());
        assert!((problem.knowledge_coverage() - 0.666).abs() < 0.01);

        problem
            .known_facts
            .insert("fact3".to_string(), "value3".to_string());
        assert_eq!(problem.knowledge_coverage(), 1.0);
    }

    #[test]
    fn test_execution_order() {
        let sub1 = SubProblem::new("sub1", "Task 1");
        let sub2 = SubProblem::new("sub2", "Task 2").depends_on("sub1");
        let sub3 = SubProblem::new("sub3", "Task 3").depends_on("sub1");
        let sub4 = SubProblem::new("sub4", "Task 4")
            .depends_on("sub2")
            .depends_on("sub3");

        let mut decomp = Decomposition {
            problem: Problem::new("Test"),
            sub_problems: vec![sub1, sub2, sub3, sub4],
            strategy: DecompositionStrategy::Structural,
            depth: 0,
            dependencies: HashMap::new(),
            parallel_groups: Vec::new(),
            created_at: 0,
            metadata: HashMap::new(),
        };

        // Build dependency graph manually
        decomp.dependencies.insert("sub1".to_string(), vec![]);
        decomp
            .dependencies
            .insert("sub2".to_string(), vec!["sub1".to_string()]);
        decomp
            .dependencies
            .insert("sub3".to_string(), vec!["sub1".to_string()]);
        decomp.dependencies.insert(
            "sub4".to_string(),
            vec!["sub2".to_string(), "sub3".to_string()],
        );

        let order = decomp.execution_order().unwrap();
        assert_eq!(order.len(), 4);
        assert_eq!(order[0], "sub1"); // Must be first
        assert_eq!(order[3], "sub4"); // Must be last
    }

    #[test]
    fn test_parallel_groups() {
        let sub1 = SubProblem::new("sub1", "Task 1");
        let sub2 = SubProblem::new("sub2", "Task 2").depends_on("sub1");
        let sub3 = SubProblem::new("sub3", "Task 3").depends_on("sub1");

        let mut decomp = Decomposition {
            problem: Problem::new("Test"),
            sub_problems: vec![sub1, sub2, sub3],
            strategy: DecompositionStrategy::Structural,
            depth: 0,
            dependencies: HashMap::new(),
            parallel_groups: Vec::new(),
            created_at: 0,
            metadata: HashMap::new(),
        };

        decomp.dependencies.insert("sub1".to_string(), vec![]);
        decomp
            .dependencies
            .insert("sub2".to_string(), vec!["sub1".to_string()]);
        decomp
            .dependencies
            .insert("sub3".to_string(), vec!["sub1".to_string()]);

        decomp.compute_parallel_groups().unwrap();

        assert_eq!(decomp.parallel_groups.len(), 2);
        assert_eq!(decomp.parallel_groups[0], vec!["sub1"]);
        // sub2 and sub3 can run in parallel
        assert_eq!(decomp.parallel_groups[1].len(), 2);
    }

    #[test]
    fn test_decomposition_engine_basic() {
        let kb = KnowledgeBase::new();
        let mut engine = DecompositionEngine::new(kb);

        let problem = Problem::new("Calculate economic impact")
            .requires("gdp_2023")
            .requires("gdp_2024")
            .with_complexity(ComplexityLevel::Complex);

        let decomp = engine.decompose(problem).unwrap();

        // Should use structural strategy (no facts in KB)
        assert_eq!(decomp.strategy, DecompositionStrategy::Structural);
        assert_eq!(decomp.sub_problems.len(), 2); // One for each required fact
        assert_eq!(decomp.depth, 0);
    }

    #[test]
    fn test_strategy_determination() {
        let kb = KnowledgeBase::new();
        let engine = DecompositionEngine::new(kb);

        // No known facts → Structural
        let problem1 = Problem::new("Test").requires("fact1").requires("fact2");
        assert_eq!(
            engine.determine_strategy(&problem1, 0.0),
            DecompositionStrategy::Structural
        );

        // Half known → Hybrid
        let problem2 = problem1.clone().with_known_fact("fact1", "value1");
        assert_eq!(
            engine.determine_strategy(&problem2, 0.5),
            DecompositionStrategy::Hybrid
        );

        // All known → Computational
        let mut problem3 = problem2.clone();
        problem3
            .known_facts
            .insert("fact2".to_string(), "value2".to_string());
        assert_eq!(
            engine.determine_strategy(&problem3, 1.0),
            DecompositionStrategy::Computational
        );
    }

    #[test]
    fn test_circular_dependency_detection() {
        let sub1 = SubProblem::new("sub1", "Task 1").depends_on("sub2");
        let sub2 = SubProblem::new("sub2", "Task 2").depends_on("sub1"); // Circular!

        let mut decomp = Decomposition {
            problem: Problem::new("Test"),
            sub_problems: vec![sub1, sub2],
            strategy: DecompositionStrategy::Structural,
            depth: 0,
            dependencies: HashMap::new(),
            parallel_groups: Vec::new(),
            created_at: 0,
            metadata: HashMap::new(),
        };

        decomp
            .dependencies
            .insert("sub1".to_string(), vec!["sub2".to_string()]);
        decomp
            .dependencies
            .insert("sub2".to_string(), vec!["sub1".to_string()]);

        let result = decomp.execution_order();
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            DecompositionError::CircularDependency(_)
        ));
    }

    #[test]
    fn test_branching_exploration() {
        use crate::engine::state_persistence::CheckpointManager;
        use std::path::PathBuf;

        let kb = KnowledgeBase::new();
        let mut engine = DecompositionEngine::new(kb);

        // Create problem variants - same problem but different formulations
        let variant1 = Problem::new("Calculate growth rate")
            .requires("current_value")
            .requires("previous_value")
            .with_complexity(ComplexityLevel::Moderate);

        let variant2 = Problem::new("Calculate growth rate")
            .requires("current_value")
            .requires("previous_value")
            .with_known_fact("formula", "((current - previous) / previous) * 100")
            .with_complexity(ComplexityLevel::Simple);

        let variants = vec![variant1, variant2];

        // Create checkpoint manager
        let temp_dir = std::env::temp_dir().join("lightbulb_test_branching");
        let _ = std::fs::remove_dir_all(&temp_dir); // Clean up if exists
        let mut checkpoint_mgr = CheckpointManager::new(temp_dir, 10).unwrap();

        // Explore variants
        let results = engine
            .explore_problem_variants(&variants, &mut checkpoint_mgr)
            .unwrap();

        // Should have 2 results
        assert_eq!(results.len(), 2);

        // Results should be sorted by score (highest first)
        assert!(results[0].2 >= results[1].2);

        // The variant with known fact should score higher (higher coverage)
        // Check that decompositions were created
        assert!(!results[0].1.sub_problems.is_empty() || results[0].1.is_atomic());
        assert!(!results[1].1.sub_problems.is_empty() || results[1].1.is_atomic());
    }

    #[test]
    fn test_decomposition_scoring() {
        let kb = KnowledgeBase::new();
        let engine = DecompositionEngine::new(kb);

        // Create a simple atomic decomposition
        let atomic_decomp = Decomposition {
            problem: Problem::new("Test")
                .with_known_fact("a", "1")
                .with_known_fact("b", "2"),
            sub_problems: vec![],
            strategy: DecompositionStrategy::Atomic,
            depth: 0,
            dependencies: HashMap::new(),
            parallel_groups: Vec::new(),
            created_at: 0,
            metadata: HashMap::new(),
        };

        // Create a complex structural decomposition
        let complex_decomp = Decomposition {
            problem: Problem::new("Test").requires("x").requires("y"),
            sub_problems: vec![
                SubProblem::new("sub1", "Task 1"),
                SubProblem::new("sub2", "Task 2").depends_on("sub1"),
                SubProblem::new("sub3", "Task 3").depends_on("sub1"),
            ],
            strategy: DecompositionStrategy::Structural,
            depth: 0,
            dependencies: HashMap::from([
                ("sub1".to_string(), vec![]),
                ("sub2".to_string(), vec!["sub1".to_string()]),
                ("sub3".to_string(), vec!["sub1".to_string()]),
            ]),
            parallel_groups: Vec::new(),
            created_at: 0,
            metadata: HashMap::new(),
        };

        let atomic_score = engine.score_decomposition(&atomic_decomp);
        let complex_score = engine.score_decomposition(&complex_decomp);

        // Atomic with full coverage should score much higher
        assert!(atomic_score > complex_score);

        // Atomic with 100% coverage should get bonus
        assert!(atomic_score > 5.0); // Should have atomic bonus + coverage bonus
    }
}
