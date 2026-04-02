//! Structural fingerprinting for knowledge chunk topology and organization
//!
//! This module provides fingerprinting capabilities for the overall structure
//! and topology of knowledge chunks, focusing on how information is organized
//! rather than its specific content.

use crate::{Fingerprint, FingerprintLevel, FingerprintMetadata, FingerprintResult};
use std::collections::HashMap;

/// Trait for types that can be fingerprinted at the structural level
pub trait StructuralFingerprintable {
    /// Extract structural components for fingerprinting
    fn get_structural_components(&self) -> StructuralComponents;
}

/// Components needed for structural-level fingerprinting
#[derive(Debug, Clone)]
pub struct StructuralComponents {
    /// Hierarchical structure information
    pub hierarchy: Vec<HierarchyNode>,
    /// Organizational patterns
    pub patterns: Vec<OrganizationalPattern>,
    /// Size and complexity metrics
    pub metrics: StructuralMetrics,
    /// Topology information
    pub topology: Option<TopologyInfo>,
}

/// A node in the hierarchical structure
#[derive(Debug, Clone)]
pub struct HierarchyNode {
    /// Identifier for this node
    pub id: String,
    /// Parent node identifier, if any
    pub parent: Option<String>,
    /// Child node identifiers
    pub children: Vec<String>,
    /// Type of this node
    pub node_type: String,
    /// Depth in the hierarchy
    pub depth: usize,
}

/// An organizational pattern found in the structure
#[derive(Debug, Clone)]
pub struct OrganizationalPattern {
    /// Type of pattern (e.g., "sequential", "hierarchical", "network")
    pub pattern_type: String,
    /// Nodes involved in this pattern
    pub nodes: Vec<String>,
    /// Strength of the pattern (0.0 to 1.0)
    pub strength: f32,
}

/// Metrics about the structural complexity
#[derive(Debug, Clone)]
pub struct StructuralMetrics {
    /// Total number of nodes
    pub node_count: usize,
    /// Maximum depth of hierarchy
    pub max_depth: usize,
    /// Average branching factor
    pub avg_branching_factor: f32,
    /// Structural complexity score
    pub complexity_score: f32,
}

/// Information about the topology
#[derive(Debug, Clone)]
pub struct TopologyInfo {
    /// Whether the structure is tree-like
    pub is_tree: bool,
    /// Whether the structure has cycles
    pub has_cycles: bool,
    /// Connectivity information
    pub connectivity: f32,
}

/// Engine for computing structural-level fingerprints
pub struct StructuralFingerprintEngine {
    /// Whether to include hierarchy information in fingerprints
    _include_hierarchy: bool,
    /// Whether to include patterns in fingerprints
    include_patterns: bool,
    /// Weight given to different structural aspects
    aspect_weights: HashMap<String, f32>,
}

impl Default for StructuralFingerprintEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl StructuralFingerprintEngine {
    /// Create a new structural fingerprint engine with default settings
    pub fn new() -> Self {
        let mut weights = HashMap::new();
        weights.insert("hierarchy".to_string(), 0.3);
        weights.insert("patterns".to_string(), 0.3);
        weights.insert("metrics".to_string(), 0.2);
        weights.insert("topology".to_string(), 0.2);

        Self {
            _include_hierarchy: true,
            include_patterns: true,
            aspect_weights: weights,
        }
    }
    /// Compute a structural-level fingerprint for the given input
    pub fn compute_fingerprint<T>(&self, input: &T) -> FingerprintResult<Fingerprint>
    where
        T: StructuralFingerprintable,
    {
        let components = input.get_structural_components();

        // Advanced structural fingerprinting algorithm with topology analysis
        let mut hash_input = Vec::new();

        // 1. Hierarchical structure encoding with topology-aware weighting
        let hierarchy_signature = self.compute_hierarchy_signature(&components.hierarchy);
        hash_input.extend_from_slice(&hierarchy_signature);

        // 2. Pattern-based structural fingerprinting
        if self.include_patterns {
            let pattern_signature = self.compute_pattern_signature(&components.patterns);
            hash_input.extend_from_slice(&pattern_signature);
        }

        // 3. Structural metrics normalization and encoding
        let metrics_signature = self.compute_metrics_signature(&components.metrics);
        hash_input.extend_from_slice(&metrics_signature);

        // 4. Topology invariant computation
        if let Some(ref topology) = components.topology {
            let topology_signature = self.compute_topology_signature(topology);
            hash_input.extend_from_slice(&topology_signature);
        }

        // 5. Structural complexity factor
        let complexity_factor = self.calculate_structural_complexity(&components);
        hash_input.extend_from_slice(&complexity_factor.to_le_bytes());

        // Include patterns if configured
        if self.include_patterns {
            if let Some(pattern_weight) = self.aspect_weights.get("patterns") {
                for pattern in &components.patterns {
                    hash_input.extend_from_slice(pattern.pattern_type.as_bytes());
                    hash_input.extend_from_slice(&pattern.strength.to_le_bytes());
                    hash_input.extend_from_slice(&pattern_weight.to_le_bytes());
                }
            }
        }

        // Include metrics with weight
        if let Some(metrics_weight) = self.aspect_weights.get("metrics") {
            hash_input.extend_from_slice(&components.metrics.node_count.to_le_bytes());
            hash_input.extend_from_slice(&components.metrics.max_depth.to_le_bytes());
            hash_input.extend_from_slice(&components.metrics.avg_branching_factor.to_le_bytes());
            hash_input.extend_from_slice(&components.metrics.complexity_score.to_le_bytes());
            hash_input.extend_from_slice(&metrics_weight.to_le_bytes());
        }

        // Include topology if available
        if let Some(ref topology) = components.topology {
            if let Some(topology_weight) = self.aspect_weights.get("topology") {
                hash_input.push(if topology.is_tree { 1 } else { 0 });
                hash_input.push(if topology.has_cycles { 1 } else { 0 });
                hash_input.extend_from_slice(&topology.connectivity.to_le_bytes());
                hash_input.extend_from_slice(&topology_weight.to_le_bytes());
            }
        }
        // 6. Advanced hash computation with structural layering
        let hash = blake3::hash(&hash_input);
        let hash_bytes: [u8; 32] = *hash.as_bytes();

        let metadata = FingerprintMetadata {
            algorithm: "structural-advanced-v1".to_string(),
            parameters: self.get_parameters(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            confidence: 0.85, // Higher confidence for structural analysis
        };

        Ok(Fingerprint::with_metadata(
            hash_bytes,
            FingerprintLevel::Structural,
            metadata,
        ))
    }

    /// Get parameters used for fingerprint computation
    fn get_parameters(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert(
            "include_patterns".to_string(),
            self.include_patterns.to_string(),
        );
        for (key, value) in &self.aspect_weights {
            params.insert(format!("weight_{}", key), value.to_string());
        }
        params
    }
    /// Compute hierarchical structure signature with topology awareness
    fn compute_hierarchy_signature(&self, hierarchy: &[HierarchyNode]) -> Vec<u8> {
        let mut signature = Vec::new();

        if let Some(hierarchy_weight) = self.aspect_weights.get("hierarchy") {
            // Sort nodes by depth for canonical ordering
            let mut sorted_nodes = hierarchy.to_vec();
            sorted_nodes.sort_by(|a, b| a.depth.cmp(&b.depth).then(a.id.cmp(&b.id)));

            for node in &sorted_nodes {
                // Encode node structural information
                signature.extend_from_slice(node.node_type.as_bytes());
                signature.extend_from_slice(&node.depth.to_le_bytes());
                signature.extend_from_slice(&node.children.len().to_le_bytes());

                // Include parent-child relationships
                if let Some(ref parent) = node.parent {
                    signature.extend_from_slice(parent.as_bytes());
                }

                signature.extend_from_slice(&hierarchy_weight.to_le_bytes());
            }
        }

        signature
    }

    /// Compute pattern signature with pattern strength weighting
    fn compute_pattern_signature(&self, patterns: &[OrganizationalPattern]) -> Vec<u8> {
        let mut signature = Vec::new();

        if let Some(pattern_weight) = self.aspect_weights.get("patterns") {
            // Sort patterns by type and strength for canonical ordering
            let mut sorted_patterns = patterns.to_vec();
            sorted_patterns.sort_by(|a, b| {
                a.pattern_type.cmp(&b.pattern_type).then(
                    b.strength
                        .partial_cmp(&a.strength)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
            });

            for pattern in &sorted_patterns {
                signature.extend_from_slice(pattern.pattern_type.as_bytes());
                signature.extend_from_slice(&pattern.strength.to_le_bytes());
                signature.extend_from_slice(&pattern.nodes.len().to_le_bytes());
                signature.extend_from_slice(&pattern_weight.to_le_bytes());
            }
        }

        signature
    }

    /// Compute metrics signature with normalized values
    fn compute_metrics_signature(&self, metrics: &StructuralMetrics) -> Vec<u8> {
        let mut signature = Vec::new();

        if let Some(metrics_weight) = self.aspect_weights.get("metrics") {
            // Normalize metrics to prevent scale bias
            let normalized_node_count = (metrics.node_count as f32).ln_1p(); // log(1+x) normalization
            let normalized_depth = metrics.max_depth as f32 / 100.0; // assume max reasonable depth of 100
            let normalized_branching = metrics.avg_branching_factor / 10.0; // assume max reasonable branching of 10
            let normalized_complexity = metrics.complexity_score.min(1.0); // clamp to [0,1]

            signature.extend_from_slice(&normalized_node_count.to_le_bytes());
            signature.extend_from_slice(&normalized_depth.to_le_bytes());
            signature.extend_from_slice(&normalized_branching.to_le_bytes());
            signature.extend_from_slice(&normalized_complexity.to_le_bytes());
            signature.extend_from_slice(&metrics_weight.to_le_bytes());
        }

        signature
    }

    /// Compute topology signature with invariant properties
    fn compute_topology_signature(&self, topology: &TopologyInfo) -> Vec<u8> {
        let mut signature = Vec::new();

        if let Some(topology_weight) = self.aspect_weights.get("topology") {
            // Encode topology as binary flags and normalized connectivity
            signature.push(if topology.is_tree { 1 } else { 0 });
            signature.push(if topology.has_cycles { 1 } else { 0 });

            // Normalize connectivity to [0,1] range
            let normalized_connectivity = topology.connectivity.clamp(0.0, 1.0);
            signature.extend_from_slice(&normalized_connectivity.to_le_bytes());
            signature.extend_from_slice(&topology_weight.to_le_bytes());
        }

        signature
    }

    /// Calculate overall structural complexity factor
    fn calculate_structural_complexity(&self, components: &StructuralComponents) -> f32 {
        let mut complexity = 0.0;
        let mut factor_count = 0;

        // Hierarchy complexity
        if !components.hierarchy.is_empty() {
            let max_depth = components
                .hierarchy
                .iter()
                .map(|h| h.depth)
                .max()
                .unwrap_or(0) as f32;
            let avg_children = components
                .hierarchy
                .iter()
                .map(|h| h.children.len())
                .sum::<usize>() as f32
                / components.hierarchy.len() as f32;

            let hierarchy_complexity = (max_depth / 10.0 + avg_children / 5.0) / 2.0;
            complexity += hierarchy_complexity.min(1.0);
            factor_count += 1;
        }

        // Pattern complexity
        if !components.patterns.is_empty() {
            let pattern_complexity = components.patterns.iter().map(|p| p.strength).sum::<f32>()
                / components.patterns.len() as f32;
            complexity += pattern_complexity;
            factor_count += 1;
        }

        // Metrics-based complexity
        let metrics_complexity = components.metrics.complexity_score;
        complexity += metrics_complexity;
        factor_count += 1;

        // Topology complexity
        if let Some(ref topology) = components.topology {
            let topology_complexity =
                if topology.has_cycles { 0.8 } else { 0.4 } + topology.connectivity * 0.4;
            complexity += topology_complexity.min(1.0);
            factor_count += 1;
        }

        if factor_count > 0 {
            complexity / factor_count as f32
        } else {
            0.5 // Default complexity
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestStructuralInput;

    impl StructuralFingerprintable for TestStructuralInput {
        fn get_structural_components(&self) -> StructuralComponents {
            StructuralComponents {
                hierarchy: vec![],
                patterns: vec![],
                metrics: StructuralMetrics {
                    node_count: 0,
                    max_depth: 0,
                    avg_branching_factor: 0.0,
                    complexity_score: 0.0,
                },
                topology: None,
            }
        }
    }

    #[test]
    fn test_structural_fingerprint_engine_creation() {
        let engine = StructuralFingerprintEngine::new();
        assert!(engine.include_patterns);
    }

    #[test]
    fn test_structural_fingerprint_computation() {
        let engine = StructuralFingerprintEngine::new();
        let input = TestStructuralInput;

        let result = engine.compute_fingerprint(&input);
        assert!(result.is_ok());

        let fingerprint = result.unwrap();
        assert_eq!(fingerprint.level, FingerprintLevel::Structural);
    }
}
