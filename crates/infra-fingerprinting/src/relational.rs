//! Relational fingerprinting for graph isomorphism signatures and relationship patterns
//!
//! This module provides fingerprinting capabilities for relationship patterns and
//! graph structures within the DynAniML system. It uses graph isomorphism techniques
//! to create position-invariant fingerprints of relationship patterns.

use crate::{
    Fingerprint, FingerprintError, FingerprintLevel, FingerprintMetadata, FingerprintResult,
};
use blake3::Hasher;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Trait for types that can be fingerprinted at the relational level
pub trait RelationalFingerprintable {
    /// Extract relational components for fingerprinting
    fn get_relational_components(&self) -> RelationalComponents;
}

/// Components needed for relational-level fingerprinting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationalComponents {
    /// Nodes in the relationship graph
    pub nodes: Vec<RelationalNode>,
    /// Edges in the relationship graph
    pub edges: Vec<RelationalEdge>,
    /// Graph-level metadata
    pub graph_metadata: GraphMetadata,
}

/// A node in the relational graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationalNode {
    /// Unique identifier for this node
    pub id: String,
    /// Type of this node
    pub node_type: String,
    /// Dimensional attributes of this node
    pub attributes: HashMap<String, String>,
    /// Weight or importance of this node
    pub weight: f32,
}

/// An edge in the relational graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationalEdge {
    /// Source node ID
    pub source: String,
    /// Target node ID
    pub target: String,
    /// Type of relationship
    pub relationship_type: String,
    /// Direction of the relationship
    pub direction: EdgeDirection,
    /// Strength or weight of the relationship
    pub weight: f32,
    /// Additional edge attributes
    pub attributes: HashMap<String, String>,
}

/// Direction of a relational edge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EdgeDirection {
    /// Directed edge from source to target
    Directed,
    /// Undirected edge (bidirectional)
    Undirected,
    /// Directed edge with reverse implication
    Bidirectional,
}

/// Metadata about the overall graph structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMetadata {
    /// Domain or context of this graph
    pub domain: String,
    /// Structural properties
    pub properties: HashMap<String, String>,
    /// Confidence in the graph structure
    pub confidence: f32,
}

/// Engine for computing relational-level fingerprints
pub struct RelationalFingerprintEngine {
    /// Whether to consider edge direction in fingerprinting
    include_direction: bool,
    /// Whether to include node attributes in fingerprinting
    include_node_attributes: bool,
    /// Whether to include edge attributes in fingerprinting
    include_edge_attributes: bool,
    /// Whether to normalize weights
    normalize_weights: bool,
}

/// Graph canonical form for isomorphism detection
#[derive(Debug, Clone)]
struct CanonicalGraph {
    /// Canonically ordered nodes
    nodes: Vec<CanonicalNode>,
    /// Canonically ordered edges
    edges: Vec<CanonicalEdge>,
    /// Graph invariants
    invariants: GraphInvariants,
}

/// Canonical representation of a node
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct CanonicalNode {
    /// Node type signature
    type_signature: String,
    /// Attribute signature
    attribute_signature: String,
    /// Degree signature
    degree_signature: String,
}

/// Canonical representation of an edge
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct CanonicalEdge {
    /// Source node canonical ID
    source_canonical: usize,
    /// Target node canonical ID
    target_canonical: usize,
    /// Edge type signature
    type_signature: String,
    /// Edge attribute signature
    attribute_signature: String,
}

/// Graph invariants for isomorphism detection
#[derive(Debug, Clone)]
struct GraphInvariants {
    /// Number of nodes
    node_count: usize,
    /// Number of edges
    edge_count: usize,
    /// Degree sequence (sorted)
    degree_sequence: Vec<usize>,
    /// Node type counts
    node_type_counts: HashMap<String, usize>,
    /// Edge type counts
    edge_type_counts: HashMap<String, usize>,
}

impl Default for RelationalFingerprintEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl RelationalFingerprintEngine {
    /// Create a new relational fingerprint engine with default settings
    pub fn new() -> Self {
        Self {
            include_direction: true,
            include_node_attributes: true,
            include_edge_attributes: false, // Often too noisy
            normalize_weights: true,
        }
    }

    /// Create an engine with custom settings
    pub fn with_config(
        include_direction: bool,
        include_node_attributes: bool,
        include_edge_attributes: bool,
        normalize_weights: bool,
    ) -> Self {
        Self {
            include_direction,
            include_node_attributes,
            include_edge_attributes,
            normalize_weights,
        }
    }

    /// Compute a relational-level fingerprint for the given input
    pub fn compute_fingerprint<T>(&self, input: &T) -> FingerprintResult<Fingerprint>
    where
        T: RelationalFingerprintable,
    {
        let components = input.get_relational_components();
        let canonical = self.canonicalize_graph(&components)?;
        let hash = self.hash_canonical_graph(&canonical)?;

        let metadata = FingerprintMetadata {
            algorithm: "relational-canonical-blake3".to_string(),
            parameters: self.get_parameters(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            confidence: self.calculate_confidence(&components),
        };

        Ok(Fingerprint::with_metadata(
            hash,
            FingerprintLevel::Relational,
            metadata,
        ))
    }

    /// Convert relational components to canonical form
    fn canonicalize_graph(
        &self,
        components: &RelationalComponents,
    ) -> FingerprintResult<CanonicalGraph> {
        // Compute graph invariants first
        let invariants = self.compute_invariants(components);

        // Create canonical nodes
        let mut canonical_nodes = Vec::new();
        for node in &components.nodes {
            let canonical_node = CanonicalNode {
                type_signature: node.node_type.clone(),
                attribute_signature: if self.include_node_attributes {
                    self.hash_attributes(&node.attributes)
                } else {
                    String::new()
                },
                degree_signature: self.compute_degree_signature(node, components),
            };
            canonical_nodes.push(canonical_node);
        }

        // Sort nodes by their canonical representation
        let mut node_indices: Vec<usize> = (0..canonical_nodes.len()).collect();
        node_indices.sort_by(|&a, &b| canonical_nodes[a].cmp(&canonical_nodes[b]));

        // Create mapping from original to canonical node IDs
        let mut node_mapping = HashMap::new();
        for (canonical_id, &original_id) in node_indices.iter().enumerate() {
            node_mapping.insert(components.nodes[original_id].id.clone(), canonical_id);
        }

        // Create canonical edges using the node mapping
        let mut canonical_edges = Vec::new();
        for edge in &components.edges {
            let source_canonical = *node_mapping
                .get(&edge.source)
                .ok_or_else(|| FingerprintError::InvalidInput("Invalid edge source".to_string()))?;
            let target_canonical = *node_mapping
                .get(&edge.target)
                .ok_or_else(|| FingerprintError::InvalidInput("Invalid edge target".to_string()))?;

            let canonical_edge = CanonicalEdge {
                source_canonical,
                target_canonical,
                type_signature: edge.relationship_type.clone(),
                attribute_signature: if self.include_edge_attributes {
                    self.hash_attributes(&edge.attributes)
                } else {
                    String::new()
                },
            };
            canonical_edges.push(canonical_edge);
        }

        // Sort edges by their canonical representation
        canonical_edges.sort();

        // Reorder nodes according to the canonical ordering
        let ordered_nodes: Vec<CanonicalNode> = node_indices
            .into_iter()
            .map(|i| canonical_nodes[i].clone())
            .collect();

        Ok(CanonicalGraph {
            nodes: ordered_nodes,
            edges: canonical_edges,
            invariants,
        })
    }

    /// Hash the canonical graph representation
    fn hash_canonical_graph(&self, canonical: &CanonicalGraph) -> FingerprintResult<[u8; 32]> {
        let mut hasher = Hasher::new();

        // Hash graph invariants first
        hasher.update(&canonical.invariants.node_count.to_le_bytes());
        hasher.update(&canonical.invariants.edge_count.to_le_bytes());

        // Hash degree sequence
        for degree in &canonical.invariants.degree_sequence {
            hasher.update(&degree.to_le_bytes());
        }

        // Hash node type counts
        let mut node_types: Vec<_> = canonical.invariants.node_type_counts.iter().collect();
        node_types.sort_by_key(|(k, _)| *k);
        for (node_type, count) in node_types {
            hasher.update(node_type.as_bytes());
            hasher.update(&count.to_le_bytes());
        }

        // Hash edge type counts
        let mut edge_types: Vec<_> = canonical.invariants.edge_type_counts.iter().collect();
        edge_types.sort_by_key(|(k, _)| *k);
        for (edge_type, count) in edge_types {
            hasher.update(edge_type.as_bytes());
            hasher.update(&count.to_le_bytes());
        }

        // Hash canonical nodes
        for node in &canonical.nodes {
            hasher.update(node.type_signature.as_bytes());
            hasher.update(node.attribute_signature.as_bytes());
            hasher.update(node.degree_signature.as_bytes());
        }

        // Hash canonical edges
        for edge in &canonical.edges {
            hasher.update(&edge.source_canonical.to_le_bytes());
            hasher.update(&edge.target_canonical.to_le_bytes());
            hasher.update(edge.type_signature.as_bytes());
            hasher.update(edge.attribute_signature.as_bytes());
        }

        Ok(hasher.finalize().into())
    }

    /// Compute graph invariants
    fn compute_invariants(&self, components: &RelationalComponents) -> GraphInvariants {
        let node_count = components.nodes.len();
        let edge_count = components.edges.len();

        // Compute degree sequence
        let mut degrees = HashMap::new();
        for edge in &components.edges {
            *degrees.entry(edge.source.clone()).or_insert(0) += 1;
            *degrees.entry(edge.target.clone()).or_insert(0) += 1;
        }

        let mut degree_sequence: Vec<usize> = degrees.values().cloned().collect();
        degree_sequence.sort_unstable();

        // Count node types
        let mut node_type_counts = HashMap::new();
        for node in &components.nodes {
            *node_type_counts.entry(node.node_type.clone()).or_insert(0) += 1;
        }

        // Count edge types
        let mut edge_type_counts = HashMap::new();
        for edge in &components.edges {
            *edge_type_counts
                .entry(edge.relationship_type.clone())
                .or_insert(0) += 1;
        }

        GraphInvariants {
            node_count,
            edge_count,
            degree_sequence,
            node_type_counts,
            edge_type_counts,
        }
    }

    /// Hash a set of attributes into a signature
    fn hash_attributes(&self, attributes: &HashMap<String, String>) -> String {
        let mut attr_pairs: Vec<_> = attributes.iter().collect();
        attr_pairs.sort_by_key(|(k, _)| *k);

        let mut hasher = blake3::Hasher::new();
        for (key, value) in attr_pairs {
            hasher.update(key.as_bytes());
            hasher.update(value.as_bytes());
        }

        hex::encode(hasher.finalize().as_bytes())
    }

    /// Compute degree signature for a node
    fn compute_degree_signature(
        &self,
        node: &RelationalNode,
        components: &RelationalComponents,
    ) -> String {
        let mut in_degree = 0;
        let mut out_degree = 0;
        let mut undirected_degree = 0;

        for edge in &components.edges {
            match edge.direction {
                EdgeDirection::Directed => {
                    if edge.source == node.id {
                        out_degree += 1;
                    }
                    if edge.target == node.id {
                        in_degree += 1;
                    }
                }
                EdgeDirection::Undirected => {
                    if edge.source == node.id || edge.target == node.id {
                        undirected_degree += 1;
                    }
                }
                EdgeDirection::Bidirectional => {
                    if edge.source == node.id {
                        out_degree += 1;
                        in_degree += 1;
                    }
                    if edge.target == node.id {
                        out_degree += 1;
                        in_degree += 1;
                    }
                }
            }
        }

        format!("{}:{}:{}", in_degree, out_degree, undirected_degree)
    }

    /// Calculate confidence score for the fingerprint
    fn calculate_confidence(&self, components: &RelationalComponents) -> f32 {
        let mut confidence = 1.0;

        // Reduce confidence for very small graphs
        if components.nodes.len() < 3 {
            confidence *= 0.8;
        }

        // Reduce confidence if no edges
        if components.edges.is_empty() {
            confidence *= 0.5;
        }

        // Increase confidence for well-connected graphs
        let connectivity = components.edges.len() as f32 / components.nodes.len().max(1) as f32;
        confidence *= (1.0 + connectivity * 0.1).min(1.2);

        confidence.min(1.0)
    }

    /// Get current engine parameters
    fn get_parameters(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert(
            "include_direction".to_string(),
            self.include_direction.to_string(),
        );
        params.insert(
            "include_node_attributes".to_string(),
            self.include_node_attributes.to_string(),
        );
        params.insert(
            "include_edge_attributes".to_string(),
            self.include_edge_attributes.to_string(),
        );
        params.insert(
            "normalize_weights".to_string(),
            self.normalize_weights.to_string(),
        );
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestGraph {
        components: RelationalComponents,
    }

    impl RelationalFingerprintable for TestGraph {
        fn get_relational_components(&self) -> RelationalComponents {
            self.components.clone()
        }
    }

    fn create_simple_graph() -> TestGraph {
        let nodes = vec![
            RelationalNode {
                id: "A".to_string(),
                node_type: "concept".to_string(),
                attributes: HashMap::new(),
                weight: 1.0,
            },
            RelationalNode {
                id: "B".to_string(),
                node_type: "concept".to_string(),
                attributes: HashMap::new(),
                weight: 1.0,
            },
        ];

        let edges = vec![RelationalEdge {
            source: "A".to_string(),
            target: "B".to_string(),
            relationship_type: "related_to".to_string(),
            direction: EdgeDirection::Directed,
            weight: 1.0,
            attributes: HashMap::new(),
        }];

        TestGraph {
            components: RelationalComponents {
                nodes,
                edges,
                graph_metadata: GraphMetadata {
                    domain: "test".to_string(),
                    properties: HashMap::new(),
                    confidence: 1.0,
                },
            },
        }
    }

    #[test]
    fn test_simple_relational_fingerprinting() {
        let engine = RelationalFingerprintEngine::new();
        let graph = create_simple_graph();

        let fingerprint = engine.compute_fingerprint(&graph).unwrap();
        assert_eq!(fingerprint.level, FingerprintLevel::Relational);
        assert!(fingerprint.metadata.is_some());
    }

    #[test]
    fn test_graph_isomorphism() {
        let engine = RelationalFingerprintEngine::new();

        // Create two isomorphic graphs with different node IDs
        let graph1 = create_simple_graph();

        let mut graph2_components = graph1.components.clone();
        graph2_components.nodes[0].id = "X".to_string();
        graph2_components.nodes[1].id = "Y".to_string();
        graph2_components.edges[0].source = "X".to_string();
        graph2_components.edges[0].target = "Y".to_string();

        let graph2 = TestGraph {
            components: graph2_components,
        };

        let fp1 = engine.compute_fingerprint(&graph1).unwrap();
        let fp2 = engine.compute_fingerprint(&graph2).unwrap();

        // Should have the same fingerprint despite different node IDs
        assert_eq!(fp1.hash, fp2.hash);
    }
}
