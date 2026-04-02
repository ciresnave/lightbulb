//! Relationship types for the DynCtx system
//!
//! Each relationship is stored in both nodes that participate in it.
//! When a relationship is created, modified, or deleted, both nodes are updated.
//! The system supports multiple layers of context through specialized relationship types.

use crate::node_arena::NodeKey;
use crate::relationship_manager::ContextLayer;
use std::fmt;

/// Represents one side of a bidirectional relationship
#[derive(Debug, Clone, PartialEq)]
pub struct RelationshipSide {
    /// The node on the other end of the relationship
    pub other_node: NodeKey,
    /// The type of relationship
    pub rel_type: RelationshipType,
    /// Whether this is the source or target side
    pub is_source: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Direction {
    Forward,
    Backward,
}

/// Core relationship types supporting different context layers
#[derive(Debug, Clone, PartialEq)]
pub enum RelationshipType {
    // Structural Context Layer
    /// Hierarchical document structure relationships
    Structural {
        /// Type of structural relationship
        kind: StructuralType,
        /// Optional metadata about the relationship
        metadata: Option<StructuralMetadata>,
    },

    // Linguistic Context Layer
    /// Language-level relationships between tokens
    Linguistic {
        /// Type of linguistic relationship
        kind: LinguisticType,
        /// Optional linguistic features
        features: Option<LinguisticFeatures>,
    },

    // Discourse Context Layer
    /// Higher-level discourse and rhetorical relationships
    Discourse {
        /// Type of discourse relationship
        kind: DiscourseType,
        /// Optional discourse properties
        properties: Option<DiscourseProperties>,
    },

    // Temporal Context Layer
    /// Time-based relationships between tokens
    Temporal {
        /// Type of temporal relationship
        order: TemporalOrder,
        /// Optional timing information
        timing: Option<TemporalTiming>,
    },

    // Social/Pragmatic Context Layer
    /// Social and usage-based relationships
    Pragmatic {
        /// Type of pragmatic relationship
        kind: PragmaticType,
        /// Optional social context
        context: Option<SocialContext>,
    },

    // Knowledge Integration Layer
    /// External knowledge and reference relationships
    Knowledge {
        /// Type of knowledge relationship
        kind: KnowledgeType,
        /// Optional knowledge metadata
        metadata: Option<KnowledgeMetadata>,
    },

    // Cognitive Context Layer
    /// Relationships based on cognitive processing and conceptual relations
    Cognitive {
        /// Type of cognitive relationship
        kind: CognitiveType,
        /// Metadata about cognitive processing
        metadata: Option<CognitiveMetadata>,
    },
}

/// Types of structural relationships in documents
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StructuralType {
    Contains,   // Parent-child relationship
    References, // Cross-references within document
    Formats,    // Formatting relationships
    Adjacent,   // Physical proximity in document
    Groups,     // Logical grouping of elements
}

/// Metadata for structural relationships
#[derive(Debug, Clone, PartialEq)]
pub struct StructuralMetadata {
    /// Base metadata shared by all relationships
    pub base: RelationshipMetadata,
    /// Parent relationship type if inherited
    pub parent_type: Option<Box<RelationshipType>>,
    /// Structural-specific metadata fields
    pub ordering: Option<u32>,
    pub weight: Option<f64>,
}

/// Types of linguistic relationships
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LinguisticType {
    DependsOn,  // Syntactic dependencies
    Modifies,   // Modification relationships
    CoRefersTo, // Coreference relationships
    Continues,  // Continuation of linguistic units
    Translates, // Translation relationships
}

/// Features for linguistic relationships
#[derive(Debug, Clone, PartialEq)]
pub struct LinguisticFeatures {
    /// Base metadata shared by all relationships
    pub base: RelationshipMetadata,
    /// Parent relationship type if inherited
    pub parent_type: Option<Box<RelationshipType>>,
    /// Linguistic-specific features
    pub pos_tag: Option<String>,
    pub dependency_type: Option<String>,
}

/// Types of discourse relationships
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DiscourseType {
    Elaborates,  // Additional detail
    Contrasts,   // Opposing viewpoints
    Exemplifies, // Examples
    Summarizes,  // Summary relationships
    Sequences,   // Sequential organization
}

/// Properties of discourse relationships
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DiscourseProperties {
    pub strength: u32, // Strength of relationship in fixed-point format (0-10000 = 0.0000-1.0000)
    pub bidirectional: bool, // Whether relationship is bidirectional
    pub discourse_role: String, // Role in discourse structure
}

impl DiscourseProperties {
    /// Creates a new discourse properties struct with fixed-point strength
    pub fn new(strength: f32, bidirectional: bool, discourse_role: String) -> Self {
        Self {
            strength: (strength * 10000.0) as u32,
            bidirectional,
            discourse_role,
        }
    }

    /// Gets the strength value as a float
    pub fn strength_float(&self) -> f32 {
        self.strength as f32 / 10000.0
    }
}

// Temporal relationships(existing enum enhanced)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TemporalOrder {
    Before,
    After,
    During,
    Overlaps,
    Contains,
    Simultaneous,
}

/// Timing information for temporal relationships
#[derive(Debug, Clone, PartialEq)]
pub struct TemporalTiming {
    pub duration: Option<std::time::Duration>,
    pub absolute_time: Option<std::time::SystemTime>,
    pub certainty: u32, // Fixed-point format (0-10000 = 0.0000-1.0000)
}

impl TemporalTiming {
    /// Creates new timing information with fixed-point certainty
    pub fn new(
        duration: Option<std::time::Duration>,
        absolute_time: Option<std::time::SystemTime>,
        certainty: f32,
    ) -> Self {
        Self {
            duration,
            absolute_time,
            certainty: (certainty * 10000.0) as u32,
        }
    }

    /// Gets the certainty value as a float
    pub fn certainty_float(&self) -> f32 {
        self.certainty as f32 / 10000.0
    }
}

/// Types of pragmatic/social relationships
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PragmaticType {
    AuthoredBy,   // Authorship
    IntendedFor,  // Target audience
    StyleMatches, // Stylistic similarity
    Registers,    // Formality level
    Citations,    // Citation relationships
}

/// Social context for pragmatic relationships
#[derive(Debug, Clone, PartialEq)]
pub struct SocialContext {
    pub formality_level: f32,
    pub cultural_context: String,
    pub domain_specific: bool,
}

/// Types of knowledge relationships
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum KnowledgeType {
    DefinesEntity, // Entity definitions
    LinksToKB,     // Knowledge base links
    CitesSource,   // External citations
    RelatesTo,     // Related concepts
    Categorizes,   // Categorical relationships
}

/// Metadata for knowledge relationships
#[derive(Debug, Clone, PartialEq)]
pub struct KnowledgeMetadata {
    pub source_reliability: f32,
    pub knowledge_domain: String,
    pub verification_status: bool,
}

/// Types of cognitive granularity for tagged concepts
#[derive(Debug, Clone, PartialEq)]
pub enum CognitiveGranularity {
    Micro,    // Morphemes, function words
    Standard, // Common words and basic concepts
    Macro,    // Complex concepts and abstractions
    Meta,     // Entire knowledge domains
}

/// Cognitive processing metadata
#[derive(Debug, Clone, PartialEq)]
pub struct CognitiveMetadata {
    pub granularity: CognitiveGranularity,
    pub complexity: u32,      // Processing complexity score (0-10000)
    pub activation_cost: u32, // Computational resources needed (0-10000)
    pub coherence_score: u32, // Internal conceptual coherence (0-10000)
    pub certainty: u32,       // Confidence in concept validity (0-10000)
}

/// Types of cognitive relationships between concepts
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CognitiveType {
    Composes,       // Concept composition relationships
    Abstracts,      // Abstraction/specification relationships
    Associates,     // Associative relationships
    Implies,        // Logical implication
    Activates,      // Cognitive activation patterns
    Custom(String), // Domain-specific relationships
}

impl RelationshipType {
    /// Creates a new structural relationship
    pub fn structural(kind: StructuralType, metadata: Option<StructuralMetadata>) -> Self {
        RelationshipType::Structural { kind, metadata }
    }

    /// Creates a new linguistic relationship
    pub fn linguistic(kind: LinguisticType, features: Option<LinguisticFeatures>) -> Self {
        RelationshipType::Linguistic { kind, features }
    }

    /// Creates a new discourse relationship
    pub fn discourse(kind: DiscourseType, properties: Option<DiscourseProperties>) -> Self {
        RelationshipType::Discourse { kind, properties }
    }

    /// Creates a new temporal relationship
    pub fn temporal(order: TemporalOrder, timing: Option<TemporalTiming>) -> Self {
        RelationshipType::Temporal { order, timing }
    }

    /// Creates a new pragmatic relationship
    pub fn pragmatic(kind: PragmaticType, context: Option<SocialContext>) -> Self {
        RelationshipType::Pragmatic { kind, context }
    }

    /// Creates a new knowledge relationship
    pub fn knowledge(kind: KnowledgeType, metadata: Option<KnowledgeMetadata>) -> Self {
        RelationshipType::Knowledge { kind, metadata }
    }

    /// Creates a new cognitive relationship
    pub fn cognitive(kind: CognitiveType, metadata: Option<CognitiveMetadata>) -> Self {
        RelationshipType::Cognitive { kind, metadata }
    }
}

impl fmt::Display for CognitiveType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CognitiveType::Composes => write!(f, "composes"),
            CognitiveType::Abstracts => write!(f, "abstracts"),
            CognitiveType::Associates => write!(f, "associates with"),
            CognitiveType::Implies => write!(f, "implies"),
            CognitiveType::Activates => write!(f, "activates"),
            CognitiveType::Custom(name) => write!(f, "custom cognitive: {name}"),
        }
    }
}

impl fmt::Display for RelationshipType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RelationshipType::Structural { kind, metadata } => {
                write!(f, "Structural({kind:?}, {metadata:?})")
            }
            RelationshipType::Linguistic { kind, features } => {
                write!(f, "Linguistic({kind:?}, {features:?})")
            }
            RelationshipType::Discourse { kind, properties } => {
                write!(f, "Discourse({kind:?}, {properties:?})")
            }
            RelationshipType::Temporal { order, timing } => {
                write!(f, "Temporal({order:?}, {timing:?})")
            }
            RelationshipType::Pragmatic { kind, context } => {
                write!(f, "Pragmatic({kind:?}, {context:?})")
            }
            RelationshipType::Knowledge { kind, metadata } => {
                write!(f, "Knowledge({kind:?}, {metadata:?})")
            }
            RelationshipType::Cognitive { kind, metadata } => {
                write!(f, "Cognitive({kind}, {metadata:?})")
            }
        }
    }
}

/// Types of associative relationships
#[derive(Debug, Clone, PartialEq)]
pub enum AssociativeType {
    Similar,                         // Similarity relationships
    Opposite,                        // Opposite/contrasting concepts
    CoOccurs,                        // Co-occurrence patterns
    PartOf { is_part: bool },        // Part/whole relationships
    Contains { is_container: bool }, // Containment relationships
    Custom(String),                  // Custom associative relationships
}

impl fmt::Display for AssociativeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AssociativeType::Similar => write!(f, "similar to"),
            AssociativeType::Opposite => write!(f, "opposite of"),
            AssociativeType::CoOccurs => write!(f, "co-occurs with"),
            AssociativeType::PartOf { is_part } => {
                if *is_part {
                    write!(f, "part of")
                } else {
                    write!(f, "has part")
                }
            }
            AssociativeType::Contains { is_container } => {
                if *is_container {
                    write!(f, "contains")
                } else {
                    write!(f, "within")
                }
            }
            AssociativeType::Custom(name) => write!(f, "custom association: {name}"),
        }
    }
}

impl AssociativeType {
    /// Creates a PartOf relationship where this node is the part
    pub fn as_part() -> Self {
        Self::PartOf { is_part: true }
    }

    /// Creates a PartOf relationship where this node is the whole
    pub fn as_whole() -> Self {
        Self::PartOf { is_part: false }
    }

    /// Creates a Contains relationship where this node is the container
    pub fn as_container() -> Self {
        Self::Contains { is_container: true }
    }

    /// Creates a Contains relationship where this node is the contained
    pub fn as_contained() -> Self {
        Self::Contains {
            is_container: false,
        }
    }

    /// Returns true if this is a symmetric relationship type
    pub fn is_symmetric(&self) -> bool {
        matches!(self, Self::Similar | Self::Opposite | Self::CoOccurs)
    }

    /// Creates the corresponding relationship for the other side
    pub fn reciprocal(&self) -> Self {
        match self {
            // Symmetric relationships are the same from both sides
            Self::Similar => Self::Similar,
            Self::Opposite => Self::Opposite,
            Self::CoOccurs => Self::CoOccurs,

            // Role-based relationships flip their role
            Self::PartOf { is_part } => Self::PartOf { is_part: !is_part },
            Self::Contains { is_container } => Self::Contains {
                is_container: !is_container,
            },

            // Custom relationships stay the same (semantics defined by user)
            Self::Custom(name) => Self::Custom(name.clone()),
        }
    }
}

/// A reference to a relationship between two nodes
#[derive(Debug, Clone)]
pub struct RelationshipRef {
    pub source: NodeKey,
    pub target: NodeKey,
    pub layer: ContextLayer,
    pub rel_type: RelationshipType,
}

/// Query criteria for finding relationships
#[derive(Debug, Clone)]
pub struct RelationshipQuery {
    /// Optional source node to filter by
    pub source: Option<NodeKey>,
    /// Optional target node to filter by  
    pub target: Option<NodeKey>,
    /// Optional relationship type to filter by
    pub rel_type: Option<RelationshipType>,
}

impl Default for RelationshipQuery {
    fn default() -> Self {
        Self::new()
    }
}

impl RelationshipQuery {
    pub fn new() -> Self {
        Self {
            source: None,
            target: None,
            rel_type: None,
        }
    }

    pub fn with_source(mut self, source: NodeKey) -> Self {
        self.source = Some(source);
        self
    }

    pub fn with_target(mut self, target: NodeKey) -> Self {
        self.target = Some(target);
        self
    }

    pub fn with_type(mut self, rel_type: RelationshipType) -> Self {
        self.rel_type = Some(rel_type);
        self
    }
}

/// Trait for relationship inheritance capabilities
pub trait InheritableRelationship {
    /// Get the parent relationship type, if any
    fn parent_type(&self) -> Option<&RelationshipType>;

    /// Check if this relationship is derived from another
    fn inherits_from(&self, other: &RelationshipType) -> bool;

    /// Get the inheritance chain from root to this type
    fn inheritance_chain(&self) -> Vec<RelationshipType>;

    /// Validate inheritance according to rules
    fn validate_inheritance(&self) -> bool;
}

/// Base trait for all relationship types
pub trait BaseRelationship {
    /// Get the context layer this relationship belongs to
    fn context_layer(&self) -> ContextLayer;

    /// Check if this relationship type can connect given node types
    fn can_connect(&self, source_type: &str, target_type: &str) -> bool;

    /// Get metadata associated with this relationship
    fn metadata(&self) -> Option<&RelationshipMetadata>;
}

/// Common metadata for all relationship types
#[derive(Debug, Clone, PartialEq)]
pub struct RelationshipMetadata {
    /// Unique identifier for the relationship type
    pub id: String,
    /// Human readable name
    pub name: String,
    /// Optional description
    pub description: Option<String>,
    /// Valid source node types
    pub valid_sources: Vec<String>,
    /// Valid target node types
    pub valid_targets: Vec<String>,
    /// Parent relationship type ID if inherited
    pub parent_id: Option<String>,
}

impl InheritableRelationship for RelationshipType {
    fn parent_type(&self) -> Option<&RelationshipType> {
        match self {
            RelationshipType::Structural { metadata, .. } => {
                metadata.as_ref().and_then(|m| m.parent_type.as_deref())
            }
            RelationshipType::Linguistic { features, .. } => {
                features.as_ref().and_then(|f| f.parent_type.as_deref())
            }
            // Add cases for other relationship types...
            _ => None,
        }
    }

    fn inherits_from(&self, other: &RelationshipType) -> bool {
        if self == other {
            return true;
        }
        match self.parent_type() {
            Some(parent) => parent.inherits_from(other),
            None => false,
        }
    }

    fn inheritance_chain(&self) -> Vec<RelationshipType> {
        let mut chain = vec![self.clone()];
        let mut current = self;
        while let Some(parent) = current.parent_type() {
            chain.push(parent.clone());
            current = parent;
        }
        chain.reverse();
        chain
    }

    fn validate_inheritance(&self) -> bool {
        // Get inheritance chain
        let chain = self.inheritance_chain();

        // Check for cycles
        let mut seen = std::collections::HashSet::new();
        for rel_type in &chain {
            if !seen.insert(format!("{:?}", rel_type)) {
                return false;
            }
        }

        // Validate layer compatibility
        let base_layer = chain[0].context_layer();
        chain.iter().all(|rt| rt.context_layer() == base_layer)
    }
}

impl BaseRelationship for RelationshipType {
    fn context_layer(&self) -> ContextLayer {
        match self {
            RelationshipType::Structural { .. } => ContextLayer::Structural,
            RelationshipType::Linguistic { .. } => ContextLayer::Linguistic,
            RelationshipType::Discourse { .. } => ContextLayer::Discourse,
            RelationshipType::Temporal { .. } => ContextLayer::Temporal,
            RelationshipType::Pragmatic { .. } => ContextLayer::Pragmatic,
            RelationshipType::Knowledge { .. } => ContextLayer::Knowledge,
            RelationshipType::Cognitive { .. } => ContextLayer::Cognitive,
        }
    }

    fn can_connect(&self, source_type: &str, target_type: &str) -> bool {
        match self {
            RelationshipType::Structural { metadata, .. } => metadata.as_ref().map_or(false, |m| {
                m.base.valid_sources.contains(&source_type.to_string())
                    && m.base.valid_targets.contains(&target_type.to_string())
            }),
            RelationshipType::Linguistic { features, .. } => features.as_ref().map_or(false, |f| {
                f.base.valid_sources.contains(&source_type.to_string())
                    && f.base.valid_targets.contains(&target_type.to_string())
            }),
            // Add cases for other relationship types...
            _ => false,
        }
    }

    fn metadata(&self) -> Option<&RelationshipMetadata> {
        match self {
            RelationshipType::Structural { metadata, .. } => metadata.as_ref().map(|m| &m.base),
            RelationshipType::Linguistic { features, .. } => features.as_ref().map(|f| &f.base),
            // Add cases for other relationship types...
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_create_temporal_relationship() {
        let rel = RelationshipType::temporal(TemporalOrder::Before, None);
        assert!(matches!(rel, RelationshipType::Temporal { .. }));
    }

    #[test]
    fn should_create_valid_structural_relationship() {
        let rel = RelationshipType::structural(StructuralType::Contains, None);
        assert!(matches!(rel, RelationshipType::Structural { .. }));
    }

    #[test]
    fn should_create_linguistic_relationship() {
        let rel = RelationshipType::linguistic(LinguisticType::DependsOn, None);
        assert!(matches!(rel, RelationshipType::Linguistic { .. }));
    }

    #[test]
    fn should_create_discourse_relationship() {
        let rel = RelationshipType::discourse(DiscourseType::Elaborates, None);
        assert!(matches!(rel, RelationshipType::Discourse { .. }));
    }

    #[test]
    fn should_create_pragmatic_relationship() {
        let rel = RelationshipType::pragmatic(PragmaticType::AuthoredBy, None);
        assert!(matches!(rel, RelationshipType::Pragmatic { .. }));
    }

    #[test]
    fn should_create_knowledge_relationship() {
        let rel = RelationshipType::knowledge(KnowledgeType::DefinesEntity, None);
        assert!(matches!(rel, RelationshipType::Knowledge { .. }));
    }

    #[test]
    fn should_create_cognitive_relationship() {
        let metadata = CognitiveMetadata {
            granularity: CognitiveGranularity::Standard,
            complexity: 5000,      // 0.5 in fixed-point format
            activation_cost: 3000, // 0.3 in fixed-point format
            coherence_score: 8000, // 0.8 in fixed-point format
            certainty: 9000,       // 0.9 in fixed-point format
        };
        let rel = RelationshipType::cognitive(CognitiveType::Composes, Some(metadata));
        assert!(matches!(rel, RelationshipType::Cognitive { .. }));
    }

    #[test]
    fn should_display_relationship_types() {
        let rel = RelationshipType::cognitive(
            CognitiveType::Composes,
            Some(CognitiveMetadata {
                granularity: CognitiveGranularity::Standard,
                complexity: 5000,      // 0.5 in fixed-point format
                activation_cost: 3000, // 0.3 in fixed-point format
                coherence_score: 8000, // 0.8 in fixed-point format
                certainty: 9000,       // 0.9 in fixed-point format
            }),
        );
        assert!(rel.to_string().contains("Cognitive"));
    }

    #[test]
    fn should_display_associative_types() {
        assert_eq!(format!("{}", AssociativeType::Similar), "similar to");
        assert_eq!(
            format!("{}", AssociativeType::Custom("requires".to_string())),
            "custom association: requires"
        );
    }
}
