use crate::node_arena::{NodeArena, NodeKey};
use crate::relationship_types::*;
use parking_lot::RwLock;
use std::collections::HashMap;

/// Manages relationships between nodes with efficient indexing and querying
pub struct RelationshipManager {
    /// Node storage and management
    arena: NodeArena,
    /// Layer-specific relationship indices
    indices: RelationshipIndices,
    /// Relationship statistics for optimization
    stats: RwLock<RelationshipStats>,
}

/// Indices for efficient relationship querying
#[derive(Default)]
struct RelationshipIndices {
    /// Index by context layer
    by_layer: HashMap<ContextLayer, Vec<(NodeKey, NodeKey)>>,
    /// Index for structural relationships
    structural: HashMap<StructuralType, Vec<(NodeKey, NodeKey)>>,
    /// Index for linguistic relationships
    linguistic: HashMap<LinguisticType, Vec<(NodeKey, NodeKey)>>,
    /// Index for discourse relationships
    discourse: HashMap<DiscourseType, Vec<(NodeKey, NodeKey)>>,
    /// Index for temporal relationships
    temporal: HashMap<TemporalOrder, Vec<(NodeKey, NodeKey)>>,
    /// Index for pragmatic relationships
    pragmatic: HashMap<PragmaticType, Vec<(NodeKey, NodeKey)>>,
    /// Index for knowledge relationships
    knowledge: HashMap<KnowledgeType, Vec<(NodeKey, NodeKey)>>,
    /// Index for cognitive relationships
    cognitive: HashMap<CognitiveType, Vec<(NodeKey, NodeKey)>>,
}

/// Context layers for relationship categorization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContextLayer {
    Structural,
    Linguistic,
    Discourse,
    Temporal,
    Pragmatic,
    Knowledge,
    Cognitive,
}

/// Statistics about relationships for optimization
#[derive(Default)]
struct RelationshipStats {
    /// Layer-specific counts
    structural_counts: HashMap<StructuralType, usize>,
    linguistic_counts: HashMap<LinguisticType, usize>,
    discourse_counts: HashMap<DiscourseType, usize>,
    temporal_counts: HashMap<TemporalOrder, usize>,
    pragmatic_counts: HashMap<PragmaticType, usize>,
    knowledge_counts: HashMap<KnowledgeType, usize>,
    cognitive_counts: HashMap<CognitiveType, usize>,
    /// Layer-specific query frequencies
    structural_frequencies: HashMap<StructuralType, usize>,
    linguistic_frequencies: HashMap<LinguisticType, usize>,
    discourse_frequencies: HashMap<DiscourseType, usize>,
    temporal_frequencies: HashMap<TemporalOrder, usize>,
    pragmatic_frequencies: HashMap<PragmaticType, usize>,
    knowledge_frequencies: HashMap<KnowledgeType, usize>,
    cognitive_frequencies: HashMap<CognitiveType, usize>,
    /// Layer-specific cache stats
    structural_cache: HashMap<StructuralType, CacheStats>,
    linguistic_cache: HashMap<LinguisticType, CacheStats>,
    discourse_cache: HashMap<DiscourseType, CacheStats>,
    temporal_cache: HashMap<TemporalOrder, CacheStats>,
    pragmatic_cache: HashMap<PragmaticType, CacheStats>,
    knowledge_cache: HashMap<KnowledgeType, CacheStats>,
    cognitive_cache: HashMap<CognitiveType, CacheStats>,
    /// Query counts and result sizes for optimization
    structural_queries: usize,
    linguistic_queries: usize,
    discourse_queries: usize,
    temporal_queries: usize,
    pragmatic_queries: usize,
    knowledge_queries: usize,
    cognitive_queries: usize,
    structural_results: usize,
    linguistic_results: usize,
    discourse_results: usize,
    temporal_results: usize,
    pragmatic_results: usize,
    knowledge_results: usize,
    cognitive_results: usize,
}

#[derive(Default)]
struct CacheStats {
    hits: usize,
    misses: usize,
}

// --- IMPLS MUST BE OUTSIDE STRUCTS ---
impl CacheStats {
    pub fn hit_rate(&self) -> f32 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f32 / total as f32
        }
    }
}

impl RelationshipStats {
    pub fn record_query(&mut self, layer: ContextLayer, rel_type: &RelationshipType) {
        match (layer, rel_type) {
            (ContextLayer::Structural, RelationshipType::Structural { kind, .. }) => {
                *self.structural_frequencies.entry(kind.clone()).or_insert(0) += 1;
            }
            (ContextLayer::Linguistic, RelationshipType::Linguistic { kind, .. }) => {
                *self.linguistic_frequencies.entry(kind.clone()).or_insert(0) += 1;
            }
            (ContextLayer::Discourse, RelationshipType::Discourse { kind, .. }) => {
                *self.discourse_frequencies.entry(kind.clone()).or_insert(0) += 1;
            }
            (ContextLayer::Temporal, RelationshipType::Temporal { order, .. }) => {
                *self.temporal_frequencies.entry(order.clone()).or_insert(0) += 1;
            }
            (ContextLayer::Pragmatic, RelationshipType::Pragmatic { kind, .. }) => {
                *self.pragmatic_frequencies.entry(kind.clone()).or_insert(0) += 1;
            }
            (ContextLayer::Knowledge, RelationshipType::Knowledge { kind, .. }) => {
                *self.knowledge_frequencies.entry(kind.clone()).or_insert(0) += 1;
            }
            (ContextLayer::Cognitive, RelationshipType::Cognitive { kind, .. }) => {
                *self.cognitive_frequencies.entry(kind.clone()).or_insert(0) += 1;
            }
            _ => {}
        }
    }

    pub fn record_cache_hit(&mut self, rel_type: &RelationshipType) {
        match rel_type {
            RelationshipType::Structural { kind, .. } => {
                self.structural_cache.entry(kind.clone()).or_default().hits += 1;
            }
            RelationshipType::Linguistic { kind, .. } => {
                self.linguistic_cache.entry(kind.clone()).or_default().hits += 1;
            }
            RelationshipType::Discourse { kind, .. } => {
                self.discourse_cache.entry(kind.clone()).or_default().hits += 1;
            }
            RelationshipType::Temporal { order, .. } => {
                self.temporal_cache.entry(order.clone()).or_default().hits += 1;
            }
            RelationshipType::Pragmatic { kind, .. } => {
                self.pragmatic_cache.entry(kind.clone()).or_default().hits += 1;
            }
            RelationshipType::Knowledge { kind, .. } => {
                self.knowledge_cache.entry(kind.clone()).or_default().hits += 1;
            }
            RelationshipType::Cognitive { kind, .. } => {
                self.cognitive_cache.entry(kind.clone()).or_default().hits += 1;
            }
        }
    }

    pub fn record_cache_miss(&mut self, rel_type: &RelationshipType) {
        match rel_type {
            RelationshipType::Structural { kind, .. } => {
                self.structural_cache
                    .entry(kind.clone())
                    .or_default()
                    .misses += 1;
            }
            RelationshipType::Linguistic { kind, .. } => {
                self.linguistic_cache
                    .entry(kind.clone())
                    .or_default()
                    .misses += 1;
            }
            RelationshipType::Discourse { kind, .. } => {
                self.discourse_cache.entry(kind.clone()).or_default().misses += 1;
            }
            RelationshipType::Temporal { order, .. } => {
                self.temporal_cache.entry(order.clone()).or_default().misses += 1;
            }
            RelationshipType::Pragmatic { kind, .. } => {
                self.pragmatic_cache.entry(kind.clone()).or_default().misses += 1;
            }
            RelationshipType::Knowledge { kind, .. } => {
                self.knowledge_cache.entry(kind.clone()).or_default().misses += 1;
            }
            RelationshipType::Cognitive { kind, .. } => {
                self.cognitive_cache.entry(kind.clone()).or_default().misses += 1;
            }
        }
    }

    /// Records statistics about a completed relationship query
    pub fn record_query_stats(
        &mut self,
        layer: ContextLayer,
        query: &RelationshipQuery,
        results: &[RelationshipRef],
    ) {
        // Record basic query info
        if let Some(rel_type) = &query.rel_type {
            self.record_query(layer, rel_type);
        }

        // Record query result size
        match layer {
            ContextLayer::Structural => {
                self.structural_queries += 1;
                self.structural_results += results.len();
            }
            ContextLayer::Linguistic => {
                self.linguistic_queries += 1;
                self.linguistic_results += results.len();
            }
            ContextLayer::Discourse => {
                self.discourse_queries += 1;
                self.discourse_results += results.len();
            }
            ContextLayer::Temporal => {
                self.temporal_queries += 1;
                self.temporal_results += results.len();
            }
            ContextLayer::Pragmatic => {
                self.pragmatic_queries += 1;
                self.pragmatic_results += results.len();
            }
            ContextLayer::Knowledge => {
                self.knowledge_queries += 1;
                self.knowledge_results += results.len();
            }
            ContextLayer::Cognitive => {
                self.cognitive_queries += 1;
                self.cognitive_results += results.len();
            }
        }
    }
}

impl Default for RelationshipManager {
    fn default() -> Self {
        Self::new()
    }
}

impl RelationshipManager {
    pub fn new() -> Self {
        Self {
            arena: NodeArena::new(),
            indices: RelationshipIndices::default(),
            stats: RwLock::new(RelationshipStats::default()),
        }
    }

    /// Query relationships based on source, target, and relationship type criteria
    pub fn query_relationships(
        &self,
        layer: ContextLayer,
        query: &RelationshipQuery,
    ) -> Vec<RelationshipRef> {
        let mut results = Vec::new();

        // Get all relationships for the given layer
        if let Some(layer_rels) = self.indices.by_layer.get(&layer) {
            // Record the query for statistics if there's a relationship type
            if let Some(ref rel_type) = query.rel_type {
                self.stats.write().record_query(layer, rel_type);
            }

            for (source, target) in layer_rels {
                // Skip if source doesn't match query
                if let Some(query_source) = query.source {
                    if query_source != *source {
                        continue;
                    }
                }

                // Skip if target doesn't match query
                if let Some(query_target) = query.target {
                    if query_target != *target {
                        continue;
                    }
                }

                // Get relationship type for this pair
                if let Some(rel_type) = self.get_relationship_type(*source, *target, layer) {
                    // Skip if type doesn't match query
                    if let Some(ref query_type) = query.rel_type {
                        if !self.relationship_types_match(query_type, &rel_type) {
                            self.stats.write().record_cache_miss(query_type);
                            continue;
                        }
                        self.stats.write().record_cache_hit(query_type);
                    }

                    results.push(RelationshipRef {
                        source: *source,
                        target: *target,
                        layer,
                        rel_type: rel_type.clone(),
                    });
                } else if let Some(ref query_type) = query.rel_type {
                    // Record cache miss for non-existent relationships
                    self.stats.write().record_cache_miss(query_type);
                }
            }
        } else if let Some(ref query_type) = query.rel_type {
            // Record cache miss when no relationships exist for the layer
            self.stats.write().record_cache_miss(query_type);
        }

        // Record final query statistics
        self.stats
            .write()
            .record_query_stats(layer, query, &results);

        results
    }

    /// Helper to check if two relationship types match, considering variants
    fn relationship_types_match(
        &self,
        query_type: &RelationshipType,
        actual_type: &RelationshipType,
    ) -> bool {
        match (query_type, actual_type) {
            (
                RelationshipType::Structural { kind: q_kind, .. },
                RelationshipType::Structural { kind: a_kind, .. },
            ) => q_kind == a_kind,

            (
                RelationshipType::Linguistic { kind: q_kind, .. },
                RelationshipType::Linguistic { kind: a_kind, .. },
            ) => q_kind == a_kind,

            (
                RelationshipType::Discourse { kind: q_kind, .. },
                RelationshipType::Discourse { kind: a_kind, .. },
            ) => q_kind == a_kind,

            (
                RelationshipType::Temporal { order: q_order, .. },
                RelationshipType::Temporal { order: a_order, .. },
            ) => q_order == a_order,

            (
                RelationshipType::Pragmatic { kind: q_kind, .. },
                RelationshipType::Pragmatic { kind: a_kind, .. },
            ) => q_kind == a_kind,

            (
                RelationshipType::Knowledge { kind: q_kind, .. },
                RelationshipType::Knowledge { kind: a_kind, .. },
            ) => q_kind == a_kind,

            (
                RelationshipType::Cognitive { kind: q_kind, .. },
                RelationshipType::Cognitive { kind: a_kind, .. },
            ) => q_kind == a_kind,

            _ => false,
        }
    }

    /// Get the relationship type between two nodes in a specific layer
    fn get_relationship_type(
        &self,
        source: NodeKey,
        target: NodeKey,
        layer: ContextLayer,
    ) -> Option<RelationshipType> {
        match layer {
            ContextLayer::Structural => {
                for (kind, pairs) in &self.indices.structural {
                    if pairs.contains(&(source, target)) {
                        return Some(RelationshipType::Structural {
                            kind: kind.clone(),
                            metadata: None,
                        });
                    }
                }
            }
            ContextLayer::Linguistic => {
                for (kind, pairs) in &self.indices.linguistic {
                    if pairs.contains(&(source, target)) {
                        return Some(RelationshipType::Linguistic {
                            kind: kind.clone(),
                            features: None,
                        });
                    }
                }
            }
            ContextLayer::Discourse => {
                for (kind, pairs) in &self.indices.discourse {
                    if pairs.contains(&(source, target)) {
                        return Some(RelationshipType::Discourse {
                            kind: kind.clone(),
                            properties: None,
                        });
                    }
                }
            }
            ContextLayer::Temporal => {
                for (kind, pairs) in &self.indices.temporal {
                    if pairs.contains(&(source, target)) {
                        return Some(RelationshipType::Temporal {
                            order: kind.clone(),
                            timing: None,
                        });
                    }
                }
            }
            ContextLayer::Pragmatic => {
                for (kind, pairs) in &self.indices.pragmatic {
                    if pairs.contains(&(source, target)) {
                        return Some(RelationshipType::Pragmatic {
                            kind: kind.clone(),
                            context: None,
                        });
                    }
                }
            }
            ContextLayer::Knowledge => {
                for (kind, pairs) in &self.indices.knowledge {
                    if pairs.contains(&(source, target)) {
                        return Some(RelationshipType::Knowledge {
                            kind: kind.clone(),
                            metadata: None,
                        });
                    }
                }
            }
            ContextLayer::Cognitive => {
                for (kind, pairs) in &self.indices.cognitive {
                    if pairs.contains(&(source, target)) {
                        return Some(RelationshipType::Cognitive {
                            kind: kind.clone(),
                            metadata: None,
                        });
                    }
                }
            }
        }
        None
    }

    /// Find discourse relationships matching the query criteria
    pub fn find_discourse_relationships(&self, query: &RelationshipQuery) -> Vec<RelationshipRef> {
        self.query_relationships(ContextLayer::Discourse, query)
    }

    /// Find pragmatic relationships matching the query criteria
    pub fn find_pragmatic_relationships(&self, query: &RelationshipQuery) -> Vec<RelationshipRef> {
        self.query_relationships(ContextLayer::Pragmatic, query)
    }

    /// Find knowledge relationships matching the query criteria
    pub fn find_knowledge_relationships(&self, query: &RelationshipQuery) -> Vec<RelationshipRef> {
        self.query_relationships(ContextLayer::Knowledge, query)
    }

    /// Find cognitive relationships matching the query criteria
    pub fn find_cognitive_relationships(&self, query: &RelationshipQuery) -> Vec<RelationshipRef> {
        self.query_relationships(ContextLayer::Cognitive, query)
    }

    /// Find structural relationships matching the query criteria
    pub fn find_structural_relationships(&self, query: &RelationshipQuery) -> Vec<RelationshipRef> {
        self.query_relationships(ContextLayer::Structural, query)
    }

    /// Find linguistic relationships matching the query criteria
    pub fn find_linguistic_relationships(&self, query: &RelationshipQuery) -> Vec<RelationshipRef> {
        self.query_relationships(ContextLayer::Linguistic, query)
    }

    /// Find temporal relationships matching the query criteria
    pub fn find_temporal_relationships(&self, query: &RelationshipQuery) -> Vec<RelationshipRef> {
        self.query_relationships(ContextLayer::Temporal, query)
    }

    /// Legacy wrapper to maintain compatibility with old API
    pub fn find_relationships(&self, source: NodeKey, rel_type: &RelationshipType) -> Vec<NodeKey> {
        let query = RelationshipQuery {
            source: Some(source),
            target: None,
            rel_type: Some(rel_type.clone()),
        };

        let layer = match rel_type {
            RelationshipType::Structural { .. } => ContextLayer::Structural,
            RelationshipType::Linguistic { .. } => ContextLayer::Linguistic,
            RelationshipType::Discourse { .. } => ContextLayer::Discourse,
            RelationshipType::Temporal { .. } => ContextLayer::Temporal,
            RelationshipType::Pragmatic { .. } => ContextLayer::Pragmatic,
            RelationshipType::Knowledge { .. } => ContextLayer::Knowledge,
            RelationshipType::Cognitive { .. } => ContextLayer::Cognitive,
        };

        // Convert RelationshipRefs to just target NodeKeys for backwards compatibility
        self.query_relationships(layer, &query)
            .into_iter()
            .map(|rel| rel.target)
            .collect()
    }

    /// Add a relationship between nodes with efficient indexing
    pub fn add_relationship(
        &mut self,
        from: NodeKey,
        to: NodeKey,
        relationship_type: RelationshipType,
    ) {
        // Add to core arena storage
        if let Some(from_node) = self.arena.get_node_mut(from) {
            from_node.relationships.push(RelationshipSide {
                other_node: to,
                rel_type: relationship_type.clone(),
                is_source: true,
            });
        }

        // Add reverse relationship
        if let Some(to_node) = self.arena.get_node_mut(to) {
            to_node.relationships.push(RelationshipSide {
                other_node: from,
                rel_type: relationship_type.clone(),
                is_source: false,
            });
        }

        // Update indices
        self.update_indices(from, to, &relationship_type);

        // Update stats
        let mut stats = self.stats.write();
        match relationship_type {
            RelationshipType::Structural { kind, .. } => {
                *stats.structural_counts.entry(kind).or_insert(0) += 1;
            }
            RelationshipType::Linguistic { kind, .. } => {
                *stats.linguistic_counts.entry(kind).or_insert(0) += 1;
            }
            RelationshipType::Discourse { kind, .. } => {
                *stats.discourse_counts.entry(kind).or_insert(0) += 1;
            }
            RelationshipType::Temporal { order, .. } => {
                *stats.temporal_counts.entry(order).or_insert(0) += 1;
            }
            RelationshipType::Pragmatic { kind, .. } => {
                *stats.pragmatic_counts.entry(kind).or_insert(0) += 1;
            }
            RelationshipType::Knowledge { kind, .. } => {
                *stats.knowledge_counts.entry(kind).or_insert(0) += 1;
            }
            RelationshipType::Cognitive { kind, .. } => {
                *stats.cognitive_counts.entry(kind).or_insert(0) += 1;
            }
        }
    }

    /// Find related nodes filtered by context layer
    pub fn find_related_in_layer(&self, node: NodeKey, layer: ContextLayer) -> Vec<NodeKey> {
        if let Some(pairs) = self.indices.by_layer.get(&layer) {
            pairs
                .iter()
                .filter_map(|(n1, n2)| {
                    if *n1 == node {
                        Some(*n2)
                    } else if *n2 == node {
                        Some(*n1)
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Find nodes with specific structural relationships

    /// Find nodes with specific linguistic relationships

    /// Update relationship indices for efficient querying
    fn update_indices(&mut self, from: NodeKey, to: NodeKey, rel_type: &RelationshipType) {
        // Add to layer index
        let layer = match rel_type {
            RelationshipType::Structural { .. } => ContextLayer::Structural,
            RelationshipType::Linguistic { .. } => ContextLayer::Linguistic,
            RelationshipType::Discourse { .. } => ContextLayer::Discourse,
            RelationshipType::Temporal { .. } => ContextLayer::Temporal,
            RelationshipType::Pragmatic { .. } => ContextLayer::Pragmatic,
            RelationshipType::Knowledge { .. } => ContextLayer::Knowledge,
            RelationshipType::Cognitive { .. } => ContextLayer::Cognitive,
        };
        self.indices
            .by_layer
            .entry(layer)
            .or_default()
            .push((from, to));

        // Add to type-specific indices
        match rel_type {
            RelationshipType::Structural { kind, .. } => {
                self.indices
                    .structural
                    .entry(kind.clone())
                    .or_default()
                    .push((from, to));
            }
            RelationshipType::Linguistic { kind, .. } => {
                self.indices
                    .linguistic
                    .entry(kind.clone())
                    .or_default()
                    .push((from, to));
            }
            RelationshipType::Discourse { kind, .. } => {
                self.indices
                    .discourse
                    .entry(kind.clone())
                    .or_default()
                    .push((from, to));
            }
            RelationshipType::Temporal { order, .. } => {
                self.indices
                    .temporal
                    .entry(order.clone())
                    .or_default()
                    .push((from, to));
            }
            RelationshipType::Pragmatic { kind, .. } => {
                self.indices
                    .pragmatic
                    .entry(kind.clone())
                    .or_default()
                    .push((from, to));
            }
            RelationshipType::Knowledge { kind, .. } => {
                self.indices
                    .knowledge
                    .entry(kind.clone())
                    .or_default()
                    .push((from, to));
            }
            RelationshipType::Cognitive { kind, .. } => {
                self.indices
                    .cognitive
                    .entry(kind.clone())
                    .or_default()
                    .push((from, to));
            }
        }
    }
    pub fn get_relationships(&self, node: NodeKey) -> Vec<RelationshipSide> {
        self.arena.get_links_safe(node).cloned().unwrap_or_default()
    }

    pub fn find_related(
        &self,
        node: NodeKey,
        relationship_type: &RelationshipType,
    ) -> Vec<NodeKey> {
        self.get_relationships(node)
            .into_iter()
            .filter(|rel| &rel.rel_type == relationship_type)
            .map(|rel| rel.other_node)
            .collect()
    }

    pub fn get_relationships_by_role(&self, node: NodeKey, role: bool) -> Vec<RelationshipSide> {
        self.get_relationships(node)
            .into_iter()
            .filter(|rel| rel.is_source == role)
            .collect()
    }

    pub fn get_source_relationships(&self, node: NodeKey) -> Vec<RelationshipSide> {
        self.get_relationships_by_role(node, true)
    }

    pub fn get_target_relationships(&self, node: NodeKey) -> Vec<RelationshipSide> {
        self.get_relationships_by_role(node, false)
    }

    pub fn get_related_by_type(&self, node: NodeKey, rel_type: RelationshipType) -> Vec<NodeKey> {
        self.get_relationships(node)
            .into_iter()
            .filter(|rel| rel.rel_type == rel_type)
            .map(|rel| rel.other_node)
            .collect()
    }

    pub fn get_related_by_type_and_role(
        &self,
        node: NodeKey,
        rel_type: RelationshipType,
        role: bool,
    ) -> Vec<NodeKey> {
        self.get_relationships(node)
            .into_iter()
            .filter(|rel| rel.rel_type == rel_type && rel.is_source == role)
            .map(|rel| rel.other_node)
            .collect()
    }
    pub fn remove_node_relationships(&mut self, node: NodeKey) {
        // Get all relationships for this node
        let relationships = self.get_relationships(node);

        // Remove the relationships from all connected nodes
        for rel in relationships {
            if let Some(other_links) = self.arena.get_links_mut_safe(rel.other_node) {
                other_links.retain(|r| r.other_node != node);
            }
        }

        // Clear relationships for this node
        if let Some(links) = self.arena.get_links_mut_safe(node) {
            links.clear();
        }
    }
    pub fn remove_relationship(
        &mut self,
        from: NodeKey,
        to: NodeKey,
        relationship_type: &RelationshipType,
    ) {
        // Remove forward relationship
        if let Some(from_links) = self.arena.get_links_mut_safe(from) {
            from_links.retain(|rel| {
                !(rel.other_node == to && rel.rel_type == *relationship_type && rel.is_source)
            });
        }

        // Remove reverse relationship
        if let Some(to_links) = self.arena.get_links_mut_safe(to) {
            to_links.retain(|rel| {
                !(rel.other_node == from && rel.rel_type == *relationship_type && !rel.is_source)
            });
        }
    }

    pub fn clear_relationships(&mut self, node: NodeKey) {
        // Get all relationships to remove
        let relationships_to_remove: Vec<(NodeKey, RelationshipType)> = self
            .get_relationships(node)
            .into_iter()
            .map(|rel| (rel.other_node, rel.rel_type))
            .collect();

        // Remove each relationship
        for (other_node, rel_type) in relationships_to_remove {
            self.remove_relationship(node, other_node, &rel_type);
        }
    }

    pub fn update_relationship_type(
        &mut self,
        from: NodeKey,
        to: NodeKey,
        old_type: &RelationshipType,
        new_type: RelationshipType,
    ) {
        if let Some(from_links) = self.arena.get_links_mut_safe(from) {
            if let Some(rel) = from_links
                .iter_mut()
                .find(|rel| rel.other_node == to && rel.rel_type == *old_type && rel.is_source)
            {
                rel.rel_type = new_type.clone();
            }
        }

        if let Some(to_links) = self.arena.get_links_mut_safe(to) {
            if let Some(rel) = to_links
                .iter_mut()
                .find(|rel| rel.other_node == from && rel.rel_type == *old_type && !rel.is_source)
            {
                rel.rel_type = new_type;
            }
        }
    }

    #[allow(dead_code)] // These methods may be used in future iterations
    fn record_query_stats(&self, layer: ContextLayer, found: bool, rel_type: &RelationshipType) {
        let mut stats = self.stats.write();
        stats.record_query(layer, rel_type);
        if found {
            stats.record_cache_hit(rel_type);
        } else {
            stats.record_cache_miss(rel_type);
        }
    }

    /// Get count of relationships by type
    pub fn get_relationship_counts(&self, layer: ContextLayer) -> HashMap<String, usize> {
        let stats = self.stats.read();
        match layer {
            ContextLayer::Structural => stats
                .structural_counts
                .iter()
                .map(|(k, v)| (format!("{k:?}"), *v))
                .collect(),
            ContextLayer::Linguistic => stats
                .linguistic_counts
                .iter()
                .map(|(k, v)| (format!("{k:?}"), *v))
                .collect(),
            ContextLayer::Discourse => stats
                .discourse_counts
                .iter()
                .map(|(k, v)| (format!("{k:?}"), *v))
                .collect(),
            ContextLayer::Temporal => stats
                .temporal_counts
                .iter()
                .map(|(k, v)| (format!("{k:?}"), *v))
                .collect(),
            ContextLayer::Pragmatic => stats
                .pragmatic_counts
                .iter()
                .map(|(k, v)| (format!("{k:?}"), *v))
                .collect(),
            ContextLayer::Knowledge => stats
                .knowledge_counts
                .iter()
                .map(|(k, v)| (format!("{k:?}"), *v))
                .collect(),
            ContextLayer::Cognitive => stats
                .cognitive_counts
                .iter()
                .map(|(k, v)| (format!("{k:?}"), *v))
                .collect(),
        }
    }

    /// Get query frequencies by type
    pub fn get_query_frequencies(&self, layer: ContextLayer) -> HashMap<String, usize> {
        let stats = self.stats.read();
        match layer {
            ContextLayer::Structural => stats
                .structural_frequencies
                .iter()
                .map(|(k, v)| (format!("{k:?}"), *v))
                .collect(),
            ContextLayer::Linguistic => stats
                .linguistic_frequencies
                .iter()
                .map(|(k, v)| (format!("{k:?}"), *v))
                .collect(),
            ContextLayer::Discourse => stats
                .discourse_frequencies
                .iter()
                .map(|(k, v)| (format!("{k:?}"), *v))
                .collect(),
            ContextLayer::Temporal => stats
                .temporal_frequencies
                .iter()
                .map(|(k, v)| (format!("{k:?}"), *v))
                .collect(),
            ContextLayer::Pragmatic => stats
                .pragmatic_frequencies
                .iter()
                .map(|(k, v)| (format!("{k:?}"), *v))
                .collect(),
            ContextLayer::Knowledge => stats
                .knowledge_frequencies
                .iter()
                .map(|(k, v)| (format!("{k:?}"), *v))
                .collect(),
            ContextLayer::Cognitive => stats
                .cognitive_frequencies
                .iter()
                .map(|(k, v)| (format!("{k:?}"), *v))
                .collect(),
        }
    }

    /// Get cache hit rates by type
    pub fn get_cache_hit_rates(&self, layer: ContextLayer) -> HashMap<String, f32> {
        let stats = self.stats.read();
        match layer {
            ContextLayer::Structural => stats
                .structural_cache
                .iter()
                .map(|(k, v)| (format!("{k:?}"), v.hit_rate()))
                .collect(),
            ContextLayer::Linguistic => stats
                .linguistic_cache
                .iter()
                .map(|(k, v)| (format!("{k:?}"), v.hit_rate()))
                .collect(),
            ContextLayer::Discourse => stats
                .discourse_cache
                .iter()
                .map(|(k, v)| (format!("{k:?}"), v.hit_rate()))
                .collect(),
            ContextLayer::Temporal => stats
                .temporal_cache
                .iter()
                .map(|(k, v)| (format!("{k:?}"), v.hit_rate()))
                .collect(),
            ContextLayer::Pragmatic => stats
                .pragmatic_cache
                .iter()
                .map(|(k, v)| (format!("{k:?}"), v.hit_rate()))
                .collect(),
            ContextLayer::Knowledge => stats
                .knowledge_cache
                .iter()
                .map(|(k, v)| (format!("{k:?}"), v.hit_rate()))
                .collect(),
            ContextLayer::Cognitive => stats
                .cognitive_cache
                .iter()
                .map(|(k, v)| (format!("{k:?}"), v.hit_rate()))
                .collect(),
        }
    }

    /// Convert generic relationship results into RelationshipRef format
    #[allow(dead_code)] // This method will be used in future iterations
    fn to_relationship_refs(
        &self,
        results: Vec<(NodeKey, NodeKey)>,
        layer: ContextLayer,
        base_type: RelationshipType,
    ) -> Vec<RelationshipRef> {
        results
            .into_iter()
            .map(|(source, target)| RelationshipRef {
                source,
                target,
                layer,
                rel_type: base_type.clone(),
            })
            .collect()
    }

    /// Create a new inherited relationship type
    pub fn create_inherited_relationship(
        &mut self,
        parent: &RelationshipType,
        metadata: RelationshipMetadata,
    ) -> Result<RelationshipType, String> {
        // Create new relationship of same variant as parent
        let new_rel = match parent {
            RelationshipType::Structural { kind, .. } => RelationshipType::Structural {
                kind: kind.clone(),
                metadata: Some(StructuralMetadata {
                    base: metadata,
                    parent_type: Some(Box::new(parent.clone())),
                    ordering: None,
                    weight: None,
                }),
            },
            RelationshipType::Linguistic { kind, .. } => RelationshipType::Linguistic {
                kind: kind.clone(),
                features: Some(LinguisticFeatures {
                    base: metadata,
                    parent_type: Some(Box::new(parent.clone())),
                    pos_tag: None,
                    dependency_type: None,
                }),
            },
            // Add cases for other relationship types...
            _ => return Err("Unsupported relationship type for inheritance".to_string()),
        };

        // Validate the new relationship
        if !new_rel.validate_inheritance() {
            return Err("Invalid inheritance chain".to_string());
        }

        Ok(new_rel)
    }

    /// Get all relationships that inherit from a given type
    pub fn get_inherited_relationships(
        &self,
        base_type: &RelationshipType,
    ) -> Vec<RelationshipType> {
        let mut inherited = Vec::new();

        // Check each context layer's index
        for relationships in self.indices.by_layer.values() {
            for &(source, target) in relationships {
                if let Some(rel) = self.get_relationship(source, target) {
                    if rel.inherits_from(base_type) {
                        inherited.push(rel);
                    }
                }
            }
        }

        inherited
    }

    /// Get the relationship between two nodes if it exists
    pub fn get_relationship(&self, source: NodeKey, target: NodeKey) -> Option<RelationshipType> {
        // First check if there's any relationship between these nodes
        for relationships in self.indices.by_layer.values() {
            if relationships.contains(&(source, target)) {
                // Look up the actual relationship from the arena
                if let Some(source_node) = self.arena.get_node(source) {
                    return source_node
                        .relationships
                        .iter()
                        .find(|r| r.other_node == target)
                        .map(|r| r.rel_type.clone());
                }
            }
        }
        None
    } // Removed unused add_relationship_to_indices method as its functionality
      // is now handled directly in the relationship management methods
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroU32;

    /// Helper function to create a test relationship manager
    fn setup_test_manager() -> RelationshipManager {
        RelationshipManager::new()
    }

    /// Helper function to create a test NodeKey
    fn create_node_key(id: u32) -> NodeKey {
        NodeKey(NonZeroU32::new(id).unwrap())
    }

    #[test]
    fn test_relationship_ref_creation() {
        let source = create_node_key(1);
        let target = create_node_key(2);
        let layer = ContextLayer::Structural;
        let rel_type = RelationshipType::Structural {
            kind: StructuralType::Contains,
            metadata: None,
        };

        let rel_ref = RelationshipRef {
            source,
            target,
            layer,
            rel_type: rel_type.clone(),
        };

        assert_eq!(rel_ref.source, source);
        assert_eq!(rel_ref.target, target);
        assert_eq!(rel_ref.layer, layer);
        assert_eq!(rel_ref.rel_type, rel_type);
    }

    #[test]
    fn test_query_relationships() {
        let manager = setup_test_manager();
        let source = create_node_key(1);
        let target = create_node_key(2);
        let layer = ContextLayer::Structural;
        let rel_type = RelationshipType::Structural {
            kind: StructuralType::Contains,
            metadata: None,
        };

        // Create a query
        let query = RelationshipQuery {
            source: Some(source),
            target: Some(target),
            rel_type: Some(rel_type.clone()),
        };

        // Query should return empty results initially
        let results = manager.query_relationships(layer, &query);
        assert!(results.is_empty());
    }

    #[test]
    fn test_statistics_tracking() {
        let manager = setup_test_manager();
        let source = create_node_key(1);
        let target = create_node_key(2);
        let layer = ContextLayer::Structural;
        let rel_type = RelationshipType::Structural {
            kind: StructuralType::Contains,
            metadata: None,
        };

        // Create and execute a query
        let query = RelationshipQuery {
            source: Some(source),
            target: Some(target),
            rel_type: Some(rel_type.clone()),
        };
        manager.query_relationships(layer, &query);

        // Verify statistics were recorded
        let stats = manager.stats.read();
        assert_eq!(stats.structural_queries, 1);
        assert_eq!(stats.structural_results, 0); // No results yet
    }

    #[test]
    fn test_cache_performance() {
        let manager = setup_test_manager();
        let source = create_node_key(1);
        let target = create_node_key(2);
        let layer = ContextLayer::Structural;
        let rel_type = RelationshipType::Structural {
            kind: StructuralType::Contains,
            metadata: None,
        };

        // Create and execute same query twice
        let query = RelationshipQuery {
            source: Some(source),
            target: Some(target),
            rel_type: Some(rel_type.clone()),
        };

        manager.query_relationships(layer, &query);
        manager.query_relationships(layer, &query); // Verify cache stats
        let stats = manager.stats.read();
        let cache_stats = stats
            .structural_cache
            .get(&StructuralType::Contains)
            .expect("Cache stats should be initialized");
        assert!(
            cache_stats.misses > 0,
            "Should have cache misses for non-existent relationships"
        );
    }

    #[test]
    fn test_relationship_types_match() {
        let manager = setup_test_manager();

        let type1 = RelationshipType::Structural {
            kind: StructuralType::Contains,
            metadata: None,
        };
        let type2 = RelationshipType::Structural {
            kind: StructuralType::Contains,
            metadata: Some(StructuralMetadata {
                base: RelationshipMetadata {
                    id: "test.structural".to_string(),
                    name: "Test Structural".to_string(),
                    description: None,
                    valid_sources: vec!["*".to_string()],
                    valid_targets: vec!["*".to_string()],
                    parent_id: None,
                },
                parent_type: None,
                ordering: Some(1),
                weight: Some(1.0),
            }),
        };
        let type3 = RelationshipType::Structural {
            kind: StructuralType::References,
            metadata: None,
        };

        // Same type, different metadata should match
        assert!(manager.relationship_types_match(&type1, &type2));
        // Different types should not match
        assert!(!manager.relationship_types_match(&type1, &type3));
    }

    #[test]
    fn test_query_without_filters() {
        let manager = setup_test_manager();
        let layer = ContextLayer::Structural;

        // Query without any filters
        let query = RelationshipQuery {
            source: None,
            target: None,
            rel_type: None,
        };

        let results = manager.query_relationships(layer, &query);
        assert!(results.is_empty()); // Should return empty for new manager
    }

    #[test]
    fn test_query_statistics_recording() {
        let manager = setup_test_manager();
        let layer = ContextLayer::Linguistic;
        let rel_type = RelationshipType::Linguistic {
            kind: LinguisticType::DependsOn,
            features: None,
        };

        // Create and execute queries
        let query = RelationshipQuery {
            source: None,
            target: None,
            rel_type: Some(rel_type),
        };

        // Execute multiple queries
        for _ in 0..3 {
            manager.query_relationships(layer, &query);
        }

        // Verify query counts
        let stats = manager.stats.read();
        assert_eq!(stats.linguistic_queries, 3);
    }

    #[test]
    fn test_multiple_layers() {
        let manager = setup_test_manager();
        let source = create_node_key(1);
        let target = create_node_key(2);

        // Test structural layer
        let structural_query = RelationshipQuery {
            source: Some(source),
            target: Some(target),
            rel_type: Some(RelationshipType::Structural {
                kind: StructuralType::Contains,
                metadata: None,
            }),
        };
        manager.query_relationships(ContextLayer::Structural, &structural_query);

        // Test linguistic layer
        let linguistic_query = RelationshipQuery {
            source: Some(source),
            target: Some(target),
            rel_type: Some(RelationshipType::Linguistic {
                kind: LinguisticType::DependsOn,
                features: None,
            }),
        };
        manager.query_relationships(ContextLayer::Linguistic, &linguistic_query);

        // Verify layer-specific stats
        let stats = manager.stats.read();
        assert_eq!(stats.structural_queries, 1);
        assert_eq!(stats.linguistic_queries, 1);
    }
}
