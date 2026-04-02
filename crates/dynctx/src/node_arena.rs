use crate::relationship_types::RelationshipSide;
use std::collections::HashMap;
use std::num::NonZeroU32; // RelationshipType unused

/// A unique identifier for a node in the arena. Re-exported at crate root.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub struct NodeKey(pub NonZeroU32);

/// A node in the relationship graph
#[derive(Debug)]
pub struct Node {
    /// The node's data
    pub data: String,
    /// The node's type
    pub node_type: String,
    /// Relationships this node participates in
    pub relationships: Vec<RelationshipSide>,
}

impl Node {
    pub fn new(data: String, node_type: String) -> Self {
        Self {
            data,
            node_type,
            relationships: Vec::new(),
        }
    }
}

/// Arena for storing nodes with relationships
pub struct NodeArena {
    nodes: HashMap<NodeKey, Node>,
    next_id: u32,
}

impl NodeArena {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            next_id: 1,
        }
    }

    pub fn allocate(&mut self, data: String, node_type: String) -> NodeKey {
        let id = NodeKey(NonZeroU32::new(self.next_id).unwrap());
        self.next_id += 1;
        self.nodes.insert(id, Node::new(data, node_type));
        id
    }

    pub fn get_node(&self, key: NodeKey) -> Option<&Node> {
        self.nodes.get(&key)
    }

    pub fn get_node_mut(&mut self, key: NodeKey) -> Option<&mut Node> {
        self.nodes.get_mut(&key)
    }

    pub fn remove_node(&mut self, key: NodeKey) -> Option<Node> {
        self.nodes.remove(&key)
    }

    pub fn get_links_safe(&self, key: NodeKey) -> Option<&Vec<RelationshipSide>> {
        self.nodes.get(&key).map(|node| &node.relationships)
    }

    pub fn get_links_mut_safe(&mut self, key: NodeKey) -> Option<&mut Vec<RelationshipSide>> {
        self.nodes.get_mut(&key).map(|node| &mut node.relationships)
    }

    pub fn add_relationship(&mut self, key: NodeKey, relationship: RelationshipSide) -> bool {
        if let Some(node) = self.nodes.get_mut(&key) {
            node.relationships.push(relationship);
            true
        } else {
            false
        }
    }

    pub fn remove_relationship(&mut self, key: NodeKey, relationship: &RelationshipSide) -> bool {
        if let Some(node) = self.nodes.get_mut(&key) {
            if let Some(idx) = node.relationships.iter().position(|r| r == relationship) {
                node.relationships.remove(idx);
                true
            } else {
                false
            }
        } else {
            false
        }
    }

    pub fn exists(&self, key: NodeKey) -> bool {
        self.nodes.contains_key(&key)
    }
}

impl Default for NodeArena {
    fn default() -> Self {
        Self::new()
    }
}
