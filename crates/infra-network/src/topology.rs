//! Network topology management for distributed systems

use crate::types::{NetworkError, NetworkResult, NodeId, PeerInfo, TopologyType};
use petgraph::{
    graph::{NodeIndex, UnGraph},
    algo::{dijkstra, has_path_connecting},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::Arc,
    time::Duration,
};
use tokio::sync::RwLock;

/// Network topology manager
#[derive(Debug, Clone)]
pub struct NetworkTopology {
    /// Topology configuration
    topology_type: TopologyType,
    /// Graph representation of the network
    graph: Arc<RwLock<UnGraph<NodeId, f64>>>,
    /// Mapping from NodeId to graph NodeIndex
    node_indices: Arc<RwLock<HashMap<NodeId, NodeIndex>>>,
    /// Peer information
    peers: Arc<RwLock<HashMap<NodeId, PeerInfo>>>,
    /// Local node ID
    local_node_id: NodeId,
}

/// Topology statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyStats {
    /// Total number of nodes
    pub node_count: usize,
    /// Total number of edges
    pub edge_count: usize,
    /// Average node degree
    pub average_degree: f64,
    /// Network diameter (longest shortest path)
    pub diameter: Option<usize>,
    /// Clustering coefficient
    pub clustering_coefficient: f64,
    /// Is the network connected?
    pub is_connected: bool,
    /// Number of connected components
    pub connected_components: usize,
}

impl NetworkTopology {
    /// Create a new network topology manager
    pub fn new(topology_type: TopologyType, local_node_id: NodeId) -> Self {
        Self {
            topology_type,
            graph: Arc::new(RwLock::new(UnGraph::new_undirected())),
            node_indices: Arc::new(RwLock::new(HashMap::new())),
            peers: Arc::new(RwLock::new(HashMap::new())),
            local_node_id,
        }
    }

    /// Add a node to the topology
    pub async fn add_node(&self, peer: PeerInfo) -> NetworkResult<()> {
        let node_id = peer.node_id;
        
        // Add to peer list
        self.peers.write().await.insert(node_id, peer);

        // Add to graph
        let mut graph = self.graph.write().await;
        let mut node_indices = self.node_indices.write().await;
        
        if !node_indices.contains_key(&node_id) {
            let index = graph.add_node(node_id);
            node_indices.insert(node_id, index);
        }

        // Apply topology-specific connection rules
        self.apply_topology_rules(node_id).await?;

        Ok(())
    }

    /// Remove a node from the topology
    pub async fn remove_node(&self, node_id: &NodeId) -> NetworkResult<()> {
        // Remove from peer list
        self.peers.write().await.remove(node_id);

        // Remove from graph
        let mut graph = self.graph.write().await;
        let mut node_indices = self.node_indices.write().await;
        
        if let Some(index) = node_indices.remove(node_id) {
            graph.remove_node(index);
        }

        Ok(())
    }

    /// Add an edge between two nodes
    pub async fn add_edge(&self, node1: NodeId, node2: NodeId, weight: f64) -> NetworkResult<()> {
        let graph = self.graph.clone();
        let node_indices = self.node_indices.clone();
        
        let indices_guard = node_indices.read().await;
        let index1 = indices_guard.get(&node1)
            .ok_or_else(|| NetworkError::PeerNotFound(node1))?;
        let index2 = indices_guard.get(&node2)
            .ok_or_else(|| NetworkError::PeerNotFound(node2))?;

        let mut graph_guard = graph.write().await;
        graph_guard.add_edge(*index1, *index2, weight);

        Ok(())
    }

    /// Remove an edge between two nodes
    pub async fn remove_edge(&self, node1: NodeId, node2: NodeId) -> NetworkResult<()> {
        let graph = self.graph.clone();
        let node_indices = self.node_indices.clone();
        
        let indices_guard = node_indices.read().await;
        let index1 = indices_guard.get(&node1)
            .ok_or_else(|| NetworkError::PeerNotFound(node1))?;
        let index2 = indices_guard.get(&node2)
            .ok_or_else(|| NetworkError::PeerNotFound(node2))?;

        let mut graph_guard = graph.write().await;
        if let Some(edge) = graph_guard.find_edge(*index1, *index2) {
            graph_guard.remove_edge(edge);
        }

        Ok(())
    }

    /// Get neighbors of a node
    pub async fn get_neighbors(&self, node_id: &NodeId) -> NetworkResult<Vec<NodeId>> {
        let graph = self.graph.read().await;
        let node_indices = self.node_indices.read().await;
        
        let index = node_indices.get(node_id)
            .ok_or_else(|| NetworkError::PeerNotFound(*node_id))?;

        let neighbors: Vec<NodeId> = graph
            .neighbors(*index)
            .map(|neighbor_index| graph[neighbor_index])
            .collect();

        Ok(neighbors)
    }

    /// Find shortest path between two nodes
    pub async fn shortest_path(&self, source: NodeId, target: NodeId) -> NetworkResult<Vec<NodeId>> {
        let graph = self.graph.read().await;
        let node_indices = self.node_indices.read().await;
        
        let source_index = node_indices.get(&source)
            .ok_or_else(|| NetworkError::PeerNotFound(source))?;
        let target_index = node_indices.get(&target)
            .ok_or_else(|| NetworkError::PeerNotFound(target))?;

        // Use Dijkstra's algorithm
        let paths = dijkstra(&*graph, *source_index, Some(*target_index), |edge| *edge.weight());
        
        if paths.contains_key(target_index) {
            // Reconstruct path (simplified - would need proper path reconstruction)
            Ok(vec![source, target]) // Placeholder
        } else {
            Err(NetworkError::NoRoute(target))
        }
    }

    /// Check if two nodes are connected
    pub async fn are_connected(&self, node1: NodeId, node2: NodeId) -> NetworkResult<bool> {
        let graph = self.graph.read().await;
        let node_indices = self.node_indices.read().await;
        
        let index1 = node_indices.get(&node1)
            .ok_or_else(|| NetworkError::PeerNotFound(node1))?;
        let index2 = node_indices.get(&node2)
            .ok_or_else(|| NetworkError::PeerNotFound(node2))?;

        Ok(has_path_connecting(&*graph, *index1, *index2, None))
    }

    /// Get topology statistics
    pub async fn get_stats(&self) -> TopologyStats {
        let graph = self.graph.read().await;
        let node_count = graph.node_count();
        let edge_count = graph.edge_count();
        
        let average_degree = if node_count > 0 {
            (edge_count * 2) as f64 / node_count as f64
        } else {
            0.0
        };

        // Calculate other metrics (simplified)
        let is_connected = node_count <= 1 || self.is_graph_connected(&*graph).await;
        
        TopologyStats {
            node_count,
            edge_count,
            average_degree,
            diameter: None, // Would implement proper diameter calculation
            clustering_coefficient: 0.0, // Would implement proper clustering coefficient
            is_connected,
            connected_components: if is_connected { 1 } else { node_count },
        }
    }

    /// Update edge weight based on network conditions
    pub async fn update_edge_weight(&self, node1: NodeId, node2: NodeId, latency: Duration) -> NetworkResult<()> {
        // Convert latency to weight (lower latency = lower weight)
        let weight = latency.as_millis() as f64;
        
        // Remove existing edge and add new one with updated weight
        let _ = self.remove_edge(node1, node2).await;
        self.add_edge(node1, node2, weight).await
    }

    /// Optimize topology based on current network conditions
    pub async fn optimize_topology(&self) -> NetworkResult<()> {
        match &self.topology_type {
            TopologyType::Mesh => {
                self.optimize_mesh().await?;
            }
            TopologyType::Star { hub_node } => {
                self.optimize_star(*hub_node).await?;
            }
            TopologyType::Ring => {
                self.optimize_ring().await?;
            }
            TopologyType::Tree { root_node } => {
                self.optimize_tree(*root_node).await?;
            }
            TopologyType::Custom => {
                // Custom optimization logic
            }
        }

        Ok(())
    }

    /// Apply topology-specific connection rules when adding a node
    async fn apply_topology_rules(&self, new_node_id: NodeId) -> NetworkResult<()> {
        match &self.topology_type {
            TopologyType::Mesh => {
                // Connect to all existing nodes
                let peers = self.peers.read().await;
                for &peer_id in peers.keys() {
                    if peer_id != new_node_id {
                        self.add_edge(new_node_id, peer_id, 1.0).await?;
                    }
                }
            }
            TopologyType::Star { hub_node } => {
                // Connect only to hub
                if new_node_id != *hub_node {
                    self.add_edge(new_node_id, *hub_node, 1.0).await?;
                }
            }
            TopologyType::Ring => {
                // Connect to form a ring
                self.maintain_ring_topology(new_node_id).await?;
            }
            TopologyType::Tree { root_node } => {
                // Connect to maintain tree structure
                self.maintain_tree_topology(new_node_id, *root_node).await?;
            }
            TopologyType::Custom => {
                // Custom connection logic
            }
        }

        Ok(())
    }

    /// Optimize mesh topology
    async fn optimize_mesh(&self) -> NetworkResult<()> {
        // Ensure all nodes are connected to all others
        let peers: Vec<NodeId> = self.peers.read().await.keys().copied().collect();
        
        for i in 0..peers.len() {
            for j in (i + 1)..peers.len() {
                let node1 = peers[i];
                let node2 = peers[j];
                
                // Add edge if not exists
                if !self.are_connected(node1, node2).await? {
                    self.add_edge(node1, node2, 1.0).await?;
                }
            }
        }

        Ok(())
    }

    /// Optimize star topology
    async fn optimize_star(&self, hub_node: NodeId) -> NetworkResult<()> {
        let peers: Vec<NodeId> = self.peers.read().await.keys().copied().collect();
        
        for &peer_id in &peers {
            if peer_id != hub_node {
                // Ensure connection to hub
                if !self.are_connected(peer_id, hub_node).await? {
                    self.add_edge(peer_id, hub_node, 1.0).await?;
                }
                
                // Remove connections between non-hub nodes
                for &other_peer in &peers {
                    if other_peer != peer_id && other_peer != hub_node {
                        let _ = self.remove_edge(peer_id, other_peer).await;
                    }
                }
            }
        }

        Ok(())
    }

    /// Optimize ring topology
    async fn optimize_ring(&self) -> NetworkResult<()> {
        let peers: Vec<NodeId> = self.peers.read().await.keys().copied().collect();
        
        if peers.len() < 3 {
            return Ok(()); // Need at least 3 nodes for a ring
        }

        // Remove all existing edges
        let graph = self.graph.clone();
        let mut graph_guard = graph.write().await;
        graph_guard.clear_edges();
        drop(graph_guard);

        // Create ring connections
        for i in 0..peers.len() {
            let current = peers[i];
            let next = peers[(i + 1) % peers.len()];
            self.add_edge(current, next, 1.0).await?;
        }

        Ok(())
    }

    /// Optimize tree topology
    async fn optimize_tree(&self, _root_node: NodeId) -> NetworkResult<()> {
        // Implement tree topology optimization
        // This is a simplified version - would need proper tree construction algorithm
        Ok(())
    }

    /// Maintain ring topology when adding a node
    async fn maintain_ring_topology(&self, new_node_id: NodeId) -> NetworkResult<()> {
        let peers: Vec<NodeId> = self.peers.read().await.keys().copied().collect();
        
        if peers.len() <= 2 {
            // Not enough nodes for a ring yet
            if peers.len() == 2 {
                let other_node = peers.iter().find(|&&id| id != new_node_id).unwrap();
                self.add_edge(new_node_id, *other_node, 1.0).await?;
            }
        } else {
            // Insert into existing ring
            // Find a good insertion point and update connections
            if let Some(&neighbor) = peers.first() {
                if neighbor != new_node_id {
                    self.add_edge(new_node_id, neighbor, 1.0).await?;
                }
            }
        }

        Ok(())
    }

    /// Maintain tree topology when adding a node
    async fn maintain_tree_topology(&self, new_node_id: NodeId, root_node: NodeId) -> NetworkResult<()> {
        if new_node_id == root_node {
            return Ok(()); // Root node doesn't need parent
        }

        // Find a suitable parent node (simplified - would use better heuristics)
        let peers: Vec<NodeId> = self.peers.read().await.keys().copied().collect();
        
        if let Some(&parent) = peers.iter().find(|&&id| id != new_node_id) {
            self.add_edge(new_node_id, parent, 1.0).await?;
        }

        Ok(())
    }

    /// Check if graph is connected
    async fn is_graph_connected(&self, graph: &UnGraph<NodeId, f64>) -> bool {
        if graph.node_count() <= 1 {
            return true;
        }

        // Use DFS to check connectivity (simplified)
        let node_indices: Vec<NodeIndex> = graph.node_indices().collect();
        if node_indices.is_empty() {
            return true;
        }

        let start_node = node_indices[0];
        
        for &target_node in &node_indices[1..] {
            if !has_path_connecting(graph, start_node, target_node, None) {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{NodeAddress, PeerInfo};

    #[tokio::test]
    async fn test_topology_creation() {
        let local_node = NodeId::new();
        let topology = NetworkTopology::new(TopologyType::Mesh, local_node);
        
        let stats = topology.get_stats().await;
        assert_eq!(stats.node_count, 0);
        assert_eq!(stats.edge_count, 0);
    }

    #[tokio::test]
    async fn test_add_remove_node() {
        let local_node = NodeId::new();
        let topology = NetworkTopology::new(TopologyType::Mesh, local_node);
        
        let peer = PeerInfo::new(
            NodeId::new(),
            NodeAddress::new("127.0.0.1:8080".parse().unwrap()),
        );
        let peer_id = peer.node_id;

        // Add node
        topology.add_node(peer).await.unwrap();
        let stats = topology.get_stats().await;
        assert_eq!(stats.node_count, 1);

        // Remove node
        topology.remove_node(&peer_id).await.unwrap();
        let stats = topology.get_stats().await;
        assert_eq!(stats.node_count, 0);
    }

    #[tokio::test]
    async fn test_mesh_topology() {
        let local_node = NodeId::new();
        let topology = NetworkTopology::new(TopologyType::Mesh, local_node);
        
        // Add multiple nodes
        for _ in 0..3 {
            let peer = PeerInfo::new(
                NodeId::new(),
                NodeAddress::new("127.0.0.1:8080".parse().unwrap()),
            );
            topology.add_node(peer).await.unwrap();
        }

        topology.optimize_topology().await.unwrap();
        
        let stats = topology.get_stats().await;
        assert_eq!(stats.node_count, 3);
        // In a complete mesh of 3 nodes, we should have 3 edges
        assert_eq!(stats.edge_count, 3);
    }

    #[tokio::test]
    async fn test_star_topology() {
        let hub_node = NodeId::new();
        let topology = NetworkTopology::new(TopologyType::Star { hub_node }, hub_node);
        
        // Add hub node
        let hub_peer = PeerInfo::new(
            hub_node,
            NodeAddress::new("127.0.0.1:8080".parse().unwrap()),
        );
        topology.add_node(hub_peer).await.unwrap();

        // Add spoke nodes
        for _ in 0..3 {
            let peer = PeerInfo::new(
                NodeId::new(),
                NodeAddress::new("127.0.0.1:8081".parse().unwrap()),
            );
            topology.add_node(peer).await.unwrap();
        }

        topology.optimize_topology().await.unwrap();
        
        let stats = topology.get_stats().await;
        assert_eq!(stats.node_count, 4); // Hub + 3 spokes
        assert_eq!(stats.edge_count, 3); // 3 connections to hub
    }
}
