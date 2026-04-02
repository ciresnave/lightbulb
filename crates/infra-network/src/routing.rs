//! Network routing engine for message delivery and path optimization.

use crate::types::{NodeId, PeerInfo, NetworkResult, RoutingStrategy};
use std::collections::HashMap;
use tokio::sync::RwLock;

/// Routing engine for managing message paths between network nodes
#[derive(Debug)]
pub struct RoutingEngine {
    /// Current routing strategy
    strategy: RoutingStrategy,
    /// Routing table mapping destinations to next hops
    routing_table: RwLock<RoutingTable>,
    /// Network topology cache
    topology_cache: RwLock<HashMap<NodeId, PeerInfo>>,
}

impl Clone for RoutingEngine {
    fn clone(&self) -> Self {
        // For routing engine cloning, we create a new instance with the same strategy
        // but empty tables (as they should be populated independently)
        Self::new(self.strategy.clone())
    }
}

/// Routing table containing paths between nodes
#[derive(Debug, Clone)]
pub struct RoutingTable {
    /// Maps destination node to next hop node
    next_hop: HashMap<NodeId, NodeId>,
    /// Maps destination node to full path
    full_paths: HashMap<NodeId, Vec<NodeId>>,
    /// Maps destination node to cost/distance
    distances: HashMap<NodeId, u32>,
}

/// Route information for a specific destination
#[derive(Debug, Clone, PartialEq)]
pub struct RouteInfo {
    /// Destination node
    pub destination: NodeId,
    /// Next hop node
    pub next_hop: NodeId,
    /// Full path to destination
    pub path: Vec<NodeId>,
    /// Cost/distance to destination
    pub cost: u32,
}

impl RoutingEngine {
    /// Create a new routing engine with the specified strategy
    pub fn new(strategy: RoutingStrategy) -> Self {
        Self {
            strategy,
            routing_table: RwLock::new(RoutingTable::new()),
            topology_cache: RwLock::new(HashMap::new()),
        }
    }

    /// Update the network topology and recompute routing table
    pub async fn update_topology(&self, peers: HashMap<NodeId, PeerInfo>) -> NetworkResult<()> {
        {
            let mut cache = self.topology_cache.write().await;
            *cache = peers.clone();
        }

        self.recompute_routes(peers).await
    }

    /// Get the next hop for routing to a destination
    pub async fn get_next_hop(&self, destination: &NodeId) -> NetworkResult<Option<NodeId>> {
        let table = self.routing_table.read().await;
        Ok(table.next_hop.get(destination).copied())
    }

    /// Get full route information for a destination
    pub async fn get_route(&self, destination: &NodeId) -> NetworkResult<Option<RouteInfo>> {
        let table = self.routing_table.read().await;
        
        if let (Some(&next_hop), Some(path), Some(&cost)) = (
            table.next_hop.get(destination),
            table.full_paths.get(destination),
            table.distances.get(destination),
        ) {
            Ok(Some(RouteInfo {
                destination: *destination,
                next_hop,
                path: path.clone(),
                cost,
            }))
        } else {
            Ok(None)
        }
    }

    /// Get all available routes
    pub async fn get_all_routes(&self) -> NetworkResult<Vec<RouteInfo>> {
        let table = self.routing_table.read().await;
        let mut routes = Vec::new();

        for (&destination, &next_hop) in &table.next_hop {
            if let (Some(path), Some(&cost)) = (
                table.full_paths.get(&destination),
                table.distances.get(&destination),
            ) {
                routes.push(RouteInfo {
                    destination,
                    next_hop,
                    path: path.clone(),
                    cost,
                });
            }
        }

        Ok(routes)
    }

    /// Remove a node from routing table (e.g., when it goes offline)
    pub async fn remove_node(&self, node_id: &NodeId) -> NetworkResult<()> {
        {
            let mut cache = self.topology_cache.write().await;
            cache.remove(node_id);
        }

        {
            let mut table = self.routing_table.write().await;
            table.remove_node(node_id);
        }

        // Recompute routes with updated topology
        let topology = self.topology_cache.read().await.clone();
        self.recompute_routes(topology).await
    }

    /// Check if a destination is reachable
    pub async fn is_reachable(&self, destination: &NodeId) -> NetworkResult<bool> {
        let table = self.routing_table.read().await;
        Ok(table.next_hop.contains_key(destination))
    }

    /// Get routing statistics
    pub async fn get_stats(&self) -> NetworkResult<RoutingStats> {
        let table = self.routing_table.read().await;
        let topology = self.topology_cache.read().await;

        Ok(RoutingStats {
            total_destinations: table.next_hop.len(),
            total_nodes: topology.len(),
            average_path_length: table.calculate_average_path_length(),
            max_distance: table.distances.values().max().copied().unwrap_or(0),
        })
    }

    /// Recompute routing table based on current topology and strategy
    async fn recompute_routes(&self, peers: HashMap<NodeId, PeerInfo>) -> NetworkResult<()> {
        let mut table = self.routing_table.write().await;
          match self.strategy {
            RoutingStrategy::Direct => {
                // Direct routing - route only to directly connected peers
                for (node_id, _) in &peers {
                    table.next_hop.insert(*node_id, *node_id);
                    table.full_paths.insert(*node_id, vec![*node_id]);
                    table.distances.insert(*node_id, 1);
                }
            }
            RoutingStrategy::Flood => {
                // Simple flooding - all peers are directly reachable
                for (node_id, _) in &peers {
                    table.next_hop.insert(*node_id, *node_id);
                    table.full_paths.insert(*node_id, vec![*node_id]);
                    table.distances.insert(*node_id, 1);
                }
            }
            RoutingStrategy::ShortestPath => {
                // Use Dijkstra's algorithm for shortest path routing
                self.compute_shortest_paths(&mut table, &peers).await?;
            }            RoutingStrategy::LoadBalanced => {
                // Use shortest path as base, with load balancing considerations
                self.compute_shortest_paths(&mut table, &peers).await?;
                // TODO: Add load balancing logic
            }
            RoutingStrategy::RandomWalk => {
                // Random walk routing - route to random neighbors
                for (node_id, _) in &peers {
                    table.next_hop.insert(*node_id, *node_id);
                    table.full_paths.insert(*node_id, vec![*node_id]);
                    table.distances.insert(*node_id, 1);
                }
            }
        }

        Ok(())
    }

    /// Compute shortest paths using Dijkstra's algorithm
    async fn compute_shortest_paths(
        &self,
        table: &mut RoutingTable,
        peers: &HashMap<NodeId, PeerInfo>,
    ) -> NetworkResult<()> {
        table.clear();

        // For simplicity, assume all peers are directly connected with cost 1
        // In a real implementation, this would use actual network topology
        for (node_id, _) in peers {
            table.next_hop.insert(*node_id, *node_id);
            table.full_paths.insert(*node_id, vec![*node_id]);
            table.distances.insert(*node_id, 1);
        }

        // TODO: Implement proper Dijkstra's algorithm for multi-hop routing
        // This would require actual network graph structure

        Ok(())
    }
}

impl RoutingTable {
    /// Create a new empty routing table
    pub fn new() -> Self {
        Self {
            next_hop: HashMap::new(),
            full_paths: HashMap::new(),
            distances: HashMap::new(),
        }
    }

    /// Remove all references to a node
    pub fn remove_node(&mut self, node_id: &NodeId) {        self.next_hop.remove(node_id);
        self.full_paths.remove(node_id);
        self.distances.remove(node_id);

        // Remove routes that go through this node
        let routes_to_remove: Vec<NodeId> = self
            .next_hop
            .iter()
            .filter(|(_, next_hop)| **next_hop == *node_id)
            .map(|(dest, _)| *dest)
            .collect();

        for dest in routes_to_remove {
            self.next_hop.remove(&dest);
            self.full_paths.remove(&dest);
            self.distances.remove(&dest);
        }
    }

    /// Clear all routing information
    pub fn clear(&mut self) {
        self.next_hop.clear();
        self.full_paths.clear();
        self.distances.clear();
    }

    /// Calculate average path length
    pub fn calculate_average_path_length(&self) -> f64 {
        if self.full_paths.is_empty() {
            return 0.0;
        }

        let total_length: usize = self.full_paths.values().map(|path| path.len()).sum();
        total_length as f64 / self.full_paths.len() as f64
    }
}

impl Default for RoutingTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Routing statistics
#[derive(Debug, Clone)]
pub struct RoutingStats {
    /// Total number of destinations in routing table
    pub total_destinations: usize,
    /// Total number of nodes in network
    pub total_nodes: usize,
    /// Average path length to destinations
    pub average_path_length: f64,
    /// Maximum distance to any destination
    pub max_distance: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{NodeAddress, PeerStatus};
    use std::collections::HashSet;fn create_test_peer(id: u32) -> (NodeId, PeerInfo) {
        let node_id = NodeId::new();
        let socket_addr = format!("127.0.0.1:{}", 8000 + id).parse().unwrap();
        let address = NodeAddress::new(socket_addr);
        let peer_info = PeerInfo {
            node_id: node_id.clone(),
            address,
            status: PeerStatus::Connected,
            last_seen: std::time::SystemTime::now(),
            latency: None,
            reliability: 1.0,
        };
        (node_id, peer_info)
    }

    #[tokio::test]
    async fn test_routing_engine_creation() {
        let engine = RoutingEngine::new(RoutingStrategy::ShortestPath);
        
        let stats = engine.get_stats().await.unwrap();
        assert_eq!(stats.total_destinations, 0);
        assert_eq!(stats.total_nodes, 0);
    }

    #[tokio::test]
    async fn test_update_topology() {
        let engine = RoutingEngine::new(RoutingStrategy::ShortestPath);
        
        let mut peers = HashMap::new();
        let (node1, peer1) = create_test_peer(1);
        let (node2, peer2) = create_test_peer(2);
        peers.insert(node1.clone(), peer1);
        peers.insert(node2.clone(), peer2);

        engine.update_topology(peers).await.unwrap();

        let stats = engine.get_stats().await.unwrap();
        assert_eq!(stats.total_destinations, 2);
        assert_eq!(stats.total_nodes, 2);
    }

    #[tokio::test]
    async fn test_get_next_hop() {
        let engine = RoutingEngine::new(RoutingStrategy::ShortestPath);
        
        let mut peers = HashMap::new();
        let (node1, peer1) = create_test_peer(1);
        peers.insert(node1.clone(), peer1);

        engine.update_topology(peers).await.unwrap();

        let next_hop = engine.get_next_hop(&node1).await.unwrap();
        assert_eq!(next_hop, Some(node1.clone()));
    }

    #[tokio::test]
    async fn test_get_route() {
        let engine = RoutingEngine::new(RoutingStrategy::ShortestPath);
        
        let mut peers = HashMap::new();
        let (node1, peer1) = create_test_peer(1);
        peers.insert(node1.clone(), peer1);

        engine.update_topology(peers).await.unwrap();

        let route = engine.get_route(&node1).await.unwrap();
        assert!(route.is_some());
        
        let route_info = route.unwrap();
        assert_eq!(route_info.destination, node1);
        assert_eq!(route_info.next_hop, node1);
        assert_eq!(route_info.cost, 1);
    }

    #[tokio::test]
    async fn test_remove_node() {
        let engine = RoutingEngine::new(RoutingStrategy::ShortestPath);
        
        let mut peers = HashMap::new();
        let (node1, peer1) = create_test_peer(1);
        let (node2, peer2) = create_test_peer(2);
        peers.insert(node1.clone(), peer1);
        peers.insert(node2.clone(), peer2);

        engine.update_topology(peers).await.unwrap();

        let stats_before = engine.get_stats().await.unwrap();
        assert_eq!(stats_before.total_destinations, 2);

        engine.remove_node(&node1).await.unwrap();

        let stats_after = engine.get_stats().await.unwrap();
        assert_eq!(stats_after.total_destinations, 1);

        let reachable = engine.is_reachable(&node1).await.unwrap();
        assert!(!reachable);
    }

    #[tokio::test]
    async fn test_is_reachable() {
        let engine = RoutingEngine::new(RoutingStrategy::ShortestPath);
        
        let mut peers = HashMap::new();
        let (node1, peer1) = create_test_peer(1);
        peers.insert(node1.clone(), peer1);

        engine.update_topology(peers).await.unwrap();

        let reachable = engine.is_reachable(&node1).await.unwrap();
        assert!(reachable);

        let unknown_node = NodeId::new();
        let not_reachable = engine.is_reachable(&unknown_node).await.unwrap();
        assert!(!not_reachable);
    }

    #[tokio::test]
    async fn test_get_all_routes() {
        let engine = RoutingEngine::new(RoutingStrategy::ShortestPath);
        
        let mut peers = HashMap::new();
        let (node1, peer1) = create_test_peer(1);
        let (node2, peer2) = create_test_peer(2);
        peers.insert(node1.clone(), peer1);
        peers.insert(node2.clone(), peer2);

        engine.update_topology(peers).await.unwrap();

        let routes = engine.get_all_routes().await.unwrap();
        assert_eq!(routes.len(), 2);
        
        let dest_ids: HashSet<NodeId> = routes.iter().map(|r| r.destination.clone()).collect();
        assert!(dest_ids.contains(&node1));
        assert!(dest_ids.contains(&node2));
    }

    #[test]
    fn test_routing_table() {
        let mut table = RoutingTable::new();
          let node1 = NodeId::new();
        let node2 = NodeId::new();
        
        table.next_hop.insert(node1.clone(), node2.clone());
        table.full_paths.insert(node1.clone(), vec![node1.clone(), node2.clone()]);
        table.distances.insert(node1.clone(), 2);

        assert_eq!(table.calculate_average_path_length(), 2.0);

        table.remove_node(&node1);
        assert!(table.next_hop.is_empty());
        assert!(table.full_paths.is_empty());
        assert!(table.distances.is_empty());
    }

    #[test]
    fn test_routing_stats() {
        let stats = RoutingStats {
            total_destinations: 5,
            total_nodes: 10,
            average_path_length: 2.5,
            max_distance: 4,
        };

        assert_eq!(stats.total_destinations, 5);
        assert_eq!(stats.total_nodes, 10);
        assert_eq!(stats.average_path_length, 2.5);
        assert_eq!(stats.max_distance, 4);
    }
}
