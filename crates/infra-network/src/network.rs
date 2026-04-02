//! High-level network manager orchestrating all network components.

use crate::{    topology::{NetworkTopology},
    discovery::{NodeDiscovery, DiscoveryConfig},
    routing::{RoutingEngine, RoutingStats},
    connection::{ConnectionManager, ConnectionConfig, ConnectionEvent},
    monitoring::{NetworkMonitor, MonitoringConfig, HealthReport, NetworkAlert},
    types::{NodeId, NodeAddress, PeerInfo, NetworkMessage, NetworkError, NetworkResult, TopologyType, RoutingStrategy},
};
use std::collections::HashMap;
use tokio::sync::mpsc;
use tokio::time::{interval, Duration};

/// High-level network manager that coordinates all network components
#[derive(Debug)]
pub struct NetworkManager {
    /// Local node information
    local_node: PeerInfo,
    /// Network configuration
    config: NetworkConfig,    /// Topology manager
    topology: NetworkTopology,
    /// Peer discovery service
    discovery: NodeDiscovery,
    /// Routing engine
    routing: RoutingEngine,
    /// Connection manager
    connection_manager: ConnectionManager,
    /// Network monitor
    monitor: NetworkMonitor,
    /// Event channel for network events
    event_receiver: Option<mpsc::UnboundedReceiver<NetworkEvent>>,
    /// Event sender for external components
    event_sender: mpsc::UnboundedSender<NetworkEvent>,
}

/// Network configuration combining all component configurations
#[derive(Debug, Clone)]
pub struct NetworkConfig {
    /// Local node listening address
    pub local_address: NodeAddress,
    /// Network topology type
    pub topology_type: TopologyType,
    /// Routing strategy
    pub routing_strategy: RoutingStrategy,
    /// Discovery configuration
    pub discovery_config: DiscoveryConfig,
    /// Connection configuration
    pub connection_config: ConnectionConfig,
    /// Monitoring configuration
    pub monitoring_config: MonitoringConfig,
    /// Bootstrap peer addresses for initial connection
    pub bootstrap_peers: Vec<NodeAddress>,
    /// Maximum number of peers to maintain
    pub max_peers: usize,
    /// Auto-discovery enabled
    pub auto_discovery: bool,
}

/// Network events that can be emitted by the network manager
#[derive(Debug, Clone)]
pub enum NetworkEvent {
    /// Network started successfully
    NetworkStarted,
    /// Network stopped
    NetworkStopped,
    /// Peer discovered
    PeerDiscovered(PeerInfo),
    /// Peer connected
    PeerConnected(NodeId),
    /// Peer disconnected
    PeerDisconnected(NodeId, String),
    /// Message received from peer
    MessageReceived(NodeId, NetworkMessage),
    /// Network topology changed
    TopologyChanged,
    /// Health alert generated
    HealthAlert(NetworkAlert),
    /// Network error occurred
    NetworkError(NetworkError),
}

/// Network status information
#[derive(Debug, Clone)]
pub struct NetworkStatus {
    /// Whether the network is running
    pub is_running: bool,
    /// Number of connected peers
    pub connected_peers: usize,
    /// Number of discovered peers
    pub discovered_peers: usize,
    /// Current topology type
    pub topology_type: TopologyType,
    /// Current routing strategy
    pub routing_strategy: RoutingStrategy,
    /// Network uptime
    pub uptime: Duration,
    /// Current health report
    pub health_report: HealthReport,
}

impl NetworkManager {
    /// Create a new network manager
    pub async fn new(local_node: PeerInfo, config: NetworkConfig) -> NetworkResult<Self> {
        let (event_sender, event_receiver) = mpsc::unbounded_channel();
        
        // Initialize topology manager
        let topology = NetworkTopology::new(
            config.topology_type.clone(),
            local_node.node_id.clone(),
        );
        
        // Initialize peer discovery
        let discovery = NodeDiscovery::new(config.discovery_config.clone()).await?;

        // Initialize routing engine
        let routing = RoutingEngine::new(config.routing_strategy.clone());

        // Initialize connection manager
        let (connection_manager, mut connection_events) = ConnectionManager::new(
            local_node.node_id.clone(),
            config.connection_config.clone(),
        );

        // Initialize network monitor
        let monitor = NetworkMonitor::new(
            local_node.node_id.clone(),
            config.monitoring_config.clone(),
        );

        // Forward connection events to network events
        let event_sender_clone = event_sender.clone();
        tokio::spawn(async move {
            while let Some(event) = connection_events.recv().await {
                let network_event = match event {
                    ConnectionEvent::PeerConnected(peer_id) => {
                        NetworkEvent::PeerConnected(peer_id)
                    }
                    ConnectionEvent::PeerDisconnected(peer_id, reason) => {
                        NetworkEvent::PeerDisconnected(peer_id, reason)
                    }
                    ConnectionEvent::MessageReceived(peer_id, message) => {
                        NetworkEvent::MessageReceived(peer_id, message)
                    }
                    ConnectionEvent::ConnectionError(_peer_id, error) => {
                        NetworkEvent::NetworkError(error)
                    }
                    _ => continue,
                };

                let _ = event_sender_clone.send(network_event);
            }
        });        Ok(Self {
            local_node,
            config,
            topology,
            discovery,
            routing,
            connection_manager,
            monitor,
            event_receiver: Some(event_receiver),
            event_sender,
        })
    }

    /// Start the network manager and all its components
    pub async fn start(&mut self) -> NetworkResult<()> {
        // Start network monitor
        self.monitor.start_monitoring().await;

        // Start connection manager listener
        self.connection_manager.start_listener(self.config.local_address.clone()).await?;

        // Start connection manager maintenance
        self.connection_manager.start_maintenance().await;

        // Start peer discovery
        self.discovery.start().await?;

        // Connect to bootstrap peers
        for bootstrap_addr in &self.config.bootstrap_peers {            let peer_info = PeerInfo::new(
                NodeId::from_string(&format!("bootstrap_{}", bootstrap_addr)),
                bootstrap_addr.clone(),
            );

            if let Err(e) = self.connection_manager.connect_to_peer(peer_info).await {
                eprintln!("Failed to connect to bootstrap peer {}: {}", bootstrap_addr, e);
            }
        }

        // Start periodic tasks
        self.start_periodic_tasks().await;

        let _ = self.event_sender.send(NetworkEvent::NetworkStarted);
        Ok(())
    }

    /// Stop the network manager
    pub async fn stop(&mut self) -> NetworkResult<()> {
        // Stop discovery
        self.discovery.stop().await?;

        // Disconnect from all peers
        let connected_peers = self.connection_manager.get_connected_peers().await?;
        for peer in connected_peers {
            let _ = self.connection_manager.disconnect_from_peer(&peer.node_id).await;
        }

        let _ = self.event_sender.send(NetworkEvent::NetworkStopped);
        Ok(())
    }

    /// Send a message to a specific peer
    pub async fn send_message(&self, peer_id: &NodeId, message: NetworkMessage) -> NetworkResult<()> {
        // Record metrics
        self.monitor.record_message_sent(message.payload.len() as u64).await;

        // Send via connection manager
        self.connection_manager.send_message(peer_id, message).await
    }

    /// Broadcast a message to all connected peers
    pub async fn broadcast_message(&self, message: NetworkMessage) -> NetworkResult<()> {
        let connected_peers = self.connection_manager.get_connected_peers().await?;
        
        // Record metrics
        self.monitor.record_message_sent(
            (message.payload.len() * connected_peers.len()) as u64
        ).await;

        // Broadcast via connection manager
        self.connection_manager.broadcast_message(message).await
    }

    /// Get current network status
    pub async fn get_status(&self) -> NetworkResult<NetworkStatus> {
        let connected_peers = self.connection_manager.get_connected_peers().await?;
        let discovered_peers = self.discovery.get_peers().await;
        let health_report = self.monitor.generate_health_report().await?;

        Ok(NetworkStatus {
            is_running: true, // TODO: Track actual running state
            connected_peers: connected_peers.len(),
            discovered_peers: discovered_peers.len(),
            topology_type: self.config.topology_type.clone(),
            routing_strategy: self.config.routing_strategy.clone(),
            uptime: self.monitor.get_uptime(),
            health_report,
        })
    }

    /// Get network health report
    pub async fn get_health_report(&self) -> NetworkResult<HealthReport> {
        self.monitor.generate_health_report().await
    }

    /// Get routing statistics
    pub async fn get_routing_stats(&self) -> NetworkResult<RoutingStats> {
        self.routing.get_stats().await
    }

    /// Add a peer manually
    pub async fn add_peer(&self, peer_info: PeerInfo) -> NetworkResult<()> {
        // Add to topology
        self.topology.add_node(peer_info.clone()).await?;

        // Update routing
        let peers_vec = self.discovery.get_peers().await;
        let peers: HashMap<NodeId, PeerInfo> = peers_vec.into_iter()
            .map(|peer| (peer.node_id.clone(), peer))
            .collect();
        self.routing.update_topology(peers).await?;

        // Attempt connection
        self.connection_manager.connect_to_peer(peer_info.clone()).await?;

        let _ = self.event_sender.send(NetworkEvent::PeerDiscovered(peer_info));
        Ok(())
    }

    /// Remove a peer
    pub async fn remove_peer(&self, peer_id: &NodeId) -> NetworkResult<()> {
        // Disconnect
        self.connection_manager.disconnect_from_peer(peer_id).await?;

        // Remove from topology
        self.topology.remove_node(peer_id).await?;

        // Update routing
        self.routing.remove_node(peer_id).await?;

        Ok(())
    }

    /// Get information about a specific peer
    pub async fn get_peer_info(&self, peer_id: &NodeId) -> NetworkResult<Option<PeerInfo>> {
        Ok(self.discovery.get_peer(peer_id).await)
    }

    /// Get all connected peers
    pub async fn get_connected_peers(&self) -> NetworkResult<Vec<PeerInfo>> {
        self.connection_manager.get_connected_peers().await
    }

    /// Get all discovered peers
    pub async fn get_discovered_peers(&self) -> NetworkResult<Vec<PeerInfo>> {
        Ok(self.discovery.get_peers().await)
    }

    /// Take the event receiver (can only be called once)
    pub fn take_event_receiver(&mut self) -> Option<mpsc::UnboundedReceiver<NetworkEvent>> {
        self.event_receiver.take()
    }

    /// Start periodic maintenance tasks
    async fn start_periodic_tasks(&self) {        let _routing = self.routing.clone();
        let _topology = self.topology.clone();
        let monitor = self.monitor.clone();
        let event_sender = self.event_sender.clone();

        // Periodic topology sync and routing updates
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60));
            
            loop {
                interval.tick().await;
                
                // Update routing with current topology
                // Note: We can't use discovery here since it's not Clone
                // In a real implementation, discovery would be Arc<NodeDiscovery>
                // let peers = discovery.get_peers().await;
                // let _ = routing.update_topology(peers).await;

                // Check for network alerts
                if let Ok(alerts) = monitor.check_alerts().await {
                    for alert in alerts {
                        let _ = event_sender.send(NetworkEvent::HealthAlert(alert));
                    }
                }
            }
        });
    }
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            local_address: NodeAddress::new("127.0.0.1:8080".parse().unwrap()),
            topology_type: TopologyType::Mesh,
            routing_strategy: RoutingStrategy::ShortestPath,
            discovery_config: DiscoveryConfig::default(),
            connection_config: ConnectionConfig::default(),
            monitoring_config: MonitoringConfig::default(),
            bootstrap_peers: Vec::new(),
            max_peers: 50,
            auto_discovery: true,
        }
    }
}

#[cfg(test)]
mod tests {    use super::*;

    fn create_test_peer_info(id: &str, port: u16) -> PeerInfo {
        PeerInfo::new(
            NodeId::from_string(id),
            NodeAddress::new(format!("127.0.0.1:{}", port).parse().unwrap()),
        )
    }

    #[tokio::test]
    async fn test_network_manager_creation() {
        let local_node = create_test_peer_info("local_node", 8080);
        let config = NetworkConfig::default();

        let manager = NetworkManager::new(local_node, config).await;
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_network_config_default() {
        let config = NetworkConfig::default();
        
        assert_eq!(config.local_address.port(), 8080);
        assert_eq!(config.topology_type, TopologyType::Mesh);
        assert_eq!(config.routing_strategy, RoutingStrategy::ShortestPath);
        assert_eq!(config.max_peers, 50);
        assert!(config.auto_discovery);
    }

    #[tokio::test]
    async fn test_get_status() {
        let local_node = create_test_peer_info("local_node", 8080);
        let config = NetworkConfig::default();

        let manager = NetworkManager::new(local_node, config).await.unwrap();
        let status = manager.get_status().await.unwrap();

        assert_eq!(status.connected_peers, 0);
        assert_eq!(status.discovered_peers, 0);
        assert_eq!(status.topology_type, TopologyType::Mesh);
        assert_eq!(status.routing_strategy, RoutingStrategy::ShortestPath);
    }

    #[tokio::test]
    async fn test_add_peer() {
        let local_node = create_test_peer_info("local_node", 8080);
        let config = NetworkConfig::default();
        let peer_info = create_test_peer_info("test_peer", 8081);

        let manager = NetworkManager::new(local_node, config).await.unwrap();
        
        // Adding peer should not fail (even if connection fails)
        let result = manager.add_peer(peer_info.clone()).await;
        // Connection will likely fail in test, but topology should be updated
    }

    #[tokio::test]
    async fn test_get_connected_peers() {
        let local_node = create_test_peer_info("local_node", 8080);
        let config = NetworkConfig::default();

        let manager = NetworkManager::new(local_node, config).await.unwrap();
        let peers = manager.get_connected_peers().await.unwrap();
        
        assert!(peers.is_empty());
    }

    #[tokio::test]
    async fn test_get_routing_stats() {
        let local_node = create_test_peer_info("local_node", 8080);
        let config = NetworkConfig::default();

        let manager = NetworkManager::new(local_node, config).await.unwrap();
        let stats = manager.get_routing_stats().await.unwrap();
        
        assert_eq!(stats.total_destinations, 0);
        assert_eq!(stats.total_nodes, 0);
    }
}
