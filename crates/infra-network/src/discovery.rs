//! Peer discovery mechanisms for dynamic network formation

use crate::types::{NetworkError, NetworkResult, NodeAddress, NodeId, PeerInfo, PeerStatus};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::Arc,
    time::{Duration, SystemTime},
};
use tokio::{
    net::{TcpListener, TcpStream, UdpSocket},
    sync::RwLock,
    time::timeout,
};

/// Discovery protocol types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiscoveryProtocol {
    /// Multicast UDP discovery
    Multicast { 
        /// Multicast address to use
        address: SocketAddr 
    },
    /// Bootstrap from known peers
    Bootstrap { 
        /// List of known bootstrap peers
        bootstrap_peers: Vec<SocketAddr> 
    },
    /// mDNS service discovery
    Mdns { 
        /// Service name for mDNS
        service_name: String 
    },
    /// DHT-based discovery
    Dht { 
        /// Bootstrap nodes for DHT
        bootstrap_nodes: Vec<SocketAddr> 
    },
    /// Static configuration
    Static { 
        /// Static list of peers
        peers: Vec<NodeAddress> 
    },
}

/// Discovery configuration
#[derive(Debug, Clone)]
pub struct DiscoveryConfig {
    /// Node's own information
    pub local_node: PeerInfo,
    /// Discovery protocol to use
    pub protocol: DiscoveryProtocol,
    /// Discovery interval
    pub discovery_interval: Duration,
    /// Peer timeout duration
    pub peer_timeout: Duration,
    /// Maximum number of peers to maintain
    pub max_peers: usize,
    /// Enable automatic peer cleanup
    pub auto_cleanup: bool,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        let node_id = NodeId::new();
        let address = NodeAddress::new("127.0.0.1:0".parse().unwrap());
        
        Self {
            local_node: PeerInfo::new(node_id, address),
            protocol: DiscoveryProtocol::Multicast {
                address: "224.0.0.251:5353".parse().unwrap(),
            },
            discovery_interval: Duration::from_secs(30),
            peer_timeout: Duration::from_secs(120),
            max_peers: 100,
            auto_cleanup: true,
        }
    }
}

/// Discovery message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiscoveryMessage {
    /// Announce presence
    Announce {
        /// Node identifier
        node_id: NodeId,
        /// Node address
        address: NodeAddress,
        /// Node capabilities
        capabilities: Vec<String>,
    },
    /// Request peer list
    PeerRequest { 
        /// Requesting node ID
        node_id: NodeId 
    },
    /// Response with peer list
    PeerResponse {
        /// Responding node ID
        node_id: NodeId,
        /// List of known peers
        peers: Vec<PeerInfo>,
    },
    /// Goodbye message
    Goodbye { 
        /// Departing node ID
        node_id: NodeId 
    },
    /// Ping for connectivity check
    Ping { 
        /// Pinging node ID
        node_id: NodeId, 
        /// Timestamp for RTT measurement
        timestamp: u64 
    },
    /// Pong response
    Pong { 
        /// Responding node ID
        node_id: NodeId, 
        /// Original timestamp from ping
        timestamp: u64 
    },
}

/// Node discovery manager
#[derive(Debug)]
pub struct NodeDiscovery {
    /// Discovery configuration
    config: DiscoveryConfig,
    /// Known peers
    peers: Arc<RwLock<HashMap<NodeId, PeerInfo>>>,
    /// UDP socket for discovery
    discovery_socket: Option<Arc<UdpSocket>>,
    /// TCP listener for incoming connections
    tcp_listener: Option<Arc<TcpListener>>,
    /// Shutdown flag
    shutdown: Arc<RwLock<bool>>,
}

impl NodeDiscovery {
    /// Create a new node discovery instance
    pub async fn new(config: DiscoveryConfig) -> NetworkResult<Self> {
        let peers = Arc::new(RwLock::new(HashMap::new()));
        let shutdown = Arc::new(RwLock::new(false));

        // Setup UDP socket for discovery
        let discovery_socket = match &config.protocol {
            DiscoveryProtocol::Multicast { address: _ } => {
                let socket = UdpSocket::bind("0.0.0.0:0").await?;
                socket.set_broadcast(true)?;
                Some(Arc::new(socket))
            }
            _ => None,
        };

        // Setup TCP listener for incoming connections
        let tcp_listener = {
            let listener = TcpListener::bind(&config.local_node.address.socket_addr).await?;
            Some(Arc::new(listener))
        };

        Ok(Self {
            config,
            peers,
            discovery_socket,
            tcp_listener,
            shutdown,
        })
    }

    /// Start the discovery process
    pub async fn start(&self) -> NetworkResult<()> {
        // Start discovery protocol
        match &self.config.protocol {
            DiscoveryProtocol::Multicast { .. } => {
                self.start_multicast_discovery().await?;
            }
            DiscoveryProtocol::Bootstrap { bootstrap_peers } => {
                self.bootstrap_from_peers(bootstrap_peers).await?;
            }
            DiscoveryProtocol::Static { peers } => {
                self.load_static_peers(peers).await?;
            }
            _ => {
                return Err(NetworkError::Configuration(
                    "Discovery protocol not yet implemented".to_string(),
                ));
            }
        }

        // Start periodic discovery
        self.start_periodic_discovery().await;

        // Start cleanup task if enabled
        if self.config.auto_cleanup {
            self.start_cleanup_task().await;
        }

        Ok(())
    }

    /// Stop the discovery process
    pub async fn stop(&self) -> NetworkResult<()> {
        *self.shutdown.write().await = true;
        
        // Send goodbye message
        if let Some(socket) = &self.discovery_socket {
            let goodbye = DiscoveryMessage::Goodbye {
                node_id: self.config.local_node.node_id,
            };
            
            if let Ok(message_bytes) = serde_json::to_vec(&goodbye) {
                if let DiscoveryProtocol::Multicast { address } = &self.config.protocol {
                    let _ = socket.send_to(&message_bytes, address).await;
                }
            }
        }

        Ok(())
    }

    /// Get current peer list
    pub async fn get_peers(&self) -> Vec<PeerInfo> {
        self.peers.read().await.values().cloned().collect()
    }

    /// Get a specific peer by ID
    pub async fn get_peer(&self, node_id: &NodeId) -> Option<PeerInfo> {
        self.peers.read().await.get(node_id).cloned()
    }

    /// Add a peer manually
    pub async fn add_peer(&self, peer: PeerInfo) -> NetworkResult<()> {
        let mut peers = self.peers.write().await;
        peers.insert(peer.node_id, peer);
        Ok(())
    }

    /// Remove a peer
    pub async fn remove_peer(&self, node_id: &NodeId) -> NetworkResult<()> {
        let mut peers = self.peers.write().await;
        peers.remove(node_id);
        Ok(())
    }

    /// Check connectivity to a peer
    pub async fn ping_peer(&self, node_id: &NodeId) -> NetworkResult<Duration> {
        let peer = self.get_peer(node_id).await
            .ok_or_else(|| NetworkError::PeerNotFound(*node_id))?;

        let start_time = std::time::Instant::now();
        
        // Try to establish TCP connection
        let connection_result = timeout(
            Duration::from_secs(5),
            TcpStream::connect(&peer.address.socket_addr)
        ).await;

        match connection_result {
            Ok(Ok(_stream)) => {
                let duration = start_time.elapsed();
                
                // Update peer info
                let mut peers = self.peers.write().await;
                if let Some(peer_info) = peers.get_mut(node_id) {
                    peer_info.status = PeerStatus::Connected;
                    peer_info.latency = Some(duration);
                    peer_info.mark_seen();
                    peer_info.update_reliability(true);
                }
                
                Ok(duration)
            }
            _ => {
                // Update peer as failed
                let mut peers = self.peers.write().await;
                if let Some(peer_info) = peers.get_mut(node_id) {
                    peer_info.status = PeerStatus::Failed;
                    peer_info.update_reliability(false);
                }
                
                Err(NetworkError::ConnectionFailed(
                    format!("Failed to ping peer {}", node_id)
                ))
            }
        }
    }

    /// Start multicast discovery
    async fn start_multicast_discovery(&self) -> NetworkResult<()> {
        if let (Some(socket), DiscoveryProtocol::Multicast { address }) = 
            (&self.discovery_socket, &self.config.protocol) {
            
            // Send announce message
            let announce = DiscoveryMessage::Announce {
                node_id: self.config.local_node.node_id,
                address: self.config.local_node.address.clone(),
                capabilities: self.config.local_node.address.capabilities.clone(),
            };

            let message_bytes = serde_json::to_vec(&announce)?;
            socket.send_to(&message_bytes, address).await?;

            // Start listening for responses
            let socket_clone = socket.clone();
            let peers_clone = self.peers.clone();
            let shutdown_clone = self.shutdown.clone();
            let local_node_id = self.config.local_node.node_id;

            tokio::spawn(async move {
                let mut buffer = [0u8; 1024];
                
                while !*shutdown_clone.read().await {
                    if let Ok((len, _addr)) = socket_clone.recv_from(&mut buffer).await {
                        if let Ok(message) = serde_json::from_slice::<DiscoveryMessage>(&buffer[..len]) {
                            Self::handle_discovery_message(
                                message,
                                &peers_clone,
                                local_node_id,
                            ).await;
                        }
                    }
                }
            });
        }

        Ok(())
    }

    /// Bootstrap from known peers
    async fn bootstrap_from_peers(&self, bootstrap_peers: &[SocketAddr]) -> NetworkResult<()> {
        for &peer_addr in bootstrap_peers {
            // Try to connect and get peer info
            if let Ok(stream) = timeout(
                Duration::from_secs(5),
                TcpStream::connect(peer_addr)
            ).await {
                if stream.is_ok() {
                    // Create peer info (in real implementation, would exchange handshake)
                    let peer_info = PeerInfo::new(
                        NodeId::new(), // Would get real ID from handshake
                        NodeAddress::new(peer_addr),
                    );
                    
                    self.add_peer(peer_info).await?;
                }
            }
        }

        Ok(())
    }

    /// Load static peers
    async fn load_static_peers(&self, static_peers: &[NodeAddress]) -> NetworkResult<()> {
        for address in static_peers {
            let peer_info = PeerInfo::new(NodeId::new(), address.clone());
            self.add_peer(peer_info).await?;
        }

        Ok(())
    }

    /// Start periodic discovery announcements
    async fn start_periodic_discovery(&self) {
        let socket = self.discovery_socket.clone();
        let protocol = self.config.protocol.clone();
        let local_node = self.config.local_node.clone();
        let interval = self.config.discovery_interval;
        let shutdown = self.shutdown.clone();

        tokio::spawn(async move {
            let mut discovery_interval = tokio::time::interval(interval);
            
            while !*shutdown.read().await {
                discovery_interval.tick().await;
                
                if let (Some(socket), DiscoveryProtocol::Multicast { address }) = 
                    (&socket, &protocol) {
                    
                    let announce = DiscoveryMessage::Announce {
                        node_id: local_node.node_id,
                        address: local_node.address.clone(),
                        capabilities: local_node.address.capabilities.clone(),
                    };

                    if let Ok(message_bytes) = serde_json::to_vec(&announce) {
                        let _ = socket.send_to(&message_bytes, &address).await;
                    }
                }
            }
        });
    }

    /// Start cleanup task for stale peers
    async fn start_cleanup_task(&self) {
        let peers = self.peers.clone();
        let timeout_duration = self.config.peer_timeout;
        let shutdown = self.shutdown.clone();

        tokio::spawn(async move {
            let mut cleanup_interval = tokio::time::interval(Duration::from_secs(60));
            
            while !*shutdown.read().await {
                cleanup_interval.tick().await;
                
                let now = SystemTime::now();
                let mut peers_guard = peers.write().await;
                
                // Remove stale peers
                peers_guard.retain(|_, peer| {
                    if let Ok(elapsed) = now.duration_since(peer.last_seen) {
                        elapsed < timeout_duration
                    } else {
                        false
                    }
                });
            }
        });
    }

    /// Handle incoming discovery messages
    async fn handle_discovery_message(
        message: DiscoveryMessage,
        peers: &Arc<RwLock<HashMap<NodeId, PeerInfo>>>,
        local_node_id: NodeId,
    ) {
        match message {
            DiscoveryMessage::Announce { node_id, address, capabilities: _ } => {
                if node_id != local_node_id {
                    let mut peer_info = PeerInfo::new(node_id, address);
                    peer_info.status = PeerStatus::Connected;
                    peer_info.mark_seen();
                    
                    peers.write().await.insert(node_id, peer_info);
                }
            }
            DiscoveryMessage::Goodbye { node_id } => {
                peers.write().await.remove(&node_id);
            }
            DiscoveryMessage::Ping { node_id, timestamp: _ } => {
                if node_id != local_node_id {
                    // Would send pong response in real implementation
                }
            }
            _ => {
                // Handle other message types
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_discovery_creation() {
        let config = DiscoveryConfig::default();
        let discovery = NodeDiscovery::new(config).await;
        assert!(discovery.is_ok());
    }

    #[tokio::test]
    async fn test_peer_management() {
        let config = DiscoveryConfig::default();
        let discovery = NodeDiscovery::new(config).await.unwrap();

        let peer = PeerInfo::new(
            NodeId::new(),
            NodeAddress::new("127.0.0.1:8080".parse().unwrap()),
        );
        let peer_id = peer.node_id;

        // Add peer
        discovery.add_peer(peer).await.unwrap();
        assert!(discovery.get_peer(&peer_id).await.is_some());

        // Remove peer
        discovery.remove_peer(&peer_id).await.unwrap();
        assert!(discovery.get_peer(&peer_id).await.is_none());
    }

    #[test]
    fn test_discovery_message_serialization() {
        let message = DiscoveryMessage::Announce {
            node_id: NodeId::new(),
            address: NodeAddress::new("127.0.0.1:8080".parse().unwrap()),
            capabilities: vec!["test".to_string()],
        };

        let serialized = serde_json::to_vec(&message).unwrap();
        let deserialized: DiscoveryMessage = serde_json::from_slice(&serialized).unwrap();

        match (message, deserialized) {
            (DiscoveryMessage::Announce { node_id: id1, .. }, 
             DiscoveryMessage::Announce { node_id: id2, .. }) => {
                assert_eq!(id1, id2);
            }
            _ => panic!("Deserialization failed"),
        }
    }
}
