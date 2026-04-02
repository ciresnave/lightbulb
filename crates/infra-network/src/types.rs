//! Core types and identifiers for network infrastructure

use serde::{Deserialize, Serialize};
use std::{
    fmt,
    net::SocketAddr,
    time::{Duration, SystemTime},
};
use uuid::Uuid;

/// Unique identifier for a network node
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub Uuid);

impl NodeId {
    /// Create a new random node ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
      /// Create a node ID from a string (deterministic)
    pub fn from_string(s: &str) -> Self {
        // Create a deterministic UUID from string using a hash
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        let hash = hasher.finish();
        
        // Convert hash to UUID bytes
        let bytes = [
            (hash >> 56) as u8, (hash >> 48) as u8, (hash >> 40) as u8, (hash >> 32) as u8,
            (hash >> 24) as u8, (hash >> 16) as u8, (hash >> 8) as u8, hash as u8,
            // Padding with zeros for a simple deterministic approach
            0, 0, 0, 0, 0, 0, 0, 0
        ];
        
        Self(Uuid::from_bytes(bytes))
    }

    /// Create a node ID from a UUID
    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }

    /// Get the underlying UUID
    pub fn as_uuid(&self) -> Uuid {
        self.0
    }
}

impl Default for NodeId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<Uuid> for NodeId {
    fn from(uuid: Uuid) -> Self {
        Self(uuid)
    }
}

/// Network address information for a node
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeAddress {
    /// Socket address for direct communication
    pub socket_addr: SocketAddr,
    /// Optional public key for secure communication
    pub public_key: Option<Vec<u8>>,
    /// Protocol capabilities
    pub capabilities: Vec<String>,
}

impl NodeAddress {
    /// Create a new node address
    pub fn new(socket_addr: SocketAddr) -> Self {
        Self {
            socket_addr,
            public_key: None,
            capabilities: vec!["basic".to_string()],
        }
    }

    /// Create a new node address from host and port
    pub fn from_host_port(host: &str, port: u16) -> Result<Self, std::net::AddrParseError> {
        let socket_addr = format!("{}:{}", host, port).parse()?;
        Ok(Self::new(socket_addr))
    }

    /// Get the host as a string
    pub fn host(&self) -> String {
        self.socket_addr.ip().to_string()
    }

    /// Get the port
    pub fn port(&self) -> u16 {
        self.socket_addr.port()
    }

    /// Add a capability to this node
    pub fn with_capability(mut self, capability: String) -> Self {
        self.capabilities.push(capability);
        self
    }

    /// Add a public key for secure communication
    pub fn with_public_key(mut self, public_key: Vec<u8>) -> Self {
        self.public_key = Some(public_key);
        self
    }

    /// Check if this node supports a capability
    pub fn supports_capability(&self, capability: &str) -> bool {
        self.capabilities.contains(&capability.to_string())
    }
}

impl fmt::Display for NodeAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.socket_addr)
    }
}

/// Information about a peer node in the network
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PeerInfo {
    /// Unique node identifier
    pub node_id: NodeId,
    /// Network address information
    pub address: NodeAddress,
    /// Connection status
    pub status: PeerStatus,
    /// Last seen timestamp
    pub last_seen: SystemTime,
    /// Network latency metrics
    pub latency: Option<Duration>,
    /// Reliability score (0.0 to 1.0)
    pub reliability: f64,
}

impl PeerInfo {
    /// Create new peer info
    pub fn new(node_id: NodeId, address: NodeAddress) -> Self {
        Self {
            node_id,
            address,
            status: PeerStatus::Disconnected,
            last_seen: SystemTime::now(),
            latency: None,
            reliability: 1.0,
        }
    }

    /// Check if peer is currently reachable
    pub fn is_reachable(&self) -> bool {
        matches!(self.status, PeerStatus::Connected | PeerStatus::Verified)
    }

    /// Update peer reliability score
    pub fn update_reliability(&mut self, success: bool) {
        if success {
            self.reliability = (self.reliability + 0.1).min(1.0);
        } else {
            self.reliability = (self.reliability - 0.2).max(0.0);
        }
    }

    /// Mark peer as seen now
    pub fn mark_seen(&mut self) {
        self.last_seen = SystemTime::now();
    }
}

/// Connection status of a peer
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PeerStatus {
    /// Not connected
    Disconnected,
    /// Peer has been discovered
    Discovered,
    /// Connection in progress
    Connecting,
    /// Connected but not verified
    Connected,
    /// Connected and identity verified
    Verified,
    /// Connection failed
    Failed,
    /// Temporarily unavailable
    Unavailable,
}

/// Network topology types supported
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopologyType {
    /// Mesh topology - all nodes connected to all others
    Mesh,    /// Star topology - central hub with spokes
    Star { 
        /// The central hub node
        hub_node: NodeId 
    },
    /// Ring topology - nodes connected in a circle
    Ring,
    /// Tree topology - hierarchical structure
    Tree { 
        /// The root node of the tree
        root_node: NodeId 
    },
    /// Custom topology
    Custom,
}

/// Routing strategy for message delivery
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoutingStrategy {
    /// Direct routing to destination
    Direct,
    /// Shortest path routing
    ShortestPath,
    /// Flood routing to all nodes
    Flood,
    /// Random walk routing
    RandomWalk,
    /// Load-balanced routing
    LoadBalanced,
}

/// Network message for inter-node communication
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NetworkMessage {
    /// Message identifier
    pub id: Uuid,
    /// Source node
    pub source: NodeId,
    /// Destination node (None for broadcast)
    pub destination: Option<NodeId>,
    /// Message type
    pub message_type: String,
    /// Message payload
    pub payload: Vec<u8>,
    /// Time-to-live for routing
    pub ttl: u8,
    /// Creation timestamp
    pub timestamp: SystemTime,
}

impl NetworkMessage {
    /// Create a new network message
    pub fn new(
        source: NodeId,
        destination: Option<NodeId>,
        message_type: String,
        payload: Vec<u8>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            source,
            destination,
            message_type,
            payload,
            ttl: 16, // Default TTL
            timestamp: SystemTime::now(),
        }
    }

    /// Check if message has expired based on TTL
    pub fn is_expired(&self) -> bool {
        self.ttl == 0
    }

    /// Decrement TTL and return if message is still valid
    pub fn decrement_ttl(&mut self) -> bool {
        if self.ttl > 0 {
            self.ttl -= 1;
            true
        } else {
            false
        }
    }

    /// Get message age
    pub fn age(&self) -> Option<Duration> {
        SystemTime::now().duration_since(self.timestamp).ok()
    }
}

/// Network statistics and metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NetworkStats {
    /// Total nodes in network
    pub total_nodes: usize,
    /// Connected nodes
    pub connected_nodes: usize,
    /// Total messages sent
    pub messages_sent: u64,
    /// Total messages received
    pub messages_received: u64,
    /// Failed message deliveries
    pub failed_deliveries: u64,
    /// Average network latency
    pub average_latency: Option<Duration>,
    /// Network uptime
    pub uptime: Duration,
    /// Bandwidth utilization
    pub bandwidth_utilization: f64,
}

impl NetworkStats {
    /// Calculate message success rate
    pub fn success_rate(&self) -> f64 {
        if self.messages_sent == 0 {
            0.0
        } else {
            (self.messages_sent - self.failed_deliveries) as f64 / self.messages_sent as f64
        }
    }

    /// Calculate network connectivity ratio
    pub fn connectivity_ratio(&self) -> f64 {
        if self.total_nodes == 0 {
            0.0
        } else {
            self.connected_nodes as f64 / self.total_nodes as f64
        }
    }
}

/// Error types for network operations
#[derive(Debug, Clone, thiserror::Error)]
pub enum NetworkError {
    /// Connection failed
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    /// Peer not found
    #[error("Peer not found: {0}")]
    PeerNotFound(NodeId),

    /// Route not available
    #[error("No route to destination: {0}")]
    NoRoute(NodeId),    /// Message too large
    #[error("Message exceeds maximum size: {size} > {max}")]
    MessageTooLarge { 
        /// Actual message size
        size: usize, 
        /// Maximum allowed size
        max: usize 
    },

    /// Network timeout
    #[error("Network operation timed out")]
    Timeout,

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),    /// IO error
    #[error("IO error: {0}")]
    Io(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// Lock error
    #[error("Lock error")]
    LockError,

    /// Route not found
    #[error("Route not found: {0}")]
    RouteNotFound(String),
}

impl From<std::io::Error> for NetworkError {
    fn from(err: std::io::Error) -> Self {
        NetworkError::Io(err.to_string())
    }
}

impl From<serde_json::Error> for NetworkError {
    fn from(err: serde_json::Error) -> Self {
        NetworkError::Serialization(err.to_string())
    }
}

/// Result type for network operations
pub type NetworkResult<T> = Result<T, NetworkError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_id_creation() {
        let id1 = NodeId::new();
        let id2 = NodeId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_peer_info_reliability() {
        let mut peer = PeerInfo::new(
            NodeId::new(),
            NodeAddress::new("127.0.0.1:8080".parse().unwrap()),
        );

        assert_eq!(peer.reliability, 1.0);

        peer.update_reliability(false);
        assert_eq!(peer.reliability, 0.8);

        peer.update_reliability(true);
        assert_eq!(peer.reliability, 0.9);
    }

    #[test]
    fn test_network_message_ttl() {
        let mut msg = NetworkMessage::new(
            NodeId::new(),
            Some(NodeId::new()),
            "test".to_string(),
            vec![1, 2, 3],
        );

        assert_eq!(msg.ttl, 16);
        assert!(!msg.is_expired());

        for _ in 0..16 {
            assert!(msg.decrement_ttl());
        }

        assert!(!msg.decrement_ttl());
        assert!(msg.is_expired());
    }

    #[test]
    fn test_network_stats() {
        let stats = NetworkStats {
            total_nodes: 10,
            connected_nodes: 8,
            messages_sent: 100,
            failed_deliveries: 5,
            ..Default::default()
        };

        assert_eq!(stats.connectivity_ratio(), 0.8);
        assert_eq!(stats.success_rate(), 0.95);
    }
}
