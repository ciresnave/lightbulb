//! # Infrastructure Network
//!
//! Network topology management and routing for DynAniML distributed systems.
//!
//! This crate provides:
//! - Network topology discovery and management  
//! - Routing algorithms for distributed communication
//! - Mesh, star, and ring topology support
//! - Peer discovery and connection management
//! - Network performance monitoring and health tracking
//! - High-level network management orchestration
//!
//! ## Example Usage
//!
//! ```rust
//! use infra_network::{NetworkManager, NetworkConfig, NodeId, NodeAddress, PeerInfo};
//! use infra_network::types::{PeerStatus, TopologyType, RoutingStrategy};
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create local node info
//! let local_node = PeerInfo {
//!     id: NodeId::new("node_1".to_string()),
//!     address: NodeAddress::new("127.0.0.1", 8080),
//!     status: PeerStatus::Connected,
//!     last_seen: std::time::SystemTime::now(),
//! };
//!
//! // Configure network
//! let mut config = NetworkConfig::default();
//! config.topology_type = TopologyType::Mesh;
//! config.routing_strategy = RoutingStrategy::ShortestPath;
//!
//! // Create and start network manager
//! let mut network = NetworkManager::new(local_node, config).await?;
//! network.start().await?;
//!
//! // Get network status
//! let status = network.get_status().await?;
//! println!("Connected peers: {}", status.connected_peers);
//! # Ok(())
//! # }
//! ```

#![deny(missing_docs)]
#![warn(clippy::all)]

pub mod types;
pub mod topology; 
pub mod discovery;
pub mod routing;
pub mod connection;
pub mod monitoring;
pub mod network;

// Re-export main types and components
pub use types::{
    NodeId, NodeAddress, PeerInfo, PeerStatus, TopologyType, RoutingStrategy,
    NetworkMessage, NetworkStats, NetworkError, NetworkResult,
};

pub use topology::{NetworkTopology, TopologyStats};
pub use discovery::{NodeDiscovery, DiscoveryConfig, DiscoveryProtocol};
pub use routing::{RoutingEngine, RoutingTable, RouteInfo, RoutingStats};
pub use connection::{ConnectionManager, ConnectionConfig, ConnectionState, ConnectionStats};
pub use monitoring::{
    NetworkMonitor, MonitoringConfig, NetworkMetrics, PeerHealth, 
    HealthReport, HealthStatus, NetworkAlert, AlertSeverity,
};
pub use network::{NetworkManager, NetworkConfig, NetworkEvent, NetworkStatus};

/// Current version of the Infrastructure Network library
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::SocketAddr;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[tokio::test]
    async fn test_basic_network_setup() -> NetworkResult<()> {
        let socket_addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();
        let address = NodeAddress::new(socket_addr);
        let node_id = NodeId::new();
        
        let local_node = PeerInfo::new(node_id, address);

        let config = NetworkConfig::default();
        let _manager = NetworkManager::new(local_node, config).await?;
        
        Ok(())
    }
}
