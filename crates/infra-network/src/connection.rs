//! Connection management for network peers.

use crate::types::{NodeId, NodeAddress, PeerInfo, PeerStatus, NetworkError, NetworkResult, NetworkMessage};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::{RwLock, mpsc};
use tokio::net::{TcpListener, TcpStream};
use tokio::time::{interval, timeout};

/// Connection manager for handling peer connections
#[derive(Debug)]
pub struct ConnectionManager {
    /// Local node information
    local_node: NodeId,
    /// Active connections to peers
    connections: Arc<RwLock<HashMap<NodeId, Connection>>>,
    /// Connection configuration
    config: ConnectionConfig,
    /// Channel for sending connection events
    event_sender: mpsc::UnboundedSender<ConnectionEvent>,
}

/// Configuration for connection management
#[derive(Debug, Clone)]
pub struct ConnectionConfig {
    /// Maximum number of concurrent connections
    pub max_connections: usize,
    /// Connection timeout duration
    pub connect_timeout: Duration,
    /// Heartbeat interval
    pub heartbeat_interval: Duration,
    /// Connection idle timeout
    pub idle_timeout: Duration,
    /// Maximum number of connection retries
    pub max_retries: u32,
    /// Retry backoff duration
    pub retry_backoff: Duration,
}

/// Represents a connection to a peer
#[derive(Debug)]
pub struct Connection {
    /// Peer information
    peer_info: PeerInfo,
    /// Connection state
    state: ConnectionState,
    /// TCP stream for communication
    stream: Option<TcpStream>,
    /// Connection establishment time
    established_at: SystemTime,
    /// Last activity time
    last_activity: SystemTime,
    /// Number of failed connection attempts
    retry_count: u32,
    /// Channel for sending messages
    message_sender: Option<mpsc::UnboundedSender<NetworkMessage>>,
}

/// Connection state
#[derive(Debug, Clone, PartialEq)]
pub enum ConnectionState {
    /// Disconnected
    Disconnected,
    /// Connecting in progress
    Connecting,
    /// Connected and active
    Connected,
    /// Connection failed
    Failed(String),
    /// Connection being closed
    Closing,
}

/// Connection events
#[derive(Debug, Clone)]
pub enum ConnectionEvent {
    /// Peer connected
    PeerConnected(NodeId),
    /// Peer disconnected
    PeerDisconnected(NodeId, String),
    /// Connection failed
    ConnectionFailed(NodeId, String),
    /// Message received from peer
    MessageReceived(NodeId, NetworkMessage),
    /// Connection error
    ConnectionError(NodeId, NetworkError),
}

/// Connection statistics
#[derive(Debug, Clone)]
pub struct ConnectionStats {
    /// Total number of active connections
    pub active_connections: usize,
    /// Total bytes sent
    pub bytes_sent: u64,
    /// Total bytes received
    pub bytes_received: u64,
    /// Total messages sent
    pub messages_sent: u64,
    /// Total messages received
    pub messages_received: u64,
    /// Connection uptime
    pub uptime: Duration,
}

impl ConnectionManager {
    /// Create a new connection manager
    pub fn new(
        local_node: NodeId,
        config: ConnectionConfig,
    ) -> (Self, mpsc::UnboundedReceiver<ConnectionEvent>) {
        let (event_sender, event_receiver) = mpsc::unbounded_channel();
        
        let manager = Self {
            local_node,
            connections: Arc::new(RwLock::new(HashMap::new())),
            config,
            event_sender,
        };

        (manager, event_receiver)
    }

    /// Start listening for incoming connections
    pub async fn start_listener(&self, address: NodeAddress) -> NetworkResult<()> {
        let listener = TcpListener::bind(format!("{}:{}", address.host(), address.port()))
            .await
            .map_err(|e| NetworkError::ConnectionFailed(format!("Failed to bind listener: {}", e)))?;

        let connections = Arc::clone(&self.connections);
        let event_sender = self.event_sender.clone();
        let config = self.config.clone();

        tokio::spawn(async move {
            loop {
                match listener.accept().await {
                    Ok((stream, addr)) => {
                        let peer_id = NodeId::from_string(&addr.to_string()); // Simplified peer identification
                        
                        if let Err(e) = Self::handle_incoming_connection(
                            stream,
                            peer_id.clone(),
                            Arc::clone(&connections),
                            event_sender.clone(),
                            config.clone(),
                        ).await {
                            let _ = event_sender.send(ConnectionEvent::ConnectionError(
                                peer_id,
                                e,
                            ));
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to accept connection: {}", e);
                    }
                }
            }
        });

        Ok(())
    }

    /// Connect to a peer
    pub async fn connect_to_peer(&self, peer_info: PeerInfo) -> NetworkResult<()> {
        let peer_id = peer_info.node_id.clone();
        
        // Check if already connected
        {
            let connections = self.connections.read().await;
            if let Some(connection) = connections.get(&peer_id) {
                if connection.state == ConnectionState::Connected {
                    return Ok(());
                }
            }
        }

        // Set connection state to connecting
        {
            let mut connections = self.connections.write().await;
            connections.insert(
                peer_id.clone(),
                Connection::new(peer_info.clone(), ConnectionState::Connecting),
            );
        }

        let address = format!("{}:{}", peer_info.address.host(), peer_info.address.port());
        
        match timeout(self.config.connect_timeout, TcpStream::connect(&address)).await {
            Ok(Ok(stream)) => {
                self.handle_successful_connection(peer_id, stream).await?;
                let _ = self.event_sender.send(ConnectionEvent::PeerConnected(peer_id));
                Ok(())
            }
            Ok(Err(e)) => {
                self.handle_connection_failure(peer_id, format!("Connection failed: {}", e)).await;
                Err(NetworkError::ConnectionFailed(format!("Failed to connect to peer: {}", e)))
            }
            Err(_) => {
                self.handle_connection_failure(peer_id, "Connection timeout".to_string()).await;
                Err(NetworkError::ConnectionFailed("Connection timeout".to_string()))
            }
        }
    }

    /// Disconnect from a peer
    pub async fn disconnect_from_peer(&self, peer_id: &NodeId) -> NetworkResult<()> {
        let mut connections = self.connections.write().await;
        
        if let Some(mut connection) = connections.remove(peer_id) {
            connection.state = ConnectionState::Closing;
            
            if let Some(stream) = connection.stream.take() {
                drop(stream); // Close the stream
            }
            
            let _ = self.event_sender.send(ConnectionEvent::PeerDisconnected(
                peer_id.clone(),
                "Disconnected by request".to_string(),
            ));
        }

        Ok(())
    }

    /// Send a message to a peer
    pub async fn send_message(&self, peer_id: &NodeId, message: NetworkMessage) -> NetworkResult<()> {
        let connections = self.connections.read().await;
        
        if let Some(connection) = connections.get(peer_id) {
            if connection.state == ConnectionState::Connected {
                if let Some(sender) = &connection.message_sender {
                    sender.send(message)
                        .map_err(|_| NetworkError::ConnectionFailed("Failed to send message".to_string()))?;
                    return Ok(());
                }
            }
        }

        Err(NetworkError::PeerNotFound(peer_id.clone()))
    }

    /// Broadcast a message to all connected peers
    pub async fn broadcast_message(&self, message: NetworkMessage) -> NetworkResult<()> {
        let connections = self.connections.read().await;
        let mut send_futures = Vec::new();

        for (_peer_id, connection) in connections.iter() {
            if connection.state == ConnectionState::Connected {
                if let Some(sender) = &connection.message_sender {
                    let msg_clone = message.clone();
                    let sender_clone = sender.clone();
                    send_futures.push(async move {
                        sender_clone.send(msg_clone)
                    });
                }
            }
        }

        // Wait for all sends to complete
        for future in send_futures {
            let _ = future.await; // Ignore individual failures
        }

        Ok(())
    }

    /// Get connection statistics
    pub async fn get_stats(&self) -> NetworkResult<ConnectionStats> {
        let connections = self.connections.read().await;
        
        let active_connections = connections
            .values()
            .filter(|c| c.state == ConnectionState::Connected)
            .count();

        // TODO: Implement proper byte and message counting
        Ok(ConnectionStats {
            active_connections,
            bytes_sent: 0,
            bytes_received: 0,
            messages_sent: 0,
            messages_received: 0,
            uptime: SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default(),
        })
    }

    /// Get information about a specific connection
    pub async fn get_connection_info(&self, peer_id: &NodeId) -> NetworkResult<Option<PeerInfo>> {
        let connections = self.connections.read().await;
        Ok(connections.get(peer_id).map(|c| c.peer_info.clone()))
    }

    /// Get all connected peers
    pub async fn get_connected_peers(&self) -> NetworkResult<Vec<PeerInfo>> {
        let connections = self.connections.read().await;
        let peers = connections
            .values()
            .filter(|c| c.state == ConnectionState::Connected)
            .map(|c| c.peer_info.clone())
            .collect();
        Ok(peers)
    }

    /// Start connection maintenance tasks
    pub async fn start_maintenance(&self) {
        let connections = Arc::clone(&self.connections);
        let config = self.config.clone();
        let event_sender = self.event_sender.clone();

        tokio::spawn(async move {
            let mut heartbeat_interval = interval(config.heartbeat_interval);
            
            loop {
                heartbeat_interval.tick().await;
                Self::perform_maintenance(
                    Arc::clone(&connections),
                    &config,
                    event_sender.clone(),
                ).await;
            }
        });
    }

    /// Handle successful connection establishment
    async fn handle_successful_connection(&self, peer_id: NodeId, stream: TcpStream) -> NetworkResult<()> {
        let (message_sender, message_receiver) = mpsc::unbounded_channel();
        
        {
            let mut connections = self.connections.write().await;
            if let Some(connection) = connections.get_mut(&peer_id) {
                connection.state = ConnectionState::Connected;
                connection.stream = Some(stream);
                connection.established_at = SystemTime::now();
                connection.last_activity = SystemTime::now();
                connection.message_sender = Some(message_sender);
            }
        }

        // Start message handler for this connection
        let connections = Arc::clone(&self.connections);
        let event_sender = self.event_sender.clone();
        
        tokio::spawn(async move {
            Self::handle_connection_messages(
                peer_id,
                message_receiver,
                connections,
                event_sender,
            ).await;
        });

        Ok(())
    }

    /// Handle connection failure
    async fn handle_connection_failure(&self, peer_id: NodeId, error: String) {
        let mut connections = self.connections.write().await;
        
        if let Some(connection) = connections.get_mut(&peer_id) {
            connection.state = ConnectionState::Failed(error.clone());
            connection.retry_count += 1;
        }

        let _ = self.event_sender.send(ConnectionEvent::ConnectionFailed(peer_id, error));
    }

    /// Handle incoming connection
    async fn handle_incoming_connection(
        stream: TcpStream,
        peer_id: NodeId,
        connections: Arc<RwLock<HashMap<NodeId, Connection>>>,
        event_sender: mpsc::UnboundedSender<ConnectionEvent>,
        _config: ConnectionConfig,
    ) -> NetworkResult<()> {
        // Create peer info from stream
        let peer_addr = stream.peer_addr()
            .map_err(|e| NetworkError::ConnectionFailed(format!("Failed to get peer address: {}", e)))?;        let peer_info = PeerInfo {
            node_id: peer_id.clone(),
            address: NodeAddress::new(peer_addr),
            status: PeerStatus::Connected,
            last_seen: SystemTime::now(),
            latency: Some(Duration::from_millis(0)),
            reliability: 1.0,
        };

        let (message_sender, message_receiver) = mpsc::unbounded_channel();
        
        {
            let mut conns = connections.write().await;
            conns.insert(
                peer_id.clone(),
                Connection {
                    peer_info,
                    state: ConnectionState::Connected,
                    stream: Some(stream),
                    established_at: SystemTime::now(),
                    last_activity: SystemTime::now(),
                    retry_count: 0,
                    message_sender: Some(message_sender),
                },
            );
        }

        let _ = event_sender.send(ConnectionEvent::PeerConnected(peer_id.clone()));

        // Start message handler
        Self::handle_connection_messages(
            peer_id,
            message_receiver,
            connections,
            event_sender,
        ).await;

        Ok(())
    }    /// Handle messages for a connection
    async fn handle_connection_messages(
        peer_id: NodeId,
        mut message_receiver: mpsc::UnboundedReceiver<NetworkMessage>,
        _connections: Arc<RwLock<HashMap<NodeId, Connection>>>,
        event_sender: mpsc::UnboundedSender<ConnectionEvent>,
    ) {
        while let Some(message) = message_receiver.recv().await {
            // TODO: Implement actual message serialization and sending over TCP
            // For now, just simulate message handling
            let _ = event_sender.send(ConnectionEvent::MessageReceived(peer_id.clone(), message));
        }
    }

    /// Perform periodic connection maintenance
    async fn perform_maintenance(
        connections: Arc<RwLock<HashMap<NodeId, Connection>>>,
        config: &ConnectionConfig,
        event_sender: mpsc::UnboundedSender<ConnectionEvent>,
    ) {
        let mut connections = connections.write().await;
        let now = SystemTime::now();
        let mut to_remove = Vec::new();

        for (peer_id, connection) in connections.iter_mut() {
            // Check for idle connections
            if let Ok(idle_time) = now.duration_since(connection.last_activity) {
                if idle_time > config.idle_timeout {
                    to_remove.push(peer_id.clone());
                    continue;
                }
            }

            // Send heartbeat if connected
            if connection.state == ConnectionState::Connected {
                // TODO: Implement actual heartbeat sending
                connection.last_activity = now;
            }
        }

        // Remove idle connections
        for peer_id in to_remove {
            connections.remove(&peer_id);
            let _ = event_sender.send(ConnectionEvent::PeerDisconnected(
                peer_id,
                "Connection idle timeout".to_string(),
            ));
        }
    }
}

impl Connection {
    /// Create a new connection
    pub fn new(peer_info: PeerInfo, state: ConnectionState) -> Self {
        let now = SystemTime::now();
        Self {
            peer_info,
            state,
            stream: None,
            established_at: now,
            last_activity: now,
            retry_count: 0,
            message_sender: None,
        }
    }

    /// Check if connection is active
    pub fn is_active(&self) -> bool {
        self.state == ConnectionState::Connected
    }

    /// Get connection duration
    pub fn duration(&self) -> Duration {
        SystemTime::now().duration_since(self.established_at).unwrap_or_default()
    }
}

impl Default for ConnectionConfig {
    fn default() -> Self {
        Self {
            max_connections: 100,
            connect_timeout: Duration::from_secs(10),
            heartbeat_interval: Duration::from_secs(30),
            idle_timeout: Duration::from_secs(300), // 5 minutes
            max_retries: 3,
            retry_backoff: Duration::from_secs(5),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::NodeAddress;    fn create_test_peer_info(id: &str, port: u16) -> PeerInfo {
        use std::net::{IpAddr, Ipv4Addr, SocketAddr};
        let socket_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), port);
        
        PeerInfo {
            node_id: NodeId::from_string(id),
            address: NodeAddress::new(socket_addr),
            status: PeerStatus::Connected,
            last_seen: SystemTime::now(),
            latency: Some(Duration::from_millis(10)),
            reliability: 1.0,
        }
    }

    #[tokio::test]
    async fn test_connection_manager_creation() {
        let local_node = NodeId::from_string("test_node");
        let config = ConnectionConfig::default();
        let (manager, _receiver) = ConnectionManager::new(local_node, config);

        let stats = manager.get_stats().await.unwrap();
        assert_eq!(stats.active_connections, 0);
    }    #[tokio::test]
    async fn test_connection_stats() {
        let local_node = NodeId::from_string("test_node");
        let config = ConnectionConfig::default();
        let (manager, _receiver) = ConnectionManager::new(local_node, config);

        let stats = manager.get_stats().await.unwrap();
        assert_eq!(stats.active_connections, 0);
        assert_eq!(stats.bytes_sent, 0);
        assert_eq!(stats.bytes_received, 0);
    }

    #[tokio::test]
    async fn test_get_connected_peers() {
        let local_node = NodeId::from_string("test_node");
        let config = ConnectionConfig::default();
        let (manager, _receiver) = ConnectionManager::new(local_node, config);

        let peers = manager.get_connected_peers().await.unwrap();
        assert!(peers.is_empty());
    }

    #[test]
    fn test_connection_creation() {
        let peer_info = create_test_peer_info("test_peer", 8080);
        let connection = Connection::new(peer_info.clone(), ConnectionState::Connected);

        assert_eq!(connection.peer_info.node_id, peer_info.node_id);
        assert_eq!(connection.state, ConnectionState::Connected);
        assert!(connection.is_active());
    }

    #[test]
    fn test_connection_state() {
        let peer_info = create_test_peer_info("test_peer", 8080);
        
        let connected = Connection::new(peer_info.clone(), ConnectionState::Connected);
        assert!(connected.is_active());

        let disconnected = Connection::new(peer_info.clone(), ConnectionState::Disconnected);
        assert!(!disconnected.is_active());

        let failed = Connection::new(peer_info, ConnectionState::Failed("Test error".to_string()));
        assert!(!failed.is_active());
    }

    #[test]
    fn test_connection_config_default() {
        let config = ConnectionConfig::default();
        
        assert_eq!(config.max_connections, 100);
        assert_eq!(config.connect_timeout, Duration::from_secs(10));
        assert_eq!(config.heartbeat_interval, Duration::from_secs(30));
        assert_eq!(config.idle_timeout, Duration::from_secs(300));
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.retry_backoff, Duration::from_secs(5));
    }

    #[test]
    fn test_connection_duration() {
        let peer_info = create_test_peer_info("test_peer", 8080);
        let connection = Connection::new(peer_info, ConnectionState::Connected);

        let duration = connection.duration();
        assert!(duration.as_millis() < 100); // Should be very recent
    }
}
