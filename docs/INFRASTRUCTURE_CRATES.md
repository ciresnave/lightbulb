# Infrastructure Crates Documentation

## Overview

This document provides detailed information about the three infrastructure crates copied from DynAniML to support Lightbulb's distributed ML architecture (M7: Federated Retrieval, M8: Distributed Training).

---

## 1. `infra-network` - Network Topology Management

### Purpose

Sophisticated network layer providing peer discovery, topology management, and intelligent routing for distributed ML systems.

### Key Capabilities

- **Peer Discovery**: Automatic node discovery using multicast and DHT
- **Topology Management**: Dynamic network topology with real-time health monitoring
- **Intelligent Routing**: Message routing with fallback strategies and load balancing
- **Connection Management**: Connection pooling, automatic reconnection, lifecycle management
- **Network Monitoring**: Real-time metrics collection and network health tracking

### Core Components

#### NetworkManager
Main entry point for network operations.

```rust
use infra_network::{NetworkManager, NodeConfig};
use std::time::Duration;

async fn setup_network() -> Result<NetworkManager, NetworkError> {
    let config = NodeConfig {
        listen_addr: "0.0.0.0:9000".to_string(),
        max_peers: 100,
        reconnect_interval: Duration::from_secs(30),
    };
    
    let mut manager = NetworkManager::new(config)?;
    manager.start().await?;
    
    Ok(manager)
}
```

#### PeerDiscovery
Automatic peer discovery with multiple strategies.

```rust
use infra_network::{PeerDiscovery, DiscoveryMethod};

async fn discover_peers() -> Result<Vec<PeerInfo>, NetworkError> {
    let discovery = PeerDiscovery::builder()
        .with_method(DiscoveryMethod::Multicast)
        .with_method(DiscoveryMethod::DHT)
        .with_bootstrap_nodes(vec![
            "/ip4/192.168.1.100/tcp/9000".parse()?,
        ])
        .build()?;
    
    discovery.start_discovery().await?;
    let peers = discovery.discovered_peers().await;
    Ok(peers)
}
```

#### TopologyManager
Network topology tracking and management.

```rust
use infra_network::{TopologyManager, TopologyEvent};

async fn monitor_topology() {
    let mut topology = TopologyManager::new();
    let mut events = topology.subscribe_events();
    
    while let Some(event) = events.recv().await {
        match event {
            TopologyEvent::PeerJoined(peer_id) => {
                println!("New peer: {}", peer_id);
                topology.update_routes().await;
            }
            TopologyEvent::PeerLeft(peer_id) => {
                println!("Peer left: {}", peer_id);
                topology.update_routes().await;
            }
            TopologyEvent::TopologyChanged => {
                println!("Topology updated");
            }
        }
    }
}
```

### Integration with Lightbulb

#### Use Case 1: Federated Knowledge Retrieval (M7)

```rust
use infra_network::NetworkManager;
use lightbulb::knowledge::FederatedRetrieval;

async fn federated_search(query: &str) -> Result<Vec<SearchResult>, Error> {
    let network = NetworkManager::from_config("network.toml")?;
    let retrieval = FederatedRetrieval::new(network);
    
    // Search across all federated nodes
    let results = retrieval.search_all_peers(query).await?;
    
    // Aggregate and rank results
    let ranked = retrieval.aggregate_results(results)?;
    Ok(ranked)
}
```

#### Use Case 2: Distributed Training Coordination (M8)

```rust
use infra_network::{NetworkManager, BroadcastChannel};

async fn coordinate_training() -> Result<(), Error> {
    let network = NetworkManager::from_config("network.toml")?;
    let channel = BroadcastChannel::new(&network, "training")?;
    
    // Broadcast training parameters
    channel.broadcast(TrainingParams {
        batch_size: 32,
        learning_rate: 0.001,
        epochs: 10,
    }).await?;
    
    // Receive gradient updates from peers
    while let Some(update) = channel.recv().await {
        process_gradient_update(update)?;
    }
    
    Ok(())
}
```

### Dependencies

- `libp2p`: Peer-to-peer networking library
- `tokio`: Async runtime
- `serde`: Serialization
- `tracing`: Structured logging

---

## 2. `infra-storage` - Multi-Backend Storage Layer

### Purpose

Unified storage abstraction supporting multiple backends (RocksDB, SQLite, Sled, in-memory) with replication, transactions, and consistency guarantees.

### Key Capabilities

- **Multi-Backend Support**: RocksDB, SQLite, Sled, in-memory
- **Unified API**: Single interface regardless of backend choice
- **Replication**: Multi-node replication with configurable consistency
- **Transaction Support**: ACID transactions across all backends
- **Batch Operations**: Efficient bulk read/write operations
- **Async/Sync APIs**: Both async and blocking interfaces

### Core Components

#### StorageBackend
Main storage interface with backend abstraction.

```rust
use infra_storage::{StorageBackend, BackendType, StorageConfig};

fn create_storage(backend: BackendType) -> Result<StorageBackend, StorageError> {
    let config = match backend {
        BackendType::RocksDB => StorageConfig::rocksdb("/data/rocks")
            .with_cache_size(100 * 1024 * 1024), // 100MB cache
        BackendType::SQLite => StorageConfig::sqlite("/data/db.sqlite")
            .with_wal_mode(true),
        BackendType::Sled => StorageConfig::sled("/data/sled")
            .with_compression(true),
        BackendType::Memory => StorageConfig::memory(),
    };
    
    StorageBackend::open(config)
}
```

#### Basic Operations

```rust
use infra_storage::StorageBackend;

fn basic_operations(storage: &StorageBackend) -> Result<(), StorageError> {
    // Put
    storage.put(b"user:1", b"Alice")?;
    
    // Get
    let value = storage.get(b"user:1")?;
    assert_eq!(value.as_deref(), Some(b"Alice".as_slice()));
    
    // Delete
    storage.delete(b"user:1")?;
    
    // Batch operations
    let mut batch = storage.batch();
    batch.put(b"user:1", b"Alice");
    batch.put(b"user:2", b"Bob");
    batch.put(b"user:3", b"Charlie");
    storage.write_batch(batch)?;
    
    Ok(())
}
```

#### Transactions

```rust
use infra_storage::{StorageBackend, Transaction};

fn transactional_update(storage: &StorageBackend) -> Result<(), StorageError> {
    let mut txn = storage.begin_transaction()?;
    
    // Read current balance
    let balance: i64 = txn.get(b"account:balance")?
        .and_then(|v| serde_json::from_slice(&v).ok())
        .unwrap_or(0);
    
    // Update balance
    let new_balance = balance + 100;
    txn.put(b"account:balance", &serde_json::to_vec(&new_balance)?)?;
    
    // Commit transaction
    txn.commit()?;
    
    Ok(())
}
```

#### Replication

```rust
use infra_storage::{ReplicatedStorage, ReplicationConfig, ConsistencyLevel};

async fn setup_replication() -> Result<ReplicatedStorage, StorageError> {
    let config = ReplicationConfig {
        replicas: vec![
            "node1.example.com:9000".to_string(),
            "node2.example.com:9000".to_string(),
            "node3.example.com:9000".to_string(),
        ],
        consistency: ConsistencyLevel::Quorum, // Require majority
        replication_factor: 3,
        async_replication: false, // Synchronous replication
    };
    
    let storage = ReplicatedStorage::new(config).await?;
    
    // Operations are automatically replicated
    storage.put(b"key", b"value").await?;
    
    Ok(storage)
}
```

### Integration with Lightbulb

#### Use Case 1: Three-Tier Caching (M5)

```rust
use infra_storage::{StorageBackend, BackendType};

struct ThreeTierCache {
    hot: StorageBackend,   // In-memory
    warm: StorageBackend,  // Sled (fast embedded)
    cold: StorageBackend,  // RocksDB (large persistent)
}

impl ThreeTierCache {
    fn new() -> Result<Self, StorageError> {
        Ok(Self {
            hot: StorageBackend::open(StorageConfig::memory())?,
            warm: StorageBackend::open(StorageConfig::sled("/cache/warm"))?,
            cold: StorageBackend::open(StorageConfig::rocksdb("/cache/cold"))?,
        })
    }
    
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, StorageError> {
        // Check hot tier first
        if let Some(value) = self.hot.get(key)? {
            return Ok(Some(value));
        }
        
        // Check warm tier
        if let Some(value) = self.warm.get(key)? {
            // Promote to hot tier
            self.hot.put(key, &value)?;
            return Ok(Some(value));
        }
        
        // Check cold tier
        if let Some(value) = self.cold.get(key)? {
            // Promote to warm tier
            self.warm.put(key, &value)?;
            return Ok(Some(value));
        }
        
        Ok(None)
    }
}
```

#### Use Case 2: Knowledge Persistence (M7)

```rust
use infra_storage::{StorageBackend, BackendType};
use lightbulb::knowledge::KnowledgeGraph;

fn persist_knowledge_graph(graph: &KnowledgeGraph) -> Result<(), Error> {
    let storage = StorageBackend::open(
        StorageConfig::rocksdb("/data/knowledge")
            .with_cache_size(500 * 1024 * 1024) // 500MB cache
            .with_compression(true)
    )?;
    
    // Store nodes
    for node in graph.nodes() {
        let key = format!("node:{}", node.id);
        let value = serde_json::to_vec(node)?;
        storage.put(key.as_bytes(), &value)?;
    }
    
    // Store edges
    for edge in graph.edges() {
        let key = format!("edge:{}:{}", edge.from, edge.to);
        let value = serde_json::to_vec(edge)?;
        storage.put(key.as_bytes(), &value)?;
    }
    
    Ok(())
}
```

### Backend Comparison

| Backend     | Best For                         | Performance | Durability | Transactions |
| ----------- | -------------------------------- | ----------- | ---------- | ------------ |
| **RocksDB** | Large datasets, high throughput  | ⭐⭐⭐⭐        | ⭐⭐⭐⭐⭐      | ✅            |
| **SQLite**  | Relational data, complex queries | ⭐⭐⭐         | ⭐⭐⭐⭐⭐      | ✅            |
| **Sled**    | Embedded systems, simplicity     | ⭐⭐⭐⭐        | ⭐⭐⭐⭐       | ✅            |
| **Memory**  | Testing, temporary data          | ⭐⭐⭐⭐⭐       | ❌          | ✅            |

### Dependencies

- `rocksdb`: High-performance key-value store
- `rusqlite`: SQLite interface
- `sled`: Embedded database
- `tokio`: Async runtime
- `serde`: Serialization

---

## 3. `infra-consensus` - Distributed Consensus

### Purpose

Raft-based consensus algorithm implementation for distributed coordination, state replication, and fault-tolerant cluster management.

### Key Capabilities

- **Raft Consensus**: Proven consensus algorithm from academia and industry
- **Leader Election**: Automatic leader election with failure detection
- **Log Replication**: Reliable state machine replication across nodes
- **Membership Changes**: Dynamic cluster membership without downtime
- **Snapshot Support**: Log compaction and efficient state snapshots
- **Metrics & Monitoring**: Real-time consensus health metrics

### Core Components

#### RaftNode
Main Raft node implementation.

```rust
use infra_consensus::{RaftNode, RaftConfig, NodeId};
use std::time::Duration;

async fn create_raft_node(node_id: u64, peers: Vec<u64>) 
    -> Result<RaftNode, ConsensusError> 
{
    let config = RaftConfig {
        node_id: NodeId(node_id),
        peers: peers.into_iter().map(NodeId).collect(),
        election_timeout: Duration::from_millis(300),
        heartbeat_interval: Duration::from_millis(100),
        max_entries_per_request: 100,
        snapshot_interval: 1000, // Snapshot every 1000 log entries
    };
    
    let node = RaftNode::new(config)?;
    node.start().await?;
    
    Ok(node)
}
```

#### Proposing Commands

```rust
use infra_consensus::{RaftNode, Command};

async fn propose_state_change(node: &RaftNode, command: Vec<u8>) 
    -> Result<Vec<u8>, ConsensusError> 
{
    // Propose command to cluster
    let result = node.propose(command).await?;
    
    // Wait for commitment (majority agreement)
    let committed_result = result.wait_committed().await?;
    
    Ok(committed_result)
}
```

#### State Machine Integration

```rust
use infra_consensus::{RaftNode, StateMachine, LogEntry};
use std::collections::HashMap;

struct KVStateMachine {
    store: HashMap<Vec<u8>, Vec<u8>>,
}

impl StateMachine for KVStateMachine {
    type Error = std::io::Error;
    
    fn apply(&mut self, entry: LogEntry) -> Result<Vec<u8>, Self::Error> {
        match entry.command {
            Command::Put { key, value } => {
                self.store.insert(key, value.clone());
                Ok(value)
            }
            Command::Get { key } => {
                Ok(self.store.get(&key).cloned().unwrap_or_default())
            }
            Command::Delete { key } => {
                self.store.remove(&key);
                Ok(vec![])
            }
        }
    }
    
    fn snapshot(&self) -> Result<Vec<u8>, Self::Error> {
        // Serialize entire state
        bincode::serialize(&self.store)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
    
    fn restore(&mut self, snapshot: &[u8]) -> Result<(), Self::Error> {
        // Restore from snapshot
        self.store = bincode::deserialize(snapshot)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        Ok(())
    }
}
```

#### Leadership Handling

```rust
use infra_consensus::{RaftNode, LeadershipEvent};

async fn handle_leadership(node: &RaftNode) {
    let mut events = node.subscribe_leadership();
    
    while let Some(event) = events.recv().await {
        match event {
            LeadershipEvent::BecameLeader => {
                println!("This node is now the leader");
                start_leader_tasks().await;
            }
            LeadershipEvent::LostLeadership => {
                println!("This node is no longer the leader");
                stop_leader_tasks().await;
            }
            LeadershipEvent::NewLeader(leader_id) => {
                println!("New leader elected: {:?}", leader_id);
                connect_to_leader(leader_id).await;
            }
        }
    }
}
```

### Integration with Lightbulb

#### Use Case 1: Federated Knowledge Coordination (M7)

```rust
use infra_consensus::RaftNode;
use lightbulb::knowledge::KnowledgeRegistry;

struct DistributedKnowledgeRegistry {
    raft: RaftNode,
    local_registry: KnowledgeRegistry,
}

impl DistributedKnowledgeRegistry {
    async fn register_knowledge(&mut self, knowledge: Knowledge) 
        -> Result<(), Error> 
    {
        // Propose knowledge registration to cluster
        let command = bincode::serialize(&RegisterCommand {
            knowledge_id: knowledge.id,
            data: knowledge.data,
        })?;
        
        self.raft.propose(command).await?;
        
        // Update local registry when committed
        self.local_registry.insert(knowledge)?;
        
        Ok(())
    }
    
    async fn query_knowledge(&self, query: &str) -> Result<Vec<Knowledge>, Error> {
        // Read from local committed state
        self.local_registry.search(query)
    }
}
```

#### Use Case 2: Training Parameter Coordination (M8)

```rust
use infra_consensus::RaftNode;

struct TrainingCoordinator {
    raft: RaftNode,
}

impl TrainingCoordinator {
    async fn update_global_model(&mut self, gradients: Vec<f32>) 
        -> Result<(), Error> 
    {
        // Propose gradient update to cluster
        let command = bincode::serialize(&GradientUpdate {
            gradients,
            timestamp: SystemTime::now(),
        })?;
        
        self.raft.propose(command).await?;
        
        Ok(())
    }
    
    async fn get_current_model(&self) -> Result<Model, Error> {
        // Read committed model state
        let state = self.raft.committed_state()?;
        Ok(state.current_model)
    }
}
```

### Raft Basics

#### Roles

- **Leader**: Handles all client requests and log replication
- **Follower**: Passive; receives and acknowledges log entries
- **Candidate**: Transitional state during election

#### Guarantees

- **Safety**: Never returns incorrect results
- **Availability**: Cluster operational if majority of nodes alive
- **Linearizability**: Operations appear atomic and ordered

#### Failure Handling

```rust
use infra_consensus::{RaftNode, HealthCheck};

async fn monitor_cluster_health(node: &RaftNode) {
    loop {
        tokio::time::sleep(Duration::from_secs(5)).await;
        
        let health = node.health_check().await;
        
        if health.leader_id.is_none() {
            eprintln!("WARNING: No leader elected");
        }
        
        if health.unhealthy_peers.len() > 0 {
            eprintln!("WARNING: Unhealthy peers: {:?}", health.unhealthy_peers);
        }
        
        println!("Cluster health: {} / {} nodes healthy",
            health.healthy_peers.len(),
            health.total_peers
        );
    }
}
```

### Dependencies

- `openraft`: Raft implementation
- `tokio`: Async runtime
- `bincode`: Binary serialization
- `serde`: Serialization framework
- `tonic`: gRPC communication

---

## Cross-Crate Integration Example

Here's how all three infrastructure crates work together for federated ML:

```rust
use infra_network::NetworkManager;
use infra_storage::{StorageBackend, BackendType};
use infra_consensus::RaftNode;

struct FederatedMLNode {
    network: NetworkManager,
    storage: StorageBackend,
    consensus: RaftNode,
}

impl FederatedMLNode {
    async fn new(node_id: u64, peers: Vec<String>) -> Result<Self, Error> {
        // Initialize network layer
        let network = NetworkManager::from_config("network.toml")?;
        network.start().await?;
        
        // Initialize storage layer
        let storage = StorageBackend::open(
            StorageConfig::rocksdb("/data/federated")
                .with_cache_size(500 * 1024 * 1024)
        )?;
        
        // Initialize consensus layer
        let consensus = RaftNode::new(RaftConfig {
            node_id: NodeId(node_id),
            peers: peers.iter().map(|p| parse_peer_id(p)).collect(),
            ..Default::default()
        })?;
        consensus.start().await?;
        
        Ok(Self { network, storage, consensus })
    }
    
    async fn share_knowledge(&mut self, knowledge: Vec<u8>) 
        -> Result<(), Error> 
    {
        // 1. Achieve consensus on knowledge sharing
        let commit_result = self.consensus.propose(knowledge.clone()).await?;
        commit_result.wait_committed().await?;
        
        // 2. Store locally
        let key = blake3::hash(&knowledge);
        self.storage.put(key.as_bytes(), &knowledge)?;
        
        // 3. Broadcast to peers
        self.network.broadcast("knowledge_update", &knowledge).await?;
        
        Ok(())
    }
    
    async fn retrieve_knowledge(&self, query: &[u8]) 
        -> Result<Option<Vec<u8>>, Error> 
    {
        // 1. Check local storage first
        if let Some(value) = self.storage.get(query)? {
            return Ok(Some(value));
        }
        
        // 2. Query peers via network
        let responses = self.network.query_all_peers("knowledge_query", query).await?;
        
        // 3. Return first valid response
        for response in responses {
            if let Some(value) = response.value {
                // Cache locally
                self.storage.put(query, &value)?;
                return Ok(Some(value));
            }
        }
        
        Ok(None)
    }
}
```

---

## Roadmap Integration

### Phase 4 (Month 6+): Distributed Systems

These infrastructure crates enable:

- ✅ **M7: Federated Retrieval**
  - Network layer: Peer discovery and communication
  - Storage layer: Distributed knowledge storage
  - Consensus layer: Agreement on shared knowledge

- ✅ **M8: Distributed Training**
  - Network layer: Gradient sharing between nodes
  - Storage layer: Checkpoint persistence
  - Consensus layer: Training parameter coordination

### Time Savings

- **Without infrastructure crates**: 10-15 weeks to build from scratch
- **With infrastructure crates**: 4-6 weeks for integration and customization
- **Net savings**: 6-9 weeks of development time

---

## Getting Started

### 1. Review the Code

```powershell
# Explore network layer
Get-Content crates\infra-network\src\lib.rs

# Explore storage layer
Get-Content crates\infra-storage\src\lib.rs

# Explore consensus layer
Get-Content crates\infra-consensus\src\raft\mod.rs
```

### 2. Run Examples

```powershell
# Network discovery example
cargo run --example peer_discovery --package infra-network

# Storage backend comparison
cargo run --example storage_backends --package infra-storage

# Raft cluster simulation
cargo run --example raft_cluster --package infra-consensus
```

### 3. Run Tests

```powershell
# Test individual crates
cargo test --package infra-network
cargo test --package infra-storage
cargo test --package infra-consensus

# Run integration tests
cargo test --test distributed_integration
```

---

## Next Steps

1. **Familiarize with APIs**: Read source code and run examples
2. **Plan M7 Integration**: Design federated retrieval architecture
3. **Prototype**: Build small proof-of-concept using these crates
4. **Optimize**: Profile and optimize for Lightbulb's specific needs
5. **Document**: Add Lightbulb-specific integration guides

---

## References

- **Source Code**: `crates/infra-network/`, `crates/infra-storage/`, `crates/infra-consensus/`
- **Integration Guide**: `docs/DYNANIML_INTEGRATION.md`
- **Roadmap**: Main project roadmap (M7, M8)
- **Raft Paper**: ["In Search of an Understandable Consensus Algorithm"](https://raft.github.io/raft.pdf)
