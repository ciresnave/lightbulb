use infra_consensus::raft::types::{Command, CommandMetadata, CommandType};
use infra_consensus::raft::{self, RaftConfig};
use tokio;

#[tokio::test]
async fn test_basic_raft_node_creation() {
    // Create configuration for a single node
    let config = RaftConfig {
        node_id: 1,
        network_config: raft::config::NetworkConfig {
            listen_addr: "127.0.0.1".to_string(),
            raft_port: 8001,
            mgmt_port: 9001,
            ..Default::default()
        },
        ..Default::default()
    };

    // Create temporary directory for the node
    let temp_dir = tempfile::tempdir().unwrap();
    let mut node_config = config.clone();
    node_config.storage_path = temp_dir.path().to_path_buf();

    // Create the node
    let node = raft::create_node(node_config.node_id, node_config)
        .await
        .unwrap();

    // Verify node was created successfully
    assert_eq!(node.node_id(), 1); // Test metrics functionality
    let metrics = node.metrics().await;
    // Metrics should be initialized with zeros
    assert_eq!(metrics.append_entries_received(), 0);
    assert_eq!(metrics.votes_cast(), 0);
    assert_eq!(metrics.leadership_changes(), 0);
}

#[tokio::test]
async fn test_command_creation() {
    // Test command creation and basic structure
    let test_command = Command {
        id: "test-1".to_string(),
        data: "test data".as_bytes().to_vec(),
        command_type: CommandType::Mutation,
        metadata: CommandMetadata {
            timestamp: chrono::Utc::now(),
            source_node: 1,
        },
    };

    // Verify command was created correctly
    assert_eq!(test_command.id, "test-1");
    assert_eq!(test_command.data, "test data".as_bytes());
    assert!(matches!(test_command.command_type, CommandType::Mutation));
    assert_eq!(test_command.metadata.source_node, 1);
}

#[tokio::test]
async fn test_multiple_nodes_creation() {
    // Create configurations for three nodes
    let configs = vec![
        RaftConfig {
            node_id: 1,
            network_config: raft::config::NetworkConfig {
                listen_addr: "127.0.0.1".to_string(),
                raft_port: 8001,
                mgmt_port: 9001,
                ..Default::default()
            },
            ..Default::default()
        },
        RaftConfig {
            node_id: 2,
            network_config: raft::config::NetworkConfig {
                listen_addr: "127.0.0.1".to_string(),
                raft_port: 8002,
                mgmt_port: 9002,
                ..Default::default()
            },
            ..Default::default()
        },
        RaftConfig {
            node_id: 3,
            network_config: raft::config::NetworkConfig {
                listen_addr: "127.0.0.1".to_string(),
                raft_port: 8003,
                mgmt_port: 9003,
                ..Default::default()
            },
            ..Default::default()
        },
    ];

    // Create temporary directories for each node
    let temp_dirs: Vec<_> = configs
        .iter()
        .map(|_| tempfile::tempdir().unwrap())
        .collect();

    // Create nodes
    let mut nodes = Vec::new();
    for (config, temp_dir) in configs.into_iter().zip(temp_dirs.iter()) {
        let mut node_config = config.clone();
        node_config.storage_path = temp_dir.path().to_path_buf();

        let node = raft::create_node(node_config.node_id, node_config)
            .await
            .unwrap();
        nodes.push(node);
    }

    // Verify nodes were created successfully
    assert_eq!(nodes.len(), 3);

    // Test that each node has the correct ID
    for (i, node) in nodes.iter().enumerate() {
        assert_eq!(node.node_id(), (i + 1) as u64);
    } // Test metrics functionality for all nodes
    for node in &nodes {
        let metrics = node.metrics().await;
        // Metrics should be initialized with zeros
        assert_eq!(metrics.append_entries_received(), 0);
        assert_eq!(metrics.votes_cast(), 0);
        assert_eq!(metrics.leadership_changes(), 0);
    }
}
