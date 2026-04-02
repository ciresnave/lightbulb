#[cfg(test)]
mod tests {
    use super::*;
    use crate::node_arena::{NodeArena, NodeKey};
    use crate::relationship_types::*;
    use crate::relationship_manager::*;

    #[test]
    fn test_relationship_inheritance() {
        let mut manager = RelationshipManager::new();

        // Create test nodes
        let source = manager.create_node("source".to_string(), "text".to_string());
        let target = manager.create_node("target".to_string(), "text".to_string());

        // Create base relationship type
        let base_metadata = RelationshipMetadata {
            id: "test.structural.base".to_string(),
            name: "Base Structural".to_string(),
            description: Some("Base structural relationship".to_string()),
            valid_sources: vec!["*".to_string()],
            valid_targets: vec!["*".to_string()],
            parent_id: None,
        };

        let base_structural = RelationshipType::Structural {
            kind: StructuralType::Contains,
            metadata: Some(StructuralMetadata {
                base: base_metadata,
                parent_type: None,
                ordering: Some(1),
                weight: Some(1.0),
            }),
        };

        // Create inherited relationship
        let inherited_metadata = RelationshipMetadata {
            id: "test.structural.inherited".to_string(),
            name: "Inherited Structural".to_string(),
            description: Some("Inherited structural relationship".to_string()),
            valid_sources: vec!["text".to_string()],
            valid_targets: vec!["text".to_string()],
            parent_id: Some("test.structural.base".to_string()),
        };

        let inherited = manager.create_inherited_relationship(
            &base_structural,
            inherited_metadata,
        ).unwrap();

        // Test inheritance chain
        assert!(inherited.inherits_from(&base_structural));
        assert_eq!(inherited.inheritance_chain().len(), 2);
        
        // Test validation
        assert!(inherited.validate_inheritance());
        
        // Test context layer consistency
        assert_eq!(inherited.context_layer(), base_structural.context_layer());
        
        // Test metadata inheritance
        let inherited_meta = inherited.metadata().unwrap();
        assert_eq!(inherited_meta.parent_id.as_ref().unwrap(), "test.structural.base");
    }

    #[test]
    fn test_invalid_inheritance() {
        let mut manager = RelationshipManager::new();

        // Create test nodes
        let source = manager.create_node("source".to_string(), "text".to_string());
        let target = manager.create_node("target".to_string(), "text".to_string());

        // Create relationships of different types
        let structural = RelationshipType::Structural {
            kind: StructuralType::Contains,
            metadata: Some(StructuralMetadata {
                base: RelationshipMetadata {
                    id: "test.structural".to_string(),
                    name: "Test Structural".to_string(),
                    description: None,
                    valid_sources: vec!["*".to_string()],
                    valid_targets: vec!["*".to_string()],
                    parent_id: None,
                },
                parent_type: None,
                ordering: None,
                weight: None,
            }),
        };

        let linguistic_metadata = RelationshipMetadata {
            id: "test.linguistic".to_string(),
            name: "Test Linguistic".to_string(),
            description: None,
            valid_sources: vec!["*".to_string()],
            valid_targets: vec!["*".to_string()],
            parent_id: Some("test.structural".to_string()),
        };

        // Try to inherit Linguistic from Structural (should fail)
        let result = manager.create_inherited_relationship(
            &structural,
            linguistic_metadata,
        );

        assert!(result.is_err());
    }
}
