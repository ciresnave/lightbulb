use core::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Direction {
    Forward,
    Backward,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TemporalOrder {
    Before,
    After,
    During,
    Overlaps,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SemanticType {
    Similar,
    Opposite,
    Implies,
    Specializes,
    References,
    Custom(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum RelationshipType {
    Sequential { direction: Direction, distance: u32 },
    Temporal { order: TemporalOrder },
    Semantic { kind: SemanticType },
    Custom { name: String, metadata: Vec<u8> },
}

impl fmt::Display for RelationshipType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RelationshipType::Sequential { direction, distance } => {
                write!(f, "Sequential({:?}, {})", direction, distance)
            }
            RelationshipType::Temporal { order } => {
                write!(f, "Temporal({:?})", order)
            }
            RelationshipType::Semantic { kind } => {
                write!(f, "Semantic({:?})", kind)
            }
            RelationshipType::Custom { name, .. } => {
                write!(f, "Custom({})", name)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_create_sequential_relationship() {
        let rel = RelationshipType::Sequential {
            direction: Direction::Forward,
            distance: 1,
        };
        assert_eq!(rel.to_string(), "Sequential(Forward, 1)");
    }

    #[test]
    fn should_create_temporal_relationship() {
        let rel = RelationshipType::Temporal {
            order: TemporalOrder::Before,
        };
        assert_eq!(rel.to_string(), "Temporal(Before)");
    }

    #[test]
    fn should_create_semantic_relationship() {
        let rel = RelationshipType::Semantic {
            kind: SemanticType::Similar,
        };
        assert_eq!(rel.to_string(), "Semantic(Similar)");
    }

    #[test]
    fn should_create_custom_relationship() {
        let rel = RelationshipType::Custom {
            name: "test".to_string(),
            metadata: vec![1, 2, 3],
        };
        assert_eq!(rel.to_string(), "Custom(test)");
    }
}
