use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

type TokenId = Uuid;

#[derive(Debug, Serialize, Deserialize, Clone)]
struct Value {
    kind: String,
    data: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct AccessTrace {
    token: TokenId,
    module: String,
    action: String,
    ts: u128,
}

#[derive(Debug)]
struct ThoughtWorkspace {
    id: Uuid,
    storage: HashMap<TokenId, Value>,
    traces: Vec<AccessTrace>,
}

impl ThoughtWorkspace {
    fn new() -> Self {
        ThoughtWorkspace {
            id: Uuid::new_v4(),
            storage: HashMap::new(),
            traces: Vec::new(),
        }
    }

    fn now_ts() -> u128 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis()
    }

    fn insert(&mut self, value: Value, producer: &str) -> TokenId {
        let id = Uuid::new_v4();
        self.storage.insert(id, value);
        self.traces.push(AccessTrace {
            token: id.clone(),
            module: producer.to_string(),
            action: "write".to_string(),
            ts: Self::now_ts(),
        });
        id
    }

    fn read(&mut self, id: &TokenId, reader: &str) -> Option<Value> {
        if let Some(v) = self.storage.get(id) {
            self.traces.push(AccessTrace {
                token: id.clone(),
                module: reader.to_string(),
                action: "read".to_string(),
                ts: Self::now_ts(),
            });
            return Some(v.clone());
        }
        None
    }

    /// Persist selected tokens to disk (simulate persisting to dynctx arena)
    fn persist(&self, tokens: &[TokenId], path: &str) -> Result<()> {
        // Collect selected tokens and their values. TokenId (Uuid) is serializable because
        // the uuid crate is configured with the `serde` feature in Cargo.toml.
        let mut out: Vec<(TokenId, &Value)> = Vec::new();
        for t in tokens {
            if let Some(v) = self.storage.get(t) {
                out.push((t.clone(), v));
            }
        }

        let mut f = File::create(path)?;
        let s = serde_json::to_string(&out)?;
        f.write_all(s.as_bytes())?;
        Ok(())
    }

    fn traces_json(&self) -> Result<String> {
        Ok(serde_json::to_string(&self.traces)?)
    }
}

fn main() -> Result<()> {
    let mut tw = ThoughtWorkspace::new();
    println!("ThoughtWorkspace PoC (id={})", tw.id);

    let t1 = tw.insert(
        Value {
            kind: "image_features".to_string(),
            data: "edges, corners".to_string(),
        },
        "edge-detector",
    );
    let t2 = tw.insert(
        Value {
            kind: "texture_features".to_string(),
            data: "coarse texture".to_string(),
        },
        "texture-detector",
    );

    // Module higher-level reads low-level token and produces a classification
    if let Some(v) = tw.read(&t1, "shape-detector") {
        println!("shape-detector saw: {}", v.data);
        let _t3 = tw.insert(
            Value {
                kind: "shape_hypothesis".to_string(),
                data: "arm-like shape".to_string(),
            },
            "shape-detector",
        );
    }

    // Persist t2 (texture) to disk as example of saving useful tokens
    tw.persist(&[t2], "thought_persist.json")?;

    println!("Traces: {}", tw.traces_json()?);
    Ok(())
}
