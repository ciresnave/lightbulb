use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModuleInfo {
    id: String,
    usage_count: usize,
    last_used: usize,
}

#[derive(Debug)]
struct CacheEntry {
    id: String,
    usage_count: usize,
    last_used: usize,
}

struct RouterPoC {
    modules: Vec<ModuleInfo>,
    // simple cache
    cache: HashMap<String, CacheEntry>,
    cache_order: VecDeque<String>,
    time: usize,
}

impl RouterPoC {
    fn new(module_count: usize) -> Self {
        let modules = (0..module_count)
            .map(|i| ModuleInfo {
                id: format!("mod-{}", i),
                usage_count: 0,
                last_used: 0,
            })
            .collect();

        Self {
            modules,
            cache: HashMap::new(),
            cache_order: VecDeque::new(),
            time: 0,
        }
    }

    // Simulate attention scores for each module for a given input
    fn attention_scores(&self) -> HashMap<String, f32> {
        let mut rng = thread_rng();
        let mut map = HashMap::new();
        for m in &self.modules {
            map.insert(m.id.clone(), rng.gen::<f32>());
        }
        map
    }

    // Predictive vote: very small model that looks at recent sequence (here: random)
    fn predictive_vote(&self) -> Vec<String> {
        // return top-2 predicted modules (random for PoC)
        let mut rng = thread_rng();
        let mut ids: Vec<String> = self.modules.iter().map(|m| m.id.clone()).collect();
        ids.shuffle(&mut rng);
        ids.into_iter().take(2).collect()
    }

    // Combined score: attention + cache heuristic + predictive boost
    fn combined_ranking(
        &self,
        attention: &HashMap<String, f32>,
        predictive: &[String],
    ) -> Vec<(String, f32)> {
        let mut scores: Vec<(String, f32)> =
            attention.iter().map(|(id, s)| (id.clone(), *s)).collect();

        for (id, score) in scores.iter_mut() {
            // LRU/LFU boost: if in cache, add small score based on usage
            if let Some(entry) = self.cache.get(id) {
                *score += 0.2 * (entry.usage_count as f32 + 1.0);
            }

            // predictive vote boost
            if predictive.contains(id) {
                *score += 0.5;
            }
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores
    }

    fn ensure_cache(&mut self, id: &str, cache_size: usize) {
        if self.cache.contains_key(id) {
            let e = self.cache.get_mut(id).unwrap();
            e.usage_count += 1;
            e.last_used = self.time;
            return;
        }

        // Evict if needed (simple LRU by last_used)
        if self.cache.len() >= cache_size {
            if let Some((evict_id, _)) = self
                .cache
                .iter()
                .min_by_key(|(_, e)| e.last_used)
                .map(|(k, v)| (k.clone(), v.last_used))
            {
                self.cache.remove(&evict_id);
            }
        }

        self.cache.insert(
            id.to_string(),
            CacheEntry {
                id: id.to_string(),
                usage_count: 1,
                last_used: self.time,
            },
        );
    }

    fn step(&mut self, cache_size: usize) {
        self.time += 1;
        let attention = self.attention_scores();
        let predictive = self.predictive_vote();
        let ranking = self.combined_ranking(&attention, &predictive);

        // pick top-3 modules to run
        let chosen: Vec<String> = ranking.iter().take(3).map(|(id, _)| id.clone()).collect();

        println!(
            "t={} predictive={:?} chosen={:?}",
            self.time, predictive, chosen
        );

        for id in chosen {
            self.ensure_cache(&id, cache_size);
        }
    }
}

fn main() {
    let mut poc = RouterPoC::new(10);
    for _ in 0..20 {
        poc.step(4);
    }
    println!("Final cache: {:?}", poc.cache.keys().collect::<Vec<_>>());
}
