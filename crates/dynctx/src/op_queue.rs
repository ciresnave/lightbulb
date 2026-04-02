// ============================================================================
// op_queue.rs — single‑writer operation queue
// ============================================================================
//! Thread‑safe MPSC queue funneling all state‑mutating operations to the
//! single inference thread.  Other threads (prefetch, I/O, oracle) obtain a
//! clone of [`OpSender`] and push ops; the inference loop owns an
//! [`OpReceiver`] and drains it between decoding steps.
//!
//! Uses `crossbeam::channel::bounded` because we want predictable
//! back‑pressure: if producers outrun the consumer, they’ll block instead of
//! unboundedly buffering.

use crossbeam::channel::{bounded, Receiver, Sender};

use crate::node_arena::NodeKey;

/// Bounded capacity of the op queue.  1 k is ample for interactive use; tune
/// if your workload performs huge batch mutations.
pub const QUEUE_CAP: usize = 1024;

/// Operation types recognised by the core mutator.
#[derive(Debug)]
#[repr(u16)]
pub enum Op {
    /// Insert `tokens` after `cursor`.
    Add { cursor: NodeKey, tokens: Vec<u32> },
    /// Excise inclusive span [start, end].  Caller guarantees ordering.
    Drop { start: NodeKey, end: NodeKey },
    /// Evict `pages` (each 64‑token block) from VRAM → RAM.
    SwapOut { pages: Vec<u32> },
    /// Bring `pages` into VRAM (caller already ensured room).
    SwapIn { pages: Vec<u32> },
}

/// Producer handle cloned by worker threads.
#[derive(Clone)]
pub struct OpSender(Sender<Op>);
/// Consumer handle owned by the inference thread.
pub struct OpReceiver(Receiver<Op>);

/// Create a bounded MPSC pair.  The receiver is `!Sync` on purpose — it must
/// live on the inference thread.
pub fn make_queue() -> (OpSender, OpReceiver) {
    let (tx, rx) = bounded::<Op>(QUEUE_CAP);
    (OpSender(tx), OpReceiver(rx))
}

impl OpSender {
    /// Non‑blocking push; returns `Err(op)` if the queue is full.
    #[inline]
    pub fn try_send(&self, op: Op) -> Result<(), Op> {
        self.0.try_send(op).map_err(|e| e.into_inner())
    }
    /// Blocking push respecting back‑pressure.
    #[inline]
    pub fn send(&self, op: Op) {
        self.0.send(op).expect("queue disconnected")
    }
}

impl OpReceiver {
    /// Drain all available ops into the supplied vec (cleared first).
    #[inline]
    pub fn drain_into(&self, out: &mut Vec<Op>) {
        out.clear();
        while let Ok(op) = self.0.try_recv() {
            out.push(op);
        }
    }
}
