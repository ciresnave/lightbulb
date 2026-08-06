//! Focused tests for the surviving `ParallelCacheBuilder` / `ParallelKvCache`
//! API — the Candle-backed KV cache the engine still runs on.
//!
//! # Why this file exists
//!
//! It replaces `batch_integration.rs`, which tested a request→slot lifecycle
//! (`prepare_batch`, `available_slots`, `release_request`, `get_cache_position`,
//! `set_cache_position`, `reset_request_state`) that has since been **deleted**
//! — those names exist nowhere in `src/` any more. That suite could not be
//! repaired, only replaced, because the layer it exercised no longer has those
//! responsibilities: slot lifecycle moved out, and what remains here is cache
//! storage plus index/mask computation.
//!
//! The Fuel-side equivalent of the deleted lifecycle is `fuel::kv_block_pool`
//! (`open`/`append`/`filled_tokens`/`discard`/`evict_blocks`/`splice`), driven
//! by `lightbulb::model_fuel::policies`. That path is block-granular rather than
//! a fixed `batch × context` rectangle, so it is a strict superset — it can
//! express prefix sharing and ragged lengths, which the old API could not.
//! Nothing here duplicates it; this covers only what the Candle path still
//! ships.
//!
//! # What is asserted
//!
//! Properties, not shapes. Each test states a claim that could be false, and
//! several would fail on a plausible implementation slip: cross-slot
//! contamination, a mask that lets a masked-out row attend, a purity contract
//! quietly becoming stateful, or ring wraparound landing on the wrong index.

use candlelight::core::{DType, Device, Tensor};
use lightbulb::engine::ParallelCacheBuilder;

const HEADS: usize = 2;
const HEAD_DIM: usize = 4;

/// Read one `[slot, head, pos, :]` vector out of a `[b, h, ctx, d]` cache.
///
/// `narrow` rather than fancy indexing so the helper cannot silently disagree
/// with the layout it is meant to inspect.
fn read_at(t: &Tensor, slot: usize, head: usize, pos: usize) -> Vec<f32> {
    t.narrow(0, slot, 1)
        .unwrap()
        .narrow(1, head, 1)
        .unwrap()
        .narrow(2, pos, 1)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

/// A `[batch, HEADS, seq, HEAD_DIM]` tensor whose every element for batch row
/// `b` is `values[b]`, so a stray write between rows is visible by value alone.
fn per_slot_tensor(values: &[f32], seq: usize, device: &Device) -> Tensor {
    let per_row = HEADS * seq * HEAD_DIM;
    let mut data = Vec::with_capacity(values.len() * per_row);
    for &v in values {
        data.extend(std::iter::repeat(v).take(per_row));
    }
    Tensor::from_vec(data, (values.len(), HEADS, seq, HEAD_DIM), device).unwrap()
}

fn builder(batch: usize, context: usize) -> (ParallelCacheBuilder, Device) {
    let device = Device::Cpu;
    let b = ParallelCacheBuilder::new(batch, context, DType::F32, &device).unwrap();
    (b, device)
}

#[test]
fn make_cache_is_zeroed_and_correctly_shaped() {
    let (builder, _device) = builder(3, 8);
    let cache = builder.make_cache(HEADS, HEAD_DIM).unwrap();

    assert_eq!(cache.k().dims(), &[3, HEADS, 8, HEAD_DIM]);
    assert_eq!(cache.v().dims(), &[3, HEADS, 8, HEAD_DIM]);

    // Zero-init is load-bearing, not cosmetic: `append` scatters into this
    // buffer, so anything never written is read back as attention over
    // whatever was left here. Uninitialised memory would show up as plausible
    // garbage logits rather than as a crash.
    let k_all = cache.k().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(k_all.iter().all(|&x| x == 0.0), "K cache is not zero-initialised");
    let v_all = cache.v().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(v_all.iter().all(|&x| x == 0.0), "V cache is not zero-initialised");
}

#[test]
fn indices_and_mask_is_pure() {
    // The doc comment promises it: "Position/indices state must be advanced
    // explicitly by the caller after the forward pass." A version that
    // advanced internally would still pass a shape check, and would then
    // double-advance every caller that also calls set_position — writing each
    // token two slots further along than intended.
    let (mut b, _device) = builder(2, 8);
    b.set_position(0, 3);
    b.set_position(1, 5);
    let before: Vec<usize> = b.positions().to_vec();

    let first = b.indices_and_mask(1, &[true, true]).unwrap();
    let after_one: Vec<usize> = b.positions().to_vec();
    let second = b.indices_and_mask(1, &[true, true]).unwrap();

    assert_eq!(before, after_one, "indices_and_mask advanced positions; it is documented as pure");
    assert_eq!(before, b.positions().to_vec(), "positions drifted across two calls");

    let m1 = first.mask().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let m2 = second.mask().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(m1, m2, "two identical calls produced different masks");
}

#[test]
fn an_inactive_slot_gets_a_permissive_mask_and_that_is_deliberate() {
    // An inactive slot's mask row is ALL ZEROS, which `IndicesAndMask`'s own
    // doc defines as "can attend" (src/cache/parallel_cache_builder.rs:32-36).
    // So the row is permitted to attend everywhere, including stale KV left by
    // a previous occupant of the same physical slot.
    //
    // That reads like a cross-request leak and is not one. The alternative —
    // an all -inf row, which is what "masked out" would literally mean — is
    // NUMERICALLY INVALID: softmax subtracts the row max, so every entry
    // becomes exp(-inf - -inf) = NaN. Verified against candlelight rather than
    // assumed: `softmax_last_dim` over a row of f32::NEG_INFINITY returns
    // [NaN, NaN, NaN, NaN]. A NaN row in a batched forward is far worse than a
    // permissive one, because NaN propagates and a discarded row does not.
    //
    // What actually protects the paused request is that its cache is NOT
    // WRITTEN while it is inactive — see `an_inactive_slot_keeps_its_cached_kv`,
    // which is the test that can fail if that protection regresses. This test
    // pins the mask as intentional so nobody "fixes" it into NaN.
    //
    // An earlier version of this test asserted the same values under the name
    // `a_masked_out_slot_attends_to_nothing`, with a failure message claiming
    // a non-zero row would mean it "can attend to stale KV" — exactly backwards
    // for additive masking. It therefore could not fail in the direction it
    // claimed to guard, and would have failed if the code were changed to match
    // its own name.
    let (mut b, _device) = builder(2, 8);
    b.set_position(0, 2);
    b.set_position(1, 2);

    let iam = b.indices_and_mask(1, &[true, false]).unwrap();
    let mask = iam.mask();
    let dims = mask.dims().to_vec();
    assert_eq!(dims.len(), 4, "mask is documented as (b, h, t, k), got {dims:?}");

    let inactive_row = mask
        .narrow(0, 1, 1)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert!(
        inactive_row.iter().all(|&x| x == 0.0),
        "the inactive slot's mask is not all-zero. If this was an intentional          change to -inf, note that softmax over an all -inf row yields NaN; the          protection this row does NOT provide is supplied by append honouring          IndicesAndMask.active instead"
    );
    assert!(
        inactive_row.iter().all(|&x| !x.is_nan()),
        "the inactive slot's mask contains NaN before softmax has even run"
    );

    // Control: the ACTIVE slot's row must contain -inf for future positions.
    // Without this the test would pass against an implementation that returned
    // an all-zero mask for every slot, which would break causal attention.
    let active_row = mask
        .narrow(0, 0, 1)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert!(
        active_row.iter().any(|&x| x == f32::NEG_INFINITY),
        "the ACTIVE slot's mask has no -inf entries, so causal masking is not          being applied and this test's all-zero assertion above is vacuous"
    );
}

#[test]
fn append_writes_each_slot_at_its_own_position_without_touching_the_others() {
    // The batch-isolation property, and the reason this file exists. Two slots
    // at different positions write different values in the same call; each must
    // land in its own row at its own position. A broadcast bug or an index built
    // from the wrong slot shows up here and essentially nowhere else, because
    // shapes stay correct throughout.
    let (mut b, device) = builder(2, 8);
    let mut cache = b.make_cache(HEADS, HEAD_DIM).unwrap();

    b.set_position(0, 2);
    b.set_position(1, 5);

    let k = per_slot_tensor(&[7.0, 9.0], 1, &device);
    let v = per_slot_tensor(&[-7.0, -9.0], 1, &device);
    let iam = b.indices_and_mask(1, &[true, true]).unwrap();
    cache.append(&k, &v, &iam).unwrap();

    assert_eq!(read_at(cache.k(), 0, 0, 2), vec![7.0; HEAD_DIM], "slot 0 K missing at its position");
    assert_eq!(read_at(cache.k(), 1, 0, 5), vec![9.0; HEAD_DIM], "slot 1 K missing at its position");
    assert_eq!(read_at(cache.v(), 0, 0, 2), vec![-7.0; HEAD_DIM], "slot 0 V missing");
    assert_eq!(read_at(cache.v(), 1, 0, 5), vec![-9.0; HEAD_DIM], "slot 1 V missing");

    // Neither slot may appear at the other's position, and nothing may land
    // anywhere else.
    assert_eq!(read_at(cache.k(), 0, 0, 5), vec![0.0; HEAD_DIM], "slot 1's write leaked into slot 0");
    assert_eq!(read_at(cache.k(), 1, 0, 2), vec![0.0; HEAD_DIM], "slot 0's write leaked into slot 1");
    for pos in [0usize, 1, 3, 4, 6, 7] {
        assert_eq!(read_at(cache.k(), 0, 0, pos), vec![0.0; HEAD_DIM], "slot 0 written at {pos}");
    }
}

#[test]
fn set_position_wraps_the_cache_index_at_the_context_boundary() {
    // The cache is a ring: index = position % context. A sequence longer than
    // the window must overwrite the oldest entry rather than run off the end,
    // and `set_position` is documented as the O(1) way to get there.
    let context = 8;
    let (mut b, device) = builder(1, context);
    let mut cache = b.make_cache(HEADS, HEAD_DIM).unwrap();

    // Position 11 is one full lap plus 3.
    b.set_position(0, context + 3);
    let k = per_slot_tensor(&[4.0], 1, &device);
    let v = per_slot_tensor(&[4.0], 1, &device);
    let iam = b.indices_and_mask(1, &[true]).unwrap();
    cache.append(&k, &v, &iam).unwrap();

    assert_eq!(
        read_at(cache.k(), 0, 0, 3),
        vec![4.0; HEAD_DIM],
        "position {} did not wrap to index 3 in a {context}-slot ring",
        context + 3
    );
}

#[test]
fn set_slot_kv_restores_one_slot_and_leaves_the_rest_alone() {
    // Prefix-cache restoration: dropping a saved prefix into one slot must not
    // disturb its neighbours, which are mid-request.
    let (b, device) = builder(3, 8);
    let mut cache = b.make_cache(HEADS, HEAD_DIM).unwrap();

    let prefix_len = 3;
    let k_slot = per_slot_tensor(&[2.0], prefix_len, &device);
    let v_slot = per_slot_tensor(&[-2.0], prefix_len, &device);

    let written = cache.set_slot_kv(1, &k_slot, &v_slot).unwrap();
    assert_eq!(written, prefix_len, "reported prefix length disagrees with the tensor");

    for pos in 0..prefix_len {
        assert_eq!(read_at(cache.k(), 1, 0, pos), vec![2.0; HEAD_DIM], "slot 1 K missing at {pos}");
        assert_eq!(read_at(cache.v(), 1, 0, pos), vec![-2.0; HEAD_DIM], "slot 1 V missing at {pos}");
        assert_eq!(read_at(cache.k(), 0, 0, pos), vec![0.0; HEAD_DIM], "slot 0 was disturbed");
        assert_eq!(read_at(cache.k(), 2, 0, pos), vec![0.0; HEAD_DIM], "slot 2 was disturbed");
    }
    // Past the prefix, slot 1 is still untouched.
    assert_eq!(read_at(cache.k(), 1, 0, prefix_len), vec![0.0; HEAD_DIM], "wrote beyond the prefix");
}

#[test]
fn reset_batch_index_clears_one_slot_only() {
    // A slot is reused when its request finishes. Resetting it must not rewind
    // the neighbours, or their next write lands on top of live KV.
    let (mut b, _device) = builder(3, 8);
    b.set_position(0, 4);
    b.set_position(1, 6);
    b.set_position(2, 2);

    b.reset_batch_index(1);

    assert_eq!(b.positions(), &[4, 0, 2], "reset_batch_index disturbed a neighbouring slot");
}

#[test]
fn an_inactive_slot_keeps_its_cached_kv() {
    // CR.1's contract, stated at parallel_model_manager.rs:1309: a request in
    // AwaitingToolResult is included in the decode batch as inactive because
    // "KV cache is preserved". A slot that is inactive this step must therefore
    // read back afterwards exactly as it did before.
    //
    // The failure this guards is not an error. `append` scatters K/V for every
    // batch row, and an inactive slot's indices are its REAL live write
    // position rather than a sentinel — so the paused request's cache gets
    // overwritten with whatever that row computed while it was masked out.
    // It resumes attending over corrupted history and produces fluent, wrong
    // output with nothing logged.
    let (mut b, device) = builder(2, 8);
    let mut cache = b.make_cache(HEADS, HEAD_DIM).unwrap();
    b.set_position(0, 0);
    b.set_position(1, 0);

    // Both slots write once. Slot 1 is the request that will pause; 7.0 is the
    // KV it must still have afterwards.
    let iam_both = b.indices_and_mask(1, &[true, true]).unwrap();
    let k1 = per_slot_tensor(&[1.0, 7.0], 1, &device);
    let v1 = per_slot_tensor(&[1.0, 7.0], 1, &device);
    cache.append(&k1, &v1, &iam_both).unwrap();
    assert_eq!(
        read_at(cache.k(), 1, 0, 0),
        vec![7.0; HEAD_DIM],
        "setup failed: slot 1 never received its KV, so the real assertion below would be vacuous"
    );

    // Slot 1 pauses. Slot 0 keeps decoding.
    let iam_one = b.indices_and_mask(1, &[true, false]).unwrap();
    let k2 = per_slot_tensor(&[2.0, 99.0], 1, &device);
    let v2 = per_slot_tensor(&[2.0, 99.0], 1, &device);
    cache.append(&k2, &v2, &iam_one).unwrap();

    assert_eq!(
        read_at(cache.k(), 1, 0, 0),
        vec![7.0; HEAD_DIM],
        "the paused slot's K cache was overwritten while it was inactive — \
         CR.1 promises this KV is preserved, and a resumed tool-call request \
         will attend over corrupted history"
    );
    assert_eq!(
        read_at(cache.v(), 1, 0, 0),
        vec![7.0; HEAD_DIM],
        "the paused slot's V cache was overwritten while it was inactive"
    );

    // Control: the ACTIVE slot must still have been written, or this test would
    // pass against an `append` that simply does nothing.
    assert_eq!(
        read_at(cache.k(), 0, 0, 0),
        vec![2.0; HEAD_DIM],
        "the active slot was NOT written — append is a no-op and the assertions above prove nothing"
    );
}
