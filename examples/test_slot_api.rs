/// Test the Slot API for state management and inspection
///
/// This demonstrates how the Slot abstraction provides type-safe access to
/// individual slot state without accidentally mixing data between requests.
///
/// **Note**: Slot is designed for state management and inspection, NOT for
/// writing K/V data. In production, K/V writes happen via batched model
/// forward passes, not per-slot operations.
use candle_core::{DType, Device};
use lightbulb::cache::parallel_cache_builder::ParallelCacheBuilder;

fn main() -> anyhow::Result<()> {
    let device = Device::cuda_if_available(0)?;
    let dtype = DType::F32;

    println!("=== Testing Slot API for State Management ===\n");

    // Create a cache builder with 4 slots
    let batch_size = 4;
    let context = 512;
    let num_heads = 8;
    let head_dim = 64;

    let mut builder = ParallelCacheBuilder::new(batch_size, context, dtype, &device)?;
    let mut cache = builder.make_cache(num_heads, head_dim)?;

    println!(
        "Created cache with {} slots, context={}\n",
        batch_size, context
    );

    // === Test 1: Using slots to set positions ===
    println!("Test 1: Setting positions through Slot API");
    {
        let mut slot0 = builder.get_slot_mut(0, &mut cache);
        slot0.set_position(5);
        println!("  Slot 0: set to position {}", slot0.position());

        // Need to drop slot0 before we can get another mutable reference
    }
    {
        let mut slot1 = builder.get_slot_mut(1, &mut cache);
        slot1.set_position(10);
        println!("  Slot 1: set to position {}", slot1.position());
    }
    {
        let mut slot2 = builder.get_slot_mut(2, &mut cache);
        slot2.set_position(3);
        println!("  Slot 2: set to position {}", slot2.position());
    }

    println!("\nVerify positions are independent:");
    println!("  Slot 0: position={}", builder.get_position(0));
    println!("  Slot 1: position={}", builder.get_position(1));
    println!("  Slot 2: position={}", builder.get_position(2));
    println!("  Slot 3: position={}", builder.get_position(3));

    // === Test 2: Resetting individual slots ===
    println!("\nTest 2: Resetting individual slots");
    {
        let mut slot1 = builder.get_slot_mut(1, &mut cache);
        println!("  Slot 1 before reset: position={}", slot1.position());
        slot1.reset();
        println!("  Slot 1 after reset: position={}", slot1.position());
    }

    println!("\nVerify only slot 1 was reset:");
    println!("  Slot 0: position={}", builder.get_position(0));
    println!("  Slot 1: position={}", builder.get_position(1));
    println!("  Slot 2: position={}", builder.get_position(2));

    // === Test 3: Querying cache indices ===
    println!("\nTest 3: Querying cache indices");
    {
        let slot0 = builder.get_slot_mut(0, &mut cache);
        println!(
            "  Slot 0: position={}, cache_index={}",
            slot0.position(),
            slot0.cache_index()
        );
        println!("  (Cache index is position % context)");
    }
    {
        let slot1 = builder.get_slot_mut(1, &mut cache);
        println!(
            "  Slot 1: position={}, cache_index={}",
            slot1.position(),
            slot1.cache_index()
        );
    }

    // === Test 4: Viewing individual slot's K/V (for inspection) ===
    println!("\nTest 4: Accessing individual slot's K/V tensors for inspection");
    {
        let slot0 = builder.get_slot_mut(0, &mut cache);
        let k_slot0 = slot0.k()?;
        let v_slot0 = slot0.v()?;

        println!("  Slot 0 K shape: {:?}", k_slot0.shape());
        println!("  Slot 0 V shape: {:?}", v_slot0.shape());
        println!(
            "  (Expected: [num_heads={}, context={}, head_dim={}])",
            num_heads, context, head_dim
        );
        println!("  Note: This is for debugging/inspection only");
    }

    // === Test 5: Full cache access for model ===
    println!("\nTest 5: Accessing full batched cache (for model forward pass)");
    let k_all = cache.k();
    let v_all = cache.v();
    println!("  Full K shape: {:?}", k_all.shape());
    println!("  Full V shape: {:?}", v_all.shape());
    println!(
        "  (Expected: [batch_size={}, num_heads={}, context={}, head_dim={}])",
        batch_size, num_heads, context, head_dim
    );
    println!("  Note: Model uses full batched tensors for parallel processing");

    println!("\n=== Slot API Test Complete ===");
    println!("\nKey benefits:");
    println!("  ✅ Type-safe per-slot state management");
    println!("  ✅ Prevents accidentally mixing slot data");
    println!("  ✅ Clear conceptual model: one slot = one request's state");
    println!("  ✅ Useful for inspection and debugging");
    println!("\nDesign note:");
    println!("  Slot is for STATE MANAGEMENT, not K/V writing");
    println!("  K/V writes happen via batched model forward passes");

    Ok(())
}
