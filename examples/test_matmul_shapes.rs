use candlelight::core::{Device, Result, Tensor};

fn main() -> Result<()> {
    let device = Device::Cpu;

    // Test matmul shapes
    let input = Tensor::randn(0f32, 1f32, &[1, 1, 64], &device)?; // [batch, seq, in]
    let weight = Tensor::randn(0f32, 1f32, &[128, 64], &device)?; // [out, in]

    println!("Input shape: {:?}", input.dims());
    println!("Weight shape: {:?}", weight.dims());

    // Try transpose
    let weight_t = weight.t()?;
    println!("Weight.t() shape: {:?}", weight_t.dims());

    // Try direct matmul (candlelight::nn::Linear style)
    let result = input.broadcast_matmul(&weight_t)?;
    println!("Result shape: {:?}", result.dims());

    Ok(())
}
