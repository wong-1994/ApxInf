use std::path::PathBuf;

use apxinf_core::Device;
use apxinf_model::GrootRuntime;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = std::env::args_os().nth(1)
        .map(PathBuf::from).ok_or("usage: groot_load_smoke MODEL_DIR")?;
    let _runtime = GrootRuntime::from_dir(&model_dir, Device::Cuda(0))?;
    println!("GR00T checkpoint load: PASS");
    Ok(())
}
