#![feature(f16)]

use expert_torch::batch_scan;
mod expert;
mod expert_torch;
fn main() {
    batch_scan();
}
