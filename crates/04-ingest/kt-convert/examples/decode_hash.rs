//! Cross-ISA bit-identity: a fixed single-machine decode, hashed with the engine's canonical
//! `hash_int`. Thread-pinned (threads=1) so the hash is independent of a device's core count — the only variable left
//! is the ISA. Run on x86 + aarch64 + the Mac and compare: equal hashes ⇒ the integer engine is bit-identical across
//! architectures on real hardware (no float, deterministic).
//!
//!   cargo run --release -p kt-convert --example decode_hash -- <model_dir> [seq_len]

use kt_convert::HfModelSource;
use kt_core::verify::hash_int;
use kt_core::{set_max_threads, CausalLM, IntTensor};
use kt_model_decoder::DecoderArch;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: decode_hash <model_dir> [seq_len]");
        std::process::exit(2);
    }
    let dir = &args[1];
    let seq: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(48);

    let src = HfModelSource::from_dir(dir).unwrap_or_else(|e| {
        eprintln!("load: {e}");
        std::process::exit(1);
    });
    eprintln!("building exact-integer decoder from {dir} …");
    let model = DecoderArch::build_decoder(&src).unwrap_or_else(|e| {
        eprintln!("build: {e}");
        std::process::exit(1);
    });
    let vocab = model.vocab();
    let n = model.n_layers();
    let tokens: Vec<usize> = (0..seq).map(|i| i % vocab.max(1)).collect();

    set_max_threads(1); // thread-count-independent → the only variable across runs is the ISA
    let mut cache = model.new_cache();
    let mut stream = Vec::with_capacity(seq * vocab);
    let mut scale0 = None;
    for (pos, &t) in tokens.iter().enumerate() {
        let li = model.decode_step(t, &mut cache, pos);
        if scale0.is_none() {
            scale0 = Some(li.scale);
        }
        stream.extend_from_slice(&li.data);
    }
    let h = hash_int(&IntTensor::new(stream, vec![seq, vocab], scale0.unwrap()));
    println!("DECODE_HASH={h} seq={seq} layers={n} vocab={vocab} arch={} threads=1", std::env::consts::ARCH);
}
