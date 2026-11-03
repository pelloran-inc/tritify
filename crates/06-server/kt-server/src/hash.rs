//! The reproducing hashes a response carries. Each is a SHA-256 over integers only, with a domain tag:
//! - **completion** (`kt-completion/1`), each part little-endian:
//!   - the prompt length (`u32`) and the prompt ids (`u32` each)
//!   - the output length and the output ids
//!   - the finish byte (`0` stop, `1` length)
//!   - the 64 hex characters of the logits hash, which covers every step's integer logits
//! - **embeddings** (`kt-embedding/1`), per input:
//!   - the token count and the ids
//!   - the scale (`mant` as `i64`, `shift` as `i32`)
//!   - the dimension and every code (`i32`)
//!
//! Two runs agree on a hash only if they agree on every one of those integers.

use kt_core::generate::{Finish, Generation};
use kt_core::IntTensor;
use sha2::{Digest, Sha256};

fn hex(h: Sha256) -> String {
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

fn ids(h: &mut Sha256, v: &[u32]) {
    h.update((v.len() as u32).to_le_bytes());
    for &x in v {
        h.update(x.to_le_bytes());
    }
}

pub(crate) fn completion(prompt: &[u32], g: &Generation) -> String {
    let mut h = Sha256::new();
    h.update(b"kt-completion/1");
    ids(&mut h, prompt);
    ids(&mut h, &g.tokens);
    h.update([match g.finish {
        Finish::Stop => 0u8,
        Finish::Length => 1u8,
    }]);
    h.update(g.logits_hash.as_bytes());
    hex(h)
}

/// The request fingerprint behind a response id (`kt-request/1`): chat or completion, the prompt ids, the budget.
pub(crate) fn request(chat: bool, prompt: &[u32], max_new: usize) -> String {
    let mut h = Sha256::new();
    h.update(b"kt-request/1");
    h.update([u8::from(chat)]);
    ids(&mut h, prompt);
    h.update((max_new as u64).to_le_bytes());
    hex(h)
}

pub(crate) fn embeddings(items: &[(Vec<u32>, IntTensor)]) -> String {
    let mut h = Sha256::new();
    h.update(b"kt-embedding/1");
    for (toks, e) in items {
        ids(&mut h, toks);
        h.update(e.scale.mant.to_le_bytes());
        h.update(e.scale.shift.to_le_bytes());
        h.update((e.data.len() as u32).to_le_bytes());
        for &v in &e.data {
            h.update(v.to_le_bytes());
        }
    }
    hex(h)
}
