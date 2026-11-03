//! A firing control, in its own process: install a deliberately wrong compute backend (the `i32`
//! accumulator that wraps) underneath a running server. `/kt/v1/verify` must then report `match: false` for an
//! answer given earlier, and `match: true` again once the correct backend is back. The API itself never offers
//! this backend.

mod common;
use common::{http, start, tiny_model};
use kt_core::backend::{install_backend, linear_cpu, reset_backend, LinearBackend};
use serde_json::{json, Value};
use std::sync::Arc;

struct WrapI32;
impl LinearBackend for WrapI32 {
    fn name(&self) -> String {
        "planted-wrap-i32".into()
    }
    fn linear(&self, x: &[i32], w: &[i32], rows: usize, out: usize, k: usize) -> Vec<i64> {
        fn dot(a: &[i32], b: &[i32]) -> i64 {
            a.iter().zip(b).fold(0i32, |s, (&x, &y)| s.wrapping_add(x.wrapping_mul(y))) as i64
        }
        linear_cpu(x, w, rows, out, k, dot)
    }
}

#[tokio::test]
async fn verify_catches_a_divergent_backend() {
    let addr = start(tiny_model()).await;
    let req: Value =
        json!({"model": "tiny", "messages": [{"role": "user", "content": "hello world"}], "max_tokens": 10});
    let answer = http(addr, "POST", "/v1/chat/completions", Some(&req.to_string())).await.json();
    let hash = answer["kt"]["hash"].as_str().unwrap().to_string();
    let check = json!({"endpoint": "/v1/chat/completions", "request": req, "hash": hash}).to_string();

    install_backend(Arc::new(WrapI32));
    let planted = http(addr, "POST", "/kt/v1/verify", Some(&check)).await.json();
    assert_eq!(planted["backend"], "planted-wrap-i32");
    assert_eq!(planted["match"], false, "a wrapping accumulator must not reproduce the hash: {planted}");

    reset_backend();
    let restored = http(addr, "POST", "/kt/v1/verify", Some(&check)).await.json();
    assert_eq!((restored["match"].as_bool(), restored["backend"].as_str()), (Some(true), Some("portable")));
}
