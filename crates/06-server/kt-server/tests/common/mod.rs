//! Test harness: a tiny synthetic decoder with a matching word-level tokenizer, a server on an ephemeral port, and
//! a raw HTTP/1.1 client (so the tests depend on nothing but the server's own stack).

#![allow(dead_code)]

use kt_convert::text::Tokenizer;
use kt_core::model::{DecoderConfig, DecoderModel, DecoderWeights, LayerWeights};
use kt_server::ServerModel;
use serde_json::Value;
use std::net::SocketAddr;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

pub const TINY_TOKENIZER: &str = r#"{
  "version": "1.0", "truncation": null, "padding": null,
  "added_tokens": [
    {"id": 0, "content": "<|im_start|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 1, "content": "<|im_end|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "normalizer": null, "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null, "decoder": null,
  "model": {"type": "WordLevel", "unk_token": "[UNK]",
            "vocab": {"<|im_start|>": 0, "<|im_end|>": 1, "user": 2, "assistant": 3, "hello": 4, "world": 5, "[UNK]": 6}}
}"#;

fn vecf(s: &mut u64, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((*s >> 40) as f64 / (1u64 << 24) as f64 * 2.0 - 1.0) as f32
        })
        .collect()
}

/// A 2-layer decoder over the tokenizer's 7 ids. It has no end-of-sequence id, so a generation runs to
/// `max_tokens` (its random weights pick `<|im_end|>` first, which would make every comparison empty).
pub fn tiny_model() -> ServerModel {
    let cfg = DecoderConfig {
        dim: 16,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 4,
        hidden: 32,
        vocab: 7,
        eps: 1e-6,
        rope_theta: 10000.0,
        max_seq: 64,
    };
    let (dim, qd, kvd, hid, voc) =
        (cfg.dim, cfg.n_heads * cfg.head_dim, cfg.n_kv_heads * cfg.head_dim, cfg.hidden, cfg.vocab);
    let mut s = 0x5032_0c00u64;
    let layers = (0..cfg.n_layers)
        .map(|_| LayerWeights {
            input_norm: vecf(&mut s, dim),
            wq: vecf(&mut s, qd * dim),
            wk: vecf(&mut s, kvd * dim),
            wv: vecf(&mut s, kvd * dim),
            wo: vecf(&mut s, dim * qd),
            wq_b: None,
            wk_b: None,
            wv_b: None,
            post_norm: vecf(&mut s, dim),
            gate: vecf(&mut s, hid * dim),
            up: vecf(&mut s, hid * dim),
            down: vecf(&mut s, dim * hid),
        })
        .collect();
    let w = DecoderWeights {
        embed: vecf(&mut s, voc * dim),
        layers,
        final_norm: vecf(&mut s, dim),
        lm_head: vecf(&mut s, voc * dim),
        cfg,
    };
    let tok = Tokenizer::from_json(TINY_TOKENIZER).unwrap();
    ServerModel { id: "tiny".into(), model: DecoderModel::build(&w), tok }
}

pub async fn start(model: ServerModel) -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, kt_server::router(model)).await.unwrap() });
    addr
}

pub struct Reply {
    pub status: u16,
    pub headers: String,
    pub body: String,
}

impl Reply {
    pub fn json(&self) -> Value {
        serde_json::from_str(&self.body).unwrap_or_else(|e| panic!("not JSON ({e}): {}", self.body))
    }
    pub fn header(&self, name: &str) -> Option<String> {
        self.headers.lines().find_map(|l| {
            let (k, v) = l.split_once(':')?;
            k.trim().eq_ignore_ascii_case(name).then(|| v.trim().to_string())
        })
    }
    /// The `data:` payloads of a server-sent-event body.
    pub fn events(&self) -> Vec<String> {
        self.body
            .lines()
            .filter_map(|l| l.strip_prefix("data: ").or_else(|| l.strip_prefix("data:")))
            .map(String::from)
            .collect()
    }
}

pub async fn http(addr: SocketAddr, method: &str, path: &str, body: Option<&str>) -> Reply {
    let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
    let b = body.unwrap_or("");
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: test\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{b}",
        b.len()
    );
    s.write_all(req.as_bytes()).await.unwrap();
    let mut raw = Vec::new();
    s.read_to_end(&mut raw).await.unwrap();
    let raw = String::from_utf8(raw).unwrap();
    let (head, rest) = raw.split_once("\r\n\r\n").unwrap();
    let status = head.split_whitespace().nth(1).unwrap().parse().unwrap();
    let headers = head.split_once("\r\n").map_or("", |(_, h)| h).to_string();
    let chunked = headers.to_ascii_lowercase().contains("transfer-encoding: chunked");
    let body = if chunked { dechunk(rest) } else { rest.to_string() };
    Reply { status, headers, body }
}

fn dechunk(mut s: &str) -> String {
    let mut out = String::new();
    while let Some((size, rest)) = s.split_once("\r\n") {
        let n = usize::from_str_radix(size.trim(), 16).unwrap_or(0);
        if n == 0 {
            break;
        }
        out.push_str(&rest[..n]);
        s = &rest[n + 2..];
    }
    out
}
