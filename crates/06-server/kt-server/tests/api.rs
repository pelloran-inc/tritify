//! The OpenAI-compatible API end to end, over real HTTP, on a tiny synthetic model. One test,
//! run in order, because the compute backend is process-wide and one step switches it.

mod common;
use common::{http, start, tiny_model};
use serde_json::{json, Value};

const CHAT: &str = r#"{"model": "tiny", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 12}"#;

fn content_of_stream(events: &[String]) -> (String, Value) {
    let mut text = String::new();
    let mut last = Value::Null;
    for e in events.iter().filter(|e| *e != "[DONE]") {
        let v: Value = serde_json::from_str(e).unwrap();
        let c = &v["choices"][0];
        text.push_str(c["delta"]["content"].as_str().or(c["text"].as_str()).unwrap_or(""));
        last = v;
    }
    (text, last)
}

#[tokio::test]
async fn the_api_answers_like_openai_and_every_answer_carries_its_hash() {
    let addr = start(tiny_model()).await;

    // models
    let r = http(addr, "GET", "/v1/models", None).await;
    assert_eq!((r.status, r.json()["data"][0]["id"].as_str()), (200, Some("tiny")));

    // chat: JSON, the hash in the body and the header, deterministic
    let r = http(addr, "POST", "/v1/chat/completions", Some(CHAT)).await;
    assert_eq!(r.status, 200, "{}", r.body);
    let v = r.json();
    let hash = v["kt"]["hash"].as_str().unwrap().to_string();
    assert_eq!(hash.len(), 64);
    assert_eq!(r.header("x-kt-hash").as_deref(), Some(hash.as_str()));
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["choices"][0]["message"]["role"], "assistant");
    assert_eq!(v["kt"]["sampling"], "greedy");
    let text = v["choices"][0]["message"]["content"].as_str().unwrap().to_string();
    let n_out = v["usage"]["completion_tokens"].as_u64().unwrap();
    println!("chat: {n_out} tokens, {:?}, finish {}, hash {hash}", text, v["choices"][0]["finish_reason"]);
    assert!(n_out > 1 && !text.is_empty(), "the comparisons below need a real answer");
    let again = http(addr, "POST", "/v1/chat/completions", Some(CHAT)).await.json();
    assert_eq!(again["kt"]["hash"], hash.as_str(), "a repeat gives the same hash");
    assert_eq!(again["id"], v["id"], "the id is derived from the request");

    // streaming carries the same text and ends with the same hash
    let streamed = CHAT.replace("\"max_tokens\"", "\"stream\": true, \"max_tokens\"");
    let r = http(addr, "POST", "/v1/chat/completions", Some(&streamed)).await;
    assert_eq!(r.status, 200);
    let events = r.events();
    assert_eq!(events.last().map(String::as_str), Some("[DONE]"));
    let (s_text, last) = content_of_stream(&events);
    assert_eq!(s_text, text, "streamed deltas reassemble the non-streamed answer");
    assert_eq!(last["kt"]["hash"], hash.as_str());
    assert_eq!(last["object"], "chat.completion.chunk");
    let ids: Vec<Value> = events
        .iter()
        .filter(|e| *e != "[DONE]")
        .map(|e| serde_json::from_str::<Value>(e).unwrap()["id"].clone())
        .collect();
    assert!(ids.iter().all(|i| *i == v["id"]), "every chunk carries the response's one id");

    // a temperature is answered greedily and says so
    let warm = CHAT.replace("\"max_tokens\"", "\"temperature\": 0.7, \"max_tokens\"");
    let w = http(addr, "POST", "/v1/chat/completions", Some(&warm)).await.json();
    assert_eq!(
        (w["kt"]["hash"].as_str(), w["kt"]["sampling"].as_str()),
        (Some(hash.as_str()), Some("greedy (temperature/top_p not applied)"))
    );

    // completions, plain and streamed
    let comp = r#"{"model": "tiny", "prompt": "hello world", "max_tokens": 8}"#;
    let c = http(addr, "POST", "/v1/completions", Some(comp)).await.json();
    assert_eq!(c["object"], "text_completion");
    let ch = c["kt"]["hash"].as_str().unwrap().to_string();
    let cs = http(
        addr,
        "POST",
        "/v1/completions",
        Some(&comp.replace("\"max_tokens\"", "\"stream\": true, \"max_tokens\"")),
    )
    .await;
    let (cs_text, cs_last) = content_of_stream(&cs.events());
    assert_eq!(
        (cs_text.as_str(), cs_last["kt"]["hash"].as_str()),
        (c["choices"][0]["text"].as_str().unwrap(), Some(ch.as_str()))
    );

    // embeddings: float and base64 encode the same integers, so the hash is the same
    let e = http(addr, "POST", "/v1/embeddings", Some(r#"{"model": "tiny", "input": ["hello world", "world"]}"#))
        .await
        .json();
    assert_eq!(e["data"].as_array().unwrap().len(), 2);
    assert_eq!(e["data"][0]["embedding"].as_array().unwrap().len(), 16);
    let eb = http(
        addr,
        "POST",
        "/v1/embeddings",
        Some(r#"{"input": ["hello world", "world"], "encoding_format": "base64"}"#),
    )
    .await
    .json();
    assert_eq!(eb["kt"]["hash"], e["kt"]["hash"]);
    assert_eq!(eb["data"][0]["embedding"].as_str().unwrap().len(), (16 * 4usize).div_ceil(3) * 4);

    // verify: the right hash matches, a wrong one does not
    let req: Value = serde_json::from_str(CHAT).unwrap();
    let ok = http(
        addr,
        "POST",
        "/kt/v1/verify",
        Some(&json!({"endpoint": "/v1/chat/completions", "request": req, "hash": hash}).to_string()),
    )
    .await
    .json();
    assert_eq!(ok["match"], true);
    let bad = http(
        addr,
        "POST",
        "/kt/v1/verify",
        Some(&json!({"endpoint": "/v1/chat/completions", "request": req, "hash": "0".repeat(64)}).to_string()),
    )
    .await
    .json();
    assert_eq!(bad["match"], false);
    let ev =
        json!({"endpoint": "/v1/embeddings", "request": {"input": ["hello world", "world"]}, "hash": e["kt"]["hash"]});
    assert_eq!(http(addr, "POST", "/kt/v1/verify", Some(&ev.to_string())).await.json()["match"], true);

    // runtime: report, switch, refuse what the host lacks; the hash does not move
    let rt = http(addr, "GET", "/kt/v1/runtime", None).await.json();
    assert_eq!(rt["host"]["arch"], std::env::consts::ARCH);
    let set = http(addr, "POST", "/kt/v1/runtime", Some(r#"{"backend": "cpu:portable", "max_threads": 1}"#)).await;
    assert_eq!(
        (set.status, set.json()["backend"].as_str(), set.json()["max_threads"].as_u64()),
        (200, Some("portable"), Some(1))
    );
    let after = http(addr, "POST", "/v1/chat/completions", Some(CHAT)).await.json();
    assert_eq!(
        (after["kt"]["backend"].as_str(), after["kt"]["hash"].as_str()),
        (Some("portable"), Some(hash.as_str()))
    );
    let lacking = if cfg!(target_arch = "aarch64") { "cpu:avx2" } else { "cpu:neon" };
    for b in [lacking, "gpu:9999", "warp-drive"] {
        let r = http(addr, "POST", "/kt/v1/runtime", Some(&json!({"backend": b}).to_string())).await;
        assert_eq!((r.status, r.json()["error"]["type"].as_str()), (400, Some("invalid_request_error")), "{b}");
    }
    http(addr, "POST", "/kt/v1/runtime", Some(r#"{"backend": "auto", "max_threads": 0}"#)).await;

    // errors in OpenAI's shape
    let wrong = http(addr, "POST", "/v1/chat/completions", Some(&CHAT.replace("\"tiny\"", "\"gpt-4\""))).await;
    assert_eq!((wrong.status, wrong.json()["error"]["code"].as_str()), (404, Some("model_not_found")));
    for (body, why) in [
        (CHAT.replace("\"max_tokens\": 12", "\"n\": 2"), "n > 1"),
        (CHAT.replace("\"max_tokens\": 12", "\"stop\": [\"x\"]"), "stop"),
        (CHAT.replace("\"max_tokens\": 12", "\"max_tokens\": 100000"), "budget"),
        ("{not json".to_string(), "malformed"),
    ] {
        let r = http(addr, "POST", "/v1/chat/completions", Some(&body)).await;
        assert_eq!(
            (r.status, r.json()["error"]["type"].as_str()),
            (400, Some("invalid_request_error")),
            "{why}: {}",
            r.body
        );
    }
}
