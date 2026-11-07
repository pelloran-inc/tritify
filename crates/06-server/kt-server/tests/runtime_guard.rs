//! On a server with the default options, an unauthenticated client cannot switch the process-global backend or thread
//! cap. `POST /kt/v1/runtime` answers 403 in OpenAI's error shape and changes neither; `GET` still reads, and says the
//! switch is off. Its own test binary, so no other test shares its process-wide backend.

mod common;
use common::{http, start, tiny_model};

#[tokio::test]
async fn a_default_server_refuses_the_runtime_switch_and_changes_nothing() {
    let addr = start(tiny_model()).await;
    let before = http(addr, "GET", "/kt/v1/runtime", None).await;
    assert_eq!(before.status, 200, "{}", before.body);
    let before = before.json();
    assert_eq!(before["switch_enabled"], false);
    let (backend, threads) = (before["backend"].clone(), before["max_threads"].clone());
    assert!(backend.is_string() && threads.is_u64(), "{before}");

    // every kind of switch is refused before its body is read: a backend, a thread cap, both, nonsense
    let other = if threads.as_u64() == Some(1) { 2 } else { 1 };
    for body in [
        r#"{"backend": "cpu:portable"}"#.to_string(),
        format!(r#"{{"max_threads": {other}}}"#),
        format!(r#"{{"backend": "cpu:portable", "max_threads": {other}}}"#),
        "{not json".to_string(),
    ] {
        let r = http(addr, "POST", "/kt/v1/runtime", Some(&body)).await;
        assert_eq!(r.status, 403, "{body}: {}", r.body);
        let e = r.json();
        assert_eq!(e["error"]["code"], "runtime_switch_disabled", "{body}");
        assert_eq!(e["error"]["type"], "invalid_request_error", "{body}");
        assert!(e["error"]["message"].as_str().unwrap().contains("--allow-runtime-switch"), "{body}");
        // both guarded globals, each unchanged
        let now = http(addr, "GET", "/kt/v1/runtime", None).await.json();
        assert_eq!(now["backend"], backend, "{body}: the backend moved");
        assert_eq!(now["max_threads"], threads, "{body}: the thread cap moved");
    }

    // the API itself still answers
    let chat = r#"{"model": "tiny", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 4}"#;
    let r = http(addr, "POST", "/v1/chat/completions", Some(chat)).await;
    assert_eq!(r.status, 200, "{}", r.body);
    assert_eq!(r.json()["kt"]["backend"], backend);
}
