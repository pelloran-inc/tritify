//! The fleet telemetry stream over a growing file. The response opens with `retry: 500`; events arrive in
//! order with `id: <run_id>/<seq>`; a reconnect with `Last-Event-ID` resumes after it with nothing lost or repeated; a
//! new run (the file truncated) streams from its `run_start`; `: idle` arrives during silence.

use std::io::Write;
use std::net::SocketAddr;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

fn line(run: &str, seq: u64, ev: &str) -> String {
    let extra = if ev == "run_start" { format!(",\"run_id\":\"{run}\"") } else { String::new() };
    format!("{{\"v\":2,\"seq\":{seq},\"t_ms\":{seq},\"ev\":\"{ev}\"{extra}}}\n")
}

async fn start(path: std::path::PathBuf) -> SocketAddr {
    let app = kt_server::fleet::router_with_idle(path, Duration::from_millis(300));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    addr
}

/// Open the stream and read until `done` holds for the text so far, or the deadline passes.
async fn read_until(addr: SocketAddr, last_id: Option<&str>, done: impl Fn(&str) -> bool) -> String {
    let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
    let hdr = last_id.map(|id| format!("Last-Event-ID: {id}\r\n")).unwrap_or_default();
    let req = format!("GET /v1/fleet/telemetry/stream HTTP/1.1\r\nHost: x\r\nAccept: text/event-stream\r\n{hdr}\r\n");
    s.write_all(req.as_bytes()).await.unwrap();
    let mut text = String::new();
    let mut buf = [0u8; 4096];
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while !done(&text) {
        let n = tokio::time::timeout_at(deadline, s.read(&mut buf)).await.expect("stream timed out").unwrap();
        assert!(n > 0, "stream closed early:\n{text}");
        text.push_str(&String::from_utf8_lossy(&buf[..n]));
    }
    text
}

fn ids(text: &str) -> Vec<String> {
    text.lines().filter_map(|l| l.strip_prefix("id: ").map(String::from)).collect()
}

#[tokio::test]
async fn the_stream_tails_resumes_and_restarts_on_a_new_run() {
    let path = std::env::temp_dir().join(format!("kt_fleet_stream_{}.ndjson", std::process::id()));
    std::fs::write(&path, format!("{}{}", line("r1", 0, "run_start"), line("r1", 1, "hb"))).unwrap();
    let addr = start(path.clone()).await;

    // the first lines, then one appended while the stream is open, then an idle comment
    let p2 = path.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(400)).await;
        let mut f = std::fs::OpenOptions::new().append(true).open(&p2).unwrap();
        f.write_all(line("r1", 2, "fetch").as_bytes()).unwrap();
    });
    let text = read_until(addr, None, |t| t.contains("id: r1/2") && t.contains(": idle")).await;
    assert!(text.contains("text/event-stream"), "{text}");
    let body = &text[text.find("\r\n\r\n").expect("headers") + 4..];
    assert!(body.contains("retry: 500"), "the stream opens with retry 500:\n{body}");
    assert!(body.find("retry: 500") < body.find("id: r1/0"), "retry comes first");
    assert_eq!(ids(&text), ["r1/0", "r1/1", "r1/2"]);
    assert!(!text.contains("event:"), "no event field");

    // a reconnect resumes after the last id it saw: nothing lost, nothing repeated
    let resumed = read_until(addr, Some("r1/1"), |t| t.contains("id: r1/2")).await;
    assert_eq!(ids(&resumed), ["r1/2"]);

    // a new run truncates the file; a stream that last saw run r1 gets run r2 from its run_start
    std::fs::write(&path, format!("{}{}", line("r2", 0, "run_start"), line("r2", 1, "end"))).unwrap();
    let next = read_until(addr, Some("r1/2"), |t| t.contains("id: r2/1")).await;
    assert_eq!(ids(&next), ["r2/0", "r2/1"]);
    std::fs::remove_file(&path).unwrap();
}
