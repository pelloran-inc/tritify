//! `GET /v1/fleet/telemetry/stream`: Server-Sent Events that tail a fleet telemetry file
//! (`tritify-fleet-telemetry/2`, NDJSON) as a coordinator writes it. A browser cannot tail a file on the coordinator;
//! this is the transport.
//! - The response opens with `retry: 500`, so a dropped stream reconnects in about half a second.
//! - Each line is one event: `id: <run_id>/<seq>`, `data: <the line>`, no `event:` field.
//! - A connect without `Last-Event-ID` streams the file's run from its first line and then tails it. With
//!   `Last-Event-ID: <run_id>/<seq>` it resumes after that line if the file still holds that run, else from the start of
//!   the run it holds.
//! - A coordinator starting a new run truncates the file; the stream then starts over with the new `run_start`.
//! - When there is nothing to send, a comment line `: idle` goes out at least every 15 s.

use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;
use serde_json::Value;
use std::convert::Infallible;
use std::path::PathBuf;
use std::time::Duration;
use tokio_stream::wrappers::ReceiverStream;

/// How often the file is polled for new lines.
const POLL: Duration = Duration::from_millis(100);

#[derive(Clone)]
struct Fleet {
    path: PathBuf,
    idle: Duration,
}

/// The fleet stream over the telemetry file at `path`, with an idle comment every 15 s.
pub fn router(path: PathBuf) -> Router {
    router_with_idle(path, Duration::from_secs(15))
}

/// The same, with another idle interval (tests use a short one).
pub fn router_with_idle(path: PathBuf, idle: Duration) -> Router {
    Router::new().route("/v1/fleet/telemetry/stream", get(stream)).with_state(Fleet { path, idle })
}

/// One complete line of the file, as the stream sends it.
#[derive(Debug, PartialEq, Eq)]
pub struct Line {
    pub id: String,
    pub data: String,
}

/// The events for the complete lines in `text`, after `resume` (`run_id`, `seq`) when the text holds that run. `run`
/// carries the current run's id from line to line; a `run_start` sets it.
pub fn lines(text: &str, run: &mut Option<String>, resume: &Option<(String, u64)>) -> Vec<Line> {
    let mut out = Vec::new();
    for l in text.lines().filter(|l| !l.trim().is_empty()) {
        let Ok(v) = serde_json::from_str::<Value>(l) else { continue };
        if v["ev"] == "run_start" {
            *run = v["run_id"].as_str().map(String::from);
        }
        let (Some(r), Some(seq)) = (run.as_deref(), v["seq"].as_u64()) else { continue };
        if resume.as_ref().is_some_and(|(rr, rs)| rr == r && seq <= *rs) {
            continue;
        }
        out.push(Line { id: format!("{r}/{seq}"), data: l.to_string() });
    }
    out
}

/// `<run_id>/<seq>` → its parts.
pub fn parse_last_event_id(s: &str) -> Option<(String, u64)> {
    let (r, seq) = s.rsplit_once('/')?;
    Some((r.to_string(), seq.parse().ok()?))
}

async fn stream(State(f): State<Fleet>, headers: HeaderMap) -> impl IntoResponse {
    let resume = headers.get("last-event-id").and_then(|v| v.to_str().ok()).and_then(parse_last_event_id);
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(256);
    tokio::spawn(async move {
        if tx.send(Ok(Event::default().retry(Duration::from_millis(500)))).await.is_err() {
            return;
        }
        let (mut offset, mut run) = (0usize, None::<String>);
        loop {
            let text = tokio::fs::read(&f.path).await.unwrap_or_default();
            if text.len() < offset {
                (offset, run) = (0, None); // the file was truncated: a new run
            }
            // only complete lines: up to the last newline
            if let Some(end) = text[offset..].iter().rposition(|&b| b == b'\n').map(|p| offset + p + 1) {
                for l in lines(&String::from_utf8_lossy(&text[offset..end]), &mut run, &resume) {
                    if tx.send(Ok(Event::default().id(l.id).data(l.data))).await.is_err() {
                        return;
                    }
                }
                offset = end;
            }
            if tx.is_closed() {
                return;
            }
            tokio::time::sleep(POLL).await;
        }
    });
    Sse::new(ReceiverStream::new(rx)).keep_alive(KeepAlive::new().interval(f.idle).text("idle"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const RUN: &str = "{\"v\":2,\"seq\":0,\"t_ms\":0,\"ev\":\"run_start\",\"run_id\":\"r1\"}\n{\"v\":2,\"seq\":1,\"t_ms\":5,\"ev\":\"hb\"}\n";

    #[test]
    fn lines_carry_run_and_seq_and_resume_after_the_last_id() {
        let mut run = None;
        let all = lines(RUN, &mut run, &None);
        assert_eq!(all.iter().map(|l| l.id.as_str()).collect::<Vec<_>>(), ["r1/0", "r1/1"]);
        let mut run = None;
        let rest = lines(RUN, &mut run, &parse_last_event_id("r1/0"));
        assert_eq!(rest.iter().map(|l| l.id.as_str()).collect::<Vec<_>>(), ["r1/1"]);
        let mut run = None;
        let other = lines(RUN, &mut run, &parse_last_event_id("r0/7"));
        assert_eq!(other.len(), 2, "another run's id resumes nothing: the file's run streams from its start");
        assert_eq!(parse_last_event_id("1793-ab/17"), Some(("1793-ab".into(), 17)));
        assert_eq!(parse_last_event_id("junk"), None);
    }

    /// A generation run's v3 lines (`token`, `shared`, a `load` on `hb`) stream exactly as v2's do; the
    /// stream reads only `run_id` and `seq`.
    #[test]
    fn a_v3_run_streams_like_a_v2_run() {
        let v3 = concat!(
            "{\"v\":3,\"seq\":0,\"t_ms\":0,\"ev\":\"run_start\",\"schema\":\"tritify-fleet-telemetry/3\",\"run_id\":\"g1\"}\n",
            "{\"v\":3,\"seq\":1,\"t_ms\":9,\"ev\":\"shared\",\"layer\":1,\"token\":0,\"step\":null,\"node\":\"coordinator\",\"rows\":1,\"us\":800}\n",
            "{\"v\":3,\"seq\":2,\"t_ms\":12,\"ev\":\"token\",\"step\":0,\"pos\":35,\"id\":7,\"text\":\"\",\"step_ms\":51}\n",
            "{\"v\":3,\"seq\":3,\"t_ms\":13,\"ev\":\"hb\",\"in_flight\":null,\"load\":[{\"node\":\"coordinator\"}]}\n",
        );
        let mut run = None;
        let all = lines(v3, &mut run, &None);
        assert_eq!(all.iter().map(|l| l.id.as_str()).collect::<Vec<_>>(), ["g1/0", "g1/1", "g1/2", "g1/3"]);
        assert!(all.iter().zip(v3.lines()).all(|(l, src)| l.data == src), "each line goes out unchanged");
        let mut run = None;
        let rest = lines(v3, &mut run, &parse_last_event_id("g1/2"));
        assert_eq!(rest.iter().map(|l| l.id.as_str()).collect::<Vec<_>>(), ["g1/3"]);
    }
}
