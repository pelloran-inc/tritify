//! The handlers. A chat or completion request is prepared (validated, tokenized, budgeted), then run on a blocking
//! thread while holding the engine lock, so it starts and ends on one backend. `/kt/v1/verify` reuses exactly the
//! same prepare-and-run path as the request it checks.

use crate::api::{
    self, ApiError, ChatRequest, Common, CompletionRequest, EmbeddingRequest, RuntimeRequest, VerifyRequest,
};
use crate::{hash, AppState};
use axum::body::Bytes;
use axum::extract::State;
use axum::http::HeaderValue;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use kt_convert::text::{Message, TextStream};
use kt_core::generate::{embed_mean, generate_greedy, Finish, Generation};
use kt_core::CausalLM;
use serde_json::{json, Value};
use std::convert::Infallible;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

type Shared = Arc<AppState>;

const DEFAULT_MAX_TOKENS: usize = 256;
const GREEDY: &str = "greedy";
const GREEDY_NOTE: &str = "greedy (temperature/top_p not applied)";

pub(crate) fn router(st: Shared) -> Router {
    Router::new()
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/completions", post(completions))
        .route("/v1/embeddings", post(embeddings))
        .route("/kt/v1/verify", post(verify))
        .route("/kt/v1/runtime", get(runtime_get).post(runtime_post))
        .with_state(st)
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// Which kind of text request: it decides the response object names.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Chat,
    Completion,
}

/// A validated, tokenized request, ready to run.
struct Prepared {
    kind: Kind,
    prompt: Vec<u32>,
    max_new: usize,
    sampling: &'static str,
    stream: bool,
}

impl Prepared {
    /// The response id: derived from the request (kind, prompt ids, budget), so it is known before generation, the
    /// same on every chunk of a stream, and the same for the same request.
    fn id(&self) -> String {
        let prefix = match self.kind {
            Kind::Chat => "chatcmpl",
            Kind::Completion => "cmpl",
        };
        format!("{prefix}-{}", &hash::request(self.kind == Kind::Chat, &self.prompt, self.max_new)[..24])
    }
}

/// What a run produced.
struct Done {
    gen: Generation,
    text: String,
    backend: String,
    hash: String,
}

fn check_common(st: &AppState, c: &Common) -> Result<(), ApiError> {
    if let Some(m) = &c.model {
        if *m != st.m.id {
            return Err(ApiError::unknown_model(m, &st.m.id));
        }
    }
    if c.n.unwrap_or(1) != 1 {
        return Err(ApiError::bad("n > 1 is not supported (greedy decoding gives one answer)"));
    }
    let stop_given = match &c.stop {
        None | Some(Value::Null) => false,
        Some(Value::Array(a)) => !a.is_empty(),
        Some(_) => true,
    };
    if stop_given {
        return Err(ApiError::bad("stop sequences are not supported yet; generation stops at the model's end token"));
    }
    Ok(())
}

fn sampling(c: &Common) -> &'static str {
    let t = c.temperature.is_some_and(|t| t > 0.0);
    let p = c.top_p.is_some_and(|p| p < 1.0);
    if t || p {
        GREEDY_NOTE
    } else {
        GREEDY
    }
}

/// The token budget: the prompt plus the generated tokens must fit the model's position tables.
fn budget(st: &AppState, prompt: usize, c: &Common) -> Result<usize, ApiError> {
    let room = st.m.model.max_positions().saturating_sub(prompt);
    if prompt == 0 {
        return Err(ApiError::bad("the prompt is empty"));
    }
    if room == 0 {
        return Err(ApiError::bad(format!(
            "the prompt has {prompt} tokens; the model covers {} positions",
            st.m.model.max_positions()
        )));
    }
    match c.max_completion_tokens.or(c.max_tokens) {
        Some(n) if n > room => Err(ApiError::bad(format!(
            "max_tokens {n} plus the prompt's {prompt} tokens exceed the model's {} positions",
            st.m.model.max_positions()
        ))),
        Some(n) => Ok(n),
        None => Ok(DEFAULT_MAX_TOKENS.min(room)),
    }
}

fn prepare_chat(st: &AppState, req: &ChatRequest) -> Result<Prepared, ApiError> {
    check_common(st, &req.common)?;
    let messages = req
        .messages
        .iter()
        .map(|m| Ok(Message { role: m.role.clone(), content: api::content_text(&m.content)? }))
        .collect::<Result<Vec<_>, ApiError>>()?;
    if messages.is_empty() {
        return Err(ApiError::bad("messages is empty"));
    }
    let prompt = st.m.tok.encode(&st.m.tok.chat_prompt(&messages), false).map_err(ApiError::internal)?;
    let max_new = budget(st, prompt.len(), &req.common)?;
    Ok(Prepared { kind: Kind::Chat, prompt, max_new, sampling: sampling(&req.common), stream: req.common.stream })
}

fn prepare_completion(st: &AppState, req: &CompletionRequest) -> Result<Prepared, ApiError> {
    check_common(st, &req.common)?;
    let text = match &req.prompt {
        Value::String(s) => s.clone(),
        Value::Array(a) if a.len() == 1 && a[0].is_string() => a[0].as_str().unwrap_or_default().to_string(),
        _ => return Err(ApiError::bad("prompt must be a string (one prompt per request)")),
    };
    let prompt = st.m.tok.encode(&text, true).map_err(ApiError::internal)?;
    let max_new = budget(st, prompt.len(), &req.common)?;
    Ok(Prepared { kind: Kind::Completion, prompt, max_new, sampling: sampling(&req.common), stream: req.common.stream })
}

/// Run a prepared request to the end on one backend; `on_text` sees each text delta (for streaming).
fn run(st: &AppState, p: &Prepared, mut on_text: impl FnMut(String)) -> Result<Done, ApiError> {
    let _engine = st.engine.lock().unwrap_or_else(|e| e.into_inner());
    let backend = kt_core::backend::backend().name();
    let mut ts = TextStream::new(&st.m.tok);
    let mut err = None;
    let gen = generate_greedy(&st.m.model, &p.prompt, p.max_new, st.m.tok.eos_ids(), |t| match ts.push(t) {
        Ok(d) if !d.is_empty() => on_text(d),
        Ok(_) => {}
        Err(e) => err = Some(e),
    });
    if let Some(e) = err {
        return Err(ApiError::internal(e));
    }
    let rest = ts.finish().map_err(ApiError::internal)?;
    if !rest.is_empty() {
        on_text(rest);
    }
    let text = st.m.tok.decode(&gen.tokens).map_err(ApiError::internal)?;
    let hash = hash::completion(&p.prompt, &gen);
    Ok(Done { gen, text, backend, hash })
}

fn finish_reason(f: Finish) -> &'static str {
    match f {
        Finish::Stop => "stop",
        Finish::Length => "length",
    }
}

fn kt_fields(p: &Prepared, d: &Done) -> Value {
    json!({"hash": d.hash, "logits_hash": d.gen.logits_hash, "backend": d.backend, "sampling": p.sampling})
}

fn usage(p: &Prepared, d: &Done) -> Value {
    let (pt, ct) = (p.prompt.len(), d.gen.tokens.len());
    json!({"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct})
}

fn body(st: &AppState, p: &Prepared, d: &Done) -> Value {
    let reason = finish_reason(d.gen.finish);
    match p.kind {
        Kind::Chat => json!({
            "id": p.id(), "object": "chat.completion", "created": now(),
            "model": st.m.id,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": d.text}, "finish_reason": reason}],
            "usage": usage(p, d), "kt": kt_fields(p, d),
        }),
        Kind::Completion => json!({
            "id": p.id(), "object": "text_completion", "created": now(), "model": st.m.id,
            "choices": [{"index": 0, "text": d.text, "logprobs": null, "finish_reason": reason}],
            "usage": usage(p, d), "kt": kt_fields(p, d),
        }),
    }
}

fn with_hash(v: Value, hash: &str) -> Response {
    let mut r = Json(v).into_response();
    if let Ok(h) = HeaderValue::from_str(hash) {
        r.headers_mut().insert("x-kt-hash", h);
    }
    r
}

/// Answer a prepared text request, as JSON or as server-sent events.
async fn respond(st: Shared, p: Prepared) -> Response {
    if !p.stream {
        let st2 = st.clone();
        return match tokio::task::spawn_blocking(move || run(&st2, &p, |_| {}).map(|d| (body(&st2, &p, &d), d.hash)))
            .await
        {
            Ok(Ok((v, h))) => with_hash(v, &h),
            Ok(Err(e)) => e.into_response(),
            Err(e) => ApiError::internal(format!("generation task failed: {e}")).into_response(),
        };
    }
    let (tx, rx) = tokio::sync::mpsc::channel::<String>(64);
    tokio::task::spawn_blocking(move || {
        let created = now();
        let id = p.id();
        let object = match p.kind {
            Kind::Chat => "chat.completion.chunk",
            Kind::Completion => "text_completion",
        };
        let chunk = |delta: &str, reason: Option<&str>, extra: Option<(Value, Value)>| -> String {
            let choice = match p.kind {
                Kind::Chat => {
                    let delta = if reason.is_some() { json!({}) } else { json!({"content": delta}) };
                    json!({"index": 0, "delta": delta, "finish_reason": reason})
                }
                Kind::Completion => json!({"index": 0, "text": delta, "logprobs": null, "finish_reason": reason}),
            };
            let mut c = json!({"id": id, "object": object, "created": created, "model": st.m.id, "choices": [choice]});
            if let Some((kt, usage)) = extra {
                c["kt"] = kt;
                c["usage"] = usage;
            }
            c.to_string()
        };
        if p.kind == Kind::Chat {
            let first = json!({"id": id, "object": object, "created": created, "model": st.m.id,
                               "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]});
            let _ = tx.blocking_send(first.to_string());
        }
        match run(&st, &p, |d| {
            let _ = tx.blocking_send(chunk(&d, None, None));
        }) {
            Ok(d) => {
                let _ = tx.blocking_send(chunk(
                    "",
                    Some(finish_reason(d.gen.finish)),
                    Some((kt_fields(&p, &d), usage(&p, &d))),
                ));
            }
            Err(e) => {
                let _ = tx.blocking_send(json!({"error": {"message": e.message, "type": e.kind}}).to_string());
            }
        }
        let _ = tx.blocking_send("[DONE]".into());
    });
    let events = ReceiverStream::new(rx).map(|s| Ok::<Event, Infallible>(Event::default().data(s)));
    Sse::new(events).keep_alive(KeepAlive::default()).into_response()
}

async fn models(State(st): State<Shared>) -> Json<Value> {
    Json(json!({"object": "list", "data": [{"id": st.m.id, "object": "model", "created": 0, "owned_by": "tritify"}]}))
}

async fn chat(State(st): State<Shared>, body: Bytes) -> Response {
    match api::parse::<ChatRequest>(&body).and_then(|r| prepare_chat(&st, &r)) {
        Ok(p) => respond(st, p).await,
        Err(e) => e.into_response(),
    }
}

async fn completions(State(st): State<Shared>, body: Bytes) -> Response {
    match api::parse::<CompletionRequest>(&body).and_then(|r| prepare_completion(&st, &r)) {
        Ok(p) => respond(st, p).await,
        Err(e) => e.into_response(),
    }
}

/// The token lists of an embeddings input: a string, a list of strings, a list of ids, or a list of id lists.
fn embedding_inputs(st: &AppState, input: &Value) -> Result<Vec<Vec<u32>>, ApiError> {
    let ids = |v: &Value| -> Option<Vec<u32>> { v.as_array()?.iter().map(|x| x.as_u64().map(|x| x as u32)).collect() };
    let enc = |s: &str| st.m.tok.encode(s, true).map_err(ApiError::internal);
    let items = match input {
        Value::String(s) => vec![enc(s)?],
        Value::Array(a) if a.iter().all(Value::is_string) => {
            a.iter().map(|s| enc(s.as_str().unwrap_or_default())).collect::<Result<_, _>>()?
        }
        Value::Array(a) if a.iter().all(Value::is_number) => vec![ids(input).ok_or_else(|| ApiError::bad("bad ids"))?],
        Value::Array(a) => {
            a.iter().map(|v| ids(v).ok_or_else(|| ApiError::bad("bad input"))).collect::<Result<_, _>>()?
        }
        _ => return Err(ApiError::bad("input must be a string, a list of strings, or token ids")),
    };
    let vocab = st.m.model.vocab();
    for t in &items {
        if t.is_empty() {
            return Err(ApiError::bad("an input is empty"));
        }
        if t.len() > st.m.model.max_positions() || t.iter().any(|&i| i as usize >= vocab) {
            return Err(ApiError::bad("an input is too long or has an id outside the vocabulary"));
        }
    }
    Ok(items)
}

fn run_embeddings(st: &AppState, req: &EmbeddingRequest) -> Result<(Value, String), ApiError> {
    if let Some(m) = &req.model {
        if *m != st.m.id {
            return Err(ApiError::unknown_model(m, &st.m.id));
        }
    }
    let base64 = match req.encoding_format.as_deref() {
        None | Some("float") => false,
        Some("base64") => true,
        Some(f) => return Err(ApiError::bad(format!("encoding_format '{f}' is not supported (float, base64)"))),
    };
    let items = embedding_inputs(st, &req.input)?;
    let _engine = st.engine.lock().unwrap_or_else(|e| e.into_inner());
    let backend = kt_core::backend::backend().name();
    let vecs: Vec<(Vec<u32>, kt_core::IntTensor)> = items
        .into_iter()
        .map(|t| {
            let toks: Vec<usize> = t.iter().map(|&i| i as usize).collect();
            let e = embed_mean(&st.m.model, &toks);
            (t, e)
        })
        .collect();
    let h = hash::embeddings(&vecs);
    let tokens: usize = vecs.iter().map(|(t, _)| t.len()).sum();
    let data: Vec<Value> = vecs
        .iter()
        .enumerate()
        .map(|(i, (_, e))| {
            let s = e.scale.to_f64(); // the payload is float, as the API defines; the hash above is over the integers
            let emb = if base64 {
                let bytes: Vec<u8> = e.data.iter().flat_map(|&c| ((c as f64 * s) as f32).to_le_bytes()).collect();
                json!(api::base64(&bytes))
            } else {
                json!(e.data.iter().map(|&c| c as f64 * s).collect::<Vec<f64>>())
            };
            json!({"object": "embedding", "index": i, "embedding": emb})
        })
        .collect();
    let body = json!({
        "object": "list", "data": data, "model": st.m.id,
        "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
        "kt": {"hash": h, "backend": backend, "pooling": "mean of the final-normed hidden state"},
    });
    Ok((body, h))
}

async fn embeddings(State(st): State<Shared>, body: Bytes) -> Response {
    let req = match api::parse::<EmbeddingRequest>(&body) {
        Ok(r) => r,
        Err(e) => return e.into_response(),
    };
    match tokio::task::spawn_blocking(move || run_embeddings(&st, &req)).await {
        Ok(Ok((v, h))) => with_hash(v, &h),
        Ok(Err(e)) => e.into_response(),
        Err(e) => ApiError::internal(format!("embedding task failed: {e}")).into_response(),
    }
}

/// Re-run a request exactly as its endpoint would (never streamed) and compare the hash.
async fn verify(State(st): State<Shared>, body: Bytes) -> Response {
    let req = match api::parse::<VerifyRequest>(&body) {
        Ok(r) => r,
        Err(e) => return e.into_response(),
    };
    let st2 = st.clone();
    let rerun = tokio::task::spawn_blocking(move || -> Result<(String, String), ApiError> {
        let inner = serde_json::to_vec(&req.request).map_err(|e| ApiError::bad(e.to_string()))?;
        let (hash, backend) = match req.endpoint.as_str() {
            "/v1/chat/completions" | "/v1/completions" => {
                let mut p = if req.endpoint == "/v1/chat/completions" {
                    prepare_chat(&st2, &api::parse::<ChatRequest>(&inner)?)?
                } else {
                    prepare_completion(&st2, &api::parse::<CompletionRequest>(&inner)?)?
                };
                p.stream = false;
                let d = run(&st2, &p, |_| {})?;
                (d.hash, d.backend)
            }
            "/v1/embeddings" => {
                let (v, h) = run_embeddings(&st2, &api::parse::<EmbeddingRequest>(&inner)?)?;
                (h, v["kt"]["backend"].as_str().unwrap_or_default().to_string())
            }
            e => return Err(ApiError::bad(format!("unknown endpoint '{e}'"))),
        };
        Ok((hash, backend))
    })
    .await;
    match rerun {
        Ok(Ok((hash, backend))) => {
            let ok = hash == body_hash(&body);
            with_hash(json!({"match": ok, "hash": hash, "expected": body_hash(&body), "backend": backend}), &hash)
        }
        Ok(Err(e)) => e.into_response(),
        Err(e) => ApiError::internal(format!("verify task failed: {e}")).into_response(),
    }
}

fn body_hash(body: &[u8]) -> String {
    api::parse::<VerifyRequest>(body).map(|r| r.hash).unwrap_or_default()
}

fn runtime_view() -> Value {
    let host = kt_runtime::detect();
    let gpus: Vec<Value> = host
        .gpus
        .iter()
        .enumerate()
        .map(|(i, g)| json!({"index": i, "name": g.name, "api": g.api, "kind": g.kind}))
        .collect();
    json!({
        "host": {"arch": host.arch, "os": host.os, "cores": host.cores,
                 "cpu_variants": host.cpu_variants.iter().map(|v| v.name()).collect::<Vec<_>>(), "gpus": gpus,
                 "cuda_devices": host.cudas},
        "auto": kt_runtime::auto_variant(&host).name(),
        "backend": kt_core::backend::backend().name(),
        "max_threads": kt_core::max_threads(),
    })
}

async fn runtime_get() -> Json<Value> {
    Json(runtime_view())
}

/// Switch the backend and/or the thread cap (between requests: it waits for the engine lock).
async fn runtime_post(State(st): State<Shared>, body: Bytes) -> Response {
    let req = match api::parse::<RuntimeRequest>(&body) {
        Ok(r) => r,
        Err(e) => return e.into_response(),
    };
    let r = tokio::task::spawn_blocking(move || -> Result<Value, ApiError> {
        let _engine = st.engine.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(b) = &req.backend {
            let cfg = kt_runtime::RuntimeConfig {
                backend: crate::parse_backend(b).map_err(ApiError::bad)?,
                max_threads: req.max_threads,
                ..Default::default()
            };
            let sel = kt_runtime::install(&cfg).map_err(ApiError::bad)?;
            let mut v = runtime_view();
            v["reason"] = json!(sel.reason);
            Ok(v)
        } else {
            if let Some(n) = req.max_threads {
                kt_core::set_max_threads(n);
            }
            Ok(runtime_view())
        }
    })
    .await;
    match r {
        Ok(Ok(v)) => Json(v).into_response(),
        Ok(Err(e)) => e.into_response(),
        Err(e) => ApiError::internal(format!("runtime task failed: {e}")).into_response(),
    }
}
