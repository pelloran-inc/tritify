//! kt-server — the product server: an OpenAI-compatible HTTP API on the exact-integer engine. Any OpenAI client
//! talks to it by changing only its base URL. Every answer carries a SHA-256 (`kt.hash`, and the `x-kt-hash` header)
//! that the same request reproduces on any backend and any repeat, and `/kt/v1/verify` re-runs a request to check it.
//!
//! Endpoints:
//! - OpenAI: `GET /v1/models`, `POST /v1/chat/completions` (with `stream`), `POST /v1/completions`,
//!   `POST /v1/embeddings`
//! - native: `POST /kt/v1/verify`, `GET` / `POST /kt/v1/runtime` (the backend overrides)
//!
//! There is no authentication. The server binds `127.0.0.1` by default, and `POST /kt/v1/runtime` switches the
//! compute backend for every client: put an authenticating proxy in front before binding it to any other address.
//!
//! One model per process, loaded through `kt-convert`. The compute backend is installed through `kt-runtime`, so
//! the engine never depends on this layer. Sampling is greedy. Cluster endpoints are not part of this server; the
//! fleet telemetry stream (below) only relays a file a coordinator writes.
//!
//! # Example
//!
//! ```
//! use kt_runtime::{BackendChoice, CpuVariant};
//!
//! // The backend names the runtime endpoint and the --backend flag accept.
//! assert_eq!(kt_server::parse_backend("auto"), Ok(BackendChoice::Auto));
//! assert_eq!(kt_server::parse_backend("cpu:portable"), Ok(BackendChoice::Cpu(Some(CpuVariant::Portable))));
//! assert_eq!(kt_server::parse_backend("gpu:0"), Ok(BackendChoice::Gpu(Some(0))));
//! assert_eq!(kt_server::parse_backend("cuda"), Ok(BackendChoice::Cuda(None)));
//! assert!(kt_server::parse_backend("cpu:quantum").is_err());
//! ```

mod api;
pub mod fleet;
mod hash;
mod routes;

use kt_convert::text::Tokenizer;
use kt_convert::{load_text_model, HfModelSource};
use kt_core::DecoderModel;
use kt_runtime::{BackendChoice, CpuVariant, RuntimeConfig};
use std::net::SocketAddr;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// The model a server serves: its public id, the engine model and its tokenizer.
pub struct ServerModel {
    pub id: String,
    pub model: DecoderModel,
    pub tok: Tokenizer,
}

impl ServerModel {
    /// Load a Hugging Face checkpoint directory (a decoder or a MoE decoder). The id is the directory's name.
    pub fn load(dir: impl AsRef<Path>) -> Result<ServerModel, String> {
        let dir = dir.as_ref();
        let d = dir.to_str().ok_or("model path is not UTF-8")?;
        let model = load_text_model(&HfModelSource::from_dir(d)?)?;
        let tok = Tokenizer::from_dir(dir)?;
        let id = dir.file_name().and_then(|s| s.to_str()).unwrap_or("model").to_string();
        Ok(ServerModel { id, model, tok })
    }

    /// Load the model a spec declares (group `decoder` or `moe`). The id is the spec's.
    pub fn from_spec(spec: &kt_convert::spec::ModelSpec) -> Result<ServerModel, String> {
        let model = kt_convert::spec::load_text_model_spec(spec)?;
        let tok = kt_convert::spec::tokenizer_for(spec)?;
        Ok(ServerModel { id: spec.id.clone(), model, tok })
    }
}

/// Shared server state. `engine` serialises generations and backend switches: a request runs start to finish on
/// one backend.
pub(crate) struct AppState {
    pub m: ServerModel,
    pub engine: Mutex<()>,
}

/// The HTTP router for one model. The backend in use is whatever is installed in the engine (see [`serve`]).
pub fn router(model: ServerModel) -> axum::Router {
    routes::router(Arc::new(AppState { m: model, engine: Mutex::new(()) }))
}

/// Install the runtime configuration, bind `listen` and serve until the process ends. This is the entry point a
/// server binary calls.
pub async fn serve(model: ServerModel, listen: SocketAddr, runtime: &RuntimeConfig) -> Result<(), String> {
    serve_with(Some(model), None, listen, runtime).await
}

/// [`serve`] with the optional parts: a model (the OpenAI API) and a fleet telemetry file to stream
/// (`GET /v1/fleet/telemetry/stream`). With no model only the stream is served.
pub async fn serve_with(
    model: Option<ServerModel>,
    telemetry: Option<std::path::PathBuf>,
    listen: SocketAddr,
    runtime: &RuntimeConfig,
) -> Result<(), String> {
    let mut app = axum::Router::new();
    if let Some(model) = model {
        let sel = kt_runtime::install(runtime)?;
        eprintln!("kt-server · model {} · backend {} ({}) · http://{listen}", model.id, sel.backend, sel.reason);
        app = app.merge(router(model));
    }
    if let Some(path) = telemetry {
        eprintln!("kt-server · fleet telemetry {} · http://{listen}/v1/fleet/telemetry/stream", path.display());
        app = app.merge(fleet::router(path));
    }
    let listener = tokio::net::TcpListener::bind(listen).await.map_err(|e| format!("bind {listen}: {e}"))?;
    axum::serve(listener, app).await.map_err(|e| format!("serve: {e}"))
}

/// Parse a backend name: `auto`, `cpu`, `cpu:<variant>` (`portable`, `neon`, `avx2`, `avx512`), `gpu`, `gpu:<i>`,
/// `cuda` or `cuda:<i>`.
pub fn parse_backend(s: &str) -> Result<BackendChoice, String> {
    match s.split_once(':') {
        None if s == "auto" => Ok(BackendChoice::Auto),
        None if s == "cpu" => Ok(BackendChoice::Cpu(None)),
        None if s == "gpu" => Ok(BackendChoice::Gpu(None)),
        None if s == "cuda" => Ok(BackendChoice::Cuda(None)),
        Some(("cpu", v)) => CpuVariant::from_name(v)
            .map(|v| BackendChoice::Cpu(Some(v)))
            .ok_or_else(|| format!("unknown CPU variant '{v}' (portable, neon, avx2, avx512)")),
        Some(("gpu", i)) => i.parse().map(|i| BackendChoice::Gpu(Some(i))).map_err(|_| format!("bad GPU index '{i}'")),
        Some(("cuda", i)) => {
            i.parse().map(|i| BackendChoice::Cuda(Some(i))).map_err(|_| format!("bad CUDA device index '{i}'"))
        }
        _ => Err(format!("unknown backend '{s}' (auto, cpu, cpu:<variant>, gpu, gpu:<index>, cuda, cuda:<index>)")),
    }
}
