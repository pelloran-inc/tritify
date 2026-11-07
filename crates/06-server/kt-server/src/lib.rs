//! kt-server — the product server: an OpenAI-compatible HTTP API on the exact-integer engine. Any OpenAI client
//! talks to it by changing only its base URL. Every answer carries a SHA-256 (`kt.hash`, and the `x-kt-hash` header)
//! that the same request reproduces on any backend and any repeat, and `/kt/v1/verify` re-runs a request to check it.
//!
//! Endpoints:
//! - OpenAI: `GET /v1/models`, `POST /v1/chat/completions` (with `stream`), `POST /v1/completions`,
//!   `POST /v1/embeddings`
//! - native: `POST /kt/v1/verify`, `GET` / `POST /kt/v1/runtime` (the backend overrides)
//!
//! # Serving / security
//! - **The defaults:** the server binds the loopback address (`127.0.0.1:8080`) and has **no authentication**. Put an
//!   authenticating reverse proxy in front before binding it to any other address; on any other address it warns on
//!   stderr.
//! - **The runtime switch is process-global, and off by default.** `POST /kt/v1/runtime` installs the compute backend
//!   and sets the thread cap for the whole process, so for every client. It answers 403 until the server is started
//!   with `--allow-runtime-switch` ([`ServeOptions::allow_runtime_switch`]). Keep it off on any server a client you do
//!   not control can reach.
//! - No switch changes an answer's bits (every backend gives the engine's integers), but it does change speed,
//!   resources and availability. `GET /kt/v1/runtime` only reads, and reports whether the switch is enabled.
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

/// How a server serves. The default is the safe posture: the runtime switch off.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ServeOptions {
    /// Whether `POST /kt/v1/runtime` may switch the process-global backend and thread cap. Off by default: a
    /// disabled POST answers 403 and changes nothing. Enable it only where every client is trusted (loopback, or
    /// behind an authenticating proxy).
    pub allow_runtime_switch: bool,
}

/// Shared server state. `engine` serialises generations and backend switches: a request runs start to finish on
/// one backend.
pub(crate) struct AppState {
    pub m: ServerModel,
    pub engine: Mutex<()>,
    pub opts: ServeOptions,
}

/// The HTTP router for one model, with the default [`ServeOptions`] (the runtime switch off). The backend in use is
/// whatever is installed in the engine (see [`serve`]).
pub fn router(model: ServerModel) -> axum::Router {
    router_with(model, ServeOptions::default())
}

/// [`router`] with the given options.
pub fn router_with(model: ServerModel, opts: ServeOptions) -> axum::Router {
    routes::router(Arc::new(AppState { m: model, engine: Mutex::new(()), opts }))
}

/// Install the runtime configuration, bind `listen` and serve until the process ends, with the default
/// [`ServeOptions`] (the runtime switch off). This is the entry point a server binary calls.
pub async fn serve(model: ServerModel, listen: SocketAddr, runtime: &RuntimeConfig) -> Result<(), String> {
    serve_with(Some(model), None, listen, runtime, ServeOptions::default()).await
}

/// [`serve`] with the optional parts: a model (the OpenAI API), a fleet telemetry file to stream
/// (`GET /v1/fleet/telemetry/stream`), and the [`ServeOptions`]. With no model only the stream is served. On an
/// address beyond loopback it warns on stderr that there is no authentication; the bind stays the caller's choice, for
/// a deployment behind a proxy.
pub async fn serve_with(
    model: Option<ServerModel>,
    telemetry: Option<std::path::PathBuf>,
    listen: SocketAddr,
    runtime: &RuntimeConfig,
    opts: ServeOptions,
) -> Result<(), String> {
    if !listen.ip().is_loopback() {
        eprintln!(
            "kt-server · WARNING: listening on {listen}, beyond this machine, with no authentication. \
             Put an authenticating reverse proxy in front."
        );
    }
    let mut app = axum::Router::new();
    if let Some(model) = model {
        let sel = kt_runtime::install(runtime)?;
        let switch = if opts.allow_runtime_switch { "on (process-global)" } else { "off" };
        eprintln!(
            "kt-server · model {} · backend {} ({}) · runtime switch {switch} · http://{listen}",
            model.id, sel.backend, sel.reason
        );
        app = app.merge(router_with(model, opts));
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
