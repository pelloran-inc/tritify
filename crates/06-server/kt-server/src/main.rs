//! `kt-server (--model <dir> | --spec <spec.json>) [--listen 127.0.0.1:8080] [--backend auto|cpu|cpu:<variant>|gpu|gpu:<i>|cuda|cuda:<i>]
//! [--threads N] [--telemetry FILE] [--allow-runtime-switch]`, or `kt-server --telemetry FILE [--listen …]` for the
//! fleet stream alone.
//!
//! Serve one model (a decoder or a MoE decoder) over the OpenAI-compatible API: a Hugging Face checkpoint directory,
//! or any checkpoint through its model spec.
//!
//! **Serving / security:**
//! - It binds the loopback address by default, and there is **no authentication**. Do not expose it beyond the
//!   machine without an authenticating reverse proxy in front. On any other address it warns on stderr.
//! - `--allow-runtime-switch` enables `POST /kt/v1/runtime`, which switches the **process-global** backend and thread
//!   cap for every client. It is off by default, and a POST then answers 403. Keep it off on an exposed server.

fn arg(a: &[String], k: &str) -> Option<String> {
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned()
}

#[tokio::main]
async fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (dir, spec, telemetry) = (arg(&a, "--model"), arg(&a, "--spec"), arg(&a, "--telemetry"));
    if (dir.is_some() && spec.is_some()) || (dir.is_none() && spec.is_none() && telemetry.is_none()) {
        eprintln!(
            "usage: kt-server (--model <dir> | --spec <spec.json>) [--listen 127.0.0.1:8080] \
             [--backend auto|cpu|cpu:<variant>|gpu|gpu:<i>|cuda|cuda:<i>] [--threads N] [--telemetry FILE] \
             [--allow-runtime-switch]\n\
             \x20      kt-server --telemetry FILE [--listen 127.0.0.1:8080]"
        );
        std::process::exit(2);
    }
    let fail = |e: String| -> ! {
        eprintln!("kt-server: {e}");
        std::process::exit(1)
    };
    let listen = arg(&a, "--listen")
        .unwrap_or_else(|| "127.0.0.1:8080".into())
        .parse()
        .unwrap_or_else(|e| fail(format!("--listen: {e}")));
    let backend =
        kt_server::parse_backend(&arg(&a, "--backend").unwrap_or_else(|| "auto".into())).unwrap_or_else(|e| fail(e));
    let max_threads = arg(&a, "--threads").map(|t| t.parse().unwrap_or_else(|e| fail(format!("--threads: {e}"))));
    let model = match (dir, spec) {
        (Some(d), _) => Some(kt_server::ServerModel::load(&d)),
        (_, Some(s)) => Some(kt_convert::spec::ModelSpec::load(&s).and_then(|s| kt_server::ServerModel::from_spec(&s))),
        _ => None,
    }
    .map(|m| m.unwrap_or_else(|e| fail(e)));
    let runtime = kt_runtime::RuntimeConfig { backend, max_threads, ..Default::default() };
    let opts = kt_server::ServeOptions { allow_runtime_switch: a.iter().any(|x| x == "--allow-runtime-switch") };
    if let Err(e) = kt_server::serve_with(model, telemetry.map(Into::into), listen, &runtime, opts).await {
        fail(e);
    }
}
