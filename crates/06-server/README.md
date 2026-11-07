# 06-server — the product server: `kt-server`, the OpenAI-compatible API with the native verify and runtime endpoints.

**Serving / security:**
- It binds the loopback address (`127.0.0.1:8080`) and has **no authentication**. Put an authenticating reverse
  proxy in front before exposing it; on any other address it warns at start.
- `POST /kt/v1/runtime` switches the **process-global** backend and thread cap for every client. It is **off by
  default** (403, `runtime_switch_disabled`) and enabled only by `--allow-runtime-switch`
  (`kt_server::ServeOptions`). Keep it off on any exposed server.
- See the README's "Serving / security" section.
