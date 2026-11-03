//! Text in and out of a language model. It reads the checkpoint's `tokenizer.json` with Hugging Face's own
//! `tokenizers` crate, so a prompt becomes exactly the ids the model was trained on. It also finds the model's
//! end-of-sequence ids and lays out a chat as a prompt.
//!
//! The chat layout is ChatML (`<|im_start|>role\ncontent<|im_end|>\n`, then an open assistant turn) when the vocabulary
//! has those tokens, and otherwise plain `role: content` lines. A checkpoint's own Jinja chat template is not
//! interpreted, so model-specific defaults it would add (for example a default system prompt) are not added.

use serde_json::Value;
use std::path::Path;

/// One chat message: `role` is `system`, `user` or `assistant`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// How a chat is laid out as a prompt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChatTemplate {
    /// `<|im_start|>role\ncontent<|im_end|>\n` per message, then `<|im_start|>assistant\n`.
    ChatMl,
    /// `role: content` per line, then `assistant:`.
    Plain,
    /// GLM-4.5's template (`chat_template.jinja`) for system, user and assistant turns without tools:
    /// `[gMASK]<sop>`, then `<|system|>\n…`, `<|user|>\n…`, `<|assistant|>\n<think>…</think>\n…` per message, then
    /// `<|assistant|>`. Without thinking (transformers' `enable_thinking=False`), each user turn ends in `/nothink` and
    /// the open assistant turn starts with an empty `<think></think>`, so the model answers directly.
    Glm4 { thinking: bool },
}

/// A checkpoint's tokenizer, its end-of-sequence ids and its chat layout.
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    eos: Vec<u32>,
    chat: ChatTemplate,
}

impl Tokenizer {
    /// Load `tokenizer.json` from a checkpoint directory. The end-of-sequence ids come from `generation_config.json`
    /// (`eos_token_id`, a number or a list), else from `tokenizer_config.json` (`eos_token`).
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Tokenizer, String> {
        let dir = dir.as_ref();
        let mut t = Tokenizer::from_file(dir.join("tokenizer.json"))?;
        let json =
            |name: &str| -> Option<Value> { serde_json::from_str(&std::fs::read_to_string(dir.join(name)).ok()?).ok() };
        let mut eos: Vec<u32> = match json("generation_config.json").and_then(|g| g.get("eos_token_id").cloned()) {
            Some(Value::Number(n)) => n.as_u64().map(|v| vec![v as u32]).unwrap_or_default(),
            Some(Value::Array(a)) => a.iter().filter_map(|v| v.as_u64().map(|v| v as u32)).collect(),
            _ => Vec::new(),
        };
        if eos.is_empty() {
            let name = json("tokenizer_config.json").and_then(|c| match c.get("eos_token")? {
                Value::String(s) => Some(s.clone()),
                Value::Object(o) => o.get("content")?.as_str().map(String::from),
                _ => None,
            });
            eos.extend(name.and_then(|n| t.inner.token_to_id(&n)));
        }
        t.eos = eos;
        Ok(t)
    }

    /// Load a `tokenizer.json` file; no end-of-sequence ids are known yet (see [`Tokenizer::from_dir`]).
    pub fn from_file(path: impl AsRef<Path>) -> Result<Tokenizer, String> {
        let inner = tokenizers::Tokenizer::from_file(path.as_ref())
            .map_err(|e| format!("tokenizer {}: {e}", path.as_ref().display()))?;
        Ok(Tokenizer::from_inner(inner))
    }

    /// Parse a `tokenizer.json` document held in memory.
    pub fn from_json(json: &str) -> Result<Tokenizer, String> {
        let inner: tokenizers::Tokenizer = json.parse().map_err(|e| format!("tokenizer.json: {e}"))?;
        Ok(Tokenizer::from_inner(inner))
    }

    fn from_inner(inner: tokenizers::Tokenizer) -> Tokenizer {
        let has = |t: &str| inner.token_to_id(t).is_some();
        let chat = if has("<|im_start|>") && has("<|im_end|>") {
            ChatTemplate::ChatMl
        } else if has("[gMASK]") && has("<sop>") && has("<|user|>") && has("<|assistant|>") {
            ChatTemplate::Glm4 { thinking: true }
        } else {
            ChatTemplate::Plain
        };
        Tokenizer { inner, eos: Vec::new(), chat }
    }

    /// Text → ids. `add_special` adds what the tokenizer's post-processor adds (a BOS token, for instance).
    pub fn encode(&self, text: &str, add_special: bool) -> Result<Vec<u32>, String> {
        Ok(self.inner.encode(text, add_special).map_err(|e| format!("encode: {e}"))?.get_ids().to_vec())
    }

    /// Ids → text, special tokens skipped.
    pub fn decode(&self, ids: &[u32]) -> Result<String, String> {
        self.inner.decode(ids, true).map_err(|e| format!("decode: {e}"))
    }

    /// Ids → text with the special tokens kept: `decode_raw(encode(s, false)) == s` for a byte-level tokenizer.
    pub fn decode_raw(&self, ids: &[u32]) -> Result<String, String> {
        self.inner.decode(ids, false).map_err(|e| format!("decode: {e}"))
    }

    /// The ids that end a generation.
    pub fn eos_ids(&self) -> &[u32] {
        &self.eos
    }

    /// Replace the end-of-sequence ids (a model declaration may name them).
    pub fn set_eos_ids(&mut self, ids: Vec<u32>) {
        self.eos = ids;
    }

    pub fn chat_template(&self) -> ChatTemplate {
        self.chat
    }

    /// Choose the chat layout (a model spec may name it).
    pub fn set_chat_template(&mut self, chat: ChatTemplate) {
        self.chat = chat;
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Lay out a chat as the prompt text the model continues (an open assistant turn at the end).
    pub fn chat_prompt(&self, messages: &[Message]) -> String {
        let mut s = String::new();
        match self.chat {
            ChatTemplate::ChatMl => {
                for m in messages {
                    s.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", m.role, m.content));
                }
                s.push_str("<|im_start|>assistant\n");
            }
            ChatTemplate::Plain => {
                for m in messages {
                    s.push_str(&format!("{}: {}\n", m.role, m.content));
                }
                s.push_str("assistant:");
            }
            ChatTemplate::Glm4 { thinking } => {
                let last_user = messages.iter().rposition(|m| m.role == "user");
                s.push_str("[gMASK]<sop>");
                for (i, m) in messages.iter().enumerate() {
                    match m.role.as_str() {
                        "system" => s.push_str(&format!("<|system|>\n{}", m.content)),
                        "user" => {
                            s.push_str(&format!("<|user|>\n{}", m.content));
                            if !thinking && !m.content.ends_with("/nothink") {
                                s.push_str("/nothink");
                            }
                        }
                        "assistant" => {
                            // a `<think>…</think>` in the content is its reasoning, kept only after the last user turn
                            let (mut reasoning, mut content) = ("", m.content.as_str());
                            if let Some((before, after)) = m.content.split_once("</think>") {
                                reasoning = before
                                    .trim_end_matches('\n')
                                    .rsplit("<think>")
                                    .next()
                                    .unwrap_or("")
                                    .trim_start_matches('\n');
                                content = after.trim_start_matches('\n');
                            }
                            s.push_str("<|assistant|>");
                            if last_user.is_some_and(|u| i > u) && !reasoning.is_empty() {
                                s.push_str(&format!("\n<think>{}</think>", reasoning.trim()));
                            } else {
                                s.push_str("\n<think></think>");
                            }
                            if !content.trim().is_empty() {
                                s.push_str(&format!("\n{}", content.trim()));
                            }
                        }
                        _ => {}
                    }
                }
                s.push_str("<|assistant|>");
                if !thinking {
                    s.push_str("\n<think></think>");
                }
            }
        }
        s
    }
}

/// Turns a growing list of generated ids into text deltas for streaming. A token may end inside a multi-byte
/// character; the delta waits until the character is complete, so no replacement character is ever sent.
pub struct TextStream<'t> {
    tok: &'t Tokenizer,
    ids: Vec<u32>,
    sent: usize,
}

impl<'t> TextStream<'t> {
    pub fn new(tok: &'t Tokenizer) -> TextStream<'t> {
        TextStream { tok, ids: Vec::new(), sent: 0 }
    }

    /// Add one id; returns the new text it completes (possibly empty).
    pub fn push(&mut self, id: u32) -> Result<String, String> {
        self.ids.push(id);
        let text = self.tok.decode(&self.ids)?;
        if text.ends_with('\u{FFFD}') || text.len() < self.sent || !text.is_char_boundary(self.sent) {
            return Ok(String::new());
        }
        let delta = text[self.sent..].to_string();
        self.sent = text.len();
        Ok(delta)
    }

    /// The text not yet sent: whatever a final incomplete character held back. Call once at the end.
    pub fn finish(&mut self) -> Result<String, String> {
        let text = self.tok.decode(&self.ids)?;
        let rest = text.get(self.sent..).unwrap_or_default().to_string();
        self.sent = text.len();
        Ok(rest)
    }
}

#[cfg(test)]
mod tests;
