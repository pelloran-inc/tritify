//! Unit tests on a tiny word-level `tokenizer.json`, so they run anywhere. The comparison against Hugging Face's
//! Python tokenizer on the real checkpoints is the `tokenizer_check` example.

use super::*;

pub(crate) const TINY: &str = r#"{
  "version": "1.0", "truncation": null, "padding": null,
  "added_tokens": [
    {"id": 0, "content": "<|im_start|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 1, "content": "<|im_end|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "normalizer": null, "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null, "decoder": null,
  "model": {"type": "WordLevel", "unk_token": "[UNK]",
            "vocab": {"<|im_start|>": 0, "<|im_end|>": 1, "user": 2, "assistant": 3, "hello": 4, "world": 5, "[UNK]": 6}}
}"#;

#[test]
fn encodes_decodes_and_finds_chatml() {
    let t = Tokenizer::from_json(TINY).unwrap();
    assert_eq!(t.encode("hello world", false).unwrap(), vec![4, 5]);
    assert_eq!(t.encode("hello mars", false).unwrap(), vec![4, 6]);
    assert_eq!(t.decode(&[4, 5]).unwrap(), "hello world");
    assert_eq!(t.decode(&[0, 4, 1]).unwrap(), "hello", "special tokens are skipped");
    assert_eq!(t.chat_template(), ChatTemplate::ChatMl);
    let p = t.chat_prompt(&[Message { role: "user".into(), content: "hello".into() }]);
    assert_eq!(p, "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n");
    assert_eq!(t.encode(&p, false).unwrap(), vec![0, 2, 4, 1, 0, 3]);
}

#[test]
fn a_vocabulary_without_chatml_gets_plain_lines() {
    let t = Tokenizer::from_json(&TINY.replace("<|im_start|>", "<s>")).unwrap();
    assert_eq!(t.chat_template(), ChatTemplate::Plain);
    assert_eq!(t.chat_prompt(&[Message { role: "user".into(), content: "hi".into() }]), "user: hi\nassistant:");
}

/// GLM-4.5's layout, character for character as transformers' `apply_chat_template` renders the checkpoint's
/// chat_template.jinja (the strings below are its output, recorded with their ids).
#[test]
fn glm4_chats_are_laid_out_as_transformers_lays_them_out() {
    let glm = TINY
        .replace("<|im_start|>", "[gMASK]")
        .replace("<|im_end|>", "<sop>")
        .replace("\"user\": 2", "\"<|user|>\": 2")
        .replace("\"assistant\": 3", "\"<|assistant|>\": 3");
    let mut t = Tokenizer::from_json(&glm).unwrap();
    assert_eq!(t.chat_template(), ChatTemplate::Glm4 { thinking: true });
    let m = |r: &str, c: &str| Message { role: r.into(), content: c.into() };
    let q = "What is the capital of France? Answer in one sentence.";
    let cases: [(Vec<Message>, &str, &str); 3] = [
        (
            vec![m("user", q)],
            "[gMASK]<sop><|user|>\nWhat is the capital of France? Answer in one sentence.<|assistant|>",
            "[gMASK]<sop><|user|>\nWhat is the capital of France? Answer in one sentence./nothink<|assistant|>\n<think></think>",
        ),
        (
            vec![m("system", "You are a concise assistant."), m("user", q)],
            "[gMASK]<sop><|system|>\nYou are a concise assistant.<|user|>\nWhat is the capital of France? Answer in one sentence.<|assistant|>",
            "[gMASK]<sop><|system|>\nYou are a concise assistant.<|user|>\nWhat is the capital of France? Answer in one sentence./nothink<|assistant|>\n<think></think>",
        ),
        (
            vec![m("user", "Hi"), m("assistant", "Hello! How can I help?"), m("user", "Name a color.")],
            "[gMASK]<sop><|user|>\nHi<|assistant|>\n<think></think>\nHello! How can I help?<|user|>\nName a color.<|assistant|>",
            "[gMASK]<sop><|user|>\nHi/nothink<|assistant|>\n<think></think>\nHello! How can I help?<|user|>\nName a color./nothink<|assistant|>\n<think></think>",
        ),
    ];
    for (msgs, thinking, direct) in cases {
        t.set_chat_template(ChatTemplate::Glm4 { thinking: true });
        assert_eq!(t.chat_prompt(&msgs), thinking);
        t.set_chat_template(ChatTemplate::Glm4 { thinking: false });
        assert_eq!(t.chat_prompt(&msgs), direct);
    }
}

#[test]
fn end_of_sequence_ids_come_from_the_checkpoint() {
    let dir = std::env::temp_dir().join(format!("kt-text-eos-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("tokenizer.json"), TINY).unwrap();
    std::fs::write(dir.join("generation_config.json"), r#"{"eos_token_id": [1, 5]}"#).unwrap();
    assert_eq!(Tokenizer::from_dir(&dir).unwrap().eos_ids(), &[1, 5]);
    std::fs::write(dir.join("generation_config.json"), r#"{"eos_token_id": 1}"#).unwrap();
    assert_eq!(Tokenizer::from_dir(&dir).unwrap().eos_ids(), &[1]);
    std::fs::remove_file(dir.join("generation_config.json")).unwrap();
    std::fs::write(dir.join("tokenizer_config.json"), r#"{"eos_token": "<|im_end|>"}"#).unwrap();
    assert_eq!(Tokenizer::from_dir(&dir).unwrap().eos_ids(), &[1]);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn the_stream_reassembles_the_text() {
    let t = Tokenizer::from_json(TINY).unwrap();
    let mut s = TextStream::new(&t);
    let text: String = [4u32, 5, 4].iter().map(|&id| s.push(id).unwrap()).collect();
    assert_eq!(text, t.decode(&[4, 5, 4]).unwrap());
}
