//! Does `kt_convert::text` lay out a GLM-4.5 chat exactly as transformers' `apply_chat_template` does? For
//! each chat in a reference made with transformers, it renders the prompt with the checkpoint's template (with or
//! without thinking), encodes it, and compares the text and the ids. With `--plant` the thinking flag is flipped; then
//! every case must differ, which shows the comparison can fail.
//!
//!   cargo run --release -p kt-convert --example chat_check -- <GLM-4.5-Air dir> <chat_reference.json> [--plant]

use kt_convert::text::{ChatTemplate, Message, Tokenizer};
use serde_json::Value;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        eprintln!("usage: chat_check <dir> <chat_reference.json> [--plant]");
        std::process::exit(2);
    }
    let plant = a.iter().any(|s| s == "--plant");
    let mut tok = Tokenizer::from_dir(&a[1]).expect("tokenizer");
    assert!(matches!(tok.chat_template(), ChatTemplate::Glm4 { .. }), "not a GLM-4.5 tokenizer");
    let reference: Value = serde_json::from_str(&std::fs::read_to_string(&a[2]).expect("reference")).expect("json");
    let mut all_ok = true;
    for c in reference["cases"].as_array().expect("cases") {
        let thinking = c["thinking"].as_bool().expect("thinking") != plant;
        tok.set_chat_template(ChatTemplate::Glm4 { thinking });
        let msgs: Vec<Message> = c["messages"]
            .as_array()
            .expect("messages")
            .iter()
            .map(|m| Message {
                role: m["role"].as_str().unwrap().into(),
                content: m["content"].as_str().unwrap().into(),
            })
            .collect();
        let text = tok.chat_prompt(&msgs);
        let ids = tok.encode(&text, false).expect("encode");
        let want: Vec<u32> = c["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let ok = text == c["text"].as_str().unwrap() && ids == want;
        all_ok &= ok;
        println!(
            "{:<12} thinking={:<5} {} ids · {}",
            c["name"].as_str().unwrap(),
            c["thinking"],
            ids.len(),
            if ok { "IDENTICAL to transformers" } else { "DIFFERS" }
        );
    }
    println!("verdict: {}", if all_ok { "every chat identical to transformers" } else { "A DIFFERENCE" });
    if !all_ok {
        std::process::exit(1);
    }
}
