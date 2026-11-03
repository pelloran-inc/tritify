//! Does `kt_convert::text` tokenize exactly like Hugging Face? For each corpus string it compares the ids
//! (with and without the special-token post-processing) against reference ids made by HF's Python `tokenizers`
//! (`make_tokenizer_reference.py`), checks the raw round trip `decode_raw(encode(s)) == s`, and checks that streaming
//! the ids one at a time reassembles the decoded text. With `--plant`, the first 100 BPE merges are removed from the
//! tokenizer; the ids must then differ for every checkpoint, which shows the comparison can fail. (Removing only the
//! first merge, `Ġ Ġ`, changes nothing for OLMoE: its tokenizer carries whitespace runs as added tokens.)
//!
//!   cargo run --release -p kt-convert --example tokenizer_check -- <models dir> <corpus.json> <reference.json> [--plant]

use kt_convert::text::{TextStream, Tokenizer};
use serde_json::Value;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!("usage: tokenizer_check <models dir> <corpus.json> <reference.json> [--plant]");
        std::process::exit(2);
    }
    let plant = a.iter().any(|s| s == "--plant");
    let read = |p: &str| std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{p}: {e}"));
    let corpus: Vec<String> = serde_json::from_str(&read(&a[2])).expect("corpus");
    let reference: Value = serde_json::from_str(&read(&a[3])).expect("reference");
    let mut all_ok = true;
    for (name, r) in reference["models"].as_object().expect("models") {
        let path = format!("{}/{name}/tokenizer.json", a[1]);
        let mut json: Value = serde_json::from_str(&read(&path)).expect("tokenizer.json");
        if plant {
            json["model"]["merges"].as_array_mut().expect("merges").drain(..100);
        }
        let tok = Tokenizer::from_json(&json.to_string()).expect("load");
        let want = |key: &str, i: usize| -> Vec<u32> {
            r[key][i].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect()
        };
        let (mut diff, mut special_diff, mut roundtrip, mut stream) = (0, 0, 0, 0);
        for (i, s) in corpus.iter().enumerate() {
            let ids = tok.encode(s, false).unwrap();
            diff += usize::from(ids != want("ids", i));
            special_diff += usize::from(tok.encode(s, true).unwrap() != want("ids_special", i));
            roundtrip += usize::from(tok.decode_raw(&ids).unwrap() != *s);
            let mut st = TextStream::new(&tok);
            let streamed: String = ids.iter().map(|&id| st.push(id).unwrap()).collect();
            stream += usize::from(streamed != tok.decode(&ids).unwrap() || streamed.contains('\u{FFFD}'));
        }
        let ok = diff + special_diff + roundtrip + stream == 0;
        all_ok &= ok;
        println!(
            "{name:<24} {} strings · ids differ {diff} · ids (special) differ {special_diff} · round trip fails \
             {roundtrip} · stream fails {stream} · {}",
            corpus.len(),
            if ok { "IDENTICAL to HF" } else { "DIFFERS" }
        );
    }
    println!("verdict: {}", if all_ok { "every id identical to Hugging Face" } else { "A DIFFERENCE" });
    if !all_ok {
        std::process::exit(1);
    }
}
