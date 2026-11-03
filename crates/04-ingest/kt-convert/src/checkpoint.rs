//! A checkpoint's tensors from the `.safetensors` headers alone: name, dtype, shape and stored bytes, without loading
//! any data. Each file starts with an 8-byte little-endian header length, then a JSON map of
//! `name → {dtype, shape, data_offsets}`. The fleet telemetry counts weight bytes from this.
//!
//! A sharded checkpoint's `model.safetensors.index.json` may carry a `total_size` that is not a byte count: GLM-4.5-Air's
//! equals its parameter count. These numbers come from the shard headers themselves.

use serde_json::Value;
use std::fs;
use std::path::Path;

/// One tensor as its header declares it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    /// The bytes the file stores for it (`data_offsets[1] − data_offsets[0]`).
    pub stored_bytes: u64,
}

impl TensorInfo {
    /// Its element count, the product of the shape.
    pub fn params(&self) -> u64 {
        self.shape.iter().product()
    }
}

/// Every tensor of the `.safetensors` files in `dir`, files in name order and tensors in header order. A file whose
/// header does not parse is skipped.
pub fn tensors(dir: &str) -> Vec<TensorInfo> {
    let mut files: Vec<_> = match fs::read_dir(Path::new(dir)) {
        Ok(e) => e.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "safetensors")).collect(),
        Err(_) => return Vec::new(),
    };
    files.sort();
    files.iter().flat_map(|p| header(p).unwrap_or_default()).collect()
}

/// The tensors one file's header declares.
pub fn header(path: &Path) -> Option<Vec<TensorInfo>> {
    use std::io::Read;
    let mut f = fs::File::open(path).ok()?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len).ok()?;
    let mut buf = vec![0u8; usize::try_from(u64::from_le_bytes(len)).ok()?];
    f.read_exact(&mut buf).ok()?;
    let Value::Object(hdr) = serde_json::from_slice::<Value>(&buf).ok()? else { return None };
    let mut out = Vec::new();
    for (name, v) in &hdr {
        if name == "__metadata__" {
            continue;
        }
        let shape: Vec<u64> = v.get("shape")?.as_array()?.iter().filter_map(Value::as_u64).collect();
        let off: Vec<u64> = v.get("data_offsets")?.as_array()?.iter().filter_map(Value::as_u64).collect();
        let dtype = v.get("dtype").and_then(Value::as_str).unwrap_or("?").to_string();
        let stored_bytes = match off.as_slice() {
            [a, b] if b >= a => b - a,
            _ => return None,
        };
        out.push(TensorInfo { name: name.clone(), dtype, shape, stored_bytes });
    }
    Some(out)
}

/// The layer index of a tensor named `model.layers.<i>.…`, else `None`.
pub fn layer_of(name: &str) -> Option<usize> {
    name.strip_prefix("model.layers.")?.split('.').next()?.parse().ok()
}

/// Parameters and stored bytes of the tensors a model built with `layers` decoder layers uses: every tensor outside
/// `model.layers`, and the layers `0..layers`. Layers at or beyond `layers` (a capped run, or a multi-token-prediction
/// layer past the decoder stack) are not counted.
pub fn totals(tensors: &[TensorInfo], layers: usize) -> (u64, u64) {
    tensors
        .iter()
        .filter(|t| layer_of(&t.name).is_none_or(|l| l < layers))
        .fold((0, 0), |(p, b), t| (p + t.params(), b + t.stored_bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn st(entries: &[(&str, &str, &[u64])]) -> Vec<u8> {
        let mut hdr = serde_json::Map::new();
        let mut off = 0u64;
        for (name, dtype, shape) in entries {
            let n: u64 = shape.iter().product::<u64>() * if *dtype == "F32" { 4 } else { 2 };
            hdr.insert(
                name.to_string(),
                serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [off, off + n]}),
            );
            off += n;
        }
        let h = serde_json::to_vec(&Value::Object(hdr)).unwrap();
        let mut out = (h.len() as u64).to_le_bytes().to_vec();
        out.extend(h);
        out.extend(vec![0u8; off as usize]);
        out
    }

    #[test]
    fn counts_the_used_layers_from_the_headers_alone() {
        let dir = std::env::temp_dir().join(format!("kt_checkpoint_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("a.safetensors"),
            st(&[("model.embed_tokens.weight", "BF16", &[10, 4]), ("model.layers.0.w", "BF16", &[4, 4])]),
        )
        .unwrap();
        fs::write(
            dir.join("b.safetensors"),
            st(&[("model.layers.1.w", "BF16", &[4, 4]), ("model.layers.2.bias", "F32", &[3])]),
        )
        .unwrap();
        let t = tensors(dir.to_str().unwrap());
        assert_eq!(t.len(), 4);
        assert_eq!(totals(&t, 1), (40 + 16, 80 + 32), "embeddings and layer 0");
        assert_eq!(totals(&t, 3), (40 + 16 + 16 + 3, 80 + 32 + 32 + 12), "every layer, an F32 tensor at 4 bytes");
        assert_eq!(layer_of("model.layers.46.eh_proj.weight"), Some(46));
        assert_eq!(layer_of("lm_head.weight"), None);
        fs::remove_dir_all(&dir).unwrap();
    }
}
