//! Hugging Face loader: `config.json` + `.safetensors` → a [`ModelSource`] the engine can consume.
//!
//! The safetensors format is read with a small dependency-free parser (8-byte LE header length · a JSON
//! header mapping each tensor name to `{dtype, shape, data_offsets}` · a concatenated data buffer). Weights
//! are dequantized to `f32` on read (F32 / F16 / BF16 — the dtypes real decoder checkpoints use).

use kt_core::plugin::ModelSource;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

#[derive(Clone, Copy)]
enum Dtype {
    F32,
    F16,
    Bf16,
}

struct Entry {
    dtype: Dtype,
    shape: Vec<usize>,
    start: usize,
    end: usize,
    shard: usize, // which shard's data buffer this tensor lives in
}

/// A `.safetensors` shard: either held in memory, or read lazily from disk (so a 13 GB sharded model is never all in
/// RAM at once). `base` is where the tensor-data buffer begins (`8 + header_len`); tensor bytes are read on demand.
enum Shard {
    Mem { data: Vec<u8>, base: usize },
    File { path: String, base: usize },
}

impl Shard {
    /// Read `[start, end)` of this shard's data buffer (relative to `base`) into a byte vector.
    fn read(&self, start: usize, end: usize) -> Option<Vec<u8>> {
        match self {
            Shard::Mem { data, base } => data.get(base + start..base + end).map(|s| s.to_vec()),
            Shard::File { path, base } => {
                let mut f = File::open(path).ok()?;
                f.seek(SeekFrom::Start((base + start) as u64)).ok()?;
                let mut buf = vec![0u8; end.checked_sub(start)?];
                f.read_exact(&mut buf).ok()?;
                Some(buf)
            }
        }
    }
}

/// A model loaded from a Hugging Face directory (or in-memory parts) — single-file OR sharded.
pub struct HfModelSource {
    config: Value,
    index: BTreeMap<String, Entry>,
    shards: Vec<Shard>,
}

/// Parse one safetensors file's header into its (name → entry) map (offsets relative to the shard's data base)
/// plus the data base. `shard` is stamped into every entry so the source knows which buffer to slice.
fn parse_shard(bytes: &[u8], shard: usize) -> Result<(Vec<(String, Entry)>, usize), String> {
    if bytes.len() < 8 {
        return Err("safetensors: file shorter than 8-byte header length".into());
    }
    let hlen = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let header_end = 8usize.checked_add(hlen).ok_or("safetensors: header length overflow")?;
    if bytes.len() < header_end {
        return Err("safetensors: header truncated".into());
    }
    let header: Value =
        serde_json::from_slice(&bytes[8..header_end]).map_err(|e| format!("safetensors header: {e}"))?;
    let obj = header.as_object().ok_or("safetensors: header is not an object")?;
    let mut out = Vec::new();
    for (name, v) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype = match v["dtype"].as_str().unwrap_or("") {
            "F32" => Dtype::F32,
            "F16" => Dtype::F16,
            "BF16" => Dtype::Bf16,
            other => return Err(format!("unsupported dtype {other:?} for tensor {name}")),
        };
        let shape = v["shape"]
            .as_array()
            .ok_or_else(|| format!("tensor {name}: no shape"))?
            .iter()
            .map(|x| x.as_u64().unwrap_or(0) as usize)
            .collect();
        let offs = v["data_offsets"].as_array().ok_or_else(|| format!("tensor {name}: no data_offsets"))?;
        let start = offs.first().and_then(|x| x.as_u64()).ok_or("bad data_offsets")? as usize;
        let end = offs.get(1).and_then(|x| x.as_u64()).ok_or("bad data_offsets")? as usize;
        out.push((name.clone(), Entry { dtype, shape, start, end, shard }));
    }
    Ok((out, header_end))
}

impl HfModelSource {
    /// Load from an in-memory `config.json` string and the bytes of a single `.safetensors` file.
    pub fn from_parts(config_json: &str, safetensors: Vec<u8>) -> Result<Self, String> {
        let config: Value = serde_json::from_str(config_json).map_err(|e| format!("config.json: {e}"))?;
        let (entries, base) = parse_shard(&safetensors, 0)?;
        let index = entries.into_iter().collect();
        Ok(HfModelSource { config, index, shards: vec![Shard::Mem { data: safetensors, base }] })
    }

    /// Load from a directory: `config.json` + either `model.safetensors` (single) or a sharded checkpoint
    /// (`model.safetensors.index.json` → `weight_map` → per-shard `.safetensors` files).
    pub fn from_dir(dir: &str) -> Result<Self, String> {
        let cfg = std::fs::read_to_string(format!("{dir}/config.json")).map_err(|e| format!("config.json: {e}"))?;
        let single = format!("{dir}/model.safetensors");
        if std::path::Path::new(&single).exists() {
            return Self::from_parts(&cfg, std::fs::read(&single).map_err(|e| format!("model.safetensors: {e}"))?);
        }
        // sharded: read the index, load each distinct shard file once, build a global tensor index.
        let idx_path = format!("{dir}/model.safetensors.index.json");
        let idx: Value = serde_json::from_str(
            &std::fs::read_to_string(&idx_path).map_err(|e| format!("no model.safetensors and no index: {e}"))?,
        )
        .map_err(|e| format!("index.json: {e}"))?;
        let wm = idx["weight_map"].as_object().ok_or("index.json: no weight_map")?;
        // distinct shard filenames, in a stable order
        let mut files: Vec<String> = wm.values().filter_map(|v| v.as_str().map(String::from)).collect();
        files.sort();
        files.dedup();
        let config: Value = serde_json::from_str(&cfg).map_err(|e| format!("config.json: {e}"))?;
        let mut shards = Vec::with_capacity(files.len());
        let mut index: BTreeMap<String, Entry> = BTreeMap::new();
        for (si, fname) in files.iter().enumerate() {
            let path = format!("{dir}/{fname}");
            // read ONLY the header (8-byte length + JSON) to build the index; tensor data stays on disk (lazy).
            let mut f = File::open(&path).map_err(|e| format!("{fname}: {e}"))?;
            let mut lb = [0u8; 8];
            f.read_exact(&mut lb).map_err(|e| format!("{fname}: {e}"))?;
            let hlen = u64::from_le_bytes(lb) as usize;
            let mut hdr = vec![0u8; 8 + hlen];
            f.seek(SeekFrom::Start(0)).map_err(|e| format!("{fname}: {e}"))?;
            f.read_exact(&mut hdr).map_err(|e| format!("{fname}: {e}"))?;
            let (entries, base) = parse_shard(&hdr, si)?;
            for (name, e) in entries {
                index.insert(name, e);
            }
            shards.push(Shard::File { path, base });
        }
        Ok(HfModelSource { config, index, shards })
    }

    pub fn tensor_names(&self) -> Vec<String> {
        self.index.keys().cloned().collect()
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    let v = match exp {
        0 => (mant as f32) * 2f32.powi(-24), // subnormal (and zero)
        0x1f => {
            if mant == 0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => (1.0 + mant as f32 / 1024.0) * 2f32.powi(exp as i32 - 15),
    };
    if sign == 1 {
        -v
    } else {
        v
    }
}

fn to_f32(dtype: Dtype, bytes: &[u8]) -> Vec<f32> {
    match dtype {
        Dtype::F32 => bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        Dtype::Bf16 => {
            bytes.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect()
        }
        Dtype::F16 => bytes.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
    }
}

impl ModelSource for HfModelSource {
    fn hf_arch_ids(&self) -> Vec<String> {
        self.config["architectures"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default()
    }

    fn config_usize(&self, key: &str) -> Option<usize> {
        self.config[key].as_u64().map(|v| v as usize)
    }

    fn config_f64(&self, key: &str) -> Option<f64> {
        self.config[key].as_f64()
    }

    fn config_bool(&self, key: &str) -> Option<bool> {
        self.config[key].as_bool()
    }

    fn tensor(&self, name: &str) -> Option<(Vec<f32>, Vec<usize>)> {
        let e = self.index.get(name)?;
        let s = self.shards.get(e.shard)?;
        let bytes = s.read(e.start, e.end)?; // in-memory slice, or a lazy on-disk read
        Some((to_f32(e.dtype, &bytes), e.shape.clone()))
    }

    fn tensor_names(&self) -> Vec<String> {
        self.index.keys().cloned().collect()
    }
}
