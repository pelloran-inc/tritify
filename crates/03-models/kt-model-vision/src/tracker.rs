//! The video-tracker memory-bank bookkeeping — the stateful "glue" that assembles the per-frame memory the
//! memory attention conditions on. This is deterministic indexing + a couple of small positional transforms
//! (not tritify data-path compute): a `FrameMemory` is stored per processed frame, and for the current frame the
//! bank concatenates the spatial memory of the relevant past frames (each with its temporal-slot encoding) and
//! the object-pointer tokens (each 256-d pointer split into 4×64-d tokens, with a projected 1-D temporal sine
//! encoding). Faithful to `_gather_memory_frame_outputs` / `_build_memory_attention_inputs` /
//! `_get_object_pointers` / `_process_object_pointers` in `modeling_sam2_video.py`.
//!
//! Composed with the verified compute links (image SAM → memory encoder → this assembly → memory attention →
//! mask decoder), this makes a fully standalone zero-float video tracker.

use crate::hiera::WeightStore;

/// What one processed frame contributes to the bank: its spatial memory (`[hw, mem_dim]` tokens + positional
/// encoding) and its object pointer (`[hidden]`).
#[derive(Clone)]
pub struct FrameMemory {
    pub maskmem: Vec<f32>,        // [hw, mem_dim]
    pub maskmem_pos: Vec<f32>,    // [hw, mem_dim]
    pub object_pointer: Vec<f32>, // [hidden]
    pub is_conditioning: bool,
    pub frame_idx: usize,
}

/// 1-D sine positional embedding (`get_1d_sine_pe`): `dim`-vector for a scalar position.
fn get_1d_sine_pe(pos: f64, dim: usize, temperature: f64) -> Vec<f64> {
    let pe_dim = dim / 2;
    let mut out = vec![0.0f64; dim];
    for k in 0..pe_dim {
        let dim_t = temperature.powf(2.0 * (k / 2) as f64 / pe_dim as f64);
        let a = pos / dim_t;
        out[k] = a.sin();
        out[pe_dim + k] = a.cos();
    }
    out
}

pub struct VideoMemoryBank {
    temporal_pos: Vec<f32>, // [num_maskmem, mem_dim] memory_temporal_positional_encoding
    ptr_proj_w: Vec<f32>,   // [mem_dim, hidden] temporal_positional_encoding_projection_layer
    ptr_proj_b: Vec<f32>,   // [mem_dim]
    num_maskmem: usize,
    mem_dim: usize,
    hidden: usize,
    max_obj_ptr: usize,
}

impl VideoMemoryBank {
    pub fn from_store(
        store: &WeightStore,
        num_maskmem: usize,
        mem_dim: usize,
        hidden: usize,
        max_obj_ptr: usize,
    ) -> Self {
        VideoMemoryBank {
            temporal_pos: store.w("memory_temporal_positional_encoding").to_vec(), // [num_maskmem,1,1,mem_dim] flat
            ptr_proj_w: store.w("temporal_positional_encoding_projection_layer.weight").to_vec(),
            ptr_proj_b: store.w("temporal_positional_encoding_projection_layer.bias").to_vec(),
            num_maskmem,
            mem_dim,
            hidden,
            max_obj_ptr,
        }
    }

    /// Projected temporal positional encoding for one object pointer at `temporal_offset` (normalized by
    /// `max_temporal_diff`): `get_1d_sine_pe(norm, hidden)` → Linear(hidden→mem_dim).
    fn pointer_tpos(&self, temporal_offset: i64, max_temporal_diff: f64) -> Vec<f32> {
        let norm = temporal_offset as f64 / max_temporal_diff;
        let pe = get_1d_sine_pe(norm, self.hidden, 10000.0);
        (0..self.mem_dim)
            .map(|o| {
                let acc: f64 = (0..self.hidden).map(|i| pe[i] * self.ptr_proj_w[o * self.hidden + i] as f64).sum();
                (acc + self.ptr_proj_b[o] as f64) as f32
            })
            .collect()
    }

    /// Assemble the memory + positional embeddings the memory attention sees at `current_frame_idx`, from the
    /// stored frames, plus `num_object_pointer_tokens`. `hw` = spatial tokens per frame (feat²).
    ///
    /// Faithful minimal bank (stride-1, as the reference supports): conditioning frames at temporal offset 0
    /// (temporal slot `num_maskmem-1`), then the most recent `num_maskmem-1` non-conditioning frames; object
    /// pointers from the conditioning frames and recent frames (each split 1→`hidden/mem_dim` tokens).
    pub fn assemble(
        &self,
        frames: &[FrameMemory],
        current_frame_idx: usize,
        num_total_frames: usize,
        hw: usize,
    ) -> (Vec<f32>, Vec<f32>, usize) {
        let mut mem: Vec<f32> = Vec::new();
        let mut pos: Vec<f32> = Vec::new();

        // --- spatial memory: conditioning frames (offset 0 → slot num_maskmem-1), then recent non-cond frames ---
        let mut spatial: Vec<(usize, &FrameMemory)> = Vec::new(); // (relative_offset, frame)
        for f in frames.iter().filter(|f| f.is_conditioning) {
            spatial.push((0, f));
        }
        for off in (1..self.num_maskmem).rev() {
            if let Some(prev) = current_frame_idx.checked_sub(off) {
                if let Some(f) = frames.iter().find(|f| !f.is_conditioning && f.frame_idx == prev) {
                    spatial.push((off, f));
                }
            }
        }
        for (off, f) in &spatial {
            let slot = if *off == 0 { self.num_maskmem - 1 } else { off - 1 };
            let tpos = &self.temporal_pos[slot * self.mem_dim..(slot + 1) * self.mem_dim];
            mem.extend_from_slice(&f.maskmem);
            for t in 0..hw {
                for (c, &tp) in tpos[..self.mem_dim].iter().enumerate() {
                    pos.push(f.maskmem_pos[t * self.mem_dim + c] + tp);
                }
            }
        }

        // --- object pointers: conditioning + recent frames (temporal_offset = current - frame_idx) ---
        let max_tdiff = (num_total_frames.min(self.max_obj_ptr) as i64 - 1).max(1) as f64;
        let num_splits = self.hidden / self.mem_dim;
        let mut num_ptr_tokens = 0usize;
        let mut ptr_frames: Vec<&FrameMemory> =
            frames.iter().filter(|f| f.is_conditioning && f.frame_idx <= current_frame_idx).collect();
        for off in 1..self.max_obj_ptr {
            if let Some(prev) = current_frame_idx.checked_sub(off) {
                if let Some(f) = frames.iter().find(|f| !f.is_conditioning && f.frame_idx == prev) {
                    ptr_frames.push(f);
                }
            }
        }
        for f in ptr_frames {
            let toff = current_frame_idx as i64 - f.frame_idx as i64;
            let ptos = self.pointer_tpos(toff, max_tdiff);
            for s in 0..num_splits {
                mem.extend_from_slice(&f.object_pointer[s * self.mem_dim..(s + 1) * self.mem_dim]);
                pos.extend_from_slice(&ptos);
                num_ptr_tokens += 1;
            }
        }
        (mem, pos, num_ptr_tokens)
    }
}
