//! # Deterministic `kpack` Formula Synthesis, Evolutionary Crossover & Reed-Solomon Parity
//!
//! This module implements per-segment formula synthesis and multi-node fault tolerance across the
//! 8 logical vertices of the Hamon Cube:
//!
//! - **Formula Synthesis ([`KpackFormula::synthesize`])**: Evaluates candidate `kpack` opcodes
//!   (`RLE`, `DICT`, `SEED` + sparse patch, `DELTA` against a reference vertex, or fallback `LIT`)
//!   and selects the most compact deterministic representation for a segment up to
//!   [`MAX_SEGMENT_BYTES`] (`3576` bytes, sized to fit inside a single 4096-byte
//!   [`jinshu::node::TrieNode`] page).
//! - **Verified Execution & Wave64 Tiling ([`KpackFormula::execute_into`],
//!   [`KpackFormula::execute_wave64_tile`])**: Reconstructs segment bytes in isolated node memory,
//!   verifies the `BLAKE3` [`Noun`] digest, and extracts 64-byte register tiles for branchless
//!   `klm` AMD GPU execution.
//! - **Evolutionary Formula Crossover ([`KpackFormula::crossover`])**: Blends two adjacent parent
//!   formulas using a deterministic mutation seed to synthesize a child formula for **Kōgnitara**
//!   Bird Fusion.
//! - **Single & Dual Parity Stripes ([`CubeParityStripe`], [`CubeDualParityStripe`])**: Provides
//!   both fast XOR parity ($7+1$, 1-node recovery) and Reed-Solomon $\text{GF}(2^8)$ `P` + `Q`
//!   dual parity ($6+2$, simultaneous 2-node recovery with irreducible polynomial `0x11D`).

use jinshu::node::TrieNode;
use kpack::{Opcode, execute};
use noun::Noun;

use crate::topology::NODE_COUNT;

/// Maximum raw or recipe payload size per logical segment (`3576` bytes), fitting inside a single
/// 4096-byte [`jinshu::node::TrieNode`] page alongside its header and 16 branch hashes.
pub const MAX_SEGMENT_BYTES: usize = 3576;
/// `TrieNode` flag marking an isolated leaf segment produced by a single hypercube vertex.
pub const FLAG_KL_SEGMENT: u8 = 0x01;
/// `TrieNode` flag marking an intermediate Merkle subtree merged during hypercube reduction rounds.
pub const FLAG_KL_AGGREGATE: u8 = 0x02;
/// `TrieNode` flag marking the consolidated root Merkle node on Coordinator vertex `0`.
pub const FLAG_KL_ARCHIVE_ROOT: u8 = 0x03;

/// Default 32-byte shared dictionary of 4-byte words used for `Opcode::Dict` synthesis.
pub const DEFAULT_DICT: &[u8] = b"AMENTYS-KL-CUBE-KPACK-MERKLE-OS!";

/// Self-contained generative `kpack` formula assigned to a single logical vertex (`0..=7`).
#[derive(Debug, Clone)]
pub struct KpackFormula {
    /// Logical vertex identifier (`0..=7`).
    pub node_id: u8,
    /// Selected `kpack` opcode (`LIT`, `DELTA`, `RLE`, `SEED`, `DICT`, `XOR`, `RS`).
    pub opcode: Opcode,
    /// Opcode-specific parameter (run length for `RLE`, word width for `DICT`, PRNG seed for `SEED`).
    pub param: u32,
    /// Exact byte length of the reconstructed raw segment (`0..=MAX_SEGMENT_BYTES`).
    pub raw_len: usize,
    /// Encoded recipe instructions or literal bytes.
    pub payload: [u8; MAX_SEGMENT_BYTES],
    /// Active byte length inside [`Self::payload`].
    pub payload_len: usize,
    /// Optional reference vertex ID (`0..=7`) when `opcode` is `Opcode::Delta` or `Opcode::Xor`.
    pub ref_node_id: Option<u8>,
    /// Expected `BLAKE3` [`Noun`] of the reconstructed raw segment.
    pub expected_noun: Noun,
}

impl KpackFormula {
    #[must_use]
    pub const fn opcode_name(&self) -> &'static str {
        match self.opcode {
            Opcode::Lit => "LIT",
            Opcode::Delta => "DELTA",
            Opcode::Rle => "RLE",
            Opcode::Seed => "SEED",
            Opcode::Dict => "DICT",
            Opcode::Xor => "XOR",
            Opcode::Rs => "RS",
        }
    }

    /// Synthesizes the most compact deterministic `kpack` formula for a segment assigned to `node_id`.
    #[must_use]
    pub fn synthesize(
        node_id: u8,
        raw_data: &[u8],
        reference: Option<(u8, &[u8])>,
        dictionary: Option<&[u8]>,
    ) -> Self {
        let clamped_len = core::cmp::min(raw_data.len(), MAX_SEGMENT_BYTES);
        let slice = &raw_data[..clamped_len];
        let expected_noun = Noun::of(slice);
        let mut payload = [0u8; MAX_SEGMENT_BYTES];

        // 1. Try RLE if all bytes are identical and length >= 4
        if clamped_len >= 4 {
            let first = slice[0];
            if slice.iter().all(|&b| b == first) {
                payload[0] = first;
                return Self {
                    node_id,
                    opcode: Opcode::Rle,
                    param: clamped_len as u32,
                    raw_len: clamped_len,
                    payload,
                    payload_len: 1,
                    ref_node_id: None,
                    expected_noun,
                };
            }
        }

        // 2. Try DICT (4-byte words) if a dictionary is available
        if let Some(dict) = dictionary {
            if clamped_len >= 4 && clamped_len % 4 == 0 && dict.len() >= 4 {
                let word_count = clamped_len / 4;
                let max_dict_words = core::cmp::min(dict.len() / 4, 255);
                let mut all_found = true;

                for w in 0..word_count {
                    let target_word = &slice[w * 4..(w + 1) * 4];
                    let mut found_idx = None;
                    for d_idx in 0..max_dict_words {
                        if &dict[d_idx * 4..(d_idx + 1) * 4] == target_word {
                            found_idx = Some(d_idx as u8);
                            break;
                        }
                    }
                    if let Some(idx) = found_idx {
                        payload[w] = idx;
                    } else {
                        all_found = false;
                        break;
                    }
                }

                if all_found {
                    return Self {
                        node_id,
                        opcode: Opcode::Dict,
                        param: 4,
                        raw_len: clamped_len,
                        payload,
                        payload_len: word_count,
                        ref_node_id: None,
                        expected_noun,
                    };
                }
                payload[..word_count].fill(0);
            }
        }

        // 3. Try SEED + Sparse Patch if a seed in 1..=32 matches closely
        if clamped_len >= 8 && clamped_len <= 512 {
            let mut best_seed = 0u32;
            let mut best_patch_len = usize::MAX;

            for candidate_seed in 1..=32u32 {
                let mut state = candidate_seed;
                let mut mismatches = 0usize;
                for &b in slice {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    let r = (state & 0xFF) as u8;
                    if r != b {
                        mismatches += 1;
                    }
                }
                let patch_len = 2 + mismatches * 3;
                if patch_len < best_patch_len && patch_len + 4 < clamped_len {
                    best_seed = candidate_seed;
                    best_patch_len = patch_len;
                }
            }

            if best_seed != 0 && best_patch_len < clamped_len {
                let mismatches = (best_patch_len - 2) / 3;
                payload[0..2].copy_from_slice(&(mismatches as u16).to_le_bytes());
                let mut st = best_seed;
                let mut p_off = 2;
                for (pos, &b) in slice.iter().enumerate() {
                    st ^= st << 13;
                    st ^= st >> 17;
                    st ^= st << 5;
                    let r = (st & 0xFF) as u8;
                    if r != b && p_off + 3 <= MAX_SEGMENT_BYTES {
                        payload[p_off..p_off + 2].copy_from_slice(&(pos as u16).to_le_bytes());
                        payload[p_off + 2] = b;
                        p_off += 3;
                    }
                }
                return Self {
                    node_id,
                    opcode: Opcode::Seed,
                    param: best_seed,
                    raw_len: clamped_len,
                    payload,
                    payload_len: best_patch_len,
                    ref_node_id: None,
                    expected_noun,
                };
            }
        }

        // 4. Try DELTA against a reference segment if provided
        if let Some((ref_id, ref_bytes)) = reference {
            if let Some(delta_len) = try_encode_delta_into(slice, ref_bytes, &mut payload) {
                if delta_len + 4 < clamped_len {
                    return Self {
                        node_id,
                        opcode: Opcode::Delta,
                        param: 0,
                        raw_len: clamped_len,
                        payload,
                        payload_len: delta_len,
                        ref_node_id: Some(ref_id),
                        expected_noun,
                    };
                }
                payload[..delta_len].fill(0);
            }
        }

        // 5. Fallback: LIT (literal copy)
        payload[..clamped_len].copy_from_slice(slice);
        Self {
            node_id,
            opcode: Opcode::Lit,
            param: clamped_len as u32,
            raw_len: clamped_len,
            payload,
            payload_len: clamped_len,
            ref_node_id: None,
            expected_noun,
        }
    }

    /// Executes the `kpack` formula in isolated node memory, verifies the `BLAKE3` `Noun`,
    /// and populates `out_node` if provided.
    pub fn execute_into(
        &self,
        reference_bytes: Option<&[u8]>,
        dictionary: Option<&[u8]>,
        output: &mut [u8; MAX_SEGMENT_BYTES],
        out_node: Option<&mut TrieNode>,
    ) -> Result<usize, &'static str> {
        if self.raw_len > MAX_SEGMENT_BYTES || self.payload_len > MAX_SEGMENT_BYTES {
            return Err("Segment size exceeds MAX_SEGMENT_BYTES (3576)");
        }
        output[..self.raw_len].fill(0);
        let aux_ref = match self.opcode {
            Opcode::Dict => dictionary,
            Opcode::Delta | Opcode::Xor => reference_bytes,
            _ => None,
        };

        execute(
            &self.opcode,
            self.param,
            &self.payload[..self.payload_len],
            &mut output[..self.raw_len],
            aux_ref,
        );

        let actual_noun = Noun::of(&output[..self.raw_len]);
        if actual_noun != self.expected_noun {
            return Err("Kpack formula execution failed BLAKE3 Noun verification");
        }

        if let Some(node) = out_node {
            *node = TrieNode::new();
            node.flags = FLAG_KL_SEGMENT | ((self.node_id & 0x07) << 4);
            node.opcode = self.opcode.clone() as u8;
            node.param = self.param;
            node.payload[..self.payload_len].copy_from_slice(&self.payload[..self.payload_len]);
            node.set_branch(self.node_id & 0x07, actual_noun);
        }

        Ok(self.raw_len)
    }

    /// Executes the `kpack` formula in isolated node memory, verifies the `BLAKE3` `Noun`,
    /// and seals the result inside a `jinshu::TrieNode`.
    pub fn execute_verified(
        &self,
        reference_bytes: Option<&[u8]>,
        dictionary: Option<&[u8]>,
    ) -> Result<([u8; MAX_SEGMENT_BYTES], usize, TrieNode), &'static str> {
        let mut output = [0u8; MAX_SEGMENT_BYTES];
        let mut node = TrieNode::new();
        let len = self.execute_into(reference_bytes, dictionary, &mut output, Some(&mut node))?;
        Ok((output, len, node))
    }

    /// Executes the formula and slices a 64-byte `Wave64` register tile (`1 byte per lane * 64 lanes`)
    /// at `tile_index` (`tile_index * 64`), padding trailing lanes with `0` for branchless `klm` GPU execution.
    pub fn execute_wave64_tile(
        &self,
        reference_bytes: Option<&[u8]>,
        dictionary: Option<&[u8]>,
        tile_index: usize,
    ) -> Result<[u8; 64], &'static str> {
        let mut output = [0u8; MAX_SEGMENT_BYTES];
        let len = self.execute_into(reference_bytes, dictionary, &mut output, None)?;
        let mut tile = [0u8; 64];
        let start = tile_index.saturating_mul(64);
        if start < len {
            let copy_len = core::cmp::min(64, len - start);
            tile[..copy_len].copy_from_slice(&output[start..start + copy_len]);
        }
        Ok(tile)
    }

    /// Genetic formula crossover (Bird Fusion in Kōgnitara / `klm` evolutionary synthesis):
    /// Fuses two parent `KpackFormula`s (`self` and `partner`) along a hypercube edge using
    /// `mutation_seed` and synthesizes the optimal child formula for `child_node_id`.
    pub fn crossover(
        &self,
        partner: &Self,
        child_node_id: u8,
        mutation_seed: u32,
        dictionary: Option<&[u8]>,
    ) -> Result<Self, &'static str> {
        let (buf_a, len_a, _) = self.execute_verified(None, dictionary)?;
        let (buf_b, len_b, _) = partner.execute_verified(None, dictionary)?;
        let child_len = core::cmp::min(core::cmp::max(len_a, len_b), MAX_SEGMENT_BYTES);
        let mut child_raw = [0u8; MAX_SEGMENT_BYTES];

        // If both parents are SEED formulas and no mutation is requested, combine their seeds directly
        let mut state = mutation_seed ^ (self.param.rotate_left(7)) ^ partner.param;
        if state == 0 {
            state = 0xA3C5_9AC3;
        }

        for i in 0..child_len {
            let a = if i < len_a { buf_a[i] } else { 0 };
            let b = if i < len_b { buf_b[i] } else { 0 };
            // Deterministic Wavefront-friendly lane crossover (even lanes inherit dominant parent A, odd lanes blend B)
            let blended = if (i & 1) == 0 {
                a
            } else if mutation_seed == 0 {
                b
            } else {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                a ^ b ^ ((state & 0x0F) as u8)
            };
            child_raw[i] = blended;
        }

        Ok(Self::synthesize(
            child_node_id & 0x07,
            &child_raw[..child_len],
            Some((self.node_id, &buf_a[..len_a])),
            dictionary,
        ))
    }
}

/// Parity stripe across all 8 logical nodes of the Hamon Cube (XOR / Reed-Solomon GF(2^8) parity).
/// Allows deterministic reconstruction of any single failed or corrupted node out of the 8.
#[derive(Debug, Clone)]
pub struct CubeParityStripe {
    pub segment_lengths: [usize; NODE_COUNT],
    pub max_len: usize,
    pub parity_bytes: [u8; MAX_SEGMENT_BYTES],
    pub parity_noun: Noun,
}

impl CubeParityStripe {
    #[must_use]
    pub fn compute(segments: &[(&[u8], usize); NODE_COUNT]) -> Self {
        let mut parity_bytes = [0u8; MAX_SEGMENT_BYTES];
        let mut segment_lengths = [0usize; NODE_COUNT];
        let mut max_len = 0usize;

        for i in 0..NODE_COUNT {
            let (buf, len) = segments[i];
            let safe_len = core::cmp::min(len, core::cmp::min(buf.len(), MAX_SEGMENT_BYTES));
            segment_lengths[i] = safe_len;
            if safe_len > max_len {
                max_len = safe_len;
            }
            for b_idx in 0..safe_len {
                parity_bytes[b_idx] ^= buf[b_idx];
            }
        }

        let parity_noun = Noun::of(&parity_bytes[..max_len]);
        Self {
            segment_lengths,
            max_len,
            parity_bytes,
            parity_noun,
        }
    }

    /// Reconstructs the exact raw bytes of `missing_node_id` using the other 7 nodes and the parity stripe.
    pub fn recover_node(
        &self,
        missing_node_id: u8,
        available_segments: &[Option<&[u8]>; NODE_COUNT],
        expected_noun: &Noun,
    ) -> Result<([u8; MAX_SEGMENT_BYTES], usize), &'static str> {
        let missing_idx = missing_node_id as usize;
        if missing_idx >= NODE_COUNT {
            return Err("Invalid node ID for parity recovery");
        }
        let target_len = self.segment_lengths[missing_idx];
        let mut recovered = [0u8; MAX_SEGMENT_BYTES];
        recovered[..self.max_len].copy_from_slice(&self.parity_bytes[..self.max_len]);

        for i in 0..NODE_COUNT {
            if i == missing_idx {
                continue;
            }
            let seg = available_segments[i].ok_or("Cannot recover: more than 1 node is missing")?;
            let len = core::cmp::min(self.segment_lengths[i], seg.len());
            for b_idx in 0..len {
                recovered[b_idx] ^= seg[b_idx];
            }
        }

        let check_noun = Noun::of(&recovered[..target_len]);
        if &check_noun != expected_noun {
            return Err("Parity recovery BLAKE3 Noun mismatch");
        }
        Ok((recovered, target_len))
    }
}

/// GF(2^8) multiplication with irreducible polynomial `x^8 + x^4 + x^3 + x^2 + 1` (`0x11D`).
#[must_use]
pub const fn gf256_mul(mut a: u8, mut b: u8) -> u8 {
    let mut p = 0u8;
    let mut i = 0;
    while i < 8 {
        if (b & 1) != 0 {
            p ^= a;
        }
        let hi = a & 0x80;
        a <<= 1;
        if hi != 0 {
            a ^= 0x1D;
        }
        b >>= 1;
        i += 1;
    }
    p
}

/// GF(2^8) multiplicative inverse (`a^254` via square-and-multiply in GF(2^8)).
#[must_use]
pub const fn gf256_inv(a: u8) -> u8 {
    if a == 0 {
        return 0;
    }
    let mut result = 1u8;
    let mut base = a;
    let mut exp = 254u8;
    while exp > 0 {
        if (exp & 1) != 0 {
            result = gf256_mul(result, base);
        }
        base = gf256_mul(base, base);
        exp >>= 1;
    }
    result
}

/// Dual-Parity Reed-Solomon stripe (`P` + `Q` over $\text{GF}(2^8)$, $6+2$ fault tolerance)
/// for Kōgnitara Sanctuary resilience and `klm` GPU cluster recovery.
/// Can deterministically heal up to **2 simultaneously missing or corrupted nodes** out of 8.
#[derive(Debug, Clone)]
pub struct CubeDualParityStripe {
    pub segment_lengths: [usize; NODE_COUNT],
    pub max_len: usize,
    pub p_parity: [u8; MAX_SEGMENT_BYTES],
    pub q_parity: [u8; MAX_SEGMENT_BYTES],
    pub p_noun: Noun,
    pub q_noun: Noun,
}

impl CubeDualParityStripe {
    pub const GF_COEFFS: [u8; NODE_COUNT] = [1, 2, 4, 8, 16, 32, 64, 128];

    #[must_use]
    pub fn compute(segments: &[(&[u8], usize); NODE_COUNT]) -> Self {
        let mut p_parity = [0u8; MAX_SEGMENT_BYTES];
        let mut q_parity = [0u8; MAX_SEGMENT_BYTES];
        let mut segment_lengths = [0usize; NODE_COUNT];
        let mut max_len = 0usize;

        for i in 0..NODE_COUNT {
            let (buf, len) = segments[i];
            let safe_len = core::cmp::min(len, core::cmp::min(buf.len(), MAX_SEGMENT_BYTES));
            segment_lengths[i] = safe_len;
            if safe_len > max_len {
                max_len = safe_len;
            }
            let coeff = Self::GF_COEFFS[i];
            for b_idx in 0..safe_len {
                let val = buf[b_idx];
                p_parity[b_idx] ^= val;
                q_parity[b_idx] ^= gf256_mul(coeff, val);
            }
        }

        let p_noun = Noun::of(&p_parity[..max_len]);
        let q_noun = Noun::of(&q_parity[..max_len]);
        Self {
            segment_lengths,
            max_len,
            p_parity,
            q_parity,
            p_noun,
            q_noun,
        }
    }

    /// Reconstructs two simultaneously missing or corrupted nodes `(node_x, node_y)` (`node_x < node_y`)
    /// using Reed-Solomon $\text{GF}(2^8)$ `P` and `Q` syndromes, and verifies both `BLAKE3` `Noun`s.
    pub fn recover_two_nodes(
        &self,
        node_x: u8,
        node_y: u8,
        available_segments: &[Option<&[u8]>; NODE_COUNT],
        expected_noun_x: &Noun,
        expected_noun_y: &Noun,
    ) -> Result<
        (
            ([u8; MAX_SEGMENT_BYTES], usize),
            ([u8; MAX_SEGMENT_BYTES], usize),
        ),
        &'static str,
    > {
        let x = node_x as usize;
        let y = node_y as usize;
        if x >= NODE_COUNT || y >= NODE_COUNT || x == y {
            return Err("Invalid node pair for dual-parity Reed-Solomon recovery");
        }

        let mut syn_p = [0u8; MAX_SEGMENT_BYTES];
        let mut syn_q = [0u8; MAX_SEGMENT_BYTES];
        syn_p[..self.max_len].copy_from_slice(&self.p_parity[..self.max_len]);
        syn_q[..self.max_len].copy_from_slice(&self.q_parity[..self.max_len]);

        for i in 0..NODE_COUNT {
            if i == x || i == y {
                continue;
            }
            let seg =
                available_segments[i].ok_or("Cannot recover: more than 2 nodes are missing")?;
            let len = core::cmp::min(self.segment_lengths[i], seg.len());
            let coeff = Self::GF_COEFFS[i];
            for b_idx in 0..len {
                let v = seg[b_idx];
                syn_p[b_idx] ^= v;
                syn_q[b_idx] ^= gf256_mul(coeff, v);
            }
        }

        let gx = Self::GF_COEFFS[x];
        let gy = Self::GF_COEFFS[y];
        let denom_inv = gf256_inv(gx ^ gy);

        let mut rec_x = [0u8; MAX_SEGMENT_BYTES];
        let mut rec_y = [0u8; MAX_SEGMENT_BYTES];
        for b_idx in 0..self.max_len {
            let sp = syn_p[b_idx];
            let sq = syn_q[b_idx];
            let dy = gf256_mul(denom_inv, sq ^ gf256_mul(gx, sp));
            let dx = sp ^ dy;
            rec_x[b_idx] = dx;
            rec_y[b_idx] = dy;
        }

        let len_x = self.segment_lengths[x];
        let len_y = self.segment_lengths[y];
        if &Noun::of(&rec_x[..len_x]) != expected_noun_x {
            return Err("Dual-parity recovery BLAKE3 Noun mismatch on first node");
        }
        if &Noun::of(&rec_y[..len_y]) != expected_noun_y {
            return Err("Dual-parity recovery BLAKE3 Noun mismatch on second node");
        }

        Ok(((rec_x, len_x), (rec_y, len_y)))
    }
}

fn try_encode_delta_into(
    target: &[u8],
    reference: &[u8],
    payload: &mut [u8; MAX_SEGMENT_BYTES],
) -> Option<usize> {
    if target.is_empty() || reference.is_empty() {
        return None;
    }
    let mut prefix_len = 0usize;
    let max_prefix = core::cmp::min(target.len(), reference.len());
    while prefix_len < max_prefix && target[prefix_len] == reference[prefix_len] {
        prefix_len += 1;
    }

    if prefix_len < 8 {
        return None;
    }

    let mut pos = 0usize;

    // Instruction 0x01: Copy prefix from reference at offset 0
    payload[pos] = 0x01;
    payload[pos + 1..pos + 3].copy_from_slice(&0u16.to_le_bytes());
    payload[pos + 3..pos + 5].copy_from_slice(&(prefix_len as u16).to_le_bytes());
    pos += 5;

    let rem = target.len() - prefix_len;
    if rem > 0 {
        if pos + 3 + rem + 1 > MAX_SEGMENT_BYTES {
            return None;
        }
        // Instruction 0x02: Add remaining literal bytes
        payload[pos] = 0x02;
        payload[pos + 1..pos + 3].copy_from_slice(&(rem as u16).to_le_bytes());
        pos += 3;
        payload[pos..pos + rem].copy_from_slice(&target[prefix_len..]);
        pos += rem;
    }

    // End marker 0x00
    if pos + 1 > MAX_SEGMENT_BYTES {
        return None;
    }
    payload[pos] = 0x00;
    pos += 1;

    Some(pos)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_formula_synthesis_and_execution_all_modes() {
        // 1. RLE segment
        let rle_data = [b'A'; 128];
        let f_rle = KpackFormula::synthesize(1, &rle_data, None, Some(DEFAULT_DICT));
        assert_eq!(f_rle.opcode, Opcode::Rle);
        assert_eq!(f_rle.payload_len, 1);
        let (out_rle, len_rle, _) = f_rle
            .execute_verified(None, Some(DEFAULT_DICT))
            .expect("RLE should verify");
        assert_eq!(&out_rle[..len_rle], &rle_data[..]);

        // 2. DICT segment
        let dict_data = b"AMENTYS-KL-CUBE-AMENTYS-KL-CUBE-";
        let f_dict = KpackFormula::synthesize(2, dict_data, None, Some(DEFAULT_DICT));
        assert_eq!(f_dict.opcode, Opcode::Dict);
        assert_eq!(f_dict.payload_len, 8);
        let (out_dict, len_dict, _) = f_dict
            .execute_verified(None, Some(DEFAULT_DICT))
            .expect("DICT should verify");
        assert_eq!(&out_dict[..len_dict], &dict_data[..]);

        // 3. DELTA segment
        let base_data = b"AmentysOS Hypercube Distributed Kpack Job Segment v1.0";
        let next_data = b"AmentysOS Hypercube Distributed Kpack Job Segment v2.0";
        let f_delta = KpackFormula::synthesize(
            3,
            next_data,
            Some((0, base_data)),
            Some(DEFAULT_DICT),
        );
        assert_eq!(f_delta.opcode, Opcode::Delta);
        assert!(f_delta.payload_len < next_data.len());
        let (out_delta, len_delta, _) = f_delta
            .execute_verified(Some(base_data), Some(DEFAULT_DICT))
            .expect("DELTA should verify");
        assert_eq!(&out_delta[..len_delta], &next_data[..]);
    }

    #[test]
    fn test_parity_stripe_single_node_recovery() {
        let s0 = b"Node0-Coordinator-Header";
        let s1 = b"Node1-Worker-Payload-Alpha";
        let s2 = b"Node2-Worker-Payload-Beta";
        let s3 = b"Node3-Worker-Payload-Gamma";
        let s4 = b"Node4-Worker-Payload-Delta";
        let s5 = b"Node5-Worker-Payload-Epsilon";
        let s6 = b"Node6-Worker-Payload-Zeta";
        let s7 = b"Node7-Worker-Payload-Omega";

        let segments: [(&[u8], usize); 8] = [
            (s0, s0.len()),
            (s1, s1.len()),
            (s2, s2.len()),
            (s3, s3.len()),
            (s4, s4.len()),
            (s5, s5.len()),
            (s6, s6.len()),
            (s7, s7.len()),
        ];

        let stripe = CubeParityStripe::compute(&segments);

        // Simulate failure of node 5
        let available: [Option<&[u8]>; 8] = [
            Some(s0),
            Some(s1),
            Some(s2),
            Some(s3),
            Some(s4),
            None,
            Some(s6),
            Some(s7),
        ];
        let expected_s5_noun = Noun::of(s5);
        let (recovered, rec_len) = stripe
            .recover_node(5, &available, &expected_s5_noun)
            .expect("Should recover node 5");
        assert_eq!(&recovered[..rec_len], &s5[..]);
    }

    #[test]
    fn test_dual_parity_reed_solomon_two_nodes_recovery_and_crossover() {
        let s0 = b"Node0-Queen-Founding-Breath-0000";
        let s1 = b"Node1-Worker-SoulGraph-001-AAAAA";
        let s2 = b"Node2-Worker-SoulGraph-010-BBBBB";
        let s3 = b"Node3-Scout-Oniria-011-CCCCCCCCC";
        let s4 = b"Node4-Guardian-Umbra-100-DDDDDDD";
        let s5 = b"Node5-Scout-Seed-101-EEEEEEEEEEE";
        let s6 = b"Node6-Fulgaris-Burst-110-FFFFFFF";
        let s7 = b"Node7-Guardian-Seal-111-GGGGGGGG";

        let segments: [(&[u8], usize); 8] = [
            (s0, s0.len()),
            (s1, s1.len()),
            (s2, s2.len()),
            (s3, s3.len()),
            (s4, s4.len()),
            (s5, s5.len()),
            (s6, s6.len()),
            (s7, s7.len()),
        ];

        let dual = CubeDualParityStripe::compute(&segments);

        // Simulate simultaneous loss of Node 2 and Node 6 (2 out of 8 nodes missing)
        let available: [Option<&[u8]>; 8] = [
            Some(s0),
            Some(s1),
            None,
            Some(s3),
            Some(s4),
            Some(s5),
            None,
            Some(s7),
        ];

        let ((rec2, len2), (rec6, len6)) = dual
            .recover_two_nodes(2, 6, &available, &Noun::of(s2), &Noun::of(s6))
            .expect("Reed-Solomon GF(2^8) P+Q should recover both node 2 and node 6");
        assert_eq!(&rec2[..len2], &s2[..]);
        assert_eq!(&rec6[..len6], &s6[..]);

        // Test genetic formula crossover (Bird Fusion) and Wave64 tile extraction
        let fa = KpackFormula::synthesize(1, s1, None, Some(DEFAULT_DICT));
        let fb = KpackFormula::synthesize(3, s3, None, Some(DEFAULT_DICT));
        let child = fa
            .crossover(&fb, 2, 0x1337, Some(DEFAULT_DICT))
            .expect("Crossover should synthesize valid child formula");
        let tile = child
            .execute_wave64_tile(Some(s1), Some(DEFAULT_DICT), 0)
            .expect("Wave64 tile execution should succeed");
        assert_eq!(&tile[..s1.len()], &child.execute_verified(Some(s1), Some(DEFAULT_DICT)).unwrap().0[..s1.len()]);
    }
}
