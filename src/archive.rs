//! # Binary `KPK1` Formula-Based Archive Format ($O(1)$ Random Access & Self-Healing)
//!
//! This module implements the serverless, zero-database `KPK1` binary container specified in
//! *Kpack — Formula-Based Archive*:
//!
//! ```text
//! +-----------------------------------------------------------------------+
//! | 1. KpkHeader (48 bytes, magic = "KPK1", version = 1, flags)           |
//! +-----------------------------------------------------------------------+
//! | 2. Shared Dictionary Table (u16 length + 4-byte word dictionary)      |
//! +-----------------------------------------------------------------------+
//! | 3. 64-Byte DMA-Aligned Recipes Table (8 vertex formulas)              |
//! |    + 64-Byte Aligned XOR / Reed-Solomon Parity Stripe                 |
//! +-----------------------------------------------------------------------+
//! | 4. CubeFS Manifest Table (8 x 56-byte cubefs::Extent descriptors)     |
//! +-----------------------------------------------------------------------+
//! | 5. O(1) Random-Access Segment Index (8 x 48-byte entries)             |
//! +-----------------------------------------------------------------------+
//! | 6. Cryptographic Footer (276 bytes: input_noun, merkle_root,          |
//! |    cubefs_root_noun, plan_seal, signed Checkpoint, trailing "KPK1")   |
//! +-----------------------------------------------------------------------+
//! ```
//!
//! - **$O(1)$ Segment Extraction ([`extract_kpk1_segment`])**: Reconstructs any single vertex
//!   segment `0..=7` directly from its indexed formula without decompressing the rest of the archive.
//! - **Transparent Parity Auto-Repair**: If a segment's reconstructed bytes fail `BLAKE3` [`Noun`]
//!   verification due to storage corruption, [`extract_segment_from_view`] automatically rebuilds
//!   the missing segment from the remaining 7 vertices and the embedded [`CubeParityStripe`].
//! - **Zero-Copy `klm` GPU DMA Alignment ([`KPK_FLAG_KLM_DMA64`])**: Aligns every recipe payload
//!   and parity stripe on 64-byte boundaries (`align_up_dma64`).

use cubefs::Extent;
use kpack::Opcode;
use noun::Noun;

use crate::hypercube::{
    Checkpoint, CubeExecutionReport, merge_hypercube_trie_nodes, verify_cube_checkpoint,
};
use crate::klm::align_up_dma64;
use crate::recipe::{
    CubeParityStripe, DEFAULT_DICT, FLAG_KL_ARCHIVE_ROOT, KpackFormula, MAX_SEGMENT_BYTES,
};
use crate::topology::{NODE_COUNT, reduction_tree_rounds};

pub const KPK_MAGIC: [u8; 4] = *b"KPK1";
pub const KPK_VERSION: u16 = 1;
pub const KPK_FLAG_PARITY_RS: u16 = 0x0001;
pub const KPK_FLAG_DICT: u16 = 0x0002;
pub const KPK_FLAG_HYPERCUBE_8: u16 = 0x0004;
pub const KPK_FLAG_KLM_DMA64: u16 = 0x0008;
pub const KPK_HASH_BLAKE3: u8 = 0;
pub const KPK_CHUNK_POW2: u8 = 12; // 2^12 = 4096 bytes per Jinshu page

pub const MAX_ARCHIVE_BYTES: usize = 36_864;
pub const HEADER_SIZE: usize = 48;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KpkHeader {
    pub magic: [u8; 4],
    pub version: u16,
    pub flags: u16,
    pub hash_algo_id: u8,
    pub chunk_base_pow2: u8,
    pub node_count: u8,
    pub reserved: u8,
    pub total_raw_size: u32,
    pub recipes_offset: u64,
    pub manifest_offset: u64,
    pub index_offset: u64,
    pub footer_offset: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KpkSegmentIndexEntry {
    pub node_id: u8,
    pub opcode: u8,
    pub ref_node_id: u8,
    pub reserved: u8,
    pub param: u32,
    pub recipe_offset: u32,
    pub recipe_len: u16,
    pub raw_len: u16,
    pub expected_noun: [u8; 32],
}

#[derive(Debug, Clone)]
pub struct KpkArchiveView {
    pub header: KpkHeader,
    pub index_entries: [KpkSegmentIndexEntry; NODE_COUNT],
    pub extents: [Extent; NODE_COUNT],
    pub formulas: [KpackFormula; NODE_COUNT],
    pub parity: CubeParityStripe,
    pub input_noun: Noun,
    pub merkle_root: Noun,
    pub cubefs_root_noun: Noun,
    pub plan_seal: Noun,
    pub checkpoint: Checkpoint,
}

impl KpkHeader {
    pub fn write_to(&self, out: &mut [u8]) -> Result<usize, &'static str> {
        if out.len() < HEADER_SIZE {
            return Err("Buffer too small for KPK1 header");
        }
        out[0..4].copy_from_slice(&self.magic);
        out[4..6].copy_from_slice(&self.version.to_le_bytes());
        out[6..8].copy_from_slice(&self.flags.to_le_bytes());
        out[8] = self.hash_algo_id;
        out[9] = self.chunk_base_pow2;
        out[10] = self.node_count;
        out[11] = self.reserved;
        out[12..16].copy_from_slice(&self.total_raw_size.to_le_bytes());
        out[16..24].copy_from_slice(&self.recipes_offset.to_le_bytes());
        out[24..32].copy_from_slice(&self.manifest_offset.to_le_bytes());
        out[32..40].copy_from_slice(&self.index_offset.to_le_bytes());
        out[40..48].copy_from_slice(&self.footer_offset.to_le_bytes());
        Ok(HEADER_SIZE)
    }

    pub fn read_from(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() < HEADER_SIZE {
            return Err("Archive shorter than KPK1 header");
        }
        if bytes[0..4] != KPK_MAGIC {
            return Err("Invalid KPK1 magic header");
        }
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        if version != KPK_VERSION {
            return Err("Unsupported KPK1 version");
        }
        let flags = u16::from_le_bytes([bytes[6], bytes[7]]);
        let hash_algo_id = bytes[8];
        let chunk_base_pow2 = bytes[9];
        let node_count = bytes[10];
        let reserved = bytes[11];
        if node_count as usize != NODE_COUNT {
            return Err("KPK1 archive node_count != 8");
        }
        let total_raw_size = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        let recipes_offset = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        let manifest_offset = u64::from_le_bytes(bytes[24..32].try_into().unwrap());
        let index_offset = u64::from_le_bytes(bytes[32..40].try_into().unwrap());
        let footer_offset = u64::from_le_bytes(bytes[40..48].try_into().unwrap());

        Ok(Self {
            magic: KPK_MAGIC,
            version,
            flags,
            hash_algo_id,
            chunk_base_pow2,
            node_count,
            reserved,
            total_raw_size,
            recipes_offset,
            manifest_offset,
            index_offset,
            footer_offset,
        })
    }
}

/// Serializes a completed `CubeExecutionReport` into the binary `KPK1` archive format:
/// `[Header] [Dictionary Table] [Recipes Table + Parity Stripe] [Manifest Extents] [Random-Access Index] [Footer + Checkpoint]`
pub fn pack_kpk1_archive(
    report: &CubeExecutionReport,
    out: &mut [u8],
) -> Result<usize, &'static str> {
    if out.len() < 4096 {
        return Err("Output buffer too small for KPK1 archive");
    }

    let mut cursor = HEADER_SIZE;

    // 1. Optional Dictionary Table
    let dict_len = DEFAULT_DICT.len();
    out[cursor..cursor + 2].copy_from_slice(&(dict_len as u16).to_le_bytes());
    cursor += 2;
    out[cursor..cursor + dict_len].copy_from_slice(DEFAULT_DICT);
    cursor += dict_len;

    // 2. Recipes Table + Parity Stripe (64-byte cache-line aligned for `klm` GPU DMA Zero-Copy)
    cursor = align_up_dma64(cursor);
    let recipes_offset = cursor as u64;
    let mut index_entries = [KpkSegmentIndexEntry {
        node_id: 0,
        opcode: 0,
        ref_node_id: 0xFF,
        reserved: 0,
        param: 0,
        recipe_offset: 0,
        recipe_len: 0,
        raw_len: 0,
        expected_noun: [0u8; 32],
    }; NODE_COUNT];

    for (i, formula) in report.formulas.iter().enumerate() {
        cursor = align_up_dma64(cursor);
        let entry_offset = cursor as u32;
        let p_len = formula.payload_len;
        if cursor + p_len > out.len() {
            return Err("Output buffer overflow while writing recipes table");
        }
        out[cursor..cursor + p_len].copy_from_slice(&formula.payload[..p_len]);
        cursor += p_len;

        let mut expected_noun = [0u8; 32];
        expected_noun.copy_from_slice(formula.expected_noun.as_bytes());
        index_entries[i] = KpkSegmentIndexEntry {
            node_id: formula.node_id,
            opcode: formula.opcode.clone() as u8,
            ref_node_id: formula.ref_node_id.unwrap_or(0xFF),
            reserved: 0,
            param: formula.param,
            recipe_offset: entry_offset,
            recipe_len: p_len as u16,
            raw_len: formula.raw_len as u16,
            expected_noun,
        };
    }

    // Write parity stripe right after the 8 node recipes (also 64-byte aligned when KPK_FLAG_KLM_DMA64 is set)
    cursor = align_up_dma64(cursor);
    let parity_len = report.parity.max_len;
    if cursor + 4 + NODE_COUNT * 2 + 32 + parity_len > out.len() {
        return Err("Output buffer overflow while writing parity stripe");
    }
    out[cursor..cursor + 2].copy_from_slice(&(parity_len as u16).to_le_bytes());
    cursor += 2;
    for i in 0..NODE_COUNT {
        let s_len = report.parity.segment_lengths[i] as u16;
        out[cursor..cursor + 2].copy_from_slice(&s_len.to_le_bytes());
        cursor += 2;
    }
    out[cursor..cursor + 32].copy_from_slice(report.parity.parity_noun.as_bytes());
    cursor += 32;
    out[cursor..cursor + parity_len].copy_from_slice(&report.parity.parity_bytes[..parity_len]);
    cursor += parity_len;

    // 3. Manifest Table (cubefs Extents per logical segment)
    let manifest_offset = cursor as u64;
    let mut logical_off = 0u64;
    for i in 0..NODE_COUNT {
        let seg_len = report.formulas[i].raw_len as u64;
        if cursor + 56 > out.len() {
            return Err("Output buffer overflow while writing manifest");
        }
        out[cursor..cursor + 8].copy_from_slice(&logical_off.to_le_bytes());
        out[cursor + 8..cursor + 16].copy_from_slice(&seg_len.to_le_bytes());
        out[cursor + 16..cursor + 48].copy_from_slice(report.segment_nouns[i].as_bytes());
        out[cursor + 48..cursor + 52].copy_from_slice(&1u32.to_le_bytes());
        out[cursor + 52..cursor + 56].copy_from_slice(&(seg_len as u32).to_le_bytes());
        cursor += 56;
        logical_off += seg_len;
    }

    // 4. Random-Access Index Table (48 bytes per node * 8 = 384 bytes)
    let index_offset = cursor as u64;
    for entry in &index_entries {
        if cursor + 48 > out.len() {
            return Err("Output buffer overflow while writing index");
        }
        out[cursor] = entry.node_id;
        out[cursor + 1] = entry.opcode;
        out[cursor + 2] = entry.ref_node_id;
        out[cursor + 3] = entry.reserved;
        out[cursor + 4..cursor + 8].copy_from_slice(&entry.param.to_le_bytes());
        out[cursor + 8..cursor + 12].copy_from_slice(&entry.recipe_offset.to_le_bytes());
        out[cursor + 12..cursor + 14].copy_from_slice(&entry.recipe_len.to_le_bytes());
        out[cursor + 14..cursor + 16].copy_from_slice(&entry.raw_len.to_le_bytes());
        out[cursor + 16..cursor + 48].copy_from_slice(&entry.expected_noun);
        cursor += 48;
    }

    // 5. Footer (input_noun, merkle_root, cubefs_root, plan_seal, Checkpoint, trailing KPK1 magic)
    let footer_offset = cursor as u64;
    if cursor + 276 > out.len() {
        return Err("Output buffer overflow while writing footer");
    }
    out[cursor..cursor + 32].copy_from_slice(report.input_noun.as_bytes());
    cursor += 32;
    out[cursor..cursor + 32].copy_from_slice(report.merkle_root.as_bytes());
    cursor += 32;
    out[cursor..cursor + 32].copy_from_slice(report.cubefs_root_noun.as_bytes());
    cursor += 32;
    let plan_seal = report.final_plan.seal();
    out[cursor..cursor + 32].copy_from_slice(plan_seal.as_bytes());
    cursor += 32;

    // Checkpoint (8 + 8 + 32 + 32 + 64 = 144 bytes)
    out[cursor..cursor + 8].copy_from_slice(&report.checkpoint.seq.to_le_bytes());
    out[cursor + 8..cursor + 16].copy_from_slice(&report.checkpoint.time.to_le_bytes());
    out[cursor + 16..cursor + 48].copy_from_slice(&report.checkpoint.root_meta_id);
    out[cursor + 48..cursor + 80].copy_from_slice(&report.checkpoint.merkle_root);
    out[cursor + 80..cursor + 144].copy_from_slice(&report.checkpoint.signature);
    cursor += 144;

    out[cursor..cursor + 4].copy_from_slice(&KPK_MAGIC);
    cursor += 4;

    // Write Header at offset 0
    let header = KpkHeader {
        magic: KPK_MAGIC,
        version: KPK_VERSION,
        flags: KPK_FLAG_PARITY_RS | KPK_FLAG_DICT | KPK_FLAG_HYPERCUBE_8 | KPK_FLAG_KLM_DMA64,
        hash_algo_id: KPK_HASH_BLAKE3,
        chunk_base_pow2: KPK_CHUNK_POW2,
        node_count: NODE_COUNT as u8,
        reserved: 0,
        total_raw_size: report.total_input_bytes as u32,
        recipes_offset,
        manifest_offset,
        index_offset,
        footer_offset,
    };
    header.write_to(&mut out[..HEADER_SIZE])?;

    Ok(cursor)
}

/// Parses and validates a binary `KPK1` archive into a structured `KpkArchiveView`.
pub fn inspect_kpk1_archive(bytes: &[u8]) -> Result<KpkArchiveView, &'static str> {
    let header = KpkHeader::read_from(bytes)?;
    let footer_off = header.footer_offset as usize;
    if footer_off + 276 > bytes.len() {
        return Err("Truncated KPK1 archive footer");
    }
    if bytes[footer_off + 272..footer_off + 276] != KPK_MAGIC {
        return Err("Missing trailing KPK1 magic in footer");
    }

    // Parse Index
    let mut index_entries = [KpkSegmentIndexEntry {
        node_id: 0,
        opcode: 0,
        ref_node_id: 0xFF,
        reserved: 0,
        param: 0,
        recipe_offset: 0,
        recipe_len: 0,
        raw_len: 0,
        expected_noun: [0u8; 32],
    }; NODE_COUNT];

    let idx_base = header.index_offset as usize;
    if idx_base + NODE_COUNT * 48 > bytes.len() {
        return Err("Truncated KPK1 index table");
    }
    for i in 0..NODE_COUNT {
        let off = idx_base + i * 48;
        let mut expected_noun = [0u8; 32];
        expected_noun.copy_from_slice(&bytes[off + 16..off + 48]);
        index_entries[i] = KpkSegmentIndexEntry {
            node_id: bytes[off],
            opcode: bytes[off + 1],
            ref_node_id: bytes[off + 2],
            reserved: bytes[off + 3],
            param: u32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap()),
            recipe_offset: u32::from_le_bytes(bytes[off + 8..off + 12].try_into().unwrap()),
            recipe_len: u16::from_le_bytes([bytes[off + 12], bytes[off + 13]]),
            raw_len: u16::from_le_bytes([bytes[off + 14], bytes[off + 15]]),
            expected_noun,
        };
    }

    // Parse Manifest Extents
    let man_base = header.manifest_offset as usize;
    if man_base + NODE_COUNT * 56 > bytes.len() {
        return Err("Truncated KPK1 manifest table");
    }
    let mut extents = [Extent {
        file_off: 0,
        len: 0,
        first_chunk_id: [0u8; 32],
        count: 0,
        last_len: 0,
    }; NODE_COUNT];

    for i in 0..NODE_COUNT {
        let off = man_base + i * 56;
        let mut first_chunk_id = [0u8; 32];
        first_chunk_id.copy_from_slice(&bytes[off + 16..off + 48]);
        extents[i] = Extent {
            file_off: u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap()),
            len: u64::from_le_bytes(bytes[off + 8..off + 16].try_into().unwrap()),
            first_chunk_id,
            count: u32::from_le_bytes(bytes[off + 48..off + 52].try_into().unwrap()),
            last_len: u32::from_le_bytes(bytes[off + 52..off + 56].try_into().unwrap()),
        };
    }

    // Parse Formulas & Parity Stripe
    let mut formulas: [KpackFormula; NODE_COUNT] = core::array::from_fn(|i| {
        let entry = &index_entries[i];
        let opcode = Opcode::from_u8(entry.opcode).unwrap_or(Opcode::Lit);
        let mut payload = [0u8; MAX_SEGMENT_BYTES];
        let r_off = entry.recipe_offset as usize;
        let r_len = entry.recipe_len as usize;
        if r_off + r_len <= bytes.len() && r_len <= MAX_SEGMENT_BYTES {
            payload[..r_len].copy_from_slice(&bytes[r_off..r_off + r_len]);
        }
        let expected_noun =
            Noun::from_bytes(&entry.expected_noun).unwrap_or_else(|| Noun::of(&entry.expected_noun));
        KpackFormula {
            node_id: entry.node_id,
            opcode,
            param: entry.param,
            raw_len: entry.raw_len as usize,
            payload,
            payload_len: r_len,
            ref_node_id: if entry.ref_node_id == 0xFF {
                None
            } else {
                Some(entry.ref_node_id)
            },
            expected_noun,
        }
    });

    // Parity stripe starts right after the last recipe (aligned to 64B if KPK_FLAG_KLM_DMA64 is set)
    let last_entry = &index_entries[NODE_COUNT - 1];
    let mut p_cursor = (last_entry.recipe_offset as usize) + (last_entry.recipe_len as usize);
    if (header.flags & KPK_FLAG_KLM_DMA64) != 0 {
        p_cursor = align_up_dma64(p_cursor);
    }
    if p_cursor + 2 + NODE_COUNT * 2 + 32 > bytes.len() {
        return Err("Truncated parity stripe in KPK1 archive");
    }
    let max_len = u16::from_le_bytes([bytes[p_cursor], bytes[p_cursor + 1]]) as usize;
    p_cursor += 2;
    let mut segment_lengths = [0usize; NODE_COUNT];
    for i in 0..NODE_COUNT {
        segment_lengths[i] =
            u16::from_le_bytes([bytes[p_cursor], bytes[p_cursor + 1]]) as usize;
        p_cursor += 2;
    }
    let parity_noun = Noun::from_bytes(&bytes[p_cursor..p_cursor + 32])
        .ok_or("Invalid parity noun in KPK1 archive")?;
    p_cursor += 32;
    if p_cursor + max_len > bytes.len() || max_len > MAX_SEGMENT_BYTES {
        return Err("Invalid parity stripe length in KPK1 archive");
    }
    let mut parity_bytes = [0u8; MAX_SEGMENT_BYTES];
    parity_bytes[..max_len].copy_from_slice(&bytes[p_cursor..p_cursor + max_len]);
    let parity = CubeParityStripe {
        segment_lengths,
        max_len,
        parity_bytes,
        parity_noun,
    };

    // Parse Footer
    let input_noun = Noun::from_bytes(&bytes[footer_off..footer_off + 32])
        .ok_or("Invalid input_noun in footer")?;
    let merkle_root = Noun::from_bytes(&bytes[footer_off + 32..footer_off + 64])
        .ok_or("Invalid merkle_root in footer")?;
    let cubefs_root_noun = Noun::from_bytes(&bytes[footer_off + 64..footer_off + 96])
        .ok_or("Invalid cubefs_root_noun in footer")?;
    let plan_seal = Noun::from_bytes(&bytes[footer_off + 96..footer_off + 128])
        .ok_or("Invalid plan_seal in footer")?;

    let cp_off = footer_off + 128;
    let seq = u64::from_le_bytes(bytes[cp_off..cp_off + 8].try_into().unwrap());
    let time = u64::from_le_bytes(bytes[cp_off + 8..cp_off + 16].try_into().unwrap());
    let mut root_meta_id = [0u8; 32];
    root_meta_id.copy_from_slice(&bytes[cp_off + 16..cp_off + 48]);
    let mut cp_merkle = [0u8; 32];
    cp_merkle.copy_from_slice(&bytes[cp_off + 48..cp_off + 80]);
    let mut signature = [0u8; 64];
    signature.copy_from_slice(&bytes[cp_off + 80..cp_off + 144]);

    let checkpoint = Checkpoint {
        seq,
        time,
        root_meta_id,
        merkle_root: cp_merkle,
        signature,
    };

    if !verify_cube_checkpoint(&checkpoint, &plan_seal) {
        return Err("KPK1 Checkpoint signature verification failed");
    }

    // Ensure formula node IDs are consistent
    for (i, f) in formulas.iter_mut().enumerate() {
        f.node_id = i as u8;
    }

    Ok(KpkArchiveView {
        header,
        index_entries,
        extents,
        formulas,
        parity,
        input_noun,
        merkle_root,
        cubefs_root_noun,
        plan_seal,
        checkpoint,
    })
}

/// Random-access $O(1)$ extraction of a single node segment (`0..=7`) from a parsed `KpkArchiveView`.
pub fn extract_segment_from_view(
    view: &KpkArchiveView,
    target_node_id: u8,
) -> Result<([u8; MAX_SEGMENT_BYTES], usize, bool), &'static str> {
    if target_node_id as usize >= NODE_COUNT {
        return Err("Segment node_id out of range 0..7");
    }
    let idx = target_node_id as usize;
    let formula = &view.formulas[idx];

    let ref_buf_opt = if let Some(ref_id) = formula.ref_node_id {
        if ref_id == target_node_id {
            return Err("Invalid self-referential delta formula");
        }
        let r_formula = &view.formulas[ref_id as usize];
        let (r_bytes, r_len, _) = r_formula.execute_verified(None, Some(DEFAULT_DICT))?;
        Some((r_bytes, r_len))
    } else {
        None
    };

    let ref_slice = ref_buf_opt.as_ref().map(|(b, l)| &b[..*l]);
    if let Ok((out, len, _)) = formula.execute_verified(ref_slice, Some(DEFAULT_DICT)) {
        return Ok((out, len, false));
    }

    // Slow path (Resilience): reconstruct all other 7 nodes and recover `target_node_id` via parity stripe!
    let mut reconstructed = [[0u8; MAX_SEGMENT_BYTES]; NODE_COUNT];
    let mut lengths = [0usize; NODE_COUNT];
    let mut ok_mask = 0u8;

    for i in 0..NODE_COUNT {
        if i == idx {
            continue;
        }
        let r_slice = view.formulas[i].ref_node_id.and_then(|rid| {
            if (ok_mask & (1 << rid)) != 0 {
                Some(&reconstructed[rid as usize][..lengths[rid as usize]])
            } else {
                None
            }
        });
        if let Ok((buf, l, _)) = view.formulas[i].execute_verified(r_slice, Some(DEFAULT_DICT)) {
            reconstructed[i] = buf;
            lengths[i] = l;
            ok_mask |= 1 << i;
        }
    }

    let available: [Option<&[u8]>; NODE_COUNT] = core::array::from_fn(|i| {
        if i == idx || (ok_mask & (1 << i)) == 0 {
            None
        } else {
            Some(&reconstructed[i][..lengths[i]])
        }
    });

    let (recovered, rec_len) =
        view.parity
            .recover_node(target_node_id, &available, &formula.expected_noun)?;
    Ok((recovered, rec_len, true))
}

/// Random-access $O(1)$ extraction of a single node segment (`0..=7`) from a `KPK1` archive.
/// If the target segment fails `BLAKE3` verification (e.g., disk corruption), it is automatically
/// repaired on the fly using the other 7 nodes and the `CubeParityStripe`.
pub fn extract_kpk1_segment(
    archive_bytes: &[u8],
    target_node_id: u8,
) -> Result<([u8; MAX_SEGMENT_BYTES], usize, bool), &'static str> {
    let view = inspect_kpk1_archive(archive_bytes)?;
    extract_segment_from_view(&view, target_node_id)
}

/// Extracts and verifies the entire payload from a `KPK1` archive, checking all 8 segment `Noun`s,
/// recomputing the 3-round hypercube Merkle root, and validating the final `input_noun`.
pub fn extract_kpk1_all(
    archive_bytes: &[u8],
    out: &mut [u8],
) -> Result<(usize, bool), &'static str> {
    let view = inspect_kpk1_archive(archive_bytes)?;
    if out.len() < view.header.total_raw_size as usize {
        return Err("Output buffer too small for extracted KPK1 archive");
    }

    let mut cursor = 0usize;
    let mut repaired_any = false;
    let mut segment_nodes: [jinshu::node::TrieNode; NODE_COUNT] =
        core::array::from_fn(|_| jinshu::node::TrieNode::new());

    for i in 0..NODE_COUNT {
        let (seg_buf, seg_len, repaired) = extract_segment_from_view(&view, i as u8)?;
        if repaired {
            repaired_any = true;
        }
        out[cursor..cursor + seg_len].copy_from_slice(&seg_buf[..seg_len]);
        cursor += seg_len;

        // Rebuild the TrieNode for Merkle tree verification
        let mut node = jinshu::node::TrieNode::new();
        node.flags = crate::recipe::FLAG_KL_SEGMENT | ((i as u8 & 0x07) << 4);
        node.opcode = view.formulas[i].opcode.clone() as u8;
        node.param = view.formulas[i].param;
        let p_len = view.formulas[i].payload_len;
        node.payload[..p_len].copy_from_slice(&view.formulas[i].payload[..p_len]);
        node.set_branch(i as u8, view.formulas[i].expected_noun.clone());
        segment_nodes[i] = node;
    }

    let whole_noun = Noun::of(&out[..cursor]);
    if whole_noun != view.input_noun {
        return Err("Extracted archive BLAKE3 input_noun mismatch");
    }

    // If no recipe payload was corrupted, also verify the 3-round hypercube Merkle root
    if !repaired_any {
        let (r1, r2, r3) = reduction_tree_rounds();
        for &(s, r) in &r1 {
            let (m, _) =
                merge_hypercube_trie_nodes(&segment_nodes[r as usize], &segment_nodes[s as usize], 1);
            segment_nodes[r as usize] = m;
        }
        for &(s, r) in &r2 {
            let (m, _) =
                merge_hypercube_trie_nodes(&segment_nodes[r as usize], &segment_nodes[s as usize], 2);
            segment_nodes[r as usize] = m;
        }
        let (s, r) = r3[0];
        let (mut root_node, _) =
            merge_hypercube_trie_nodes(&segment_nodes[r as usize], &segment_nodes[s as usize], 3);
        root_node.flags = FLAG_KL_ARCHIVE_ROOT;
        root_node.param = cursor as u32;
        let calc_bytes = root_node.calculate_noun();
        let calc_merkle = Noun::from_bytes(&calc_bytes).unwrap_or_else(|| Noun::of(&calc_bytes));
        if calc_merkle != view.merkle_root {
            return Err("Hypercube Merkle Root verification failed");
        }
    }

    Ok((cursor, repaired_any))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hypercube::execute_distributed_cube_job;
    use crate::topology::CubeConfig;

    #[test]
    fn test_kpk1_pack_inspect_random_access_and_self_healing() {
        let cfg = CubeConfig::default();
        let mut original = [0u8; 800];
        // Segment 0 (0..100): RLE
        original[0..100].fill(b'K');
        // Segment 1 (100..200): DICT
        for chunk in original[100..196].chunks_exact_mut(32) {
            chunk.copy_from_slice(DEFAULT_DICT);
        }
        original[196..200].copy_from_slice(b"OS!!");
        // Segments 2..8: structured data
        for i in 200..800 {
            original[i] = ((i * 7 + 13) & 0xFF) as u8;
        }

        let report = execute_distributed_cube_job(&cfg, &original, 7, 1_700_000_123)
            .expect("Job should succeed");

        let mut archive_buf = [0u8; MAX_ARCHIVE_BYTES];
        let archive_len =
            pack_kpk1_archive(&report, &mut archive_buf).expect("Packing KPK1 should succeed");
        let archive_slice = &mut archive_buf[..archive_len];

        // 1. Inspect archive
        let view = inspect_kpk1_archive(archive_slice).expect("Inspect should succeed");
        assert_eq!(view.header.total_raw_size, 800);
        assert_eq!(view.merkle_root, report.merkle_root);

        // 2. Random-access extraction of segment 0 and segment 5
        let (seg0, len0, rep0) =
            extract_kpk1_segment(archive_slice, 0).expect("Extract segment 0");
        assert!(!rep0);
        assert_eq!(&seg0[..len0], &original[0..100]);

        let (seg5, len5, rep5) =
            extract_kpk1_segment(archive_slice, 5).expect("Extract segment 5");
        assert!(!rep5);
        assert_eq!(&seg5[..len5], &original[500..600]);

        // 3. Full extraction
        let mut restored = [0u8; 800];
        let (restored_len, repaired) =
            extract_kpk1_all(archive_slice, &mut restored).expect("Full extraction should succeed");
        assert_eq!(restored_len, 800);
        assert!(!repaired);
        assert_eq!(&restored[..], &original[..]);

        // 4. Simulate corruption of Node 4's recipe payload inside the KPK1 archive and verify Reed-Solomon/XOR parity auto-repair!
        let node4_recipe_off = view.index_entries[4].recipe_offset as usize;
        archive_slice[node4_recipe_off] ^= 0xFF;

        let mut healed = [0u8; 800];
        let (healed_len, was_repaired) = extract_kpk1_all(archive_slice, &mut healed)
            .expect("Self-healing extraction should succeed via parity stripe");
        assert_eq!(healed_len, 800);
        assert!(was_repaired);
        assert_eq!(&healed[..], &original[..]);
    }
}
