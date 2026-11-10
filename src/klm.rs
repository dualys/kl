//! # `klm` AMD GPU Compute Bridge — Zero-Copy DMA, Conflict-Free `LDS` & `Wave64` `DPP`
//!
//! This module prepares `kl` data structures and execution primitives for the **`klm`** AMD GPU
//! compute framework (targeting RDNA 3/4 and CDNA 2/3/4 architectures inside Amentys OS):
//!
//! - **64-Byte Cache-Line Aligned Zero-Copy Descriptors ([`KlmSlotDescriptor`], [`KlmDispatchPacket`])**:
//!   All dispatch headers and per-vertex recipe slots are `#[repr(C, align(64))]` so an archive or
//!   dispatch packet can be mapped directly into GPU virtual memory without CPU marshaling.
//! - **32-Bank Conflict-Free `LDS` Addressing ([`lds_bank_conflict_free_offset`], [`lds_bank_of`])**:
//!   Shifts each vertex's Local Data Share (`LDS`) base address by `vertex_id * 4` bytes (one
//!   32-bit `LDS` bank), guaranteeing that simultaneous accesses across all 8 Hamon Cube vertices
//!   hit 8 distinct `LDS` banks in a single clock cycle.
//! - **In-Register `Wave64` `DPP` Butterfly Reduction ([`execute_wave64_dpp_butterfly`])**:
//!   Executes the 3-round Hamon Cube reduction across 64 lanes using AMD hardware `DPP` row XOR
//!   swizzles (`row_xmask: 1, 2, 4`), aggregating all 8 sub-cubes in 3 instruction cycles.
//! - **Formula-Compressed Quantized Tensor Tiles ([`KlmTensorTileQ8`])**: Synthesizes `Int8`
//!   weight tiles into compact `kpack` formulas (`RLE`, `DICT`, `SEED`, `DELTA`, `LIT`) and
//!   executes on-the-fly `Wave64` dot products directly from formula state.

use noun::Noun;

use crate::hypercube::CubeExecutionReport;
use crate::recipe::{DEFAULT_DICT, KpackFormula, MAX_SEGMENT_BYTES};
use crate::topology::{
    NODE_COUNT, WAVE64_LANES, Wave64CubeLattice, WaveMode, dpp_xor_partner,
};

/// AMD GPU cache-line & wavefront transaction alignment (64 bytes).
pub const KLM_DMA_ALIGNMENT: usize = 64;
/// Number of 32-bit (4-byte) banks in an AMD Compute Unit Local Data Share (`LDS`).
pub const KLM_LDS_BANKS: usize = 32;
/// Width of a single `LDS` bank in bytes (32-bit word).
pub const KLM_LDS_BANK_WIDTH: usize = 4;
/// Hardware DPP row XOR masks corresponding to the 3 rounds of Hamon Cube reduction.
pub const KLM_DPP_ROW_XMASKS: [u8; 3] = [0x01, 0x02, 0x04];

/// Rounds `offset` up to the next multiple of `KLM_DMA_ALIGNMENT` (64 bytes).
#[must_use]
pub const fn align_up_dma64(offset: usize) -> usize {
    (offset + (KLM_DMA_ALIGNMENT - 1)) & !(KLM_DMA_ALIGNMENT - 1)
}

/// Computes a bank-conflict-free `LDS` byte offset for `vertex_id` (`0..8`) and local `byte_offset`.
/// By shifting each vertex's base address by `vertex_id * 4` bytes (1 `LDS` bank), simultaneous reads
/// across the 8 Hamon Cube vertices hit 8 distinct `LDS` banks in a single clock cycle.
#[must_use]
pub const fn lds_bank_conflict_free_offset(
    vertex_id: u8,
    slot_stride_aligned: usize,
    byte_offset: usize,
) -> usize {
    let v = (vertex_id & 0x07) as usize;
    v * slot_stride_aligned + v * KLM_LDS_BANK_WIDTH + byte_offset
}

/// Returns the `LDS` bank index (`0..32`) accessed by a 32-bit word at `lds_byte_addr`.
#[must_use]
pub const fn lds_bank_of(lds_byte_addr: usize) -> u8 {
    ((lds_byte_addr / KLM_LDS_BANK_WIDTH) % KLM_LDS_BANKS) as u8
}

/// 64-byte cache-line aligned slot descriptor for a single Hamon Cube vertex in the `klm` framework.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KlmSlotDescriptor {
    pub node_id: u8,
    pub opcode: u8,
    pub ref_node_id: u8,
    pub lds_primary_bank: u8,
    pub param: u32,
    pub dma_recipe_offset: u32,
    pub recipe_len: u16,
    pub raw_len: u16,
    pub lds_base_offset: u32,
    pub reserved: [u8; 12],
    pub expected_noun: [u8; 32],
}

/// 64-byte aligned zero-copy dispatch packet prepared by `kl` for the `klm` AMD GPU runtime.
#[repr(C, align(64))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KlmDispatchPacket {
    pub magic: [u8; 4],
    pub version: u16,
    pub wave_lanes: u8,
    pub dpp_xmasks: [u8; 3],
    pub node_count: u8,
    pub bank_conflict_free: u8,
    pub lds_slot_stride: u32,
    pub total_lds_bytes: u32,
    pub reserved_hdr: [u8; 12],
    pub merkle_root: [u8; 32],
    pub cubefs_root: [u8; 32],
    pub plan_seal: [u8; 32],
    pub slots: [KlmSlotDescriptor; NODE_COUNT],
}

impl KlmDispatchPacket {
    pub const MAGIC: [u8; 4] = *b"KLMD";
    pub const VERSION: u16 = 1;

    /// Builds a hardware-ready `KlmDispatchPacket` from a completed `CubeExecutionReport`.
    #[must_use]
    pub fn from_execution_report(report: &CubeExecutionReport, wave_mode: WaveMode) -> Self {
        let max_raw = report
            .formulas
            .iter()
            .map(|f| f.raw_len)
            .max()
            .unwrap_or(64);
        // Align each LDS slot to 128 bytes (32 banks * 4 bytes) so the +4B per-vertex shift guarantees distinct banks
        let lds_slot_stride = ((max_raw + 127) & !127).max(128);
        let mut dma_cursor = align_up_dma64(64);

        let slots = core::array::from_fn(|i| {
            let f = &report.formulas[i];
            let dma_recipe_offset = dma_cursor as u32;
            dma_cursor = align_up_dma64(dma_cursor + f.payload_len.max(1));

            let lds_base = lds_bank_conflict_free_offset(i as u8, lds_slot_stride, 0);
            let lds_primary_bank = lds_bank_of(lds_base);
            let mut expected_noun = [0u8; 32];
            expected_noun.copy_from_slice(f.expected_noun.as_bytes());

            KlmSlotDescriptor {
                node_id: i as u8,
                opcode: f.opcode.clone() as u8,
                ref_node_id: f.ref_node_id.unwrap_or(0xFF),
                lds_primary_bank,
                param: f.param,
                dma_recipe_offset,
                recipe_len: f.payload_len as u16,
                raw_len: f.raw_len as u16,
                lds_base_offset: lds_base as u32,
                reserved: [0u8; 12],
                expected_noun,
            }
        });

        let total_lds_bytes =
            (lds_bank_conflict_free_offset(7, lds_slot_stride, lds_slot_stride)) as u32;

        let mut merkle_root = [0u8; 32];
        merkle_root.copy_from_slice(report.merkle_root.as_bytes());
        let mut cubefs_root = [0u8; 32];
        cubefs_root.copy_from_slice(report.cubefs_root_noun.as_bytes());
        let mut plan_seal = [0u8; 32];
        plan_seal.copy_from_slice(report.final_plan.seal().as_bytes());

        Self {
            magic: Self::MAGIC,
            version: Self::VERSION,
            wave_lanes: wave_mode.lanes() as u8,
            dpp_xmasks: KLM_DPP_ROW_XMASKS,
            node_count: NODE_COUNT as u8,
            bank_conflict_free: 1,
            lds_slot_stride: lds_slot_stride as u32,
            total_lds_bytes,
            reserved_hdr: [0u8; 12],
            merkle_root,
            cubefs_root,
            plan_seal,
            slots,
        }
    }

    /// Verifies that all 8 vertex slots are 64-byte DMA aligned and map to 8 distinct `LDS` banks (`0..8`).
    pub fn validate_hardware_constraints(&self) -> Result<(), &'static str> {
        if self.magic != Self::MAGIC {
            return Err("Invalid KLMD dispatch packet magic");
        }
        let mut bank_mask = 0u32;
        for (i, slot) in self.slots.iter().enumerate() {
            if slot.node_id as usize != i {
                return Err("Slot node_id mismatch in KlmDispatchPacket");
            }
            if (slot.dma_recipe_offset as usize) % KLM_DMA_ALIGNMENT != 0 {
                return Err("Recipe DMA offset is not 64-byte cache-line aligned");
            }
            let bank_bit = 1u32 << (slot.lds_primary_bank & 0x1F);
            if (bank_mask & bank_bit) != 0 {
                return Err("LDS bank conflict detected across Hamon Cube vertices");
            }
            bank_mask |= bank_bit;
        }
        Ok(())
    }

    /// Computes the cryptographic `Noun` sealing this `KlmDispatchPacket`.
    #[must_use]
    pub fn packet_noun(&self) -> Noun {
        let mut buf = [0u8; 256];
        buf[0..4].copy_from_slice(&self.magic);
        buf[4..6].copy_from_slice(&self.version.to_le_bytes());
        buf[6] = self.wave_lanes;
        buf[7..10].copy_from_slice(&self.dpp_xmasks);
        buf[10..14].copy_from_slice(&self.lds_slot_stride.to_le_bytes());
        buf[14..18].copy_from_slice(&self.total_lds_bytes.to_le_bytes());
        buf[18..50].copy_from_slice(&self.merkle_root);
        buf[50..82].copy_from_slice(&self.cubefs_root);
        buf[82..114].copy_from_slice(&self.plan_seal);
        for (i, s) in self.slots.iter().enumerate() {
            let off = 114 + i * 16;
            buf[off] = s.node_id;
            buf[off + 1] = s.opcode;
            buf[off + 2] = s.lds_primary_bank;
            buf[off + 3] = s.ref_node_id;
            buf[off + 4..off + 8].copy_from_slice(&s.dma_recipe_offset.to_le_bytes());
            buf[off + 8..off + 12].copy_from_slice(&s.lds_base_offset.to_le_bytes());
            buf[off + 12..off + 16].copy_from_slice(&s.expected_noun[0..4]);
        }
        Noun::of(&buf[..242])
    }
}

/// Trace of an in-register `Wave64` butterfly reduction using AMD `DPP` XOR masks (`1`, `2`, `4`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Wave64ButterflyReport {
    pub initial_vgpr: [u32; WAVE64_LANES],
    pub after_round1_xor1: [u32; WAVE64_LANES],
    pub after_round2_xor2: [u32; WAVE64_LANES],
    pub after_round3_xor4: [u32; WAVE64_LANES],
    pub subcube_sums: [u32; NODE_COUNT],
    pub wave64_total_sum: u32,
}

/// Executes a 3-round intra-subcube `DPP` butterfly reduction (`row_xmask: 1, 2, 4`) followed by
/// an inter-subcube `LDS` reduction across all 64 lanes of an AMD `Wave64` wavefront.
#[must_use]
pub fn execute_wave64_dpp_butterfly(vgpr_input: &[u32; WAVE64_LANES]) -> Wave64ButterflyReport {
    let mut r1 = *vgpr_input;
    for lane in 0..WAVE64_LANES {
        let partner = dpp_xor_partner(lane as u8, 1) as usize;
        r1[lane] = vgpr_input[lane].wrapping_add(vgpr_input[partner]);
    }

    let mut r2 = r1;
    for lane in 0..WAVE64_LANES {
        let partner = dpp_xor_partner(lane as u8, 2) as usize;
        r2[lane] = r1[lane].wrapping_add(r1[partner]);
    }

    let mut r3 = r2;
    for lane in 0..WAVE64_LANES {
        let partner = dpp_xor_partner(lane as u8, 3) as usize;
        r3[lane] = r2[lane].wrapping_add(r2[partner]);
    }

    // At the end of Round 3 (3 clock cycles of DPP instructions), the leader lane (vertex 0)
    // of each of the 8 sub-cubes holds the exact sum of its 8-node Hamon Cube!
    let mut subcube_sums = [0u32; NODE_COUNT];
    let mut wave64_total_sum = 0u32;
    for sub in 0..NODE_COUNT {
        let leader_lane = Wave64CubeLattice::lane_from_parts(sub as u8, 0) as usize;
        subcube_sums[sub] = r3[leader_lane];
        wave64_total_sum = wave64_total_sum.wrapping_add(subcube_sums[sub]);
    }

    Wave64ButterflyReport {
        initial_vgpr: *vgpr_input,
        after_round1_xor1: r1,
        after_round2_xor2: r2,
        after_round3_xor4: r3,
        subcube_sums,
        wave64_total_sum,
    }
}

/// Quantized `Int8` tensor tile stored as a `KpackFormula` for `klm` on-the-fly synthesis inside GPU `LDS` / `VGPR`s.
#[derive(Debug, Clone)]
pub struct KlmTensorTileQ8 {
    pub vertex_id: u8,
    pub scale_q16: u32,
    pub zero_point: i8,
    pub formula: KpackFormula,
}

impl KlmTensorTileQ8 {
    /// Quantizes or wraps a raw `i8` weight tile for `vertex_id` and synthesizes its compact `KpackFormula`.
    #[must_use]
    pub fn synthesize_tile(
        vertex_id: u8,
        weights_i8: &[i8],
        scale_q16: u32,
        zero_point: i8,
        reference: Option<(u8, &[u8])>,
    ) -> Self {
        let len = core::cmp::min(weights_i8.len(), MAX_SEGMENT_BYTES);
        let mut raw_u8 = [0u8; MAX_SEGMENT_BYTES];
        for i in 0..len {
            raw_u8[i] = weights_i8[i] as u8;
        }
        let formula = KpackFormula::synthesize(
            vertex_id & 0x07,
            &raw_u8[..len],
            reference,
            Some(DEFAULT_DICT),
        );
        Self {
            vertex_id: vertex_id & 0x07,
            scale_q16,
            zero_point,
            formula,
        }
    }

    /// Materializes a 64-lane `Wave64` `i32` register vector from this formula at `tile_index`
    /// and computes the partial dot product with an activation vector `activations_i8` (`64` lanes)
    /// reduced across the 8 Hamon sub-cubes via `execute_wave64_dpp_butterfly`.
    pub fn execute_wave64_dot_product(
        &self,
        reference_bytes: Option<&[u8]>,
        tile_index: usize,
        activations_i8: &[i8; WAVE64_LANES],
    ) -> Result<(i32, Wave64ButterflyReport), &'static str> {
        let weight_bytes =
            self.formula
                .execute_wave64_tile(reference_bytes, Some(DEFAULT_DICT), tile_index)?;
        let mut lane_products = [0u32; WAVE64_LANES];
        for lane in 0..WAVE64_LANES {
            let w = (weight_bytes[lane] as i8 as i32) - (self.zero_point as i32);
            let a = activations_i8[lane] as i32;
            lane_products[lane] = (w * a) as u32;
        }
        let report = execute_wave64_dpp_butterfly(&lane_products);
        Ok((report.wave64_total_sum as i32, report))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hypercube::execute_distributed_cube_job;
    use crate::topology::CubeConfig;

    #[test]
    fn test_klm_dispatch_packet_dma_alignment_and_lds_bank_conflict_free() {
        let cfg = CubeConfig::default();
        let mut payload = [0u8; 1024];
        for (i, b) in payload.iter_mut().enumerate() {
            *b = ((i * 5 + 1) & 0xFF) as u8;
        }
        let report = execute_distributed_cube_job(&cfg, &payload, 1, 1_700_000_000)
            .expect("Cube job should succeed");

        let packet = KlmDispatchPacket::from_execution_report(&report, WaveMode::Wave64);
        assert_eq!(core::mem::align_of::<KlmSlotDescriptor>(), 64);
        assert_eq!(core::mem::align_of::<KlmDispatchPacket>(), 64);
        assert!(packet.validate_hardware_constraints().is_ok());

        // Verify all 8 vertices map to banks 0..7 without any collision
        for (i, slot) in packet.slots.iter().enumerate() {
            assert_eq!(slot.lds_primary_bank, i as u8);
            assert_eq!((slot.dma_recipe_offset as usize) % KLM_DMA_ALIGNMENT, 0);
        }
    }

    #[test]
    fn test_klm_wave64_dpp_butterfly_and_q8_tensor_tile() {
        let mut vgpr = [0u32; WAVE64_LANES];
        for i in 0..WAVE64_LANES {
            vgpr[i] = (i as u32) + 1;
        }
        let b_report = execute_wave64_dpp_butterfly(&vgpr);
        // Sum of 1..=64 is 64 * 65 / 2 = 2080
        assert_eq!(b_report.wave64_total_sum, 2080);
        // Subcube 0 sum is 1+2+3+4+5+6+7+8 = 36
        assert_eq!(b_report.subcube_sums[0], 36);

        // Test Q8 Tensor Tile formula synthesis and Wave64 dot product
        let weights = [2i8; 64];
        let activations = [3i8; 64];
        let tile = KlmTensorTileQ8::synthesize_tile(0, &weights, 65536, 0, None);
        // Since all weights are identical (2), KpackFormula automatically compresses the tensor tile with RLE (1 byte)!
        assert_eq!(tile.formula.payload_len, 1);

        let (dot, _) = tile
            .execute_wave64_dot_product(None, 0, &activations)
            .expect("Wave64 dot product from RLE formula");
        assert_eq!(dot, 64 * 2 * 3);
    }
}
