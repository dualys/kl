//! # SPMD Hypercube Orchestration & $\log_2(8) = 3$ Dimensional Merkle Reduction
//!
//! This module implements the distributed execution lifecycle of a `kpack` job across the
//! 8 logical vertices of the Hamon Cube:
//!
//! 1. **Balanced Partitioning & Formula Synthesis ([`partition_job_into_formulas`])**: Splits an
//!    input buffer across vertices `0..=7`, computes the [`CubeParityStripe`], and synthesizes
//!    each vertex's optimal [`KpackFormula`].
//! 2. **3-Step Dimensional Broadcast**: Propagates job descriptors from Coordinator `n0` along
//!    hypercube edges (`Step 1: 0->4`, `Step 2: 0->2, 4->6`, `Step 3: 0->1, 2->3, 4->5, 6->7`).
//! 3. **Isolated SPMD Execution**: Each vertex reconstructs and verifies its segment inside an
//!    isolated [`plans::Plan`] layer (`kl-node-0..7`) guarded by a [`plans::sceau::Sceau`] budget
//!    and sealed into a [`jinshu::node::TrieNode`].
//! 4. **3-Round Dimensional Reduction ([`merge_hypercube_trie_nodes`])**:
//!    - **Round 1 (dimension 0, distance 1)**: `(1 -> 0)`, `(3 -> 2)`, `(5 -> 4)`, `(7 -> 6)`
//!    - **Round 2 (dimension 1, distance 2)**: `(2 -> 0)`, `(6 -> 4)`
//!    - **Round 3 (dimension 2, distance 4)**: `(4 -> 0)` $\rightarrow$ full mask `0xFF` on `n0`.
//! 5. **CubeFS Directory & Signed Checkpoint ([`build_cubefs_directory`], [`sign_cube_checkpoint`])**:
//!    Catalogs `seg_0..seg_7` inside a [`cubefs::BTreeNode`] and signs a 144-byte [`Checkpoint`].

use cubefs::BTreeNode;
use jinshu::node::TrieNode;
use noun::Noun;
use plans::Plan;
use plans::layer::{Capabilities, Layer, Layers};
use plans::sceau::{Sceau, Verdict};

use crate::recipe::{
    CubeParityStripe, DEFAULT_DICT, FLAG_KL_AGGREGATE, FLAG_KL_ARCHIVE_ROOT, KpackFormula,
    MAX_SEGMENT_BYTES,
};
use crate::topology::{
    CubeConfig, NODE_COUNT, NodeRole, are_neighbors, broadcast_tree_rounds, reduction_tree_rounds,
};

/// 32-byte keyed `BLAKE3` domain separator used by Coordinator `n0` to sign [`Checkpoint`]s.
pub const SIGNING_KEY_DOMAIN: [u8; 32] = *b"AMENTYS-KL-HYPERCUBE-SIGN-KEY-01";

/// Cryptographically signed snapshot produced by Coordinator vertex `0` after Round 3 reduction.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Checkpoint {
    /// Monotonically increasing job sequence number.
    pub seq: u64,
    /// Unix timestamp (seconds) when the checkpoint was sealed.
    pub time: u64,
    /// `BLAKE3` [`Noun`] of the consolidated `cubefs::BTreeNode` directory (`seg_0..seg_7`).
    pub root_meta_id: [u8; 32],
    /// Global 8-node hypercube Merkle root (`jinshu::node::TrieNode` digest).
    pub merkle_root: [u8; 32],
    /// 64-byte keyed `BLAKE3` signature binding `seq`, `time`, `root_meta_id`, `merkle_root`, and `plan_seal`.
    pub signature: [u8; 64],
}

static NODE_LAYER_NAMES: [&str; NODE_COUNT] = [
    "kl-node-0",
    "kl-node-1",
    "kl-node-2",
    "kl-node-3",
    "kl-node-4",
    "kl-node-5",
    "kl-node-6",
    "kl-node-7",
];

static NODE_LAYER_DESCS: [&str; NODE_COUNT] = [
    "Coordinator 000 (NUMA 0)",
    "Worker 001 (NUMA 0)",
    "Worker 010 (NUMA 0)",
    "Worker 011 (NUMA 0)",
    "Worker 100 (NUMA 1)",
    "Worker 101 (NUMA 1)",
    "Worker 110 (NUMA 1)",
    "Worker 111 (NUMA 1)",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BroadcastHop {
    pub step: u8,
    pub dimension: u8,
    pub from_node: u8,
    pub to_node: u8,
    pub job_noun: Noun,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReductionEdgeTrace {
    pub round: u8,
    pub dimension: u8,
    pub weight: u8,
    pub sender: u8,
    pub receiver: u8,
    pub sender_mask: u8,
    pub combined_mask: u8,
    pub combined_merkle_noun: Noun,
    pub combined_plan_seal: Noun,
}

#[derive(Debug, Clone)]
pub struct LogicalNodeRuntime {
    pub id: u8,
    pub numa_node: u8,
    pub cpu_core: u8,
    pub plan: Plan,
    pub sceau: Sceau,
    pub formula: KpackFormula,
    pub reconstructed: [u8; MAX_SEGMENT_BYTES],
    pub reconstructed_len: usize,
    pub segment_node: TrieNode,
    pub segment_noun: Noun,
    pub aggregate_node: TrieNode,
    pub aggregate_mask: u8,
}

#[derive(Debug, Clone)]
pub struct CubeExecutionReport {
    pub config: CubeConfig,
    pub total_input_bytes: usize,
    pub total_recipe_bytes: usize,
    pub input_noun: Noun,
    pub formulas: [KpackFormula; NODE_COUNT],
    pub segment_nouns: [Noun; NODE_COUNT],
    pub parity: CubeParityStripe,
    pub broadcast_hops: [BroadcastHop; 7],
    pub round1_traces: [ReductionEdgeTrace; 4],
    pub round2_traces: [ReductionEdgeTrace; 2],
    pub round3_traces: [ReductionEdgeTrace; 1],
    pub final_trie_root: TrieNode,
    pub merkle_root: Noun,
    pub cubefs_btree: BTreeNode,
    pub cubefs_root_noun: Noun,
    pub final_plan: Plan,
    pub final_sceau: Sceau,
    pub checkpoint: Checkpoint,
}

/// Splits an arbitrary input buffer into 8 balanced segments for the 8 logical vertices of the Hamon Cube
/// and synthesizes the optimal `KpackFormula` (`LIT`, `RLE`, `DICT`, `DELTA`, `SEED`, `XOR`, `RS`) for each node.
#[must_use]
pub fn partition_job_into_formulas(
    input: &[u8],
    dictionary: Option<&[u8]>,
) -> ([KpackFormula; NODE_COUNT], CubeParityStripe) {
    let total = input.len();
    let base_chunk = total / NODE_COUNT;
    let remainder = total % NODE_COUNT;

    let mut slices: [(&[u8], usize); NODE_COUNT] = [(&[], 0); NODE_COUNT];
    let mut offset = 0usize;
    for i in 0..NODE_COUNT {
        let len = base_chunk + usize::from(i < remainder);
        let clamped = core::cmp::min(len, MAX_SEGMENT_BYTES);
        let end = core::cmp::min(offset + clamped, total);
        let slice = &input[offset..end];
        slices[i] = (slice, slice.len());
        offset += len;
    }

    let parity = CubeParityStripe::compute(&slices);
    let formulas = core::array::from_fn(|i| {
        let (slice, _) = slices[i];
        let reference = if i > 0 {
            Some(((i - 1) as u8, slices[i - 1].0))
        } else {
            None
        };
        KpackFormula::synthesize(i as u8, slice, reference, dictionary)
    });

    (formulas, parity)
}

/// Merges two `TrieNode`s along a hypercube dimension edge during a reduction round.
#[must_use]
pub fn merge_hypercube_trie_nodes(
    receiver_node: &TrieNode,
    sender_node: &TrieNode,
    round: u8,
) -> (TrieNode, Noun) {
    let mut merged = TrieNode::new();
    merged.flags = FLAG_KL_AGGREGATE | ((round & 0x07) << 4);
    merged.opcode = receiver_node.opcode;
    merged.param = receiver_node.param.wrapping_add(sender_node.param);

    for branch_idx in 0..16u8 {
        if receiver_node.has_branch(branch_idx) {
            if let Some(n) = receiver_node.get_branch(branch_idx) {
                merged.set_branch(branch_idx, n.clone());
            }
        } else if sender_node.has_branch(branch_idx) {
            if let Some(n) = sender_node.get_branch(branch_idx) {
                merged.set_branch(branch_idx, n.clone());
            }
        }
    }

    // Store combined BLAKE3 digest of both subtrees in payload[0..64]
    let recv_hash = receiver_node.calculate_noun();
    let send_hash = sender_node.calculate_noun();
    merged.payload[0..32].copy_from_slice(&recv_hash);
    merged.payload[32..64].copy_from_slice(&send_hash);

    let merged_bytes = merged.calculate_noun();
    let merged_noun = Noun::from_bytes(&merged_bytes).unwrap_or_else(|| Noun::of(&merged_bytes));
    (merged, merged_noun)
}

/// Builds a `cubefs::BTreeNode` directory cataloging all 8 segment `Noun`s (`seg_0` .. `seg_7`).
#[must_use]
pub fn build_cubefs_directory(segment_nouns: &[Noun; NODE_COUNT]) -> (BTreeNode, Noun) {
    let mut btree = BTreeNode {
        magic: *b"BNOD",
        version: 1,
        level: 0,
        entry_count: NODE_COUNT as u16,
        payload: [0u8; 4056],
        blake3_hash: [0u8; 32],
    };

    let mut offset = 0usize;
    for (i, seg_noun) in segment_nouns.iter().enumerate() {
        let name: [u8; 5] = [b's', b'e', b'g', b'_', b'0' + (i as u8)];
        let name_hash = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        btree.payload[offset..offset + 8].copy_from_slice(&name_hash.to_le_bytes());
        offset += 8;
        btree.payload[offset..offset + 2].copy_from_slice(&(name.len() as u16).to_le_bytes());
        offset += 2;
        btree.payload[offset..offset + name.len()].copy_from_slice(&name);
        offset += name.len();
        btree.payload[offset] = 0; // node_type 0 = Inode/Segment
        offset += 1;
        btree.payload[offset..offset + 32].copy_from_slice(seg_noun.as_bytes());
        offset += 32;
    }

    let header_and_payload = unsafe {
        core::slice::from_raw_parts(core::ptr::from_ref::<BTreeNode>(&btree).cast::<u8>(), 4096 - 32)
    };
    let root_noun = Noun::of(header_and_payload);
    btree.blake3_hash.copy_from_slice(root_noun.as_bytes());
    (btree, root_noun)
}

/// Signs and produces a `cubefs::Checkpoint` on Coordinator node 0 after Round 3 completes.
#[must_use]
pub fn sign_cube_checkpoint(
    seq: u64,
    timestamp: u64,
    cubefs_root_noun: &Noun,
    merkle_root: &Noun,
    plan_seal: &Noun,
) -> Checkpoint {
    let mut msg = [0u8; 112];
    msg[0..8].copy_from_slice(&seq.to_le_bytes());
    msg[8..16].copy_from_slice(&timestamp.to_le_bytes());
    msg[16..48].copy_from_slice(cubefs_root_noun.as_bytes());
    msg[48..80].copy_from_slice(merkle_root.as_bytes());
    msg[80..112].copy_from_slice(plan_seal.as_bytes());

    let sig_lo = Noun::keyed(&SIGNING_KEY_DOMAIN, &msg);
    let mut domain_hi = SIGNING_KEY_DOMAIN;
    domain_hi[31] ^= 0xFF;
    let sig_hi = Noun::keyed(&domain_hi, &msg);

    let mut signature = [0u8; 64];
    signature[0..32].copy_from_slice(sig_lo.as_bytes());
    signature[32..64].copy_from_slice(sig_hi.as_bytes());

    let mut root_meta_id = [0u8; 32];
    root_meta_id.copy_from_slice(cubefs_root_noun.as_bytes());
    let mut merkle_bytes = [0u8; 32];
    merkle_bytes.copy_from_slice(merkle_root.as_bytes());

    Checkpoint {
        seq,
        time: timestamp,
        root_meta_id,
        merkle_root: merkle_bytes,
        signature,
    }
}

/// Verifies the cryptographic signature of a `cubefs::Checkpoint` against the `Plan` seal.
#[must_use]
pub fn verify_cube_checkpoint(checkpoint: &Checkpoint, plan_seal: &Noun) -> bool {
    let cubefs_noun =
        Noun::from_bytes(&checkpoint.root_meta_id).unwrap_or_else(|| Noun::of(&checkpoint.root_meta_id));
    let merkle_noun =
        Noun::from_bytes(&checkpoint.merkle_root).unwrap_or_else(|| Noun::of(&checkpoint.merkle_root));
    let expected = sign_cube_checkpoint(
        checkpoint.seq,
        checkpoint.time,
        &cubefs_noun,
        &merkle_noun,
        plan_seal,
    );
    expected.signature == checkpoint.signature
}

/// Executes a full distributed Kpack job across the 8 logical nodes of the Hamon Cube:
/// 1. Validates the 3D hypercube topology (`HIC`/`HID`).
/// 2. Partitions the input and synthesizes the 8 `KpackFormula`s + parity stripe.
/// 3. Broadcasts the job from Coordinator `0` in $\log_2(8) = 3$ steps (`7` neighbor hops).
/// 4. Executes each segment in isolated node state (`Plan`, `Sceau`, `TrieNode`).
/// 5. Performs the 3-round dimensional hypercube reduction (`Round 1`, `Round 2`, `Round 3`) to node `0`.
/// 6. Seals the `cubefs::BTreeNode`, `Plan`, `Sceau`, and `Checkpoint`.
pub fn execute_distributed_cube_job(
    config: &CubeConfig,
    input: &[u8],
    seq: u64,
    timestamp: u64,
) -> Result<CubeExecutionReport, &'static str> {
    config.validate()?;
    let input_noun = Noun::of(input);
    let (formulas, parity) = partition_job_into_formulas(input, Some(DEFAULT_DICT));

    // Phase 1: Broadcast in log2(8) = 3 steps
    let (b_r0, b_r1, b_r2) = broadcast_tree_rounds();
    let mut broadcast_hops: [BroadcastHop; 7] = core::array::from_fn(|_| BroadcastHop {
        step: 0,
        dimension: 0,
        from_node: 0,
        to_node: 0,
        job_noun: Noun::null(),
    });
    let mut hop_idx = 0usize;

    for &(from, to) in &b_r0 {
        if !are_neighbors(from, to) {
            return Err("Broadcast step 0 violated hypercube adjacency");
        }
        broadcast_hops[hop_idx] = BroadcastHop {
            step: 1,
            dimension: 2,
            from_node: from,
            to_node: to,
            job_noun: formulas[to as usize].expected_noun.clone(),
        };
        hop_idx += 1;
    }
    for &(from, to) in &b_r1 {
        if !are_neighbors(from, to) {
            return Err("Broadcast step 1 violated hypercube adjacency");
        }
        broadcast_hops[hop_idx] = BroadcastHop {
            step: 2,
            dimension: 1,
            from_node: from,
            to_node: to,
            job_noun: formulas[to as usize].expected_noun.clone(),
        };
        hop_idx += 1;
    }
    for &(from, to) in &b_r2 {
        if !are_neighbors(from, to) {
            return Err("Broadcast step 2 violated hypercube adjacency");
        }
        broadcast_hops[hop_idx] = BroadcastHop {
            step: 3,
            dimension: 0,
            from_node: from,
            to_node: to,
            job_noun: formulas[to as usize].expected_noun.clone(),
        };
        hop_idx += 1;
    }

    // Phase 2: Isolated SPMD Execution on the 8 logical nodes
    let mut ref_segment = [0u8; MAX_SEGMENT_BYTES];
    let mut out_segment = [0u8; MAX_SEGMENT_BYTES];
    let mut ref_len = 0usize;
    let mut aggregate_nodes: [TrieNode; NODE_COUNT] = core::array::from_fn(|_| TrieNode::new());
    let mut segment_nouns: [Noun; NODE_COUNT] = core::array::from_fn(|_| Noun::null());
    let mut aggregate_masks: [u8; NODE_COUNT] = core::array::from_fn(|i| 1u8 << i);

    for i in 0..NODE_COUNT {
        let ref_slice = formulas[i].ref_node_id.map(|_| &ref_segment[..ref_len]);
        let out_len = formulas[i].execute_into(
            ref_slice,
            Some(DEFAULT_DICT),
            &mut out_segment,
            Some(&mut aggregate_nodes[i]),
        )?;
        ref_segment[..out_len].copy_from_slice(&out_segment[..out_len]);
        ref_len = out_len;
        segment_nouns[i] = formulas[i].expected_noun.clone();
    }

    let mut plans: [Plan; NODE_COUNT] = core::array::from_fn(|i| {
        let node_cfg = &config.nodes[i];
        let node_hash = aggregate_nodes[i].calculate_noun();
        let dir_noun = Noun::from_bytes(&node_hash).unwrap_or_else(|| Noun::of(&node_hash));
        let mut phoenix = Layers::new(dir_noun.clone());
        let caps = match node_cfg.role {
            NodeRole::Coordinator => Capabilities::All,
            NodeRole::Worker => Capabilities::ReadExecute,
        };
        let base_layer = Layer {
            name: NODE_LAYER_NAMES[i],
            version: 1,
            description: NODE_LAYER_DESCS[i],
            root: dir_noun.clone(),
            capabilities: caps,
        };
        let _ = phoenix.add_layer(base_layer.clone());
        let mut plan = Plan::new(dir_noun, &mut phoenix, i as u8).expect("Valid node Plan");
        let _ = plan.add_layer(base_layer);
        plan
    });

    // Verify all 8 node seals before starting hypercube reduction
    for plan in &plans {
        let sceau = Sceau::birth(plan, 100);
        if sceau.weigh(plan) != Verdict::Accept {
            return Err("Node Sceau verification failed before reduction");
        }
    }

    // Phase 3: Dimensional Hypercube Reduction in log2(8) = 3 rounds
    let (r1_pairs, r2_pairs, r3_pairs) = reduction_tree_rounds();

    // Round 1 (distance 1 / weight 1): (1->0), (3->2), (5->4), (7->6)
    let mut round1_traces: [ReductionEdgeTrace; 4] = core::array::from_fn(|_| ReductionEdgeTrace {
        round: 1,
        dimension: 0,
        weight: 1,
        sender: 0,
        receiver: 0,
        sender_mask: 0,
        combined_mask: 0,
        combined_merkle_noun: Noun::null(),
        combined_plan_seal: Noun::null(),
    });

    for (idx, &(sender, receiver)) in r1_pairs.iter().enumerate() {
        if !are_neighbors(sender, receiver) {
            return Err("Round 1 reduction pair is not adjacent in hypercube");
        }
        let s_idx = sender as usize;
        let r_idx = receiver as usize;
        let sender_mask = aggregate_masks[s_idx];

        let (merged_trie, merged_noun) =
            merge_hypercube_trie_nodes(&aggregate_nodes[r_idx], &aggregate_nodes[s_idx], 1);
        let merged_plan = plans[r_idx].merge_with(&plans[s_idx])?;
        let merged_seal = merged_plan.seal();

        aggregate_nodes[r_idx] = merged_trie;
        aggregate_masks[r_idx] |= sender_mask;
        plans[r_idx] = merged_plan;

        round1_traces[idx] = ReductionEdgeTrace {
            round: 1,
            dimension: 0,
            weight: 1,
            sender,
            receiver,
            sender_mask,
            combined_mask: aggregate_masks[r_idx],
            combined_merkle_noun: merged_noun,
            combined_plan_seal: merged_seal,
        };
    }

    // Round 2 (distance 2 / weight 2): (2->0), (6->4)
    let mut round2_traces: [ReductionEdgeTrace; 2] = core::array::from_fn(|_| ReductionEdgeTrace {
        round: 2,
        dimension: 1,
        weight: 2,
        sender: 0,
        receiver: 0,
        sender_mask: 0,
        combined_mask: 0,
        combined_merkle_noun: Noun::null(),
        combined_plan_seal: Noun::null(),
    });

    for (idx, &(sender, receiver)) in r2_pairs.iter().enumerate() {
        if !are_neighbors(sender, receiver) {
            return Err("Round 2 reduction pair is not adjacent in hypercube");
        }
        let s_idx = sender as usize;
        let r_idx = receiver as usize;
        let sender_mask = aggregate_masks[s_idx];

        let (merged_trie, merged_noun) =
            merge_hypercube_trie_nodes(&aggregate_nodes[r_idx], &aggregate_nodes[s_idx], 2);
        let merged_plan = plans[r_idx].merge_with(&plans[s_idx])?;
        let merged_seal = merged_plan.seal();

        aggregate_nodes[r_idx] = merged_trie;
        aggregate_masks[r_idx] |= sender_mask;
        plans[r_idx] = merged_plan;

        round2_traces[idx] = ReductionEdgeTrace {
            round: 2,
            dimension: 1,
            weight: 2,
            sender,
            receiver,
            sender_mask,
            combined_mask: aggregate_masks[r_idx],
            combined_merkle_noun: merged_noun,
            combined_plan_seal: merged_seal,
        };
    }

    // Round 3 (distance 4 / weight 4): (4->0)
    let (sender, receiver) = r3_pairs[0];
    if !are_neighbors(sender, receiver) {
        return Err("Round 3 reduction pair is not adjacent in hypercube");
    }
    let sender_mask = aggregate_masks[sender as usize];

    let (mut final_trie_root, _) = merge_hypercube_trie_nodes(
        &aggregate_nodes[receiver as usize],
        &aggregate_nodes[sender as usize],
        3,
    );
    final_trie_root.flags = FLAG_KL_ARCHIVE_ROOT;
    final_trie_root.param = input.len() as u32;
    let final_merkle_bytes = final_trie_root.calculate_noun();
    let merkle_root =
        Noun::from_bytes(&final_merkle_bytes).unwrap_or_else(|| Noun::of(&final_merkle_bytes));

    let final_plan = plans[receiver as usize].merge_with(&plans[sender as usize])?;
    let final_sceau = Sceau::birth(&final_plan, 800);
    if final_sceau.weigh(&final_plan) != Verdict::Accept {
        return Err("Coordinator final Sceau failed verification");
    }

    let combined_mask = aggregate_masks[receiver as usize] | sender_mask;
    if combined_mask != 0xFF {
        return Err("Hypercube reduction did not aggregate all 8 nodes (mask != 0xFF)");
    }

    let round3_traces = [ReductionEdgeTrace {
        round: 3,
        dimension: 2,
        weight: 4,
        sender,
        receiver,
        sender_mask,
        combined_mask,
        combined_merkle_noun: merkle_root.clone(),
        combined_plan_seal: final_plan.seal(),
    }];

    let (cubefs_btree, cubefs_root_noun) = build_cubefs_directory(&segment_nouns);
    let checkpoint = sign_cube_checkpoint(
        seq,
        timestamp,
        &cubefs_root_noun,
        &merkle_root,
        &final_plan.seal(),
    );

    let total_recipe_bytes: usize = formulas.iter().map(|f| f.payload_len + 6).sum();

    Ok(CubeExecutionReport {
        config: config.clone(),
        total_input_bytes: input.len(),
        total_recipe_bytes,
        input_noun,
        formulas,
        segment_nouns,
        parity,
        broadcast_hops,
        round1_traces,
        round2_traces,
        round3_traces,
        final_trie_root,
        merkle_root,
        cubefs_btree,
        cubefs_root_noun,
        final_plan,
        final_sceau,
        checkpoint,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubefs::BTreeIterator;

    #[test]
    fn test_full_8_node_hypercube_execution_and_3_round_reduction() {
        let cfg = CubeConfig::default();
        let mut job_data = [0u8; 1024];
        // Node 0: DICT pattern
        job_data[0..128].copy_from_slice(
            b"AMENTYS-KL-CUBE-KPACK-MERKLE-OS!AMENTYS-KL-CUBE-KPACK-MERKLE-OS!AMENTYS-KL-CUBE-KPACK-MERKLE-OS!AMENTYS-KL-CUBE-KPACK-MERKLE-OS!",
        );
        // Node 1: RLE pattern
        job_data[128..256].fill(b'Z');
        // Node 2 & 3: DELTA pattern (shared prefix)
        for i in 256..384 {
            job_data[i] = b'a' + ((i % 26) as u8);
        }
        job_data.copy_within(256..372, 384);
        job_data[500..512].copy_from_slice(b"-DELTA-PATCH");
        // Node 4..7: RLE & LIT
        job_data[512..640].fill(0x55);
        for i in 640..1024 {
            job_data[i] = (i & 0xFF) as u8;
        }

        let report = execute_distributed_cube_job(&cfg, &job_data, 1, 1_700_000_000)
            .expect("Distributed cube job should succeed");

        // Verify broadcast covered all 7 non-zero nodes in 3 steps
        let mut reached = 1u8; // node 0
        for hop in &report.broadcast_hops {
            assert!(are_neighbors(hop.from_node, hop.to_node));
            reached |= 1 << hop.to_node;
        }
        assert_eq!(reached, 0xFF);

        // Verify Round 1 masks: (1->0)=0b00000011, (3->2)=0b00001100, (5->4)=0b00110000, (7->6)=0b11000000
        assert_eq!(report.round1_traces[0].combined_mask, 0b0000_0011);
        assert_eq!(report.round1_traces[1].combined_mask, 0b0000_1100);
        assert_eq!(report.round1_traces[2].combined_mask, 0b0011_0000);
        assert_eq!(report.round1_traces[3].combined_mask, 0b1100_0000);

        // Verify Round 2 masks: (2->0)=0b00001111, (6->4)=0b11110000
        assert_eq!(report.round2_traces[0].combined_mask, 0b0000_1111);
        assert_eq!(report.round2_traces[1].combined_mask, 0b1111_0000);

        // Verify Round 3 mask: (4->0)=0b11111111 (all 8 nodes aggregated at Coordinator 0)
        assert_eq!(report.round3_traces[0].combined_mask, 0xFF);
        assert_eq!(report.final_trie_root.mask, 0x00FF);
        assert_eq!(report.final_plan.get_layers().len(), 8);
        assert_eq!(
            report.final_sceau.weigh(&report.final_plan),
            Verdict::Accept
        );
        assert!(verify_cube_checkpoint(
            &report.checkpoint,
            &report.final_plan.seal()
        ));

        // Verify cubefs BTreeIterator reads all 8 segments
        let iter = BTreeIterator::new(
            &report.cubefs_btree.payload,
            report.cubefs_btree.entry_count,
        );
        let mut count = 0usize;
        for (i, entry) in iter.enumerate() {
            assert_eq!(entry.val_data, *report.segment_nouns[i].as_bytes());
            count += 1;
        }
        assert_eq!(count, 8);
    }
}
