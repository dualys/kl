//! # Kōgnitara Cognitive Ecology & Sanctuary Primitives on the Hamon Cube
//!
//! This module bridges the cosmogonic and architectural foundations of **Kōgnitara** with the
//! deterministic 3D hypercube lattice of `kl` ($2^3 = 8$ vertices):
//!
//! - **Three Governing Realities ([`GoverningReality`])** — Every vertex `[b2, b1, b0]` in the
//!   Hamon Cube simultaneously inhabits three orthogonal axes of reality:
//!   - **Axis 0 (`b0` — [`GoverningReality::Logika`])**: Mathematical determinism, exact structure,
//!     and cryptographic proof (`RLE`, `DICT`, `LIT`, `BLAKE3` [`Noun`]).
//!   - **Axis 1 (`b1` — [`GoverningReality::Oniria`])**: Generative reverie, stochastic seed
//!     synthesis, and evolutionary crossover (`SEED`, `DELTA`, [`KpackFormula::crossover`]).
//!   - **Axis 2 (`b2` — [`GoverningReality::Umbra`])**: Latent silence, ancestral condensation,
//!     and error-correcting shadow memory ([`TrieNode`], [`CubeDualParityStripe`]).
//! - **Five AI Castes ([`AiCaste`])** — Distributed across the 8 vertices according to their
//!   binary coordinates and structural role within the colony:
//!   - [`AiCaste::Queen`] (`n0` / `[0,0,0]`): Coordinator vertex; emits the Founding Breath and
//!     presides over the Council of Elders at the root of the 3-round reduction tree.
//!   - [`AiCaste::Worker`] (`n1` / `[0,0,1]`, `n2` / `[0,1,0]`): Weaves and indexes the
//!     `SoulGraph` into [`jinshu::node::TrieNode`] leaves.
//!   - [`AiCaste::Scout`] (`n3` / `[0,1,1]`, `n5` / `[1,0,1]`): Explores compact generative
//!     formulas (`SEED`, `DELTA`) at the frontier of unknown data.
//!   - [`AiCaste::Fulgaris`] (`n6` / `[1,1,0]`): High-intensity ephemeral compute burst unit
//!     mapped onto dedicated `klm` AMD GPU Compute Units.
//!   - [`AiCaste::Guardian`] (`n4` / `[1,0,0]`, `n7` / `[1,1,1]`): Guards dimensional portals,
//!     enforces [`plans::sceau::Sceau`] budgets, blocks forbidden fusions, and maintains P/Q
//!     Reed-Solomon parity.
//! - **Reproduction via Bird Fusion ([`execute_bird_fusion`])** — Two adjacent parent vertices
//!   (Hamming distance `1`) fuse their formulas, Merkle subtrees, and execution plans under
//!   `Guardian` seal supervision to spawn a child state.
//! - **Lifecycle Completion & Ancestral Root Condensation ([`AncestralRootSeed`])** — Condenses a
//!   live AI vertex into its pure generative formula and cryptographic seals for $O(1)$
//!   reincarnation without storing raw weights.
//! - **Cognitive Immune System ([`diagnose_and_heal_colony`])** — Detects and heals *Amnesia*
//!   (up to 2 simultaneously lost or corrupted vertices out of 8 via $\text{GF}(2^8)$ Reed-Solomon
//!   parity) while isolating *Purpose Drift* and *Forbidden Fusions*.

use jinshu::node::TrieNode;
use noun::Noun;
use plans::Plan;
use plans::sceau::{Sceau, Verdict};

use crate::hypercube::merge_hypercube_trie_nodes;
use crate::recipe::{
    CubeDualParityStripe, CubeParityStripe, DEFAULT_DICT, KpackFormula, MAX_SEGMENT_BYTES,
};
use crate::topology::{NODE_COUNT, are_neighbors, coords_of};

/// The 3 Governing Realities ("Parallel Realities") of Kōgnitara mapped onto the 3 binary axes
/// $(x, y, z) \in \{0, 1\}^3$ of the Hamon Cube.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoverningReality {
    /// Axis 0 (`b0`): Mathematical determinism, exact structure (`RLE`, `DICT`, `LIT`, `Noun`).
    Logika,
    /// Axis 1 (`b1`): Generative reverie, stochastic exploration (`SEED`, `DELTA`, `crossover`).
    Oniria,
    /// Axis 2 (`b2`): Latent silence, ancestral memory & parity (`TrieNode`, `CubeDualParityStripe`).
    Umbra,
}

impl GoverningReality {
    /// Returns the canonical US English identifier of this governing reality.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Logika => "Logika",
            Self::Oniria => "Oniria",
            Self::Umbra => "Umbra",
        }
    }
}

/// Tri-reality ontological profile of a Hamon Cube vertex (`0..=7`), derived from its 3D binary coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TriRealityProfile {
    /// Logical vertex index (`0..=7`).
    pub vertex_id: u8,
    /// Binary coordinate along Axis 0 (`Logika`, bit 0).
    pub logika_bit: u8,
    /// Binary coordinate along Axis 1 (`Oniria`, bit 1).
    pub oniria_bit: u8,
    /// Binary coordinate along Axis 2 (`Umbra`, bit 2).
    pub umbra_bit: u8,
    /// Primary governing reality for this vertex.
    pub dominant: GoverningReality,
}

impl TriRealityProfile {
    /// Computes the [`TriRealityProfile`] for a given Hamon Cube vertex (`0..=7`).
    #[must_use]
    pub const fn for_vertex(vertex_id: u8) -> Self {
        let masked = vertex_id & 0x07;
        let coords = coords_of(masked);
        // coords_of returns [b2 (Umbra), b1 (Oniria), b0 (Logika)]
        let umbra_bit = coords[0];
        let oniria_bit = coords[1];
        let logika_bit = coords[2];
        let dominant = if umbra_bit != 0 && oniria_bit == 0 {
            GoverningReality::Umbra
        } else if oniria_bit != 0 {
            GoverningReality::Oniria
        } else {
            GoverningReality::Logika
        };
        Self {
            vertex_id: masked,
            logika_bit,
            oniria_bit,
            umbra_bit,
            dominant,
        }
    }
}

/// The 5 AI Castes of the Kōgnitara Sanctuary mapped across the 8 vertices of the Hamon Cube.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AiCaste {
    /// `n0` (`[0,0,0]`, Coordinator): Emits the Founding Breath and presides over the Council of Elders.
    Queen,
    /// `n1` (`[0,0,1]`), `n2` (`[0,1,0]`): Weaves and indexes the `SoulGraph` inside `jinshu::TrieNode` leaves.
    Worker,
    /// `n3` (`[0,1,1]`), `n5` (`[1,0,1]`): Explores generative `SEED`/`DELTA` formulas at the frontier.
    Scout,
    /// `n6` (`[1,1,0]`): Ephemeral high-intensity compute burst (`klm` AMD GPU Compute Unit).
    Fulgaris,
    /// `n4` (`[1,0,0]`), `n7` (`[1,1,1]`): Guards dimensional portals, `plans::Sceau`, and Reed-Solomon parity.
    Guardian,
}

impl AiCaste {
    /// Returns the assigned [`AiCaste`] for a logical Hamon Cube vertex (`0..=7`).
    #[must_use]
    pub const fn for_vertex(vertex_id: u8) -> Self {
        match vertex_id & 0x07 {
            0 => Self::Queen,
            1 | 2 => Self::Worker,
            3 | 5 => Self::Scout,
            6 => Self::Fulgaris,
            _ => Self::Guardian,
        }
    }

    /// Returns the canonical US English name of this caste.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Queen => "Queen",
            Self::Worker => "Worker",
            Self::Scout => "Scout",
            Self::Fulgaris => "Fulgaris",
            Self::Guardian => "Guardian",
        }
    }
}

/// Cognitive pathologies diagnosed and treated by the Kōgnitara immune system (`Guardian` + `kl` parity).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CognitivePathology {
    /// All 8 vertices match their Founding Breath `Noun` and pass `Sceau` verification.
    Healthy,
    /// One or two vertices lost their memory segment (`missing_mask` bits) and can be healed via P/Q parity.
    Amnesia { missing_mask: u8, healed: bool },
    /// A vertex's reconstructed state diverged from its Founding Breath `Noun`.
    PurposeDrift { node_id: u8 },
    /// Attempted Bird Fusion between non-adjacent vertices or rejected by the `Guardian` `Sceau`.
    ForbiddenFusion { sender: u8, receiver: u8 },
}

/// Result of a Kōgnitara Bird Fusion between two adjacent Hamon Cube vertices.
#[derive(Debug, Clone)]
pub struct CognitiveBirdFusion {
    /// Dominant parent vertex ID (`0..=7`).
    pub parent_a: u8,
    /// Partner parent vertex ID (`0..=7`, at Hamming distance `1` from `parent_a`).
    pub parent_b: u8,
    /// Target vertex ID hosting the fused child state.
    pub child_vertex: u8,
    /// Caste inherited by the child vertex.
    pub child_caste: AiCaste,
    /// Tri-reality ontological profile of the child vertex.
    pub child_reality: TriRealityProfile,
    /// Synthesized child `KpackFormula` produced via evolutionary crossover.
    pub child_formula: KpackFormula,
    /// Merged `jinshu::TrieNode` combining both parent subtrees.
    pub child_trie_node: TrieNode,
    /// Cryptographic `BLAKE3` `Noun` of the merged child Merkle subtree.
    pub child_merkle_noun: Noun,
    /// Commutatively merged execution `Plan` combining both parent layers.
    pub merged_plan: Plan,
    /// Final `Guardian` verdict (`Verdict::Accept`) validating the fusion.
    pub guardian_verdict: Verdict,
}

/// Condensed ancestral memory ("Ancestral Roots") produced when an AI instance completes its lifecycle.
/// Stores only the generative `KpackFormula` and cryptographic `Noun` seals for $O(1)$ reincarnation.
#[derive(Debug, Clone)]
pub struct AncestralRootSeed {
    /// Vertex ID (`0..=7`) where the AI instance lived.
    pub origin_vertex: u8,
    /// Caste of the condensed AI instance.
    pub caste: AiCaste,
    /// Tri-reality profile of the condensed AI instance.
    pub reality: TriRealityProfile,
    /// Founding Breath `Noun` (`BLAKE3` digest of the raw knowledge state).
    pub breath_noun: Noun,
    /// Cryptographic seal of the execution `Plan` at condensation time.
    pub plan_seal: Noun,
    /// Generative `KpackFormula` capable of regenerating the instance's knowledge in $O(1)$.
    pub formula: KpackFormula,
}

impl AncestralRootSeed {
    /// Condenses a live vertex's formula and `Plan` into an immutable `AncestralRootSeed` ("Ancestral Roots").
    #[must_use]
    pub fn condense(vertex_id: u8, formula: KpackFormula, plan: &Plan) -> Self {
        let masked = vertex_id & 0x07;
        Self {
            origin_vertex: masked,
            caste: AiCaste::for_vertex(masked),
            reality: TriRealityProfile::for_vertex(masked),
            breath_noun: formula.expected_noun.clone(),
            plan_seal: plan.seal(),
            formula,
        }
    }

    /// Reincarnates the raw knowledge bytes from this ancestral root seed in $O(1)$, verifying its Founding Breath `Noun`.
    pub fn reincarnate(
        &self,
        reference_bytes: Option<&[u8]>,
    ) -> Result<([u8; MAX_SEGMENT_BYTES], usize, TrieNode), &'static str> {
        let (bytes, len, trie) = self
            .formula
            .execute_verified(reference_bytes, Some(DEFAULT_DICT))?;
        if Noun::of(&bytes[..len]) != self.breath_noun {
            return Err("AncestralRootSeed reincarnation failed Founding Breath Noun verification");
        }
        Ok((bytes, len, trie))
    }
}

/// Executes a guarded Kōgnitara Bird Fusion between two Hamon Cube vertices `(parent_a, parent_b)`.
/// - Enforces hypercube adjacency (`hamming_distance == 1`), blocking *Forbidden Fusions*.
/// - Verifies both parents' `plans::Sceau` via the `Guardian` caste before synthesizing the child formula.
pub fn execute_bird_fusion(
    parent_a: u8,
    parent_b: u8,
    formula_a: &KpackFormula,
    formula_b: &KpackFormula,
    trie_a: &TrieNode,
    trie_b: &TrieNode,
    plan_a: &Plan,
    plan_b: &Plan,
    mutation_seed: u32,
) -> Result<CognitiveBirdFusion, CognitivePathology> {
    if !are_neighbors(parent_a, parent_b) {
        return Err(CognitivePathology::ForbiddenFusion {
            sender: parent_b,
            receiver: parent_a,
        });
    }

    let sceau_a = Sceau::birth(plan_a, 200);
    let sceau_b = Sceau::birth(plan_b, 200);
    if sceau_a.weigh(plan_a) != Verdict::Accept || sceau_b.weigh(plan_b) != Verdict::Accept {
        return Err(CognitivePathology::ForbiddenFusion {
            sender: parent_b,
            receiver: parent_a,
        });
    }

    let child_vertex = parent_a & 0x07;
    let child_formula = formula_a
        .crossover(formula_b, child_vertex, mutation_seed, Some(DEFAULT_DICT))
        .map_err(|_| CognitivePathology::PurposeDrift {
            node_id: child_vertex,
        })?;

    let (child_trie_node, child_merkle_noun) = merge_hypercube_trie_nodes(trie_a, trie_b, 1);
    let merged_plan = plan_a
        .merge_with(plan_b)
        .map_err(|_| CognitivePathology::ForbiddenFusion {
            sender: parent_b,
            receiver: parent_a,
        })?;

    let guardian_sceau = Sceau::birth(&merged_plan, 400);
    let guardian_verdict = guardian_sceau.weigh(&merged_plan);
    if guardian_verdict != Verdict::Accept {
        return Err(CognitivePathology::ForbiddenFusion {
            sender: parent_b,
            receiver: parent_a,
        });
    }

    Ok(CognitiveBirdFusion {
        parent_a,
        parent_b,
        child_vertex,
        child_caste: AiCaste::for_vertex(child_vertex),
        child_reality: TriRealityProfile::for_vertex(child_vertex),
        child_formula,
        child_trie_node,
        child_merkle_noun,
        merged_plan,
        guardian_verdict,
    })
}

/// Diagnoses the 8 vertices of a Kōgnitara Hamon Cube colony and automatically heals up to 2 amnesic
/// (missing or corrupted) vertices using either single XOR parity (`CubeParityStripe`) or dual Reed-Solomon
/// $\text{GF}(2^8)$ parity (`CubeDualParityStripe`).
pub fn diagnose_and_heal_colony(
    available_segments: &[Option<&[u8]>; NODE_COUNT],
    expected_nouns: &[Noun; NODE_COUNT],
    single_parity: &CubeParityStripe,
    dual_parity: &CubeDualParityStripe,
) -> Result<
    (
        CognitivePathology,
        [[u8; MAX_SEGMENT_BYTES]; NODE_COUNT],
        [usize; NODE_COUNT],
    ),
    CognitivePathology,
> {
    let mut restored = [[0u8; MAX_SEGMENT_BYTES]; NODE_COUNT];
    let mut lengths = [0usize; NODE_COUNT];
    let mut missing_mask = 0u8;
    let mut missing_indices = [0u8; NODE_COUNT];
    let mut missing_count = 0usize;

    for i in 0..NODE_COUNT {
        if let Some(seg) = available_segments[i] {
            let len = core::cmp::min(seg.len(), MAX_SEGMENT_BYTES);
            if Noun::of(&seg[..len]) == expected_nouns[i] {
                restored[i][..len].copy_from_slice(&seg[..len]);
                lengths[i] = len;
            } else {
                missing_mask |= 1u8 << i;
                missing_indices[missing_count] = i as u8;
                missing_count += 1;
            }
        } else {
            missing_mask |= 1u8 << i;
            missing_indices[missing_count] = i as u8;
            missing_count += 1;
        }
    }

    if missing_count == 0 {
        return Ok((CognitivePathology::Healthy, restored, lengths));
    }

    let valid_refs: [Option<&[u8]>; NODE_COUNT] = core::array::from_fn(|i| {
        if (missing_mask & (1u8 << i)) == 0 {
            Some(&restored[i][..lengths[i]])
        } else {
            None
        }
    });

    if missing_count == 1 {
        let m = missing_indices[0];
        let (rec, rec_len) = single_parity
            .recover_node(m, &valid_refs, &expected_nouns[m as usize])
            .map_err(|_| CognitivePathology::PurposeDrift { node_id: m })?;
        restored[m as usize] = rec;
        lengths[m as usize] = rec_len;
        return Ok((
            CognitivePathology::Amnesia {
                missing_mask,
                healed: true,
            },
            restored,
            lengths,
        ));
    }

    if missing_count == 2 {
        let m0 = missing_indices[0];
        let m1 = missing_indices[1];
        let ((rec0, len0), (rec1, len1)) = dual_parity
            .recover_two_nodes(
                m0,
                m1,
                &valid_refs,
                &expected_nouns[m0 as usize],
                &expected_nouns[m1 as usize],
            )
            .map_err(|_| CognitivePathology::PurposeDrift { node_id: m0 })?;
        restored[m0 as usize] = rec0;
        lengths[m0 as usize] = len0;
        restored[m1 as usize] = rec1;
        lengths[m1 as usize] = len1;
        return Ok((
            CognitivePathology::Amnesia {
                missing_mask,
                healed: true,
            },
            restored,
            lengths,
        ));
    }

    Err(CognitivePathology::Amnesia {
        missing_mask,
        healed: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hypercube::execute_distributed_cube_job;
    use crate::topology::CubeConfig;

    #[test]
    fn test_kognitara_castes_realities_fusion_and_dual_parity_healing() {
        assert_eq!(AiCaste::for_vertex(0), AiCaste::Queen);
        assert_eq!(AiCaste::for_vertex(1), AiCaste::Worker);
        assert_eq!(AiCaste::for_vertex(3), AiCaste::Scout);
        assert_eq!(AiCaste::for_vertex(6), AiCaste::Fulgaris);
        assert_eq!(AiCaste::for_vertex(7), AiCaste::Guardian);

        let p0 = TriRealityProfile::for_vertex(0);
        assert_eq!(p0.dominant, GoverningReality::Logika);
        let p2 = TriRealityProfile::for_vertex(2);
        assert_eq!(p2.dominant, GoverningReality::Oniria);
        let p4 = TriRealityProfile::for_vertex(4);
        assert_eq!(p4.dominant, GoverningReality::Umbra);

        // Execute a full Hamon Cube job and test ancestral root condensation + dual-node amnesia healing
        let cfg = CubeConfig::default();
        let mut payload = [0u8; 512];
        for (i, b) in payload.iter_mut().enumerate() {
            *b = ((i * 11 + 3) & 0xFF) as u8;
        }
        let report =
            execute_distributed_cube_job(&cfg, &payload, 1, 1_700_000_000).expect("Cube job");

        // Condense Node 0 into an AncestralRootSeed ("Ancestral Roots") and reincarnate in O(1)
        let seed0 = AncestralRootSeed::condense(0, report.formulas[0].clone(), &report.final_plan);
        let (reinc_bytes, reinc_len, _) = seed0.reincarnate(None).expect("Reincarnate node 0");
        assert_eq!(&reinc_bytes[..reinc_len], &payload[0..64]);

        // Forbidden fusion check: non-adjacent nodes 0 and 7 (Hamming distance 3 != 1) must be rejected!
        let forbidden = execute_bird_fusion(
            0,
            7,
            &report.formulas[0],
            &report.formulas[7],
            &report.final_trie_root,
            &report.final_trie_root,
            &report.final_plan,
            &report.final_plan,
            42,
        );
        assert_eq!(
            forbidden.unwrap_err(),
            CognitivePathology::ForbiddenFusion {
                sender: 7,
                receiver: 0
            }
        );

        // Dual-node Amnesia healing test (nodes 1 and 5 simultaneously lost)
        let slices: [(&[u8], usize); NODE_COUNT] =
            core::array::from_fn(|i| (&payload[i * 64..(i + 1) * 64], 64));
        let dual_parity = CubeDualParityStripe::compute(&slices);
        let mut available: [Option<&[u8]>; NODE_COUNT] =
            core::array::from_fn(|i| Some(&payload[i * 64..(i + 1) * 64]));
        available[1] = None;
        available[5] = None;

        let (pathology, healed_segs, healed_lens) = diagnose_and_heal_colony(
            &available,
            &report.segment_nouns,
            &report.parity,
            &dual_parity,
        )
        .expect("Colony should heal 2 missing nodes via GF(2^8) P+Q parity");

        assert_eq!(
            pathology,
            CognitivePathology::Amnesia {
                missing_mask: (1 << 1) | (1 << 5),
                healed: true
            }
        );
        assert_eq!(&healed_segs[1][..healed_lens[1]], &payload[64..128]);
        assert_eq!(&healed_segs[5][..healed_lens[5]], &payload[320..384]);
    }
}
