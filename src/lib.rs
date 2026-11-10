//! # `kl` — Kpack Lattice & Distributed 3D Hypercube Execution Engine
//!
//! **`kl`** (**K**pack **L**attice / **K**ernel **L**ink) is the distributed execution engine,
//! self-healing formula-based archive layer, and hardware interconnect bridge of the **Amentys OS**
//! ecosystem (`noun`, `kpack`, `jinshu`, `cubefs`, `plans`, `wuu`).
//!
//! Designed to compile strictly in `#![no_std]` environments (outside the OS socket/process
//! `transport` feature), `kl` partitions any workload across **8 isolated logical nodes** organized
//! as a **3D binary hypercube ($2^3 = 8$ vertices, `000` to `111`, the *Hamon Cube*)**, synthesizes
//! deterministic generative formulas per vertex, and consolidates the resulting Merkle state in
//! exactly **$\log_2(8) = 3$ dimensional reduction rounds** toward Coordinator vertex `0`.
//!
//! ## Core Architectural Pillars
//!
//! 1. **Logical & Hardware Hypercube Topology ([`topology`])** — Models the 8-vertex *Hamon
//!    Interconnect Cube* (`HIC`) and *Hardware Interconnect Deployment* (`HID`), including
//!    1-bit Hamming neighbor adjacency, deterministic `e-cube` routing, CPU core / NUMA socket
//!    affinity, AMD GPU Compute Unit (`CU` / `WGP`) targets, `DPP` register lane XOR routing, and
//!    the 64-lane 6D hierarchical [`topology::Wave64CubeLattice`] ($2^3 \times 2^3$).
//! 2. **Generative Formula Synthesis & Reed-Solomon Parity ([`recipe`])** — Automatically encodes
//!    each vertex segment into the smallest deterministic `kpack` program (`LIT`, `RLE`, `DICT`,
//!    `SEED`, `DELTA`, `XOR`, `RS`), supports evolutionary formula crossover
//!    ([`recipe::KpackFormula::crossover`]), and protects the 8 vertices with both single XOR parity
//!    ([`recipe::CubeParityStripe`]) and dual Reed-Solomon $\text{GF}(2^8)$ $P+Q$ parity
//!    ([`recipe::CubeDualParityStripe`]) capable of healing 2 simultaneous node failures.
//! 3. **SPMD Orchestration & 3-Round Merkle Reduction ([`hypercube`])** — Broadcasts jobs in 3 hops,
//!    executes formulas in isolated node [`plans::Plan`] sandboxes guarded by [`plans::sceau::Sceau`],
//!    merges [`jinshu::node::TrieNode`] subtrees across rounds `1 -> 2 -> 3`, builds the
//!    [`cubefs::BTreeNode`] segment directory, and signs the final [`hypercube::Checkpoint`].
//! 4. **Zero-Database Self-Healing `KPK1` Archive ([`archive`])** — Packs all 8 vertex formulas,
//!    parity stripes, `cubefs` extents, $O(1)$ random-access index entries, and Merkle checkpoints
//!    into a 64-byte DMA-aligned binary container with transparent parity auto-repair.
//! 5. **`klm` AMD GPU Compute Bridge ([`klm`])** — Prepares 64-byte cache-line aligned zero-copy
//!    dispatch descriptors ([`klm::KlmDispatchPacket`]), 32-bank conflict-free `LDS` shared memory
//!    addresses, 3-cycle `Wave64` `DPP` butterfly reductions, and formula-compressed `Int8` tensor
//!    tiles ([`klm::KlmTensorTileQ8`]).
//! 6. **Kōgnitara Cognitive Sanctuary Primitives ([`kognitara`])** — Maps the 3 Governing Realities
//!    (`Logika`, `Oniria`, `Umbra`) onto the 3 binary axes of the Hamon Cube, assigns the 5 AI
//!    Castes (`Queen`, `Worker`, `Scout`, `Fulgaris`, `Guardian`) to the 8 vertices, and implements
//!    guarded Bird Fusion, Ancestral Root Condensation, and the Cognitive Immune System.
//! 7. **Authenticated Loopback & Multi-Process Transport ([`transport`])** — Pins workers to
//!    hardware CPU cores via `sched_setaffinity` and exchanges BLAKE3-authenticated `KLWP` wire
//!    packets over TCP or QUIC/UDP loopback sockets (`127.0.0.1:8000..8007`).
//! 8. **CLI & 4-Panel `wuu` TUI Dashboard ([`cli`])** — Exposes `init`, `topology`, `run`,
//!    `inspect`, `extract`, `deploy`, `dashboard`, `klm`, and `worker` commands.

#![cfg_attr(not(any(test, feature = "std")), no_std)]

pub mod archive;
pub mod cli;
pub mod hypercube;
pub mod klm;
pub mod kognitara;
pub mod recipe;
pub mod topology;

#[cfg(any(test, feature = "std"))]
pub mod transport;
