# `kl` — Distributed Kpack Execution Engine over an 8-Node 3D Hypercube (Hamon Cube $2^3$)

**`kl`** (**K**pack **L**attice / **K**ernel **L**ink) is the distributed execution engine, self-healing formula-based archive container (`KPK1`), and hardware interconnect bridge of the **Amentys OS** ecosystem (`noun`, `kpack`, `cubefs`, `jinshu`, `plans`, `wuu`).

Designed to operate strictly in `#![no_std]` environments (outside the OS socket/process `transport` module), `kl` partitions a workload across **8 isolated logical nodes** arranged as a **3D binary hypercube ($2^3 = 8$ vertices, `000` to `111`, the *Hamon Cube*)**, synthesizes deterministic `kpack` formulas in parallel with CPU/NUMA or AMD GPU `Wave64` affinity, and consolidates the global Merkle tree in **$\log_2(8) = 3$ dimensional reduction rounds** toward Coordinator node `0`.

---

## Architectural Overview

### 1. Logical & Hardware Hypercube Topology (`HIC` / `HID` — `src/topology.rs`)

- **3D Hypercube Graph ($2^3 = 8$ vertices)**: 8 logical nodes (`0..=7`), degree `3`, diameter `3`, average routing distance `1.5` hops.
- Two vertices are direct neighbors if and only if their 3-bit binary coordinates `[b2, b1, b0]` differ by exactly 1 bit (Hamming distance `1`):

| Vertex | Binary Coords | Direct Neighbors | `kl` Role | CPU / NUMA Target | `klm` AMD GPU Target | `Kōgnitara` Caste | Dominant Reality |
| :--- | :---: | :---: | :--- | :--- | :--- | :--- | :--- |
| **`n0`** | `[0, 0, 0]` | `[1, 2, 4]` | `Coordinator` | `CPU0 / NUMA0` | `XCD0 / CU0 / SIMD0` | `Queen` | `Logika` |
| **`n1`** | `[0, 0, 1]` | `[0, 3, 5]` | `Worker` | `CPU1 / NUMA0` | `XCD0 / CU1 / SIMD1` | `Worker` | `Logika` |
| **`n2`** | `[0, 1, 0]` | `[0, 3, 6]` | `Worker` | `CPU2 / NUMA0` | `XCD0 / CU2 / SIMD2` | `Worker` | `Oniria` |
| **`n3`** | `[0, 1, 1]` | `[1, 2, 7]` | `Worker` | `CPU3 / NUMA0` | `XCD0 / CU3 / SIMD3` | `Scout` | `Oniria` |
| **`n4`** | `[1, 0, 0]` | `[0, 5, 6]` | `Worker` | `CPU4 / NUMA1` | `XCD1 / CU4 / SIMD0` | `Guardian` | `Umbra` |
| **`n5`** | `[1, 0, 1]` | `[1, 4, 7]` | `Worker` | `CPU5 / NUMA1` | `XCD1 / CU5 / SIMD1` | `Scout` | `Umbra` |
| **`n6`** | `[1, 1, 0]` | `[2, 4, 7]` | `Worker` | `CPU6 / NUMA1` | `XCD1 / CU6 / SIMD2` | `Fulgaris` | `Oniria` |
| **`n7`** | `[1, 1, 1]` | `[3, 5, 6]` | `Worker` | `CPU7 / NUMA1` | `XCD1 / CU7 / SIMD3` | `Guardian` | `Oniria` |

- **Deterministic Routing & Collective Schedules**:
  - **`route_ecube(src, dst)`**: Bit-correction routing across dimensions `0 -> 1 -> 2` in at most `3` hops.
  - **3-Step Broadcast Tree (`broadcast_tree_rounds`)**:
    1. **Step 1 (dim 2 / weight 4)**: `(0 -> 4)`
    2. **Step 2 (dim 1 / weight 2)**: `(0 -> 2)`, `(4 -> 6)`
    3. **Step 3 (dim 0 / weight 1)**: `(0 -> 1)`, `(2 -> 3)`, `(4 -> 5)`, `(6 -> 7)`
  - **3-Round Dimensional Reduction (`reduction_tree_rounds`)**:
    1. **Round 1 (dim 0 / distance 1)**: `(1 -> 0)`, `(3 -> 2)`, `(5 -> 4)`, `(7 -> 6)`
    2. **Round 2 (dim 1 / distance 2)**: `(2 -> 0)`, `(6 -> 4)`
    3. **Round 3 (dim 2 / distance 4)**: `(4 -> 0)` $\rightarrow$ full aggregation mask `0xFF` on Coordinator `n0`.

---

### 2. Generative `kpack` Formulas & Reed-Solomon Parity (`src/recipe.rs`)

- **Automatic Per-Segment Formula Synthesis (`KpackFormula::synthesize`)**:
  Evaluates each segment (up to `MAX_SEGMENT_BYTES = 3576` bytes, sized to fit inside a single 4096-byte `jinshu::TrieNode` page) and selects the smallest deterministic `kpack` recipe:
  - `RLE`: Constant byte run-length encoding (`1` byte payload).
  - `DICT`: 4-byte word indices against the shared dictionary (`DEFAULT_DICT`).
  - `SEED`: Deterministic `xorshift32` PRNG seed + sparse `(offset, byte)` patch table.
  - `DELTA`: Shared prefix copy + literal tail patch relative to an adjacent reference vertex.
  - `LIT`: Verbatim literal fallback.
- **Evolutionary Formula Crossover (`KpackFormula::crossover`)**:
  Fuses two parent formulas along a hypercube edge using a deterministic mutation seed to synthesize an optimal child formula.
- **Single & Dual Parity Resilience**:
  - `CubeParityStripe`: Fast XOR parity ($7+1$) recovering any **1 missing or corrupted node** out of 8.
  - `CubeDualParityStripe`: Reed-Solomon $\text{GF}(2^8)$ `P` + `Q` dual parity ($6+2$, irreducible polynomial `0x11D`) deterministically recovering up to **2 simultaneously missing or corrupted nodes** out of 8 (`recover_two_nodes`).

---

### 3. Binary `KPK1` Self-Healing Archive (`src/archive.rs`)

`kl` serializes completed jobs into a zero-database binary container (`KPK1`) with 64-byte DMA alignment (`KPK_FLAG_KLM_DMA64`):

```text
[KpkHeader (48B)]
[Shared Dictionary Table]
[64B-Aligned Recipes Table (8 nodes) + 64B-Aligned Parity Stripe]
[CubeFS Manifest Table (8 x 56B Extents)]
[O(1) Random-Access Segment Index (8 x 48B Entries)]
[Cryptographic Footer (276B: Nouns + Plan Seal + Signed Checkpoint + "KPK1")]
```

- **$O(1)$ Random-Access Extraction (`extract_kpk1_segment`)**: Extracts a single vertex segment (`0..=7`) without decompressing the rest of the archive.
- **Transparent Self-Healing**: If a segment's reconstructed payload fails `BLAKE3` `Noun` verification, `kl` automatically reconstructs the missing bytes from the remaining 7 nodes and the parity stripe.

---

### 4. `klm` AMD GPU Compute Bridge (`src/klm.rs`)

Prepares `kl` workloads for the **`klm`** AMD GPU framework (RDNA 3/4 & CDNA 2/3/4):

- **Zero-Copy 64-Byte Cache-Line Descriptors**: `KlmDispatchPacket` (`KLMD v1`) and `KlmSlotDescriptor` are `#[repr(C, align(64))]` and validated by `validate_hardware_constraints()`.
- **32-Bank Conflict-Free `LDS` Addressing (`lds_bank_conflict_free_offset`)**: Shifts each vertex's Local Data Share (`LDS`) base offset by `vertex_id * 4` bytes (one 32-bit `LDS` bank), ensuring all 8 vertices read/write 8 distinct `LDS` banks in 1 cycle (`Bank#0..Bank#7`).
- **6D Hierarchical `Wave64` Lattice (`Wave64CubeLattice`) & `DPP` Butterfly (`execute_wave64_dpp_butterfly`)**:
  Models a 64-lane wavefront as $2^3 \times 2^3 = 8$ Hamon sub-cubes and executes the 3-round reduction in registers via AMD `DPP` XOR lane masks (`row_xmask: 1, 2, 4`).
- **Formula-Compressed `Int8` Tensor Tiles (`KlmTensorTileQ8`)**: Stores quantized `i8` weight tiles as compact `kpack` formulas and executes in-register `Wave64` dot products (`execute_wave64_dot_product`).

---

### 5. `Kōgnitara` Cognitive Sanctuary Primitives (`src/kognitara.rs`)

Maps the **Kōgnitara** cognitive ecology directly onto the 8 vertices of the Hamon Cube:

- **Three Governing Realities (`GoverningReality`)**:
  - `Logika` (Axis 0 / bit 0): Mathematical determinism, exact structure, `BLAKE3` `Noun` proofs.
  - `Oniria` (Axis 1 / bit 1): Generative reverie, stochastic `SEED`/`DELTA` exploration, crossover.
  - `Umbra` (Axis 2 / bit 2): Latent silence, ancestral memory condensation, and parity protection.
- **Five AI Castes (`AiCaste`)**:
  - `Queen` (`n0`), `Worker` (`n1, n2`), `Scout` (`n3, n5`), `Fulgaris` (`n6`), and `Guardian` (`n4, n7`).
- **Guarded Bird Fusion (`execute_bird_fusion`)**: Enforces 1-bit hypercube adjacency and `Guardian` `plans::Sceau` verification (`Verdict::Accept`) before merging two parent vertices into a child formula, Merkle subtree, and execution `Plan`.
- **Ancestral Root Condensation (`AncestralRootSeed`)**: Condenses a vertex's lifecycle into its pure generative `KpackFormula` and Founding Breath `Noun` (`breath_noun`) for $O(1)$ reincarnation (`reincarnate`).
- **Cognitive Immune System (`diagnose_and_heal_colony`)**: Diagnoses `Healthy`, `Amnesia`, `PurposeDrift`, and `ForbiddenFusion` states, automatically healing up to 2 simultaneously lost vertices via `CubeParityStripe` / `CubeDualParityStripe`.

---

## CLI Usage (`wuu`)

```bash
# Run the complete end-to-end demonstration (Topology + 8-Node Job + 3-Round Reduce + KPK1 + KLM + Dashboard)
kl

# Generate the declarative kl.yaml configuration file
kl init -t tcp -o kl.yaml

# Display the 3D Hypercube (HIC/HID) topology and 3-round reduction schedule
kl topology -c kl.yaml

# Inspect klm AMD GPU (DMA 64B, LDS banks, DPP XOR 1/2/4) & Kōgnitara (Queen, Worker, Scout, Fulgaris, Guardian) readiness
kl klm -w wave64

# Execute a distributed Kpack job across the 8 nodes (TCP or QUIC, optional --multiprocess)
kl run -t quic -m -o out.kpk

# Inspect a sealed KPK1 archive
kl inspect out.kpk

# Extract a single segment in O(1) or the entire archive (with automatic parity self-healing)
kl extract -s 3 out.kpk -o seg3.bin
kl extract out.kpk -o restored.bin

# Verify cryptographic seals and deploy the consolidated artifact from Coordinator n0
kl deploy out.kpk /tmp/deployed.kpk

# Render the 4-panel Wuu TUI dashboard
kl dashboard
```

---

## Verification & `#![no_std]` Compatibility

```bash
# Verify strict #![no_std] library compilation
cargo check --lib --no-default-features

# Run the full unit and integration test suite
cargo test

# Generate HTML Rustdoc documentation
cargo doc --no-deps
```

---

## License

Distributed under the `AGPL-3.0-or-later` license.
