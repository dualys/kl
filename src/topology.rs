//! # 3D Hypercube Topology (`HIC` / `HID`) & `Wave64` 6D Lattice
//!
//! This module models both the logical *Hamon Interconnect Cube* (`HIC`) and the hardware
//! *Hardware Interconnect Deployment* (`HID`) for `kl`:
//!
//! - **3D Hypercube ($2^3 = 8$ vertices)**: Each vertex `0..=7` is addressed by its 3-bit binary
//!   coordinates `[b2, b1, b0]` ([`coords_of`]) and connects to exactly 3 direct neighbors at
//!   Hamming distance `1` ([`neighbors_of`]). The graph has degree `3`, diameter `3`, and an
//!   average distance of `1.5` hops from Coordinator vertex `0`.
//! - **Deterministic Routing & Collective Trees**: Provides bit-correction `e-cube` routing
//!   ([`route_ecube`]), the 3-step spanning broadcast tree ([`broadcast_tree_rounds`]), and the
//!   $\log_2(8) = 3$ dimensional reduction schedule ([`reduction_tree_rounds`]).
//! - **Hardware Targets ([`ComputeTarget`])**: Maps each vertex onto either an isolated CPU core /
//!   NUMA socket (`CPU0..CPU7`, `NUMA 0` for `0..=3`, `NUMA 1` for `4..=7`) or an AMD GPU Compute
//!   Unit (`XCD`, `CU`, `SIMD`, `Wave32`/`Wave64`, `LDS` quota) for the `klm` framework.
//! - **AMD GPU `DPP` & `Wave64` 6D Lattice ([`dpp_xor_partner`], [`Wave64CubeLattice`])**: Maps
//!   the 3 reduction rounds onto single-cycle hardware `DPP` register lane swaps (`row_xmask: 1, 2, 4`)
//!   and decomposes a 64-lane `Wave64` wavefront into an $8 \times 8$ hierarchy of 8 Hamon Cubes.
//! - **Declarative YAML Configuration ([`CubeConfig::to_yaml`], [`CubeConfig::from_yaml`])**:
//!   Serializes, parses, and cryptographically seals (`KL3D` [`Noun`]) the `kl.yaml` cluster specification.

use core::fmt::Write;
use heapless::String;
use noun::Noun;

/// Dimensionality of the base Hamon Cube ($D = 3$).
pub const CUBE_DIM: usize = 3;
/// Number of logical vertices in the 3D Hamon Cube ($2^3 = 8$).
pub const NODE_COUNT: usize = 8;
/// Number of lanes in an AMD RDNA native `Wave32` wavefront (4 Hamon Cubes).
pub const WAVE32_LANES: usize = 32;
/// Number of lanes in an AMD CDNA/RDNA `Wave64` wavefront (8 Hamon Cubes = $2^3 \times 2^3$).
pub const WAVE64_LANES: usize = 64;
/// Number of 8-vertex Hamon Cubes packed inside a single 64-lane `Wave64` wavefront.
pub const SUBCUBES_PER_WAVE64: usize = WAVE64_LANES / NODE_COUNT;
/// Default loopback TCP/QUIC base port (`8000` for `n0` through `8007` for `n7`).
pub const DEFAULT_BASE_PORT: u16 = 8000;
/// Default per-vertex Local Data Share (`LDS`) quota in bytes (`8192` bytes).
pub const DEFAULT_LDS_QUOTA_BYTES: u32 = 8192;

/// Execution wavefront width for `klm` AMD GPU Compute Units / Workgroup Processors (WGP).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveMode {
    /// RDNA native 32-lane wavefront (4 Hamon Cubes per wavefront).
    Wave32,
    /// CDNA / RDNA wide 64-lane wavefront (8 Hamon Cubes = $2^3 \times 2^3$ 6D hypercube).
    Wave64,
}

impl WaveMode {
    #[must_use]
    pub const fn lanes(&self) -> usize {
        match self {
            Self::Wave32 => WAVE32_LANES,
            Self::Wave64 => WAVE64_LANES,
        }
    }

    #[must_use]
    pub const fn subcube_count(&self) -> usize {
        self.lanes() / NODE_COUNT
    }

    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Wave32 => "wave32",
            Self::Wave64 => "wave64",
        }
    }
}

/// Hardware Interconnect Deployment (`HID`) compute target for a logical Hamon Cube vertex.
/// Prepares `kl` for both CPU multi-core pinning and the `klm` AMD GPU framework.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeTarget {
    /// Dedicated CPU core and NUMA socket inside Amentys OS.
    CpuCore { core: u8, numa_node: u8 },
    /// Dedicated AMD GPU Compute Unit (CU / WGP) slice for the `klm` framework.
    AmdComputeUnit {
        xcd_id: u8,
        cu_index: u8,
        simd_unit: u8,
        wave_mode: WaveMode,
        lds_quota_bytes: u32,
    },
}

impl ComputeTarget {
    #[must_use]
    pub const fn default_cpu_for_node(id: u8) -> Self {
        let masked = id & 0x07;
        Self::CpuCore {
            core: masked,
            numa_node: if masked < 4 { 0 } else { 1 },
        }
    }

    #[must_use]
    pub const fn default_amd_cu_for_node(id: u8, wave_mode: WaveMode) -> Self {
        let masked = id & 0x07;
        Self::AmdComputeUnit {
            xcd_id: if masked < 4 { 0 } else { 1 },
            cu_index: masked,
            simd_unit: masked & 0x03,
            wave_mode,
            lds_quota_bytes: DEFAULT_LDS_QUOTA_BYTES,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeRole {
    Coordinator,
    Worker,
}

impl NodeRole {
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Coordinator => "coordinator",
            Self::Worker => "worker",
        }
    }

    #[must_use]
    pub fn from_str(s: &str) -> Option<Self> {
        let trimmed = s.trim();
        if trimmed == "coordinator" {
            Some(Self::Coordinator)
        } else if trimmed == "worker" {
            Some(Self::Worker)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportKind {
    Tcp,
    Quic,
}

impl TransportKind {
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Tcp => "tcp",
            Self::Quic => "quic",
        }
    }

    #[must_use]
    pub fn from_str(s: &str) -> Option<Self> {
        let trimmed = s.trim();
        if trimmed == "tcp" {
            Some(Self::Tcp)
        } else if trimmed == "quic" {
            Some(Self::Quic)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeConfig {
    pub id: u8,
    pub coords: [u8; CUBE_DIM],
    pub neighbors: [u8; CUBE_DIM],
    pub role: NodeRole,
    pub cpu_core: u8,
    pub numa_node: u8,
    pub ip: [u8; 4],
    pub port: u16,
}

impl NodeConfig {
    #[must_use]
    pub const fn default_for_id(id: u8, base_port: u16) -> Self {
        let masked = id & 0x07;
        Self {
            id: masked,
            coords: coords_of(masked),
            neighbors: neighbors_of(masked),
            role: if masked == 0 {
                NodeRole::Coordinator
            } else {
                NodeRole::Worker
            },
            cpu_core: masked,
            numa_node: if masked < 4 { 0 } else { 1 },
            ip: [127, 0, 0, 1],
            port: base_port.wrapping_add(masked as u16),
        }
    }

    #[must_use]
    pub fn endpoint_string(&self) -> String<32> {
        let mut s = String::new();
        let _ = write!(
            s,
            "{}.{}.{}.{}:{}",
            self.ip[0], self.ip[1], self.ip[2], self.ip[3], self.port
        );
        s
    }

    #[must_use]
    pub fn coords_string(&self) -> String<16> {
        let mut s = String::new();
        let _ = write!(
            s,
            "[{},{},{}]",
            self.coords[0], self.coords[1], self.coords[2]
        );
        s
    }

    #[must_use]
    pub fn bits_string(&self) -> String<8> {
        let mut s = String::new();
        let _ = write!(
            s,
            "{}{}{}",
            self.coords[0], self.coords[1], self.coords[2]
        );
        s
    }

    #[must_use]
    pub const fn cpu_target(&self) -> ComputeTarget {
        ComputeTarget::CpuCore {
            core: self.cpu_core,
            numa_node: self.numa_node,
        }
    }

    #[must_use]
    pub const fn amd_gpu_target(&self, wave_mode: WaveMode) -> ComputeTarget {
        ComputeTarget::AmdComputeUnit {
            xcd_id: self.numa_node,
            cu_index: self.cpu_core,
            simd_unit: self.id & 0x03,
            wave_mode,
            lds_quota_bytes: DEFAULT_LDS_QUOTA_BYTES,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CubeConfig {
    pub transport: TransportKind,
    pub base_port: u16,
    pub nodes: [NodeConfig; NODE_COUNT],
}

impl Default for CubeConfig {
    fn default() -> Self {
        Self::new(DEFAULT_BASE_PORT, TransportKind::Tcp)
    }
}

impl CubeConfig {
    #[must_use]
    pub const fn new(base_port: u16, transport: TransportKind) -> Self {
        Self {
            transport,
            base_port,
            nodes: [
                NodeConfig::default_for_id(0, base_port),
                NodeConfig::default_for_id(1, base_port),
                NodeConfig::default_for_id(2, base_port),
                NodeConfig::default_for_id(3, base_port),
                NodeConfig::default_for_id(4, base_port),
                NodeConfig::default_for_id(5, base_port),
                NodeConfig::default_for_id(6, base_port),
                NodeConfig::default_for_id(7, base_port),
            ],
        }
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        let mut coordinator_count = 0usize;
        for i in 0..NODE_COUNT {
            let node = &self.nodes[i];
            if node.id as usize != i {
                return Err("Node ID mismatch in hypercube table");
            }
            if node.coords != coords_of(node.id) {
                return Err("Node binary coordinates do not match hypercube vertex ID");
            }
            let expected_neighbors = neighbors_of(node.id);
            for &nb in &expected_neighbors {
                if !node.neighbors.contains(&nb) {
                    return Err("Node neighbor list violates 1-bit Hamming distance");
                }
                if hamming_distance(node.id, nb) != 1 {
                    return Err("Invalid edge in hypercube: Hamming distance != 1");
                }
            }
            if node.role == NodeRole::Coordinator {
                coordinator_count += 1;
            }
            for j in (i + 1)..NODE_COUNT {
                if self.nodes[j].ip == node.ip && self.nodes[j].port == node.port {
                    return Err("Duplicate endpoint address/port across logical nodes");
                }
            }
        }
        if coordinator_count != 1 || self.nodes[0].role != NodeRole::Coordinator {
            return Err("Node 0 must be the unique coordinator of the Hamon Cube");
        }
        Ok(())
    }

    #[must_use]
    pub fn config_noun(&self) -> Noun {
        let mut raw = [0u8; 160];
        raw[0..4].copy_from_slice(b"KL3D");
        raw[4] = match self.transport {
            TransportKind::Tcp => 0,
            TransportKind::Quic => 1,
        };
        raw[5..7].copy_from_slice(&self.base_port.to_le_bytes());
        let mut offset = 8;
        for node in &self.nodes {
            raw[offset] = node.id;
            raw[offset + 1..offset + 4].copy_from_slice(&node.coords);
            raw[offset + 4..offset + 7].copy_from_slice(&node.neighbors);
            raw[offset + 7] = match node.role {
                NodeRole::Coordinator => 1,
                NodeRole::Worker => 0,
            };
            raw[offset + 8] = node.cpu_core;
            raw[offset + 9] = node.numa_node;
            raw[offset + 10..offset + 14].copy_from_slice(&node.ip);
            raw[offset + 14..offset + 16].copy_from_slice(&node.port.to_le_bytes());
            offset += 16;
        }
        Noun::of(&raw[..offset])
    }

    #[must_use]
    pub fn to_yaml(&self) -> String<2048> {
        let mut out = String::new();
        let _ = writeln!(out, "topology: \"hypercube_3d\"");
        let _ = writeln!(out, "transport: \"{}\"", self.transport.as_str());
        let _ = writeln!(out, "nodes:");
        for node in &self.nodes {
            let _ = writeln!(out, "  - id: {}", node.id);
            let _ = writeln!(
                out,
                "    coords: [{},{},{}]",
                node.coords[0], node.coords[1], node.coords[2]
            );
            let _ = writeln!(
                out,
                "    neighbors: [{},{},{}]",
                node.neighbors[0], node.neighbors[1], node.neighbors[2]
            );
            let _ = writeln!(out, "    role: {}", node.role.as_str());
            let _ = writeln!(out, "    cpu_core: {}", node.cpu_core);
            let _ = writeln!(out, "    numa_node: {}", node.numa_node);
            let _ = writeln!(
                out,
                "    endpoint: \"{}.{}.{}.{}:{}\"",
                node.ip[0], node.ip[1], node.ip[2], node.ip[3], node.port
            );
        }
        out
    }

    pub fn from_yaml(yaml: &str) -> Result<Self, &'static str> {
        let mut config = Self::default();
        let mut current_id: Option<usize> = None;
        let mut seen_nodes = 0u8;

        for raw_line in yaml.lines() {
            let line = match raw_line.find('#') {
                Some(idx) => &raw_line[..idx],
                None => raw_line,
            }
            .trim();
            if line.is_empty() || line == "nodes:" {
                continue;
            }
            if let Some(rest) = line.strip_prefix("topology:") {
                let val = rest.trim().trim_matches('"');
                if val != "hypercube_3d" {
                    return Err("Unsupported topology: expected hypercube_3d");
                }
                continue;
            }
            if let Some(rest) = line.strip_prefix("transport:") {
                let val = rest.trim().trim_matches('"');
                config.transport =
                    TransportKind::from_str(val).ok_or("Invalid transport in YAML")?;
                continue;
            }
            let normalized = line.strip_prefix('-').map_or(line, str::trim);
            if let Some(rest) = normalized.strip_prefix("id:") {
                let id = parse_u8(rest.trim()).ok_or("Invalid node id in YAML")?;
                if id as usize >= NODE_COUNT {
                    return Err("Node id out of range 0..7");
                }
                current_id = Some(id as usize);
                config.nodes[id as usize].id = id;
                seen_nodes |= 1 << id;
                continue;
            }
            let Some(idx) = current_id else {
                continue;
            };
            if let Some(rest) = normalized.strip_prefix("coords:") {
                config.nodes[idx].coords =
                    parse_triple_u8(rest.trim()).ok_or("Invalid coords in YAML")?;
            } else if let Some(rest) = normalized.strip_prefix("neighbors:") {
                config.nodes[idx].neighbors =
                    parse_triple_u8(rest.trim()).ok_or("Invalid neighbors in YAML")?;
            } else if let Some(rest) = normalized.strip_prefix("role:") {
                config.nodes[idx].role =
                    NodeRole::from_str(rest.trim()).ok_or("Invalid role in YAML")?;
            } else if let Some(rest) = normalized.strip_prefix("cpu_core:") {
                config.nodes[idx].cpu_core =
                    parse_u8(rest.trim()).ok_or("Invalid cpu_core in YAML")?;
            } else if let Some(rest) = normalized.strip_prefix("numa_node:") {
                config.nodes[idx].numa_node =
                    parse_u8(rest.trim()).ok_or("Invalid numa_node in YAML")?;
            } else if let Some(rest) = normalized.strip_prefix("endpoint:") {
                let (ip, port) =
                    parse_endpoint(rest.trim()).ok_or("Invalid endpoint in YAML")?;
                config.nodes[idx].ip = ip;
                config.nodes[idx].port = port;
                if idx == 0 {
                    config.base_port = port;
                }
            }
        }

        if seen_nodes != 0xFF {
            return Err("YAML configuration must define all 8 logical nodes (0..7)");
        }
        config.validate()?;
        Ok(config)
    }
}

/// Returns the 3D binary coordinates `[b2, b1, b0]` for a vertex `id` in `0..8`.
#[must_use]
pub const fn coords_of(id: u8) -> [u8; CUBE_DIM] {
    [(id >> 2) & 1, (id >> 1) & 1, id & 1]
}

/// Converts 3D binary coordinates `[b2, b1, b0]` back to vertex `id` in `0..8`.
#[must_use]
pub const fn id_from_coords(coords: [u8; CUBE_DIM]) -> u8 {
    ((coords[0] & 1) << 2) | ((coords[1] & 1) << 1) | (coords[2] & 1)
}

/// Returns the neighbor of `id` along dimension `dim` (`0` = weight 1, `1` = weight 2, `2` = weight 4).
#[must_use]
pub const fn neighbor_along_dimension(id: u8, dim: usize) -> u8 {
    (id & 0x07) ^ (1u8 << (dim & 0x03))
}

/// Returns the 3 direct neighbors of `id` in sorted ascending order.
#[must_use]
pub const fn neighbors_of(id: u8) -> [u8; CUBE_DIM] {
    let mut n = [
        neighbor_along_dimension(id, 0),
        neighbor_along_dimension(id, 1),
        neighbor_along_dimension(id, 2),
    ];
    if n[0] > n[1] {
        let tmp = n[0];
        n[0] = n[1];
        n[1] = tmp;
    }
    if n[1] > n[2] {
        let tmp = n[1];
        n[1] = n[2];
        n[2] = tmp;
    }
    if n[0] > n[1] {
        let tmp = n[0];
        n[0] = n[1];
        n[1] = tmp;
    }
    n
}

/// Returns the Hamming distance (number of differing bits) between two node IDs.
#[must_use]
pub const fn hamming_distance(a: u8, b: u8) -> u8 {
    ((a & 0x07) ^ (b & 0x07)).count_ones() as u8
}

/// Returns true if two nodes are directly connected by an edge in the 3D hypercube.
#[must_use]
pub const fn are_neighbors(a: u8, b: u8) -> bool {
    hamming_distance(a, b) == 1
}

/// Deterministic e-cube routing from `src` to `dst` in at most 3 hops.
/// Returns `(path_nodes, hop_count)` where `path_nodes[0] == src` and `path_nodes[hop_count] == dst`.
#[must_use]
pub const fn route_ecube(src: u8, dst: u8) -> ([u8; 4], usize) {
    let mut path = [src & 0x07, 0, 0, 0];
    let mut current = src & 0x07;
    let target = dst & 0x07;
    let mut hops = 0usize;

    let mut dim = 0usize;
    while dim < CUBE_DIM {
        let mask = 1u8 << dim;
        if (current ^ target) & mask != 0 {
            current ^= mask;
            hops += 1;
            path[hops] = current;
        }
        dim += 1;
    }
    (path, hops)
}

/// Returns the 3 rounds of the dimensional hypercube broadcast starting at node 0:
/// - Step 0 (dim 2 / weight 4): `(0 -> 4)`
/// - Step 1 (dim 1 / weight 2): `(0 -> 2), (4 -> 6)`
/// - Step 2 (dim 0 / weight 1): `(0 -> 1), (2 -> 3), (4 -> 5), (6 -> 7)`
#[must_use]
pub const fn broadcast_tree_rounds() -> ([(u8, u8); 1], [(u8, u8); 2], [(u8, u8); 4]) {
    ([(0, 4)], [(0, 2), (4, 6)], [(0, 1), (2, 3), (4, 5), (6, 7)])
}

/// Returns the 3 rounds of dimensional hypercube reduction toward node 0 (Coordinator),
/// exactly matching the Hamon Cube specification (page 7):
/// - Round 1 (distance 1 / weight 1): `(1 -> 0), (3 -> 2), (5 -> 4), (7 -> 6)`
/// - Round 2 (distance 2 / weight 2): `(2 -> 0), (6 -> 4)`
/// - Round 3 (distance 4 / weight 4): `(4 -> 0)`
#[must_use]
pub const fn reduction_tree_rounds() -> ([(u8, u8); 4], [(u8, u8); 2], [(u8, u8); 1]) {
    ([(1, 0), (3, 2), (5, 4), (7, 6)], [(2, 0), (6, 4)], [(4, 0)])
}

/// Computes the AMD GPU hardware `DPP` (`Data Parallel Primitives`) / `ds_swizzle_b32` partner lane
/// for a given `lane_id` during hypercube reduction `round` (`1..=3` -> `XOR 1`, `XOR 2`, `XOR 4`).
#[must_use]
pub const fn dpp_xor_partner(lane_id: u8, round: u8) -> u8 {
    let shift = match round {
        1 => 0,
        2 => 1,
        3 => 2,
        _ => (round.saturating_sub(1)) & 0x07,
    };
    lane_id ^ (1u8 << shift)
}

/// Hierarchical 6D Hypercube Lattice ($2^3 \times 2^3 = 64$ lanes) modeling a full AMD `Wave64` wavefront
/// for the `klm` framework:
/// - Bits `0..=2` (`lane_id & 0x07`): inner 3D Hamon Cube vertex (register-level DPP `row_xmask: 1, 2, 4`).
/// - Bits `3..=5` (`(lane_id >> 3) & 0x07`): outer 3D Cube of 8 sub-cubes (shared `LDS` butterfly exchange).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Wave64CubeLattice;

impl Wave64CubeLattice {
    /// Returns the outer sub-cube index (`0..8`) for a `Wave64` lane (`0..64`).
    #[must_use]
    pub const fn subcube_of(lane_id: u8) -> u8 {
        (lane_id >> 3) & 0x07
    }

    /// Returns the inner Hamon Cube vertex (`0..8`) for a `Wave64` lane (`0..64`).
    #[must_use]
    pub const fn vertex_of(lane_id: u8) -> u8 {
        lane_id & 0x07
    }

    /// Reconstructs a `Wave64` lane index (`0..64`) from `(subcube_id, vertex_id)`.
    #[must_use]
    pub const fn lane_from_parts(subcube_id: u8, vertex_id: u8) -> u8 {
        ((subcube_id & 0x07) << 3) | (vertex_id & 0x07)
    }

    /// Returns the 6D binary coordinates `[s2, s1, s0, v2, v1, v0]` for a `Wave64` lane (`0..64`).
    #[must_use]
    pub const fn coords_6d(lane_id: u8) -> [u8; 6] {
        let masked = lane_id & 0x3F;
        [
            (masked >> 5) & 1,
            (masked >> 4) & 1,
            (masked >> 3) & 1,
            (masked >> 2) & 1,
            (masked >> 1) & 1,
            masked & 1,
        ]
    }

    /// Returns the intra-subcube partner lane along inner dimension `dim` (`0..3`, DPP register swap).
    #[must_use]
    pub const fn intra_cube_partner(lane_id: u8, dim: usize) -> u8 {
        (lane_id & 0x3F) ^ (1u8 << (dim & 0x03))
    }

    /// Returns the inter-subcube partner lane along outer dimension `dim` (`0..3`, LDS butterfly swap).
    #[must_use]
    pub const fn inter_cube_partner(lane_id: u8, dim: usize) -> u8 {
        (lane_id & 0x3F) ^ (1u8 << (3 + (dim & 0x03)))
    }
}

fn parse_u8(s: &str) -> Option<u8> {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return None;
    }
    let mut val = 0u16;
    for b in trimmed.bytes() {
        if !b.is_ascii_digit() {
            return None;
        }
        val = val * 10 + (b - b'0') as u16;
        if val > 255 {
            return None;
        }
    }
    Some(val as u8)
}

fn parse_u16(s: &str) -> Option<u16> {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return None;
    }
    let mut val = 0u32;
    for b in trimmed.bytes() {
        if !b.is_ascii_digit() {
            return None;
        }
        val = val * 10 + (b - b'0') as u32;
        if val > 65535 {
            return None;
        }
    }
    Some(val as u16)
}

fn parse_triple_u8(s: &str) -> Option<[u8; 3]> {
    let inner = s.trim().strip_prefix('[')?.strip_suffix(']')?;
    let mut out = [0u8; 3];
    let mut count = 0usize;
    for part in inner.split(',') {
        if count >= 3 {
            return None;
        }
        out[count] = parse_u8(part)?;
        count += 1;
    }
    if count == 3 {
        Some(out)
    } else {
        None
    }
}

fn parse_endpoint(s: &str) -> Option<([u8; 4], u16)> {
    let unquoted = s.trim().trim_matches('"');
    let colon = unquoted.rfind(':')?;
    let ip_str = &unquoted[..colon];
    let port_str = &unquoted[colon + 1..];
    let port = parse_u16(port_str)?;

    let mut ip = [0u8; 4];
    let mut count = 0usize;
    for part in ip_str.split('.') {
        if count >= 4 {
            return None;
        }
        ip[count] = parse_u8(part)?;
        count += 1;
    }
    if count == 4 {
        Some((ip, port))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hypercube_neighbors_and_coords_match_spec() {
        let cfg = CubeConfig::default();
        assert!(cfg.validate().is_ok());

        // Spec page 2 & page 8-9 examples:
        assert_eq!(cfg.nodes[0].coords, [0, 0, 0]);
        assert_eq!(cfg.nodes[0].neighbors, [1, 2, 4]);
        assert_eq!(cfg.nodes[0].role, NodeRole::Coordinator);
        assert_eq!(cfg.nodes[0].numa_node, 0);

        assert_eq!(cfg.nodes[3].coords, [0, 1, 1]);
        assert_eq!(cfg.nodes[3].neighbors, [1, 2, 7]);
        assert_eq!(cfg.nodes[3].numa_node, 0);

        assert_eq!(cfg.nodes[4].coords, [1, 0, 0]);
        assert_eq!(cfg.nodes[4].neighbors, [0, 5, 6]);
        assert_eq!(cfg.nodes[4].numa_node, 1);

        assert_eq!(cfg.nodes[7].coords, [1, 1, 1]);
        assert_eq!(cfg.nodes[7].neighbors, [3, 5, 6]);
        assert_eq!(cfg.nodes[7].numa_node, 1);
    }

    #[test]
    fn test_diameter_and_average_distance() {
        let mut max_dist = 0u8;
        let mut total_dist_from_zero = 0u32;
        for dst in 0..8u8 {
            let d = hamming_distance(0, dst);
            if d > max_dist {
                max_dist = d;
            }
            total_dist_from_zero += d as u32;
            let (path, hops) = route_ecube(0, dst);
            assert_eq!(hops as u8, d);
            assert_eq!(path[0], 0);
            assert_eq!(path[hops], dst);
            for h in 0..hops {
                assert!(are_neighbors(path[h], path[h + 1]));
            }
        }
        // Diameter is 3, sum of distances from node 0 to all 8 nodes is 0 + 3*1 + 3*2 + 1*3 = 12 -> 12/8 = 1.5
        assert_eq!(max_dist, 3);
        assert_eq!(total_dist_from_zero, 12);
    }

    #[test]
    fn test_yaml_roundtrip_and_spec_parse() {
        let cfg = CubeConfig::default();
        let yaml = cfg.to_yaml();
        let parsed = CubeConfig::from_yaml(yaml.as_str()).expect("YAML should parse cleanly");
        assert_eq!(cfg, parsed);
        assert_eq!(cfg.config_noun(), parsed.config_noun());
    }

    #[test]
    fn test_wave64_cube_lattice_and_dpp_xor_routing() {
        // Verify DPP XOR partner matches reduction_tree_rounds
        let (r1, r2, r3) = reduction_tree_rounds();
        for &(s, r) in &r1 {
            assert_eq!(dpp_xor_partner(r, 1), s);
        }
        for &(s, r) in &r2 {
            assert_eq!(dpp_xor_partner(r, 2), s);
        }
        for &(s, r) in &r3 {
            assert_eq!(dpp_xor_partner(r, 3), s);
        }

        // Verify 64-lane 6D hypercube decomposition (8 subcubes x 8 vertices)
        for lane in 0..64u8 {
            let sub = Wave64CubeLattice::subcube_of(lane);
            let vtx = Wave64CubeLattice::vertex_of(lane);
            assert!(sub < 8);
            assert!(vtx < 8);
            assert_eq!(Wave64CubeLattice::lane_from_parts(sub, vtx), lane);

            for dim in 0..3 {
                let intra = Wave64CubeLattice::intra_cube_partner(lane, dim);
                assert_eq!(Wave64CubeLattice::subcube_of(intra), sub);
                assert_eq!(
                    hamming_distance(Wave64CubeLattice::vertex_of(intra), vtx),
                    1
                );

                let inter = Wave64CubeLattice::inter_cube_partner(lane, dim);
                assert_eq!(Wave64CubeLattice::vertex_of(inter), vtx);
                assert_eq!(
                    hamming_distance(Wave64CubeLattice::subcube_of(inter), sub),
                    1
                );
            }
        }
    }
}
