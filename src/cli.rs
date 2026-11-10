//! # CLI Specification & 4-Panel `wuu` TUI Dashboard
//!
//! This module defines the command-line interface and terminal user interface (TUI) renderers
//! for `kl`, built atop the `#![no_std]`-compatible [`wuu`] framework:
//!
//! - [`kl_cli`]: Declarative command hierarchy (`init`, `topology`, `run`, `inspect`, `extract`,
//!   `deploy`, `dashboard`, `klm`, and `worker`).
//! - [`render_topology`]: Renders the 8-vertex `HIC`/`HID` hypercube table (`VERTEX`, `COORDS`,
//!   `NEIGHBORS`, `ROLE`, `AFFINITY`, `ENDPOINT`) and the $\log_2(8) = 3$ reduction schedule.
//! - [`render_execution_report`]: Renders the synthesized `kpack` formulas for nodes `0..=7` and
//!   the per-round Merkle reduction traces (`r1`, `r2`, `r3-root`).
//! - [`render_archive_inspection`]: Renders the segment index, opcodes, byte offsets, and `BLAKE3`
//!   `Noun` digests of a binary `KPK1` archive.
//! - [`render_dashboard`]: Renders the 4-panel `wuu` TUI dashboard (*HIC/HID 3D Cube*,
//!   *Kpack 8-Node Formulas*, *Hypercube 3-Round Reduce*, *Plans, Seal & Checkpoint*).
//! - [`render_klm_readiness`]: Renders the hardware dispatch table (`VERTEX`, `CASTE`, `REALITY`,
//!   `AMD GPU TARGET`, `LDS BANK & OFFSET`, `DMA64 RECIPE`) and `Wave64` `DPP` butterfly trace.
//! - [`execute_cli_no_std`]: Pure `#![no_std]` in-memory CLI dispatcher for kernel and embedded
//!   environments.

use core::fmt::Write;
use heapless::Vec;
use noun::Noun;
use wuu::{
    Arg, ColorMode, Command, Context, DropDownTerm, MessageLevel, Panel, ProgressBar, StackString,
    Status, Table, TableStyle, Wuu, write_ok_custom, write_styled_message,
};

use crate::archive::{KpkArchiveView, inspect_kpk1_archive};
use crate::hypercube::{CubeExecutionReport, execute_distributed_cube_job};
use crate::klm::{KlmDispatchPacket, execute_wave64_dpp_butterfly};
use crate::kognitara::{AiCaste, TriRealityProfile};
use crate::topology::{
    ComputeTarget, CubeConfig, NODE_COUNT, TransportKind, WAVE64_LANES, WaveMode,
};

const INIT_ARGS: [Arg<'static>; 2] = [
    Arg::new("transport")
        .short('t')
        .long("transport")
        .takes_value("PROTO")
        .help("Internal interconnect transport protocol: tcp or quic (default: tcp)"),
    Arg::new("output")
        .short('o')
        .long("output")
        .takes_value("FILE")
        .help("Declarative YAML configuration file path to write (default: kl.yaml)"),
];

const TOPOLOGY_ARGS: [Arg<'static>; 1] = [Arg::new("config")
    .short('c')
    .long("config")
    .takes_value("FILE")
    .help("Path to the kl.yaml configuration file to inspect")];

const RUN_ARGS: [Arg<'static>; 5] = [
    Arg::new("transport")
        .short('t')
        .long("transport")
        .takes_value("PROTO")
        .help("Loopback transport protocol across the 8 nodes: tcp or quic (default: tcp)"),
    Arg::new("multiprocess")
        .short('m')
        .long("multiprocess")
        .help("Spawn 8 isolated OS processes (1 Coordinator + 7 Workers)"),
    Arg::new("data")
        .short('d')
        .long("data")
        .takes_value("PAYLOAD")
        .help("Inline payload string to partition across the 8 hypercube nodes"),
    Arg::new("input")
        .short('i')
        .long("input")
        .takes_value("PATH")
        .help("Input file path to synthesize and execute across the hypercube"),
    Arg::new("output")
        .short('o')
        .long("output")
        .takes_value("ARCHIVE")
        .help("Output path for the sealed KPK1 archive (default: out.kpk)"),
];

const INSPECT_ARGS: [Arg<'static>; 1] = [Arg::new("archive")
    .takes_value("ARCHIVE")
    .required()
    .help("Path to the KPK1 archive to inspect")];

const EXTRACT_ARGS: [Arg<'static>; 3] = [
    Arg::new("segment")
        .short('s')
        .long("segment")
        .takes_value("NODE_ID")
        .help("Extract only a single logical node segment (0..7) in O(1)"),
    Arg::new("output")
        .short('o')
        .long("output")
        .takes_value("PATH")
        .help("Output file path for the reconstructed bytes"),
    Arg::new("archive")
        .takes_value("ARCHIVE")
        .required()
        .help("Path to the KPK1 archive to extract"),
];

const DEPLOY_ARGS: [Arg<'static>; 2] = [
    Arg::new("archive")
        .takes_value("ARCHIVE")
        .required()
        .help("Consolidated KPK1 archive sealed by Coordinator node 0"),
    Arg::new("target")
        .takes_value("TARGET")
        .required()
        .help("Deployment destination (target file path or endpoint)"),
];

const DASHBOARD_ARGS: [Arg<'static>; 2] = [
    Arg::new("cols")
        .long("cols")
        .takes_value("COLS")
        .help("Dashboard width in columns (default: 80)"),
    Arg::new("rows")
        .long("rows")
        .takes_value("ROWS")
        .help("Dashboard height in rows (default: 24)"),
];

const WORKER_ARGS: [Arg<'static>; 2] = [
    Arg::new("id")
        .long("id")
        .takes_value("NODE_ID")
        .required()
        .help("Logical node identifier (0..7)"),
    Arg::new("cpu")
        .long("cpu")
        .takes_value("CORE")
        .help("CPU core index to pin this worker process to"),
];

const KLM_ARGS: [Arg<'static>; 1] = [Arg::new("wave")
    .short('w')
    .long("wave")
    .takes_value("MODE")
    .help("AMD GPU wavefront width for klm: wave32 or wave64 (default: wave64)")];

const SUBCOMMANDS: [Command<'static>; 9] = [
    Command::new("init")
        .about("Generate declarative kl.yaml configuration (8-node HIC/HID topology)")
        .args(&INIT_ARGS),
    Command::new("topology")
        .about("Display the 3D Hypercube (2^3) topology, CPU/NUMA affinity, and 3 reduction rounds")
        .args(&TOPOLOGY_ARGS),
    Command::new("run")
        .about("Execute a distributed Kpack job across the 8 logical nodes and reduce in 3 hops")
        .args(&RUN_ARGS),
    Command::new("inspect")
        .about("Inspect a KPK1 archive (Header, 8 Kpack formulas, RS parity, Merkle root & Seal)")
        .args(&INSPECT_ARGS),
    Command::new("extract")
        .about("Extract a single O(1) segment or the full archive with parity self-healing")
        .args(&EXTRACT_ARGS),
    Command::new("deploy")
        .about("Verify cryptographic seals and deploy the consolidated artifact from Coordinator 0")
        .args(&DEPLOY_ARGS),
    Command::new("dashboard")
        .about("Render the 4-panel Wuu TUI dashboard (3D Cube, Formulas, 3-Round Reduce, Seal)")
        .args(&DASHBOARD_ARGS),
    Command::new("klm")
        .about("Prepare and inspect DMA/LDS/DPP descriptors for the klm framework & Kognitara")
        .args(&KLM_ARGS),
    Command::new("worker")
        .about("Internal subcommand for executing an isolated logical worker node (0..7)")
        .args(&WORKER_ARGS),
];

/// Returns the root `kl` CLI specification built with [`wuu::Command`].
#[must_use]
pub const fn kl_cli() -> Command<'static> {
    Command::new("kl")
        .about("Distributed Kpack job execution engine over 8 logical nodes (Hamon Cube 2^3)")
        .subcommands(&SUBCOMMANDS)
}

/// Formats the first 4 bytes (8 hex digits) of a [`Noun`].
#[must_use]
pub fn format_noun_short(noun: &Noun) -> StackString<256> {
    let b = noun.as_bytes();
    wuu::format_stack!("{:02x}{:02x}{:02x}{:02x}", b[0], b[1], b[2], b[3])
}

/// Renders the 8-node HIC/HID hypercube topology table and the 3-round reduction schedule.
pub fn render_topology<W: Write>(
    config: &CubeConfig,
    color_mode: ColorMode,
    output: &mut W,
) -> Result<(), &'static str> {
    let cfg_noun = format_noun_short(&config.config_noun());
    let header_msg = wuu::format_stack!(
        "HIC/HID Topology: hypercube_3d (2^3 = 8 nodes, degree=3, diameter=3, avg_hops=1.5) transport={} noun={}",
        config.transport.as_str(),
        cfg_noun.as_str()
    );
    write_styled_message(output, MessageLevel::Info, header_msg.as_str(), color_mode)
        .map_err(|_| "Failed to write topology header")?;

    let mut id_strs: Vec<StackString<256>, NODE_COUNT> = Vec::new();
    let mut coord_strs: Vec<StackString<256>, NODE_COUNT> = Vec::new();
    let mut neigh_strs: Vec<StackString<256>, NODE_COUNT> = Vec::new();
    let mut role_strs: Vec<&'static str, NODE_COUNT> = Vec::new();
    let mut hw_strs: Vec<StackString<256>, NODE_COUNT> = Vec::new();
    let mut ep_strs: Vec<StackString<256>, NODE_COUNT> = Vec::new();

    for node in &config.nodes {
        let _ = id_strs.push(wuu::format_stack!(
            "n{} ({}{}{})",
            node.id,
            node.coords[0],
            node.coords[1],
            node.coords[2]
        ));
        let _ = coord_strs.push(wuu::format_stack!(
            "[{},{},{}]",
            node.coords[0],
            node.coords[1],
            node.coords[2]
        ));
        let _ = neigh_strs.push(wuu::format_stack!(
            "[{},{},{}]",
            node.neighbors[0],
            node.neighbors[1],
            node.neighbors[2]
        ));
        let _ = role_strs.push(node.role.as_str());
        let _ = hw_strs.push(wuu::format_stack!(
            "CPU{} / NUMA{}",
            node.cpu_core,
            node.numa_node
        ));
        let _ = ep_strs.push(wuu::format_stack!(
            "{}.{}.{}.{}:{}",
            node.ip[0],
            node.ip[1],
            node.ip[2],
            node.ip[3],
            node.port
        ));
    }

    let mut cell_rows: Vec<[&str; 6], NODE_COUNT> = Vec::new();
    for i in 0..NODE_COUNT {
        let _ = cell_rows.push([
            id_strs[i].as_str(),
            coord_strs[i].as_str(),
            neigh_strs[i].as_str(),
            role_strs[i],
            hw_strs[i].as_str(),
            ep_strs[i].as_str(),
        ]);
    }

    let mut row_refs: Vec<&[&str], NODE_COUNT> = Vec::new();
    for r in &cell_rows {
        let _ = row_refs.push(r.as_slice());
    }

    let headers = ["VERTEX", "COORDS", "NEIGHBORS", "ROLE", "AFFINITY", "ENDPOINT"];
    let table = Table::new(&headers, row_refs.as_slice()).with_style(TableStyle::Ascii);
    table
        .write_to(output)
        .map_err(|_| "Failed to write topology table")?;

    writeln!(
        output,
        "Reduction Schedule (log2(8) = 3 rounds -> Coordinator n0):"
    )
    .map_err(|_| "Write error")?;
    writeln!(
        output,
        "  Round 1 (dim 0, dist 1): (1->0), (3->2), (5->4), (7->6)"
    )
    .map_err(|_| "Write error")?;
    writeln!(output, "  Round 2 (dim 1, dist 2): (2->0), (6->4)")
        .map_err(|_| "Write error")?;
    writeln!(output, "  Round 3 (dim 2, dist 4): (4->0) [Full 0..7]")
        .map_err(|_| "Write error")?;

    Ok(())
}

/// Renders a completed `CubeExecutionReport` (8-node Kpack formulas + 3-round reduction + Merkle seal).
pub fn render_execution_report<W: Write>(
    report: &CubeExecutionReport,
    color_mode: ColorMode,
    output: &mut W,
) -> Result<(), &'static str> {
    let merkle_short = format_noun_short(&report.merkle_root);
    let input_short = format_noun_short(&report.input_noun);
    let seal_short = format_noun_short(&report.final_plan.seal());

    let banner = wuu::format_stack!(
        "KL Job Complete: raw={}B recipes={}B input_noun={} merkle_root={} plan_seal={}",
        report.total_input_bytes,
        report.total_recipe_bytes,
        input_short.as_str(),
        merkle_short.as_str(),
        seal_short.as_str()
    );
    write_styled_message(output, MessageLevel::Info, banner.as_str(), color_mode)
        .map_err(|_| "Write error")?;

    let mut node_strs: Vec<StackString<256>, NODE_COUNT> = Vec::new();
    let mut op_strs: Vec<&'static str, NODE_COUNT> = Vec::new();
    let mut size_strs: Vec<StackString<256>, NODE_COUNT> = Vec::new();
    let mut noun_strs: Vec<StackString<256>, NODE_COUNT> = Vec::new();

    for i in 0..NODE_COUNT {
        let f = &report.formulas[i];
        let ncfg = &report.config.nodes[i];
        let _ = node_strs.push(wuu::format_stack!(
            "n{} (CPU{}/NUMA{})",
            i,
            ncfg.cpu_core,
            ncfg.numa_node
        ));
        let _ = op_strs.push(f.opcode_name());
        let _ = size_strs.push(wuu::format_stack!("{}B -> {}B", f.raw_len, f.payload_len));
        let _ = noun_strs.push(format_noun_short(&f.expected_noun));
    }

    let mut cell_rows: Vec<[&str; 4], NODE_COUNT> = Vec::new();
    for i in 0..NODE_COUNT {
        let _ = cell_rows.push([
            node_strs[i].as_str(),
            op_strs[i],
            size_strs[i].as_str(),
            noun_strs[i].as_str(),
        ]);
    }

    let mut row_refs: Vec<&[&str], NODE_COUNT> = Vec::new();
    for r in &cell_rows {
        let _ = row_refs.push(r.as_slice());
    }

    let headers = ["LOGICAL NODE", "OPCODE", "RAW -> RECIPE", "BLAKE3 NOUN"];
    let table = Table::new(&headers, row_refs.as_slice()).with_style(TableStyle::Ascii);
    table.write_to(output).map_err(|_| "Write error")?;

    // Show the 3 reduction rounds
    for t in &report.round1_traces {
        let n_short = format_noun_short(&t.combined_merkle_noun);
        let desc = wuu::format_stack!(
            "Round 1 (dist 1): n{} -> n{} mask=0b{:08b} merkle={}",
            t.sender,
            t.receiver,
            t.combined_mask,
            n_short.as_str()
        );
        write_ok_custom(output, desc.as_str(), "r1", 64, color_mode)
            .map_err(|_| "Write error")?;
    }
    for t in &report.round2_traces {
        let n_short = format_noun_short(&t.combined_merkle_noun);
        let desc = wuu::format_stack!(
            "Round 2 (dist 2): n{} -> n{} mask=0b{:08b} merkle={}",
            t.sender,
            t.receiver,
            t.combined_mask,
            n_short.as_str()
        );
        write_ok_custom(output, desc.as_str(), "r2", 64, color_mode)
            .map_err(|_| "Write error")?;
    }
    for t in &report.round3_traces {
        let n_short = format_noun_short(&t.combined_merkle_noun);
        let desc = wuu::format_stack!(
            "Round 3 (dist 4): n{} -> n{} mask=0b{:08b} merkle={}",
            t.sender,
            t.receiver,
            t.combined_mask,
            n_short.as_str()
        );
        write_ok_custom(output, desc.as_str(), "r3-root", 64, color_mode)
            .map_err(|_| "Write error")?;
    }

    Ok(())
}

/// Renders an inspected `KPK1` archive view using `wuu::Table`.
pub fn render_archive_inspection<W: Write>(
    view: &KpkArchiveView,
    color_mode: ColorMode,
    output: &mut W,
) -> Result<(), &'static str> {
    let merkle_short = format_noun_short(&view.merkle_root);
    let input_short = format_noun_short(&view.input_noun);
    let parity_short = format_noun_short(&view.parity.parity_noun);
    let info = wuu::format_stack!(
        "KPK1 v{} | nodes={} | raw={}B | input={} | merkle={} | parity={}",
        view.header.version,
        view.header.node_count,
        view.header.total_raw_size,
        input_short.as_str(),
        merkle_short.as_str(),
        parity_short.as_str()
    );
    write_styled_message(output, MessageLevel::Info, info.as_str(), color_mode)
        .map_err(|_| "Write error")?;

    let mut seg_strs: Vec<StackString<256>, NODE_COUNT> = Vec::new();
    let mut op_strs: Vec<&'static str, NODE_COUNT> = Vec::new();
    let mut off_strs: Vec<StackString<256>, NODE_COUNT> = Vec::new();
    let mut noun_strs: Vec<StackString<256>, NODE_COUNT> = Vec::new();

    for i in 0..NODE_COUNT {
        let entry = &view.index_entries[i];
        let f = &view.formulas[i];
        let _ = seg_strs.push(wuu::format_stack!("seg_{} (n{})", i, entry.node_id));
        let _ = op_strs.push(f.opcode_name());
        let _ = off_strs.push(wuu::format_stack!(
            "@{} ({}B -> {}B)",
            entry.recipe_offset,
            entry.recipe_len,
            entry.raw_len
        ));
        let _ = noun_strs.push(format_noun_short(&f.expected_noun));
    }

    let mut cell_rows: Vec<[&str; 4], NODE_COUNT> = Vec::new();
    for i in 0..NODE_COUNT {
        let _ = cell_rows.push([
            seg_strs[i].as_str(),
            op_strs[i],
            off_strs[i].as_str(),
            noun_strs[i].as_str(),
        ]);
    }

    let mut row_refs: Vec<&[&str], NODE_COUNT> = Vec::new();
    for r in &cell_rows {
        let _ = row_refs.push(r.as_slice());
    }

    let headers = ["SEGMENT", "OPCODE", "OFFSET & SIZE", "BLAKE3 NOUN"];
    let table = Table::new(&headers, row_refs.as_slice()).with_style(TableStyle::Ascii);
    table.write_to(output).map_err(|_| "Write error")?;

    Ok(())
}

/// Renders the 4-panel `wuu` Dashboard for the Hamon Cube (`kl`).
pub fn render_dashboard<W: Write>(
    report: &CubeExecutionReport,
    cols: usize,
    rows: usize,
    color_mode: ColorMode,
    output: &mut W,
) -> Result<(), &'static str> {
    let merkle_short = format_noun_short(&report.merkle_root);
    let seal_short = format_noun_short(&report.final_plan.seal());
    let parity_short = format_noun_short(&report.parity.parity_noun);

    let topo_panel_text = wuu::format_stack!(
        "Topology: hypercube_3d (2^3)\nNodes: 8 (n0 Coordinator, n1..n7)\nNUMA0: CPU0..3 | NUMA1: CPU4..7\nTransport: {} (ports {}..{})",
        report.config.transport.as_str(),
        report.config.nodes[0].port,
        report.config.nodes[7].port
    );

    let recipe_panel_text = wuu::format_stack!(
        "Input: {} B -> Recipes: {} B\nn0:{} n1:{} n2:{} n3:{}\nn4:{} n5:{} n6:{} n7:{}\nParity RS/XOR: {}",
        report.total_input_bytes,
        report.total_recipe_bytes,
        report.formulas[0].opcode_name(),
        report.formulas[1].opcode_name(),
        report.formulas[2].opcode_name(),
        report.formulas[3].opcode_name(),
        report.formulas[4].opcode_name(),
        report.formulas[5].opcode_name(),
        report.formulas[6].opcode_name(),
        report.formulas[7].opcode_name(),
        parity_short.as_str()
    );

    let reduction_panel_text = wuu::format_stack!(
        "R1 (d=1): (1->0)(3->2)(5->4)(7->6)\nR2 (d=2): (2->0)[0..3] (6->4)[4..7]\nR3 (d=4): (4->0)[0..7 mask=0xFF]\nMerkle Root: {}",
        merkle_short.as_str()
    );

    let sceau_panel_text = wuu::format_stack!(
        "Plan Layers: {}/8 merged\nPlan Seal: {}\nSceau Verdict: Accept (budget={})\nCheckpoint Seq: #{}",
        report.final_plan.get_layers().len(),
        seal_short.as_str(),
        report.final_sceau.budget,
        report.checkpoint.seq
    );

    let panels = [
        Panel::new(
            "HIC/HID 3D Cube",
            topo_panel_text.as_str(),
            Status::Focused,
            DropDownTerm::new("hic"),
        ),
        Panel::new(
            "Kpack 8-Node Formulas",
            recipe_panel_text.as_str(),
            Status::Normal,
            DropDownTerm::new("kpack"),
        ),
        Panel::new(
            "Hypercube 3-Round Reduce",
            reduction_panel_text.as_str(),
            Status::Normal,
            DropDownTerm::new("reduce"),
        ),
        Panel::new(
            "Plans, Sceau & Checkpoint",
            sceau_panel_text.as_str(),
            Status::Normal,
            DropDownTerm::open_with("kl", merkle_short.as_str(), "status"),
        ),
    ];

    if wuu::TERMINAL.lock().is_some() {
        let ui = Wuu::new(Context::Server { cols, rows });
        ui.render_dashboard("KL HYPERCUBE 2^3 - AMENTYS OS", &panels);
    }

    write_styled_message(
        output,
        MessageLevel::Info,
        "=== KL HYPERCUBE 2^3 DASHBOARD (AMENTYS OS) ===",
        color_mode,
    )
    .map_err(|_| "Write error")?;

    let mut progress = ProgressBar::with_message(8, "Hypercube Nodes Aggregated");
    progress.set_width(16);
    progress.set_position(8);
    progress.write_finish(output).map_err(|_| "Write error")?;

    for p in &panels {
        writeln!(output, "[{}] {}", p.title, p.content.replace('\n', " | "))
            .map_err(|_| "Write error")?;
    }

    Ok(())
}

/// Renders the `klm` AMD GPU & Kōgnitara Sanctuary readiness table for the 8 Hamon Cube vertices.
pub fn render_klm_readiness<W: Write>(
    report: &CubeExecutionReport,
    wave_mode: WaveMode,
    color_mode: ColorMode,
    output: &mut W,
) -> Result<(), &'static str> {
    let packet = KlmDispatchPacket::from_execution_report(report, wave_mode);
    packet.validate_hardware_constraints()?;
    let pkt_noun = format_noun_short(&packet.packet_noun());

    let banner = wuu::format_stack!(
        "KLM Dispatch Ready (KLMD v{}): mode={} ({} lanes) | dpp_xmask=[1,2,4] | lds={}B (bank-conflict-free) | dma_align=64B | noun={}",
        packet.version,
        wave_mode.as_str(),
        packet.wave_lanes,
        packet.total_lds_bytes,
        pkt_noun.as_str()
    );
    write_styled_message(output, MessageLevel::Info, banner.as_str(), color_mode)
        .map_err(|_| "Write error")?;

    let mut vtx_strs: Vec<StackString<256>, NODE_COUNT> = Vec::new();
    let mut caste_strs: Vec<&'static str, NODE_COUNT> = Vec::new();
    let mut real_strs: Vec<&'static str, NODE_COUNT> = Vec::new();
    let mut cu_strs: Vec<StackString<256>, NODE_COUNT> = Vec::new();
    let mut lds_strs: Vec<StackString<256>, NODE_COUNT> = Vec::new();
    let mut dma_strs: Vec<StackString<256>, NODE_COUNT> = Vec::new();

    for i in 0..NODE_COUNT {
        let slot = &packet.slots[i];
        let node = &report.config.nodes[i];
        let caste = AiCaste::for_vertex(i as u8);
        let reality = TriRealityProfile::for_vertex(i as u8);
        let gpu_target = node.amd_gpu_target(wave_mode);

        let _ = vtx_strs.push(wuu::format_stack!(
            "n{} ({}{}{})",
            i,
            node.coords[0],
            node.coords[1],
            node.coords[2]
        ));
        let _ = caste_strs.push(caste.as_str());
        let _ = real_strs.push(reality.dominant.as_str());
        if let ComputeTarget::AmdComputeUnit {
            xcd_id,
            cu_index,
            simd_unit,
            ..
        } = gpu_target
        {
            let _ = cu_strs.push(wuu::format_stack!(
                "XCD{}/CU{}/SIMD{}",
                xcd_id,
                cu_index,
                simd_unit
            ));
        } else {
            let _ = cu_strs.push(wuu::format_stack!("CPU{}", node.cpu_core));
        }
        let _ = lds_strs.push(wuu::format_stack!(
            "Bank#{} @+{}B",
            slot.lds_primary_bank,
            slot.lds_base_offset
        ));
        let _ = dma_strs.push(wuu::format_stack!(
            "@{} ({} {}B)",
            slot.dma_recipe_offset,
            report.formulas[i].opcode_name(),
            slot.recipe_len
        ));
    }

    let mut cell_rows: Vec<[&str; 6], NODE_COUNT> = Vec::new();
    for i in 0..NODE_COUNT {
        let _ = cell_rows.push([
            vtx_strs[i].as_str(),
            caste_strs[i],
            real_strs[i],
            cu_strs[i].as_str(),
            lds_strs[i].as_str(),
            dma_strs[i].as_str(),
        ]);
    }

    let mut row_refs: Vec<&[&str], NODE_COUNT> = Vec::new();
    for r in &cell_rows {
        let _ = row_refs.push(r.as_slice());
    }

    let headers = [
        "VERTEX",
        "CASTE",
        "REALITY",
        "AMD GPU TARGET",
        "LDS BANK & OFFSET",
        "DMA64 RECIPE",
    ];
    let table = Table::new(&headers, row_refs.as_slice()).with_style(TableStyle::Ascii);
    table.write_to(output).map_err(|_| "Write error")?;

    let mut sample_vgpr = [0u32; WAVE64_LANES];
    for lane in 0..WAVE64_LANES {
        sample_vgpr[lane] = report.formulas[lane & 0x07].raw_len as u32;
    }
    let bfly = execute_wave64_dpp_butterfly(&sample_vgpr);
    let dpp_msg = wuu::format_stack!(
        "Wave64 DPP Butterfly (XOR 1->2->4): subcube0={}B | wave64_sum={}B across 8 Hamon sub-cubes",
        bfly.subcube_sums[0],
        bfly.wave64_total_sum
    );
    write_ok_custom(output, dpp_msg.as_str(), "klm-dpp", 68, color_mode)
        .map_err(|_| "Write error")?;

    Ok(())
}

/// Pure `#![no_std]` CLI executor for in-memory testing and embedded kernel integration.
pub fn execute_cli_no_std<W: Write>(
    config: &CubeConfig,
    sample_payload: &[u8],
    argv: &[&str],
    color_mode: ColorMode,
    output: &mut W,
) -> Result<(), &'static str> {
    if argv.is_empty() || argv[0] == "help" || argv[0] == "--help" || argv[0] == "-h" {
        let cmd = kl_cli();
        cmd.write_help(output).map_err(|_| "Help write error")?;
        return Ok(());
    }

    let root_cmd = kl_cli();
    let parsed = root_cmd
        .parse(argv)
        .map_err(|_| "Invalid CLI command or option")?;

    let Some(sub_name) = parsed.subcommand_name() else {
        root_cmd
            .write_help(output)
            .map_err(|_| "Help write error")?;
        return Ok(());
    };
    let sub = parsed
        .subcommand_matches()
        .map_err(|_| "Invalid subcommand arguments")?
        .ok_or("Missing subcommand")?;

    match sub_name {
        "init" => {
            let transport = sub
                .value("transport")
                .and_then(TransportKind::from_str)
                .unwrap_or(config.transport);
            let cfg = CubeConfig::new(config.base_port, transport);
            let yaml = cfg.to_yaml();
            writeln!(output, "{yaml}").map_err(|_| "Write error")
        }
        "topology" => render_topology(config, color_mode, output),
        "run" => {
            let payload = sub
                .value("data")
                .map_or(sample_payload, str::as_bytes);
            let report = execute_distributed_cube_job(config, payload, 1, 1_700_000_000)?;
            render_execution_report(&report, color_mode, output)
        }
        "dashboard" => {
            let cols = sub
                .value("cols")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(80);
            let rows = sub
                .value("rows")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(24);
            let report = execute_distributed_cube_job(config, sample_payload, 1, 1_700_000_000)?;
            render_dashboard(&report, cols, rows, color_mode, output)
        }
        "klm" => {
            let wave_mode = match sub.value("wave") {
                Some("wave32") => WaveMode::Wave32,
                _ => WaveMode::Wave64,
            };
            let report = execute_distributed_cube_job(config, sample_payload, 1, 1_700_000_000)?;
            render_klm_readiness(&report, wave_mode, color_mode, output)
        }
        "inspect" => {
            let _ = inspect_kpk1_archive;
            Err("Use std file I/O entrypoint for archive file inspection")
        }
        _ => Err("Subcommand requires OS runtime"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_topology_run_and_dashboard_rendering() {
        let cfg = CubeConfig::default();
        let sample = [b'K'; 512];

        let mut out_topo = StackString::<4096>::new();
        execute_cli_no_std(
            &cfg,
            &sample,
            &["topology"],
            ColorMode::Plain,
            &mut out_topo,
        )
        .expect("topology CLI should succeed");
        assert!(out_topo.as_str().contains("hypercube_3d"));
        assert!(out_topo.as_str().contains("Round 3 (dim 2, dist 4): (4->0)"));

        let mut out_run = StackString::<4096>::new();
        execute_cli_no_std(&cfg, &sample, &["run"], ColorMode::Plain, &mut out_run)
            .expect("run CLI should succeed");
        assert!(out_run.as_str().contains("RLE"));
        assert!(out_run.as_str().contains("Round 3 (dist 4): n4 -> n0"));

        let mut out_dash = StackString::<4096>::new();
        execute_cli_no_std(
            &cfg,
            &sample,
            &["dashboard"],
            ColorMode::Plain,
            &mut out_dash,
        )
        .expect("dashboard CLI should succeed");
        assert!(out_dash.as_str().contains("HIC/HID 3D Cube"));
        assert!(out_dash.as_str().contains("Hypercube 3-Round Reduce"));

        let mut out_klm = StackString::<4096>::new();
        execute_cli_no_std(&cfg, &sample, &["klm"], ColorMode::Plain, &mut out_klm)
            .expect("klm CLI should succeed");
        assert!(out_klm.as_str().contains("KLM Dispatch Ready (KLMD v1)"));
        assert!(out_klm.as_str().contains("Queen"));
        assert!(out_klm.as_str().contains("Guardian"));
        assert!(out_klm.as_str().contains("Fulgaris"));
    }
}
