//! # `kl` Binary Entrypoint
//!
//! OS command-line entrypoint for `kl`. When invoked without arguments, runs the complete
//! end-to-end demonstration:
//! 1. Renders the 8-node 3D hypercube (`HIC`/`HID`) topology and 3-round reduction schedule.
//! 2. Executes a 1024-byte multi-formula `kpack` job across all 8 loopback nodes with CPU core pinning.
//! 3. Packs, seals, and verifies a 64-byte DMA-aligned `KPK1` archive.
//! 4. Renders the `klm` AMD GPU (`Wave64` / `DPP` / `LDS`) and `Kōgnitara` (`Queen`, `Worker`,
//!    `Scout`, `Fulgaris`, `Guardian`) readiness table.
//! 5. Renders the 4-panel `wuu` TUI dashboard.

use core::fmt::Write as FmtWrite;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use kl::archive::{
    MAX_ARCHIVE_BYTES, extract_kpk1_all, extract_kpk1_segment, inspect_kpk1_archive,
    pack_kpk1_archive,
};
use kl::cli::{
    format_noun_short, kl_cli, render_archive_inspection, render_dashboard,
    render_execution_report, render_klm_readiness, render_topology,
};
use kl::recipe::{DEFAULT_DICT, MAX_SEGMENT_BYTES};
use kl::topology::{CubeConfig, NODE_COUNT, TransportKind, WaveMode};
use kl::transport::{
    execute_loopback_cube_job, execute_multiprocess_cube_job, pin_current_to_cpu_core,
};
use wuu::{ColorMode, write_ko_custom, write_ok_custom, write_styled_message};

struct StdoutWriter;

impl FmtWrite for StdoutWriter {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        print!("{s}");
        Ok(())
    }
}

fn current_unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(1_700_000_000)
}

fn build_demo_job_payload() -> [u8; 1024] {
    let mut job_data = [0u8; 1024];
    // Node 0 (0..128): DICT 4-byte words from DEFAULT_DICT
    for chunk in job_data[0..128].chunks_exact_mut(32) {
        chunk.copy_from_slice(DEFAULT_DICT);
    }
    // Node 1 (128..256): RLE run of 'K'
    job_data[128..256].fill(b'K');
    // Node 2 (256..384): Base structured payload
    for i in 256..384 {
        job_data[i] = b'A' + ((i % 26) as u8);
    }
    // Node 3 (384..512): DELTA relative to Node 2 (shares 112-byte prefix + 16-byte tail patch)
    job_data.copy_within(256..368, 384);
    job_data[496..512].copy_from_slice(b"-KL-DELTA-PATCH!");
    // Node 4 (512..640): RLE run of 0xAA
    job_data[512..640].fill(0xAA);
    // Node 5 (640..768): DICT words
    for chunk in job_data[640..768].chunks_exact_mut(32) {
        chunk.copy_from_slice(DEFAULT_DICT);
    }
    // Node 6 (768..896) & Node 7 (896..1024): DELTA + LIT
    for i in 768..896 {
        job_data[i] = b'0' + ((i % 10) as u8);
    }
    job_data.copy_within(768..880, 896);
    job_data[1008..1024].copy_from_slice(b"-CUBE-VERTEX-111");
    job_data
}

fn run_default_demo(out: &mut StdoutWriter) -> Result<(), &'static str> {
    let cfg = CubeConfig::default();
    render_topology(&cfg, ColorMode::Ansi, out)?;

    let demo_payload = build_demo_job_payload();
    let ts = current_unix_timestamp();
    let (report, net_stats) = execute_loopback_cube_job(&cfg, &demo_payload, 1, ts)?;

    render_execution_report(&report, ColorMode::Ansi, out)?;

    let net_desc = wuu::format_stack!(
        "Loopback {} sockets: {} packets ({} B) | CPU pinned: {}/8 | {} us",
        net_stats.transport.as_str(),
        net_stats.packets_exchanged,
        net_stats.bytes_transferred,
        net_stats.cpu_pinned_count,
        net_stats.elapsed_micros
    );
    write_ok_custom(out, net_desc.as_str(), "net-hic", 68, ColorMode::Ansi)
        .map_err(|_| "Write error")?;

    let mut archive_buf = [0u8; MAX_ARCHIVE_BYTES];
    let archive_len = pack_kpk1_archive(&report, &mut archive_buf)?;
    let mut restored = [0u8; 1024];
    let (restored_len, _) = extract_kpk1_all(&archive_buf[..archive_len], &mut restored)?;
    if restored_len != demo_payload.len() || restored != demo_payload {
        return Err("Demo KPK1 roundtrip verification failed");
    }

    let kpk_desc = wuu::format_stack!(
        "KPK1 archive sealed ({} B, DMA64 aligned) & verified 8/8 nodes + parity stripe",
        archive_len
    );
    write_ok_custom(out, kpk_desc.as_str(), "kpk1", 68, ColorMode::Ansi)
        .map_err(|_| "Write error")?;

    render_klm_readiness(&report, WaveMode::Wave64, ColorMode::Ansi, out)?;
    render_dashboard(&report, 80, 24, ColorMode::Ansi, out)
}

fn run_cli_from_env() -> Result<(), &'static str> {
    let raw_args: std::vec::Vec<std::string::String> = std::env::args().skip(1).collect();
    let mut out = StdoutWriter;

    if raw_args.is_empty() {
        return run_default_demo(&mut out);
    }

    let arg_refs: std::vec::Vec<&str> = raw_args.iter().map(std::string::String::as_str).collect();
    if arg_refs[0] == "help" || arg_refs[0] == "--help" || arg_refs[0] == "-h" {
        let cmd = kl_cli();
        cmd.write_help(&mut out).map_err(|_| "Help write error")?;
        return Ok(());
    }

    let root_cmd = kl_cli();
    let parsed = root_cmd
        .parse(&arg_refs)
        .map_err(|_| "Invalid CLI command or option")?;

    let Some(sub_name) = parsed.subcommand_name() else {
        root_cmd
            .write_help(&mut out)
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
                .unwrap_or(TransportKind::Tcp);
            let out_path = sub.value("output").unwrap_or("kl.yaml");
            let cfg = CubeConfig::new(8000, transport);
            let yaml = cfg.to_yaml();
            fs::write(out_path, yaml.as_str()).map_err(|_| "Failed to write YAML configuration file")?;
            let noun_short = format_noun_short(&cfg.config_noun());
            let desc = wuu::format_stack!(
                "wrote {} (hypercube_3d, transport={}, noun={})",
                out_path,
                transport.as_str(),
                noun_short.as_str()
            );
            write_ok_custom(&mut out, desc.as_str(), "init", 64, ColorMode::Ansi)
                .map_err(|_| "Write error")
        }
        "topology" => {
            let cfg = if let Some(path) = sub.value("config") {
                let text =
                    fs::read_to_string(path).map_err(|_| "Failed to read YAML configuration file")?;
                CubeConfig::from_yaml(&text)?
            } else if let Ok(text) = fs::read_to_string("kl.yaml") {
                CubeConfig::from_yaml(&text).unwrap_or_default()
            } else {
                CubeConfig::default()
            };
            render_topology(&cfg, ColorMode::Ansi, &mut out)
        }
        "run" => {
            let transport = sub
                .value("transport")
                .and_then(TransportKind::from_str)
                .unwrap_or(TransportKind::Tcp);
            let cfg = CubeConfig::new(8000, transport);

            let input_vec: std::vec::Vec<u8> = if let Some(path) = sub.value("input") {
                fs::read(path).map_err(|_| "Failed to read --input file")?
            } else if let Some(text) = sub.value("data") {
                text.as_bytes().to_vec()
            } else {
                build_demo_job_payload().to_vec()
            };

            let max_job_bytes = MAX_SEGMENT_BYTES * NODE_COUNT;
            let clamped_slice = if input_vec.len() > max_job_bytes {
                &input_vec[..max_job_bytes]
            } else {
                &input_vec[..]
            };

            let ts = current_unix_timestamp();
            let (report, stats) = if sub.flag("multiprocess") {
                execute_multiprocess_cube_job(&cfg, clamped_slice, 1, ts)?
            } else {
                execute_loopback_cube_job(&cfg, clamped_slice, 1, ts)?
            };

            render_execution_report(&report, ColorMode::Ansi, &mut out)?;

            let out_archive_path = sub.value("output").unwrap_or("out.kpk");
            let mut archive_buf = [0u8; MAX_ARCHIVE_BYTES];
            let archive_len = pack_kpk1_archive(&report, &mut archive_buf)?;
            fs::write(out_archive_path, &archive_buf[..archive_len])
                .map_err(|_| "Failed to write KPK1 archive")?;

            let desc = wuu::format_stack!(
                "sealed {} ({} B, {} packets via {}, multiprocess={})",
                out_archive_path,
                archive_len,
                stats.packets_exchanged,
                stats.transport.as_str(),
                stats.multiprocess
            );
            write_ok_custom(&mut out, desc.as_str(), "kpk1", 68, ColorMode::Ansi)
                .map_err(|_| "Write error")
        }
        "inspect" => {
            let path = sub.value("archive").ok_or("KPK1 archive path required")?;
            let bytes = fs::read(path).map_err(|_| "Failed to read KPK1 archive")?;
            let view = inspect_kpk1_archive(&bytes)?;
            render_archive_inspection(&view, ColorMode::Ansi, &mut out)
        }
        "extract" => {
            let path = sub.value("archive").ok_or("KPK1 archive path required")?;
            let bytes = fs::read(path).map_err(|_| "Failed to read KPK1 archive")?;

            if let Some(seg_str) = sub.value("segment") {
                let node_id = seg_str
                    .parse::<u8>()
                    .map_err(|_| "Invalid segment identifier (expected: 0..7)")?;
                let (seg_buf, seg_len, repaired) = extract_kpk1_segment(&bytes, node_id)?;
                if let Some(out_path) = sub.value("output") {
                    fs::write(out_path, &seg_buf[..seg_len])
                        .map_err(|_| "Failed to write extracted segment")?;
                } else if let Ok(txt) = core::str::from_utf8(&seg_buf[..seg_len]) {
                    println!("{txt}");
                }
                let desc = wuu::format_stack!(
                    "extracted segment n{} ({} B, parity_repaired={})",
                    node_id,
                    seg_len,
                    repaired
                );
                write_ok_custom(&mut out, desc.as_str(), "extract", 64, ColorMode::Ansi)
                    .map_err(|_| "Write error")
            } else {
                let view = inspect_kpk1_archive(&bytes)?;
                let mut restored = vec![0u8; view.header.total_raw_size as usize];
                let (len, repaired) = extract_kpk1_all(&bytes, &mut restored)?;
                if let Some(out_path) = sub.value("output") {
                    fs::write(out_path, &restored[..len])
                        .map_err(|_| "Failed to write extracted file")?;
                }
                let desc = wuu::format_stack!(
                    "extracted full archive ({} B, parity_repaired={})",
                    len,
                    repaired
                );
                write_ok_custom(&mut out, desc.as_str(), "extract", 64, ColorMode::Ansi)
                    .map_err(|_| "Write error")
            }
        }
        "deploy" => {
            let archive_path = sub.value("archive").ok_or("KPK1 archive required")?;
            let target_path = sub.value("target").ok_or("Deployment target required")?;
            let bytes = fs::read(archive_path).map_err(|_| "Failed to read KPK1 archive")?;
            let view = inspect_kpk1_archive(&bytes)?;
            let mut restored = vec![0u8; view.header.total_raw_size as usize];
            let _ = extract_kpk1_all(&bytes, &mut restored)?;
            fs::write(target_path, &bytes)
                .map_err(|_| "Failed to deploy archive to target")?;
            let merkle_short = format_noun_short(&view.merkle_root);
            let seal_short = format_noun_short(&view.plan_seal);
            let desc = wuu::format_stack!(
                "Coordinator n0 deployed {} -> {} (merkle={}, seal={})",
                archive_path,
                target_path,
                merkle_short.as_str(),
                seal_short.as_str()
            );
            write_ok_custom(&mut out, desc.as_str(), "deployed", 68, ColorMode::Ansi)
                .map_err(|_| "Write error")
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
            let cfg = CubeConfig::default();
            let demo_payload = build_demo_job_payload();
            let (report, _) =
                execute_loopback_cube_job(&cfg, &demo_payload, 1, current_unix_timestamp())?;
            render_dashboard(&report, cols, rows, ColorMode::Ansi, &mut out)
        }
        "klm" => {
            let wave_mode = match sub.value("wave") {
                Some("wave32") => WaveMode::Wave32,
                _ => WaveMode::Wave64,
            };
            let cfg = CubeConfig::default();
            let demo_payload = build_demo_job_payload();
            let (report, _) =
                execute_loopback_cube_job(&cfg, &demo_payload, 1, current_unix_timestamp())?;
            render_klm_readiness(&report, wave_mode, ColorMode::Ansi, &mut out)
        }
        "worker" => {
            let id = sub
                .value("id")
                .and_then(|s| s.parse::<u8>().ok())
                .unwrap_or(1);
            let cpu = sub
                .value("cpu")
                .and_then(|s| s.parse::<u8>().ok())
                .unwrap_or(id);
            let _ = pin_current_to_cpu_core(cpu);
            Ok(())
        }
        _ => {
            let _ = write_styled_message(
                &mut out,
                wuu::MessageLevel::Error,
                "Unknown subcommand",
                ColorMode::Ansi,
            );
            Err("Unknown subcommand")
        }
    }
}

fn main() {
    if let Err(err) = run_cli_from_env() {
        let mut out = StdoutWriter;
        let _ = write_ko_custom(&mut out, "kl error", err, 60, ColorMode::Ansi);
        std::process::exit(1);
    }
}
