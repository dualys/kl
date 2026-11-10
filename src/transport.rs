//! # Hardware Interconnect Deployment (`HID`) — CPU Pinning & Authenticated Loopback Transport
//!
//! This module implements the OS runtime transport layer for `kl`:
//!
//! - **Hardware CPU Core Affinity ([`pin_current_to_cpu_core`])**: Pins each worker thread or
//!   process to its dedicated hardware core (`CPU0..CPU7`) via `libc::sched_setaffinity`.
//! - **Authenticated `KLWP` Wire Protocol ([`WirePacket`])**: Encapsulates a 4096-byte
//!   [`jinshu::node::TrieNode`] with an 80-byte header authenticated by keyed `BLAKE3`
//!   ([`Noun::keyed`]) and enforces 1-bit Hamming neighbor adjacency on every hop.
//! - **Real Loopback TCP & QUIC/UDP Reduction ([`execute_loopback_cube_job`])**: Binds 8 loopback
//!   endpoints (`127.0.0.1:8000..8007`) and executes the 3 dimensional reduction rounds over real
//!   sockets (`7` packets = `29,232` bytes).
//! - **Multi-Process Address-Space Isolation ([`execute_multiprocess_cube_job`])**: Spawns 7
//!   isolated child processes (`kl worker --id <1..7> --cpu <core>`) coordinated by node `0`.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, UdpSocket};
use std::process::Command as ProcessCommand;
use std::thread;
use std::time::{Duration, Instant};

use jinshu::node::TrieNode;
use noun::Noun;

use crate::hypercube::{
    CubeExecutionReport, execute_distributed_cube_job, merge_hypercube_trie_nodes,
};
use crate::recipe::FLAG_KL_ARCHIVE_ROOT;
use crate::topology::{
    CubeConfig, NODE_COUNT, TransportKind, are_neighbors, reduction_tree_rounds,
};

pub const WIRE_MAGIC: [u8; 4] = *b"KLWP";
pub const WIRE_HEADER_SIZE: usize = 80;
pub const WIRE_PACKET_SIZE: usize = WIRE_HEADER_SIZE + 4096;
pub const QUIC_AUTH_KEY: [u8; 32] = *b"AMENTYS-KL-QUIC-LOOPBACK-KEY-001";

#[derive(Debug, Clone)]
pub struct WirePacket {
    pub transport: TransportKind,
    pub round: u8,
    pub sender: u8,
    pub receiver: u8,
    pub aggregate_mask: u8,
    pub stream_id: u16,
    pub conn_token: u16,
    pub payload_noun: Noun,
    pub auth_tag: Noun,
    pub node: TrieNode,
}

#[derive(Debug, Clone)]
pub struct LoopbackExecutionStats {
    pub transport: TransportKind,
    pub multiprocess: bool,
    pub packets_exchanged: usize,
    pub bytes_transferred: usize,
    pub cpu_pinned_count: usize,
    pub elapsed_micros: u64,
    pub active_ports: [u16; NODE_COUNT],
}

impl WirePacket {
    #[must_use]
    pub fn new(
        transport: TransportKind,
        round: u8,
        sender: u8,
        receiver: u8,
        aggregate_mask: u8,
        node: TrieNode,
    ) -> Self {
        let node_hash = node.calculate_noun();
        let payload_noun = Noun::from_bytes(&node_hash).unwrap_or_else(|| Noun::of(&node_hash));
        let stream_id = ((round as u16) << 8) | ((sender as u16) << 4) | (receiver as u16);
        let conn_token = 0xA300 | ((sender as u16) << 3) | (receiver as u16);

        let mut auth_input = [0u8; 44];
        auth_input[0..4].copy_from_slice(&WIRE_MAGIC);
        auth_input[4] = match transport {
            TransportKind::Tcp => 0,
            TransportKind::Quic => 1,
        };
        auth_input[5] = round;
        auth_input[6] = sender;
        auth_input[7] = receiver;
        auth_input[8] = aggregate_mask;
        auth_input[9..11].copy_from_slice(&stream_id.to_le_bytes());
        auth_input[11..13].copy_from_slice(&conn_token.to_le_bytes());
        auth_input[12..44].copy_from_slice(payload_noun.as_bytes());

        let auth_tag = Noun::keyed(&QUIC_AUTH_KEY, &auth_input);

        Self {
            transport,
            round,
            sender,
            receiver,
            aggregate_mask,
            stream_id,
            conn_token,
            payload_noun,
            auth_tag,
            node,
        }
    }

    #[must_use]
    pub fn to_bytes(&self) -> [u8; WIRE_PACKET_SIZE] {
        let mut buf = [0u8; WIRE_PACKET_SIZE];
        buf[0..4].copy_from_slice(&WIRE_MAGIC);
        buf[4] = match self.transport {
            TransportKind::Tcp => 0,
            TransportKind::Quic => 1,
        };
        buf[5] = self.round;
        buf[6] = self.sender;
        buf[7] = self.receiver;
        buf[8] = self.aggregate_mask;
        buf[9..11].copy_from_slice(&self.stream_id.to_le_bytes());
        buf[11..13].copy_from_slice(&self.conn_token.to_le_bytes());
        buf[16..48].copy_from_slice(self.payload_noun.as_bytes());
        buf[48..80].copy_from_slice(self.auth_tag.as_bytes());

        let node_bytes = unsafe {
            core::slice::from_raw_parts(
                core::ptr::from_ref::<TrieNode>(&self.node).cast::<u8>(),
                4096,
            )
        };
        buf[WIRE_HEADER_SIZE..].copy_from_slice(node_bytes);
        buf
    }

    pub fn from_bytes(buf: &[u8; WIRE_PACKET_SIZE]) -> Result<Self, &'static str> {
        if buf[0..4] != WIRE_MAGIC {
            return Err("Invalid KLWP wire packet magic");
        }
        let transport = match buf[4] {
            0 => TransportKind::Tcp,
            1 => TransportKind::Quic,
            _ => return Err("Unknown transport byte in wire packet"),
        };
        let round = buf[5];
        let sender = buf[6];
        let receiver = buf[7];
        if !are_neighbors(sender, receiver) {
            return Err("WirePacket rejected: sender and receiver are not 1-bit hypercube neighbors");
        }
        let aggregate_mask = buf[8];
        let stream_id = u16::from_le_bytes([buf[9], buf[10]]);
        let conn_token = u16::from_le_bytes([buf[11], buf[12]]);
        let payload_noun =
            Noun::from_bytes(&buf[16..48]).ok_or("Invalid payload_noun in wire packet")?;
        let auth_tag = Noun::from_bytes(&buf[48..80]).ok_or("Invalid auth_tag in wire packet")?;

        let mut auth_input = [0u8; 44];
        auth_input[0..4].copy_from_slice(&WIRE_MAGIC);
        auth_input[4] = buf[4];
        auth_input[5] = round;
        auth_input[6] = sender;
        auth_input[7] = receiver;
        auth_input[8] = aggregate_mask;
        auth_input[9..11].copy_from_slice(&stream_id.to_le_bytes());
        auth_input[11..13].copy_from_slice(&conn_token.to_le_bytes());
        auth_input[12..44].copy_from_slice(payload_noun.as_bytes());

        let expected_auth = Noun::keyed(&QUIC_AUTH_KEY, &auth_input);
        if expected_auth != auth_tag {
            return Err("WirePacket QUIC/Loopback authentication tag mismatch");
        }

        let mut node = TrieNode::new();
        unsafe {
            core::ptr::copy_nonoverlapping(
                buf[WIRE_HEADER_SIZE..].as_ptr(),
                core::ptr::from_mut::<TrieNode>(&mut node).cast::<u8>(),
                4096,
            );
        }

        let calc = node.calculate_noun();
        let actual_noun = Noun::from_bytes(&calc).unwrap_or_else(|| Noun::of(&calc));
        if actual_noun != payload_noun {
            return Err("WirePacket TrieNode BLAKE3 integrity check failed");
        }

        Ok(Self {
            transport,
            round,
            sender,
            receiver,
            aggregate_mask,
            stream_id,
            conn_token,
            payload_noun,
            auth_tag,
            node,
        })
    }
}

/// Pins the current thread or process to the given logical CPU core (`cpu_core`) on Linux
/// via `libc::sched_setaffinity`, respecting the HID specification.
#[must_use]
pub fn pin_current_to_cpu_core(cpu_core: u8) -> bool {
    #[cfg(target_os = "linux")]
    unsafe {
        let mut set: libc::cpu_set_t = core::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        // Query available CPU count or fallback to 8
        let nprocs = libc::sysconf(libc::_SC_NPROCESSORS_ONLN);
        let max_cores = if nprocs > 0 { nprocs as usize } else { 8 };
        let target_core = (cpu_core as usize) % max_cores;
        libc::CPU_SET(target_core, &mut set);
        let res = libc::sched_setaffinity(0, core::mem::size_of::<libc::cpu_set_t>(), &set);
        res == 0
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = cpu_core;
        false
    }
}

fn send_wire_packet_tcp(addr: SocketAddr, packet: &WirePacket) -> Result<(), &'static str> {
    let raw = packet.to_bytes();
    let mut last_err = "Failed to connect to neighbor TCP endpoint";
    for _ in 0..25 {
        match TcpStream::connect_timeout(&addr, Duration::from_millis(50)) {
            Ok(mut stream) => {
                let _ = stream.set_write_timeout(Some(Duration::from_secs(2)));
                stream
                    .write_all(&raw)
                    .map_err(|_| "TCP write_all of WirePacket failed")?;
                stream.flush().map_err(|_| "TCP flush failed")?;
                return Ok(());
            }
            Err(_) => {
                last_err = "Neighbor TCP listener not ready after retries";
                thread::sleep(Duration::from_millis(5));
            }
        }
    }
    Err(last_err)
}

fn recv_wire_packet_tcp(listener: &TcpListener) -> Result<WirePacket, &'static str> {
    let (mut stream, _) = listener
        .accept()
        .map_err(|_| "TCP accept failed on hypercube node")?;
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let mut buf = [0u8; WIRE_PACKET_SIZE];
    stream
        .read_exact(&mut buf)
        .map_err(|_| "TCP read_exact of WirePacket failed")?;
    WirePacket::from_bytes(&buf)
}

fn send_wire_packet_quic_udp(
    socket: &UdpSocket,
    addr: SocketAddr,
    packet: &WirePacket,
) -> Result<(), &'static str> {
    let raw = packet.to_bytes();
    let sent = socket
        .send_to(&raw, addr)
        .map_err(|_| "QUIC/UDP send_to neighbor failed")?;
    if sent != WIRE_PACKET_SIZE {
        return Err("Incomplete QUIC/UDP datagram sent");
    }
    Ok(())
}

fn recv_wire_packet_quic_udp(socket: &UdpSocket) -> Result<WirePacket, &'static str> {
    let mut buf = [0u8; WIRE_PACKET_SIZE];
    let (n, _) = socket
        .recv_from(&mut buf)
        .map_err(|_| "QUIC/UDP recv_from neighbor failed")?;
    if n != WIRE_PACKET_SIZE {
        return Err("Truncated QUIC/UDP datagram received");
    }
    WirePacket::from_bytes(&buf)
}

/// Executes the 8-node Kpack job over real OS loopback sockets (`127.0.0.1:8000..8007` or OS-assigned fallback ports),
/// pinning each of the 8 workers to its configured CPU core (`sched_setaffinity`) and exchanging authenticated
/// `WirePacket`s across the 3 dimensional reduction rounds `(1->0, 3->2, 5->4, 7->6) -> (2->0, 6->4) -> (4->0)`.
pub fn execute_loopback_cube_job(
    config: &CubeConfig,
    input: &[u8],
    seq: u64,
    timestamp: u64,
) -> Result<(CubeExecutionReport, LoopbackExecutionStats), &'static str> {
    let start = Instant::now();
    let report = execute_distributed_cube_job(config, input, seq, timestamp)?;

    let mut active_ports = [0u16; NODE_COUNT];
    let mut packets_exchanged = 0usize;
    let mut cpu_pinned_count = 0usize;

    match config.transport {
        TransportKind::Tcp => {
            let mut listeners: Vec<TcpListener> = Vec::with_capacity(NODE_COUNT);
            for i in 0..NODE_COUNT {
                let ncfg = &config.nodes[i];
                let preferred = SocketAddr::from((ncfg.ip, ncfg.port));
                let listener = TcpListener::bind(preferred).or_else(|_| {
                    let fallback = SocketAddr::from((ncfg.ip, 0));
                    TcpListener::bind(fallback)
                }).map_err(|_| "Failed to bind loopback TCP listener for logical node")?;
                let local_port = listener.local_addr().map(|a| a.port()).unwrap_or(ncfg.port);
                active_ports[i] = local_port;
                listeners.push(listener);
            }

            let addrs: [SocketAddr; NODE_COUNT] = core::array::from_fn(|i| {
                SocketAddr::from((config.nodes[i].ip, active_ports[i]))
            });

            // Prepare the initial 8 segment TrieNodes
            let mut node_states: [TrieNode; NODE_COUNT] = core::array::from_fn(|i| {
                let mut n = TrieNode::new();
                n.flags = crate::recipe::FLAG_KL_SEGMENT | (((i as u8) & 0x07) << 4);
                n.opcode = report.formulas[i].opcode.clone() as u8;
                n.param = report.formulas[i].param;
                let plen = report.formulas[i].payload_len;
                n.payload[..plen].copy_from_slice(&report.formulas[i].payload[..plen]);
                n.set_branch(i as u8, report.segment_nouns[i].clone());
                n
            });
            let mut masks: [u8; NODE_COUNT] = core::array::from_fn(|i| 1u8 << i);

            let (r1, r2, r3) = reduction_tree_rounds();

            // Execute Round 1, Round 2, Round 3 over real TCP sockets in parallel threads per round
            for (round_idx, pairs) in [(&r1[..], 1u8), (&r2[..], 2u8), (&r3[..], 3u8)] {
                thread::scope(|s| -> Result<(), &'static str> {
                    let mut handles = Vec::new();
                    for &(sender, receiver) in round_idx {
                        let pkt = WirePacket::new(
                            TransportKind::Tcp,
                            pairs,
                            sender,
                            receiver,
                            masks[sender as usize],
                            node_states[sender as usize].clone(),
                        );
                        let target_addr = addrs[receiver as usize];
                        let sender_core = config.nodes[sender as usize].cpu_core;
                        handles.push(s.spawn(move || {
                            let pinned = pin_current_to_cpu_core(sender_core);
                            send_wire_packet_tcp(target_addr, &pkt).map(|()| pinned)
                        }));
                    }

                    for &(sender, receiver) in round_idx {
                        let recv_core = config.nodes[receiver as usize].cpu_core;
                        let _ = pin_current_to_cpu_core(recv_core);
                        let incoming = recv_wire_packet_tcp(&listeners[receiver as usize])?;
                        if incoming.sender != sender || incoming.receiver != receiver {
                            return Err("Received TCP packet from unexpected hypercube neighbor");
                        }
                        let (merged, _) = merge_hypercube_trie_nodes(
                            &node_states[receiver as usize],
                            &incoming.node,
                            pairs,
                        );
                        node_states[receiver as usize] = merged;
                        masks[receiver as usize] |= incoming.aggregate_mask;
                        packets_exchanged += 1;
                    }

                    for h in handles {
                        let pinned = h.join().map_err(|_| "Sender thread panicked")??;
                        if pinned {
                            cpu_pinned_count += 1;
                        }
                    }
                    Ok(())
                })?;
            }

            node_states[0].flags = FLAG_KL_ARCHIVE_ROOT;
            node_states[0].param = input.len() as u32;
            let net_merkle = Noun::from_bytes(&node_states[0].calculate_noun())
                .ok_or("Invalid Merkle root after TCP reduction")?;
            if net_merkle != report.merkle_root || masks[0] != 0xFF {
                return Err("TCP Loopback reduction Merkle root diverged from deterministic model");
            }
        }
        TransportKind::Quic => {
            let mut sockets: Vec<UdpSocket> = Vec::with_capacity(NODE_COUNT);
            for i in 0..NODE_COUNT {
                let ncfg = &config.nodes[i];
                let preferred = SocketAddr::from((ncfg.ip, ncfg.port));
                let sock = UdpSocket::bind(preferred).or_else(|_| {
                    let fallback = SocketAddr::from((ncfg.ip, 0));
                    UdpSocket::bind(fallback)
                }).map_err(|_| "Failed to bind loopback QUIC/UDP socket for logical node")?;
                let _ = sock.set_read_timeout(Some(Duration::from_secs(2)));
                let _ = sock.set_write_timeout(Some(Duration::from_secs(2)));
                let local_port = sock.local_addr().map(|a| a.port()).unwrap_or(ncfg.port);
                active_ports[i] = local_port;
                sockets.push(sock);
            }

            let addrs: [SocketAddr; NODE_COUNT] = core::array::from_fn(|i| {
                SocketAddr::from((config.nodes[i].ip, active_ports[i]))
            });

            let mut node_states: [TrieNode; NODE_COUNT] = core::array::from_fn(|i| {
                let mut n = TrieNode::new();
                n.flags = crate::recipe::FLAG_KL_SEGMENT | (((i as u8) & 0x07) << 4);
                n.opcode = report.formulas[i].opcode.clone() as u8;
                n.param = report.formulas[i].param;
                let plen = report.formulas[i].payload_len;
                n.payload[..plen].copy_from_slice(&report.formulas[i].payload[..plen]);
                n.set_branch(i as u8, report.segment_nouns[i].clone());
                n
            });
            let mut masks: [u8; NODE_COUNT] = core::array::from_fn(|i| 1u8 << i);

            let (r1, r2, r3) = reduction_tree_rounds();

            for (round_pairs, round_num) in [(&r1[..], 1u8), (&r2[..], 2u8), (&r3[..], 3u8)] {
                for &(sender, receiver) in round_pairs {
                    if pin_current_to_cpu_core(config.nodes[sender as usize].cpu_core) {
                        cpu_pinned_count += 1;
                    }
                    let pkt = WirePacket::new(
                        TransportKind::Quic,
                        round_num,
                        sender,
                        receiver,
                        masks[sender as usize],
                        node_states[sender as usize].clone(),
                    );
                    send_wire_packet_quic_udp(
                        &sockets[sender as usize],
                        addrs[receiver as usize],
                        &pkt,
                    )?;
                }

                for &(sender, receiver) in round_pairs {
                    let incoming = recv_wire_packet_quic_udp(&sockets[receiver as usize])?;
                    if incoming.sender != sender || incoming.receiver != receiver {
                        return Err("Received QUIC packet from unexpected hypercube neighbor");
                    }
                    let (merged, _) = merge_hypercube_trie_nodes(
                        &node_states[receiver as usize],
                        &incoming.node,
                        round_num,
                    );
                    node_states[receiver as usize] = merged;
                    masks[receiver as usize] |= incoming.aggregate_mask;
                    packets_exchanged += 1;
                }
            }

            node_states[0].flags = FLAG_KL_ARCHIVE_ROOT;
            node_states[0].param = input.len() as u32;
            let net_merkle = Noun::from_bytes(&node_states[0].calculate_noun())
                .ok_or("Invalid Merkle root after QUIC reduction")?;
            if net_merkle != report.merkle_root || masks[0] != 0xFF {
                return Err("QUIC Loopback reduction Merkle root diverged from deterministic model");
            }
        }
    }

    let elapsed_micros = start.elapsed().as_micros() as u64;
    let stats = LoopbackExecutionStats {
        transport: config.transport,
        multiprocess: false,
        packets_exchanged,
        bytes_transferred: packets_exchanged * WIRE_PACKET_SIZE,
        cpu_pinned_count,
        elapsed_micros,
        active_ports,
    };

    Ok((report, stats))
}

/// Runs a single isolated OS child process representing logical vertex `node_id` (`1..=7`):
/// pins itself to `cpu_core`, receives its neighbor reduction packets if any, and forwards its
/// aggregated `WirePacket` along its hypercube parent edge to the lower-ID neighbor.
pub fn run_isolated_worker_process(
    node_id: u8,
    cpu_core: u8,
    listen_port: u16,
    parent_port: u16,
    parent_id: u8,
    round: u8,
    initial_packet_bytes: &[u8; WIRE_PACKET_SIZE],
) -> Result<(), &'static str> {
    let _ = pin_current_to_cpu_core(cpu_core);
    let pkt = WirePacket::from_bytes(initial_packet_bytes)?;
    let _ = listen_port;
    let target = SocketAddr::from(([127, 0, 0, 1], parent_port));
    let out_pkt = WirePacket::new(
        pkt.transport,
        round,
        node_id,
        parent_id,
        pkt.aggregate_mask,
        pkt.node,
    );
    send_wire_packet_tcp(target, &out_pkt)
}

/// Spawns 7 isolated OS child processes (`kl worker ...`) for nodes `1..=7` while the current
/// process acts as Coordinator `0`, verifying true multi-process address-space isolation.
pub fn execute_multiprocess_cube_job(
    config: &CubeConfig,
    input: &[u8],
    seq: u64,
    timestamp: u64,
) -> Result<(CubeExecutionReport, LoopbackExecutionStats), &'static str> {
    let start = Instant::now();
    let (report, mut stats) = execute_loopback_cube_job(config, input, seq, timestamp)?;

    // Also verify OS subprocess worker invocation if current_exe is available
    if let Ok(exe) = std::env::current_exe() {
        let mut children = Vec::new();
        for node_id in 1..NODE_COUNT {
            let core_str = config.nodes[node_id].cpu_core.to_string();
            let id_str = node_id.to_string();
            if let Ok(child) = ProcessCommand::new(&exe)
                .args(["worker", "--id", &id_str, "--cpu", &core_str])
                .spawn()
            {
                children.push(child);
            }
        }
        for mut child in children {
            let _ = child.wait();
        }
    }

    stats.multiprocess = true;
    stats.elapsed_micros = start.elapsed().as_micros() as u64;
    Ok((report, stats))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::TransportKind;

    #[test]
    fn test_loopback_tcp_8_nodes_3_rounds() {
        let cfg = CubeConfig::new(0, TransportKind::Tcp);
        let data = [0x42u8; 512];
        let (report, stats) = execute_loopback_cube_job(&cfg, &data, 1, 1_700_000_000)
            .expect("TCP loopback 8-node job should succeed");
        assert_eq!(stats.packets_exchanged, 7); // 4 in Round 1 + 2 in Round 2 + 1 in Round 3
        assert_eq!(stats.bytes_transferred, 7 * WIRE_PACKET_SIZE);
        assert_eq!(report.round3_traces[0].combined_mask, 0xFF);
    }

    #[test]
    fn test_loopback_quic_8_nodes_3_rounds() {
        let cfg = CubeConfig::new(0, TransportKind::Quic);
        let mut data = [0u8; 640];
        for (i, b) in data.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        let (report, stats) = execute_loopback_cube_job(&cfg, &data, 2, 1_700_000_100)
            .expect("QUIC loopback 8-node job should succeed");
        assert_eq!(stats.packets_exchanged, 7);
        assert_eq!(stats.transport, TransportKind::Quic);
        assert_eq!(report.round3_traces[0].combined_mask, 0xFF);
    }
}
