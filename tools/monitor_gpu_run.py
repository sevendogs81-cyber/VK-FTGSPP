#!/usr/bin/env python3
"""Poll nvidia-smi (+ optional train_metrics) into CSV / JSONL / text log beside a run.

Example (run in a second tmux pane while training):
  python tools/monitor_gpu_run.py \\
    --run-dir /path/to/run \\
    --interval 5 \\
    --gpus 0,1
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


# nvidia-smi GPU indices (not Vulkan device indices).
SMI_QUERY = [
    "index",
    "uuid",
    "name",
    "pci.bus_id",
    "driver_version",
    "pstate",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "memory.free",
    "temperature.gpu",
    "temperature.memory",
    "power.draw",
    "power.limit",
    "enforced.power.limit",
    "clocks.current.graphics",
    "clocks.current.sm",
    "clocks.current.memory",
    "clocks.current.video",
    "clocks.max.sm",
    "clocks.max.memory",
    "clocks_throttle_reasons.active",
    "pcie.link.gen.current",
    "pcie.link.gen.max",
    "pcie.link.width.current",
    "pcie.link.width.max",
    "fan.speed",
    "compute_mode",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _nvidia_smi_rows() -> list[dict[str, str]]:
    cmd = [
        "nvidia-smi",
        f"--query-gpu={','.join(SMI_QUERY)}",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    rows: list[dict[str, str]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(SMI_QUERY):
            # throttle reason strings can contain commas; recover by joining tail
            if len(parts) > len(SMI_QUERY):
                head = parts[: SMI_QUERY.index("clocks_throttle_reasons.active")]
                # find throttle field then remaining fixed fields after it
                throttle_idx = SMI_QUERY.index("clocks_throttle_reasons.active")
                n_after = len(SMI_QUERY) - throttle_idx - 1
                if n_after >= 0 and len(parts) >= throttle_idx + 1 + n_after:
                    throttle = ",".join(parts[throttle_idx : len(parts) - n_after])
                    tail = parts[len(parts) - n_after :]
                    parts = head + [throttle] + tail
            if len(parts) != len(SMI_QUERY):
                continue
        rows.append(dict(zip(SMI_QUERY, parts)))
    return rows


def _compute_apps(keep: set[int]) -> dict[int, list[dict[str, str]]]:
    """Per-GPU compute processes (pid / name / used memory)."""
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,gpu_bus_id,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError):
        return {}

    # Map bus_id / uuid via a second lightweight query keyed by index
    meta_cmd = [
        "nvidia-smi",
        "--query-gpu=index,uuid,pci.bus_id",
        "--format=csv,noheader,nounits",
    ]
    try:
        meta_out = subprocess.check_output(meta_cmd, text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError):
        return {}

    uuid_to_idx: dict[str, int] = {}
    bus_to_idx: dict[str, int] = {}
    for line in meta_out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        if idx not in keep:
            continue
        uuid_to_idx[parts[1]] = idx
        bus_to_idx[parts[2]] = idx

    by_gpu: dict[int, list[dict[str, str]]] = {i: [] for i in keep}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        uuid, bus, pid, name, mem = parts[0], parts[1], parts[2], parts[3], parts[4]
        idx = uuid_to_idx.get(uuid)
        if idx is None:
            idx = bus_to_idx.get(bus)
        if idx is None or idx not in keep:
            continue
        by_gpu[idx].append(
            {
                "pid": pid,
                "process_name": name,
                "used_gpu_memory_mib": mem,
            }
        )
    return by_gpu


def _host_mem() -> dict[str, float]:
    """Host RAM from /proc/meminfo (MiB)."""
    data: dict[str, float] = {}
    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError:
        return data
    wanted = {
        "MemTotal": "host_mem_total_mib",
        "MemAvailable": "host_mem_available_mib",
        "MemFree": "host_mem_free_mib",
        "Cached": "host_mem_cached_mib",
        "SwapTotal": "host_swap_total_mib",
        "SwapFree": "host_swap_free_mib",
    }
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        if key not in wanted:
            continue
        try:
            kib = float(rest.strip().split()[0])
        except (ValueError, IndexError):
            continue
        data[wanted[key]] = round(kib / 1024.0, 1)
    if "host_mem_total_mib" in data and "host_mem_available_mib" in data:
        data["host_mem_used_mib"] = round(
            data["host_mem_total_mib"] - data["host_mem_available_mib"], 1
        )
    if "host_swap_total_mib" in data and "host_swap_free_mib" in data:
        data["host_swap_used_mib"] = round(
            data["host_swap_total_mib"] - data["host_swap_free_mib"], 1
        )
    return data


def _tail_train_metrics(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    last: dict[str, object] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
    return last


def _pick_metrics(row: dict[str, object] | None) -> dict[str, object]:
    if not row:
        return {}
    keys = (
        "step",
        "iteration",
        "steps_per_second",
        "views_per_second",
        "data_seconds",
        "backward_seconds",
        "gradient_download_seconds",
        "reduction_seconds",
        "gradient_upload_seconds",
        "gradient_host_memcpy_seconds",
        "gradient_exchange_wall_seconds",
        "gradient_async_pipeline",
        "gradient_persistent_host_mapping",
        "optimizer_seconds",
        "world_size",
        "active_world_size",
        "configured_world_size",
        "views_per_step",
        "views_seen",
        "transport",
        "pair_codec_policy",
        "pair_gaussian_codec",
        "pair_sparse",
        "pair_active_sh_degree",
        "pair_active_sh_floats",
        "pair_payload_bytes",
        "pair_peer_payload_bytes",
        "pair_local_visible",
        "pair_peer_visible",
        "collective_algorithm",
        "reduction_tree_abi",
        "sparse_visible_count",
        "global_union_count",
        "gradient_payload_bytes",
        "sparse_input_bytes",
        "sparse_union_bytes",
        "d2h_gpu_copy_seconds",
        "host_reduce_seconds",
        "host_merge_seconds",
        "h2d_gpu_copy_seconds",
        "scatter_seconds",
        "collective_wall_seconds",
        "cpu_memcpy_bytes",
        "cpu_reduction_bytes",
        "pcie_d2h_bytes",
        "pcie_h2d_bytes",
        "parallel_efficiency",
        "total_loss",
        "image_loss",
        "l1",
        "ssim",
        "grad_xyz_rms",
        "grad_scaling_rms",
        "grad_oc_rms",
        "grad_rotation_rms",
    )
    out: dict[str, object] = {}
    for key in keys:
        if key in row:
            out[f"train_{key}"] = row[key]
    return out


def _format_log_line(
    stamp: str,
    elapsed: float,
    samples: list[dict[str, object]],
    train: dict[str, object],
    host: dict[str, float],
) -> str:
    bits = []
    for sample in samples:
        procs = sample.get("compute_apps") or []
        proc_bit = ""
        if isinstance(procs, list) and procs:
            proc_bit = " apps=" + ";".join(
                f"{p.get('process_name', '?')}({p.get('used_gpu_memory_mib', '?')}MiB)"
                for p in procs
            )
        bits.append(
            f"smi{sample['smi_index']}: "
            f"util={sample['util_gpu_pct']}% "
            f"mem={sample['mem_used_mib']}/{sample['mem_total_mib']}MiB "
            f"T={sample['temp_c']}C "
            f"P={sample['power_w']}/{sample['power_limit_w']}W "
            f"sm={sample['clock_sm_mhz']}MHz "
            f"pcie={sample['pcie_gen']}x{sample['pcie_width']} "
            f"pstate={sample['pstate']} "
            f"throttle={sample['throttle_active']}"
            f"{proc_bit}"
        )
    host_bit = ""
    if host:
        host_bit = (
            f" | host_mem={host.get('host_mem_used_mib', '?')}/"
            f"{host.get('host_mem_total_mib', '?')}MiB "
            f"swap={host.get('host_swap_used_mib', '?')}MiB"
        )
    train_bit = ""
    train_step = train.get("train_step", train.get("train_iteration"))
    if train_step not in (None, ""):
        train_bit = (
            f" | step={train_step} "
            f"steps/s={train.get('train_steps_per_second')} "
            f"views/s={train.get('train_views_per_second')} "
            f"bw={train.get('train_backward_seconds')} "
            f"dl={train.get('train_gradient_download_seconds')} "
            f"exch={train.get('train_gradient_exchange_wall_seconds')} "
            f"opt={train.get('train_optimizer_seconds')} "
            f"transport={train.get('train_transport')} "
            f"codec={train.get('train_pair_gaussian_codec')} "
            f"algorithm={train.get('train_collective_algorithm')} "
            f"loss={train.get('train_total_loss')} "
            f"g_xyz={train.get('train_grad_xyz_rms')} "
            f"g_scale={train.get('train_grad_scaling_rms')} "
            f"g_oc={train.get('train_grad_oc_rms')} "
            f"g_rot={train.get('train_grad_rotation_rms')}"
        )
    return (
        f"[{stamp}] elapsed={elapsed:.0f}s || "
        + " || ".join(bits)
        + host_bit
        + train_bit
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Experiment output directory (writes gpu_monitor.* here)",
    )
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between samples")
    parser.add_argument(
        "--gpus",
        default="0,1",
        help="Comma-separated nvidia-smi GPU indices (default: 0,1)",
    )
    parser.add_argument(
        "--no-train-metrics",
        action="store_true",
        help="Do not append latest train_metrics.jsonl fields",
    )
    parser.add_argument(
        "--no-processes",
        action="store_true",
        help="Skip per-GPU compute process listing",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Sample once and exit",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    keep = {int(x) for x in args.gpus.split(",") if x.strip() != ""}
    csv_path = run_dir / "gpu_monitor.csv"
    jsonl_path = run_dir / "gpu_monitor.jsonl"
    log_path = run_dir / "gpu_monitor.log"
    metrics_path = run_dir / "train_metrics.jsonl"

    fieldnames = [
        "timestamp",
        "elapsed_s",
        "smi_index",
        "uuid",
        "name",
        "pci_bus_id",
        "driver_version",
        "pstate",
        "util_gpu_pct",
        "util_mem_pct",
        "mem_used_mib",
        "mem_total_mib",
        "mem_free_mib",
        "temp_c",
        "temp_mem_c",
        "power_w",
        "power_limit_w",
        "enforced_power_limit_w",
        "clock_graphics_mhz",
        "clock_sm_mhz",
        "clock_mem_mhz",
        "clock_video_mhz",
        "clock_sm_max_mhz",
        "clock_mem_max_mhz",
        "throttle_active",
        "pcie_gen",
        "pcie_gen_max",
        "pcie_width",
        "pcie_width_max",
        "fan_pct",
        "compute_mode",
        "compute_apps_json",
        "host_mem_used_mib",
        "host_mem_total_mib",
        "host_mem_available_mib",
        "host_swap_used_mib",
        "train_step",
        "train_iteration",
        "train_steps_per_second",
        "train_views_per_second",
        "train_data_seconds",
        "train_backward_seconds",
        "train_gradient_download_seconds",
        "train_reduction_seconds",
        "train_gradient_upload_seconds",
        "train_gradient_host_memcpy_seconds",
        "train_gradient_exchange_wall_seconds",
        "train_gradient_async_pipeline",
        "train_gradient_persistent_host_mapping",
        "train_optimizer_seconds",
        "train_world_size",
        "train_active_world_size",
        "train_configured_world_size",
        "train_views_per_step",
        "train_views_seen",
        "train_transport",
        "train_pair_codec_policy",
        "train_pair_gaussian_codec",
        "train_pair_sparse",
        "train_pair_active_sh_degree",
        "train_pair_active_sh_floats",
        "train_pair_payload_bytes",
        "train_pair_peer_payload_bytes",
        "train_pair_local_visible",
        "train_pair_peer_visible",
        "train_collective_algorithm",
        "train_reduction_tree_abi",
        "train_sparse_visible_count",
        "train_global_union_count",
        "train_gradient_payload_bytes",
        "train_sparse_input_bytes",
        "train_sparse_union_bytes",
        "train_d2h_gpu_copy_seconds",
        "train_host_reduce_seconds",
        "train_host_merge_seconds",
        "train_h2d_gpu_copy_seconds",
        "train_scatter_seconds",
        "train_collective_wall_seconds",
        "train_cpu_memcpy_bytes",
        "train_cpu_reduction_bytes",
        "train_pcie_d2h_bytes",
        "train_pcie_h2d_bytes",
        "train_parallel_efficiency",
        "train_total_loss",
        "train_image_loss",
        "train_l1",
        "train_ssim",
        "train_grad_xyz_rms",
        "train_grad_scaling_rms",
        "train_grad_oc_rms",
        "train_grad_rotation_rms",
    ]

    new_file = not csv_path.exists()
    csv_file = csv_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
    if new_file:
        writer.writeheader()
        csv_file.flush()

    print(
        f"[monitor] run_dir={run_dir}\n"
        f"[monitor] csv={csv_path}\n"
        f"[monitor] jsonl={jsonl_path}\n"
        f"[monitor] log={log_path}\n"
        f"[monitor] interval={args.interval}s gpus={sorted(keep)}",
        flush=True,
    )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"# gpu_monitor started {_now_iso()} interval={args.interval}s "
            f"gpus={sorted(keep)}\n"
        )

    t0 = time.monotonic()
    try:
        while True:
            stamp = _now_iso()
            elapsed = time.monotonic() - t0
            try:
                smi_rows = _nvidia_smi_rows()
            except (OSError, subprocess.CalledProcessError) as exc:
                print(f"[monitor] nvidia-smi failed: {exc}", file=sys.stderr, flush=True)
                smi_rows = []

            apps = {} if args.no_processes else _compute_apps(keep)
            host = _host_mem()
            train = (
                {}
                if args.no_train_metrics
                else _pick_metrics(_tail_train_metrics(metrics_path))
            )
            samples: list[dict[str, object]] = []
            for raw in smi_rows:
                try:
                    idx = int(raw["index"])
                except ValueError:
                    continue
                if idx not in keep:
                    continue
                gpu_apps = apps.get(idx, [])
                sample: dict[str, object] = {
                    "timestamp": stamp,
                    "elapsed_s": round(elapsed, 1),
                    "smi_index": idx,
                    "uuid": raw["uuid"],
                    "name": raw["name"],
                    "pci_bus_id": raw["pci.bus_id"],
                    "driver_version": raw["driver_version"],
                    "pstate": raw["pstate"],
                    "util_gpu_pct": raw["utilization.gpu"],
                    "util_mem_pct": raw["utilization.memory"],
                    "mem_used_mib": raw["memory.used"],
                    "mem_total_mib": raw["memory.total"],
                    "mem_free_mib": raw["memory.free"],
                    "temp_c": raw["temperature.gpu"],
                    "temp_mem_c": raw["temperature.memory"],
                    "power_w": raw["power.draw"],
                    "power_limit_w": raw["power.limit"],
                    "enforced_power_limit_w": raw["enforced.power.limit"],
                    "clock_graphics_mhz": raw["clocks.current.graphics"],
                    "clock_sm_mhz": raw["clocks.current.sm"],
                    "clock_mem_mhz": raw["clocks.current.memory"],
                    "clock_video_mhz": raw["clocks.current.video"],
                    "clock_sm_max_mhz": raw["clocks.max.sm"],
                    "clock_mem_max_mhz": raw["clocks.max.memory"],
                    "throttle_active": raw["clocks_throttle_reasons.active"],
                    "pcie_gen": raw["pcie.link.gen.current"],
                    "pcie_gen_max": raw["pcie.link.gen.max"],
                    "pcie_width": raw["pcie.link.width.current"],
                    "pcie_width_max": raw["pcie.link.width.max"],
                    "fan_pct": raw["fan.speed"],
                    "compute_mode": raw["compute_mode"],
                    "compute_apps": gpu_apps,
                    "compute_apps_json": json.dumps(gpu_apps, ensure_ascii=False),
                    "host_mem_used_mib": host.get("host_mem_used_mib", ""),
                    "host_mem_total_mib": host.get("host_mem_total_mib", ""),
                    "host_mem_available_mib": host.get("host_mem_available_mib", ""),
                    "host_swap_used_mib": host.get("host_swap_used_mib", ""),
                    "train_iteration": train.get("train_iteration", ""),
                    "train_steps_per_second": train.get("train_steps_per_second", ""),
                    "train_views_per_second": train.get("train_views_per_second", ""),
                    "train_data_seconds": train.get("train_data_seconds", ""),
                    "train_backward_seconds": train.get("train_backward_seconds", ""),
                    "train_gradient_download_seconds": train.get(
                        "train_gradient_download_seconds", ""
                    ),
                    "train_reduction_seconds": train.get("train_reduction_seconds", ""),
                    "train_gradient_upload_seconds": train.get(
                        "train_gradient_upload_seconds", ""
                    ),
                    "train_gradient_host_memcpy_seconds": train.get(
                        "train_gradient_host_memcpy_seconds", ""
                    ),
                    "train_gradient_exchange_wall_seconds": train.get(
                        "train_gradient_exchange_wall_seconds", ""
                    ),
                    "train_gradient_async_pipeline": train.get(
                        "train_gradient_async_pipeline", ""
                    ),
                    "train_gradient_persistent_host_mapping": train.get(
                        "train_gradient_persistent_host_mapping", ""
                    ),
                    "train_optimizer_seconds": train.get("train_optimizer_seconds", ""),
                    "train_world_size": train.get("train_world_size", ""),
                    "train_active_world_size": train.get(
                        "train_active_world_size", ""
                    ),
                    "train_configured_world_size": train.get(
                        "train_configured_world_size", ""
                    ),
                    "train_views_per_step": train.get("train_views_per_step", ""),
                    "train_views_seen": train.get("train_views_seen", ""),
                    "train_transport": train.get("train_transport", ""),
                    "train_collective_algorithm": train.get(
                        "train_collective_algorithm", ""
                    ),
                    "train_reduction_tree_abi": train.get(
                        "train_reduction_tree_abi", ""
                    ),
                    "train_sparse_visible_count": train.get(
                        "train_sparse_visible_count", ""
                    ),
                    "train_global_union_count": train.get(
                        "train_global_union_count", ""
                    ),
                    "train_gradient_payload_bytes": train.get(
                        "train_gradient_payload_bytes", ""
                    ),
                    "train_sparse_input_bytes": train.get(
                        "train_sparse_input_bytes", ""
                    ),
                    "train_sparse_union_bytes": train.get(
                        "train_sparse_union_bytes", ""
                    ),
                    "train_d2h_gpu_copy_seconds": train.get(
                        "train_d2h_gpu_copy_seconds", ""
                    ),
                    "train_host_reduce_seconds": train.get(
                        "train_host_reduce_seconds", ""
                    ),
                    "train_host_merge_seconds": train.get(
                        "train_host_merge_seconds", ""
                    ),
                    "train_h2d_gpu_copy_seconds": train.get(
                        "train_h2d_gpu_copy_seconds", ""
                    ),
                    "train_scatter_seconds": train.get(
                        "train_scatter_seconds", ""
                    ),
                    "train_collective_wall_seconds": train.get(
                        "train_collective_wall_seconds", ""
                    ),
                    "train_cpu_memcpy_bytes": train.get(
                        "train_cpu_memcpy_bytes", ""
                    ),
                    "train_cpu_reduction_bytes": train.get(
                        "train_cpu_reduction_bytes", ""
                    ),
                    "train_pcie_d2h_bytes": train.get(
                        "train_pcie_d2h_bytes", ""
                    ),
                    "train_pcie_h2d_bytes": train.get(
                        "train_pcie_h2d_bytes", ""
                    ),
                    "train_parallel_efficiency": train.get(
                        "train_parallel_efficiency", ""
                    ),
                    "train_total_loss": train.get("train_total_loss", ""),
                    "train_image_loss": train.get("train_image_loss", ""),
                    "train_l1": train.get("train_l1", ""),
                    "train_ssim": train.get("train_ssim", ""),
                    "train_grad_xyz_rms": train.get("train_grad_xyz_rms", ""),
                    "train_grad_scaling_rms": train.get(
                        "train_grad_scaling_rms", ""
                    ),
                    "train_grad_oc_rms": train.get("train_grad_oc_rms", ""),
                    "train_grad_rotation_rms": train.get(
                        "train_grad_rotation_rms", ""
                    ),
                }
                samples.append(sample)
                writer.writerow(sample)

            csv_file.flush()
            with jsonl_path.open("a", encoding="utf-8") as handle:
                for sample in samples:
                    handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

            line = _format_log_line(stamp, elapsed, samples, train, host)
            print(line, flush=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

            if args.once:
                break
            time.sleep(max(0.5, float(args.interval)))
    except KeyboardInterrupt:
        print("\n[monitor] stopped", flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"# gpu_monitor stopped {_now_iso()}\n")
    finally:
        csv_file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
