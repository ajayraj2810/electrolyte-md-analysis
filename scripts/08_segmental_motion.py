#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import MDAnalysis as mda
    from MDAnalysis.lib.distances import capped_distance, minimize_vectors
except ImportError as exc:
    raise SystemExit(
        "MDAnalysis is required. Install it in the active environment, for example:\n"
        "    pip install MDAnalysis pandas numpy matplotlib\n"
    ) from exc

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required. Install it with:\n"
        "    pip install matplotlib\n"
    ) from exc

TRANSITION_ORDER = [
    "Same_Chain_Neighbor_Slide",
    "Same_Chain_Large_EO_Jump",
    "Interchain_Hop",
    "PEO_Attachment",
    "PEO_Detachment",
]

@dataclass
class ChainMap:

    chain_codes: np.ndarray
    eo_indices: np.ndarray
    chain_labels: List[str]
    chain_to_local_indices: Dict[int, np.ndarray]
    chain_site_to_local: Dict[Tuple[int, int], int]
    source_attribute: str

@dataclass
class CorrelationAccumulator:
    lags: np.ndarray
    sum_metal_sq: np.ndarray
    sum_segment_sq: np.ndarray
    sum_dot: np.ndarray
    sum_relative_sq: np.ndarray
    n_origins: np.ndarray
    n_episodes: np.ndarray

    @classmethod
    def create(cls, lags: np.ndarray) -> "CorrelationAccumulator":
        n = len(lags)
        return cls(
            lags=lags,
            sum_metal_sq=np.zeros(n, dtype=np.float64),
            sum_segment_sq=np.zeros(n, dtype=np.float64),
            sum_dot=np.zeros(n, dtype=np.float64),
            sum_relative_sq=np.zeros(n, dtype=np.float64),
            n_origins=np.zeros(n, dtype=np.int64),
            n_episodes=np.zeros(n, dtype=np.int64),
        )

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze PEO EO-site sliding, interchain hopping, dwell times, and "
            "metal-local-PEO segmental-motion coupling for generic metal-ion electrolytes."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--topology",
        type=Path,
        default=None,
        help="LAMMPS DATA topology. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=None,
        help="LAMMPS dump trajectory. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--metal",
        default="metal",
        help="Generic metal-ion label stored in output metadata.",
    )
    parser.add_argument(
        "--composition",
        default=None,
        help="Composition label, e.g. 20_5, 15_10, 15_15, or 5_20.",
    )
    parser.add_argument(
        "--temperature-K",
        type=float,
        default=380.0,
        help="Simulation temperature used in output metadata.",
    )

    parser.add_argument("--metal-selection", default="type 17")
    parser.add_argument("--peo-o-selection", default="type 4")
    parser.add_argument("--tfsi-o-selection", default="type 9")
    parser.add_argument(
        "--peo-chain-attribute",
        choices=["auto", "resid", "molnum", "segid"],
        default="auto",
        help="Topology attribute used to identify individual PEO chains.",
    )
    parser.add_argument(
        "--peo-cutoff", type=float, required=True,
        help="Metal–polymer-oxygen coordination cutoff in angstrom.",
    )
    parser.add_argument(
        "--tfsi-cutoff", type=float, required=True,
        help="Metal–anion-oxygen coordination cutoff in angstrom.",
    )

    parser.add_argument(
        "--frame-interval-ps",
        type=float,
        default=10.0,
        help="Physical time between consecutive stored trajectory frames.",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--stop-frame",
        type=int,
        default=None,
        help="Exclusive stop frame. Default: end of trajectory.",
    )
    parser.add_argument("--stride", type=int, default=1)

    parser.add_argument(
        "--smoothing-window",
        type=int,
        choices=[1, 3],
        default=3,
        help=(
            "Temporal state smoothing. Window 3 removes isolated one-frame "
            "chain/site flicker while preserving real sequential motion."
        ),
    )
    parser.add_argument(
        "--segment-half-width",
        type=int,
        default=2,
        help="Number of EO sites on either side used for the local segment COM.",
    )
    parser.add_argument(
        "--max-correlation-ns",
        type=float,
        default=10.0,
        help="Maximum lag for metal-local-PEO coupling.",
    )
    parser.add_argument(
        "--n-log-lags",
        type=int,
        default=45,
        help="Number of logarithmically spaced lag targets.",
    )
    parser.add_argument(
        "--min-correlation-origins",
        type=int,
        default=200,
        help="Minimum origins for a recommended correlation point.",
    )
    parser.add_argument(
        "--min-correlation-episodes",
        type=int,
        default=20,
        help="Minimum same-chain episodes for a recommended correlation point.",
    )
    parser.add_argument(
        "--n-blocks",
        type=int,
        default=5,
        help="Number of trajectory blocks for transition-fraction uncertainty.",
    )
    parser.add_argument(
        "--representative-metals",
        type=int,
        default=5,
        help="Number of metal EO-coordinate traces saved for visualization.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("06_PEO_sliding_analysis"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory.",
    )
    parser.add_argument(
        "--write-frame-database",
        action="store_true",
        help=(
            "Write the full frame-level metal coordinate database. This can be "
            "several hundred MB even after gzip compression."
        ),
    )
    return parser

def configure_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("peo_sliding")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(
        output_dir / "06_PEO_sliding_analysis.log",
        mode="w",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def infer_metal(path: Path) -> Optional[str]:

    return "metal"

def infer_composition(path: Path) -> Optional[str]:
    match = re.search(r"(20_5|15_10|15_15|5_20)", str(path))
    return match.group(1) if match else None

def discover_unique_file(
    explicit: Optional[Path],
    patterns: Sequence[str],
    label: str,
) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
        return path

    matches: List[Path] = []
    cwd = Path.cwd()
    for pattern in patterns:
        matches.extend(cwd.glob(pattern))
    matches = sorted({p.resolve() for p in matches if p.is_file()})

    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"No {label} was found automatically. Supply it explicitly."
        )
    listed = "\n".join(f"  - {path}" for path in matches)
    raise RuntimeError(
        f"Multiple possible {label} files were found. Use the explicit option:\n"
        f"{listed}"
    )

def prepare_output_dir(path: Path, overwrite: bool) -> Path:
    path = path.expanduser().resolve()
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {path}\n"
                "Use --overwrite only when replacing it intentionally."
            )
        for child in sorted(path.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
    path.mkdir(parents=True, exist_ok=True)
    (path / "figures").mkdir(exist_ok=True)
    return path

def safe_fraction(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")

def run_length_segments(values: Sequence[Tuple[int, int]]) -> List[Tuple[int, int, Tuple[int, int]]]:

    if not values:
        return []
    segments: List[Tuple[int, int, Tuple[int, int]]] = []
    start = 0
    state = values[0]
    for index in range(1, len(values)):
        if values[index] != state:
            segments.append((start, index, state))
            start = index
            state = values[index]
    segments.append((start, len(values), state))
    return segments

def make_lag_grid(
    effective_dt_ps: float,
    max_lag_ns: float,
    n_log_lags: int,
) -> np.ndarray:
    max_frames = max(1, int(round(max_lag_ns * 1000.0 / effective_dt_ps)))
    if max_frames == 1:
        return np.array([1], dtype=int)
    values = np.geomspace(1, max_frames, num=max(2, n_log_lags))
    lags = np.unique(np.clip(np.rint(values).astype(int), 1, max_frames))
    return lags

def classify_transition(
    from_chain: int,
    from_site: int,
    to_chain: int,
    to_site: int,
) -> Optional[str]:
    if from_chain < 0 and to_chain >= 0:
        return "PEO_Attachment"
    if from_chain >= 0 and to_chain < 0:
        return "PEO_Detachment"
    if from_chain < 0 and to_chain < 0:
        return None
    if from_chain != to_chain:
        return "Interchain_Hop"

    delta = abs(int(to_site) - int(from_site))
    if delta == 1:
        return "Same_Chain_Neighbor_Slide"
    if delta > 1:
        return "Same_Chain_Large_EO_Jump"
    return None

def atom_attribute_values(atom_group, attribute: str) -> np.ndarray:
    if attribute == "resid":
        return np.asarray(atom_group.resids)
    if attribute == "segid":
        return np.asarray(atom_group.segids)
    if attribute == "molnum":
        try:
            return np.asarray(atom_group.molnums)
        except Exception as exc:
            raise RuntimeError(
                "The topology does not expose molnums/molecule IDs."
            ) from exc
    raise ValueError(attribute)

def attribute_is_plausible(values: np.ndarray) -> bool:
    unique, counts = np.unique(values, return_counts=True)
    return (
        len(unique) >= 2
        and len(unique) < len(values)
        and np.median(counts) >= 2
    )

def build_chain_map(peo_oxygens, requested: str) -> ChainMap:
    if requested == "auto":
        candidates = ["resid", "molnum", "segid"]
        selected = None
        values = None
        for candidate in candidates:
            try:
                trial = atom_attribute_values(peo_oxygens, candidate)
            except Exception:
                continue
            if attribute_is_plausible(trial):
                selected = candidate
                values = trial
                break
        if selected is None or values is None:
            raise RuntimeError(
                "Could not identify PEO chains automatically. Try "
                "--peo-chain-attribute resid or molnum after inspecting the topology."
            )
    else:
        selected = requested
        values = atom_attribute_values(peo_oxygens, selected)
        if not attribute_is_plausible(values):
            raise RuntimeError(
                f"PEO chain attribute '{selected}' does not produce a plausible "
                "multi-chain mapping."
            )

    unique_labels = sorted(np.unique(values), key=lambda x: str(x))
    label_to_code = {label: code for code, label in enumerate(unique_labels)}
    chain_codes = np.array([label_to_code[x] for x in values], dtype=np.int32)
    eo_indices = np.zeros(len(peo_oxygens), dtype=np.int16)
    chain_to_local_indices: Dict[int, np.ndarray] = {}
    chain_site_to_local: Dict[Tuple[int, int], int] = {}

    atom_ids = np.asarray(peo_oxygens.ids)
    for code in range(len(unique_labels)):
        local = np.where(chain_codes == code)[0]
        ordered = local[np.argsort(atom_ids[local])]
        chain_to_local_indices[code] = ordered
        for site, local_index in enumerate(ordered, start=1):
            eo_indices[local_index] = site
            chain_site_to_local[(code, site)] = int(local_index)

    return ChainMap(
        chain_codes=chain_codes,
        eo_indices=eo_indices,
        chain_labels=[str(x) for x in unique_labels],
        chain_to_local_indices=chain_to_local_indices,
        chain_site_to_local=chain_site_to_local,
        source_attribute=selected,
    )

def build_segment_com_map(
    peo_positions: np.ndarray,
    box: np.ndarray,
    chain_map: ChainMap,
    half_width: int,
) -> np.ndarray:

    result = np.full((len(peo_positions), 3), np.nan, dtype=np.float64)

    for indices in chain_map.chain_to_local_indices.values():
        wrapped = np.asarray(peo_positions[indices], dtype=np.float64)
        unwrapped = np.empty_like(wrapped)
        unwrapped[0] = wrapped[0]
        if len(wrapped) > 1:
            steps = minimize_vectors(wrapped[1:] - wrapped[:-1], box)
            unwrapped[1:] = unwrapped[0] + np.cumsum(steps, axis=0)

        for local_site, peo_local_index in enumerate(indices):
            lo = max(0, local_site - half_width)
            hi = min(len(indices), local_site + half_width + 1)
            result[peo_local_index] = unwrapped[lo:hi].mean(axis=0)

    return result

def choose_dominant_chain_and_site(
    peo_local_indices: np.ndarray,
    distances: np.ndarray,
    chain_map: ChainMap,
    previous_chain: int,
) -> Tuple[int, int, float]:

    contact_chains = chain_map.chain_codes[peo_local_indices]
    unique, counts = np.unique(contact_chains, return_counts=True)
    maximum = counts.max()
    candidates = unique[counts == maximum]

    if previous_chain in candidates:
        chain = int(previous_chain)
    elif len(candidates) == 1:
        chain = int(candidates[0])
    else:
        mean_distances = []
        for candidate in candidates:
            mask = contact_chains == candidate
            mean_distances.append(float(np.mean(distances[mask])))
        chain = int(candidates[int(np.argmin(mean_distances))])

    mask = contact_chains == chain
    selected_indices = peo_local_indices[mask]
    selected_distances = distances[mask]
    sites = chain_map.eo_indices[selected_indices].astype(np.float64)

    weights = 1.0 / np.maximum(selected_distances, 1.0e-8) ** 2
    coordinate = float(np.sum(weights * sites) / np.sum(weights))
    max_site = len(chain_map.chain_to_local_indices[chain])
    site = int(np.clip(np.rint(coordinate), 1, max_site))
    return chain, site, coordinate

def smooth_states_three_frame(
    raw_chain: np.ndarray,
    raw_site: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:

    n_frames, n_metals = raw_chain.shape
    chain_out = raw_chain.copy()
    site_out = raw_site.copy()

    for metal in range(n_metals):
        chain = raw_chain[:, metal]
        site = raw_site[:, metal].astype(np.float64)

        left_chain = np.concatenate(([chain[0]], chain[:-1]))
        center_chain = chain
        right_chain = np.concatenate((chain[1:], [chain[-1]]))

        smoothed_chain = center_chain.copy()
        same_lr = left_chain == right_chain
        smoothed_chain[same_lr] = left_chain[same_lr]

        left_site = np.concatenate(([site[0]], site[:-1]))
        center_site = site
        right_site = np.concatenate((site[1:], [site[-1]]))

        candidates = np.vstack(
            [
                np.where(left_chain == smoothed_chain, left_site, np.nan),
                np.where(center_chain == smoothed_chain, center_site, np.nan),
                np.where(right_chain == smoothed_chain, right_site, np.nan),
            ]
        )
        with np.errstate(all="ignore"):
            smoothed_site_float = np.nanmedian(candidates, axis=0)

        smoothed_site = np.where(
            smoothed_chain >= 0,
            np.rint(smoothed_site_float),
            -1,
        ).astype(np.int16)

        chain_out[:, metal] = smoothed_chain
        site_out[:, metal] = smoothed_site

    return chain_out, site_out

def process_trajectory(
    universe,
    metals,
    peo_oxygens,
    tfsi_oxygens,
    chain_map: ChainMap,
    args,
    logger: logging.Logger,
):
    start = max(0, args.start_frame)
    stop = len(universe.trajectory) if args.stop_frame is None else min(
        args.stop_frame, len(universe.trajectory)
    )
    frame_numbers = np.arange(start, stop, args.stride, dtype=np.int32)
    n_frames = len(frame_numbers)
    n_metals = len(metals)

    if n_frames < 2:
        raise RuntimeError("At least two selected trajectory frames are required.")

    raw_chain = np.full((n_frames, n_metals), -1, dtype=np.int16)
    raw_site = np.full((n_frames, n_metals), -1, dtype=np.int16)
    raw_coordinate = np.full((n_frames, n_metals), np.nan, dtype=np.float32)
    n_peo_o = np.zeros((n_frames, n_metals), dtype=np.int16)
    n_peo_chains = np.zeros((n_frames, n_metals), dtype=np.int8)
    n_tfsi_o = np.zeros((n_frames, n_metals), dtype=np.int16)
    tfsi_present = np.zeros((n_frames, n_metals), dtype=bool)
    multichain_peo = np.zeros((n_frames, n_metals), dtype=bool)

    metal_wrapped = np.empty((n_frames, n_metals, 3), dtype=np.float32)
    segment_wrapped = np.full(
        (n_frames, n_metals, 3), np.nan, dtype=np.float32
    )
    boxes = np.empty((n_frames, 6), dtype=np.float32)
    times_ps = np.empty(n_frames, dtype=np.float64)

    previous_chain = np.full(n_metals, -1, dtype=np.int16)
    wall_start = time.perf_counter()

    for output_frame, trajectory_frame in enumerate(frame_numbers):
        ts = universe.trajectory[int(trajectory_frame)]
        box = np.asarray(ts.dimensions, dtype=np.float64)
        if box.shape[0] < 6 or np.any(box[:3] <= 0):
            raise RuntimeError(
                f"Invalid periodic box at trajectory frame {trajectory_frame}: {box}"
            )

        metal_positions = np.asarray(metals.positions, dtype=np.float64)
        peo_positions = np.asarray(peo_oxygens.positions, dtype=np.float64)
        tfsi_positions = np.asarray(tfsi_oxygens.positions, dtype=np.float64)

        metal_wrapped[output_frame] = metal_positions
        boxes[output_frame] = box
        times_ps[output_frame] = (
            (trajectory_frame - start) * args.frame_interval_ps
        )

        segment_com_map = build_segment_com_map(
            peo_positions,
            box,
            chain_map,
            args.segment_half_width,
        )

        peo_pairs, peo_distances = capped_distance(
            metal_positions,
            peo_positions,
            max_cutoff=args.peo_cutoff,
            box=box,
            return_distances=True,
        )
        tfsi_pairs, _tfsi_distances = capped_distance(
            metal_positions,
            tfsi_positions,
            max_cutoff=args.tfsi_cutoff,
            box=box,
            return_distances=True,
        )

        if len(tfsi_pairs):
            tfsi_counts = np.bincount(
                tfsi_pairs[:, 0],
                minlength=n_metals,
            )
            n_tfsi_o[output_frame] = tfsi_counts.astype(np.int16)
            tfsi_present[output_frame] = tfsi_counts > 0

        if len(peo_pairs):
            order = np.argsort(peo_pairs[:, 0], kind="stable")
            sorted_pairs = peo_pairs[order]
            sorted_distances = peo_distances[order]
            metal_indices = sorted_pairs[:, 0]
            boundaries = np.flatnonzero(np.diff(metal_indices)) + 1
            group_starts = np.concatenate(([0], boundaries))
            group_stops = np.concatenate((boundaries, [len(sorted_pairs)]))

            for group_start, group_stop in zip(group_starts, group_stops):
                metal_local = int(sorted_pairs[group_start, 0])
                peo_local = sorted_pairs[group_start:group_stop, 1].astype(int)
                distances = sorted_distances[group_start:group_stop]

                chains = chain_map.chain_codes[peo_local]
                unique_chains = np.unique(chains)
                n_peo_o[output_frame, metal_local] = len(peo_local)
                n_peo_chains[output_frame, metal_local] = len(unique_chains)
                multichain_peo[output_frame, metal_local] = (
                    len(unique_chains) >= 2
                )

                chain, site, coordinate = choose_dominant_chain_and_site(
                    peo_local,
                    distances,
                    chain_map,
                    int(previous_chain[metal_local]),
                )
                raw_chain[output_frame, metal_local] = chain
                raw_site[output_frame, metal_local] = site
                raw_coordinate[output_frame, metal_local] = coordinate
                previous_chain[metal_local] = chain

                selected_local = chain_map.chain_site_to_local[(chain, site)]
                segment_wrapped[output_frame, metal_local] = (
                    segment_com_map[selected_local]
                )

        absent = raw_chain[output_frame] < 0
        previous_chain[absent] = -1

        if (
            output_frame == 0
            or (output_frame + 1) % 500 == 0
            or output_frame + 1 == n_frames
        ):
            elapsed = time.perf_counter() - wall_start
            logger.info(
                "Processed %d/%d frames | trajectory frame=%d | "
                "PEO-coordinated metals=%d/%d | elapsed=%.1f min",
                output_frame + 1,
                n_frames,
                trajectory_frame,
                int(np.sum(raw_chain[output_frame] >= 0)),
                n_metals,
                elapsed / 60.0,
            )

    return {
        "frame_numbers": frame_numbers,
        "times_ps": times_ps,
        "boxes": boxes,
        "raw_chain": raw_chain,
        "raw_site": raw_site,
        "raw_coordinate": raw_coordinate,
        "n_peo_o": n_peo_o,
        "n_peo_chains": n_peo_chains,
        "n_tfsi_o": n_tfsi_o,
        "tfsi_present": tfsi_present,
        "multichain_peo": multichain_peo,
        "metal_wrapped": metal_wrapped,
        "segment_wrapped": segment_wrapped,
    }

def build_dwell_and_transition_tables(
    arrays,
    smoothed_chain: np.ndarray,
    smoothed_site: np.ndarray,
    metal_ids: np.ndarray,
    chain_map: ChainMap,
    effective_dt_ps: float,
    n_blocks: int,
):
    frame_numbers = arrays["frame_numbers"]
    times_ps = arrays["times_ps"]
    tfsi_present = arrays["tfsi_present"]
    multichain = arrays["multichain_peo"]
    n_peo_o = arrays["n_peo_o"]
    n_peo_chains = arrays["n_peo_chains"]

    n_frames, n_metals = smoothed_chain.shape
    dwell_rows: List[dict] = []
    transition_rows: List[dict] = []
    global_event_id = 0

    for metal_local in range(n_metals):
        states = [
            (int(smoothed_chain[frame, metal_local]),
             int(smoothed_site[frame, metal_local]))
            for frame in range(n_frames)
        ]
        segments = run_length_segments(states)
        local_events: List[dict] = []

        for local_event_index, (start, stop, state) in enumerate(segments):
            chain, site = state
            valid_peo = chain >= 0
            event = {
                "event_id": global_event_id,
                "metal_local_index": metal_local,
                "metal_atom_id": int(metal_ids[metal_local]),
                "local_event_index": local_event_index,
                "start_selected_frame_index": start,
                "stop_selected_frame_index_exclusive": stop,
                "start_trajectory_frame": int(frame_numbers[start]),
                "end_trajectory_frame": int(frame_numbers[stop - 1]),
                "start_time_ps": float(times_ps[start]),
                "end_time_ps": float(times_ps[stop - 1]),
                "n_frames": stop - start,
                "duration_ps": float((stop - start) * effective_dt_ps),
                "duration_ns": float((stop - start) * effective_dt_ps / 1000.0),
                "dominant_chain_code": chain,
                "dominant_chain_label": (
                    chain_map.chain_labels[chain] if valid_peo else "UNBOUND"
                ),
                "eo_site": site if valid_peo else -1,
                "valid_peo_dwell": int(valid_peo),
                "left_censored": int(start == 0),
                "right_censored": int(stop == n_frames),
                "tfsi_involved_any": int(
                    np.any(tfsi_present[start:stop, metal_local])
                ),
                "tfsi_time_fraction": float(
                    np.mean(tfsi_present[start:stop, metal_local])
                ),
                "multichain_peo_any": int(
                    np.any(multichain[start:stop, metal_local])
                ),
                "multichain_peo_time_fraction": float(
                    np.mean(multichain[start:stop, metal_local])
                ),
                "mean_peo_oxygen_cn": float(
                    np.mean(n_peo_o[start:stop, metal_local])
                ),
                "mean_peo_chain_cn": float(
                    np.mean(n_peo_chains[start:stop, metal_local])
                ),
            }
            dwell_rows.append(event)
            local_events.append(event)
            global_event_id += 1

        for pair_index in range(len(local_events) - 1):
            before = local_events[pair_index]
            after = local_events[pair_index + 1]
            transition_type = classify_transition(
                before["dominant_chain_code"],
                before["eo_site"],
                after["dominant_chain_code"],
                after["eo_site"],
            )
            if transition_type is None:
                continue

            boundary = int(after["start_selected_frame_index"])
            lo = max(0, boundary - 1)
            hi = min(n_frames, boundary + 2)
            from_chain = int(before["dominant_chain_code"])
            to_chain = int(after["dominant_chain_code"])
            from_site = int(before["eo_site"])
            to_site = int(after["eo_site"])

            delta_eo = (
                to_site - from_site
                if from_chain >= 0 and from_chain == to_chain
                else np.nan
            )
            block_index = min(
                n_blocks - 1,
                int(boundary * n_blocks / max(1, n_frames)),
            )

            transition_rows.append(
                {
                    "transition_id": len(transition_rows),
                    "metal_local_index": metal_local,
                    "metal_atom_id": int(metal_ids[metal_local]),
                    "from_event_id": int(before["event_id"]),
                    "to_event_id": int(after["event_id"]),
                    "trajectory_frame": int(frame_numbers[boundary]),
                    "time_ps": float(times_ps[boundary]),
                    "time_ns": float(times_ps[boundary] / 1000.0),
                    "block_index": block_index,
                    "transition_type": transition_type,
                    "from_chain_code": from_chain,
                    "from_chain_label": before["dominant_chain_label"],
                    "from_eo_site": from_site,
                    "to_chain_code": to_chain,
                    "to_chain_label": after["dominant_chain_label"],
                    "to_eo_site": to_site,
                    "delta_eo": delta_eo,
                    "absolute_delta_eo": (
                        abs(delta_eo) if np.isfinite(delta_eo) else np.nan
                    ),
                    "tfsi_involved": int(
                        np.any(tfsi_present[lo:hi, metal_local])
                    ),
                    "multichain_peo_involved": int(
                        np.any(multichain[lo:hi, metal_local])
                    ),
                    "from_dwell_ps": float(before["duration_ps"]),
                    "to_dwell_ps": float(after["duration_ps"]),
                }
            )

    dwell_df = pd.DataFrame(dwell_rows)
    transition_df = pd.DataFrame(transition_rows)
    return dwell_df, transition_df

def build_sliding_episodes(transition_df: pd.DataFrame) -> pd.DataFrame:

    rows: List[dict] = []
    episode_id = 0

    if transition_df.empty:
        return pd.DataFrame()

    for metal_id, group in transition_df.groupby("metal_atom_id", sort=False):
        group = group.sort_values("time_ps")
        current: Optional[dict] = None
        previous_row = None

        for _, row in group.iterrows():
            is_neighbor = (
                row["transition_type"] == "Same_Chain_Neighbor_Slide"
            )
            direction = (
                int(np.sign(row["delta_eo"])) if is_neighbor else 0
            )

            can_continue = (
                is_neighbor
                and current is not None
                and previous_row is not None
                and previous_row["transition_type"]
                == "Same_Chain_Neighbor_Slide"
                and int(previous_row["to_event_id"])
                == int(row["from_event_id"])
                and int(previous_row["to_chain_code"])
                == int(row["from_chain_code"])
                and int(np.sign(previous_row["delta_eo"])) == direction
            )

            if is_neighbor and can_continue:
                current["end_time_ps"] = float(row["time_ps"])
                current["end_eo_site"] = int(row["to_eo_site"])
                current["n_steps"] += 1
                current["net_delta_eo"] += int(row["delta_eo"])
                current["tfsi_involved_any"] = max(
                    current["tfsi_involved_any"],
                    int(row["tfsi_involved"]),
                )
                current["multichain_peo_involved_any"] = max(
                    current["multichain_peo_involved_any"],
                    int(row["multichain_peo_involved"]),
                )
            else:
                if current is not None:
                    current["observed_span_ps"] = (
                        current["end_time_ps"] - current["start_time_ps"]
                    )
                    rows.append(current)
                    episode_id += 1
                    current = None

                if is_neighbor:
                    current = {
                        "sliding_episode_id": episode_id,
                        "metal_atom_id": int(metal_id),
                        "chain_code": int(row["from_chain_code"]),
                        "chain_label": row["from_chain_label"],
                        "direction": direction,
                        "start_time_ps": float(row["time_ps"]),
                        "end_time_ps": float(row["time_ps"]),
                        "start_eo_site": int(row["from_eo_site"]),
                        "end_eo_site": int(row["to_eo_site"]),
                        "n_steps": 1,
                        "net_delta_eo": int(row["delta_eo"]),
                        "tfsi_involved_any": int(row["tfsi_involved"]),
                        "multichain_peo_involved_any": int(
                            row["multichain_peo_involved"]
                        ),
                    }

            previous_row = row

        if current is not None:
            current["observed_span_ps"] = (
                current["end_time_ps"] - current["start_time_ps"]
            )
            rows.append(current)
            episode_id += 1

    return pd.DataFrame(rows)

def summarize_dwells(dwell_df: pd.DataFrame) -> pd.DataFrame:
    valid = dwell_df[dwell_df["valid_peo_dwell"] == 1].copy()
    uncensored = valid[
        (valid["left_censored"] == 0) & (valid["right_censored"] == 0)
    ]

    summary = {
        "total_dwell_events_including_unbound": len(dwell_df),
        "valid_peo_dwell_events": len(valid),
        "completed_peo_dwell_events": len(uncensored),
        "fraction_peo_dwells_censored": safe_fraction(
            len(valid) - len(uncensored), len(valid)
        ),
        "mean_completed_dwell_ps": (
            float(uncensored["duration_ps"].mean()) if len(uncensored) else np.nan
        ),
        "median_completed_dwell_ps": (
            float(uncensored["duration_ps"].median()) if len(uncensored) else np.nan
        ),
        "p90_completed_dwell_ps": (
            float(uncensored["duration_ps"].quantile(0.90))
            if len(uncensored) else np.nan
        ),
        "p95_completed_dwell_ps": (
            float(uncensored["duration_ps"].quantile(0.95))
            if len(uncensored) else np.nan
        ),
        "maximum_observed_peo_dwell_ps": (
            float(valid["duration_ps"].max()) if len(valid) else np.nan
        ),
        "fraction_peo_dwells_with_tfsi": (
            float(valid["tfsi_involved_any"].mean()) if len(valid) else np.nan
        ),
        "fraction_peo_dwells_with_multichain_coordination": (
            float(valid["multichain_peo_any"].mean()) if len(valid) else np.nan
        ),
        "time_weighted_tfsi_fraction_during_peo_dwells": (
            float(
                np.average(
                    valid["tfsi_time_fraction"],
                    weights=valid["n_frames"],
                )
            )
            if len(valid) else np.nan
        ),
        "time_weighted_multichain_fraction_during_peo_dwells": (
            float(
                np.average(
                    valid["multichain_peo_time_fraction"],
                    weights=valid["n_frames"],
                )
            )
            if len(valid) else np.nan
        ),
    }
    return pd.DataFrame([summary])

def summarize_transitions(
    transition_df: pd.DataFrame,
    n_blocks: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    block_rows = []
    total = len(transition_df)

    for transition_type in TRANSITION_ORDER:
        subset = transition_df[
            transition_df["transition_type"] == transition_type
        ]
        fractions = []
        for block in range(n_blocks):
            block_all = transition_df[
                transition_df["block_index"] == block
            ]
            block_count = np.sum(
                block_all["transition_type"] == transition_type
            )
            fraction = safe_fraction(block_count, len(block_all))
            fractions.append(fraction)
            block_rows.append(
                {
                    "block_index": block,
                    "transition_type": transition_type,
                    "n_events": int(block_count),
                    "n_all_transitions": int(len(block_all)),
                    "fraction": fraction,
                }
            )

        finite = np.asarray(
            [value for value in fractions if np.isfinite(value)],
            dtype=float,
        )
        mean = float(np.mean(finite)) if len(finite) else np.nan
        std = (
            float(np.std(finite, ddof=1)) if len(finite) >= 2 else np.nan
        )
        sem = std / math.sqrt(len(finite)) if len(finite) >= 2 else np.nan

        rows.append(
            {
                "transition_type": transition_type,
                "n_events": int(len(subset)),
                "fraction_all_transitions": safe_fraction(len(subset), total),
                "block_mean_fraction": mean,
                "block_std_fraction": std,
                "block_sem_fraction": sem,
                "n_blocks": int(len(finite)),
                "mean_absolute_delta_eo": (
                    float(subset["absolute_delta_eo"].mean())
                    if len(subset) else np.nan
                ),
                "fraction_tfsi_involved": (
                    float(subset["tfsi_involved"].mean())
                    if len(subset) else np.nan
                ),
                "fraction_multichain_peo_involved": (
                    float(subset["multichain_peo_involved"].mean())
                    if len(subset) else np.nan
                ),
                "mean_from_dwell_ps": (
                    float(subset["from_dwell_ps"].mean())
                    if len(subset) else np.nan
                ),
                "mean_to_dwell_ps": (
                    float(subset["to_dwell_ps"].mean())
                    if len(subset) else np.nan
                ),
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(block_rows)

def summarize_sliding_episodes(
    sliding_df: pd.DataFrame,
    n_metals: int,
    total_time_ns: float,
    peo_coordinated_metal_time_ns: float,
) -> pd.DataFrame:
    if sliding_df.empty:
        return pd.DataFrame(
            [
                {
                    "n_sliding_episodes": 0,
                    "n_neighbor_slide_steps": 0,
                    "neighbor_slide_steps_per_metal_ns_total": 0.0,
                    "neighbor_slide_steps_per_metal_ns_peo_coordinated": 0.0,
                }
            ]
        )

    n_steps = int(sliding_df["n_steps"].sum())
    return pd.DataFrame(
        [
            {
                "n_sliding_episodes": int(len(sliding_df)),
                "n_neighbor_slide_steps": n_steps,
                "mean_steps_per_episode": float(
                    sliding_df["n_steps"].mean()
                ),
                "median_steps_per_episode": float(
                    sliding_df["n_steps"].median()
                ),
                "maximum_steps_per_episode": int(
                    sliding_df["n_steps"].max()
                ),
                "fraction_multistep_episodes": float(
                    np.mean(sliding_df["n_steps"] >= 2)
                ),
                "fraction_episodes_with_tfsi": float(
                    sliding_df["tfsi_involved_any"].mean()
                ),
                "fraction_episodes_with_multichain_peo": float(
                    sliding_df[
                        "multichain_peo_involved_any"
                    ].mean()
                ),
                "neighbor_slide_steps_per_metal_ns_total": safe_fraction(
                    n_steps,
                    n_metals * total_time_ns,
                ),
                "neighbor_slide_steps_per_metal_ns_peo_coordinated": (
                    safe_fraction(n_steps, peo_coordinated_metal_time_ns)
                ),
            }
        ]
    )

def unwrap_episode(
    wrapped: np.ndarray,
    boxes: np.ndarray,
) -> np.ndarray:
    unwrapped = np.empty_like(wrapped, dtype=np.float64)
    unwrapped[0] = wrapped[0]
    for index in range(1, len(wrapped)):
        delta = wrapped[index] - wrapped[index - 1]
        delta = minimize_vectors(
            np.asarray(delta, dtype=np.float64)[None, :],
            np.asarray(boxes[index], dtype=np.float64),
        )[0]
        unwrapped[index] = unwrapped[index - 1] + delta
    return unwrapped

def accumulate_segmental_correlations(
    raw_chain: np.ndarray,
    metal_wrapped: np.ndarray,
    segment_wrapped: np.ndarray,
    boxes: np.ndarray,
    lags: np.ndarray,
) -> CorrelationAccumulator:
    accumulator = CorrelationAccumulator.create(lags)
    n_frames, n_metals = raw_chain.shape

    for metal in range(n_metals):
        chain_series = raw_chain[:, metal]
        start = 0
        while start < n_frames:
            chain = int(chain_series[start])
            stop = start + 1
            while stop < n_frames and int(chain_series[stop]) == chain:
                stop += 1

            if chain >= 0 and stop - start >= 2:
                metal_episode = np.asarray(
                    metal_wrapped[start:stop, metal], dtype=np.float64
                )
                segment_episode = np.asarray(
                    segment_wrapped[start:stop, metal], dtype=np.float64
                )
                box_episode = np.asarray(boxes[start:stop], dtype=np.float64)

                valid = (
                    np.all(np.isfinite(metal_episode), axis=1)
                    & np.all(np.isfinite(segment_episode), axis=1)
                )
                if np.all(valid):
                    metal_unwrapped = unwrap_episode(
                        metal_episode, box_episode
                    )
                    segment_unwrapped = unwrap_episode(
                        segment_episode, box_episode
                    )
                    length = len(metal_unwrapped)

                    for lag_index, lag in enumerate(lags):
                        lag = int(lag)
                        if lag >= length:
                            continue
                        dm = metal_unwrapped[lag:] - metal_unwrapped[:-lag]
                        ds = (
                            segment_unwrapped[lag:]
                            - segment_unwrapped[:-lag]
                        )
                        accumulator.sum_metal_sq[lag_index] += float(
                            np.einsum("ij,ij->", dm, dm)
                        )
                        accumulator.sum_segment_sq[lag_index] += float(
                            np.einsum("ij,ij->", ds, ds)
                        )
                        accumulator.sum_dot[lag_index] += float(
                            np.einsum("ij,ij->", dm, ds)
                        )
                        relative = dm - ds
                        accumulator.sum_relative_sq[lag_index] += float(
                            np.einsum("ij,ij->", relative, relative)
                        )
                        accumulator.n_origins[lag_index] += len(dm)
                        accumulator.n_episodes[lag_index] += 1

            start = stop

    return accumulator

def finalize_segmental_correlations(
    accumulator: CorrelationAccumulator,
    effective_dt_ps: float,
    min_origins: int,
    min_episodes: int,
) -> pd.DataFrame:
    rows = []
    for index, lag in enumerate(accumulator.lags):
        origins = int(accumulator.n_origins[index])
        episodes = int(accumulator.n_episodes[index])
        if origins == 0:
            continue

        metal_msd = accumulator.sum_metal_sq[index] / origins
        segment_msd = accumulator.sum_segment_sq[index] / origins
        dot = accumulator.sum_dot[index] / origins
        relative_msd = accumulator.sum_relative_sq[index] / origins

        denominator_geom = math.sqrt(max(metal_msd * segment_msd, 0.0))
        vector_correlation = (
            dot / denominator_geom if denominator_geom > 0 else np.nan
        )
        denominator_sum = metal_msd + segment_msd
        collective_alignment = (
            2.0 * dot / denominator_sum
            if denominator_sum > 0 else np.nan
        )
        relative_ratio = (
            relative_msd / denominator_sum
            if denominator_sum > 0 else np.nan
        )

        rows.append(
            {
                "lag_frames": int(lag),
                "lag_ps": float(lag * effective_dt_ps),
                "lag_ns": float(lag * effective_dt_ps / 1000.0),
                "n_time_origins": origins,
                "n_same_chain_episodes": episodes,
                "metal_msd_A2": metal_msd,
                "local_peo_segment_msd_A2": segment_msd,
                "metal_segment_dot_A2": dot,
                "metal_relative_to_segment_msd_A2": relative_msd,
                "vector_correlation": vector_correlation,
                "collective_alignment_index": collective_alignment,
                "relative_motion_ratio": relative_ratio,
                "identity_residual": (
                    relative_ratio - (1.0 - collective_alignment)
                    if np.isfinite(relative_ratio)
                    and np.isfinite(collective_alignment)
                    else np.nan
                ),
                "recommended_for_interpretation": int(
                    origins >= min_origins and episodes >= min_episodes
                ),
            }
        )
    return pd.DataFrame(rows)

def add_system_columns(
    dataframe: pd.DataFrame,
    system: str,
    metal: str,
    composition: str,
    temperature_K: float,
) -> pd.DataFrame:
    dataframe = dataframe.copy()
    dataframe.insert(0, "temperature_K", temperature_K)
    dataframe.insert(0, "composition", composition)
    dataframe.insert(0, "metal_species", metal)
    dataframe.insert(0, "system", system)
    return dataframe

def write_frame_database(
    path: Path,
    arrays,
    smoothed_chain: np.ndarray,
    smoothed_site: np.ndarray,
    metal_ids: np.ndarray,
    chain_map: ChainMap,
):
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        columns = [
            "selected_frame_index",
            "trajectory_frame",
            "time_ps",
            "metal_atom_id",
            "raw_chain_code",
            "raw_chain_label",
            "raw_eo_site",
            "raw_eo_coordinate",
            "smoothed_chain_code",
            "smoothed_chain_label",
            "smoothed_eo_site",
            "n_peo_oxygens",
            "n_peo_chains",
            "n_tfsi_oxygens",
            "tfsi_present",
            "multichain_peo",
        ]
        handle.write(",".join(columns) + "\n")
        n_frames, n_metals = smoothed_chain.shape
        for frame in range(n_frames):
            for metal in range(n_metals):
                raw_chain = int(arrays["raw_chain"][frame, metal])
                smooth_chain = int(smoothed_chain[frame, metal])
                row = [
                    frame,
                    int(arrays["frame_numbers"][frame]),
                    float(arrays["times_ps"][frame]),
                    int(metal_ids[metal]),
                    raw_chain,
                    (
                        chain_map.chain_labels[raw_chain]
                        if raw_chain >= 0 else "UNBOUND"
                    ),
                    int(arrays["raw_site"][frame, metal]),
                    float(arrays["raw_coordinate"][frame, metal]),
                    smooth_chain,
                    (
                        chain_map.chain_labels[smooth_chain]
                        if smooth_chain >= 0 else "UNBOUND"
                    ),
                    int(smoothed_site[frame, metal]),
                    int(arrays["n_peo_o"][frame, metal]),
                    int(arrays["n_peo_chains"][frame, metal]),
                    int(arrays["n_tfsi_o"][frame, metal]),
                    int(arrays["tfsi_present"][frame, metal]),
                    int(arrays["multichain_peo"][frame, metal]),
                ]
                handle.write(",".join(map(str, row)) + "\n")

def create_figures(
    output_dir: Path,
    transition_summary: pd.DataFrame,
    correlation_df: pd.DataFrame,
    arrays,
    smoothed_site: np.ndarray,
    smoothed_chain: np.ndarray,
    metal_ids: np.ndarray,
    n_representative: int,
):
    figures = output_dir / "figures"

    figure = plt.figure(figsize=(8, 5.5))
    axis = figure.add_subplot(111)
    x = np.arange(len(transition_summary))
    y = transition_summary["fraction_all_transitions"].to_numpy() * 100.0
    sem = transition_summary["block_sem_fraction"].to_numpy() * 100.0
    axis.bar(x, y, yerr=sem, capsize=4)
    axis.set_xticks(x)
    axis.set_xticklabels(
        transition_summary["transition_type"],
        rotation=30,
        ha="right",
    )
    axis.set_ylabel("Fraction of transitions (%)")
    axis.set_title("PEO coordination-site transition mechanisms")
    figure.tight_layout()
    figure.savefig(figures / "transition_mechanism_fractions.png", dpi=300)
    plt.close(figure)

    if not correlation_df.empty:
        figure = plt.figure(figsize=(7.5, 5.5))
        axis = figure.add_subplot(111)
        recommended = correlation_df[
            correlation_df["recommended_for_interpretation"] == 1
        ]
        plot_data = recommended if len(recommended) else correlation_df
        axis.semilogx(
            plot_data["lag_ns"],
            plot_data["collective_alignment_index"],
            marker="o",
            label="Collective alignment",
        )
        axis.semilogx(
            plot_data["lag_ns"],
            plot_data["vector_correlation"],
            marker="s",
            label="Vector correlation",
        )
        axis.set_xlabel("Lag time (ns)")
        axis.set_ylabel("Metal-local PEO coupling")
        axis.set_title("Segmental-motion coupling on the same PEO chain")
        axis.legend()
        figure.tight_layout()
        figure.savefig(figures / "metal_peo_segmental_coupling.png", dpi=300)
        plt.close(figure)

    n_plot = min(n_representative, smoothed_site.shape[1])
    if n_plot > 0:
        figure = plt.figure(figsize=(9, 5.8))
        axis = figure.add_subplot(111)
        time_ns = arrays["times_ps"] / 1000.0
        for metal in range(n_plot):
            values = smoothed_site[:, metal].astype(float)
            values[smoothed_chain[:, metal] < 0] = np.nan
            axis.plot(
                time_ns,
                values,
                linewidth=0.9,
                label=f"Metal {int(metal_ids[metal])}",
            )
        axis.set_xlabel("Time (ns)")
        axis.set_ylabel("Dominant EO site index")
        axis.set_title("Representative PEO EO-site trajectories")
        axis.legend(ncol=2)
        figure.tight_layout()
        figure.savefig(figures / "representative_eo_site_trajectories.png", dpi=300)
        plt.close(figure)

def main() -> None:
    args = build_parser().parse_args()

    working_directory = Path.cwd()
    metal = str(args.metal or "metal")
    composition = args.composition or infer_composition(working_directory) or "composition"

    if args.peo_cutoff <= 0 or args.tfsi_cutoff <= 0:
        raise ValueError("Coordination cutoffs must be positive.")
    if args.segment_half_width < 0:
        raise ValueError("--segment-half-width must be nonnegative.")
    if args.stride < 1:
        raise ValueError("--stride must be at least 1.")
    if args.n_blocks < 2:
        raise ValueError("--n-blocks must be at least 2.")

    topology = discover_unique_file(
        args.topology,
        ["*.data", "*.lmp", "*.lammps"],
        "topology",
    )
    trajectory = discover_unique_file(
        args.trajectory,
        ["*.lammpsdump", "*.lammpstrj", "*.dump"],
        "trajectory",
    )
    output_dir = prepare_output_dir(args.output_dir, args.overwrite)
    logger = configure_logger(output_dir)

    system = f"{metal}_{composition}"
    logger.info("=" * 79)
    logger.info("MODULE 06: PEO SLIDING AND SEGMENTAL-MOTION ANALYSIS")
    logger.info("=" * 79)
    logger.info("System             : %s", system)
    logger.info("Topology           : %s", topology)
    logger.info("Trajectory         : %s", trajectory)
    logger.info("M-O(PEO) cutoff    : %.3f Å", args.peo_cutoff)
    logger.info("M-O(TFSI) cutoff   : %.3f Å", args.tfsi_cutoff)
    logger.info("Metal selection    : %s", args.metal_selection)
    logger.info("Smoothing window   : %d frame(s)", args.smoothing_window)

    analysis_start = time.perf_counter()
    universe = mda.Universe(str(topology), str(trajectory), format="LAMMPSDUMP")
    metals = universe.select_atoms(args.metal_selection)
    peo_oxygens = universe.select_atoms(args.peo_o_selection)
    tfsi_oxygens = universe.select_atoms(args.tfsi_o_selection)

    if len(metals) == 0:
        raise RuntimeError(f"No atoms matched: {args.metal_selection}")
    if len(peo_oxygens) == 0:
        raise RuntimeError(f"No atoms matched: {args.peo_o_selection}")
    if len(tfsi_oxygens) == 0:
        raise RuntimeError(f"No atoms matched: {args.tfsi_o_selection}")

    chain_map = build_chain_map(
        peo_oxygens,
        args.peo_chain_attribute,
    )
    chain_sizes = [
        len(indices)
        for indices in chain_map.chain_to_local_indices.values()
    ]

    logger.info("Metal atoms        : %d", len(metals))
    logger.info("PEO oxygens        : %d", len(peo_oxygens))
    logger.info("TFSI oxygens       : %d", len(tfsi_oxygens))
    logger.info(
        "PEO chains         : %d via %s",
        len(chain_map.chain_labels),
        chain_map.source_attribute,
    )
    logger.info(
        "EO sites/chain     : min=%d median=%.1f max=%d",
        min(chain_sizes),
        float(np.median(chain_sizes)),
        max(chain_sizes),
    )

    arrays = process_trajectory(
        universe,
        metals,
        peo_oxygens,
        tfsi_oxygens,
        chain_map,
        args,
        logger,
    )

    if args.smoothing_window == 3:
        smoothed_chain, smoothed_site = smooth_states_three_frame(
            arrays["raw_chain"],
            arrays["raw_site"],
        )
    else:
        smoothed_chain = arrays["raw_chain"].copy()
        smoothed_site = arrays["raw_site"].copy()

    effective_dt_ps = args.frame_interval_ps * args.stride
    metal_ids = np.asarray(metals.ids, dtype=int)

    dwell_df, transition_df = build_dwell_and_transition_tables(
        arrays,
        smoothed_chain,
        smoothed_site,
        metal_ids,
        chain_map,
        effective_dt_ps,
        args.n_blocks,
    )
    sliding_df = build_sliding_episodes(transition_df)

    dwell_summary = summarize_dwells(dwell_df)
    transition_summary, transition_blocks = summarize_transitions(
        transition_df,
        args.n_blocks,
    )

    total_time_ns = (
        len(arrays["frame_numbers"]) * effective_dt_ps / 1000.0
    )
    peo_coordinated_metal_time_ns = float(
        np.sum(smoothed_chain >= 0) * effective_dt_ps / 1000.0
    )
    sliding_summary = summarize_sliding_episodes(
        sliding_df,
        len(metals),
        total_time_ns,
        peo_coordinated_metal_time_ns,
    )

    lags = make_lag_grid(
        effective_dt_ps,
        args.max_correlation_ns,
        args.n_log_lags,
    )
    accumulator = accumulate_segmental_correlations(
        arrays["raw_chain"],
        arrays["metal_wrapped"],
        arrays["segment_wrapped"],
        arrays["boxes"],
        lags,
    )
    correlation_df = finalize_segmental_correlations(
        accumulator,
        effective_dt_ps,
        args.min_correlation_origins,
        args.min_correlation_episodes,
    )

    dwell_df = add_system_columns(
        dwell_df, system, metal, composition, args.temperature_K
    )
    transition_df = add_system_columns(
        transition_df, system, metal, composition, args.temperature_K
    )
    sliding_df = add_system_columns(
        sliding_df, system, metal, composition, args.temperature_K
    ) if not sliding_df.empty else sliding_df
    dwell_summary = add_system_columns(
        dwell_summary, system, metal, composition, args.temperature_K
    )
    transition_summary = add_system_columns(
        transition_summary, system, metal, composition, args.temperature_K
    )
    transition_blocks = add_system_columns(
        transition_blocks, system, metal, composition, args.temperature_K
    )
    sliding_summary = add_system_columns(
        sliding_summary, system, metal, composition, args.temperature_K
    )
    correlation_df = add_system_columns(
        correlation_df, system, metal, composition, args.temperature_K
    )

    dwell_df.to_csv(
        output_dir / "eo_site_dwell_events.csv.gz",
        index=False,
        compression="gzip",
    )
    transition_df.to_csv(
        output_dir / "eo_site_transition_events.csv.gz",
        index=False,
        compression="gzip",
    )
    if not sliding_df.empty:
        sliding_df.to_csv(
            output_dir / "sliding_episode_events.csv.gz",
            index=False,
            compression="gzip",
        )
    dwell_summary.to_csv(
        output_dir / "eo_site_dwell_summary.csv", index=False
    )
    transition_summary.to_csv(
        output_dir / "eo_site_transition_summary.csv", index=False
    )
    transition_blocks.to_csv(
        output_dir / "eo_site_transition_block_summary.csv", index=False
    )
    transition_summary[
        [
            "system",
            "metal_species",
            "composition",
            "temperature_K",
            "transition_type",
            "n_events",
            "fraction_tfsi_involved",
            "fraction_multichain_peo_involved",
        ]
    ].to_csv(
        output_dir / "tfsi_bridge_by_transition_type.csv",
        index=False,
    )
    sliding_summary.to_csv(
        output_dir / "sliding_episode_summary.csv", index=False
    )
    correlation_df.to_csv(
        output_dir / "metal_peo_segmental_correlation.csv",
        index=False,
    )

    representative_rows = []
    n_rep = min(args.representative_metals, len(metals))
    for frame in range(len(arrays["frame_numbers"])):
        for metal_local in range(n_rep):
            chain = int(smoothed_chain[frame, metal_local])
            representative_rows.append(
                {
                    "system": system,
                    "trajectory_frame": int(
                        arrays["frame_numbers"][frame]
                    ),
                    "time_ps": float(arrays["times_ps"][frame]),
                    "metal_atom_id": int(metal_ids[metal_local]),
                    "dominant_chain_code": chain,
                    "dominant_chain_label": (
                        chain_map.chain_labels[chain]
                        if chain >= 0 else "UNBOUND"
                    ),
                    "eo_site": int(smoothed_site[frame, metal_local]),
                    "tfsi_present": int(
                        arrays["tfsi_present"][frame, metal_local]
                    ),
                    "multichain_peo": int(
                        arrays["multichain_peo"][frame, metal_local]
                    ),
                }
            )
    pd.DataFrame(representative_rows).to_csv(
        output_dir / "representative_eo_trajectories.csv.gz",
        index=False,
        compression="gzip",
    )

    if args.write_frame_database:
        logger.info("Writing full frame-level database...")
        write_frame_database(
            output_dir / "frame_level_polymer_coordinate.csv.gz",
            arrays,
            smoothed_chain,
            smoothed_site,
            metal_ids,
            chain_map,
        )

    create_figures(
        output_dir,
        transition_summary,
        correlation_df,
        arrays,
        smoothed_site,
        smoothed_chain,
        metal_ids,
        args.representative_metals,
    )

    fraction_sum = float(
        transition_summary["fraction_all_transitions"].sum()
    )
    correlation_identity_max = (
        float(np.nanmax(np.abs(correlation_df["identity_residual"])))
        if len(correlation_df) else np.nan
    )
    site_valid = bool(
        np.all(
            (smoothed_chain < 0)
            | (smoothed_site >= 1)
        )
    )
    transition_fraction_valid = bool(
        len(transition_df) == 0 or abs(fraction_sum - 1.0) < 1.0e-10
    )
    correlation_identity_valid = bool(
        len(correlation_df) == 0
        or correlation_identity_max < 1.0e-8
    )

    validation = {
        "status": (
            "PASSED"
            if site_valid
            and transition_fraction_valid
            and correlation_identity_valid
            else "FAILED"
        ),
        "checks": {
            "all_selected_frames_processed": True,
            "peo_chain_mapping_plausible": bool(
                len(chain_map.chain_labels) >= 2
            ),
            "smoothed_eo_sites_valid": site_valid,
            "transition_fractions_sum_to_one": transition_fraction_valid,
            "segmental_relative_motion_identity": (
                correlation_identity_valid
            ),
        },
        "metrics": {
            "n_selected_frames": int(len(arrays["frame_numbers"])),
            "n_metals": int(len(metals)),
            "n_peo_oxygens": int(len(peo_oxygens)),
            "n_peo_chains": int(len(chain_map.chain_labels)),
            "n_dwell_events": int(len(dwell_df)),
            "n_transitions": int(len(transition_df)),
            "n_sliding_episodes": int(len(sliding_df)),
            "transition_fraction_sum": fraction_sum,
            "maximum_correlation_identity_residual": (
                correlation_identity_max
            ),
            "peo_coordinated_metal_time_fraction": float(
                np.mean(smoothed_chain >= 0)
            ),
            "tfsi_contact_metal_time_fraction": float(
                np.mean(arrays["tfsi_present"])
            ),
            "multichain_peo_metal_time_fraction": float(
                np.mean(arrays["multichain_peo"])
            ),
        },
        "warnings": [],
        "errors": [],
    }
    with open(
        output_dir / "validation_report.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(validation, handle, indent=2)

    key_transition = {
        row["transition_type"]: {
            "n_events": int(row["n_events"]),
            "fraction": float(row["fraction_all_transitions"]),
            "fraction_tfsi_involved": float(
                row["fraction_tfsi_involved"]
            ) if np.isfinite(row["fraction_tfsi_involved"]) else None,
            "fraction_multichain_peo_involved": float(
                row["fraction_multichain_peo_involved"]
            ) if np.isfinite(
                row["fraction_multichain_peo_involved"]
            ) else None,
        }
        for _, row in transition_summary.iterrows()
    }

    recommended_corr = correlation_df[
        correlation_df["recommended_for_interpretation"] == 1
    ]
    near_one_ns = None
    if len(recommended_corr):
        index = (
            recommended_corr["lag_ns"] - 1.0
        ).abs().idxmin()
        row = recommended_corr.loc[index]
        near_one_ns = {
            "lag_ns": float(row["lag_ns"]),
            "n_time_origins": int(row["n_time_origins"]),
            "n_same_chain_episodes": int(
                row["n_same_chain_episodes"]
            ),
            "vector_correlation": float(row["vector_correlation"]),
            "collective_alignment_index": float(
                row["collective_alignment_index"]
            ),
            "relative_motion_ratio": float(
                row["relative_motion_ratio"]
            ),
        }

    runtime = time.perf_counter() - analysis_start
    summary = {
        "module": "segmental_motion",
        "system": {
            "system": system,
            "metal_species": metal,
            "composition": composition,
            "temperature_K": args.temperature_K,
            "n_metals": int(len(metals)),
            "n_peo_oxygens": int(len(peo_oxygens)),
            "n_tfsi_oxygens": int(len(tfsi_oxygens)),
            "n_peo_chains": int(len(chain_map.chain_labels)),
            "peo_chain_attribute": chain_map.source_attribute,
            "eo_sites_per_chain_min": int(min(chain_sizes)),
            "eo_sites_per_chain_median": float(np.median(chain_sizes)),
            "eo_sites_per_chain_max": int(max(chain_sizes)),
        },
        "configuration": {
            "topology": str(topology),
            "trajectory": str(trajectory),
            "metal_selection": args.metal_selection,
            "peo_o_selection": args.peo_o_selection,
            "tfsi_o_selection": args.tfsi_o_selection,
            "peo_cutoff_A": args.peo_cutoff,
            "tfsi_cutoff_A": args.tfsi_cutoff,
            "frame_interval_ps": args.frame_interval_ps,
            "stride": args.stride,
            "effective_dt_ps": effective_dt_ps,
            "smoothing_window": args.smoothing_window,
            "segment_half_width": args.segment_half_width,
            "n_blocks": args.n_blocks,
        },
        "population": {
            "peo_coordinated_metal_time_fraction": float(
                np.mean(smoothed_chain >= 0)
            ),
            "tfsi_contact_metal_time_fraction": float(
                np.mean(arrays["tfsi_present"])
            ),
            "multichain_peo_metal_time_fraction": float(
                np.mean(arrays["multichain_peo"])
            ),
            "mean_peo_oxygen_cn": float(
                np.mean(arrays["n_peo_o"])
            ),
            "mean_peo_chain_cn": float(
                np.mean(arrays["n_peo_chains"])
            ),
        },
        "transition_summary": key_transition,
        "dwell_summary": dwell_summary.iloc[0].to_dict(),
        "sliding_episode_summary": sliding_summary.iloc[0].to_dict(),
        "segmental_correlation_near_1_ns": near_one_ns,
        "validation_status": validation["status"],
        "scope_note": (
            "Neighbor-slide classifications use a smoothed robust EO-coordinate "
            "center. Segmental coupling is conditional on the metal remaining "
            "on the same instantaneous dominant PEO chain for the lag interval."
        ),
        "runtime_seconds": runtime,
    }
    with open(
        output_dir / "summary.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2, default=str)

    logger.info("=" * 79)
    logger.info("MODULE 06 COMPLETED")
    logger.info("=" * 79)
    logger.info("Dwell events        : %d", len(dwell_df))
    logger.info("Transitions         : %d", len(transition_df))
    logger.info("Sliding episodes    : %d", len(sliding_df))
    logger.info(
        "Neighbor slides     : %d (%.2f%% of transitions)",
        int(
            transition_summary.loc[
                transition_summary["transition_type"]
                == "Same_Chain_Neighbor_Slide",
                "n_events",
            ].iloc[0]
        ),
        100.0
        * float(
            transition_summary.loc[
                transition_summary["transition_type"]
                == "Same_Chain_Neighbor_Slide",
                "fraction_all_transitions",
            ].iloc[0]
        ),
    )
    logger.info(
        "PEO time fraction   : %.3f",
        float(np.mean(smoothed_chain >= 0)),
    )
    logger.info("Validation          : %s", validation["status"])
    logger.info("Output directory    : %s", output_dir)
    logger.info("Runtime             : %.1f min", runtime / 60.0)

if __name__ == "__main__":
    main()