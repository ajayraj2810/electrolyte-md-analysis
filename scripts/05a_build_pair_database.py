#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import math
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import MDAnalysis as mda
    from MDAnalysis.lib.distances import capped_distance
except ImportError as exc:
    raise ImportError(
        "MDAnalysis is required. Activate your environment and install it with:\n"
        "    python -m pip install MDAnalysis"
    ) from exc

DEFAULT_TOPOLOGY = "system.data"
DEFAULT_TRAJECTORY = "system.lammpsdump"
DEFAULT_OUTPUT_DIR = "05E1_pair_database"

DEFAULT_FRAME_INTERVAL_PS = 10.0
DEFAULT_START_FRAME = 0
DEFAULT_STOP_FRAME: Optional[int] = None
DEFAULT_STRIDE = 1

METAL_ATOM_TYPE = 17
TFSI_REFERENCE_ATOM_TYPE = 7
TFSI_OXYGEN_ATOM_TYPE = 9

LOG_EVERY_N_FRAMES = 100
FLUSH_EVERY_N_FRAMES = 100
CSV_FLOAT_FORMAT = ".8g"

PAIR_COLUMNS = [
    "system",
    "metal_species",
    "composition",
    "temperature_K",
    "frame",
    "trajectory_timestep",
    "time_ps",
    "pair_id",
    "pair_instance_id",
    "metal_id",
    "metal_resid",
    "tfsi_resid",
    "tfsi_reference_atom_id",
    "n_coordinating_oxygens",
    "min_metal_oxygen_distance_A",
    "mean_metal_oxygen_distance_A",
    "coordinating_oxygen_ids",
    "continuing_pair",
    "new_pair_event",
    "pair_age_frames",
    "pair_age_ps",
    "pair_event_start_frame",
    "pair_event_start_time_ps",
    "metal_x_A",
    "metal_y_A",
    "metal_z_A",
    "tfsi_x_A",
    "tfsi_y_A",
    "tfsi_z_A",
    "box_a_A",
    "box_b_A",
    "box_c_A",
    "box_alpha_deg",
    "box_beta_deg",
    "box_gamma_deg",
]

FRAME_SUMMARY_COLUMNS = [
    "system",
    "metal_species",
    "composition",
    "temperature_K",
    "frame",
    "trajectory_timestep",
    "time_ps",
    "n_metals",
    "n_tfsi_residues",
    "n_metal_oxygen_contacts",
    "n_unique_metal_tfsi_pairs",
    "n_coordinated_metals",
    "fraction_coordinated_metals",
    "n_coordinated_tfsi",
    "fraction_coordinated_tfsi",
    "mean_pairs_per_metal_all",
    "mean_pairs_per_coordinated_metal",
    "max_pairs_for_one_metal",
    "mean_coordinating_oxygens_per_pair",
    "max_coordinating_oxygens_per_pair",
    "mean_min_pair_distance_A",
    "n_continuing_pairs",
    "n_new_pair_events",
    "n_ended_pair_events",
    "box_volume_A3",
]

@dataclass(frozen=True)
class SystemIdentity:
    system: str
    metal_species: str
    composition: str
    temperature_K: float
    topology: str
    trajectory: str

@dataclass(frozen=True)
class AnalysisConfig:
    topology: Path
    trajectory: Path
    output_dir: Path

    frame_interval_ps: float
    start_frame: int
    stop_frame: Optional[int]
    stride: int

    metal_atom_type: int
    tfsi_reference_atom_type: int
    tfsi_oxygen_atom_type: int
    metal_oxygen_cutoff_A: float

    overwrite: bool = False
    save_oxygen_ids: bool = True
    log_every_n_frames: int = LOG_EVERY_N_FRAMES
    flush_every_n_frames: int = FLUSH_EVERY_N_FRAMES

    def validate(self) -> None:
        if not self.topology.exists():
            raise FileNotFoundError(f"Topology not found: {self.topology}")
        if not self.trajectory.exists():
            raise FileNotFoundError(f"Trajectory not found: {self.trajectory}")
        if self.frame_interval_ps <= 0:
            raise ValueError("frame_interval_ps must be positive.")
        if self.start_frame < 0:
            raise ValueError("start_frame cannot be negative.")
        if self.stop_frame is not None and self.stop_frame <= self.start_frame:
            raise ValueError("stop_frame must be greater than start_frame.")
        if self.stride <= 0:
            raise ValueError("stride must be a positive integer.")
        if self.metal_oxygen_cutoff_A <= 0:
            raise ValueError("metal_oxygen_cutoff_A must be positive.")
        if self.log_every_n_frames <= 0:
            raise ValueError("log_every_n_frames must be positive.")
        if self.flush_every_n_frames <= 0:
            raise ValueError("flush_every_n_frames must be positive.")

@dataclass
class ValidationReport:
    status: str = "NOT_RUN"
    checks: Dict[str, bool] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def add_check(self, name: str, passed: bool) -> None:
        self.checks[name] = bool(passed)

    def add_metric(self, name: str, value: Any) -> None:
        self.metrics[name] = make_json_safe(value)

    def warn(self, message: str) -> None:
        self.warnings.append(str(message))

    def error(self, message: str) -> None:
        self.errors.append(str(message))

    def finalize(self) -> None:
        if self.errors:
            self.status = "FAILED"
        elif self.checks and all(self.checks.values()):
            self.status = "PASSED"
        elif self.checks:
            self.status = "FAILED"
        else:
            self.status = "NOT_RUN"

    def to_dict(self) -> Dict[str, Any]:
        self.finalize()
        return make_json_safe(asdict(self))

@dataclass
class PairState:
    pair_instance_id: str
    event_number: int
    age_frames: int
    start_frame: int
    start_time_ps: float

class PairTracker:

    def __init__(self) -> None:
        self.active: Dict[str, PairState] = {}
        self.event_counts: Dict[str, int] = {}
        self.total_started_events = 0
        self.total_ended_events = 0
        self.max_pair_age_frames = 0

    def update(
        self,
        current_pair_ids: Sequence[str],
        frame: int,
        time_ps: float,
    ) -> Tuple[Dict[str, PairState], int, int]:
        current_set = set(current_pair_ids)
        previous_set = set(self.active)

        ended = previous_set - current_set
        continuing = previous_set & current_set
        new = current_set - previous_set

        for pair_id in ended:
            del self.active[pair_id]

        for pair_id in continuing:
            state = self.active[pair_id]
            state.age_frames += 1
            self.max_pair_age_frames = max(
                self.max_pair_age_frames,
                state.age_frames,
            )

        for pair_id in new:
            event_number = self.event_counts.get(pair_id, 0) + 1
            self.event_counts[pair_id] = event_number
            state = PairState(
                pair_instance_id=f"{pair_id}_E{event_number:04d}",
                event_number=event_number,
                age_frames=1,
                start_frame=int(frame),
                start_time_ps=float(time_ps),
            )
            self.active[pair_id] = state
            self.total_started_events += 1
            self.max_pair_age_frames = max(self.max_pair_age_frames, 1)

        self.total_ended_events += len(ended)
        return self.active, len(new), len(ended)

    def close_remaining_events(self) -> int:
        remaining = len(self.active)
        self.total_ended_events += remaining
        self.active.clear()
        return remaining

class PairDatabaseWriter:

    def __init__(self, path: Path, fieldnames: Sequence[str]) -> None:
        self.path = path
        self.fieldnames = list(fieldnames)
        self.handle = None
        self.writer = None
        self.rows_written = 0

    def __enter__(self) -> "PairDatabaseWriter":
        self.handle = gzip.open(
            self.path,
            mode="wt",
            encoding="utf-8",
            newline="",
            compresslevel=6,
        )
        self.writer = csv.DictWriter(
            self.handle,
            fieldnames=self.fieldnames,
            extrasaction="raise",
        )
        self.writer.writeheader()
        return self

    def write_rows(self, rows: Sequence[Mapping[str, Any]]) -> None:
        if not rows:
            return
        assert self.writer is not None
        self.writer.writerows(rows)
        self.rows_written += len(rows)

    def flush(self) -> None:
        if self.handle is not None:
            self.handle.flush()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.handle is not None:
            self.handle.close()

class OutputManager:
    def __init__(self, root: Path, overwrite: bool = False) -> None:
        self.root = root.resolve()
        self.figures = self.root / "figures"
        self.overwrite = overwrite

        self.pair_database = self.root / "pair_database.csv.gz"
        self.frame_summary = self.root / "frame_pair_summary.csv"
        self.validation_report = self.root / "validation_report.json"
        self.metadata = self.root / "metadata.json"
        self.log_file = self.root / "05E1_pair_database.log"

    def prepare(self) -> None:
        if self.root.exists() and not self.overwrite:
            protected = [
                self.pair_database,
                self.frame_summary,
                self.validation_report,
                self.metadata,
            ]
            existing = [path for path in protected if path.exists()]
            if existing:
                listing = "\n".join(f"  - {path}" for path in existing)
                raise FileExistsError(
                    "Output files already exist. Use --overwrite to replace them:\n"
                    f"{listing}"
                )

        self.root.mkdir(parents=True, exist_ok=True)
        self.figures.mkdir(parents=True, exist_ok=True)

def configure_logging(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("05E1_pair_database")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def make_json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "__dataclass_fields__"):
        return make_json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(v) for v in value]
    return value

def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(make_json_safe(payload), handle, indent=2, sort_keys=False)
        handle.write("\n")
    tmp_path.replace(path)

def format_float(value: float) -> str:
    if value is None or not np.isfinite(value):
        return ""
    return format(float(value), CSV_FLOAT_FORMAT)

def detect_metal_species(*texts: str) -> str:

    return "metal"

def detect_composition(*texts: str) -> str:
    joined = " ".join(str(text) for text in texts)
    matches = re.findall(
        r"(?<!\d)(20_5|15_10|15_15|5_20)(?!\d)",
        joined,
    )
    unique = sorted(set(matches))

    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        raise ValueError(f"Ambiguous composition labels detected: {unique}")

    raise ValueError(
        "Could not detect composition. Pass --composition explicitly."
    )

def detect_temperature_from_path(*texts: str, default: float = 380.0) -> float:
    joined = " ".join(str(text) for text in texts)
    candidates = [
        float(value)
        for value in re.findall(
            r"(?<!\d)(2\d{2}|3\d{2}|4\d{2}|5\d{2})(?:K)?(?!\d)",
            joined,
        )
    ]
    plausible = sorted(set(x for x in candidates if 200 <= x <= 600))
    return plausible[0] if len(plausible) == 1 else float(default)

def build_system_identity(
    topology: Path,
    trajectory: Path,
    metal_override: Optional[str],
    composition_override: Optional[str],
    temperature_K: float,
) -> SystemIdentity:
    metal = str(metal_override or "metal")
    composition = composition_override or detect_composition(topology, trajectory)

    return SystemIdentity(
        system=f"{metal}_{composition}",
        metal_species=metal,
        composition=composition,
        temperature_K=float(temperature_K),
        topology=str(topology.resolve()),
        trajectory=str(trajectory.resolve()),
    )

def load_universe(topology: Path, trajectory: Path) -> mda.Universe:
    try:
        return mda.Universe(
            str(topology),
            str(trajectory),
            topology_format="DATA",
            format="LAMMPSDUMP",
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load topology/trajectory with MDAnalysis.\n"
            f"Topology: {topology}\n"
            f"Trajectory: {trajectory}\n"
            f"Original error: {exc}"
        ) from exc

def select_atoms_by_type(
    universe: mda.Universe,
    atom_type: int,
    label: str,
):
    selection = universe.select_atoms(f"type {atom_type}")
    if selection.n_atoms == 0:
        raise ValueError(
            f"No {label} atoms found using selection 'type {atom_type}'."
        )
    return selection

def require_topology_attributes(
    atom_group,
    attributes: Sequence[str],
    label: str,
) -> None:
    missing: List[str] = []
    for attribute in attributes:
        try:
            getattr(atom_group, attribute)
        except (AttributeError, mda.exceptions.NoDataError):
            missing.append(attribute)

    if missing:
        raise ValueError(
            f"{label} atoms are missing required topology attributes: {missing}. "
            "The LAMMPS data file must preserve atom IDs and molecule/residue IDs."
        )

def build_tfsi_reference_mapping(
    tfsi_reference_atoms,
    tfsi_oxygen_atoms,
) -> Tuple[Dict[int, int], Dict[int, int]]:

    require_topology_attributes(
        tfsi_reference_atoms,
        ["ids", "resids"],
        "TFSI reference",
    )
    require_topology_attributes(
        tfsi_oxygen_atoms,
        ["ids", "resids"],
        "TFSI oxygen",
    )

    ref_resids = np.asarray(tfsi_reference_atoms.resids, dtype=np.int64)
    unique_ref, ref_counts = np.unique(ref_resids, return_counts=True)

    bad_ref = unique_ref[ref_counts != 1]
    if bad_ref.size:
        preview = bad_ref[:10].tolist()
        raise ValueError(
            "Expected exactly one type-7 TFSI reference atom per TFSI residue. "
            f"Residues violating this rule include: {preview}"
        )

    oxygen_resids = np.unique(
        np.asarray(tfsi_oxygen_atoms.resids, dtype=np.int64)
    )
    missing_reference = sorted(set(oxygen_resids) - set(unique_ref))
    extra_reference = sorted(set(unique_ref) - set(oxygen_resids))

    if missing_reference:
        raise ValueError(
            "Some TFSI oxygen residues have no type-7 reference atom. "
            f"First missing residues: {missing_reference[:10]}"
        )

    if extra_reference:
        raise ValueError(
            "Some type-7 reference residues contain no selected TFSI oxygen. "
            f"First unmatched residues: {extra_reference[:10]}"
        )

    resid_to_local_index = {
        int(resid): int(local_index)
        for local_index, resid in enumerate(ref_resids)
    }
    resid_to_atom_id = {
        int(resid): int(atom_id)
        for resid, atom_id in zip(ref_resids, tfsi_reference_atoms.ids)
    }

    return resid_to_local_index, resid_to_atom_id

def summarize_topology(
    universe: mda.Universe,
    metal_atoms,
    tfsi_reference_atoms,
    tfsi_oxygen_atoms,
) -> Dict[str, Any]:
    require_topology_attributes(metal_atoms, ["ids", "resids"], "Metal")
    require_topology_attributes(tfsi_oxygen_atoms, ["ids", "resids"], "TFSI oxygen")

    tfsi_resids = np.asarray(tfsi_oxygen_atoms.resids, dtype=np.int64)
    unique_tfsi_resids, oxygen_counts = np.unique(
        tfsi_resids,
        return_counts=True,
    )

    return {
        "n_total_atoms": int(universe.atoms.n_atoms),
        "n_metal_atoms": int(metal_atoms.n_atoms),
        "n_tfsi_reference_atoms": int(tfsi_reference_atoms.n_atoms),
        "n_tfsi_oxygen_atoms": int(tfsi_oxygen_atoms.n_atoms),
        "n_tfsi_residues": int(unique_tfsi_resids.size),
        "tfsi_oxygen_count_per_residue": {
            "minimum": int(oxygen_counts.min()),
            "maximum": int(oxygen_counts.max()),
            "mean": float(oxygen_counts.mean()),
            "median": float(np.median(oxygen_counts)),
        },
        "metal_atom_id_min": int(np.min(metal_atoms.ids)),
        "metal_atom_id_max": int(np.max(metal_atoms.ids)),
        "metal_resid_min": int(np.min(metal_atoms.resids)),
        "metal_resid_max": int(np.max(metal_atoms.resids)),
        "tfsi_resid_min": int(np.min(unique_tfsi_resids)),
        "tfsi_resid_max": int(np.max(unique_tfsi_resids)),
    }

def validate_initial_box(universe: mda.Universe) -> Dict[str, Any]:
    universe.trajectory[0]
    dimensions = np.asarray(universe.trajectory.ts.dimensions, dtype=float)

    if dimensions.size < 6:
        raise ValueError("Trajectory lacks complete periodic box dimensions.")

    lengths = dimensions[:3]
    angles = dimensions[3:6]

    if np.any(~np.isfinite(lengths)) or np.any(lengths <= 0):
        raise ValueError(f"Invalid box lengths: {lengths}")
    if np.any(~np.isfinite(angles)) or np.any(angles <= 0):
        raise ValueError(f"Invalid box angles: {angles}")

    return {
        "box_lengths_A": lengths.tolist(),
        "box_angles_deg": angles.tolist(),
        "box_volume_A3": float(universe.trajectory.ts.volume),
    }

def resolve_frame_selection(
    n_frames: int,
    start: int,
    stop: Optional[int],
    stride: int,
) -> Tuple[int, int, int, int]:
    resolved_stop = n_frames if stop is None else min(int(stop), n_frames)

    if start >= n_frames:
        raise ValueError(
            f"start_frame={start} is outside trajectory with {n_frames} frames."
        )

    n_selected = len(range(start, resolved_stop, stride))
    if n_selected == 0:
        raise ValueError("Selected frame range is empty.")

    return int(start), int(resolved_stop), int(stride), int(n_selected)

def frame_time_ps(frame_index: int, frame_interval_ps: float) -> float:
    return float(frame_index) * float(frame_interval_ps)

def get_trajectory_timestep(ts) -> Optional[int]:

    data = getattr(ts, "data", {})
    for key in ("step", "timestep", "time_step"):
        if key in data:
            try:
                return int(data[key])
            except (TypeError, ValueError):
                pass
    return None

def format_eta(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds_int = int(round(seconds))
    hours, rem = divmod(seconds_int, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    return f"{minutes:d}m {secs:02d}s"

def detect_frame_pairs(
    metal_atoms,
    tfsi_oxygen_atoms,
    cutoff_A: float,
    box: np.ndarray,
) -> Tuple[List[Dict[str, Any]], int]:

    metal_positions = np.asarray(metal_atoms.positions, dtype=np.float64)
    oxygen_positions = np.asarray(tfsi_oxygen_atoms.positions, dtype=np.float64)

    contact_indices, contact_distances = capped_distance(
        metal_positions,
        oxygen_positions,
        max_cutoff=float(cutoff_A),
        min_cutoff=None,
        box=np.asarray(box, dtype=np.float64),
        method=None,
        return_distances=True,
    )

    n_contacts = int(len(contact_distances))
    if n_contacts == 0:
        return [], 0

    metal_ids = np.asarray(metal_atoms.ids, dtype=np.int64)
    metal_resids = np.asarray(metal_atoms.resids, dtype=np.int64)
    oxygen_ids = np.asarray(tfsi_oxygen_atoms.ids, dtype=np.int64)
    oxygen_resids = np.asarray(tfsi_oxygen_atoms.resids, dtype=np.int64)

    grouped: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for (metal_local, oxygen_local), distance in zip(
        contact_indices,
        contact_distances,
    ):
        m_local = int(metal_local)
        o_local = int(oxygen_local)

        metal_id = int(metal_ids[m_local])
        tfsi_resid = int(oxygen_resids[o_local])
        key = (metal_id, tfsi_resid)

        if key not in grouped:
            grouped[key] = {
                "metal_local_index": m_local,
                "metal_id": metal_id,
                "metal_resid": int(metal_resids[m_local]),
                "tfsi_resid": tfsi_resid,
                "oxygen_ids": [],
                "distances": [],
            }

        grouped[key]["oxygen_ids"].append(int(oxygen_ids[o_local]))
        grouped[key]["distances"].append(float(distance))

    pair_records: List[Dict[str, Any]] = []

    for key in sorted(grouped):
        record = grouped[key]
        distances = np.asarray(record.pop("distances"), dtype=np.float64)
        oxygen_id_list = sorted(set(record.pop("oxygen_ids")))

        record.update(
            {
                "oxygen_ids": oxygen_id_list,
                "n_coordinating_oxygens": int(len(oxygen_id_list)),
                "min_distance_A": float(np.min(distances)),
                "mean_distance_A": float(np.mean(distances)),
            }
        )
        pair_records.append(record)

    return pair_records, n_contacts

def process_trajectory(
    universe: mda.Universe,
    metal_atoms,
    tfsi_reference_atoms,
    tfsi_oxygen_atoms,
    tfsi_resid_to_ref_local: Mapping[int, int],
    tfsi_resid_to_ref_atom_id: Mapping[int, int],
    identity: SystemIdentity,
    config: AnalysisConfig,
    outputs: OutputManager,
    logger: logging.Logger,
    validation: ValidationReport,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:

    start, stop, stride, n_selected = resolve_frame_selection(
        n_frames=len(universe.trajectory),
        start=config.start_frame,
        stop=config.stop_frame,
        stride=config.stride,
    )

    tracker = PairTracker()
    frame_summaries: List[Dict[str, Any]] = []

    n_frames_processed = 0
    total_contacts = 0
    total_pairs = 0
    max_pairs_in_frame = 0
    max_distance_seen = 0.0
    min_distance_seen = math.inf
    duplicate_pair_rows = 0
    invalid_distance_rows = 0
    invalid_oxygen_count_rows = 0
    missing_reference_rows = 0
    all_frames_have_valid_box = True
    previous_processed_frame: Optional[int] = None
    frame_gap_mismatches = 0

    pending_rows: List[Dict[str, Any]] = []
    process_start = time.perf_counter()

    logger.info("-" * 79)
    logger.info("Beginning trajectory pair detection")
    logger.info("Selected frames    : %d", n_selected)
    logger.info("Progress interval  : every %d frames", config.log_every_n_frames)
    logger.info("Database output    : %s", outputs.pair_database)
    logger.info("-" * 79)

    with PairDatabaseWriter(outputs.pair_database, PAIR_COLUMNS) as writer:
        trajectory_slice = universe.trajectory[start:stop:stride]

        for selected_index, ts in enumerate(trajectory_slice, start=1):
            frame = int(ts.frame)
            time_ps = frame_time_ps(frame, config.frame_interval_ps)
            trajectory_timestep = get_trajectory_timestep(ts)

            if previous_processed_frame is not None:
                if frame - previous_processed_frame != stride:
                    frame_gap_mismatches += 1
            previous_processed_frame = frame

            dimensions = np.asarray(ts.dimensions, dtype=np.float64)
            if (
                dimensions.size < 6
                or np.any(~np.isfinite(dimensions))
                or np.any(dimensions[:3] <= 0)
            ):
                all_frames_have_valid_box = False
                raise ValueError(
                    f"Invalid periodic box at frame {frame}: {dimensions}"
                )

            pair_records, n_contacts = detect_frame_pairs(
                metal_atoms=metal_atoms,
                tfsi_oxygen_atoms=tfsi_oxygen_atoms,
                cutoff_A=config.metal_oxygen_cutoff_A,
                box=dimensions,
            )

            pair_ids = [
                f"M{record['metal_id']}_T{record['tfsi_resid']}"
                for record in pair_records
            ]

            if len(pair_ids) != len(set(pair_ids)):
                duplicate_pair_rows += len(pair_ids) - len(set(pair_ids))

            previous_active_ids = set(tracker.active)
            active_states, n_new_events, n_ended_events = tracker.update(
                current_pair_ids=pair_ids,
                frame=frame,
                time_ps=time_ps,
            )

            metal_positions = np.asarray(
                metal_atoms.positions,
                dtype=np.float64,
            )
            tfsi_reference_positions = np.asarray(
                tfsi_reference_atoms.positions,
                dtype=np.float64,
            )

            frame_rows: List[Dict[str, Any]] = []
            metal_pair_counts: Dict[int, int] = {}
            coordinated_tfsi_resids = set()
            per_pair_oxygen_counts: List[int] = []
            per_pair_min_distances: List[float] = []

            for record in pair_records:
                metal_id = int(record["metal_id"])
                tfsi_resid = int(record["tfsi_resid"])
                pair_id = f"M{metal_id}_T{tfsi_resid}"
                state = active_states[pair_id]

                continuing_pair = pair_id in previous_active_ids
                new_pair_event = not continuing_pair

                ref_local_index = tfsi_resid_to_ref_local.get(tfsi_resid)
                ref_atom_id = tfsi_resid_to_ref_atom_id.get(tfsi_resid)

                if ref_local_index is None or ref_atom_id is None:
                    missing_reference_rows += 1
                    raise ValueError(
                        f"No TFSI reference atom mapping for residue {tfsi_resid}."
                    )

                m_local = int(record["metal_local_index"])
                metal_xyz = metal_positions[m_local]
                tfsi_xyz = tfsi_reference_positions[ref_local_index]

                min_distance = float(record["min_distance_A"])
                mean_distance = float(record["mean_distance_A"])
                n_oxygens = int(record["n_coordinating_oxygens"])

                if min_distance > config.metal_oxygen_cutoff_A + 1.0e-6:
                    invalid_distance_rows += 1
                if n_oxygens < 1:
                    invalid_oxygen_count_rows += 1

                min_distance_seen = min(min_distance_seen, min_distance)
                max_distance_seen = max(max_distance_seen, min_distance)

                oxygen_ids_string = (
                    ";".join(str(x) for x in record["oxygen_ids"])
                    if config.save_oxygen_ids
                    else ""
                )

                row = {
                    "system": identity.system,
                    "metal_species": identity.metal_species,
                    "composition": identity.composition,
                    "temperature_K": format_float(identity.temperature_K),
                    "frame": frame,
                    "trajectory_timestep": (
                        "" if trajectory_timestep is None else trajectory_timestep
                    ),
                    "time_ps": format_float(time_ps),
                    "pair_id": pair_id,
                    "pair_instance_id": state.pair_instance_id,
                    "metal_id": metal_id,
                    "metal_resid": int(record["metal_resid"]),
                    "tfsi_resid": tfsi_resid,
                    "tfsi_reference_atom_id": int(ref_atom_id),
                    "n_coordinating_oxygens": n_oxygens,
                    "min_metal_oxygen_distance_A": format_float(min_distance),
                    "mean_metal_oxygen_distance_A": format_float(mean_distance),
                    "coordinating_oxygen_ids": oxygen_ids_string,
                    "continuing_pair": int(continuing_pair),
                    "new_pair_event": int(new_pair_event),
                    "pair_age_frames": int(state.age_frames),
                    "pair_age_ps": format_float(
                        (state.age_frames - 1)
                        * config.frame_interval_ps
                        * config.stride
                    ),
                    "pair_event_start_frame": int(state.start_frame),
                    "pair_event_start_time_ps": format_float(
                        state.start_time_ps
                    ),
                    "metal_x_A": format_float(metal_xyz[0]),
                    "metal_y_A": format_float(metal_xyz[1]),
                    "metal_z_A": format_float(metal_xyz[2]),
                    "tfsi_x_A": format_float(tfsi_xyz[0]),
                    "tfsi_y_A": format_float(tfsi_xyz[1]),
                    "tfsi_z_A": format_float(tfsi_xyz[2]),
                    "box_a_A": format_float(dimensions[0]),
                    "box_b_A": format_float(dimensions[1]),
                    "box_c_A": format_float(dimensions[2]),
                    "box_alpha_deg": format_float(dimensions[3]),
                    "box_beta_deg": format_float(dimensions[4]),
                    "box_gamma_deg": format_float(dimensions[5]),
                }

                frame_rows.append(row)
                metal_pair_counts[metal_id] = metal_pair_counts.get(metal_id, 0) + 1
                coordinated_tfsi_resids.add(tfsi_resid)
                per_pair_oxygen_counts.append(n_oxygens)
                per_pair_min_distances.append(min_distance)

            pending_rows.extend(frame_rows)

            if selected_index % config.flush_every_n_frames == 0:
                writer.write_rows(pending_rows)
                pending_rows.clear()
                writer.flush()

            n_pairs = len(pair_records)
            n_coordinated_metals = len(metal_pair_counts)
            n_coordinated_tfsi = len(coordinated_tfsi_resids)
            n_metals = int(metal_atoms.n_atoms)
            n_tfsi_residues = int(len(tfsi_resid_to_ref_local))

            max_pairs_for_one_metal = (
                max(metal_pair_counts.values()) if metal_pair_counts else 0
            )

            frame_summary = {
                "system": identity.system,
                "metal_species": identity.metal_species,
                "composition": identity.composition,
                "temperature_K": identity.temperature_K,
                "frame": frame,
                "trajectory_timestep": trajectory_timestep,
                "time_ps": time_ps,
                "n_metals": n_metals,
                "n_tfsi_residues": n_tfsi_residues,
                "n_metal_oxygen_contacts": n_contacts,
                "n_unique_metal_tfsi_pairs": n_pairs,
                "n_coordinated_metals": n_coordinated_metals,
                "fraction_coordinated_metals": (
                    n_coordinated_metals / n_metals if n_metals else np.nan
                ),
                "n_coordinated_tfsi": n_coordinated_tfsi,
                "fraction_coordinated_tfsi": (
                    n_coordinated_tfsi / n_tfsi_residues
                    if n_tfsi_residues
                    else np.nan
                ),
                "mean_pairs_per_metal_all": (
                    n_pairs / n_metals if n_metals else np.nan
                ),
                "mean_pairs_per_coordinated_metal": (
                    n_pairs / n_coordinated_metals
                    if n_coordinated_metals
                    else 0.0
                ),
                "max_pairs_for_one_metal": max_pairs_for_one_metal,
                "mean_coordinating_oxygens_per_pair": (
                    float(np.mean(per_pair_oxygen_counts))
                    if per_pair_oxygen_counts
                    else 0.0
                ),
                "max_coordinating_oxygens_per_pair": (
                    int(max(per_pair_oxygen_counts))
                    if per_pair_oxygen_counts
                    else 0
                ),
                "mean_min_pair_distance_A": (
                    float(np.mean(per_pair_min_distances))
                    if per_pair_min_distances
                    else np.nan
                ),
                "n_continuing_pairs": int(n_pairs - n_new_events),
                "n_new_pair_events": int(n_new_events),
                "n_ended_pair_events": int(n_ended_events),
                "box_volume_A3": float(ts.volume),
            }
            frame_summaries.append(frame_summary)

            n_frames_processed += 1
            total_contacts += n_contacts
            total_pairs += n_pairs
            max_pairs_in_frame = max(max_pairs_in_frame, n_pairs)

            if (
                selected_index == 1
                or selected_index % config.log_every_n_frames == 0
                or selected_index == n_selected
            ):
                elapsed = time.perf_counter() - process_start
                rate = selected_index / elapsed if elapsed > 0 else np.nan
                remaining = n_selected - selected_index
                eta = remaining / rate if rate > 0 else np.nan
                percent = 100.0 * selected_index / n_selected

                logger.info(
                    "Progress %6.2f%% | %d/%d frames | frame=%d | "
                    "pairs=%d | contacts=%d | elapsed=%s | ETA=%s",
                    percent,                    selected_index,
                    n_selected,
                    frame,
                    n_pairs,
                    n_contacts,
                    format_eta(elapsed),
                    format_eta(eta),
                )

        writer.write_rows(pending_rows)
        pending_rows.clear()
        writer.flush()
        rows_written = writer.rows_written

    remaining_events_closed = tracker.close_remaining_events()
    processing_seconds = time.perf_counter() - process_start

    frame_summary_df = pd.DataFrame(
        frame_summaries,
        columns=FRAME_SUMMARY_COLUMNS,
    )
    frame_summary_df.to_csv(
        outputs.frame_summary,
        index=False,
        float_format="%.8g",
    )

    validation.add_check(
        "all_selected_frames_processed",
        n_frames_processed == n_selected,
    )
    validation.add_check("frame_sequence_valid", frame_gap_mismatches == 0)
    validation.add_check(
        "pair_rows_equal_streamed_rows",
        total_pairs == rows_written,
    )
    validation.add_check(
        "no_duplicate_pairs_within_frames",
        duplicate_pair_rows == 0,
    )
    validation.add_check(
        "all_pair_distances_within_cutoff",
        invalid_distance_rows == 0,
    )
    validation.add_check(
        "all_pairs_have_at_least_one_oxygen",
        invalid_oxygen_count_rows == 0,
    )
    validation.add_check(
        "all_pairs_have_tfsi_reference_atom",
        missing_reference_rows == 0,
    )
    validation.add_check(
        "periodic_box_valid_for_all_frames",
        all_frames_have_valid_box,
    )
    validation.add_check(
        "frame_summary_row_count_valid",
        len(frame_summary_df) == n_selected,
    )

    validation.add_metric("n_frames_processed", n_frames_processed)
    validation.add_metric("n_pair_rows_written", rows_written)
    validation.add_metric("total_metal_oxygen_contacts", total_contacts)
    validation.add_metric("mean_pairs_per_frame", total_pairs / n_frames_processed)
    validation.add_metric("max_pairs_in_one_frame", max_pairs_in_frame)
    validation.add_metric(
        "minimum_pair_distance_A",
        None if not np.isfinite(min_distance_seen) else min_distance_seen,
    )
    validation.add_metric("maximum_pair_distance_A", max_distance_seen)
    validation.add_metric(
        "total_pair_events_started",
        tracker.total_started_events,
    )
    validation.add_metric(
        "total_pair_events_ended_including_final_frame",
        tracker.total_ended_events,
    )
    validation.add_metric(
        "events_closed_at_final_frame",
        remaining_events_closed,
    )
    validation.add_metric(
        "maximum_consecutive_pair_age_frames",
        tracker.max_pair_age_frames,
    )
    validation.add_metric(
        "maximum_consecutive_pair_age_ps",
        max(0, tracker.max_pair_age_frames - 1)
        * config.frame_interval_ps
        * config.stride,
    )
    validation.add_metric("processing_seconds", processing_seconds)
    validation.add_metric(
        "processing_frames_per_second",
        n_frames_processed / processing_seconds
        if processing_seconds > 0
        else None,
    )

    processing_summary = {
        "n_frames_processed": n_frames_processed,
        "n_pair_rows_written": rows_written,
        "total_metal_oxygen_contacts": total_contacts,
        "mean_pairs_per_frame": total_pairs / n_frames_processed,
        "maximum_pairs_in_one_frame": max_pairs_in_frame,
        "total_pair_events_started": tracker.total_started_events,
        "maximum_consecutive_pair_age_frames": tracker.max_pair_age_frames,
        "processing_seconds": processing_seconds,
        "processing_frames_per_second": (
            n_frames_processed / processing_seconds
            if processing_seconds > 0
            else None
        ),
    }

    return frame_summaries, processing_summary

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a frame-resolved metal–TFSI coordination-pair database "
            "for a generic metal-ion polymer electrolyte."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--topology",
        type=Path,
        default=Path(DEFAULT_TOPOLOGY),
        help="LAMMPS data topology.",
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=Path(DEFAULT_TRAJECTORY),
        help="LAMMPS dump trajectory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help="Output directory.",
    )
    parser.add_argument(
        "--metal",
        default="metal",
        help="Generic metal-ion label stored in output metadata.",
    )
    parser.add_argument(
        "--composition",
        choices=["20_5", "15_10", "15_15", "5_20"],
        default=None,
        help="Override automatic composition detection.",
    )
    parser.add_argument(
        "--temperature-K",
        type=float,
        default=None,
        help="Simulation temperature in kelvin; otherwise detected or set to 380 K.",
    )
    parser.add_argument(
        "--frame-interval-ps",
        type=float,
        default=DEFAULT_FRAME_INTERVAL_PS,
        help="Time between saved trajectory frames.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=DEFAULT_START_FRAME,
        help="First frame to process, inclusive.",
    )
    parser.add_argument(
        "--stop-frame",
        type=int,
        default=DEFAULT_STOP_FRAME,
        help="Last frame boundary, exclusive.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_STRIDE,
        help="Frame stride.",
    )
    parser.add_argument(
        "--metal-type",
        type=int,
        default=METAL_ATOM_TYPE,
        help="LAMMPS atom type for metal ions.",
    )
    parser.add_argument(
        "--tfsi-reference-type",
        type=int,
        default=TFSI_REFERENCE_ATOM_TYPE,
        help="LAMMPS atom type for the TFSI reference/COM atom.",
    )
    parser.add_argument(
        "--tfsi-oxygen-type",
        type=int,
        default=TFSI_OXYGEN_ATOM_TYPE,
        help="LAMMPS atom type for TFSI oxygen.",
    )
    parser.add_argument(
        "--cutoff-A",
        type=float,
        default=None,
        help="Override automatic metal–oxygen cutoff.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=LOG_EVERY_N_FRAMES,
        help="Print progress every N processed frames.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=FLUSH_EVERY_N_FRAMES,
        help="Flush pair rows to gzip every N processed frames.",
    )
    parser.add_argument(
        "--do-not-save-oxygen-ids",
        action="store_true",
        help="Do not store coordinating TFSI oxygen IDs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output files.",
    )

    return parser

def config_from_args(
    args: argparse.Namespace,
) -> Tuple[AnalysisConfig, SystemIdentity]:
    topology = args.topology.resolve()
    trajectory = args.trajectory.resolve()

    temperature = (
        float(args.temperature_K)
        if args.temperature_K is not None
        else detect_temperature_from_path(topology, trajectory, default=380.0)
    )

    identity = build_system_identity(
        topology=topology,
        trajectory=trajectory,
        metal_override=args.metal,
        composition_override=args.composition,
        temperature_K=temperature,
    )

    cutoff = float(args.cutoff_A)

    config = AnalysisConfig(
        topology=topology,
        trajectory=trajectory,
        output_dir=args.output_dir.resolve(),
        frame_interval_ps=float(args.frame_interval_ps),
        start_frame=int(args.start_frame),
        stop_frame=args.stop_frame,
        stride=int(args.stride),
        metal_atom_type=int(args.metal_type),
        tfsi_reference_atom_type=int(args.tfsi_reference_type),
        tfsi_oxygen_atom_type=int(args.tfsi_oxygen_type),
        metal_oxygen_cutoff_A=cutoff,
        overwrite=bool(args.overwrite),
        save_oxygen_ids=not bool(args.do_not_save_oxygen_ids),
        log_every_n_frames=int(args.log_every),
        flush_every_n_frames=int(args.flush_every),
    )
    config.validate()
    return config, identity

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config, identity = config_from_args(args)

    outputs = OutputManager(config.output_dir, overwrite=config.overwrite)
    outputs.prepare()
    logger = configure_logging(outputs.log_file)

    overall_start = time.perf_counter()

    logger.info("=" * 79)
    logger.info("MODULE 5E1: METAL–TFSI PAIR DATABASE BUILDER — PARTS 1 + 2")
    logger.info("=" * 79)
    logger.info("System             : %s", identity.system)
    logger.info("Metal species      : %s", identity.metal_species)
    logger.info("Composition        : %s", identity.composition)
    logger.info("Temperature        : %.2f K", identity.temperature_K)
    logger.info("Topology           : %s", config.topology)
    logger.info("Trajectory         : %s", config.trajectory)
    logger.info("Output directory   : %s", outputs.root)
    logger.info("Metal atom type    : %d", config.metal_atom_type)
    logger.info("TFSI ref atom type : %d", config.tfsi_reference_atom_type)
    logger.info("TFSI oxygen type   : %d", config.tfsi_oxygen_atom_type)
    logger.info("Metal/O cutoff     : %.3f Å", config.metal_oxygen_cutoff_A)
    logger.info("Frame interval     : %.3f ps", config.frame_interval_ps)
    logger.info(
        "Frame range        : start=%s stop=%s stride=%s",
        config.start_frame,
        config.stop_frame,
        config.stride,
    )

    logger.info("Loading topology and trajectory index...")
    load_start = time.perf_counter()
    universe = load_universe(config.topology, config.trajectory)
    logger.info(
        "Topology/trajectory loaded in %s",
        format_eta(time.perf_counter() - load_start),
    )

    metal_atoms = select_atoms_by_type(
        universe,
        config.metal_atom_type,
        "metal",
    )
    tfsi_reference_atoms = select_atoms_by_type(
        universe,
        config.tfsi_reference_atom_type,
        "TFSI reference",
    )
    tfsi_oxygen_atoms = select_atoms_by_type(
        universe,
        config.tfsi_oxygen_atom_type,
        "TFSI oxygen",
    )

    tfsi_resid_to_ref_local, tfsi_resid_to_ref_atom_id = (
        build_tfsi_reference_mapping(
            tfsi_reference_atoms,
            tfsi_oxygen_atoms,
        )
    )

    topology_summary = summarize_topology(
        universe,
        metal_atoms,
        tfsi_reference_atoms,
        tfsi_oxygen_atoms,
    )
    box_summary = validate_initial_box(universe)

    start, stop, stride, n_selected = resolve_frame_selection(
        n_frames=len(universe.trajectory),
        start=config.start_frame,
        stop=config.stop_frame,
        stride=config.stride,
    )

    validation = ValidationReport()
    validation.add_check("topology_loaded", True)
    validation.add_check("trajectory_loaded", True)
    validation.add_check("metal_selection_nonempty", metal_atoms.n_atoms > 0)
    validation.add_check(
        "tfsi_reference_selection_nonempty",
        tfsi_reference_atoms.n_atoms > 0,
    )
    validation.add_check(
        "tfsi_oxygen_selection_nonempty",
        tfsi_oxygen_atoms.n_atoms > 0,
    )
    validation.add_check(
        "one_tfsi_reference_per_tfsi_residue",
        topology_summary["n_tfsi_reference_atoms"]
        == topology_summary["n_tfsi_residues"],
    )
    validation.add_check(
        "tfsi_residue_mapping_available",
        topology_summary["n_tfsi_residues"] > 0,
    )
    validation.add_check("initial_periodic_box_valid", True)
    validation.add_check("frame_range_nonempty", n_selected > 0)

    validation.add_metric("n_trajectory_frames", len(universe.trajectory))
    validation.add_metric("n_selected_frames", n_selected)
    validation.add_metric("first_selected_frame", start)
    validation.add_metric("last_selected_frame", start + (n_selected - 1) * stride)
    validation.add_metric(
        "selected_duration_ns",
        (
            frame_time_ps(
                start + (n_selected - 1) * stride,
                config.frame_interval_ps,
            )
            - frame_time_ps(start, config.frame_interval_ps)
        )
        / 1000.0,
    )

    logger.info("Total atoms        : %d", topology_summary["n_total_atoms"])
    logger.info("Metal atoms        : %d", topology_summary["n_metal_atoms"])
    logger.info(
        "TFSI ref atoms     : %d",
        topology_summary["n_tfsi_reference_atoms"],
    )
    logger.info(
        "TFSI oxygen atoms  : %d",
        topology_summary["n_tfsi_oxygen_atoms"],
    )
    logger.info("TFSI residues      : %d", topology_summary["n_tfsi_residues"])
    logger.info("Trajectory frames  : %d", len(universe.trajectory))
    logger.info("Selected frames    : %d", n_selected)
    logger.info("Initial volume     : %.3f Å³", box_summary["box_volume_A3"])

    metadata = {
        "module": "05E1_build_pair_database",
        "implementation_stage": "parts_1_and_2_pair_database_generation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "identity": asdict(identity),
        "configuration": asdict(config),
        "pair_definition": {
            "criterion": (
                "At least one TFSI oxygen lies within the metal–oxygen cutoff."
            ),
            "cutoff_A": config.metal_oxygen_cutoff_A,
            "pair_age_definition": (
                "Consecutive presence across processed frames; age resets after "
                "any absence."
            ),
            "coordinate_storage": (
                "Wrapped metal and type-7 TFSI reference coordinates."
            ),
        },
        "topology_summary": topology_summary,
        "initial_box": box_summary,
        "selected_frame_summary": {
            "n_frames": n_selected,
            "first_frame": start,
            "last_frame": start + (n_selected - 1) * stride,
            "stride": stride,
            "first_time_ps": frame_time_ps(
                start,
                config.frame_interval_ps,
            ),
            "last_time_ps": frame_time_ps(
                start + (n_selected - 1) * stride,
                config.frame_interval_ps,
            ),
        },
    }
    write_json(outputs.metadata, metadata)

    try:
        _, processing_summary = process_trajectory(
            universe=universe,
            metal_atoms=metal_atoms,
            tfsi_reference_atoms=tfsi_reference_atoms,
            tfsi_oxygen_atoms=tfsi_oxygen_atoms,
            tfsi_resid_to_ref_local=tfsi_resid_to_ref_local,
            tfsi_resid_to_ref_atom_id=tfsi_resid_to_ref_atom_id,
            identity=identity,
            config=config,
            outputs=outputs,
            logger=logger,
            validation=validation,
        )
    except Exception as exc:
        validation.error(f"{type(exc).__name__}: {exc}")
        write_json(outputs.validation_report, validation.to_dict())
        logger.exception("Module 5E1 failed during trajectory processing.")
        raise

    metadata["processing_summary"] = processing_summary
    metadata["completed_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["overall_runtime_seconds"] = time.perf_counter() - overall_start
    write_json(outputs.metadata, metadata)
    write_json(outputs.validation_report, validation.to_dict())

    logger.info("=" * 79)
    logger.info("MODULE 5E1 PARTS 1 + 2 COMPLETED")
    logger.info("=" * 79)
    logger.info("Pair rows written  : %d", processing_summary["n_pair_rows_written"])
    logger.info(
        "Mean pairs/frame   : %.6f",
        processing_summary["mean_pairs_per_frame"],
    )
    logger.info(
        "Pair events        : %d",
        processing_summary["total_pair_events_started"],
    )
    logger.info(
        "Maximum pair age   : %d frames",
        processing_summary["maximum_consecutive_pair_age_frames"],
    )
    logger.info(
        "Processing speed   : %.4f frames/s",
        processing_summary["processing_frames_per_second"],
    )
    logger.info("Validation status  : %s", validation.status)
    logger.info("Pair database      : %s", outputs.pair_database)
    logger.info("Frame summary      : %s", outputs.frame_summary)
    logger.info("Validation report  : %s", outputs.validation_report)
    logger.info("Metadata           : %s", outputs.metadata)
    logger.info(
        "Overall runtime    : %s",
        format_eta(time.perf_counter() - overall_start),
    )
    logger.info(
        "Part 3 will add final cross-checks, event summaries, and figures."
    )

if __name__ == "__main__":
    main()