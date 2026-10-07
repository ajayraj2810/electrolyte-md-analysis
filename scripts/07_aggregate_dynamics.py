#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import logging
import math
import sys
import time
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_INPUT_DIR = "05E1_pair_database"
DEFAULT_OUTPUT_SUBDIR = "05E3_bridge_network"
DEFAULT_CHUNK_SIZE = 200_000
DEFAULT_MAX_LAG_NS = 20.0
DEFAULT_N_LOG_LAGS = 50
DEFAULT_LINEAR_LAGS = 20
DEFAULT_LINEAGE_JACCARD = 0.50
DEFAULT_MAX_INTERNAL_NODES = 32
DEFAULT_BOOTSTRAP_REPLICATES = 100
DEFAULT_BOOTSTRAP_SEED = 20260720
DEFAULT_MIN_EVENTS = 20
DEFAULT_MIN_ORIGINS = 200
DEFAULT_MAX_STEP_A = 25.0
DEFAULT_MAX_CLOSURE_A = 1.0e-3
DEFAULT_FIGURE_DPI = 300

DEFAULT_METAL_MASS_G_MOL = 40.0
DEFAULT_METAL_CHARGE_NUMBER = 1
TFSI_MASS = 280.14

PAIR_COLS = [
    "frame", "time_ps", "metal_id", "metal_resid", "tfsi_resid",
    "tfsi_reference_atom_id",
    "metal_x_A", "metal_y_A", "metal_z_A",
    "tfsi_x_A", "tfsi_y_A", "tfsi_z_A",
    "box_a_A", "box_b_A", "box_c_A",
    "box_alpha_deg", "box_beta_deg", "box_gamma_deg",
]

BRIDGE_LAG_COLS = [
    "system", "metal_species", "composition", "temperature_K",
    "bridge_instance_id", "bridge_id", "tfsi_resid",
    "metal1_id", "metal2_id", "completed_event",
    "left_censored", "right_censored", "persistence_class",
    "occupancy_time_ps", "occupancy_time_ns", "event_n_frames",
    "lag_frames", "lag_ps", "lag_ns", "n_time_origins",
    "sum_m1_sq", "sum_m2_sq", "sum_tfsi_sq",
    "sum_relative_sq", "sum_midpoint_sq", "sum_bridge_com_sq",
    "sum_tfsi_relative_midpoint_sq", "sum_metal_dot",
    "sum_m1_tfsi_dot", "sum_m2_tfsi_dot",
    "sum_directional_cosine", "n_directional_cosine",
]

CLUSTER_LAG_COLS = [
    "system", "metal_species", "composition", "temperature_K",
    "aggregate_instance_id", "aggregate_key", "completed_event",
    "left_censored", "right_censored", "persistence_class",
    "size_class", "metal_count_class", "n_metals", "n_tfsi",
    "n_total_ions", "net_charge_e", "occupancy_time_ps",
    "occupancy_time_ns", "event_n_frames", "lag_frames",
    "lag_ps", "lag_ns", "n_time_origins",
    "sum_com_sq", "sum_rg_change_sq",
    "sum_internal_rearrangement", "n_internal_origins",
]

@dataclass(frozen=True)
class Config:
    input_dir: Path
    output_dir: Path
    chunk_size: int
    max_lag_ns: float
    n_log_lags: int
    linear_lags: int
    lineage_jaccard: float
    max_internal_nodes: int
    bootstrap_replicates: int
    bootstrap_seed: int
    min_events: int
    min_origins: int
    max_step_A: float
    max_closure_A: float
    figure_dpi: int
    overwrite: bool

    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.max_lag_ns <= 0:
            raise ValueError("max_lag_ns must be positive")
        if self.n_log_lags < 5 or self.linear_lags < 1:
            raise ValueError("invalid lag-grid settings")
        if not 0 < self.lineage_jaccard <= 1:
            raise ValueError("lineage_jaccard must be in (0, 1]")
        if self.max_internal_nodes < 2:
            raise ValueError("max_internal_nodes must be at least 2")
        if self.bootstrap_replicates < 0:
            raise ValueError("bootstrap_replicates cannot be negative")
        if self.min_events < 1 or self.min_origins < 1:
            raise ValueError("support thresholds must be positive")

@dataclass(frozen=True)
class Metadata:
    system: str
    metal_species: str
    composition: str
    temperature_K: float
    frame_dt_ps: float
    stride: int
    effective_dt_ps: float
    first_frame: int
    last_frame: int
    n_frames: int
    n_metals: int
    n_tfsi: int
    metal_mass: float
    metal_charge_number: int
    tfsi_mass: float

@dataclass
class Validation:
    status: str = "NOT_RUN"
    checks: Dict[str, bool] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def check(self, name: str, value: bool) -> None:
        self.checks[name] = bool(value)

    def metric(self, name: str, value: Any) -> None:
        self.metrics[name] = json_safe(value)

    def warn(self, text: str) -> None:
        self.warnings.append(str(text))

    def finalize(self) -> None:
        if self.errors or (self.checks and not all(self.checks.values())):
            self.status = "FAILED"
        elif self.warnings:
            self.status = "PASSED_WITH_WARNINGS"
        elif self.checks:
            self.status = "PASSED"
        else:
            self.status = "NOT_RUN"

    def as_dict(self) -> Dict[str, Any]:
        self.finalize()
        return json_safe(asdict(self))

class Paths:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.figures = root / "figures"
        self.frame_summary = root / "frame_network_summary.csv"
        self.bridge_events = root / "bridge_event_summary.csv"
        self.cluster_events = root / "aggregate_event_summary.csv"
        self.lineages = root / "aggregate_lineage_summary.csv"
        self.bridge_lag = root / "bridge_event_lag_statistics.csv.gz"
        self.cluster_lag = root / "aggregate_event_lag_statistics.csv.gz"
        self.bridge_all = root / "bridge_comotion_all.csv"
        self.bridge_persistence = root / "bridge_comotion_by_persistence.csv"
        self.cluster_all = root / "aggregate_transport_all.csv"
        self.cluster_size = root / "aggregate_transport_by_size.csv"
        self.cluster_metals = root / "aggregate_transport_by_metal_count.csv"
        self.cluster_persistence = root / "aggregate_transport_by_persistence.csv"
        self.size_distribution = root / "aggregate_size_distribution.csv"
        self.lag_grid = root / "lag_grid.csv"
        self.validation = root / "validation_report.json"
        self.summary = root / "summary.json"
        self.log = root / "05E3_bridge_network.log"

    def prepare(self, overwrite: bool) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.figures.mkdir(parents=True, exist_ok=True)
        outputs = [
            self.frame_summary, self.bridge_events, self.cluster_events,
            self.lineages, self.bridge_lag, self.cluster_lag,
            self.bridge_all, self.bridge_persistence, self.cluster_all,
            self.cluster_size, self.cluster_metals, self.cluster_persistence,
            self.size_distribution, self.lag_grid, self.validation,
            self.summary, self.log,
        ]
        existing = [p for p in outputs if p.exists()]
        if existing and not overwrite:
            raise FileExistsError(
                "05E3 outputs exist. Use --overwrite.\n"
                + "\n".join(str(p) for p in existing)
            )
        if overwrite:
            for p in existing:
                p.unlink()

class GzipWriter:
    def __init__(self, path: Path, columns: Sequence[str]) -> None:
        self.path = path
        self.columns = list(columns)
        self.handle = None
        self.writer = None
        self.rows = 0

    def __enter__(self) -> "GzipWriter":
        self.handle = gzip.open(
            self.path, "wt", encoding="utf-8", newline="", compresslevel=6
        )
        self.writer = csv.DictWriter(
            self.handle, fieldnames=self.columns, extrasaction="raise"
        )
        self.writer.writeheader()
        return self

    def write_rows(self, rows: Sequence[Mapping[str, Any]]) -> None:
        if not rows:
            return
        self.writer.writerows(rows)
        self.rows += len(rows)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.handle is not None:
            self.handle.close()

def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "__dataclass_fields__"):
        return json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value

def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2)
        handle.write("\n")
    tmp.replace(path)

def configure_logging(path: Path) -> logging.Logger:
    logger = logging.getLogger("05E3")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    file_handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def format_runtime(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "unknown"
    value = int(round(seconds))
    hours, rem = divmod(value, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    return f"{minutes:d}m {secs:02d}s"

def require_columns(columns: Iterable[str], required: Sequence[str], label: str) -> None:
    missing = [c for c in required if c not in set(columns)]
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")

def classify_persistence(time_ps: float) -> str:
    if time_ps <= 50:
        return "transient_10_50_ps"
    if time_ps <= 500:
        return "short_50_500_ps"
    if time_ps <= 5000:
        return "intermediate_0.5_5_ns"
    return "persistent_gt_5_ns"

def classify_size(n: int) -> str:
    if n == 2:
        return "pair_cluster"
    if n <= 4:
        return "small_3_4"
    if n <= 8:
        return "medium_5_8"
    if n <= 16:
        return "large_9_16"
    return "network_gt_16"

def classify_metal_count(n: int) -> str:
    if n == 1:
        return "one_metal"
    if n == 2:
        return "two_metals"
    return "three_plus_metals"

def membership_key(metals: Sequence[int], tfsis: Sequence[int]) -> str:
    return (
        "M:" + ",".join(map(str, metals))
        + "|T:" + ",".join(map(str, tfsis))
    )

def membership_hash(key: str) -> str:
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:14]

def build_lag_grid(meta: Metadata, cfg: Config) -> np.ndarray:
    max_frames = min(
        meta.n_frames - 1,
        int(math.floor(cfg.max_lag_ns * 1000 / meta.effective_dt_ps)),
    )
    linear_max = min(cfg.linear_lags, max_frames)
    linear = np.arange(1, linear_max + 1, dtype=np.int32)
    if max_frames <= linear_max:
        return linear
    log = np.unique(
        np.rint(
            np.logspace(
                np.log10(linear_max + 1),
                np.log10(max_frames),
                cfg.n_log_lags,
            )
        ).astype(np.int32)
    )
    return np.unique(np.concatenate([linear, log[log > linear_max]])).astype(np.int32)

def box_matrix(dim: np.ndarray) -> np.ndarray:
    a, b, c, alpha_d, beta_d, gamma_d = map(float, dim)
    alpha, beta, gamma = map(math.radians, (alpha_d, beta_d, gamma_d))
    sg = math.sin(gamma)
    if min(a, b, c) <= 0 or abs(sg) < 1e-12:
        raise ValueError("Invalid periodic box")
    av = np.array([a, 0.0, 0.0])
    bv = np.array([b * math.cos(gamma), b * sg, 0.0])
    cx = c * math.cos(beta)
    cy = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / sg
    cz = math.sqrt(max(0.0, c * c - cx * cx - cy * cy))
    matrix = np.vstack([av, bv, np.array([cx, cy, cz])])
    if abs(np.linalg.det(matrix)) < 1e-10:
        raise ValueError("Singular periodic box")
    return matrix

def mic(delta: np.ndarray, dim: np.ndarray) -> np.ndarray:
    delta = np.asarray(delta, dtype=float)
    dim = np.asarray(dim, dtype=float)
    if np.allclose(dim[3:6], 90.0, atol=1e-4, rtol=0):
        lengths = dim[:3]
        return delta - lengths * np.rint(delta / lengths)
    matrix = box_matrix(dim)
    frac = delta @ np.linalg.inv(matrix)
    frac -= np.rint(frac)
    return frac @ matrix

def unwrap_series(wrapped: np.ndarray, boxes: np.ndarray) -> Tuple[np.ndarray, float]:
    wrapped = np.asarray(wrapped, dtype=float)
    out = np.empty_like(wrapped)
    if len(wrapped) == 0:
        return out, 0.0
    out[0] = wrapped[0]
    maximum = 0.0
    for i in range(1, len(wrapped)):
        delta = mic(
            wrapped[i] - wrapped[i - 1],
            0.5 * (boxes[i] + boxes[i - 1]),
        )
        maximum = max(maximum, float(np.linalg.norm(delta)))
        out[i] = out[i - 1] + delta
    return out, maximum

class UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}

    def add(self, x: str) -> None:
        if x not in self.parent:
            self.parent[x] = x

    def find(self, x: str) -> str:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        self.add(a)
        self.add(b)
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def components(self) -> List[Set[str]]:
        groups: Dict[str, Set[str]] = defaultdict(set)
        for x in self.parent:
            groups[self.find(x)].add(x)
        return list(groups.values())

@dataclass
class BridgeFrame:
    bridge_id: str
    tfsi: int
    m1: int
    m2: int
    r1: np.ndarray
    r2: np.ndarray
    rt: np.ndarray
    box: np.ndarray

@dataclass
class ClusterFrame:
    key: str
    metals: Tuple[int, ...]
    tfsis: Tuple[int, ...]
    nodes: Tuple[str, ...]
    positions: Optional[np.ndarray]
    masses: Optional[np.ndarray]
    com: np.ndarray
    rg: float
    closure: float
    is_percolating: bool
    winding_rank: int
    winding_axes: str
    winding_vectors: str
    n_bridge_tfsi: int
    n_bridge_edges: int
    charge: int
    size_class: str
    metal_class: str
    box: np.ndarray

@dataclass
class BridgeBuffer:
    instance: str
    bridge_id: str
    tfsi: int
    m1: int
    m2: int
    start_frame: int
    frames: List[int] = field(default_factory=list)
    times: List[float] = field(default_factory=list)
    r1: List[np.ndarray] = field(default_factory=list)
    r2: List[np.ndarray] = field(default_factory=list)
    rt: List[np.ndarray] = field(default_factory=list)
    boxes: List[np.ndarray] = field(default_factory=list)

    def append(self, frame: int, time_ps: float, data: BridgeFrame) -> None:
        self.frames.append(int(frame))
        self.times.append(float(time_ps))
        self.r1.append(np.asarray(data.r1, dtype=float))
        self.r2.append(np.asarray(data.r2, dtype=float))
        self.rt.append(np.asarray(data.rt, dtype=float))
        self.boxes.append(np.asarray(data.box, dtype=float))

@dataclass
class ClusterBuffer:
    instance: str
    key: str
    metals: Tuple[int, ...]
    tfsis: Tuple[int, ...]
    nodes: Tuple[str, ...]
    masses: Optional[np.ndarray]
    size_class: str
    metal_class: str
    charge: int
    start_frame: int
    frames: List[int] = field(default_factory=list)
    times: List[float] = field(default_factory=list)
    com: List[np.ndarray] = field(default_factory=list)
    rg: List[float] = field(default_factory=list)
    internal: List[Optional[np.ndarray]] = field(default_factory=list)
    boxes: List[np.ndarray] = field(default_factory=list)
    closure: List[float] = field(default_factory=list)
    percolating: List[bool] = field(default_factory=list)
    winding_rank: List[int] = field(default_factory=list)
    winding_axes: List[str] = field(default_factory=list)

    def append(self, frame: int, time_ps: float, data: ClusterFrame) -> None:
        self.frames.append(int(frame))
        self.times.append(float(time_ps))
        self.com.append(np.asarray(data.com, dtype=float))
        self.rg.append(float(data.rg))
        self.internal.append(
            None if data.positions is None
            else np.asarray(data.positions, dtype=float) - np.asarray(data.com, dtype=float)
        )
        self.boxes.append(np.asarray(data.box, dtype=float))
        self.closure.append(float(data.closure))
        self.percolating.append(bool(data.is_percolating))
        self.winding_rank.append(int(data.winding_rank))
        self.winding_axes.append(str(data.winding_axes))

@dataclass
class LineageStats:
    lineage_id: str
    first_frame: int
    last_frame: int
    n_frames: int = 0
    sum_size: int = 0
    min_size: int = 10**9
    max_size: int = 0
    max_metals: int = 0
    max_tfsi: int = 0
    merges: int = 0
    splits: int = 0
    jaccard_sum: float = 0.0
    jaccard_n: int = 0

    def update(
        self,
        frame: int,
        size: int,
        n_metals: int,
        n_tfsi: int,
        jaccard: float,
        merge_count: int,
        split: int,
    ) -> None:
        self.last_frame = int(frame)
        self.n_frames += 1
        self.sum_size += int(size)
        self.min_size = min(self.min_size, int(size))
        self.max_size = max(self.max_size, int(size))
        self.max_metals = max(self.max_metals, int(n_metals))
        self.max_tfsi = max(self.max_tfsi, int(n_tfsi))
        self.merges += int(merge_count > 1)
        self.splits += int(split)
        if np.isfinite(jaccard):
            self.jaccard_sum += float(jaccard)
            self.jaccard_n += 1

def canonical_winding_vector(vector: np.ndarray) -> Tuple[int, int, int]:

    values = np.asarray(vector, dtype=int).copy()
    for value in values:
        if value < 0:
            values *= -1
            break
        if value > 0:
            break
    return tuple(int(value) for value in values)

def reconstruct_component(
    component: Set[str],
    adjacency: Mapping[str, Set[str]],
    wrapped: Mapping[str, np.ndarray],
    box: np.ndarray,
    meta: Metadata,
    max_internal_nodes: int,
) -> Tuple[
    Tuple[str, ...],
    Optional[np.ndarray],
    Optional[np.ndarray],
    np.ndarray,
    float,
    float,
    bool,
    int,
    str,
    str,
]:

    metals = sorted(
        (x for x in component if x.startswith("M")),
        key=lambda x: int(x[1:]),
    )
    tfsis = sorted(
        (x for x in component if x.startswith("T")),
        key=lambda x: int(x[1:]),
    )
    order = tuple(metals + tfsis)
    anchor = order[0]

    matrix = box_matrix(box)
    inverse = np.linalg.inv(matrix)

    placed: Dict[str, np.ndarray] = {
        anchor: np.asarray(wrapped[anchor], dtype=float)
    }
    queue = deque([anchor])
    max_residual = 0.0
    winding_vectors: Set[Tuple[int, int, int]] = set()

    while queue:
        current = queue.popleft()
        current_wrapped = np.asarray(wrapped[current], dtype=float)

        for neighbor in adjacency[current]:
            neighbor_wrapped = np.asarray(wrapped[neighbor], dtype=float)
            edge_delta = mic(neighbor_wrapped - current_wrapped, box)
            candidate = placed[current] + edge_delta

            if neighbor not in placed:
                placed[neighbor] = candidate
                queue.append(neighbor)
            else:

                raw_closure = candidate - placed[neighbor]
                fractional_closure = raw_closure @ inverse
                integer_winding = np.rint(fractional_closure).astype(int)
                residual_fractional = fractional_closure - integer_winding
                residual_cartesian = residual_fractional @ matrix

                residual_norm = float(np.linalg.norm(residual_cartesian))
                max_residual = max(max_residual, residual_norm)

                if np.any(integer_winding != 0):
                    winding_vectors.add(
                        canonical_winding_vector(integer_winding)
                    )

    if len(placed) != len(component):
        raise ValueError(
            "Periodic component reconstruction did not visit all graph nodes."
        )

    positions = np.vstack([placed[x] for x in order])
    masses = np.array(
        [
            meta.metal_mass if x.startswith("M") else meta.tfsi_mass
            for x in order
        ],
        dtype=float,
    )
    total_mass = float(masses.sum())
    com = np.sum(positions * masses[:, None], axis=0) / total_mass
    centered = positions - com
    rg = math.sqrt(
        float(
            np.sum(
                masses * np.einsum("ij,ij->i", centered, centered)
            )
            / total_mass
        )
    )

    ordered_windings = sorted(winding_vectors)
    is_percolating = bool(ordered_windings)
    if ordered_windings:
        winding_array = np.asarray(ordered_windings, dtype=float)
        winding_rank = int(
            np.linalg.matrix_rank(winding_array, tol=1.0e-10)
        )
    else:
        winding_rank = 0

    axis_names = ["x", "y", "z"]
    active_axes = [
        axis_names[axis]
        for axis in range(3)
        if any(vector[axis] != 0 for vector in ordered_windings)
    ]
    winding_axes = "".join(active_axes) if active_axes else "none"
    winding_text = ";".join(
        ",".join(str(value) for value in vector)
        for vector in ordered_windings
    )

    if len(order) <= max_internal_nodes:
        return (
            order,
            positions,
            masses,
            com,
            rg,
            max_residual,
            is_percolating,
            winding_rank,
            winding_axes,
            winding_text,
        )

    return (
        order,
        None,
        None,
        com,
        rg,
        max_residual,
        is_percolating,
        winding_rank,
        winding_axes,
        winding_text,
    )

def frame_graph(
    group: pd.DataFrame,
    meta: Metadata,
    cfg: Config,
) -> Tuple[Dict[str, BridgeFrame], List[ClusterFrame], Dict[str, Any]]:
    frame = int(group["frame"].iloc[0])
    time_ps = float(group["time_ps"].iloc[0])
    if group.duplicated(["metal_id", "tfsi_resid"]).any():
        raise ValueError(f"Duplicate pair edge at frame {frame}")

    box = group[
        ["box_a_A", "box_b_A", "box_c_A", "box_alpha_deg", "box_beta_deg", "box_gamma_deg"]
    ].iloc[0].to_numpy(dtype=float)

    metal_pos: Dict[int, np.ndarray] = {}
    tfsi_pos: Dict[int, np.ndarray] = {}
    metal_to_tfsi: Dict[int, Set[int]] = defaultdict(set)
    tfsi_to_metal: Dict[int, Set[int]] = defaultdict(set)
    adjacency: Dict[str, Set[str]] = defaultdict(set)
    uf = UnionFind()

    for row in group.itertuples(index=False):
        m, t = int(row.metal_id), int(row.tfsi_resid)
        rm = np.array([row.metal_x_A, row.metal_y_A, row.metal_z_A], dtype=float)
        rt = np.array([row.tfsi_x_A, row.tfsi_y_A, row.tfsi_z_A], dtype=float)

        if m in metal_pos and np.linalg.norm(mic(rm - metal_pos[m], box)) > 1e-5:
            raise ValueError(f"Inconsistent metal coordinate: frame {frame}, metal {m}")
        if t in tfsi_pos and np.linalg.norm(mic(rt - tfsi_pos[t], box)) > 1e-5:
            raise ValueError(f"Inconsistent TFSI coordinate: frame {frame}, TFSI {t}")
        metal_pos.setdefault(m, rm)
        tfsi_pos.setdefault(t, rt)

        mn, tn = f"M{m}", f"T{t}"
        uf.union(mn, tn)
        adjacency[mn].add(tn)
        adjacency[tn].add(mn)
        metal_to_tfsi[m].add(t)
        tfsi_to_metal[t].add(m)

    bridges: Dict[str, BridgeFrame] = {}
    for t, metals_set in tfsi_to_metal.items():
        metals = sorted(metals_set)
        for m1, m2 in itertools.combinations(metals, 2):
            key = f"T{t}_M{m1}_M{m2}"
            bridges[key] = BridgeFrame(
                key, t, m1, m2, metal_pos[m1], metal_pos[m2], tfsi_pos[t], box
            )

    wrapped = {f"M{k}": v for k, v in metal_pos.items()}
    wrapped.update({f"T{k}": v for k, v in tfsi_pos.items()})

    clusters: List[ClusterFrame] = []
    max_closure = 0.0
    for component in uf.components():
        metals = tuple(sorted(int(x[1:]) for x in component if x.startswith("M")))
        tfsis = tuple(sorted(int(x[1:]) for x in component if x.startswith("T")))
        (
            nodes,
            positions,
            masses,
            com,
            rg,
            closure,
            is_percolating,
            winding_rank,
            winding_axes,
            winding_vectors,
        ) = reconstruct_component(
            component, adjacency, wrapped, box, meta, cfg.max_internal_nodes
        )
        max_closure = max(max_closure, closure)
        n_bridge_tfsi = sum(len(tfsi_to_metal[t]) >= 2 for t in tfsis)
        n_bridge_edges = sum(
            math.comb(len(tfsi_to_metal[t]), 2)
            for t in tfsis if len(tfsi_to_metal[t]) >= 2
        )
        size = len(metals) + len(tfsis)
        key = membership_key(metals, tfsis)
        clusters.append(
            ClusterFrame(
                key=key,
                metals=metals,
                tfsis=tfsis,
                nodes=nodes,
                positions=positions,
                masses=masses,
                com=com,
                rg=rg,
                closure=closure,
                is_percolating=bool(is_percolating),
                winding_rank=int(winding_rank),
                winding_axes=str(winding_axes),
                winding_vectors=str(winding_vectors),
                n_bridge_tfsi=int(n_bridge_tfsi),
                n_bridge_edges=int(n_bridge_edges),
                charge=meta.metal_charge_number * len(metals) - len(tfsis),
                size_class=classify_size(size),
                metal_class=classify_metal_count(len(metals)),
                box=box,
            )
        )
    clusters.sort(key=lambda c: (-(len(c.metals) + len(c.tfsis)), c.key))

    coordinated_metals = len(metal_pos)
    largest = clusters[0] if clusters else None
    metals_in_multimetal = sum(
        len(c.metals) for c in clusters if len(c.metals) >= 2
    )
    percolating_clusters = [c for c in clusters if c.is_percolating]
    metals_in_percolating = sum(len(c.metals) for c in percolating_clusters)

    metrics = {
        "frame": frame,
        "time_ps": time_ps,
        "n_pairs": len(group),
        "n_coordinated_metals": coordinated_metals,
        "n_coordinated_tfsi": len(tfsi_pos),
        "n_bridging_tfsi": sum(len(v) >= 2 for v in tfsi_to_metal.values()),
        "n_bridge_edges": len(bridges),
        "n_aggregates": len(clusters),
        "mean_aggregate_size": (
            float(np.mean([len(c.metals) + len(c.tfsis) for c in clusters]))
            if clusters else 0.0
        ),
        "largest_aggregate_size": (
            len(largest.metals) + len(largest.tfsis) if largest else 0
        ),
        "largest_aggregate_metals": len(largest.metals) if largest else 0,
        "largest_aggregate_tfsi": len(largest.tfsis) if largest else 0,
        "fraction_coordinated_metals_in_multimetal_aggregates": (
            metals_in_multimetal / coordinated_metals if coordinated_metals else 0.0
        ),
        "n_neutral_aggregates": sum(c.charge == 0 for c in clusters),
        "n_cationic_aggregates": sum(c.charge > 0 for c in clusters),
        "n_anionic_aggregates": sum(c.charge < 0 for c in clusters),
        "n_percolating_aggregates": len(percolating_clusters),
        "largest_percolating_aggregate_size": (
            max(
                len(c.metals) + len(c.tfsis)
                for c in percolating_clusters
            )
            if percolating_clusters
            else 0
        ),
        "fraction_coordinated_metals_in_percolating_aggregates": (
            metals_in_percolating / coordinated_metals
            if coordinated_metals
            else 0.0
        ),
        "maximum_percolation_winding_rank": (
            max(c.winding_rank for c in percolating_clusters)
            if percolating_clusters
            else 0
        ),
        "maximum_cycle_closure_residual_A": max_closure,
    }
    return bridges, clusters, metrics

def bridge_event_statistics(
    buffer: BridgeBuffer,
    lags: np.ndarray,
    meta: Metadata,
    right_censored: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], float, float]:
    frames = np.asarray(buffer.frames, dtype=int)
    r1, max1 = unwrap_series(np.vstack(buffer.r1), np.vstack(buffer.boxes))
    r2, max2 = unwrap_series(np.vstack(buffer.r2), np.vstack(buffer.boxes))
    rt, maxt = unwrap_series(np.vstack(buffer.rt), np.vstack(buffer.boxes))
    n = len(frames)
    occupancy_ps = n * meta.effective_dt_ps
    left = buffer.start_frame == meta.first_frame
    completed = not left and not right_censored
    pclass = classify_persistence(occupancy_ps)
    total_mass = 2 * meta.metal_mass + meta.tfsi_mass
    rows: List[Dict[str, Any]] = []

    for lag in lags:
        lag = int(lag)
        if lag >= n:
            continue
        d1 = r1[lag:] - r1[:-lag]
        d2 = r2[lag:] - r2[:-lag]
        dt = rt[lag:] - rt[:-lag]
        relative = d1 - d2
        midpoint = 0.5 * (d1 + d2)
        bridge_com = (
            meta.metal_mass * d1 + meta.metal_mass * d2 + meta.tfsi_mass * dt
        ) / total_mass
        tfsi_rel_mid = dt - midpoint
        m1sq = np.einsum("ij,ij->i", d1, d1)
        m2sq = np.einsum("ij,ij->i", d2, d2)
        tsq = np.einsum("ij,ij->i", dt, dt)
        relsq = np.einsum("ij,ij->i", relative, relative)
        midsq = np.einsum("ij,ij->i", midpoint, midpoint)
        comsq = np.einsum("ij,ij->i", bridge_com, bridge_com)
        trmsq = np.einsum("ij,ij->i", tfsi_rel_mid, tfsi_rel_mid)
        dot12 = np.einsum("ij,ij->i", d1, d2)
        dot1t = np.einsum("ij,ij->i", d1, dt)
        dot2t = np.einsum("ij,ij->i", d2, dt)
        denom = np.sqrt(m1sq * m2sq)
        valid = denom > 1e-20
        cosine = dot12[valid] / denom[valid] if np.any(valid) else np.array([])

        rows.append({
            "system": meta.system,
            "metal_species": meta.metal_species,
            "composition": meta.composition,
            "temperature_K": meta.temperature_K,
            "bridge_instance_id": buffer.instance,
            "bridge_id": buffer.bridge_id,
            "tfsi_resid": buffer.tfsi,
            "metal1_id": buffer.m1,
            "metal2_id": buffer.m2,
            "completed_event": int(completed),
            "left_censored": int(left),
            "right_censored": int(right_censored),
            "persistence_class": pclass,
            "occupancy_time_ps": occupancy_ps,
            "occupancy_time_ns": occupancy_ps / 1000,
            "event_n_frames": n,
            "lag_frames": lag,
            "lag_ps": lag * meta.effective_dt_ps,
            "lag_ns": lag * meta.effective_dt_ps / 1000,
            "n_time_origins": len(m1sq),
            "sum_m1_sq": float(m1sq.sum()),
            "sum_m2_sq": float(m2sq.sum()),
            "sum_tfsi_sq": float(tsq.sum()),
            "sum_relative_sq": float(relsq.sum()),
            "sum_midpoint_sq": float(midsq.sum()),
            "sum_bridge_com_sq": float(comsq.sum()),
            "sum_tfsi_relative_midpoint_sq": float(trmsq.sum()),
            "sum_metal_dot": float(dot12.sum()),
            "sum_m1_tfsi_dot": float(dot1t.sum()),
            "sum_m2_tfsi_dot": float(dot2t.sum()),
            "sum_directional_cosine": float(cosine.sum()) if cosine.size else 0.0,
            "n_directional_cosine": int(cosine.size),
        })

    continuity = np.array_equal(
        frames,
        np.arange(frames[0], frames[0] + n * meta.stride, meta.stride),
    )
    summary = {
        "system": meta.system,
        "metal_species": meta.metal_species,
        "composition": meta.composition,
        "temperature_K": meta.temperature_K,
        "bridge_instance_id": buffer.instance,
        "bridge_id": buffer.bridge_id,
        "tfsi_resid": buffer.tfsi,
        "metal1_id": buffer.m1,
        "metal2_id": buffer.m2,
        "first_frame": int(frames[0]),
        "last_frame": int(frames[-1]),
        "n_observed_frames": n,
        "occupancy_time_ps": occupancy_ps,
        "occupancy_time_ns": occupancy_ps / 1000,
        "persistence_class": pclass,
        "left_censored": int(left),
        "right_censored": int(right_censored),
        "completed_event": int(completed),
        "continuity_valid": int(continuity),
        "maximum_metal_step_A": max(max1, max2),
        "maximum_tfsi_step_A": maxt,
    }
    return rows, summary, max(max1, max2), maxt

def cluster_event_statistics(
    buffer: ClusterBuffer,
    lags: np.ndarray,
    meta: Metadata,
    right_censored: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], float, float]:
    frames = np.asarray(buffer.frames, dtype=int)
    com, max_step_all = unwrap_series(
        np.vstack(buffer.com),
        np.vstack(buffer.boxes),
    )
    rg = np.asarray(buffer.rg, dtype=float)
    n = len(frames)
    occupancy_ps = n * meta.effective_dt_ps
    left = buffer.start_frame == meta.first_frame
    completed = not left and not right_censored
    pclass = classify_persistence(occupancy_ps)

    percolating_flags = np.asarray(buffer.percolating, dtype=bool)
    percolating_event = bool(np.any(percolating_flags))
    n_percolating_frames = int(np.sum(percolating_flags))
    fraction_percolating_frames = (
        n_percolating_frames / n if n else 0.0
    )
    maximum_winding_rank = (        int(max(buffer.winding_rank))
        if buffer.winding_rank
        else 0
    )
    winding_axes_observed = ",".join(
        sorted(
            {
                axes
                for axes in buffer.winding_axes
                if axes and axes != "none"
            }
        )
    )

    internal_ok = (
        not percolating_event
        and buffer.masses is not None
        and all(x is not None for x in buffer.internal)
    )
    internal = np.stack(buffer.internal) if internal_ok else None
    masses = np.asarray(buffer.masses, dtype=float) if internal_ok else None
    rows: List[Dict[str, Any]] = []

    if not percolating_event:
        for lag in lags:
            lag = int(lag)
            if lag >= n:
                continue

            dcom = com[lag:] - com[:-lag]
            comsq = np.einsum("ij,ij->i", dcom, dcom)
            drg = rg[lag:] - rg[:-lag]

            if internal_ok:
                delta = internal[lag:] - internal[:-lag]
                node_sq = np.einsum("tij,tij->ti", delta, delta)
                weighted = (
                    np.sum(node_sq * masses[None, :], axis=1)
                    / masses.sum()
                )
                sum_internal = float(weighted.sum())
                n_internal = len(weighted)
            else:
                sum_internal = 0.0
                n_internal = 0

            rows.append({
                "system": meta.system,
                "metal_species": meta.metal_species,
                "composition": meta.composition,
                "temperature_K": meta.temperature_K,
                "aggregate_instance_id": buffer.instance,
                "aggregate_key": buffer.key,
                "completed_event": int(completed),
                "left_censored": int(left),
                "right_censored": int(right_censored),
                "persistence_class": pclass,
                "size_class": buffer.size_class,
                "metal_count_class": buffer.metal_class,
                "n_metals": len(buffer.metals),
                "n_tfsi": len(buffer.tfsis),
                "n_total_ions": len(buffer.metals) + len(buffer.tfsis),
                "net_charge_e": buffer.charge,
                "occupancy_time_ps": occupancy_ps,
                "occupancy_time_ns": occupancy_ps / 1000,
                "event_n_frames": n,
                "lag_frames": lag,
                "lag_ps": lag * meta.effective_dt_ps,
                "lag_ns": lag * meta.effective_dt_ps / 1000,
                "n_time_origins": len(comsq),
                "sum_com_sq": float(comsq.sum()),
                "sum_rg_change_sq": float(np.sum(drg * drg)),
                "sum_internal_rearrangement": sum_internal,
                "n_internal_origins": n_internal,
            })

    continuity = np.array_equal(
        frames,
        np.arange(
            frames[0],
            frames[0] + n * meta.stride,
            meta.stride,
        ),
    )

    summary = {
        "system": meta.system,
        "metal_species": meta.metal_species,
        "composition": meta.composition,
        "temperature_K": meta.temperature_K,
        "aggregate_instance_id": buffer.instance,
        "aggregate_key": buffer.key,
        "metal_ids": ";".join(map(str, buffer.metals)),
        "tfsi_resids": ";".join(map(str, buffer.tfsis)),
        "n_metals": len(buffer.metals),
        "n_tfsi": len(buffer.tfsis),
        "n_total_ions": len(buffer.metals) + len(buffer.tfsis),
        "size_class": buffer.size_class,
        "metal_count_class": buffer.metal_class,
        "net_charge_e": buffer.charge,
        "first_frame": int(frames[0]),
        "last_frame": int(frames[-1]),
        "n_observed_frames": n,
        "occupancy_time_ps": occupancy_ps,
        "occupancy_time_ns": occupancy_ps / 1000,
        "persistence_class": pclass,
        "left_censored": int(left),
        "right_censored": int(right_censored),
        "completed_event": int(completed),
        "continuity_valid": int(continuity),
        "is_percolating_event": int(percolating_event),
        "n_percolating_frames": n_percolating_frames,
        "fraction_percolating_frames": fraction_percolating_frames,
        "maximum_winding_rank": maximum_winding_rank,
        "winding_axes_observed": winding_axes_observed,
        "aggregate_transport_eligible": int(not percolating_event),
        "aggregate_transport_exclusion_reason": (
            "periodic_percolation"
            if percolating_event
            else ""
        ),
        "internal_coordinates_available": int(internal_ok),
        "mean_radius_of_gyration_A": float(rg.mean()),
        "maximum_cycle_closure_residual_A": float(max(buffer.closure)),
        "maximum_com_step_all_events_A": max_step_all,
        "maximum_com_step_A": (
            max_step_all if not percolating_event else np.nan
        ),
    }

    maximum_eligible_step = (
        max_step_all if not percolating_event else 0.0
    )
    return rows, summary, maximum_eligible_step, max_step_all

class BridgeTracker:
    def __init__(self, meta: Metadata, lags: np.ndarray, writer: GzipWriter) -> None:
        self.meta = meta
        self.lags = lags
        self.writer = writer
        self.active: Dict[str, BridgeBuffer] = {}
        self.counts: Dict[str, int] = defaultdict(int)
        self.summaries: List[Dict[str, Any]] = []
        self.max_metal_step = 0.0
        self.max_tfsi_step = 0.0
        self.discontinuous = 0

    def update(self, frame: int, time_ps: float, current: Mapping[str, BridgeFrame]) -> None:
        for key in set(self.active) - set(current):
            self._close(key, False)
        for key, data in current.items():
            if key not in self.active:
                self.counts[key] += 1
                self.active[key] = BridgeBuffer(
                    instance=f"{key}_E{self.counts[key]:05d}",
                    bridge_id=key,
                    tfsi=data.tfsi,
                    m1=data.m1,
                    m2=data.m2,
                    start_frame=frame,
                )
            self.active[key].append(frame, time_ps, data)

    def break_all(self) -> None:
        for key in list(self.active):
            self._close(key, False)

    def close_final(self, final_frame_has_pairs: bool) -> None:
        for key in list(self.active):
            self._close(key, final_frame_has_pairs)

    def _close(self, key: str, right: bool) -> None:
        buffer = self.active.pop(key)
        rows, summary, max_metal, max_tfsi = bridge_event_statistics(
            buffer, self.lags, self.meta, right
        )
        self.writer.write_rows(rows)
        self.summaries.append(summary)
        self.max_metal_step = max(self.max_metal_step, max_metal)
        self.max_tfsi_step = max(self.max_tfsi_step, max_tfsi)
        self.discontinuous += int(summary["continuity_valid"] != 1)

class ClusterTracker:
    def __init__(self, meta: Metadata, lags: np.ndarray, writer: GzipWriter) -> None:
        self.meta = meta
        self.lags = lags
        self.writer = writer
        self.active: Dict[str, ClusterBuffer] = {}
        self.counts: Dict[str, int] = defaultdict(int)
        self.summaries: List[Dict[str, Any]] = []
        self.max_com_step = 0.0
        self.max_com_step_all_events = 0.0
        self.max_closure = 0.0
        self.discontinuous = 0
        self.percolating_events_excluded = 0

    def update(self, frame: int, time_ps: float, current: Mapping[str, ClusterFrame]) -> None:
        for key in set(self.active) - set(current):
            self._close(key, False)
        for key, data in current.items():
            if key not in self.active:
                self.counts[key] += 1
                self.active[key] = ClusterBuffer(
                    instance=f"C{membership_hash(key)}_E{self.counts[key]:05d}",
                    key=key,
                    metals=data.metals,
                    tfsis=data.tfsis,
                    nodes=data.nodes,
                    masses=data.masses,
                    size_class=data.size_class,
                    metal_class=data.metal_class,
                    charge=data.charge,
                    start_frame=frame,
                )
            self.active[key].append(frame, time_ps, data)

    def break_all(self) -> None:
        for key in list(self.active):
            self._close(key, False)

    def close_final(self, final_frame_has_pairs: bool) -> None:
        for key in list(self.active):
            self._close(key, final_frame_has_pairs)

    def _close(self, key: str, right: bool) -> None:
        buffer = self.active.pop(key)
        rows, summary, max_step, max_step_all = cluster_event_statistics(
            buffer, self.lags, self.meta, right
        )
        self.writer.write_rows(rows)
        self.summaries.append(summary)
        self.max_com_step = max(self.max_com_step, max_step)
        self.max_com_step_all_events = max(
            self.max_com_step_all_events,
            max_step_all,
        )
        self.percolating_events_excluded += int(
            summary["is_percolating_event"] == 1
        )
        self.max_closure = max(
            self.max_closure,
            summary["maximum_cycle_closure_residual_A"],
        )
        self.discontinuous += int(summary["continuity_valid"] != 1)

class LineageTracker:
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.previous: List[Tuple[str, Set[str]]] = []
        self.next_id = 1
        self.stats: Dict[str, LineageStats] = {}

    def reset(self) -> None:
        self.previous = []

    def assign(self, frame: int, clusters: Sequence[ClusterFrame]) -> None:
        current_sets = [set(c.nodes) for c in clusters]
        previous_sets = [x[1] for x in self.previous]
        candidates: List[Tuple[float, int, int]] = []
        current_counts = [0] * len(current_sets)
        previous_counts = [0] * len(previous_sets)

        for pi, pset in enumerate(previous_sets):
            for ci, cset in enumerate(current_sets):
                union = pset | cset
                score = len(pset & cset) / len(union) if union else 0.0
                if score >= self.threshold:
                    candidates.append((score, pi, ci))
                    current_counts[ci] += 1
                    previous_counts[pi] += 1

        candidates.sort(reverse=True)
        used_p: Set[int] = set()
        used_c: Set[int] = set()
        matches: Dict[int, Tuple[int, float]] = {}
        for score, pi, ci in candidates:
            if pi in used_p or ci in used_c:
                continue
            used_p.add(pi)
            used_c.add(ci)
            matches[ci] = (pi, score)

        new_previous: List[Tuple[str, Set[str]]] = []
        for ci, cluster in enumerate(clusters):
            if ci in matches:
                pi, score = matches[ci]
                lineage_id = self.previous[pi][0]
                split = int(previous_counts[pi] > 1)
            else:
                lineage_id = f"L{self.next_id:07d}"
                self.next_id += 1
                score = np.nan
                split = 0

            if lineage_id not in self.stats:
                self.stats[lineage_id] = LineageStats(
                    lineage_id=lineage_id,
                    first_frame=frame,
                    last_frame=frame,
                )
            size = len(cluster.metals) + len(cluster.tfsis)
            self.stats[lineage_id].update(
                frame=frame,
                size=size,
                n_metals=len(cluster.metals),
                n_tfsi=len(cluster.tfsis),
                jaccard=score,
                merge_count=current_counts[ci],
                split=split,
            )
            new_previous.append((lineage_id, current_sets[ci]))
        self.previous = new_previous

    def dataframe(self, meta: Metadata) -> pd.DataFrame:
        rows = []
        for item in self.stats.values():
            rows.append({
                "system": meta.system,
                "metal_species": meta.metal_species,
                "composition": meta.composition,
                "temperature_K": meta.temperature_K,
                "lineage_id": item.lineage_id,
                "first_frame": item.first_frame,
                "last_frame": item.last_frame,
                "n_frames": item.n_frames,
                "observed_span_ps": (
                    (item.last_frame - item.first_frame) * meta.frame_dt_ps
                ),
                "mean_total_ions": item.sum_size / item.n_frames,
                "minimum_total_ions": item.min_size,
                "maximum_total_ions": item.max_size,
                "maximum_metals": item.max_metals,
                "maximum_tfsi": item.max_tfsi,
                "n_merge_frames": item.merges,
                "n_split_frames": item.splits,
                "mean_jaccard_from_previous": (
                    item.jaccard_sum / item.jaccard_n if item.jaccard_n else np.nan
                ),
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df.sort_values(
                ["n_frames", "maximum_total_ions"],
                ascending=[False, False],
                inplace=True,
                ignore_index=True,
            )
        return df

def process_database(
    pair_path: Path,
    frame_summary_path: Path,
    meta: Metadata,
    cfg: Config,
    paths: Paths,
    lags: np.ndarray,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    header = pd.read_csv(pair_path, compression="gzip", nrows=0).columns
    require_columns(header, PAIR_COLS, "pair_database.csv.gz")
    original = pd.read_csv(frame_summary_path, low_memory=False)
    require_columns(
        original.columns,
        ["frame", "time_ps", "n_unique_metal_tfsi_pairs"],
        "frame_pair_summary.csv",
    )
    original.sort_values("frame", inplace=True, ignore_index=True)

    dtypes = {
        "frame": "int32",
        "time_ps": "float64",
        "metal_id": "int32",
        "metal_resid": "int32",
        "tfsi_resid": "int32",
        "tfsi_reference_atom_id": "int32",
    }
    for c in PAIR_COLS:
        if c not in dtypes:
            dtypes[c] = "float64"

    frame_rows: List[Dict[str, Any]] = []
    size_counter: Counter = Counter()
    pair_rows = 0
    frames_with_pairs = 0
    previous_frame: Optional[int] = None
    frame_order_valid = True
    max_closure = 0.0
    start = time.perf_counter()

    with GzipWriter(paths.bridge_lag, BRIDGE_LAG_COLS) as bridge_writer, \
         GzipWriter(paths.cluster_lag, CLUSTER_LAG_COLS) as cluster_writer:

        bridge_tracker = BridgeTracker(meta, lags, bridge_writer)
        cluster_tracker = ClusterTracker(meta, lags, cluster_writer)
        lineage_tracker = LineageTracker(cfg.lineage_jaccard)

        carry = pd.DataFrame(columns=PAIR_COLS)
        reader = pd.read_csv(
            pair_path,
            compression="gzip",
            usecols=PAIR_COLS,
            dtype=dtypes,
            chunksize=cfg.chunk_size,
            low_memory=False,
        )

        processed = 0

        def process_one(frame_group: pd.DataFrame) -> None:
            nonlocal previous_frame, frame_order_valid, frames_with_pairs
            nonlocal processed, max_closure

            frame = int(frame_group["frame"].iloc[0])
            if previous_frame is not None:
                if frame <= previous_frame:
                    frame_order_valid = False
                if frame - previous_frame > meta.stride:
                    bridge_tracker.break_all()
                    cluster_tracker.break_all()
                    lineage_tracker.reset()
            previous_frame = frame

            bridges, clusters, metrics = frame_graph(frame_group, meta, cfg)
            bridge_tracker.update(frame, metrics["time_ps"], bridges)
            cluster_tracker.update(
                frame,
                metrics["time_ps"],
                {c.key: c for c in clusters},
            )
            lineage_tracker.assign(frame, clusters)

            frame_rows.append({
                "system": meta.system,
                "metal_species": meta.metal_species,
                "composition": meta.composition,
                "temperature_K": meta.temperature_K,
                **metrics,
            })
            for c in clusters:
                size_counter[
                    (
                        len(c.metals) + len(c.tfsis),
                        len(c.metals),
                        len(c.tfsis),
                        c.charge,
                        int(c.is_percolating),
                    )
                ] += 1
            max_closure = max(max_closure, metrics["maximum_cycle_closure_residual_A"])
            frames_with_pairs += 1
            processed += 1
            if processed == 1 or processed % 500 == 0:
                logger.info(
                    "Processed %d pair-containing frames | frame=%d | bridges=%d | "
                    "aggregates=%d | elapsed=%s",
                    processed,
                    frame,
                    len(bridges),
                    len(clusters),
                    format_runtime(time.perf_counter() - start),
                )

        for chunk_number, chunk in enumerate(reader, start=1):
            pair_rows += len(chunk)
            if not carry.empty:
                chunk = pd.concat([carry, chunk], ignore_index=True)
            last_frame = int(chunk["frame"].iloc[-1])
            complete = chunk[chunk["frame"] != last_frame]
            carry = chunk[chunk["frame"] == last_frame].copy()

            for _, group in complete.groupby("frame", sort=False):
                process_one(group)

            logger.info(
                "Read chunk %d | pair rows=%d | carry rows=%d",
                chunk_number,
                pair_rows,
                len(carry),
            )

        if not carry.empty:
            for _, group in carry.groupby("frame", sort=False):
                process_one(group)

        final_frame_has_pairs = previous_frame == meta.last_frame
        bridge_tracker.close_final(final_frame_has_pairs)
        cluster_tracker.close_final(final_frame_has_pairs)

    derived = pd.DataFrame(frame_rows)
    full = original[
        ["frame", "time_ps", "n_unique_metal_tfsi_pairs"]
    ].copy()
    full.rename(
        columns={"n_unique_metal_tfsi_pairs": "n_pairs_from_module_5e1"},
        inplace=True,
    )
    merged = full.merge(
        derived,
        on=["frame", "time_ps"],
        how="left",
    )
    zero_columns = [
        "n_pairs", "n_coordinated_metals", "n_coordinated_tfsi",
        "n_bridging_tfsi", "n_bridge_edges", "n_aggregates",
        "mean_aggregate_size", "largest_aggregate_size",
        "largest_aggregate_metals", "largest_aggregate_tfsi",
        "fraction_coordinated_metals_in_multimetal_aggregates",
        "n_neutral_aggregates", "n_cationic_aggregates",
        "n_anionic_aggregates", "n_percolating_aggregates",
        "largest_percolating_aggregate_size",
        "fraction_coordinated_metals_in_percolating_aggregates",
        "maximum_percolation_winding_rank",
        "maximum_cycle_closure_residual_A",
    ]
    for c in zero_columns:
        merged[c] = merged[c].fillna(0)
    for c, value in [
        ("system", meta.system),
        ("metal_species", meta.metal_species),
        ("composition", meta.composition),
        ("temperature_K", meta.temperature_K),
    ]:
        merged[c] = merged[c].fillna(value)
    merged["pair_count_matches_module_5e1"] = (
        merged["n_pairs"].astype(int)
        == merged["n_pairs_from_module_5e1"].astype(int)
    ).astype(int)
    merged.to_csv(paths.frame_summary, index=False, float_format="%.8g")

    bridge_df = pd.DataFrame(bridge_tracker.summaries)
    cluster_df = pd.DataFrame(cluster_tracker.summaries)
    if not bridge_df.empty:
        bridge_df.sort_values(["first_frame", "bridge_id"], inplace=True, ignore_index=True)
    if not cluster_df.empty:
        cluster_df.sort_values(
            ["first_frame", "n_total_ions", "aggregate_instance_id"],
            inplace=True,
            ignore_index=True,
        )
    bridge_df.to_csv(paths.bridge_events, index=False, float_format="%.8g")
    cluster_df.to_csv(paths.cluster_events, index=False, float_format="%.8g")

    lineage_df = lineage_tracker.dataframe(meta)
    lineage_df.to_csv(paths.lineages, index=False, float_format="%.8g")

    total_size = sum(size_counter.values())
    size_rows = []
    for (size, nm, nt, charge, is_percolating), count in sorted(
        size_counter.items()
    ):
        size_rows.append({
            "system": meta.system,
            "metal_species": meta.metal_species,
            "composition": meta.composition,
            "temperature_K": meta.temperature_K,
            "n_total_ions": size,
            "n_metals": nm,
            "n_tfsi": nt,
            "net_charge_e": charge,
            "is_percolating": int(is_percolating),
            "aggregate_transport_eligible": int(not is_percolating),
            "size_class": classify_size(size),
            "metal_count_class": classify_metal_count(nm),
            "count": count,
            "fraction": count / total_size if total_size else np.nan,
        })
    size_df = pd.DataFrame(size_rows)
    size_df.to_csv(paths.size_distribution, index=False, float_format="%.8g")

    metrics = {
        "n_pair_rows_read": pair_rows,
        "n_frames_with_pairs": frames_with_pairs,
        "frame_order_valid": frame_order_valid,
        "maximum_cluster_cycle_closure_residual_A": max_closure,
        "n_bridge_events": len(bridge_df),
        "n_aggregate_events": len(cluster_df),
        "n_lineages": len(lineage_df),
        "n_bridge_event_lag_rows": bridge_writer.rows,
        "n_aggregate_event_lag_rows": cluster_writer.rows,
        "maximum_bridge_metal_step_A": bridge_tracker.max_metal_step,
        "maximum_bridge_tfsi_step_A": bridge_tracker.max_tfsi_step,
        "maximum_aggregate_com_step_A": cluster_tracker.max_com_step,
        "maximum_aggregate_com_step_all_events_A": (
            cluster_tracker.max_com_step_all_events
        ),
        "n_percolating_aggregate_events_excluded": (
            cluster_tracker.percolating_events_excluded
        ),
        "n_percolating_aggregate_frame_observations": int(
            merged["n_percolating_aggregates"].sum()
        ),
        "fraction_frames_with_percolating_aggregate": float(
            (merged["n_percolating_aggregates"] > 0).mean()
        ),
        "bridge_discontinuous_events": bridge_tracker.discontinuous,
        "aggregate_discontinuous_events": cluster_tracker.discontinuous,
        "processing_seconds": time.perf_counter() - start,
    }
    return merged, bridge_df, cluster_df, lineage_df, size_df, metrics

BRIDGE_SUMS = [
    "n_time_origins", "sum_m1_sq", "sum_m2_sq", "sum_tfsi_sq",
    "sum_relative_sq", "sum_midpoint_sq", "sum_bridge_com_sq",
    "sum_tfsi_relative_midpoint_sq", "sum_metal_dot",
    "sum_m1_tfsi_dot", "sum_m2_tfsi_dot",
    "sum_directional_cosine", "n_directional_cosine",
]
CLUSTER_SUMS = [
    "n_time_origins", "sum_com_sq", "sum_rg_change_sq",
    "sum_internal_rearrangement", "n_internal_origins",
]

def bridge_metrics(s: Mapping[str, float]) -> Dict[str, float]:
    n = float(s["n_time_origins"])
    if n <= 0:
        return {}
    m1 = float(s["sum_m1_sq"]) / n
    m2 = float(s["sum_m2_sq"]) / n
    tfsi = float(s["sum_tfsi_sq"]) / n
    relative = float(s["sum_relative_sq"]) / n
    midpoint = float(s["sum_midpoint_sq"]) / n
    bcom = float(s["sum_bridge_com_sq"]) / n
    trm = float(s["sum_tfsi_relative_midpoint_sq"]) / n
    dot = float(s["sum_metal_dot"]) / n
    d1t = float(s["sum_m1_tfsi_dot"]) / n
    d2t = float(s["sum_m2_tfsi_dot"]) / n
    corr = dot / math.sqrt(m1 * m2) if m1 > 0 and m2 > 0 else np.nan
    alignment = 2 * dot / (m1 + m2) if m1 + m2 > 0 else np.nan
    rel_ratio = relative / (m1 + m2) if m1 + m2 > 0 else np.nan
    ncos = float(s["n_directional_cosine"])
    cosine = float(s["sum_directional_cosine"]) / ncos if ncos > 0 else np.nan
    a1 = 2 * d1t / (m1 + tfsi) if m1 + tfsi > 0 else np.nan
    a2 = 2 * d2t / (m2 + tfsi) if m2 + tfsi > 0 else np.nan
    return {
        "metal1_msd_A2": m1,
        "metal2_msd_A2": m2,
        "tfsi_msd_A2": tfsi,
        "metal_relative_msd_A2": relative,
        "metal_midpoint_msd_A2": midpoint,
        "bridge_com_msd_A2": bcom,
        "tfsi_relative_midpoint_msd_A2": trm,
        "metal_vector_correlation": corr,
        "metal_collective_alignment": alignment,
        "metal_relative_motion_ratio": rel_ratio,
        "mean_metal_directional_cosine": cosine,
        "mean_metal_tfsi_alignment": float(np.nanmean([a1, a2])),
    }

def cluster_metrics(s: Mapping[str, float]) -> Dict[str, float]:
    n = float(s["n_time_origins"])
    if n <= 0:
        return {}
    com = float(s["sum_com_sq"]) / n
    rg = float(s["sum_rg_change_sq"]) / n
    ni = float(s["n_internal_origins"])
    internal = (
        float(s["sum_internal_rearrangement"]) / ni if ni > 0 else np.nan
    )
    translation = (
        com / (com + internal)
        if np.isfinite(internal) and com + internal > 0
        else np.nan
    )
    return {
        "aggregate_com_msd_A2": com,
        "rg_change_msd_A2": rg,
        "internal_rearrangement_msd_A2": internal,
        "translation_fraction": translation,
    }

def bootstrap_intervals(
    group: pd.DataFrame,
    sum_cols: Sequence[str],
    metric_fn,
    metric_names: Sequence[str],
    cfg: Config,
    rng: np.random.Generator,
) -> Dict[str, Tuple[float, float]]:
    if cfg.bootstrap_replicates == 0 or len(group) < 2:
        return {name: (np.nan, np.nan) for name in metric_names}
    values = group[list(sum_cols)].to_numpy(dtype=float)
    samples = {name: np.empty(cfg.bootstrap_replicates) for name in metric_names}
    for i in range(cfg.bootstrap_replicates):
        idx = rng.integers(0, len(values), size=len(values))
        metrics = metric_fn(dict(zip(sum_cols, values[idx].sum(axis=0))))
        for name in metric_names:
            samples[name][i] = metrics.get(name, np.nan)
    return {
        name: (
            float(np.nanpercentile(arr, 2.5)),
            float(np.nanpercentile(arr, 97.5)),
        )
        for name, arr in samples.items()
    }

def aggregate_lag_file(
    path: Path,
    kind: str,
    group_column: Optional[str],
    group_type: str,
    meta: Metadata,
    cfg: Config,
    rng: np.random.Generator,
) -> pd.DataFrame:
    df = pd.read_csv(path, compression="gzip", low_memory=False)
    if df.empty:
        return pd.DataFrame()
    if group_column is None:
        df["_group"] = "all_bridges" if kind == "bridge" else "all_aggregates"
        group_column = "_group"

    if kind == "bridge":
        sum_cols = BRIDGE_SUMS
        metric_fn = bridge_metrics
        id_col = "bridge_instance_id"
        metric_names = [
            "metal1_msd_A2", "metal2_msd_A2", "tfsi_msd_A2",
            "metal_relative_msd_A2", "metal_midpoint_msd_A2",
            "bridge_com_msd_A2", "tfsi_relative_midpoint_msd_A2",
            "metal_vector_correlation", "metal_collective_alignment",
            "metal_relative_motion_ratio", "mean_metal_directional_cosine",
            "mean_metal_tfsi_alignment",
        ]
    else:
        sum_cols = CLUSTER_SUMS
        metric_fn = cluster_metrics
        id_col = "aggregate_instance_id"
        metric_names = [
            "aggregate_com_msd_A2", "rg_change_msd_A2",
            "internal_rearrangement_msd_A2", "translation_fraction",
        ]

    rows: List[Dict[str, Any]] = []
    for (label, lag), group in df.groupby([group_column, "lag_frames"], observed=True):
        sums = group[sum_cols].sum()
        metrics = metric_fn(sums)
        intervals = bootstrap_intervals(
            group, sum_cols, metric_fn, metric_names, cfg, rng
        )
        n_events = int(group[id_col].nunique())
        n_origins = int(sums["n_time_origins"])
        row = {
            "system": meta.system,
            "metal_species": meta.metal_species,
            "composition": meta.composition,
            "temperature_K": meta.temperature_K,
            "group_type": group_type,
            "group_label": str(label),
            "lag_frames": int(lag),
            "lag_ps": int(lag) * meta.effective_dt_ps,
            "lag_ns": int(lag) * meta.effective_dt_ps / 1000,
            "n_events": n_events,
            "n_time_origins": n_origins,
            "recommended_for_interpretation": int(
                n_events >= cfg.min_events and n_origins >= cfg.min_origins
            ),
            **metrics,
        }
        if kind == "cluster":
            row["n_internal_origins"] = int(sums["n_internal_origins"])
        for name, (low, high) in intervals.items():
            row[f"{name}_ci_low"] = low
            row[f"{name}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)

def validate_results(
    frame_df: pd.DataFrame,
    bridge_events: pd.DataFrame,
    cluster_events: pd.DataFrame,
    bridge_all: pd.DataFrame,
    cluster_all: pd.DataFrame,
    metrics: Mapping[str, Any],
    meta: Metadata,
    cfg: Config,
    upstream: Mapping[str, Any],
) -> Validation:
    v = Validation()
    v.check(
        "module_5e1_validation_acceptable",
        upstream.get("status") in {"PASSED", "PASSED_WITH_WARNINGS"},
    )
    v.check("all_selected_frames_present", len(frame_df) == meta.n_frames)
    v.check(
        "pair_counts_match_module_5e1",
        bool((frame_df["pair_count_matches_module_5e1"] == 1).all()),
    )
    v.check("pair_database_frame_order_valid", bool(metrics["frame_order_valid"]))
    v.check(
        "bridge_events_continuous",
        int(metrics["bridge_discontinuous_events"]) == 0,
    )
    v.check(
        "aggregate_events_continuous",
        int(metrics["aggregate_discontinuous_events"]) == 0,
    )
    v.check(
        "cluster_cycle_closure_acceptable",
        float(metrics["maximum_cluster_cycle_closure_residual_A"])
        <= cfg.max_closure_A,
    )

    if not bridge_all.empty:
        v.check(
            "bridge_alignment_bounded",
            bool(
                bridge_all["metal_collective_alignment"]
                .dropna()
                .between(-1.000001, 1.000001)
                .all()
            ),
        )
        v.check(
            "bridge_relative_motion_identity",
            bool(
                np.allclose(
                    bridge_all["metal_relative_motion_ratio"],
                    1 - bridge_all["metal_collective_alignment"],
                    rtol=1e-7,
                    atol=1e-9,
                    equal_nan=True,
                )
            ),
        )
    else:
        v.check("bridge_alignment_bounded", True)
        v.check("bridge_relative_motion_identity", True)
        v.warn("No TFSI-mediated metal bridges were detected.")

    if not cluster_all.empty:
        vals = cluster_all["translation_fraction"].dropna()
        v.check(
            "aggregate_translation_fraction_bounded",
            bool(vals.between(-1e-9, 1.000001).all()),
        )
    else:
        v.check("aggregate_translation_fraction_bounded", True)

    for name, value in metrics.items():
        v.metric(name, value)
    v.metric("mean_bridging_tfsi_per_frame", frame_df["n_bridging_tfsi"].mean())
    v.metric("mean_bridge_edges_per_frame", frame_df["n_bridge_edges"].mean())
    v.metric("mean_largest_aggregate_size", frame_df["largest_aggregate_size"].mean())
    v.metric("maximum_aggregate_size", frame_df["largest_aggregate_size"].max())
    v.metric(
        "mean_percolating_aggregates_per_frame",
        frame_df["n_percolating_aggregates"].mean(),
    )
    v.metric(
        "fraction_frames_with_percolating_aggregate",
        (frame_df["n_percolating_aggregates"] > 0).mean(),
    )
    v.metric(
        "n_percolating_aggregate_events_excluded",
        metrics["n_percolating_aggregate_events_excluded"],
    )
    v.check(
        "percolating_events_excluded_from_aggregate_transport",
        int(metrics["n_percolating_aggregate_events_excluded"]) >= 0,
    )

    if metrics["maximum_bridge_metal_step_A"] > cfg.max_step_A:
        v.warn("A bridged-metal frame step exceeds the configured threshold.")
    if metrics["maximum_bridge_tfsi_step_A"] > cfg.max_step_A:
        v.warn("A bridging-TFSI frame step exceeds the configured threshold.")
    if metrics["maximum_aggregate_com_step_A"] > cfg.max_step_A:
        v.warn("A nonpercolating, transport-eligible aggregate COM frame step exceeds the configured threshold.")
    return v

def configure_plots() -> None:
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 9,
        "savefig.bbox": "tight",
    })

def save_plot(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.savefig(path, dpi=dpi)
    plt.close(fig)

def make_figures(
    frame_df: pd.DataFrame,
    size_df: pd.DataFrame,
    bridge_persistence: pd.DataFrame,
    cluster_size: pd.DataFrame,
    cluster_events: pd.DataFrame,
    meta: Metadata,
    cfg: Config,
    paths: Paths,
) -> None:
    configure_plots()
    label = f"{meta.metal_species} {meta.composition}"
    frame_df = frame_df.copy()
    frame_df["time_ns"] = frame_df["time_ps"] / 1000

    fig, ax = plt.subplots(figsize=(7.5, 4.7))
    ax.plot(frame_df["time_ns"], frame_df["n_bridge_edges"], lw=0.7)
    ax.set(xlabel="Time (ns)", ylabel="TFSI-mediated metal bridge edges",
           title=f"{label}: bridge-network population")
    ax.grid(alpha=0.25)
    save_plot(fig, paths.figures / "bridge_population_vs_time.png", cfg.figure_dpi)

    fig, ax = plt.subplots(figsize=(7.5, 4.7))
    ax.plot(frame_df["time_ns"], frame_df["largest_aggregate_size"], lw=0.7)
    ax.set(xlabel="Time (ns)", ylabel="Largest aggregate size",
           title=f"{label}: largest coordinated aggregate")
    ax.grid(alpha=0.25)
    save_plot(fig, paths.figures / "largest_aggregate_vs_time.png", cfg.figure_dpi)

    if not size_df.empty:
        grouped = size_df.groupby("n_total_ions", as_index=False)["count"].sum()
        grouped["fraction"] = grouped["count"] / grouped["count"].sum()
        fig, ax = plt.subplots(figsize=(7.0, 4.7))
        ax.bar(grouped["n_total_ions"], grouped["fraction"])
        ax.set(xlabel="Aggregate size (coordinated ions)",
               ylabel="Fraction of aggregate-frame observations",
               title=f"{label}: aggregate-size distribution")
        ax.grid(axis="y", alpha=0.25)
        save_plot(fig, paths.figures / "aggregate_size_distribution.png", cfg.figure_dpi)

    def group_plot(df: pd.DataFrame, metric: str, ylabel: str, filename: str) -> None:
        if df.empty:
            return
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        for name, group in df.groupby("group_label"):
            group = group[group["recommended_for_interpretation"] == 1]
            if group.empty:
                continue
            ax.plot(group["lag_ns"], group[metric], marker="o", ms=3, lw=1.2, label=name)
        ax.set_xscale("log")
        ax.set(xlabel="Lag time (ns)", ylabel=ylabel, title=f"{label}: {ylabel}")
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
        save_plot(fig, paths.figures / filename, cfg.figure_dpi)

    group_plot(
        bridge_persistence,
        "metal_collective_alignment",
        "Bridged-metal collective alignment",
        "bridge_alignment_by_persistence.png",
    )
    group_plot(
        bridge_persistence,
        "metal_relative_motion_ratio",
        "Bridged-metal relative-motion ratio",
        "bridge_relative_motion_by_persistence.png",
    )

    if not cluster_size.empty:
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        for name, group in cluster_size.groupby("group_label"):
            group = group[
                (group["recommended_for_interpretation"] == 1)
                & (group["aggregate_com_msd_A2"] > 0)
            ]
            if group.empty:
                continue
            ax.plot(
                group["lag_ns"], group["aggregate_com_msd_A2"],
                marker="o", ms=3, lw=1.2, label=name
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set(xlabel="Lag time (ns)", ylabel="Aggregate COM MSD (Å$^2$)",
               title=f"{label}: aggregate COM transport")
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
        save_plot(fig, paths.figures / "aggregate_com_msd_by_size.png", cfg.figure_dpi)

    group_plot(
        cluster_size,
        "translation_fraction",
        "Aggregate translation fraction",
        "aggregate_translation_fraction_by_size.png",
    )

    if not cluster_events.empty:
        durations = cluster_events.loc[
            cluster_events["completed_event"] == 1, "occupancy_time_ns"
        ].to_numpy(dtype=float)
        durations = durations[durations > 0]
        if durations.size:
            bins = (
                np.logspace(np.log10(durations.min()), np.log10(durations.max()), 45)
                if durations.max() > durations.min()
                else np.array([durations.min() * 0.8, durations.min() * 1.2])
            )
            fig, ax = plt.subplots(figsize=(7.0, 4.7))
            ax.hist(durations, bins=bins)
            ax.set_xscale("log")
            ax.set(xlabel="Exact-membership aggregate lifetime (ns)",
                   ylabel="Completed events",
                   title=f"{label}: aggregate-event lifetimes")
            ax.grid(axis="y", alpha=0.25)
            save_plot(
                fig,
                paths.figures / "aggregate_event_lifetime_distribution.png",
                cfg.figure_dpi,            )

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="TFSI bridge-network and aggregate transport analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-dir", type=Path, default=Path(DEFAULT_INPUT_DIR))
    p.add_argument("--output-subdir", default=DEFAULT_OUTPUT_SUBDIR)
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    p.add_argument("--max-lag-ns", type=float, default=DEFAULT_MAX_LAG_NS)
    p.add_argument("--n-log-lags", type=int, default=DEFAULT_N_LOG_LAGS)
    p.add_argument("--linear-lags", type=int, default=DEFAULT_LINEAR_LAGS)
    p.add_argument(
        "--lineage-jaccard", type=float, default=DEFAULT_LINEAGE_JACCARD
    )
    p.add_argument(
        "--max-internal-nodes", type=int, default=DEFAULT_MAX_INTERNAL_NODES
    )
    p.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    p.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    p.add_argument("--min-events", type=int, default=DEFAULT_MIN_EVENTS)
    p.add_argument("--min-origins", type=int, default=DEFAULT_MIN_ORIGINS)
    p.add_argument("--max-step-A", type=float, default=DEFAULT_MAX_STEP_A)
    p.add_argument("--max-closure-A", type=float, default=DEFAULT_MAX_CLOSURE_A)
    p.add_argument("--metal-mass-g-mol", type=float, default=DEFAULT_METAL_MASS_G_MOL,
                   help="Metal-ion molar mass used for mass-weighted bridge/aggregate COM motion")
    p.add_argument("--metal-charge-number", type=int, default=DEFAULT_METAL_CHARGE_NUMBER,
                   help="Formal metal-ion charge number used for aggregate net charge")
    p.add_argument("--figure-dpi", type=int, default=DEFAULT_FIGURE_DPI)
    p.add_argument("--overwrite", action="store_true")
    return p

def main() -> None:
    args = parser().parse_args()
    input_dir = args.input_dir.resolve()
    cfg = Config(
        input_dir=input_dir,
        output_dir=input_dir / args.output_subdir,
        chunk_size=args.chunk_size,
        max_lag_ns=args.max_lag_ns,
        n_log_lags=args.n_log_lags,
        linear_lags=args.linear_lags,
        lineage_jaccard=args.lineage_jaccard,
        max_internal_nodes=args.max_internal_nodes,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        min_events=args.min_events,
        min_origins=args.min_origins,
        max_step_A=args.max_step_A,
        max_closure_A=args.max_closure_A,
        figure_dpi=args.figure_dpi,
        overwrite=args.overwrite,
    )
    cfg.validate()
    if args.metal_charge_number <= 0:
        raise ValueError("--metal-charge-number must be positive")

    input_paths = {
        "pairs": input_dir / "pair_database.csv.gz",
        "frames": input_dir / "frame_pair_summary.csv",
        "metadata": input_dir / "metadata.json",
        "validation": input_dir / "final_validation_report.json",
    }
    missing = [p for p in input_paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(map(str, missing)))

    paths = Paths(cfg.output_dir)
    paths.prepare(cfg.overwrite)
    logger = configure_logging(paths.log)
    start = time.perf_counter()

    metadata_json = read_json(input_paths["metadata"])
    identity = metadata_json.get("identity", {})
    configuration = metadata_json.get("configuration", {})
    selected = metadata_json.get("selected_frame_summary", {})
    topology = metadata_json.get("topology_summary", {})
    metal = str(identity.get("metal_species", "metal"))
    frame_dt = float(configuration.get("frame_interval_ps", 10.0))
    stride = int(configuration.get("stride", selected.get("stride", 1)))
    meta = Metadata(
        system=str(identity.get("system")),
        metal_species=metal,
        composition=str(identity.get("composition")),
        temperature_K=float(identity.get("temperature_K", np.nan)),
        frame_dt_ps=frame_dt,
        stride=stride,
        effective_dt_ps=frame_dt * stride,
        first_frame=int(selected.get("first_frame", 0)),
        last_frame=int(selected.get("last_frame", 0)),
        n_frames=int(selected.get("n_frames", 0)),
        n_metals=int(topology.get("n_metal_atoms", 0)),
        n_tfsi=int(topology.get("n_tfsi_residues", 0)),
        metal_mass=float(args.metal_mass_g_mol),
        metal_charge_number=int(args.metal_charge_number),
        tfsi_mass=TFSI_MASS,
    )

    logger.info("=" * 79)
    logger.info("MODULE 5E3: BRIDGE-NETWORK AND AGGREGATE TRANSPORT")
    logger.info("=" * 79)
    logger.info("System              : %s", meta.system)
    logger.info("Input directory     : %s", cfg.input_dir)
    logger.info("Output directory    : %s", cfg.output_dir)
    logger.info("Maximum lag         : %.3f ns", cfg.max_lag_ns)
    logger.info("Lineage Jaccard     : %.3f", cfg.lineage_jaccard)
    logger.info("Max internal nodes  : %d", cfg.max_internal_nodes)

    lags = build_lag_grid(meta, cfg)
    lag_df = pd.DataFrame({
        "lag_frames": lags,
        "lag_ps": lags * meta.effective_dt_ps,
        "lag_ns": lags * meta.effective_dt_ps / 1000,
    })
    lag_df.to_csv(paths.lag_grid, index=False, float_format="%.8g")
    logger.info("Lag points          : %d", len(lags))

    (
        frame_df,
        bridge_events,
        cluster_events,
        lineage_df,
        size_df,
        processing,
    ) = process_database(
        input_paths["pairs"],
        input_paths["frames"],
        meta,
        cfg,
        paths,
        lags,
        logger,
    )

    rng = np.random.default_rng(cfg.bootstrap_seed)
    bridge_all = aggregate_lag_file(
        paths.bridge_lag, "bridge", None, "all_bridges", meta, cfg, rng
    )
    bridge_persistence = aggregate_lag_file(
        paths.bridge_lag, "bridge", "persistence_class",
        "persistence_class", meta, cfg, rng
    )
    cluster_all = aggregate_lag_file(
        paths.cluster_lag, "cluster", None, "all_aggregates", meta, cfg, rng
    )
    cluster_size = aggregate_lag_file(
        paths.cluster_lag, "cluster", "size_class", "size_class", meta, cfg, rng
    )
    cluster_metals = aggregate_lag_file(
        paths.cluster_lag, "cluster", "metal_count_class",
        "metal_count_class", meta, cfg, rng
    )
    cluster_persistence = aggregate_lag_file(
        paths.cluster_lag, "cluster", "persistence_class",
        "persistence_class", meta, cfg, rng
    )

    for df, path in [
        (bridge_all, paths.bridge_all),
        (bridge_persistence, paths.bridge_persistence),
        (cluster_all, paths.cluster_all),
        (cluster_size, paths.cluster_size),
        (cluster_metals, paths.cluster_metals),
        (cluster_persistence, paths.cluster_persistence),
    ]:
        df.to_csv(path, index=False, float_format="%.8g")

    validation = validate_results(
        frame_df,
        bridge_events,
        cluster_events,
        bridge_all,
        cluster_all,
        processing,
        meta,
        cfg,
        read_json(input_paths["validation"]),
    )
    write_json(paths.validation, validation.as_dict())

    make_figures(
        frame_df,
        size_df,
        bridge_persistence,
        cluster_size,
        cluster_events,
        meta,
        cfg,
        paths,
    )

    validation.finalize()
    summary = {
        "module": "05E3_bridge_network_aggregate_transport_percolation_corrected",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "system": asdict(meta),
        "configuration": asdict(cfg),
        "processing": processing,
        "network": {
            "mean_bridging_tfsi_per_frame": float(frame_df["n_bridging_tfsi"].mean()),
            "mean_bridge_edges_per_frame": float(frame_df["n_bridge_edges"].mean()),
            "mean_aggregates_per_frame": float(frame_df["n_aggregates"].mean()),
            "mean_largest_aggregate_size": float(
                frame_df["largest_aggregate_size"].mean()
            ),
            "maximum_aggregate_size": int(frame_df["largest_aggregate_size"].max()),
            "mean_fraction_coordinated_metals_in_multimetal_aggregates": float(
                frame_df[
                    "fraction_coordinated_metals_in_multimetal_aggregates"
                ].mean()
            ),
            "mean_percolating_aggregates_per_frame": float(
                frame_df["n_percolating_aggregates"].mean()
            ),
            "fraction_frames_with_percolating_aggregate": float(
                (frame_df["n_percolating_aggregates"] > 0).mean()
            ),
            "mean_fraction_coordinated_metals_in_percolating_aggregates": float(
                frame_df[
                    "fraction_coordinated_metals_in_percolating_aggregates"
                ].mean()
            ),
            "maximum_percolating_aggregate_size": int(
                frame_df["largest_percolating_aggregate_size"].max()
            ),
        },
        "bridge_events": {
            "n_events": len(bridge_events),
            "median_lifetime_ps": (
                float(bridge_events["occupancy_time_ps"].median())
                if not bridge_events.empty else None
            ),
            "maximum_lifetime_ps": (
                float(bridge_events["occupancy_time_ps"].max())
                if not bridge_events.empty else None
            ),
        },
        "aggregate_events": {
            "n_events": len(cluster_events),
            "median_lifetime_ps": (
                float(cluster_events["occupancy_time_ps"].median())
                if not cluster_events.empty else None
            ),
            "maximum_lifetime_ps": (
                float(cluster_events["occupancy_time_ps"].max())
                if not cluster_events.empty else None
            ),
            "fraction_with_internal_coordinates": (
                float(cluster_events["internal_coordinates_available"].mean())
                if not cluster_events.empty else None
            ),
            "n_percolating_events_excluded_from_transport": (
                int(cluster_events["is_percolating_event"].sum())
                if not cluster_events.empty else 0
            ),
            "fraction_percolating_events": (
                float(cluster_events["is_percolating_event"].mean())
                if not cluster_events.empty else None
            ),
            "n_transport_eligible_events": (
                int(cluster_events["aggregate_transport_eligible"].sum())
                if not cluster_events.empty else 0
            ),
        },
        "lineages": {
            "n_lineages": len(lineage_df),
            "maximum_lineage_frames": (
                int(lineage_df["n_frames"].max()) if not lineage_df.empty else 0
            ),
            "maximum_lineage_size": (
                int(lineage_df["maximum_total_ions"].max())
                if not lineage_df.empty else 0
            ),
        },
        "validation_status": validation.status,
        "warnings": validation.warnings,
        "scope_note": (
            "Aggregate statistics include only ions participating in at least "
            "one metal–TFSI edge. Uncoordinated ions are absent. Periodically "
            "percolating aggregate events remain in structural and lineage "
            "statistics but are excluded from aggregate COM/internal transport."
        ),
        "runtime_seconds": time.perf_counter() - start,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(paths.summary, summary)

    logger.info("=" * 79)
    logger.info("MODULE 5E3 COMPLETED")
    logger.info("=" * 79)
    logger.info("Pair rows read       : %d", processing["n_pair_rows_read"])
    logger.info("Bridge events        : %d", len(bridge_events))
    logger.info("Aggregate events     : %d", len(cluster_events))
    logger.info("Aggregate lineages   : %d", len(lineage_df))
    logger.info(
        "Percolating events   : %d excluded from aggregate transport",
        processing["n_percolating_aggregate_events_excluded"],
    )
    logger.info(
        "Frames percolating   : %.2f%%",
        100.0 * processing["fraction_frames_with_percolating_aggregate"],
    )
    logger.info(
        "Mean bridge edges    : %.6f/frame",
        float(frame_df["n_bridge_edges"].mean()),
    )
    logger.info(
        "Maximum cluster size : %d ions",
        int(frame_df["largest_aggregate_size"].max()),
    )
    logger.info("Validation status   : %s", validation.status)
    logger.info("Warnings            : %d", len(validation.warnings))
    logger.info("Summary             : %s", paths.summary)
    logger.info("Runtime             : %s", format_runtime(time.perf_counter() - start))

if __name__ == "__main__":
    main()