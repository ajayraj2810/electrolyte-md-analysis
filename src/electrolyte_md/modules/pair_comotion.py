#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05E2_pair_comotion_analysis.py
==============================

Module 5E2 — Metal–TFSI Pair Co-motion Analysis

Purpose
-------
Determine whether coordinated metal ions and TFSI ions translate together,
move independently while remaining associated, or undergo strong internal
rearrangement.

This script reads the validated Module 5E1 outputs only. It does NOT reopen
the topology or LAMMPS trajectory.

Required inputs
---------------
05E1_pair_database/
├── pair_database.csv.gz
├── pair_event_summary.csv
├── metadata.json
└── final_validation_report.json

Core observables
----------------
For every surviving pair event and lag time:

1. Metal MSD:
       <|Δr_M|²>

2. TFSI-reference MSD:
       <|Δr_T|²>

3. Relative MSD:
       <|Δr_M - Δr_T|²>

4. Pair midpoint MSD:
       <|(Δr_M + Δr_T)/2|²>

5. Mass-weighted pair COM MSD:
       <|(m_M Δr_M + m_T Δr_T)/(m_M + m_T)|²>

6. Displacement dot product:
       <Δr_M · Δr_T>

7. Vector correlation coefficient:
       <Δr_M · Δr_T> /
       sqrt(<|Δr_M|²><|Δr_T|²>)

8. Collective alignment index:
       2<Δr_M · Δr_T> /
       (<|Δr_M|²> + <|Δr_T|²>)

9. Relative-motion ratio:
       <|Δr_M - Δr_T|²> /
       (<|Δr_M|²> + <|Δr_T|²>)

10. Mean directional cosine:
       <cos θ> =
       <(Δr_M · Δr_T)/(|Δr_M||Δr_T|)>

Interpretation
--------------
Strong co-motion:
    positive vector correlation,
    collective alignment approaching +1,
    low relative-motion ratio,
    and pair-COM motion larger than internal relative motion.

Independent motion:
    vector correlation near 0,
    relative-motion ratio near 1.

Opposing motion:
    negative correlation,
    relative-motion ratio greater than 1.

Persistence classes
-------------------
Transient:       10–50 ps
Short-lived:     >50–500 ps
Intermediate:    >0.5–5 ns
Persistent:      >5 ns

Censored pair events are retained in the "all_observed" analysis because their
observed continuous segments contain valid displacement information. Separate
"completed_only" curves are also generated.

Periodic-boundary treatment
---------------------------
Stored coordinates are wrapped. Each pair event is unwrapped independently
using frame-to-frame minimum-image displacements and the full triclinic box
matrix reconstructed from box lengths and angles.

Outputs
-------
05E1_pair_database/05E2_pair_comotion/
├── pair_comotion_all.csv
├── pair_comotion_by_lifetime.csv
├── pair_comotion_by_formation.csv
├── pair_comotion_by_coordination.csv
├── pair_comotion_event_lag_statistics.csv.gz
├── pair_comotion_event_summary.csv
├── lag_grid.csv
├── validation_report.json
├── summary.json
├── 05E2_pair_comotion.log
└── figures/
    ├── all_pair_msd_components.png
    ├── all_pair_correlation_metrics.png
    ├── lifetime_alignment_comparison.png
    ├── lifetime_relative_motion_comparison.png
    ├── formation_alignment_comparison.png
    ├── coordination_alignment_comparison.png
    └── lag_sample_support.png

Author: Ajay Dwivedi
"""

from __future__ import annotations

import argparse
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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# DEFAULTS
# =============================================================================

DEFAULT_INPUT_DIR = "05E1_pair_database"
DEFAULT_OUTPUT_SUBDIR = "05E2_pair_comotion"

DEFAULT_CHUNK_SIZE = 200_000
DEFAULT_MAX_LAG_NS = 20.0
DEFAULT_N_LOG_LAGS = 55
DEFAULT_LINEAR_LAG_FRAMES = 20

DEFAULT_BOOTSTRAP_REPLICATES = 300
DEFAULT_BOOTSTRAP_SEED = 20260720
DEFAULT_CI_LEVEL = 0.95

DEFAULT_MIN_EVENTS_FOR_INTERPRETATION = 20
DEFAULT_MIN_ORIGINS_FOR_INTERPRETATION = 200
DEFAULT_DIRECTION_EPSILON_A = 1.0e-10
DEFAULT_MAX_REASONABLE_STEP_A = 25.0
DEFAULT_FIGURE_DPI = 300

# Molecular masses used for the mass-weighted pair COM displacement.
DEFAULT_METAL_MASS_G_MOL = 40.0  # override with --metal-mass-g-mol
TFSI_MASS_G_MOL = 280.14

PAIR_DATABASE_COLUMNS = [
    "pair_instance_id",
    "frame",
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

EVENT_SUMMARY_REQUIRED_COLUMNS = [
    "pair_instance_id",
    "pair_id",
    "event_number",
    "reformation_event",
    "metal_id",
    "metal_resid",
    "tfsi_resid",
    "first_frame",
    "last_frame",
    "n_observed_frames",
    "occupancy_time_ps",
    "occupancy_time_ns",
    "left_censored",
    "right_censored",
    "fully_censored",
    "completed_event",
    "mean_coordinating_oxygens",
    "maximum_coordinating_oxygens",
    "fraction_multi_oxygen_frames",
]

EVENT_LAG_COLUMNS = [
    "system",
    "metal_species",
    "composition",
    "temperature_K",
    "pair_instance_id",
    "pair_id",
    "event_number",
    "reformation_event",
    "formation_class",
    "completed_event",
    "left_censored",
    "right_censored",
    "lifetime_class",
    "coordination_class",
    "occupancy_time_ps",
    "occupancy_time_ns",
    "event_n_frames",
    "lag_frames",
    "lag_ps",
    "lag_ns",
    "n_time_origins",
    "sum_metal_displacement_sq_A2",
    "sum_tfsi_displacement_sq_A2",
    "sum_relative_displacement_sq_A2",
    "sum_midpoint_displacement_sq_A2",
    "sum_pair_com_displacement_sq_A2",
    "sum_displacement_dot_A2",
    "sum_directional_cosine",
    "n_directional_cosine",
]

AGGREGATE_COLUMNS = [
    "system",
    "metal_species",
    "composition",
    "temperature_K",
    "analysis_scope",
    "group_type",
    "group_label",
    "lag_frames",
    "lag_ps",
    "lag_ns",
    "n_events",
    "n_time_origins",
    "metal_msd_A2",
    "tfsi_msd_A2",
    "relative_msd_A2",
    "midpoint_msd_A2",
    "pair_com_msd_A2",
    "mean_displacement_dot_A2",
    "vector_correlation",
    "collective_alignment_index",
    "relative_motion_ratio",
    "mean_directional_cosine",
    "metal_msd_ci_low_A2",
    "metal_msd_ci_high_A2",
    "tfsi_msd_ci_low_A2",
    "tfsi_msd_ci_high_A2",
    "relative_msd_ci_low_A2",
    "relative_msd_ci_high_A2",
    "pair_com_msd_ci_low_A2",
    "pair_com_msd_ci_high_A2",
    "vector_correlation_ci_low",
    "vector_correlation_ci_high",
    "collective_alignment_ci_low",
    "collective_alignment_ci_high",
    "relative_motion_ratio_ci_low",
    "relative_motion_ratio_ci_high",
    "directional_cosine_ci_low",
    "directional_cosine_ci_high",
    "recommended_for_interpretation",
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass(frozen=True)
class AnalysisConfig:
    input_dir: Path
    output_dir: Path

    chunk_size: int
    max_lag_ns: float
    n_log_lags: int
    linear_lag_frames: int

    bootstrap_replicates: int
    bootstrap_seed: int
    ci_level: float

    min_events_for_interpretation: int
    min_origins_for_interpretation: int
    direction_epsilon_A: float
    max_reasonable_step_A: float
    figure_dpi: int

    overwrite: bool = False
    save_event_lag_statistics: bool = True

    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if self.max_lag_ns <= 0:
            raise ValueError("max_lag_ns must be positive.")
        if self.n_log_lags < 5:
            raise ValueError("n_log_lags must be at least 5.")
        if self.linear_lag_frames < 1:
            raise ValueError("linear_lag_frames must be positive.")
        if self.bootstrap_replicates < 0:
            raise ValueError("bootstrap_replicates cannot be negative.")
        if not 0.0 < self.ci_level < 1.0:
            raise ValueError("ci_level must lie between 0 and 1.")
        if self.min_events_for_interpretation < 1:
            raise ValueError(
                "min_events_for_interpretation must be positive."
            )
        if self.min_origins_for_interpretation < 1:
            raise ValueError(
                "min_origins_for_interpretation must be positive."
            )
        if self.direction_epsilon_A <= 0:
            raise ValueError("direction_epsilon_A must be positive.")
        if self.max_reasonable_step_A <= 0:
            raise ValueError("max_reasonable_step_A must be positive.")
        if self.figure_dpi < 72:
            raise ValueError("figure_dpi must be at least 72.")


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
        if self.errors or (self.checks and not all(self.checks.values())):
            self.status = "FAILED"
        elif self.warnings:
            self.status = "PASSED_WITH_WARNINGS"
        elif self.checks:
            self.status = "PASSED"
        else:
            self.status = "NOT_RUN"

    def to_dict(self) -> Dict[str, Any]:
        self.finalize()
        return make_json_safe(asdict(self))


@dataclass(frozen=True)
class SystemMetadata:
    system: str
    metal_species: str
    composition: str
    temperature_K: float
    frame_interval_ps: float
    stride: int
    effective_frame_interval_ps: float
    first_frame: int
    last_frame: int
    n_selected_frames: int
    metal_mass_g_mol: float
    tfsi_mass_g_mol: float


@dataclass
class CoordinateArrays:
    event_index: np.ndarray
    frame: np.ndarray
    metal_position: np.ndarray
    tfsi_position: np.ndarray
    box_dimensions: np.ndarray
    n_rows: int


class OutputPaths:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.figures = self.root / "figures"

        self.all_pairs = self.root / "pair_comotion_all.csv"
        self.by_lifetime = self.root / "pair_comotion_by_lifetime.csv"
        self.by_formation = self.root / "pair_comotion_by_formation.csv"
        self.by_coordination = (
            self.root / "pair_comotion_by_coordination.csv"
        )
        self.event_lag_statistics = (
            self.root / "pair_comotion_event_lag_statistics.csv.gz"
        )
        self.event_summary = (
            self.root / "pair_comotion_event_summary.csv"
        )
        self.lag_grid = self.root / "lag_grid.csv"
        self.validation_report = self.root / "validation_report.json"
        self.summary = self.root / "summary.json"
        self.log_file = self.root / "05E2_pair_comotion.log"

        self.figure_all_msd = (
            self.figures / "all_pair_msd_components.png"
        )
        self.figure_all_correlation = (
            self.figures / "all_pair_correlation_metrics.png"
        )
        self.figure_lifetime_alignment = (
            self.figures / "lifetime_alignment_comparison.png"
        )
        self.figure_lifetime_relative = (
            self.figures / "lifetime_relative_motion_comparison.png"
        )
        self.figure_formation_alignment = (
            self.figures / "formation_alignment_comparison.png"
        )
        self.figure_coordination_alignment = (
            self.figures / "coordination_alignment_comparison.png"
        )
        self.figure_sample_support = (
            self.figures / "lag_sample_support.png"
        )

    def all_output_files(self) -> List[Path]:
        return [
            self.all_pairs,
            self.by_lifetime,
            self.by_formation,
            self.by_coordination,
            self.event_lag_statistics,
            self.event_summary,
            self.lag_grid,
            self.validation_report,
            self.summary,
            self.log_file,
            self.figure_all_msd,
            self.figure_all_correlation,
            self.figure_lifetime_alignment,
            self.figure_lifetime_relative,
            self.figure_formation_alignment,
            self.figure_coordination_alignment,
            self.figure_sample_support,
        ]

    def prepare(self, overwrite: bool) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.figures.mkdir(parents=True, exist_ok=True)

        existing = [path for path in self.all_output_files() if path.exists()]
        if existing and not overwrite:
            listing = "\n".join(f"  - {path}" for path in existing)
            raise FileExistsError(
                "Module 5E2 output files already exist. "
                "Use --overwrite to replace them:\n"
                f"{listing}"
            )

        if overwrite:
            for path in existing:
                path.unlink()


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def configure_logging(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("05E2_pair_comotion")
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
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(make_json_safe(payload), handle, indent=2)
        handle.write("\n")
    temporary.replace(path)


def require_columns(
    columns: Iterable[str],
    required: Sequence[str],
    label: str,
) -> None:
    available = set(columns)
    missing = [column for column in required if column not in available]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def percentile_interval(
    values: np.ndarray,
    ci_level: float,
) -> Tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return np.nan, np.nan

    alpha = 1.0 - ci_level
    return (
        float(np.percentile(array, 100.0 * alpha / 2.0)),
        float(np.percentile(array, 100.0 * (1.0 - alpha / 2.0))),
    )


def classify_lifetime(occupancy_time_ps: float) -> str:
    value = float(occupancy_time_ps)
    if value <= 50.0:
        return "transient_10_50_ps"
    if value <= 500.0:
        return "short_50_500_ps"
    if value <= 5000.0:
        return "intermediate_0.5_5_ns"
    return "persistent_gt_5_ns"


def classify_formation(reformation_event: int) -> str:
    return (
        "reformation_event"
        if int(reformation_event) == 1
        else "first_formation_event"
    )


def classify_coordination(
    fraction_multi_oxygen_frames: float,
) -> str:
    fraction = float(fraction_multi_oxygen_frames)
    if fraction <= 0.0:
        return "strictly_monodentate"
    if fraction <= 0.25:
        return "occasional_multioxygen"
    return "multioxygen_enriched"


def format_runtime(seconds: float) -> str:
    seconds_int = int(round(seconds))
    hours, remainder = divmod(seconds_int, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    return f"{minutes:d}m {secs:02d}s"


# =============================================================================
# INPUT AND METADATA
# =============================================================================

def resolve_input_paths(input_dir: Path) -> Dict[str, Path]:
    paths = {
        "pair_database": input_dir / "pair_database.csv.gz",
        "event_summary": input_dir / "pair_event_summary.csv",
        "metadata": input_dir / "metadata.json",
        "final_validation": input_dir / "final_validation_report.json",
    }

    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        listing = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Required Module 5E1 outputs are missing:\n"
            f"{listing}"
        )

    return paths


def resolve_system_metadata(
    metadata: Mapping[str, Any],
    metal_mass_g_mol: float,
) -> SystemMetadata:
    identity = metadata.get("identity", {})
    configuration = metadata.get("configuration", {})
    selected = metadata.get("selected_frame_summary", {})

    metal_species = str(identity.get("metal_species", "metal"))

    frame_interval_ps = float(
        configuration.get("frame_interval_ps", 10.0)
    )
    stride = int(
        configuration.get("stride", selected.get("stride", 1))
    )

    return SystemMetadata(
        system=str(identity.get("system", "unknown")),
        metal_species=metal_species,
        composition=str(identity.get("composition", "unknown")),
        temperature_K=float(identity.get("temperature_K", np.nan)),
        frame_interval_ps=frame_interval_ps,
        stride=stride,
        effective_frame_interval_ps=frame_interval_ps * stride,
        first_frame=int(selected.get("first_frame", 0)),
        last_frame=int(selected.get("last_frame", 0)),
        n_selected_frames=int(selected.get("n_frames", 0)),
        metal_mass_g_mol=float(metal_mass_g_mol),
        tfsi_mass_g_mol=TFSI_MASS_G_MOL,
    )


def load_event_summary(path: Path) -> pd.DataFrame:
    event_df = pd.read_csv(path, low_memory=False)
    require_columns(
        event_df.columns,
        EVENT_SUMMARY_REQUIRED_COLUMNS,
        "pair_event_summary.csv",
    )

    event_df = event_df.copy()
    event_df["pair_instance_id"] = (
        event_df["pair_instance_id"].astype(str)
    )
    event_df["lifetime_class"] = event_df[
        "occupancy_time_ps"
    ].map(classify_lifetime)
    event_df["formation_class"] = event_df[
        "reformation_event"
    ].map(classify_formation)
    event_df["coordination_class"] = event_df[
        "fraction_multi_oxygen_frames"
    ].map(classify_coordination)

    event_df.sort_values(
        ["first_frame", "metal_id", "tfsi_resid", "event_number"],
        inplace=True,
        ignore_index=True,
    )
    event_df["event_index"] = np.arange(
        len(event_df),
        dtype=np.int32,
    )

    return event_df


def build_lag_grid(
    max_event_frames: int,
    effective_dt_ps: float,
    max_lag_ns: float,
    n_log_lags: int,
    linear_lag_frames: int,
) -> np.ndarray:
    requested_max_frames = int(
        math.floor(max_lag_ns * 1000.0 / effective_dt_ps)
    )
    maximum_valid_lag = min(
        requested_max_frames,
        max_event_frames - 1,
    )

    if maximum_valid_lag < 1:
        raise ValueError(
            "No event contains enough frames for displacement analysis."
        )

    linear_max = min(linear_lag_frames, maximum_valid_lag)
    linear = np.arange(1, linear_max + 1, dtype=np.int32)

    if maximum_valid_lag <= linear_max:
        return linear

    logarithmic = np.unique(
        np.rint(
            np.logspace(
                np.log10(linear_max + 1),
                np.log10(maximum_valid_lag),
                n_log_lags,
            )
        ).astype(np.int32)
    )
    logarithmic = logarithmic[
        (logarithmic > linear_max)
        & (logarithmic <= maximum_valid_lag)
    ]

    return np.unique(
        np.concatenate([linear, logarithmic])
    ).astype(np.int32)


# =============================================================================
# MEMORY-EFFICIENT DATABASE LOADING
# =============================================================================

def allocate_coordinate_arrays(n_rows: int) -> CoordinateArrays:
    return CoordinateArrays(
        event_index=np.empty(n_rows, dtype=np.int32),
        frame=np.empty(n_rows, dtype=np.int32),
        metal_position=np.empty((n_rows, 3), dtype=np.float64),
        tfsi_position=np.empty((n_rows, 3), dtype=np.float64),
        box_dimensions=np.empty((n_rows, 6), dtype=np.float32),
        n_rows=n_rows,
    )


def load_coordinate_arrays(
    pair_database: Path,
    event_df: pd.DataFrame,
    config: AnalysisConfig,
    logger: logging.Logger,
) -> CoordinateArrays:
    expected_rows = int(event_df["n_observed_frames"].sum())
    arrays = allocate_coordinate_arrays(expected_rows)

    event_index_map = pd.Series(
        event_df["event_index"].to_numpy(dtype=np.int32),
        index=event_df["pair_instance_id"].astype(str),
    )

    dtypes = {
        "pair_instance_id": "string",
        "frame": "int32",
        "metal_x_A": "float64",
        "metal_y_A": "float64",
        "metal_z_A": "float64",
        "tfsi_x_A": "float64",
        "tfsi_y_A": "float64",
        "tfsi_z_A": "float64",
        "box_a_A": "float32",
        "box_b_A": "float32",
        "box_c_A": "float32",
        "box_alpha_deg": "float32",
        "box_beta_deg": "float32",
        "box_gamma_deg": "float32",
    }

    cursor = 0
    start_time = time.perf_counter()

    reader = pd.read_csv(
        pair_database,
        compression="gzip",
        usecols=PAIR_DATABASE_COLUMNS,
        dtype=dtypes,
        chunksize=config.chunk_size,
        low_memory=False,
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        count = len(chunk)
        end = cursor + count

        if end > expected_rows:
            raise ValueError(
                "pair_database.csv.gz contains more rows than expected "
                "from pair_event_summary.csv."
            )

        mapped = chunk["pair_instance_id"].astype(str).map(
            event_index_map
        )
        if mapped.isna().any():
            unknown = (
                chunk.loc[mapped.isna(), "pair_instance_id"]
                .astype(str)
                .drop_duplicates()
                .head(10)
                .tolist()
            )
            raise ValueError(
                "Pair database contains pair_instance_id values absent from "
                f"pair_event_summary.csv: {unknown}"
            )

        arrays.event_index[cursor:end] = mapped.to_numpy(
            dtype=np.int32
        )
        arrays.frame[cursor:end] = chunk["frame"].to_numpy(
            dtype=np.int32
        )
        arrays.metal_position[cursor:end, :] = chunk[
            ["metal_x_A", "metal_y_A", "metal_z_A"]
        ].to_numpy(dtype=np.float64)
        arrays.tfsi_position[cursor:end, :] = chunk[
            ["tfsi_x_A", "tfsi_y_A", "tfsi_z_A"]
        ].to_numpy(dtype=np.float64)
        arrays.box_dimensions[cursor:end, :] = chunk[
            [
                "box_a_A",
                "box_b_A",
                "box_c_A",
                "box_alpha_deg",
                "box_beta_deg",
                "box_gamma_deg",
            ]
        ].to_numpy(dtype=np.float32)

        cursor = end
        elapsed = time.perf_counter() - start_time
        logger.info(
            "Loaded coordinate chunk %d | rows=%d/%d | elapsed=%s",
            chunk_number,
            cursor,
            expected_rows,
            format_runtime(elapsed),
        )

    if cursor != expected_rows:
        raise ValueError(
            f"Loaded {cursor} pair rows, expected {expected_rows}."
        )

    logger.info(
        "Coordinate arrays loaded in %s",
        format_runtime(time.perf_counter() - start_time),
    )
    return arrays


def sort_coordinate_arrays(
    arrays: CoordinateArrays,
    logger: logging.Logger,
) -> None:
    logger.info("Sorting pair rows by event and frame...")
    start = time.perf_counter()

    order = np.lexsort((arrays.frame, arrays.event_index))

    arrays.event_index = arrays.event_index[order]
    arrays.frame = arrays.frame[order]
    arrays.metal_position = arrays.metal_position[order, :]
    arrays.tfsi_position = arrays.tfsi_position[order, :]
    arrays.box_dimensions = arrays.box_dimensions[order, :]

    logger.info(
        "Pair rows sorted in %s",
        format_runtime(time.perf_counter() - start),
    )


# =============================================================================
# TRICLINIC BOX AND UNWRAPPING
# =============================================================================

def box_matrix_from_dimensions(
    dimensions: np.ndarray,
) -> np.ndarray:
    """
    Construct a triclinic box matrix whose rows are the box vectors.

    Input:
        [a, b, c, alpha, beta, gamma]

    Angles:
        alpha = angle(b, c)
        beta  = angle(a, c)
        gamma = angle(a, b)
    """

    a, b, c, alpha_deg, beta_deg, gamma_deg = [
        float(value) for value in dimensions
    ]

    if min(a, b, c) <= 0:
        raise ValueError(f"Invalid box lengths: {dimensions[:3]}")

    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)
    gamma = math.radians(gamma_deg)

    sin_gamma = math.sin(gamma)
    if abs(sin_gamma) < 1.0e-12:
        raise ValueError(
            f"Invalid triclinic box angle gamma={gamma_deg}"
        )

    a_vector = np.array([a, 0.0, 0.0], dtype=np.float64)
    b_vector = np.array(
        [b * math.cos(gamma), b * sin_gamma, 0.0],
        dtype=np.float64,
    )

    c_x = c * math.cos(beta)
    c_y = c * (
        math.cos(alpha)
        - math.cos(beta) * math.cos(gamma)
    ) / sin_gamma

    c_z_sq = c * c - c_x * c_x - c_y * c_y
    if c_z_sq < -1.0e-7:
        raise ValueError(
            f"Invalid box geometry gives c_z²={c_z_sq}"
        )
    c_z = math.sqrt(max(0.0, c_z_sq))
    c_vector = np.array([c_x, c_y, c_z], dtype=np.float64)

    matrix = np.vstack([a_vector, b_vector, c_vector])

    determinant = float(np.linalg.det(matrix))
    if not np.isfinite(determinant) or abs(determinant) < 1.0e-10:
        raise ValueError("Singular periodic box matrix.")

    return matrix


def minimum_image_delta(
    delta_cartesian: np.ndarray,
    dimensions: np.ndarray,
) -> np.ndarray:
    matrix = box_matrix_from_dimensions(dimensions)
    inverse = np.linalg.inv(matrix)

    fractional = np.asarray(delta_cartesian, dtype=np.float64) @ inverse
    fractional -= np.rint(fractional)

    return fractional @ matrix


def unwrap_event_positions(
    wrapped_positions: np.ndarray,
    box_dimensions: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Unwrap one continuous pair-event trajectory.

    Orthorhombic boxes are handled with a fully vectorized minimum-image
    calculation. A general triclinic fallback is retained for tilted boxes.
    """

    n_frames = len(wrapped_positions)
    unwrapped = np.empty_like(wrapped_positions, dtype=np.float64)

    if n_frames == 0:
        return unwrapped, 0.0

    unwrapped[0] = wrapped_positions[0]

    if n_frames == 1:
        return unwrapped, 0.0

    wrapped_deltas = np.diff(
        wrapped_positions,
        axis=0,
    ).astype(np.float64, copy=False)
    angles = np.asarray(
        box_dimensions[:, 3:6],
        dtype=np.float64,
    )
    orthorhombic = bool(
        np.allclose(
            angles,
            90.0,
            rtol=0.0,
            atol=1.0e-4,
        )
    )

    if orthorhombic:
        averaged_lengths = 0.5 * (
            np.asarray(box_dimensions[1:, :3], dtype=np.float64)
            + np.asarray(box_dimensions[:-1, :3], dtype=np.float64)
        )

        if np.any(~np.isfinite(averaged_lengths)) or np.any(
            averaged_lengths <= 0
        ):
            raise ValueError(
                "Invalid orthorhombic box lengths during unwrapping."
            )

        mic_deltas = wrapped_deltas - averaged_lengths * np.rint(
            wrapped_deltas / averaged_lengths
        )
    else:
        mic_deltas = np.empty_like(wrapped_deltas)

        for index, delta in enumerate(wrapped_deltas):
            averaged_box = 0.5 * (
                box_dimensions[index + 1]
                + box_dimensions[index]
            )
            mic_deltas[index] = minimum_image_delta(
                delta,
                averaged_box,
            )

    unwrapped[1:] = (
        wrapped_positions[0]
        + np.cumsum(mic_deltas, axis=0)
    )

    step_lengths = np.linalg.norm(mic_deltas, axis=1)
    maximum_step = (
        float(np.max(step_lengths))
        if step_lengths.size
        else 0.0
    )

    return unwrapped, maximum_step


# =============================================================================
# EVENT-LEVEL CO-MOTION CALCULATION
# =============================================================================

def compute_event_lag_statistics(
    event_row: pd.Series,
    frames: np.ndarray,
    metal_wrapped: np.ndarray,
    tfsi_wrapped: np.ndarray,
    box_dimensions: np.ndarray,
    lag_frames: np.ndarray,
    metadata: SystemMetadata,
    direction_epsilon_A: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    metal_unwrapped, max_metal_step = unwrap_event_positions(
        metal_wrapped,
        box_dimensions,
    )
    tfsi_unwrapped, max_tfsi_step = unwrap_event_positions(
        tfsi_wrapped,
        box_dimensions,
    )

    n_frames = len(frames)
    expected_frames = np.arange(
        frames[0],
        frames[0] + n_frames * metadata.stride,
        metadata.stride,
        dtype=np.int32,
    )
    frame_continuity_valid = bool(
        np.array_equal(frames, expected_frames)
    )

    metal_mass = metadata.metal_mass_g_mol
    tfsi_mass = metadata.tfsi_mass_g_mol
    total_mass = metal_mass + tfsi_mass

    records: List[Dict[str, Any]] = []

    for lag in lag_frames:
        lag_int = int(lag)
        if lag_int >= n_frames:
            continue

        metal_displacement = (
            metal_unwrapped[lag_int:]
            - metal_unwrapped[:-lag_int]
        )
        tfsi_displacement = (
            tfsi_unwrapped[lag_int:]
            - tfsi_unwrapped[:-lag_int]
        )

        relative_displacement = (
            metal_displacement - tfsi_displacement
        )
        midpoint_displacement = 0.5 * (
            metal_displacement + tfsi_displacement
        )
        pair_com_displacement = (
            metal_mass * metal_displacement
            + tfsi_mass * tfsi_displacement
        ) / total_mass

        metal_sq = np.einsum(
            "ij,ij->i",
            metal_displacement,
            metal_displacement,
        )
        tfsi_sq = np.einsum(
            "ij,ij->i",
            tfsi_displacement,
            tfsi_displacement,
        )
        relative_sq = np.einsum(
            "ij,ij->i",
            relative_displacement,
            relative_displacement,
        )
        midpoint_sq = np.einsum(
            "ij,ij->i",
            midpoint_displacement,
            midpoint_displacement,
        )
        pair_com_sq = np.einsum(
            "ij,ij->i",
            pair_com_displacement,
            pair_com_displacement,
        )
        dot = np.einsum(
            "ij,ij->i",
            metal_displacement,
            tfsi_displacement,
        )

        magnitude_product = np.sqrt(metal_sq * tfsi_sq)
        valid_direction = magnitude_product > (
            direction_epsilon_A**2
        )

        if np.any(valid_direction):
            directional_cosine = (
                dot[valid_direction]
                / magnitude_product[valid_direction]
            )
            sum_directional_cosine = float(
                np.sum(directional_cosine)
            )
            n_directional_cosine = int(
                directional_cosine.size
            )
        else:
            sum_directional_cosine = 0.0
            n_directional_cosine = 0

        records.append(
            {
                "system": metadata.system,
                "metal_species": metadata.metal_species,
                "composition": metadata.composition,
                "temperature_K": metadata.temperature_K,
                "pair_instance_id": str(
                    event_row["pair_instance_id"]
                ),
                "pair_id": str(event_row["pair_id"]),
                "event_number": int(event_row["event_number"]),
                "reformation_event": int(
                    event_row["reformation_event"]
                ),
                "formation_class": str(
                    event_row["formation_class"]
                ),
                "completed_event": int(
                    event_row["completed_event"]
                ),
                "left_censored": int(
                    event_row["left_censored"]
                ),
                "right_censored": int(
                    event_row["right_censored"]
                ),
                "lifetime_class": str(
                    event_row["lifetime_class"]
                ),
                "coordination_class": str(
                    event_row["coordination_class"]
                ),
                "occupancy_time_ps": float(
                    event_row["occupancy_time_ps"]
                ),
                "occupancy_time_ns": float(
                    event_row["occupancy_time_ns"]
                ),
                "event_n_frames": n_frames,
                "lag_frames": lag_int,
                "lag_ps": (
                    lag_int
                    * metadata.effective_frame_interval_ps
                ),
                "lag_ns": (
                    lag_int
                    * metadata.effective_frame_interval_ps
                    / 1000.0
                ),
                "n_time_origins": int(len(metal_sq)),
                "sum_metal_displacement_sq_A2": float(
                    np.sum(metal_sq)
                ),
                "sum_tfsi_displacement_sq_A2": float(
                    np.sum(tfsi_sq)
                ),
                "sum_relative_displacement_sq_A2": float(
                    np.sum(relative_sq)
                ),
                "sum_midpoint_displacement_sq_A2": float(
                    np.sum(midpoint_sq)
                ),
                "sum_pair_com_displacement_sq_A2": float(
                    np.sum(pair_com_sq)
                ),
                "sum_displacement_dot_A2": float(
                    np.sum(dot)
                ),
                "sum_directional_cosine": (
                    sum_directional_cosine
                ),
                "n_directional_cosine": (
                    n_directional_cosine
                ),
            }
        )

    net_metal = metal_unwrapped[-1] - metal_unwrapped[0]
    net_tfsi = tfsi_unwrapped[-1] - tfsi_unwrapped[0]
    net_relative = net_metal - net_tfsi
    net_pair_com = (
        metal_mass * net_metal + tfsi_mass * net_tfsi
    ) / total_mass

    metal_steps = np.diff(metal_unwrapped, axis=0)
    tfsi_steps = np.diff(tfsi_unwrapped, axis=0)

    metal_path_length = (
        float(np.sum(np.linalg.norm(metal_steps, axis=1)))
        if n_frames > 1
        else 0.0
    )
    tfsi_path_length = (
        float(np.sum(np.linalg.norm(tfsi_steps, axis=1)))
        if n_frames > 1
        else 0.0
    )

    net_metal_norm = float(np.linalg.norm(net_metal))
    net_tfsi_norm = float(np.linalg.norm(net_tfsi))
    net_denominator = net_metal_norm * net_tfsi_norm

    event_summary = {
        "system": metadata.system,
        "metal_species": metadata.metal_species,
        "composition": metadata.composition,
        "temperature_K": metadata.temperature_K,
        "pair_instance_id": str(event_row["pair_instance_id"]),
        "pair_id": str(event_row["pair_id"]),
        "event_number": int(event_row["event_number"]),
        "reformation_event": int(event_row["reformation_event"]),
        "formation_class": str(event_row["formation_class"]),
        "completed_event": int(event_row["completed_event"]),
        "left_censored": int(event_row["left_censored"]),
        "right_censored": int(event_row["right_censored"]),
        "lifetime_class": str(event_row["lifetime_class"]),
        "coordination_class": str(
            event_row["coordination_class"]
        ),
        "occupancy_time_ps": float(event_row["occupancy_time_ps"]),
        "occupancy_time_ns": float(event_row["occupancy_time_ns"]),
        "n_frames": n_frames,
        "first_frame": int(frames[0]),
        "last_frame": int(frames[-1]),
        "frame_continuity_valid": int(
            frame_continuity_valid
        ),
        "maximum_metal_step_A": max_metal_step,
        "maximum_tfsi_step_A": max_tfsi_step,
        "metal_path_length_A": metal_path_length,
        "tfsi_path_length_A": tfsi_path_length,
        "net_metal_displacement_A": net_metal_norm,
        "net_tfsi_displacement_A": net_tfsi_norm,
        "net_relative_displacement_A": float(
            np.linalg.norm(net_relative)
        ),
        "net_pair_com_displacement_A": float(
            np.linalg.norm(net_pair_com)
        ),
        "net_displacement_dot_A2": float(
            np.dot(net_metal, net_tfsi)
        ),
        "net_directional_cosine": (
            float(np.dot(net_metal, net_tfsi) / net_denominator)
            if net_denominator > direction_epsilon_A**2
            else np.nan
        ),
        "mean_coordinating_oxygens": float(
            event_row["mean_coordinating_oxygens"]
        ),
        "maximum_coordinating_oxygens": int(
            event_row["maximum_coordinating_oxygens"]
        ),
        "fraction_multi_oxygen_frames": float(
            event_row["fraction_multi_oxygen_frames"]
        ),
    }

    return records, event_summary


def process_all_events(
    arrays: CoordinateArrays,
    event_df: pd.DataFrame,
    lag_frames: np.ndarray,
    metadata: SystemMetadata,
    config: AnalysisConfig,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    event_lag_records: List[Dict[str, Any]] = []
    event_summaries: List[Dict[str, Any]] = []

    unique_indices, starts, counts = np.unique(
        arrays.event_index,
        return_index=True,
        return_counts=True,
    )

    expected_event_indices = np.arange(
        len(event_df),
        dtype=np.int32,
    )
    if not np.array_equal(unique_indices, expected_event_indices):
        missing = sorted(
            set(expected_event_indices.tolist())
            - set(unique_indices.tolist())
        )
        raise ValueError(
            f"Coordinate database is missing event indices: {missing[:10]}"
        )

    maximum_metal_step = 0.0
    maximum_tfsi_step = 0.0
    discontinuous_events = 0
    start_time = time.perf_counter()

    for event_counter, (
        event_index,
        start,
        count,
    ) in enumerate(
        zip(unique_indices, starts, counts),
        start=1,
    ):
        stop = int(start + count)
        event_row = event_df.iloc[int(event_index)]

        records, event_summary = compute_event_lag_statistics(
            event_row=event_row,
            frames=arrays.frame[start:stop],
            metal_wrapped=arrays.metal_position[start:stop, :],
            tfsi_wrapped=arrays.tfsi_position[start:stop, :],
            box_dimensions=arrays.box_dimensions[start:stop, :],
            lag_frames=lag_frames,
            metadata=metadata,
            direction_epsilon_A=config.direction_epsilon_A,
        )

        event_lag_records.extend(records)
        event_summaries.append(event_summary)

        maximum_metal_step = max(
            maximum_metal_step,
            event_summary["maximum_metal_step_A"],
        )
        maximum_tfsi_step = max(
            maximum_tfsi_step,
            event_summary["maximum_tfsi_step_A"],
        )
        if event_summary["frame_continuity_valid"] != 1:
            discontinuous_events += 1

        if (
            event_counter == 1
            or event_counter % 250 == 0
            or event_counter == len(event_df)
        ):
            elapsed = time.perf_counter() - start_time
            rate = event_counter / elapsed if elapsed > 0 else np.nan
            remaining = len(event_df) - event_counter
            eta = remaining / rate if rate > 0 else np.nan

            logger.info(
                "Event progress %d/%d | event-lag rows=%d | "
                "elapsed=%s | ETA=%s",
                event_counter,
                len(event_df),
                len(event_lag_records),
                format_runtime(elapsed),
                format_runtime(eta),
            )

    event_lag_df = pd.DataFrame(
        event_lag_records,
        columns=EVENT_LAG_COLUMNS,
    )
    event_summary_df = pd.DataFrame(event_summaries)

    processing_metrics = {
        "n_events_processed": int(len(event_df)),
        "n_event_lag_rows": int(len(event_lag_df)),
        "maximum_metal_frame_step_A": maximum_metal_step,
        "maximum_tfsi_frame_step_A": maximum_tfsi_step,
        "n_discontinuous_events": discontinuous_events,
        "event_processing_seconds": (
            time.perf_counter() - start_time
        ),
    }

    return event_lag_df, event_summary_df, processing_metrics


# =============================================================================
# AGGREGATION AND BOOTSTRAP
# =============================================================================

SUM_COLUMNS = [
    "n_time_origins",
    "sum_metal_displacement_sq_A2",
    "sum_tfsi_displacement_sq_A2",
    "sum_relative_displacement_sq_A2",
    "sum_midpoint_displacement_sq_A2",
    "sum_pair_com_displacement_sq_A2",
    "sum_displacement_dot_A2",
    "sum_directional_cosine",
    "n_directional_cosine",
]


def metrics_from_sums(sums: Mapping[str, float]) -> Dict[str, float]:
    n_origins = float(sums["n_time_origins"])
    n_directional = float(sums["n_directional_cosine"])

    if n_origins <= 0:
        return {
            "metal_msd_A2": np.nan,
            "tfsi_msd_A2": np.nan,
            "relative_msd_A2": np.nan,
            "midpoint_msd_A2": np.nan,
            "pair_com_msd_A2": np.nan,
            "mean_displacement_dot_A2": np.nan,
            "vector_correlation": np.nan,
            "collective_alignment_index": np.nan,
            "relative_motion_ratio": np.nan,
            "mean_directional_cosine": np.nan,
        }

    metal_msd = (
        float(sums["sum_metal_displacement_sq_A2"])
        / n_origins
    )
    tfsi_msd = (
        float(sums["sum_tfsi_displacement_sq_A2"])
        / n_origins
    )
    relative_msd = (
        float(sums["sum_relative_displacement_sq_A2"])
        / n_origins
    )
    midpoint_msd = (
        float(sums["sum_midpoint_displacement_sq_A2"])
        / n_origins
    )
    pair_com_msd = (
        float(sums["sum_pair_com_displacement_sq_A2"])
        / n_origins
    )
    mean_dot = (
        float(sums["sum_displacement_dot_A2"])
        / n_origins
    )

    denominator_correlation = math.sqrt(
        max(0.0, metal_msd * tfsi_msd)
    )
    denominator_alignment = metal_msd + tfsi_msd

    vector_correlation = (
        mean_dot / denominator_correlation
        if denominator_correlation > 0
        else np.nan
    )
    collective_alignment = (
        2.0 * mean_dot / denominator_alignment
        if denominator_alignment > 0
        else np.nan
    )
    relative_motion_ratio = (
        relative_msd / denominator_alignment
        if denominator_alignment > 0
        else np.nan
    )
    mean_directional_cosine = (
        float(sums["sum_directional_cosine"])
        / n_directional
        if n_directional > 0
        else np.nan
    )

    return {
        "metal_msd_A2": metal_msd,
        "tfsi_msd_A2": tfsi_msd,
        "relative_msd_A2": relative_msd,
        "midpoint_msd_A2": midpoint_msd,
        "pair_com_msd_A2": pair_com_msd,
        "mean_displacement_dot_A2": mean_dot,
        "vector_correlation": vector_correlation,
        "collective_alignment_index": collective_alignment,
        "relative_motion_ratio": relative_motion_ratio,
        "mean_directional_cosine": mean_directional_cosine,
    }


def bootstrap_group_metrics(
    group: pd.DataFrame,
    config: AnalysisConfig,
    rng: np.random.Generator,
) -> Dict[str, Tuple[float, float]]:
    metric_names = [
        "metal_msd_A2",
        "tfsi_msd_A2",
        "relative_msd_A2",
        "pair_com_msd_A2",
        "vector_correlation",
        "collective_alignment_index",
        "relative_motion_ratio",
        "mean_directional_cosine",
    ]

    if config.bootstrap_replicates == 0 or len(group) < 2:
        return {
            name: (np.nan, np.nan)
            for name in metric_names
        }

    values = group[SUM_COLUMNS].to_numpy(dtype=np.float64)
    n_events = len(group)

    bootstrap_values = {
        name: np.empty(
            config.bootstrap_replicates,
            dtype=np.float64,
        )
        for name in metric_names
    }

    for replicate in range(config.bootstrap_replicates):
        sampled_indices = rng.integers(
            0,
            n_events,
            size=n_events,
        )
        sampled_sums_array = np.sum(
            values[sampled_indices, :],
            axis=0,
        )
        sampled_sums = dict(
            zip(SUM_COLUMNS, sampled_sums_array)
        )
        metrics = metrics_from_sums(sampled_sums)

        for name in metric_names:
            bootstrap_values[name][replicate] = metrics[name]

    return {
        name: percentile_interval(
            array,
            config.ci_level,
        )
        for name, array in bootstrap_values.items()
    }


def aggregate_event_lag_data(
    event_lag_df: pd.DataFrame,
    group_type: str,
    group_column: Optional[str],
    analysis_scope: str,
    config: AnalysisConfig,
    rng: np.random.Generator,
    metadata: SystemMetadata,
    logger: logging.Logger,
) -> pd.DataFrame:
    if analysis_scope == "completed_only":
        scoped = event_lag_df[
            event_lag_df["completed_event"] == 1
        ].copy()
    elif analysis_scope == "all_observed":
        scoped = event_lag_df
    else:
        raise ValueError(
            f"Unsupported analysis_scope: {analysis_scope}"
        )

    if group_column is None:
        scoped = scoped.copy()
        scoped["_group_label"] = "all_pairs"
        actual_group_column = "_group_label"
    else:
        actual_group_column = group_column

    records: List[Dict[str, Any]] = []
    grouped = scoped.groupby(
        [actual_group_column, "lag_frames"],
        sort=True,
        observed=True,
    )

    n_groups = grouped.ngroups
    start_time = time.perf_counter()

    for group_counter, (
        (group_label, lag_frames),
        group,
    ) in enumerate(grouped, start=1):
        total_sums = group[SUM_COLUMNS].sum()
        metrics = metrics_from_sums(total_sums)

        confidence_intervals = bootstrap_group_metrics(
            group=group,
            config=config,
            rng=rng,
        )

        n_events = int(group["pair_instance_id"].nunique())
        n_origins = int(total_sums["n_time_origins"])

        recommended = int(
            n_events >= config.min_events_for_interpretation
            and n_origins >= config.min_origins_for_interpretation
        )

        record = {
            "system": metadata.system,
            "metal_species": metadata.metal_species,
            "composition": metadata.composition,
            "temperature_K": metadata.temperature_K,
            "analysis_scope": analysis_scope,
            "group_type": group_type,
            "group_label": str(group_label),
            "lag_frames": int(lag_frames),
            "lag_ps": (
                int(lag_frames)
                * metadata.effective_frame_interval_ps
            ),
            "lag_ns": (
                int(lag_frames)
                * metadata.effective_frame_interval_ps
                / 1000.0
            ),
            "n_events": n_events,
            "n_time_origins": n_origins,
            **metrics,
            "metal_msd_ci_low_A2": (
                confidence_intervals["metal_msd_A2"][0]
            ),
            "metal_msd_ci_high_A2": (
                confidence_intervals["metal_msd_A2"][1]
            ),
            "tfsi_msd_ci_low_A2": (
                confidence_intervals["tfsi_msd_A2"][0]
            ),
            "tfsi_msd_ci_high_A2": (
                confidence_intervals["tfsi_msd_A2"][1]
            ),
            "relative_msd_ci_low_A2": (
                confidence_intervals["relative_msd_A2"][0]
            ),
            "relative_msd_ci_high_A2": (
                confidence_intervals["relative_msd_A2"][1]
            ),
            "pair_com_msd_ci_low_A2": (
                confidence_intervals["pair_com_msd_A2"][0]
            ),
            "pair_com_msd_ci_high_A2": (
                confidence_intervals["pair_com_msd_A2"][1]
            ),
            "vector_correlation_ci_low": (
                confidence_intervals["vector_correlation"][0]
            ),
            "vector_correlation_ci_high": (
                confidence_intervals["vector_correlation"][1]
            ),
            "collective_alignment_ci_low": (
                confidence_intervals[
                    "collective_alignment_index"
                ][0]
            ),
            "collective_alignment_ci_high": (
                confidence_intervals[
                    "collective_alignment_index"
                ][1]
            ),
            "relative_motion_ratio_ci_low": (
                confidence_intervals[
                    "relative_motion_ratio"
                ][0]
            ),
            "relative_motion_ratio_ci_high": (
                confidence_intervals[
                    "relative_motion_ratio"
                ][1]
            ),
            "directional_cosine_ci_low": (
                confidence_intervals[
                    "mean_directional_cosine"
                ][0]
            ),
            "directional_cosine_ci_high": (
                confidence_intervals[
                    "mean_directional_cosine"
                ][1]
            ),
            "recommended_for_interpretation": recommended,
        }
        records.append(record)

        if (
            group_counter == 1
            or group_counter % 100 == 0
            or group_counter == n_groups
        ):
            logger.info(
                "Aggregating %s/%s | %d/%d groups | elapsed=%s",
                group_type,
                analysis_scope,
                group_counter,
                n_groups,
                format_runtime(time.perf_counter() - start_time),
            )

    return pd.DataFrame(records, columns=AGGREGATE_COLUMNS)


# =============================================================================
# VALIDATION
# =============================================================================

def validate_results(
    event_df: pd.DataFrame,
    arrays: CoordinateArrays,
    event_lag_df: pd.DataFrame,
    event_summary_df: pd.DataFrame,
    all_curve_df: pd.DataFrame,
    processing_metrics: Mapping[str, Any],
    metadata: SystemMetadata,
    final_5e1_validation: Mapping[str, Any],
    config: AnalysisConfig,
) -> ValidationReport:
    report = ValidationReport()

    acceptable_5e1_status = final_5e1_validation.get(
        "status"
    ) in {"PASSED", "PASSED_WITH_WARNINGS"}

    report.add_check(
        "module_5e1_validation_acceptable",
        acceptable_5e1_status,
    )
    report.add_check(
        "coordinate_row_count_matches_event_summary",
        arrays.n_rows == int(event_df["n_observed_frames"].sum()),
    )
    report.add_check(
        "all_events_processed",
        len(event_summary_df) == len(event_df),
    )
    report.add_check(
        "event_indices_complete",
        arrays.event_index.min() == 0
        and arrays.event_index.max() == len(event_df) - 1,
    )
    report.add_check(
        "all_event_frame_sequences_continuous",
        int(processing_metrics["n_discontinuous_events"]) == 0,
    )
    report.add_check(
        "event_lag_table_nonempty",
        not event_lag_df.empty,
    )
    report.add_check(
        "all_pair_curve_nonempty",
        not all_curve_df.empty,
    )
    report.add_check(
        "all_time_origin_counts_positive",
        bool((event_lag_df["n_time_origins"] > 0).all()),
    )
    report.add_check(
        "all_lag_times_positive",
        bool((event_lag_df["lag_ps"] > 0).all()),
    )
    report.add_check(
        "relative_msd_nonnegative",
        bool(
            (
                all_curve_df["relative_msd_A2"]
                >= -1.0e-10
            ).all()
        ),
    )
    report.add_check(
        "vector_correlation_within_bounds",
        bool(
            (
                all_curve_df["vector_correlation"]
                .dropna()
                .between(-1.000001, 1.000001)
            ).all()
        ),
    )
    report.add_check(
        "collective_alignment_within_bounds",
        bool(
            (
                all_curve_df["collective_alignment_index"]
                .dropna()
                .between(-1.000001, 1.000001)
            ).all()
        ),
    )
    report.add_check(
        "relative_motion_identity_valid",
        bool(
            np.allclose(
                all_curve_df["relative_motion_ratio"],
                1.0
                - all_curve_df[
                    "collective_alignment_index"
                ],
                rtol=1.0e-7,
                atol=1.0e-9,
                equal_nan=True,
            )
        ),
    )

    report.add_metric("n_events", len(event_df))
    report.add_metric("n_coordinate_rows", arrays.n_rows)
    report.add_metric("n_event_lag_rows", len(event_lag_df))
    report.add_metric(
        "n_lag_points",
        int(all_curve_df["lag_frames"].nunique()),
    )
    report.add_metric(
        "maximum_analyzed_lag_ns",
        float(all_curve_df["lag_ns"].max()),
    )
    report.add_metric(
        "maximum_metal_frame_step_A",
        processing_metrics["maximum_metal_frame_step_A"],
    )
    report.add_metric(
        "maximum_tfsi_frame_step_A",
        processing_metrics["maximum_tfsi_frame_step_A"],
    )
    report.add_metric(
        "n_discontinuous_events",
        processing_metrics["n_discontinuous_events"],
    )
    report.add_metric(
        "bootstrap_replicates",
        config.bootstrap_replicates,
    )

    if (
        float(processing_metrics["maximum_metal_frame_step_A"])
        > config.max_reasonable_step_A
    ):
        report.warn(
            "At least one unwrapped metal frame-to-frame step exceeds "
            f"{config.max_reasonable_step_A:.2f} Å. Inspect the event summary "
            "for possible coordinate or periodic-boundary artifacts."
        )

    if (
        float(processing_metrics["maximum_tfsi_frame_step_A"])
        > config.max_reasonable_step_A
    ):
        report.warn(
            "At least one unwrapped TFSI frame-to-frame step exceeds "
            f"{config.max_reasonable_step_A:.2f} Å. Inspect the event summary "
            "for possible coordinate or periodic-boundary artifacts."
        )

    unsupported = all_curve_df[
        all_curve_df["recommended_for_interpretation"] == 0
    ]
    if not unsupported.empty:
        first_unsupported_lag = float(
            unsupported["lag_ns"].min()
        )
        report.warn(
            "Some lag points have fewer than "
            f"{config.min_events_for_interpretation} events or "
            f"{config.min_origins_for_interpretation} time origins. "
            f"The first unsupported all-pair lag is "
            f"{first_unsupported_lag:.4g} ns."
        )

    return report


# =============================================================================
# FIGURES
# =============================================================================

def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 120,
            "savefig.bbox": "tight",
        }
    )


def save_and_close(
    figure: plt.Figure,
    path: Path,
    dpi: int,
) -> None:
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def recommended_subset(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    return dataframe[
        dataframe["recommended_for_interpretation"] == 1
    ].copy()


def plot_all_msd_components(
    all_curve_df: pd.DataFrame,
    outputs: OutputPaths,
    config: AnalysisConfig,
    system_label: str,
) -> None:
    data = recommended_subset(
        all_curve_df[
            all_curve_df["analysis_scope"] == "all_observed"
        ]
    )

    fig, axis = plt.subplots(figsize=(7.4, 5.1))

    for column, label in [
        ("metal_msd_A2", "Metal MSD"),
        ("tfsi_msd_A2", "TFSI-reference MSD"),
        ("relative_msd_A2", "Relative MSD"),
        ("pair_com_msd_A2", "Pair COM MSD"),
    ]:
        positive = data[
            (data["lag_ns"] > 0)
            & (data[column] > 0)
        ]
        axis.plot(
            positive["lag_ns"],
            positive[column],            marker="o",
            markersize=3,
            linewidth=1.2,
            label=label,
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Lag time (ns)")
    axis.set_ylabel("Mean-squared displacement (Å$^2$)")
    axis.set_title(f"{system_label}: surviving-pair displacement components")
    axis.legend(frameon=False)
    axis.grid(alpha=0.25)
    save_and_close(
        fig,
        outputs.figure_all_msd,
        config.figure_dpi,
    )


def plot_all_correlation_metrics(
    all_curve_df: pd.DataFrame,
    outputs: OutputPaths,
    config: AnalysisConfig,
    system_label: str,
) -> None:
    data = recommended_subset(
        all_curve_df[
            all_curve_df["analysis_scope"] == "all_observed"
        ]
    )

    fig, axis = plt.subplots(figsize=(7.4, 5.1))
    for column, label in [
        ("vector_correlation", "Vector correlation"),
        (
            "collective_alignment_index",
            "Collective alignment index",
        ),
        (
            "mean_directional_cosine",
            "Mean directional cosine",
        ),
        (
            "relative_motion_ratio",
            "Relative-motion ratio",
        ),
    ]:
        axis.plot(
            data["lag_ns"],
            data[column],
            marker="o",
            markersize=3,
            linewidth=1.2,
            label=label,
        )

    axis.axhline(0.0, linewidth=0.8)
    axis.axhline(1.0, linewidth=0.8, linestyle="--")
    axis.set_xscale("log")
    axis.set_xlabel("Lag time (ns)")
    axis.set_ylabel("Dimensionless metric")
    axis.set_title(f"{system_label}: pair co-motion metrics")
    axis.legend(frameon=False)
    axis.grid(alpha=0.25)
    save_and_close(
        fig,
        outputs.figure_all_correlation,
        config.figure_dpi,
    )


def plot_group_metric(
    grouped_df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    config: AnalysisConfig,
) -> None:
    data = grouped_df[
        grouped_df["analysis_scope"] == "all_observed"
    ]

    fig, axis = plt.subplots(figsize=(7.6, 5.2))

    for label, group in data.groupby(
        "group_label",
        sort=True,
    ):
        supported = recommended_subset(group)
        if supported.empty:
            continue
        axis.plot(
            supported["lag_ns"],
            supported[metric],
            marker="o",
            markersize=3,
            linewidth=1.2,
            label=str(label),
        )

    axis.set_xscale("log")
    axis.set_xlabel("Lag time (ns)")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.legend(frameon=False)
    axis.grid(alpha=0.25)
    save_and_close(fig, output_path, config.figure_dpi)


def plot_sample_support(
    all_curve_df: pd.DataFrame,
    outputs: OutputPaths,
    config: AnalysisConfig,
    system_label: str,
) -> None:
    data = all_curve_df[
        all_curve_df["analysis_scope"] == "all_observed"
    ]

    fig, axis = plt.subplots(figsize=(7.2, 4.9))
    axis.plot(
        data["lag_ns"],
        data["n_events"],
        marker="o",
        markersize=3,
        linewidth=1.2,
        label="Contributing pair events",
    )
    axis.plot(
        data["lag_ns"],
        data["n_time_origins"],
        marker="o",
        markersize=3,
        linewidth=1.2,
        label="Time origins",
    )
    axis.axhline(
        config.min_events_for_interpretation,
        linestyle="--",
        linewidth=0.9,
        label="Minimum event threshold",
    )
    axis.axhline(
        config.min_origins_for_interpretation,
        linestyle=":",
        linewidth=0.9,
        label="Minimum origin threshold",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Lag time (ns)")
    axis.set_ylabel("Sample count")
    axis.set_title(f"{system_label}: lag-time statistical support")
    axis.legend(frameon=False)
    axis.grid(alpha=0.25)
    save_and_close(
        fig,
        outputs.figure_sample_support,
        config.figure_dpi,
    )


# =============================================================================
# SUMMARY HELPERS
# =============================================================================

def nearest_supported_lag_row(
    dataframe: pd.DataFrame,
    target_lag_ns: float,
) -> Optional[pd.Series]:
    supported = dataframe[
        (dataframe["analysis_scope"] == "all_observed")
        & (dataframe["recommended_for_interpretation"] == 1)
    ]
    if supported.empty:
        return None

    index = (
        supported["lag_ns"] - target_lag_ns
    ).abs().idxmin()
    return supported.loc[index]


def summarize_key_lags(
    all_curve_df: pd.DataFrame,
    targets_ns: Sequence[float],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for target in targets_ns:
        row = nearest_supported_lag_row(
            all_curve_df,
            target,
        )
        if row is None:
            continue

        records.append(
            {
                "requested_lag_ns": float(target),
                "actual_lag_ns": float(row["lag_ns"]),
                "n_events": int(row["n_events"]),
                "n_time_origins": int(row["n_time_origins"]),
                "metal_msd_A2": float(row["metal_msd_A2"]),
                "tfsi_msd_A2": float(row["tfsi_msd_A2"]),
                "relative_msd_A2": float(
                    row["relative_msd_A2"]
                ),
                "pair_com_msd_A2": float(
                    row["pair_com_msd_A2"]
                ),
                "vector_correlation": float(
                    row["vector_correlation"]
                ),
                "collective_alignment_index": float(
                    row["collective_alignment_index"]
                ),
                "relative_motion_ratio": float(
                    row["relative_motion_ratio"]
                ),
                "mean_directional_cosine": float(
                    row["mean_directional_cosine"]
                ),
            }
        )

    return records


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate lag-dependent metal–TFSI pair co-motion from the "
            "validated Module 5E1 pair database."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(DEFAULT_INPUT_DIR),
        help="Directory containing Module 5E1 outputs.",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default=DEFAULT_OUTPUT_SUBDIR,
        help="Subdirectory created inside input-dir.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Rows per chunk while reading pair_database.csv.gz.",
    )
    parser.add_argument(
        "--max-lag-ns",
        type=float,
        default=DEFAULT_MAX_LAG_NS,
        help="Maximum lag time requested.",
    )
    parser.add_argument(
        "--n-log-lags",
        type=int,
        default=DEFAULT_N_LOG_LAGS,
        help="Number of logarithmically spaced lag candidates.",
    )
    parser.add_argument(
        "--linear-lag-frames",
        type=int,
        default=DEFAULT_LINEAR_LAG_FRAMES,
        help="Include every lag from 1 through this many frames.",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
        help="Event-level bootstrap replicates per group and lag.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
        help="Random seed for bootstrap confidence intervals.",
    )
    parser.add_argument(
        "--ci-level",
        type=float,
        default=DEFAULT_CI_LEVEL,
        help="Bootstrap confidence level.",
    )
    parser.add_argument(
        "--min-events",
        type=int,
        default=DEFAULT_MIN_EVENTS_FOR_INTERPRETATION,
        help="Minimum contributing events for an interpretable lag.",
    )
    parser.add_argument(
        "--min-origins",
        type=int,
        default=DEFAULT_MIN_ORIGINS_FOR_INTERPRETATION,
        help="Minimum displacement time origins for interpretation.",
    )
    parser.add_argument(
        "--max-reasonable-step-A",
        type=float,
        default=DEFAULT_MAX_REASONABLE_STEP_A,
        help="Warn if an unwrapped one-frame step exceeds this value.",
    )
    parser.add_argument(
        "--metal-mass-g-mol",
        type=float,
        default=DEFAULT_METAL_MASS_G_MOL,
        help="Metal-ion molar mass used only for mass-weighted pair COM motion.",
    )
    parser.add_argument(
        "--figure-dpi",
        type=int,
        default=DEFAULT_FIGURE_DPI,
        help="PNG figure resolution.",
    )
    parser.add_argument(
        "--do-not-save-event-lag-statistics",
        action="store_true",
        help="Do not write the compressed event-by-lag sufficient-statistics file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing Module 5E2 outputs.",
    )

    return parser


def config_from_args(args: argparse.Namespace) -> AnalysisConfig:
    input_dir = args.input_dir.resolve()
    output_dir = input_dir / args.output_subdir

    config = AnalysisConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        chunk_size=int(args.chunk_size),
        max_lag_ns=float(args.max_lag_ns),
        n_log_lags=int(args.n_log_lags),
        linear_lag_frames=int(args.linear_lag_frames),
        bootstrap_replicates=int(args.bootstrap_replicates),
        bootstrap_seed=int(args.bootstrap_seed),
        ci_level=float(args.ci_level),
        min_events_for_interpretation=int(args.min_events),
        min_origins_for_interpretation=int(args.min_origins),
        direction_epsilon_A=DEFAULT_DIRECTION_EPSILON_A,
        max_reasonable_step_A=float(
            args.max_reasonable_step_A
        ),
        figure_dpi=int(args.figure_dpi),
        overwrite=bool(args.overwrite),
        save_event_lag_statistics=not bool(
            args.do_not_save_event_lag_statistics
        ),
    )
    config.validate()
    return config


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = config_from_args(args)

    input_paths = resolve_input_paths(config.input_dir)
    outputs = OutputPaths(config.output_dir)
    outputs.prepare(overwrite=config.overwrite)
    logger = configure_logging(outputs.log_file)

    overall_start = time.perf_counter()

    logger.info("=" * 79)
    logger.info("MODULE 5E2: METAL–TFSI PAIR CO-MOTION ANALYSIS")
    logger.info("=" * 79)
    logger.info("Input directory     : %s", config.input_dir)
    logger.info("Output directory    : %s", config.output_dir)
    logger.info("Maximum lag         : %.3f ns", config.max_lag_ns)
    logger.info(
        "Bootstrap replicates: %d",
        config.bootstrap_replicates,
    )

    metadata_json = read_json(input_paths["metadata"])
    final_5e1_validation = read_json(
        input_paths["final_validation"]
    )
    metadata = resolve_system_metadata(metadata_json, args.metal_mass_g_mol)

    logger.info("System              : %s", metadata.system)
    logger.info("Metal species       : %s", metadata.metal_species)
    logger.info("Composition         : %s", metadata.composition)
    logger.info(
        "Temperature         : %.2f metal",
        metadata.temperature_K,
    )
    logger.info(
        "Effective frame dt  : %.3f ps",
        metadata.effective_frame_interval_ps,
    )
    logger.info(
        "Metal mass          : %.5f g/mol",
        metadata.metal_mass_g_mol,
    )
    logger.info(
        "TFSI mass           : %.5f g/mol",
        metadata.tfsi_mass_g_mol,
    )

    event_df = load_event_summary(
        input_paths["event_summary"]
    )
    logger.info("Pair events         : %d", len(event_df))
    logger.info(
        "Pair database rows  : %d",
        int(event_df["n_observed_frames"].sum()),
    )

    lag_frames = build_lag_grid(
        max_event_frames=int(
            event_df["n_observed_frames"].max()
        ),
        effective_dt_ps=metadata.effective_frame_interval_ps,
        max_lag_ns=config.max_lag_ns,
        n_log_lags=config.n_log_lags,
        linear_lag_frames=config.linear_lag_frames,
    )

    lag_grid_df = pd.DataFrame(
        {
            "lag_frames": lag_frames,
            "lag_ps": (
                lag_frames
                * metadata.effective_frame_interval_ps
            ),
            "lag_ns": (
                lag_frames
                * metadata.effective_frame_interval_ps
                / 1000.0
            ),
        }
    )
    lag_grid_df.to_csv(
        outputs.lag_grid,
        index=False,
        float_format="%.8g",
    )

    logger.info("Lag points          : %d", len(lag_frames))
    logger.info(
        "Actual maximum lag  : %.3f ns",
        float(lag_grid_df["lag_ns"].max()),
    )

    arrays = load_coordinate_arrays(
        pair_database=input_paths["pair_database"],
        event_df=event_df,
        config=config,
        logger=logger,
    )
    sort_coordinate_arrays(arrays, logger)

    (
        event_lag_df,
        event_summary_df,
        processing_metrics,
    ) = process_all_events(
        arrays=arrays,
        event_df=event_df,
        lag_frames=lag_frames,
        metadata=metadata,
        config=config,
        logger=logger,
    )

    event_summary_df.to_csv(
        outputs.event_summary,
        index=False,
        float_format="%.8g",
    )

    if config.save_event_lag_statistics:
        event_lag_df.to_csv(
            outputs.event_lag_statistics,
            index=False,
            compression="gzip",
            float_format="%.8g",
        )

    rng = np.random.default_rng(config.bootstrap_seed)

    all_curves: List[pd.DataFrame] = []
    lifetime_curves: List[pd.DataFrame] = []
    formation_curves: List[pd.DataFrame] = []
    coordination_curves: List[pd.DataFrame] = []

    for scope in ("all_observed", "completed_only"):
        all_curves.append(
            aggregate_event_lag_data(
                event_lag_df=event_lag_df,
                group_type="all_pairs",
                group_column=None,
                analysis_scope=scope,
                config=config,
                rng=rng,
                metadata=metadata,
                logger=logger,
            )
        )
        lifetime_curves.append(
            aggregate_event_lag_data(
                event_lag_df=event_lag_df,
                group_type="lifetime_class",
                group_column="lifetime_class",
                analysis_scope=scope,
                config=config,
                rng=rng,
                metadata=metadata,
                logger=logger,
            )
        )
        formation_curves.append(
            aggregate_event_lag_data(
                event_lag_df=event_lag_df,
                group_type="formation_class",
                group_column="formation_class",
                analysis_scope=scope,
                config=config,
                rng=rng,
                metadata=metadata,
                logger=logger,
            )
        )
        coordination_curves.append(
            aggregate_event_lag_data(
                event_lag_df=event_lag_df,
                group_type="coordination_class",
                group_column="coordination_class",
                analysis_scope=scope,
                config=config,
                rng=rng,
                metadata=metadata,
                logger=logger,
            )
        )

    all_curve_df = pd.concat(
        all_curves,
        ignore_index=True,
    )
    lifetime_curve_df = pd.concat(
        lifetime_curves,
        ignore_index=True,
    )
    formation_curve_df = pd.concat(
        formation_curves,
        ignore_index=True,
    )
    coordination_curve_df = pd.concat(
        coordination_curves,
        ignore_index=True,
    )

    all_curve_df.to_csv(
        outputs.all_pairs,
        index=False,
        float_format="%.8g",
    )
    lifetime_curve_df.to_csv(
        outputs.by_lifetime,
        index=False,
        float_format="%.8g",
    )
    formation_curve_df.to_csv(
        outputs.by_formation,
        index=False,
        float_format="%.8g",
    )
    coordination_curve_df.to_csv(
        outputs.by_coordination,
        index=False,
        float_format="%.8g",
    )

    validation = validate_results(
        event_df=event_df,
        arrays=arrays,
        event_lag_df=event_lag_df,
        event_summary_df=event_summary_df,
        all_curve_df=all_curve_df,
        processing_metrics=processing_metrics,
        metadata=metadata,
        final_5e1_validation=final_5e1_validation,
        config=config,
    )
    write_json(
        outputs.validation_report,
        validation.to_dict(),
    )

    key_lag_summary = summarize_key_lags(
        all_curve_df=all_curve_df,
        targets_ns=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
    )

    persistence_counts = (
        event_df.groupby("lifetime_class", observed=True)
        .agg(
            n_events=("pair_instance_id", "size"),
            n_completed_events=("completed_event", "sum"),
            total_pair_frames=("n_observed_frames", "sum"),
            median_occupancy_time_ps=(
                "occupancy_time_ps",
                "median",
            ),
            maximum_occupancy_time_ps=(
                "occupancy_time_ps",
                "max",
            ),
        )
        .reset_index()
        .to_dict(orient="records")
    )

    formation_counts = (
        event_df.groupby("formation_class", observed=True)
        .agg(
            n_events=("pair_instance_id", "size"),
            total_pair_frames=("n_observed_frames", "sum"),
        )
        .reset_index()
        .to_dict(orient="records")
    )

    coordination_counts = (
        event_df.groupby("coordination_class", observed=True)
        .agg(
            n_events=("pair_instance_id", "size"),
            total_pair_frames=("n_observed_frames", "sum"),
        )
        .reset_index()
        .to_dict(orient="records")
    )

    configure_matplotlib()
    system_label = (
        f"{metadata.metal_species} {metadata.composition}"
    )

    plot_all_msd_components(
        all_curve_df,
        outputs,
        config,
        system_label,
    )
    plot_all_correlation_metrics(
        all_curve_df,
        outputs,
        config,
        system_label,
    )
    plot_group_metric(
        grouped_df=lifetime_curve_df,
        metric="collective_alignment_index",
        ylabel="Collective alignment index",
        title=f"{system_label}: co-motion by pair persistence",
        output_path=outputs.figure_lifetime_alignment,
        config=config,
    )
    plot_group_metric(
        grouped_df=lifetime_curve_df,
        metric="relative_motion_ratio",
        ylabel="Relative-motion ratio",
        title=f"{system_label}: internal motion by pair persistence",
        output_path=outputs.figure_lifetime_relative,
        config=config,
    )
    plot_group_metric(
        grouped_df=formation_curve_df,
        metric="collective_alignment_index",
        ylabel="Collective alignment index",
        title=f"{system_label}: first formation vs reformation",
        output_path=outputs.figure_formation_alignment,
        config=config,
    )
    plot_group_metric(
        grouped_df=coordination_curve_df,
        metric="collective_alignment_index",
        ylabel="Collective alignment index",
        title=f"{system_label}: co-motion by coordination mode",
        output_path=outputs.figure_coordination_alignment,
        config=config,
    )
    plot_sample_support(
        all_curve_df,
        outputs,
        config,
        system_label,
    )

    validation.finalize()

    summary = {
        "module": "05E2_pair_comotion_analysis",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "system": asdict(metadata),
        "configuration": asdict(config),
        "lag_grid": {
            "n_lags": int(len(lag_frames)),
            "minimum_lag_ns": float(
                lag_grid_df["lag_ns"].min()
            ),
            "maximum_lag_ns": float(
                lag_grid_df["lag_ns"].max()
            ),
        },
        "event_processing": processing_metrics,
        "event_class_counts": {
            "lifetime": persistence_counts,
            "formation": formation_counts,
            "coordination": coordination_counts,
        },
        "key_lag_summary_all_observed": key_lag_summary,
        "validation_status": validation.status,
        "warnings": validation.warnings,
        "runtime_seconds": time.perf_counter() - overall_start,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(outputs.summary, summary)

    logger.info("=" * 79)
    logger.info("MODULE 5E2 COMPLETED")
    logger.info("=" * 79)
    logger.info(
        "Events processed         : %d",
        processing_metrics["n_events_processed"],
    )
    logger.info(
        "Event-lag rows           : %d",
        processing_metrics["n_event_lag_rows"],
    )
    logger.info(
        "Maximum metal step       : %.6f Å",
        processing_metrics["maximum_metal_frame_step_A"],
    )
    logger.info(
        "Maximum TFSI step        : %.6f Å",
        processing_metrics["maximum_tfsi_frame_step_A"],
    )
    logger.info(
        "Discontinuous events     : %d",
        processing_metrics["n_discontinuous_events"],
    )
    logger.info(
        "Validation status        : %s",
        validation.status,
    )
    logger.info(
        "Warnings                 : %d",
        len(validation.warnings),
    )
    logger.info("All-pair curves          : %s", outputs.all_pairs)
    logger.info(
        "Lifetime curves          : %s",
        outputs.by_lifetime,
    )
    logger.info(
        "Formation curves         : %s",
        outputs.by_formation,
    )
    logger.info(
        "Coordination curves      : %s",
        outputs.by_coordination,
    )
    logger.info(
        "Event summary            : %s",
        outputs.event_summary,
    )
    if config.save_event_lag_statistics:
        logger.info(
            "Event-lag statistics    : %s",
            outputs.event_lag_statistics,
        )
    logger.info(
        "Validation report        : %s",
        outputs.validation_report,
    )
    logger.info("Summary                  : %s", outputs.summary)
    logger.info(
        "Figures directory        : %s",
        outputs.figures,
    )
    logger.info(
        "Total runtime            : %s",
        format_runtime(time.perf_counter() - overall_start),
    )


if __name__ == "__main__":
    main()