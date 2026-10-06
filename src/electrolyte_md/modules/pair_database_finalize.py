#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05E1_finalize_pair_database.py
==============================

Module 5E1 — Metal–TFSI Pair Database Builder
Part 3: final event analysis, cross-validation, population convergence,
censoring-aware pair lifetimes, distributions, and publication-ready figures.

This script does NOT read the LAMMPS trajectory. It uses only the outputs from
05E1_build_pair_database.py Parts 1 + 2:

Required inputs
---------------
05E1_pair_database/
├── pair_database.csv.gz
├── frame_pair_summary.csv
├── validation_report.json
└── metadata.json

Outputs
-------
05E1_pair_database/
├── pair_event_summary.csv
├── pair_identity_summary.csv
├── pair_lifetime_summary.csv
├── pair_population_block_summary.csv
├── pair_population_equilibration_sensitivity.csv
├── pair_population_autocorrelation.csv
├── metal_oxygen_distance_histogram.csv
├── oxygen_multiplicity_summary.csv
├── distance_outliers.csv
├── pair_survival_probability.csv
├── final_validation_report.json
├── part3_summary.json
└── figures/
    ├── pair_population_vs_time.png
    ├── pair_population_block_average.png
    ├── pair_count_distribution.png
    ├── pair_population_autocorrelation.png
    ├── pair_lifetime_distribution.png
    ├── pair_survival_probability.png
    ├── oxygen_multiplicity_distribution.png
    └── metal_oxygen_distance_distribution.png

Lifetime conventions
--------------------
n_observed_frames
    Number of saved frames in which the pair event is present.

observed_span_ps
    (last_frame - first_frame) × frame_interval_ps. A one-frame event has a
    span of zero.

occupancy_time_ps
    n_observed_frames × effective_frame_interval_ps. This is used for event
    lifetime distributions and Kaplan–Meier survival analysis because every
    observed event has at least one sampling interval of exposure.

Censoring
---------
left_censored
    The pair is already present in the first selected frame; its true start is
    unknown.

right_censored
    The pair remains present in the final selected frame; its true end is
    unknown.

Kaplan–Meier analysis excludes left-censored events because standard
right-censored Kaplan–Meier estimation cannot correctly incorporate unknown
start times. Right-censored events that are not left-censored are retained.

Author: Ajay Dwivedi
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# DEFAULTS
# =============================================================================

DEFAULT_INPUT_DIR = "05E1_pair_database"
DEFAULT_CHUNK_SIZE = 200_000
DEFAULT_POPULATION_BLOCK_NS = 10.0
DEFAULT_ROLLING_WINDOW_NS = 1.0
DEFAULT_ACF_MAX_LAG_NS = 20.0
DEFAULT_DISTANCE_OUTLIER_THRESHOLD_A = 1.80
DEFAULT_DISTANCE_HISTOGRAM_BINS = 200
DEFAULT_FIGURE_DPI = 300

REQUIRED_INPUT_FILES = {
    "pair_database": "pair_database.csv.gz",
    "frame_summary": "frame_pair_summary.csv",
    "validation_report": "validation_report.json",
    "metadata": "metadata.json",
}

REQUIRED_PAIR_COLUMNS = [
    "pair_id",
    "pair_instance_id",
    "frame",
    "time_ps",
    "metal_id",
    "metal_resid",
    "tfsi_resid",
    "n_coordinating_oxygens",
    "min_metal_oxygen_distance_A",
    "mean_metal_oxygen_distance_A",
    "coordinating_oxygen_ids",
    "continuing_pair",
    "new_pair_event",
    "pair_age_frames",
]

REQUIRED_FRAME_COLUMNS = [
    "frame",
    "time_ps",
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
    "mean_min_pair_distance_A",
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass(frozen=True)
class Part3Config:
    input_dir: Path
    chunk_size: int
    population_block_ns: float
    rolling_window_ns: float
    acf_max_lag_ns: float
    distance_outlier_threshold_A: float
    distance_histogram_bins: int
    figure_dpi: int
    overwrite: bool = False

    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if self.population_block_ns <= 0:
            raise ValueError("population_block_ns must be positive.")
        if self.rolling_window_ns <= 0:
            raise ValueError("rolling_window_ns must be positive.")
        if self.acf_max_lag_ns <= 0:
            raise ValueError("acf_max_lag_ns must be positive.")
        if self.distance_outlier_threshold_A <= 0:
            raise ValueError("distance_outlier_threshold_A must be positive.")
        if self.distance_histogram_bins < 20:
            raise ValueError("distance_histogram_bins must be at least 20.")
        if self.figure_dpi < 72:
            raise ValueError("figure_dpi must be at least 72.")


@dataclass
class FinalValidationReport:
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


@dataclass
class EventAccumulator:
    pair_instance_id: str
    pair_id: str
    metal_id: int
    metal_resid: int
    tfsi_resid: int

    first_frame: int
    last_frame: int
    first_time_ps: float
    last_time_ps: float

    n_rows: int = 0
    sum_min_distance_A: float = 0.0
    min_distance_A: float = math.inf
    max_min_distance_A: float = -math.inf

    sum_mean_distance_A: float = 0.0
    sum_n_coordinating_oxygens: int = 0
    max_n_coordinating_oxygens: int = 0
    n_multi_oxygen_frames: int = 0

    first_pair_age_frames: int = 0
    last_pair_age_frames: int = 0
    n_new_pair_flags: int = 0
    n_continuing_pair_flags: int = 0

    def update_row(
        self,
        frame: int,
        time_ps: float,
        min_distance_A: float,
        mean_distance_A: float,
        n_coordinating_oxygens: int,
        pair_age_frames: int,
        new_pair_event: int,
        continuing_pair: int,
    ) -> None:
        if self.n_rows == 0:
            self.first_frame = int(frame)
            self.first_time_ps = float(time_ps)
            self.first_pair_age_frames = int(pair_age_frames)

        self.last_frame = int(frame)
        self.last_time_ps = float(time_ps)
        self.last_pair_age_frames = int(pair_age_frames)
        self.n_rows += 1

        self.sum_min_distance_A += float(min_distance_A)
        self.min_distance_A = min(self.min_distance_A, float(min_distance_A))
        self.max_min_distance_A = max(
            self.max_min_distance_A,
            float(min_distance_A),
        )

        self.sum_mean_distance_A += float(mean_distance_A)
        self.sum_n_coordinating_oxygens += int(n_coordinating_oxygens)
        self.max_n_coordinating_oxygens = max(
            self.max_n_coordinating_oxygens,
            int(n_coordinating_oxygens),
        )
        if int(n_coordinating_oxygens) > 1:
            self.n_multi_oxygen_frames += 1

        self.n_new_pair_flags += int(new_pair_event)
        self.n_continuing_pair_flags += int(continuing_pair)


class OutputPaths:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.figures = self.root / "figures"

        self.event_summary = self.root / "pair_event_summary.csv"
        self.identity_summary = self.root / "pair_identity_summary.csv"
        self.lifetime_summary = self.root / "pair_lifetime_summary.csv"
        self.population_blocks = self.root / "pair_population_block_summary.csv"
        self.equilibration_sensitivity = (
            self.root / "pair_population_equilibration_sensitivity.csv"
        )
        self.population_acf = self.root / "pair_population_autocorrelation.csv"
        self.distance_histogram = (
            self.root / "metal_oxygen_distance_histogram.csv"
        )
        self.oxygen_multiplicity = (
            self.root / "oxygen_multiplicity_summary.csv"
        )
        self.distance_outliers = self.root / "distance_outliers.csv"
        self.survival_probability = (
            self.root / "pair_survival_probability.csv"
        )
        self.final_validation = self.root / "final_validation_report.json"
        self.part3_summary = self.root / "part3_summary.json"
        self.log_file = self.root / "05E1_part3_finalize.log"

        self.figure_population = (
            self.figures / "pair_population_vs_time.png"
        )
        self.figure_population_blocks = (
            self.figures / "pair_population_block_average.png"
        )
        self.figure_pair_count_distribution = (
            self.figures / "pair_count_distribution.png"
        )
        self.figure_population_acf = (
            self.figures / "pair_population_autocorrelation.png"
        )
        self.figure_lifetime_distribution = (
            self.figures / "pair_lifetime_distribution.png"
        )
        self.figure_survival = (
            self.figures / "pair_survival_probability.png"
        )
        self.figure_oxygen_multiplicity = (
            self.figures / "oxygen_multiplicity_distribution.png"
        )
        self.figure_distance_distribution = (
            self.figures / "metal_oxygen_distance_distribution.png"
        )

    def output_files(self) -> List[Path]:
        return [
            self.event_summary,
            self.identity_summary,
            self.lifetime_summary,
            self.population_blocks,
            self.equilibration_sensitivity,
            self.population_acf,
            self.distance_histogram,
            self.oxygen_multiplicity,
            self.distance_outliers,
            self.survival_probability,
            self.final_validation,
            self.part3_summary,
            self.log_file,
            self.figure_population,
            self.figure_population_blocks,
            self.figure_pair_count_distribution,
            self.figure_population_acf,
            self.figure_lifetime_distribution,
            self.figure_survival,
            self.figure_oxygen_multiplicity,
            self.figure_distance_distribution,
        ]

    def prepare(self, overwrite: bool) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.figures.mkdir(parents=True, exist_ok=True)

        existing = [path for path in self.output_files() if path.exists()]
        if existing and not overwrite:
            listing = "\n".join(f"  - {path}" for path in existing)
            raise FileExistsError(
                "Part 3 output files already exist. Use --overwrite to replace:\n"
                f"{listing}"
            )

        if overwrite:
            for path in existing:
                path.unlink()


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def configure_logging(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("05E1_part3")
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


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(make_json_safe(payload), handle, indent=2)
        handle.write("\n")
    temporary.replace(path)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require_columns(
    dataframe_columns: Iterable[str],
    required: Sequence[str],
    label: str,
) -> None:
    columns = set(dataframe_columns)
    missing = [column for column in required if column not in columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def extract_event_number(pair_instance_id: str) -> Optional[int]:
    match = re.search(r"_E(\d+)$", str(pair_instance_id))
    return int(match.group(1)) if match else None


def safe_weighted_mean(
    values: pd.Series,
    weights: pd.Series,
) -> float:
    values_array = np.asarray(values, dtype=float)
    weights_array = np.asarray(weights, dtype=float)
    valid = (
        np.isfinite(values_array)
        & np.isfinite(weights_array)
        & (weights_array > 0)
    )
    if not np.any(valid):
        return np.nan
    return float(
        np.sum(values_array[valid] * weights_array[valid])
        / np.sum(weights_array[valid])
    )


def percentile_or_nan(values: np.ndarray, percentile: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.percentile(values, percentile))


def linear_trend(
    time_ns: np.ndarray,
    values: np.ndarray,
) -> Dict[str, float]:
    x = np.asarray(time_ns, dtype=float)
    y = np.asarray(values, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if x.size < 3 or np.allclose(x, x[0]):
        return {
            "slope_per_ns": np.nan,
            "intercept": np.nan,
            "r_squared": np.nan,
            "slope_standard_error": np.nan,
            "net_change_over_window": np.nan,
        }

    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residuals = y - fitted

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    sxx = float(np.sum((x - np.mean(x)) ** 2))
    if x.size > 2 and sxx > 0:
        residual_variance = ss_res / (x.size - 2)
        slope_se = math.sqrt(residual_variance / sxx)
    else:
        slope_se = np.nan

    return {
        "slope_per_ns": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "slope_standard_error": float(slope_se),
        "net_change_over_window": float(slope * (x[-1] - x[0])),
    }


def autocorrelation_fft(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([], dtype=float)

    x = x - np.mean(x)
    variance = float(np.var(x))
    if variance == 0:
        result = np.zeros(x.size, dtype=float)
        result[0] = 1.0
        return result

    n = x.size
    fft_length = 1 << (2 * n - 1).bit_length()
    transformed = np.fft.rfft(x, n=fft_length)
    raw = np.fft.irfft(
        transformed * np.conjugate(transformed),
        n=fft_length,
    )[:n]

    normalization = np.arange(n, 0, -1, dtype=float)
    autocovariance = raw / normalization
    return autocovariance / autocovariance[0]


def integrated_autocorrelation_time(acf: np.ndarray) -> float:
    """
    Initial-positive-sequence estimate in frames.

    Returns tau_int = 0.5 + sum_{lag >= 1} acf(lag) until the first
    nonpositive value.
    """

    if acf.size == 0 or not np.isfinite(acf[0]):
        return np.nan

    tau = 0.5
    for value in acf[1:]:
        if not np.isfinite(value) or value <= 0:
            break
        tau += float(value)
    return float(tau)


def kaplan_meier(
    durations_ps: np.ndarray,
    event_observed: np.ndarray,
) -> pd.DataFrame:
    """
    Standard Kaplan–Meier estimator for right-censored observations.

    Left-censored events must be excluded before calling this function.
    """

    durations = np.asarray(durations_ps, dtype=float)
    observed = np.asarray(event_observed, dtype=bool)

    valid = np.isfinite(durations) & (durations > 0)
    durations = durations[valid]
    observed = observed[valid]

    columns = [
        "time_ps",
        "time_ns",
        "n_at_risk",
        "n_events",
        "n_censored",
        "survival_probability",
    ]

    if durations.size == 0:
        return pd.DataFrame(columns=columns)

    unique_times = np.sort(np.unique(durations))
    n_at_risk = int(durations.size)
    survival = 1.0
    records: List[Dict[str, Any]] = []

    for current_time in unique_times:
        at_time = durations == current_time
        n_events = int(np.sum(at_time & observed))
        n_censored = int(np.sum(at_time & ~observed))

        risk_before = n_at_risk
        if risk_before > 0 and n_events > 0:
            survival *= 1.0 - n_events / risk_before

        records.append(
            {
                "time_ps": float(current_time),
                "time_ns": float(current_time / 1000.0),
                "n_at_risk": risk_before,
                "n_events": n_events,
                "n_censored": n_censored,
                "survival_probability": float(survival),
            }
        )

        n_at_risk -= n_events + n_censored

    return pd.DataFrame(records, columns=columns)


def median_survival_time(km_df: pd.DataFrame) -> float:
    if km_df.empty:
        return np.nan
    below = km_df[km_df["survival_probability"] <= 0.5]
    if below.empty:
        return np.nan
    return float(below.iloc[0]["time_ps"])


# =============================================================================
# INPUT SETUP
# =============================================================================

def resolve_input_paths(input_dir: Path) -> Dict[str, Path]:
    paths = {
        key: input_dir / filename
        for key, filename in REQUIRED_INPUT_FILES.items()
    }

    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        listing = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Required Part 1/2 outputs are missing:\n{listing}")

    return paths


def read_pair_header(pair_database: Path) -> List[str]:
    return pd.read_csv(
        pair_database,
        compression="gzip",
        nrows=0,
    ).columns.tolist()


def resolve_metadata(
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    identity = metadata.get("identity", {})
    configuration = metadata.get("configuration", {})
    selected = metadata.get("selected_frame_summary", {})
    topology = metadata.get("topology_summary", {})
    pair_definition = metadata.get("pair_definition", {})

    frame_interval_ps = float(configuration.get("frame_interval_ps", 10.0))
    stride = int(configuration.get("stride", selected.get("stride", 1)))
    effective_frame_interval_ps = frame_interval_ps * stride

    first_frame = int(selected.get("first_frame", 0))
    last_frame = int(selected.get("last_frame", first_frame))
    n_selected_frames = int(
        selected.get("n_frames", last_frame - first_frame + 1)
    )

    cutoff_A = float(
        pair_definition.get(
            "cutoff_A",
            configuration.get("metal_oxygen_cutoff_A", np.nan),
        )
    )

    return {
        "system": identity.get("system", "unknown"),
        "metal_species": identity.get("metal_species", "unknown"),
        "composition": identity.get("composition", "unknown"),
        "temperature_K": float(identity.get("temperature_K", np.nan)),
        "frame_interval_ps": frame_interval_ps,
        "stride": stride,
        "effective_frame_interval_ps": effective_frame_interval_ps,
        "first_frame": first_frame,
        "last_frame": last_frame,
        "n_selected_frames": n_selected_frames,
        "cutoff_A": cutoff_A,
        "n_metals": int(topology.get("n_metal_atoms", 0)),
        "n_tfsi_residues": int(topology.get("n_tfsi_residues", 0)),
    }


# =============================================================================
# STREAMING PAIR DATABASE ANALYSIS
# =============================================================================

def initialize_distance_histogram(
    cutoff_A: float,
    n_bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    upper = cutoff_A if np.isfinite(cutoff_A) and cutoff_A > 0 else 5.0
    edges = np.linspace(0.0, upper, n_bins + 1)
    counts = np.zeros(n_bins, dtype=np.int64)
    return edges, counts


def stream_pair_database(
    pair_database: Path,
    config: Part3Config,
    resolved: Mapping[str, Any],
    outputs: OutputPaths,
    logger: logging.Logger,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, Any],
]:
    """
    Stream the large pair database and create:
      - event-level accumulators,
      - distance histogram,
      - oxygen multiplicity counts,
      - distance-outlier table.
    """

    usecols = REQUIRED_PAIR_COLUMNS
    dtypes = {
        "pair_id": "string",
        "pair_instance_id": "string",
        "frame": "int32",
        "time_ps": "float64",
        "metal_id": "int32",
        "metal_resid": "int32",
        "tfsi_resid": "int32",
        "n_coordinating_oxygens": "int16",
        "min_metal_oxygen_distance_A": "float64",
        "mean_metal_oxygen_distance_A": "float64",
        "coordinating_oxygen_ids": "string",
        "continuing_pair": "int8",
        "new_pair_event": "int8",
        "pair_age_frames": "int32",
    }

    event_accumulators: Dict[str, EventAccumulator] = {}
    multiplicity_counter: Counter = Counter()

    distance_edges, distance_counts = initialize_distance_histogram(
        cutoff_A=float(resolved["cutoff_A"]),
        n_bins=config.distance_histogram_bins,
    )

    outlier_columns = [
        "pair_instance_id",
        "pair_id",
        "frame",
        "time_ps",
        "metal_id",
        "metal_resid",
        "tfsi_resid",
        "n_coordinating_oxygens",
        "min_metal_oxygen_distance_A",
        "mean_metal_oxygen_distance_A",
        "coordinating_oxygen_ids",
        "pair_age_frames",
    ]

    outlier_handle = outputs.distance_outliers.open(
        "w",
        encoding="utf-8",
        newline="",
    )
    outlier_writer = csv.DictWriter(
        outlier_handle,
        fieldnames=outlier_columns,
    )
    outlier_writer.writeheader()

    n_rows = 0
    n_chunks = 0
    n_outliers = 0
    global_min_distance = math.inf
    global_max_distance = -math.inf
    rows_above_cutoff = 0
    rows_without_positive_oxygen_count = 0
    duplicate_event_frame_rows = 0

    seen_event_frame_pairs: set = set()
    process_start = time.perf_counter()

    logger.info("Streaming pair database in chunks of %d rows", config.chunk_size)

    try:
        reader = pd.read_csv(
            pair_database,
            compression="gzip",
            usecols=usecols,
            dtype=dtypes,
            chunksize=config.chunk_size,
            low_memory=False,
        )

        for chunk in reader:
            n_chunks += 1
            n_rows += len(chunk)

            distances = chunk["min_metal_oxygen_distance_A"].to_numpy(
                dtype=float
            )
            finite_distances = distances[np.isfinite(distances)]

            if finite_distances.size:
                global_min_distance = min(
                    global_min_distance,
                    float(np.min(finite_distances)),
                )
                global_max_distance = max(
                    global_max_distance,
                    float(np.max(finite_distances)),
                )
                histogram, _ = np.histogram(
                    finite_distances,
                    bins=distance_edges,
                )
                distance_counts += histogram.astype(np.int64)

            cutoff_A = float(resolved["cutoff_A"])
            if np.isfinite(cutoff_A):
                rows_above_cutoff += int(
                    np.sum(distances > cutoff_A + 1.0e-6)
                )

            oxygen_counts = chunk[
                "n_coordinating_oxygens"
            ].to_numpy(dtype=int)
            rows_without_positive_oxygen_count += int(
                np.sum(oxygen_counts < 1)
            )
            multiplicity_counter.update(oxygen_counts.tolist())

            outliers = chunk[
                chunk["min_metal_oxygen_distance_A"]
                < config.distance_outlier_threshold_A
            ]
            if not outliers.empty:
                for row in outliers[outlier_columns].itertuples(
                    index=False,
                    name=None,
                ):
                    outlier_writer.writerow(
                        dict(zip(outlier_columns, row))
                    )
                n_outliers += len(outliers)

            for row in chunk.itertuples(index=False):
                instance_id = str(row.pair_instance_id)
                event_frame_key = (instance_id, int(row.frame))

                if event_frame_key in seen_event_frame_pairs:
                    duplicate_event_frame_rows += 1
                else:
                    seen_event_frame_pairs.add(event_frame_key)

                accumulator = event_accumulators.get(instance_id)
                if accumulator is None:
                    accumulator = EventAccumulator(
                        pair_instance_id=instance_id,
                        pair_id=str(row.pair_id),
                        metal_id=int(row.metal_id),
                        metal_resid=int(row.metal_resid),
                        tfsi_resid=int(row.tfsi_resid),
                        first_frame=int(row.frame),
                        last_frame=int(row.frame),
                        first_time_ps=float(row.time_ps),
                        last_time_ps=float(row.time_ps),
                    )
                    event_accumulators[instance_id] = accumulator

                accumulator.update_row(
                    frame=int(row.frame),
                    time_ps=float(row.time_ps),
                    min_distance_A=float(
                        row.min_metal_oxygen_distance_A
                    ),
                    mean_distance_A=float(
                        row.mean_metal_oxygen_distance_A
                    ),
                    n_coordinating_oxygens=int(
                        row.n_coordinating_oxygens
                    ),
                    pair_age_frames=int(row.pair_age_frames),
                    new_pair_event=int(row.new_pair_event),
                    continuing_pair=int(row.continuing_pair),
                )

            elapsed = time.perf_counter() - process_start
            logger.info(
                "Chunk %d | rows=%d | events=%d | outliers=%d | elapsed=%.1f s",
                n_chunks,
                n_rows,
                len(event_accumulators),
                n_outliers,
                elapsed,
            )
    finally:
        outlier_handle.close()

    effective_dt_ps = float(resolved["effective_frame_interval_ps"])
    first_selected_frame = int(resolved["first_frame"])
    last_selected_frame = int(resolved["last_frame"])
    stride = int(resolved["stride"])

    event_records: List[Dict[str, Any]] = []

    for accumulator in event_accumulators.values():
        expected_n_rows = (
            (accumulator.last_frame - accumulator.first_frame) // stride
        ) + 1

        left_censored = (
            accumulator.first_frame == first_selected_frame
        )
        right_censored = (
            accumulator.last_frame == last_selected_frame
        )
        fully_censored = left_censored and right_censored
        completed_event = not left_censored and not right_censored

        event_number = extract_event_number(
            accumulator.pair_instance_id
        )
        reformation_event = (
            event_number is not None and event_number > 1
        )

        observed_span_ps = (
            accumulator.last_frame - accumulator.first_frame
        ) * float(resolved["frame_interval_ps"])
        occupancy_time_ps = accumulator.n_rows * effective_dt_ps

        event_records.append(
            {
                "system": resolved["system"],
                "metal_species": resolved["metal_species"],
                "composition": resolved["composition"],
                "temperature_K": resolved["temperature_K"],
                "pair_instance_id": accumulator.pair_instance_id,
                "pair_id": accumulator.pair_id,
                "event_number": event_number,
                "reformation_event": int(reformation_event),
                "metal_id": accumulator.metal_id,
                "metal_resid": accumulator.metal_resid,
                "tfsi_resid": accumulator.tfsi_resid,
                "first_frame": accumulator.first_frame,
                "last_frame": accumulator.last_frame,
                "first_time_ps": accumulator.first_time_ps,
                "last_time_ps": accumulator.last_time_ps,
                "n_observed_frames": accumulator.n_rows,
                "expected_consecutive_frames": expected_n_rows,
                "continuity_valid": int(
                    accumulator.n_rows == expected_n_rows
                ),
                "observed_span_ps": observed_span_ps,
                "observed_span_ns": observed_span_ps / 1000.0,
                "occupancy_time_ps": occupancy_time_ps,
                "occupancy_time_ns": occupancy_time_ps / 1000.0,
                "left_censored": int(left_censored),
                "right_censored": int(right_censored),
                "fully_censored": int(fully_censored),
                "completed_event": int(completed_event),
                "mean_min_distance_A": (
                    accumulator.sum_min_distance_A
                    / accumulator.n_rows
                ),
                "minimum_distance_A": accumulator.min_distance_A,
                "maximum_min_distance_A": (
                    accumulator.max_min_distance_A
                ),                "mean_contact_distance_A": (
                    accumulator.sum_mean_distance_A
                    / accumulator.n_rows
                ),
                "mean_coordinating_oxygens": (
                    accumulator.sum_n_coordinating_oxygens
                    / accumulator.n_rows
                ),
                "maximum_coordinating_oxygens": (
                    accumulator.max_n_coordinating_oxygens
                ),
                "n_multi_oxygen_frames": (
                    accumulator.n_multi_oxygen_frames
                ),
                "fraction_multi_oxygen_frames": (
                    accumulator.n_multi_oxygen_frames
                    / accumulator.n_rows
                ),
                "first_pair_age_frames": (
                    accumulator.first_pair_age_frames
                ),
                "last_pair_age_frames": (
                    accumulator.last_pair_age_frames
                ),
                "n_new_pair_flags": accumulator.n_new_pair_flags,
                "n_continuing_pair_flags": (
                    accumulator.n_continuing_pair_flags
                ),
            }
        )

    event_df = pd.DataFrame(event_records)
    event_df.sort_values(
        ["first_frame", "metal_id", "tfsi_resid", "event_number"],
        inplace=True,
        ignore_index=True,
    )

    distance_midpoints = 0.5 * (
        distance_edges[:-1] + distance_edges[1:]
    )
    total_distance_counts = int(np.sum(distance_counts))
    bin_widths = np.diff(distance_edges)
    density = (
        distance_counts / (total_distance_counts * bin_widths)
        if total_distance_counts > 0
        else np.zeros_like(distance_midpoints)
    )

    distance_histogram_df = pd.DataFrame(
        {
            "bin_left_A": distance_edges[:-1],
            "bin_right_A": distance_edges[1:],
            "bin_center_A": distance_midpoints,
            "count": distance_counts,
            "probability_density_per_A": density,
        }
    )

    multiplicity_rows = []
    total_multiplicity = sum(multiplicity_counter.values())
    for multiplicity in sorted(multiplicity_counter):
        count = int(multiplicity_counter[multiplicity])
        multiplicity_rows.append(
            {
                "n_coordinating_oxygens": int(multiplicity),
                "count": count,
                "fraction": (
                    count / total_multiplicity
                    if total_multiplicity
                    else np.nan
                ),
            }
        )
    multiplicity_df = pd.DataFrame(multiplicity_rows)

    streaming_metrics = {
        "n_pair_rows_read": n_rows,
        "n_chunks": n_chunks,
        "n_pair_events_reconstructed": len(event_df),
        "n_distance_outliers": n_outliers,
        "distance_outlier_threshold_A": (
            config.distance_outlier_threshold_A
        ),
        "minimum_distance_A": (
            None
            if not np.isfinite(global_min_distance)
            else global_min_distance
        ),
        "maximum_distance_A": (
            None
            if not np.isfinite(global_max_distance)
            else global_max_distance
        ),
        "rows_above_cutoff": rows_above_cutoff,
        "rows_without_positive_oxygen_count": (
            rows_without_positive_oxygen_count
        ),
        "duplicate_pair_instance_frame_rows": (
            duplicate_event_frame_rows
        ),
        "streaming_seconds": time.perf_counter() - process_start,
    }

    return (
        event_df,
        distance_histogram_df,
        multiplicity_df,
        streaming_metrics,
    )


# =============================================================================
# EVENT AND IDENTITY SUMMARIES
# =============================================================================

def build_pair_identity_summary(
    event_df: pd.DataFrame,
    n_selected_frames: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for pair_id, group in event_df.groupby("pair_id", sort=True):
        weights = group["n_observed_frames"]

        completed_lifetimes = group.loc[
            group["completed_event"] == 1,
            "occupancy_time_ps",
        ].to_numpy(dtype=float)

        rows.append(
            {
                "system": group["system"].iloc[0],
                "metal_species": group["metal_species"].iloc[0],
                "composition": group["composition"].iloc[0],
                "temperature_K": group["temperature_K"].iloc[0],
                "pair_id": pair_id,
                "metal_id": int(group["metal_id"].iloc[0]),
                "metal_resid": int(group["metal_resid"].iloc[0]),
                "tfsi_resid": int(group["tfsi_resid"].iloc[0]),
                "n_pair_events": int(len(group)),
                "n_reformation_events": int(
                    group["reformation_event"].sum()
                ),
                "n_completed_events": int(
                    group["completed_event"].sum()
                ),
                "n_left_censored_events": int(
                    group["left_censored"].sum()
                ),
                "n_right_censored_events": int(
                    group["right_censored"].sum()
                ),
                "n_fully_censored_events": int(
                    group["fully_censored"].sum()
                ),
                "first_observed_frame": int(
                    group["first_frame"].min()
                ),
                "last_observed_frame": int(
                    group["last_frame"].max()
                ),
                "total_observed_frames": int(
                    group["n_observed_frames"].sum()
                ),
                "trajectory_occupancy_fraction": (
                    float(group["n_observed_frames"].sum())
                    / n_selected_frames
                ),
                "maximum_event_occupancy_time_ps": float(
                    group["occupancy_time_ps"].max()
                ),
                "mean_completed_event_lifetime_ps": (
                    float(np.mean(completed_lifetimes))
                    if completed_lifetimes.size
                    else np.nan
                ),
                "median_completed_event_lifetime_ps": (
                    float(np.median(completed_lifetimes))
                    if completed_lifetimes.size
                    else np.nan
                ),
                "weighted_mean_min_distance_A": safe_weighted_mean(
                    group["mean_min_distance_A"],
                    weights,
                ),
                "minimum_distance_A": float(
                    group["minimum_distance_A"].min()
                ),
                "weighted_mean_coordinating_oxygens": safe_weighted_mean(
                    group["mean_coordinating_oxygens"],
                    weights,
                ),
                "weighted_fraction_multi_oxygen_frames": (
                    float(group["n_multi_oxygen_frames"].sum())
                    / float(group["n_observed_frames"].sum())
                ),
            }
        )

    result = pd.DataFrame(rows)
    result.sort_values(
        [
            "trajectory_occupancy_fraction",
            "metal_id",
            "tfsi_resid",
        ],
        ascending=[False, True, True],
        inplace=True,
        ignore_index=True,
    )
    return result


def summarize_lifetime_subset(
    label: str,
    subset: pd.DataFrame,
) -> Dict[str, Any]:
    durations = subset["occupancy_time_ps"].to_numpy(dtype=float)
    return {
        "event_class": label,
        "n_events": int(len(subset)),
        "mean_occupancy_time_ps": (
            float(np.mean(durations)) if durations.size else np.nan
        ),
        "median_occupancy_time_ps": (
            float(np.median(durations)) if durations.size else np.nan
        ),
        "std_occupancy_time_ps": (
            float(np.std(durations, ddof=1))
            if durations.size > 1
            else np.nan
        ),
        "minimum_occupancy_time_ps": (
            float(np.min(durations)) if durations.size else np.nan
        ),
        "p25_occupancy_time_ps": percentile_or_nan(durations, 25),
        "p75_occupancy_time_ps": percentile_or_nan(durations, 75),
        "p90_occupancy_time_ps": percentile_or_nan(durations, 90),
        "p95_occupancy_time_ps": percentile_or_nan(durations, 95),
        "maximum_occupancy_time_ps": (
            float(np.max(durations)) if durations.size else np.nan
        ),
        "mean_occupancy_time_ns": (
            float(np.mean(durations) / 1000.0)
            if durations.size
            else np.nan
        ),
        "median_occupancy_time_ns": (
            float(np.median(durations) / 1000.0)
            if durations.size
            else np.nan
        ),
        "maximum_occupancy_time_ns": (
            float(np.max(durations) / 1000.0)
            if durations.size
            else np.nan
        ),
    }


def build_lifetime_summary(event_df: pd.DataFrame) -> pd.DataFrame:
    subsets = [
        ("all_events", event_df),
        (
            "completed_events",
            event_df[event_df["completed_event"] == 1],
        ),
        (
            "left_censored_events",
            event_df[event_df["left_censored"] == 1],
        ),
        (
            "right_censored_events",
            event_df[event_df["right_censored"] == 1],
        ),
        (
            "fully_censored_events",
            event_df[event_df["fully_censored"] == 1],
        ),
        (
            "first_formation_events",
            event_df[event_df["reformation_event"] == 0],
        ),
        (
            "reformation_events",
            event_df[event_df["reformation_event"] == 1],
        ),
    ]

    return pd.DataFrame(
        [
            summarize_lifetime_subset(label, subset)
            for label, subset in subsets
        ]
    )


# =============================================================================
# POPULATION ANALYSIS
# =============================================================================

def analyze_population(
    frame_summary_path: Path,
    resolved: Mapping[str, Any],
    config: Part3Config,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, Any],
]:
    frame_df = pd.read_csv(frame_summary_path, low_memory=False)
    require_columns(
        frame_df.columns,
        REQUIRED_FRAME_COLUMNS,
        "frame_pair_summary.csv",
    )

    frame_df.sort_values("frame", inplace=True, ignore_index=True)
    frame_df["time_ns"] = frame_df["time_ps"] / 1000.0

    effective_dt_ps = float(resolved["effective_frame_interval_ps"])
    rolling_frames = max(
        1,
        int(
            round(
                config.rolling_window_ns
                * 1000.0
                / effective_dt_ps
            )
        ),
    )
    frame_df["rolling_mean_pairs"] = (
        frame_df["n_unique_metal_tfsi_pairs"]
        .rolling(
            window=rolling_frames,
            center=True,
            min_periods=max(1, rolling_frames // 4),
        )
        .mean()
    )

    frames_per_block = max(
        1,
        int(
            round(
                config.population_block_ns
                * 1000.0
                / effective_dt_ps
            )
        ),
    )
    frame_df["population_block_index"] = (
        np.arange(len(frame_df), dtype=int) // frames_per_block
    )

    block_rows: List[Dict[str, Any]] = []

    for block_index, group in frame_df.groupby(
        "population_block_index",
        sort=True,
    ):
        pair_values = group[
            "n_unique_metal_tfsi_pairs"
        ].to_numpy(dtype=float)
        n_group = len(group)

        block_rows.append(
            {
                "system": resolved["system"],
                "metal_species": resolved["metal_species"],
                "composition": resolved["composition"],
                "temperature_K": resolved["temperature_K"],
                "block_index": int(block_index),
                "start_frame": int(group["frame"].iloc[0]),
                "end_frame": int(group["frame"].iloc[-1]),
                "start_time_ns": float(group["time_ns"].iloc[0]),
                "end_time_ns": float(group["time_ns"].iloc[-1]),
                "midpoint_time_ns": float(
                    0.5
                    * (
                        group["time_ns"].iloc[0]
                        + group["time_ns"].iloc[-1]
                    )
                ),
                "n_frames": int(n_group),
                "expected_frames_per_full_block": int(
                    frames_per_block
                ),
                "complete_block": int(
                    n_group >= 0.90 * frames_per_block
                ),
                "mean_pair_count": float(np.mean(pair_values)),
                "std_pair_count": (
                    float(np.std(pair_values, ddof=1))
                    if n_group > 1
                    else np.nan
                ),
                "sem_pair_count": (
                    float(
                        np.std(pair_values, ddof=1)
                        / math.sqrt(n_group)
                    )
                    if n_group > 1
                    else np.nan
                ),
                "minimum_pair_count": int(np.min(pair_values)),
                "maximum_pair_count": int(np.max(pair_values)),
                "mean_coordinated_metal_fraction": float(
                    group["fraction_coordinated_metals"].mean()
                ),
                "mean_coordinated_tfsi_fraction": float(
                    group["fraction_coordinated_tfsi"].mean()
                ),
                "mean_contacts_per_frame": float(
                    group["n_metal_oxygen_contacts"].mean()
                ),
                "mean_oxygen_multiplicity_per_pair": float(
                    group[
                        "mean_coordinating_oxygens_per_pair"
                    ].mean()
                ),
                "mean_min_pair_distance_A": float(
                    group["mean_min_pair_distance_A"].mean()
                ),
            }
        )

    block_df = pd.DataFrame(block_rows)

    n_frames = len(frame_df)
    midpoint = n_frames // 2
    first_half = frame_df.iloc[:midpoint]
    second_half = frame_df.iloc[midpoint:]

    pair_counts = frame_df[
        "n_unique_metal_tfsi_pairs"
    ].to_numpy(dtype=float)
    time_ns = frame_df["time_ns"].to_numpy(dtype=float)
    full_trend = linear_trend(time_ns, pair_counts)

    sensitivity_starts = [
        0.0,
        10.0,
        20.0,
        50.0,
        100.0,
    ]
    maximum_time_ns = float(frame_df["time_ns"].max())
    sensitivity_rows: List[Dict[str, Any]] = []

    for requested_start_ns in sensitivity_starts:
        if requested_start_ns >= maximum_time_ns:
            continue

        subset = frame_df[
            frame_df["time_ns"] >= requested_start_ns
        ]
        values = subset[
            "n_unique_metal_tfsi_pairs"
        ].to_numpy(dtype=float)
        trend = linear_trend(
            subset["time_ns"].to_numpy(dtype=float),
            values,
        )

        sensitivity_rows.append(
            {
                "requested_start_time_ns": requested_start_ns,
                "actual_start_time_ns": float(
                    subset["time_ns"].iloc[0]
                ),
                "end_time_ns": float(
                    subset["time_ns"].iloc[-1]
                ),
                "n_frames": int(len(subset)),
                "mean_pair_count": float(np.mean(values)),
                "std_pair_count": float(
                    np.std(values, ddof=1)
                ),
                "sem_naive_pair_count": float(
                    np.std(values, ddof=1)
                    / math.sqrt(len(values))
                ),
                **trend,
            }
        )

    sensitivity_df = pd.DataFrame(sensitivity_rows)

    acf_full = autocorrelation_fft(pair_counts)
    max_lag_frames = min(
        len(acf_full) - 1,
        int(
            round(
                config.acf_max_lag_ns
                * 1000.0
                / effective_dt_ps
            )
        ),
    )
    lag_frames = np.arange(max_lag_frames + 1, dtype=int)
    acf = acf_full[: max_lag_frames + 1]

    acf_df = pd.DataFrame(
        {
            "lag_frames": lag_frames,
            "lag_ps": lag_frames * effective_dt_ps,
            "lag_ns": lag_frames * effective_dt_ps / 1000.0,
            "autocorrelation": acf,
        }
    )

    tau_int_frames = integrated_autocorrelation_time(acf_full)
    tau_int_ps = tau_int_frames * effective_dt_ps
    effective_sample_size = (
        len(pair_counts) / (2.0 * tau_int_frames)
        if np.isfinite(tau_int_frames) and tau_int_frames > 0
        else np.nan
    )

    corrected_sem = (
        float(np.std(pair_counts, ddof=1))
        / math.sqrt(effective_sample_size)
        if np.isfinite(effective_sample_size)
        and effective_sample_size > 1
        else np.nan
    )

    complete_blocks = block_df[block_df["complete_block"] == 1]
    block_trend = linear_trend(
        complete_blocks["midpoint_time_ns"].to_numpy(dtype=float),
        complete_blocks["mean_pair_count"].to_numpy(dtype=float),
    )

    population_summary = {
        "n_frames": n_frames,
        "first_time_ns": float(frame_df["time_ns"].iloc[0]),
        "last_time_ns": float(frame_df["time_ns"].iloc[-1]),
        "mean_pair_count": float(np.mean(pair_counts)),
        "std_pair_count": float(np.std(pair_counts, ddof=1)),
        "minimum_pair_count": int(np.min(pair_counts)),
        "maximum_pair_count": int(np.max(pair_counts)),
        "first_half_mean_pair_count": float(
            first_half["n_unique_metal_tfsi_pairs"].mean()
        ),
        "second_half_mean_pair_count": float(
            second_half["n_unique_metal_tfsi_pairs"].mean()
        ),
        "second_minus_first_half_pair_count": float(
            second_half["n_unique_metal_tfsi_pairs"].mean()
            - first_half["n_unique_metal_tfsi_pairs"].mean()
        ),
        "relative_half_change_percent": float(
            100.0
            * (
                second_half[
                    "n_unique_metal_tfsi_pairs"
                ].mean()
                - first_half[
                    "n_unique_metal_tfsi_pairs"
                ].mean()
            )
            / first_half[
                "n_unique_metal_tfsi_pairs"
            ].mean()
        ),
        "full_frame_trend": full_trend,
        "complete_block_trend": block_trend,
        "rolling_window_frames": rolling_frames,
        "rolling_window_ns": config.rolling_window_ns,
        "frames_per_population_block": frames_per_block,
        "population_block_ns": config.population_block_ns,
        "integrated_autocorrelation_time_frames": tau_int_frames,
        "integrated_autocorrelation_time_ps": tau_int_ps,
        "integrated_autocorrelation_time_ns": tau_int_ps / 1000.0,
        "effective_sample_size": effective_sample_size,
        "autocorrelation_corrected_sem_pair_count": corrected_sem,
        "mean_coordinated_metal_fraction": float(
            frame_df["fraction_coordinated_metals"].mean()
        ),
        "mean_coordinated_tfsi_fraction": float(
            frame_df["fraction_coordinated_tfsi"].mean()
        ),
        "mean_pairs_per_metal_all": float(
            frame_df["mean_pairs_per_metal_all"].mean()
        ),
        "mean_pairs_per_coordinated_metal": float(
            frame_df[
                "mean_pairs_per_coordinated_metal"
            ].mean()
        ),
        "mean_contacts_per_frame": float(
            frame_df["n_metal_oxygen_contacts"].mean()
        ),
        "mean_oxygen_multiplicity_per_pair": float(
            frame_df[
                "mean_coordinating_oxygens_per_pair"
            ].mean()
        ),
    }

    return (
        frame_df,
        block_df,
        sensitivity_df,
        acf_df,
        population_summary,
    )


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
            "legend.fontsize": 10,
            "figure.dpi": 120,
            "savefig.bbox": "tight",
        }
    )


def save_and_close(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_pair_population(
    frame_df: pd.DataFrame,
    block_df: pd.DataFrame,
    outputs: OutputPaths,
    config: Part3Config,
    system_label: str,
) -> None:
    fig, axis = plt.subplots(figsize=(8.0, 5.0))
    axis.plot(
        frame_df["time_ns"],
        frame_df["n_unique_metal_tfsi_pairs"],
        linewidth=0.45,
        alpha=0.45,
        label="Frame-level count",
    )
    axis.plot(
        frame_df["time_ns"],
        frame_df["rolling_mean_pairs"],
        linewidth=1.6,
        label=f"{config.rolling_window_ns:g} ns rolling mean",
    )

    complete = block_df[block_df["complete_block"] == 1]
    axis.plot(
        complete["midpoint_time_ns"],
        complete["mean_pair_count"],
        marker="o",
        linewidth=1.1,
        markersize=3.5,
        label=f"{config.population_block_ns:g} ns block mean",
    )

    axis.set_xlabel("Time (ns)")
    axis.set_ylabel("Number of metal–TFSI pairs")
    axis.set_title(f"{system_label}: pair population")
    axis.legend(frameon=False)
    axis.grid(alpha=0.25)
    save_and_close(fig, outputs.figure_population, config.figure_dpi)


def plot_population_blocks(
    block_df: pd.DataFrame,
    outputs: OutputPaths,
    config: Part3Config,
    system_label: str,
) -> None:
    complete = block_df[block_df["complete_block"] == 1]

    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.errorbar(
        complete["midpoint_time_ns"],
        complete["mean_pair_count"],
        yerr=complete["sem_pair_count"],
        marker="o",
        capsize=3,
        linewidth=1.2,
    )
    axis.set_xlabel("Block midpoint (ns)")
    axis.set_ylabel("Mean number of metal–TFSI pairs")
    axis.set_title(
        f"{system_label}: {config.population_block_ns:g} ns block averages"
    )
    axis.grid(alpha=0.25)
    save_and_close(
        fig,
        outputs.figure_population_blocks,
        config.figure_dpi,
    )


def plot_pair_count_distribution(
    frame_df: pd.DataFrame,
    outputs: OutputPaths,
    config: Part3Config,
    system_label: str,
) -> None:
    values = frame_df[
        "n_unique_metal_tfsi_pairs"
    ].to_numpy(dtype=float)
    lower = int(np.min(values))
    upper = int(np.max(values))
    bins = np.arange(lower - 0.5, upper + 1.5, 1.0)

    fig, axis = plt.subplots(figsize=(6.5, 4.7))
    axis.hist(values, bins=bins, density=True)
    axis.set_xlabel("Number of metal–TFSI pairs per frame")
    axis.set_ylabel("Probability")
    axis.set_title(f"{system_label}: pair-count distribution")
    axis.grid(axis="y", alpha=0.25)
    save_and_close(
        fig,
        outputs.figure_pair_count_distribution,
        config.figure_dpi,
    )


def plot_population_acf(
    acf_df: pd.DataFrame,
    outputs: OutputPaths,
    config: Part3Config,
    system_label: str,
) -> None:
    fig, axis = plt.subplots(figsize=(6.8, 4.6))
    axis.plot(
        acf_df["lag_ns"],
        acf_df["autocorrelation"],
        linewidth=1.3,
    )
    axis.axhline(0.0, linewidth=0.8)
    axis.set_xlabel("Lag time (ns)")
    axis.set_ylabel("Autocorrelation")
    axis.set_title(f"{system_label}: pair-population autocorrelation")
    axis.grid(alpha=0.25)
    save_and_close(
        fig,
        outputs.figure_population_acf,
        config.figure_dpi,
    )


def plot_lifetime_distribution(
    event_df: pd.DataFrame,
    outputs: OutputPaths,
    config: Part3Config,
    system_label: str,
) -> None:
    completed = event_df[event_df["completed_event"] == 1]
    lifetimes_ns = completed[
        "occupancy_time_ns"
    ].to_numpy(dtype=float)
    lifetimes_ns = lifetimes_ns[
        np.isfinite(lifetimes_ns) & (lifetimes_ns > 0)
    ]

    fig, axis = plt.subplots(figsize=(6.8, 4.8))

    if lifetimes_ns.size:
        minimum = float(np.min(lifetimes_ns))
        maximum = float(np.max(lifetimes_ns))

        if maximum > minimum:
            bins = np.logspace(
                np.log10(minimum),
                np.log10(maximum),
                45,
            )
        else:
            bins = np.array(
                [minimum * 0.8, minimum * 1.2]
            )

        axis.hist(lifetimes_ns, bins=bins)
        axis.set_xscale("log")
    else:
        axis.text(
            0.5,
            0.5,
            "No completed events",
            transform=axis.transAxes,
            ha="center",
            va="center",
        )

    axis.set_xlabel("Completed-event occupancy time (ns)")
    axis.set_ylabel("Number of events")
    axis.set_title(f"{system_label}: completed pair-event lifetimes")
    axis.grid(axis="y", alpha=0.25)
    save_and_close(
        fig,
        outputs.figure_lifetime_distribution,
        config.figure_dpi,
    )


def plot_survival_probability(
    km_df: pd.DataFrame,
    outputs: OutputPaths,
    config: Part3Config,
    system_label: str,
) -> None:
    fig, axis = plt.subplots(figsize=(6.8, 4.8))

    if not km_df.empty:
        axis.step(
            km_df["time_ns"],
            km_df["survival_probability"],
            where="post",
            linewidth=1.5,
        )
        positive_times = km_df["time_ns"] > 0
        if positive_times.any():
            axis.set_xscale("log")
    else:
        axis.text(
            0.5,
            0.5,
            "No eligible events",
            transform=axis.transAxes,
            ha="center",
            va="center",
        )

    axis.set_xlabel("Pair-event occupancy time (ns)")
    axis.set_ylabel("Survival probability")
    axis.set_ylim(0.0, 1.03)
    axis.set_title(
        f"{system_label}: Kaplan–Meier pair survival"
    )
    axis.grid(alpha=0.25)
    save_and_close(
        fig,
        outputs.figure_survival,
        config.figure_dpi,
    )


def plot_oxygen_multiplicity(
    multiplicity_df: pd.DataFrame,
    outputs: OutputPaths,
    config: Part3Config,
    system_label: str,
) -> None:
    fig, axis = plt.subplots(figsize=(6.4, 4.6))
    if not multiplicity_df.empty:
        axis.bar(
            multiplicity_df["n_coordinating_oxygens"],
            multiplicity_df["fraction"],
        )
        axis.set_xticks(
            multiplicity_df["n_coordinating_oxygens"]
        )
    axis.set_xlabel("Coordinating TFSI oxygens per pair")
    axis.set_ylabel("Fraction of pair-frame observations")
    axis.set_title(f"{system_label}: oxygen multiplicity")
    axis.grid(axis="y", alpha=0.25)
    save_and_close(
        fig,
        outputs.figure_oxygen_multiplicity,
        config.figure_dpi,
    )


def plot_distance_distribution(
    distance_histogram_df: pd.DataFrame,
    outputs: OutputPaths,
    config: Part3Config,
    system_label: str,
    outlier_threshold_A: float,
) -> None:
    fig, axis = plt.subplots(figsize=(6.8, 4.8))
    axis.plot(
        distance_histogram_df["bin_center_A"],
        distance_histogram_df["probability_density_per_A"],
        linewidth=1.4,
    )
    axis.axvline(
        outlier_threshold_A,
        linestyle="--",
        linewidth=1.0,
        label=f"Diagnostic threshold = {outlier_threshold_A:.2f} Å",
    )
    axis.set_xlabel("Minimum metal–oxygen distance (Å)")
    axis.set_ylabel("Probability density (Å$^{-1}$)")
    axis.set_title(f"{system_label}: metal–TFSI oxygen distance")
    axis.legend(frameon=False)
    axis.grid(alpha=0.25)
    save_and_close(
        fig,
        outputs.figure_distance_distribution,
        config.figure_dpi,
    )


# =============================================================================
# VALIDATION AND SUMMARY
# =============================================================================

def compare_with_part2_validation(
    part2_validation: Mapping[str, Any],
    event_df: pd.DataFrame,
    frame_df: pd.DataFrame,
    streaming_metrics: Mapping[str, Any],
    resolved: Mapping[str, Any],
    report: FinalValidationReport,
) -> None:
    part2_status = part2_validation.get("status", "UNKNOWN")
    part2_metrics = part2_validation.get("metrics", {})

    report.add_check(
        "part2_validation_was_successful",
        part2_status == "PASSED",
    )
    report.add_check(
        "pair_database_rows_match_part2",
        int(streaming_metrics["n_pair_rows_read"])
        == int(part2_metrics.get("n_pair_rows_written", -1)),
    )
    report.add_check(
        "event_count_matches_part2",
        len(event_df)
        == int(part2_metrics.get("total_pair_events_started", -1)),
    )
    report.add_check(
        "frame_summary_count_matches_selected_frames",
        len(frame_df) == int(resolved["n_selected_frames"]),
    )
    report.add_check(
        "all_reconstructed_events_are_consecutive",
        bool((event_df["continuity_valid"] == 1).all()),
    )
    report.add_check(
        "all_events_have_one_new_pair_flag",
        bool((event_df["n_new_pair_flags"] == 1).all()),
    )
    report.add_check(
        "no_duplicate_pair_instance_frame_rows",
        int(
            streaming_metrics[
                "duplicate_pair_instance_frame_rows"
            ]
        )
        == 0,
    )
    report.add_check(
        "all_pair_rows_have_positive_oxygen_count",
        int(
            streaming_metrics[
                "rows_without_positive_oxygen_count"
            ]
        )
        == 0,
    )
    report.add_check(
        "all_pair_rows_are_within_cutoff",
        int(streaming_metrics["rows_above_cutoff"]) == 0,
    )
    report.add_check(
        "maximum_event_age_matches_part2",
        int(event_df["n_observed_frames"].max())
        == int(
            part2_metrics.get(
                "maximum_consecutive_pair_age_frames",
                -1,
            )
        ),
    )
    report.add_check(
        "event_rows_sum_to_pair_database_rows",
        int(event_df["n_observed_frames"].sum())
        == int(streaming_metrics["n_pair_rows_read"]),
    )

    report.add_metric("part2_status", part2_status)
    report.add_metric(
        "n_pair_rows_read",
        streaming_metrics["n_pair_rows_read"],
    )
    report.add_metric("n_pair_events", len(event_df))
    report.add_metric(
        "n_unique_pair_identities",
        event_df["pair_id"].nunique(),
    )
    report.add_metric(        "n_completed_events",
        int(event_df["completed_event"].sum()),
    )
    report.add_metric(
        "n_left_censored_events",
        int(event_df["left_censored"].sum()),
    )
    report.add_metric(
        "n_right_censored_events",
        int(event_df["right_censored"].sum()),
    )
    report.add_metric(
        "n_fully_censored_events",
        int(event_df["fully_censored"].sum()),
    )
    report.add_metric(
        "n_distance_outliers",
        streaming_metrics["n_distance_outliers"],
    )
    report.add_metric(
        "distance_outlier_threshold_A",
        streaming_metrics["distance_outlier_threshold_A"],
    )
    report.add_metric(
        "minimum_distance_A",
        streaming_metrics["minimum_distance_A"],
    )
    report.add_metric(
        "maximum_distance_A",
        streaming_metrics["maximum_distance_A"],
    )

    if int(streaming_metrics["n_distance_outliers"]) > 0:
        report.warn(
            f"{streaming_metrics['n_distance_outliers']} pair-frame rows "
            f"have minimum metal–oxygen distances below "
            f"{streaming_metrics['distance_outlier_threshold_A']:.2f} Å. "
            "Inspect distance_outliers.csv before physical interpretation."
        )

    if int(event_df["fully_censored"].sum()) > 0:
        report.warn(
            f"{int(event_df['fully_censored'].sum())} pair event(s) span the "
            "entire selected trajectory and are both left- and right-censored."
        )


def add_population_validation(
    population_summary: Mapping[str, Any],
    km_df: pd.DataFrame,
    report: FinalValidationReport,
) -> None:
    relative_change = float(
        population_summary["relative_half_change_percent"]
    )
    full_net_change = float(
        population_summary["full_frame_trend"][
            "net_change_over_window"
        ]
    )
    mean_pair_count = float(population_summary["mean_pair_count"])

    report.add_metric(
        "mean_pair_count",
        population_summary["mean_pair_count"],
    )
    report.add_metric(
        "first_half_mean_pair_count",
        population_summary["first_half_mean_pair_count"],
    )
    report.add_metric(
        "second_half_mean_pair_count",
        population_summary["second_half_mean_pair_count"],
    )
    report.add_metric(
        "relative_half_change_percent",
        relative_change,
    )
    report.add_metric(
        "full_frame_slope_pairs_per_ns",
        population_summary["full_frame_trend"][
            "slope_per_ns"
        ],
    )
    report.add_metric(
        "estimated_full_window_pair_change",
        full_net_change,
    )
    report.add_metric(
        "population_integrated_autocorrelation_time_ns",
        population_summary[
            "integrated_autocorrelation_time_ns"
        ],
    )
    report.add_metric(
        "population_effective_sample_size",
        population_summary["effective_sample_size"],
    )
    report.add_metric(
        "mean_coordinated_metal_fraction",
        population_summary["mean_coordinated_metal_fraction"],
    )
    report.add_metric(
        "mean_coordinated_tfsi_fraction",
        population_summary["mean_coordinated_tfsi_fraction"],
    )

    report.add_check(
        "population_mean_is_finite",
        np.isfinite(mean_pair_count),
    )

    if not km_df.empty:
        survival = km_df[
            "survival_probability"
        ].to_numpy(dtype=float)
        monotonic = bool(np.all(np.diff(survival) <= 1.0e-12))
        bounded = bool(
            np.all((survival >= -1.0e-12) & (survival <= 1.0 + 1.0e-12))
        )
    else:
        monotonic = True
        bounded = True

    report.add_check(
        "kaplan_meier_survival_is_monotonic",
        monotonic,
    )
    report.add_check(
        "kaplan_meier_survival_is_bounded",
        bounded,
    )

    if abs(relative_change) >= 5.0:
        report.warn(
            f"The second-half mean pair population differs from the first "
            f"half by {relative_change:.2f}%. This indicates a meaningful "
            "long-timescale population drift or incomplete equilibration."
        )

    if np.isfinite(full_net_change) and mean_pair_count > 0:
        relative_net_change = 100.0 * full_net_change / mean_pair_count
        report.add_metric(
            "trend_implied_relative_change_percent",
            relative_net_change,
        )
        if abs(relative_net_change) >= 5.0:
            report.warn(
                f"The fitted population trend implies a "
                f"{relative_net_change:.2f}% change relative to the mean "
                "over the selected trajectory."
            )


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize Module 5E1 using the pair database and frame summary "
            "without rereading the trajectory."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(DEFAULT_INPUT_DIR),
        help="Directory containing Parts 1 + 2 outputs.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Rows per pandas chunk when reading pair_database.csv.gz.",
    )
    parser.add_argument(
        "--population-block-ns",
        type=float,
        default=DEFAULT_POPULATION_BLOCK_NS,
        help="Block length for pair-population convergence analysis.",
    )
    parser.add_argument(
        "--rolling-window-ns",
        type=float,
        default=DEFAULT_ROLLING_WINDOW_NS,
        help="Rolling-mean window for the population time-series figure.",
    )
    parser.add_argument(
        "--acf-max-lag-ns",
        type=float,
        default=DEFAULT_ACF_MAX_LAG_NS,
        help="Maximum lag shown in the pair-population autocorrelation.",
    )
    parser.add_argument(
        "--distance-outlier-threshold-A",
        type=float,
        default=DEFAULT_DISTANCE_OUTLIER_THRESHOLD_A,
        help="Export pair rows below this minimum metal–oxygen distance.",
    )
    parser.add_argument(
        "--distance-histogram-bins",
        type=int,
        default=DEFAULT_DISTANCE_HISTOGRAM_BINS,
        help="Number of distance-histogram bins.",
    )
    parser.add_argument(
        "--figure-dpi",
        type=int,
        default=DEFAULT_FIGURE_DPI,
        help="PNG output resolution.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing Part 3 outputs.",
    )

    return parser


def config_from_args(args: argparse.Namespace) -> Part3Config:
    config = Part3Config(
        input_dir=args.input_dir.resolve(),
        chunk_size=int(args.chunk_size),
        population_block_ns=float(args.population_block_ns),
        rolling_window_ns=float(args.rolling_window_ns),
        acf_max_lag_ns=float(args.acf_max_lag_ns),
        distance_outlier_threshold_A=float(
            args.distance_outlier_threshold_A
        ),
        distance_histogram_bins=int(
            args.distance_histogram_bins
        ),
        figure_dpi=int(args.figure_dpi),
        overwrite=bool(args.overwrite),
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
    outputs = OutputPaths(config.input_dir)
    outputs.prepare(overwrite=config.overwrite)
    logger = configure_logging(outputs.log_file)

    overall_start = time.perf_counter()

    logger.info("=" * 79)
    logger.info("MODULE 5E1 PART 3: FINAL ANALYSIS AND VALIDATION")
    logger.info("=" * 79)
    logger.info("Input directory     : %s", config.input_dir)
    logger.info("Pair database       : %s", input_paths["pair_database"])
    logger.info("Frame summary       : %s", input_paths["frame_summary"])
    logger.info("Chunk size          : %d", config.chunk_size)
    logger.info(
        "Population block   : %.3f ns",
        config.population_block_ns,
    )
    logger.info(
        "Outlier threshold  : %.3f Å",
        config.distance_outlier_threshold_A,
    )

    metadata = read_json(input_paths["metadata"])
    part2_validation = read_json(
        input_paths["validation_report"]
    )
    resolved = resolve_metadata(metadata)

    logger.info("System              : %s", resolved["system"])
    logger.info("Metal species       : %s", resolved["metal_species"])
    logger.info("Composition         : %s", resolved["composition"])
    logger.info(
        "Temperature         : %.2f metal",
        resolved["temperature_K"],
    )
    logger.info(
        "Effective frame dt  : %.3f ps",
        resolved["effective_frame_interval_ps"],
    )
    logger.info(
        "Selected frames     : %d",
        resolved["n_selected_frames"],
    )
    logger.info("Pair cutoff         : %.3f Å", resolved["cutoff_A"])

    pair_header = read_pair_header(
        input_paths["pair_database"]
    )
    require_columns(
        pair_header,
        REQUIRED_PAIR_COLUMNS,
        "pair_database.csv.gz",
    )

    final_validation = FinalValidationReport()
    final_validation.add_check(
        "required_input_files_present",
        True,
    )
    final_validation.add_check(
        "required_pair_database_columns_present",
        True,
    )
    final_validation.add_check(
        "metadata_frame_interval_valid",
        resolved["effective_frame_interval_ps"] > 0,
    )

    (
        event_df,
        distance_histogram_df,
        multiplicity_df,
        streaming_metrics,
    ) = stream_pair_database(
        pair_database=input_paths["pair_database"],
        config=config,
        resolved=resolved,
        outputs=outputs,
        logger=logger,
    )

    identity_df = build_pair_identity_summary(
        event_df=event_df,
        n_selected_frames=int(resolved["n_selected_frames"]),
    )
    lifetime_df = build_lifetime_summary(event_df)

    (
        frame_df,
        block_df,
        sensitivity_df,
        acf_df,
        population_summary,
    ) = analyze_population(
        frame_summary_path=input_paths["frame_summary"],
        resolved=resolved,
        config=config,
    )

    km_eligible = event_df[event_df["left_censored"] == 0].copy()
    km_durations = km_eligible[
        "occupancy_time_ps"
    ].to_numpy(dtype=float)
    km_event_observed = (
        km_eligible["right_censored"].to_numpy(dtype=int) == 0
    )
    km_df = kaplan_meier(
        durations_ps=km_durations,
        event_observed=km_event_observed,
    )

    median_km_ps = median_survival_time(km_df)

    event_df.to_csv(
        outputs.event_summary,
        index=False,
        float_format="%.8g",
    )
    identity_df.to_csv(
        outputs.identity_summary,
        index=False,
        float_format="%.8g",
    )
    lifetime_df.to_csv(
        outputs.lifetime_summary,
        index=False,
        float_format="%.8g",
    )
    block_df.to_csv(
        outputs.population_blocks,
        index=False,
        float_format="%.8g",
    )
    sensitivity_df.to_csv(
        outputs.equilibration_sensitivity,
        index=False,
        float_format="%.8g",
    )
    acf_df.to_csv(
        outputs.population_acf,
        index=False,
        float_format="%.8g",
    )
    distance_histogram_df.to_csv(
        outputs.distance_histogram,
        index=False,
        float_format="%.8g",
    )
    multiplicity_df.to_csv(
        outputs.oxygen_multiplicity,
        index=False,
        float_format="%.8g",
    )
    km_df.to_csv(
        outputs.survival_probability,
        index=False,
        float_format="%.8g",
    )

    compare_with_part2_validation(
        part2_validation=part2_validation,
        event_df=event_df,
        frame_df=frame_df,
        streaming_metrics=streaming_metrics,
        resolved=resolved,
        report=final_validation,
    )
    add_population_validation(
        population_summary=population_summary,
        km_df=km_df,
        report=final_validation,
    )

    completed_lifetimes = event_df.loc[
        event_df["completed_event"] == 1,
        "occupancy_time_ps",
    ].to_numpy(dtype=float)

    part3_summary = {
        "module": "05E1_finalize_pair_database",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "system": resolved,
        "streaming_metrics": streaming_metrics,
        "event_summary": {
            "n_pair_events": int(len(event_df)),
            "n_unique_pair_identities": int(
                event_df["pair_id"].nunique()
            ),
            "n_completed_events": int(
                event_df["completed_event"].sum()
            ),
            "n_left_censored_events": int(
                event_df["left_censored"].sum()
            ),
            "n_right_censored_events": int(
                event_df["right_censored"].sum()
            ),
            "n_fully_censored_events": int(
                event_df["fully_censored"].sum()
            ),
            "n_reformation_events": int(
                event_df["reformation_event"].sum()
            ),
            "mean_completed_event_lifetime_ps": (
                float(np.mean(completed_lifetimes))
                if completed_lifetimes.size
                else None
            ),
            "median_completed_event_lifetime_ps": (
                float(np.median(completed_lifetimes))
                if completed_lifetimes.size
                else None
            ),
            "p90_completed_event_lifetime_ps": (
                percentile_or_nan(completed_lifetimes, 90)
            ),
            "maximum_completed_event_lifetime_ps": (
                float(np.max(completed_lifetimes))
                if completed_lifetimes.size
                else None
            ),
            "kaplan_meier_eligible_events": int(
                len(km_eligible)
            ),
            "kaplan_meier_median_survival_ps": (
                median_km_ps
            ),
            "kaplan_meier_median_survival_ns": (
                median_km_ps / 1000.0
                if np.isfinite(median_km_ps)
                else None
            ),
        },
        "population_summary": population_summary,
        "oxygen_multiplicity_summary": (
            multiplicity_df.to_dict(orient="records")
        ),
        "final_validation_status": None,
    }

    configure_matplotlib()
    system_label = (
        f"{resolved['metal_species']} {resolved['composition']}"
    )

    plot_pair_population(
        frame_df,
        block_df,
        outputs,
        config,
        system_label,
    )
    plot_population_blocks(
        block_df,
        outputs,
        config,
        system_label,
    )
    plot_pair_count_distribution(
        frame_df,
        outputs,
        config,
        system_label,
    )
    plot_population_acf(
        acf_df,
        outputs,
        config,
        system_label,
    )
    plot_lifetime_distribution(
        event_df,
        outputs,
        config,
        system_label,
    )
    plot_survival_probability(
        km_df,
        outputs,
        config,
        system_label,
    )
    plot_oxygen_multiplicity(
        multiplicity_df,
        outputs,
        config,
        system_label,
    )
    plot_distance_distribution(
        distance_histogram_df,
        outputs,
        config,
        system_label,
        config.distance_outlier_threshold_A,
    )

    final_validation.finalize()
    part3_summary["final_validation_status"] = (
        final_validation.status
    )
    part3_summary["completed_utc"] = (
        datetime.now(timezone.utc).isoformat()
    )
    part3_summary["runtime_seconds"] = (
        time.perf_counter() - overall_start
    )

    write_json(
        outputs.final_validation,
        final_validation.to_dict(),
    )
    write_json(outputs.part3_summary, part3_summary)

    logger.info("=" * 79)
    logger.info("MODULE 5E1 PART 3 COMPLETED")
    logger.info("=" * 79)
    logger.info(
        "Pair rows analyzed       : %d",
        streaming_metrics["n_pair_rows_read"],
    )
    logger.info("Pair events reconstructed: %d", len(event_df))
    logger.info(
        "Unique pair identities   : %d",
        event_df["pair_id"].nunique(),
    )
    logger.info(
        "Completed events         : %d",
        int(event_df["completed_event"].sum()),
    )
    logger.info(
        "Left-censored events     : %d",
        int(event_df["left_censored"].sum()),
    )
    logger.info(
        "Right-censored events    : %d",
        int(event_df["right_censored"].sum()),
    )
    logger.info(
        "Fully censored events    : %d",
        int(event_df["fully_censored"].sum()),
    )
    logger.info(
        "Distance outliers        : %d below %.3f Å",
        streaming_metrics["n_distance_outliers"],
        config.distance_outlier_threshold_A,
    )
    logger.info(
        "Mean pair population     : %.6f",
        population_summary["mean_pair_count"],
    )
    logger.info(
        "First-half mean          : %.6f",
        population_summary["first_half_mean_pair_count"],
    )
    logger.info(
        "Second-half mean         : %.6f",
        population_summary["second_half_mean_pair_count"],
    )
    logger.info(
        "Relative half change     : %.3f%%",
        population_summary["relative_half_change_percent"],
    )
    logger.info(
        "Population slope         : %.8f pairs/ns",
        population_summary["full_frame_trend"][
            "slope_per_ns"
        ],
    )
    logger.info(
        "Population tau_int       : %.6f ns",
        population_summary[
            "integrated_autocorrelation_time_ns"
        ],
    )
    if np.isfinite(median_km_ps):
        logger.info(
            "KM median survival      : %.6f ns",
            median_km_ps / 1000.0,
        )
    else:
        logger.info(
            "KM median survival      : not reached"
        )
    logger.info(
        "Final validation         : %s",
        final_validation.status,
    )
    logger.info(
        "Warnings                 : %d",
        len(final_validation.warnings),
    )
    logger.info(
        "Event summary            : %s",
        outputs.event_summary,
    )
    logger.info(
        "Identity summary         : %s",
        outputs.identity_summary,
    )
    logger.info(
        "Lifetime summary         : %s",
        outputs.lifetime_summary,
    )
    logger.info(
        "Final validation report  : %s",
        outputs.final_validation,
    )
    logger.info(
        "Part 3 summary           : %s",
        outputs.part3_summary,
    )
    logger.info(
        "Figures directory        : %s",
        outputs.figures,
    )
    logger.info(
        "Runtime                  : %.2f s",
        time.perf_counter() - overall_start,
    )


if __name__ == "__main__":
    main()