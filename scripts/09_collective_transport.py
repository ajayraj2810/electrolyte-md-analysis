from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import re
import warnings

import MDAnalysis as mda
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WORK_DIR = Path.cwd()
OUTPUT_DIR = WORK_DIR / "collective_transport_v2"

TOPOLOGY_FILE: Path | None = None
TRAJECTORY_FILE: Path | None = None

FRAME_INTERVAL_PS = 10.0
TEMPERATURE_K = 380.0

SPECIES_SELECTIONS = {
    "metal": "type 17",
    "EMIM": "type 18",
    "TFSI": "type 7",
}

METAL_CHARGE_NUMBER = 1.0
FORMAL_CHARGE_NUMBERS = {
    "metal": METAL_CHARGE_NUMBER,
    "EMIM": +1.0,
    "TFSI": -1.0,
}

MODEL_CURRENT_CHARGE_SCALE = 0.8

METAL_MASS_G_MOL = 40.0
MOLECULAR_MASSES_G_MOL = {
    "metal": METAL_MASS_G_MOL,
    "EMIM": 111.17,
    "TFSI": 280.15,
}

REMOVE_IONIC_BARYCENTER_DRIFT = True

MAX_LAG_NS = 100.0
ORIGIN_STRIDE = 1
BLOCK_ORIGIN_STRIDE = 2
ORIGIN_CHUNK_SIZE = 250

N_EARLY_LINEAR_LAGS = 25
FULL_DENSE_LAG_END_NS = 20.0
FULL_DENSE_LAG_STEP_NS = 0.25
FULL_LATE_LAG_STEP_NS = 1.0

BLOCK_LAG_STEP_NS = 0.5

BLOCK_SCHEMES = (
    {
        "name": "5x40ns",
        "block_length_ns": 40.0,
        "maximum_lag_ns": 20.0,
        "expected_blocks": 5,
        "fit_windows_ns": (
            (2.0, 8.0),
            (5.0, 12.0),
            (8.0, 16.0),
            (10.0, 20.0),
        ),
    },
    {
        "name": "4x50ns",
        "block_length_ns": 50.0,
        "maximum_lag_ns": 25.0,
        "expected_blocks": 4,
        "fit_windows_ns": (
            (2.0, 8.0),
            (5.0, 12.0),
            (8.0, 16.0),
            (10.0, 20.0),
            (12.0, 25.0),
        ),
    },
)

FULL_FIT_WINDOWS_NS = (
    (5.0, 20.0),
    (10.0, 30.0),
    (20.0, 50.0),
    (30.0, 60.0),
    (40.0, 80.0),
    (50.0, 100.0),
)
SLIDING_WINDOW_LENGTHS_NS = (10.0, 15.0, 20.0, 30.0)
SLIDING_WINDOW_START_STEP_NS = 5.0
CUMULATIVE_START_TIMES_NS = (2.0, 5.0, 10.0, 20.0)
CUMULATIVE_END_STEP_NS = 5.0

STABLE_BETA_MIN = 0.90
STABLE_BETA_MAX = 1.10
STABLE_R2_MIN = 0.98
STABLE_LOCAL_WINDOW_CV_MAX = 0.25
STABLE_LOCAL_NEIGHBORS_MIN = 3
STABLE_BLOCK_CV_MAX = 0.75
STABLE_FULL_BLOCK_RELATIVE_DIFFERENCE_MAX = 0.35
STABLE_SELF_NE_MIN = 0.70
STABLE_SELF_NE_MAX = 1.30

MIN_FIT_POINTS = 6

SAVE_UNWRAPPED_COORDINATES = False
COORDINATE_DTYPE = np.float32

DIFFUSION_COEFFICIENTS_NM2_NS = {}

ELEMENTARY_CHARGE_C = 1.602176634e-19
BOLTZMANN_J_K = 1.380649e-23
ANGSTROM_TO_M = 1.0e-10
PS_TO_S = 1.0e-12
A3_TO_M3 = 1.0e-30
NM2_NS_TO_M2_S = 1.0e-9
S_M_TO_MS_CM = 10.0

SPECIES_ORDER = ("metal", "EMIM", "TFSI")
PAIR_ORDER = (
    ("metal", "metal"),
    ("EMIM", "EMIM"),
    ("TFSI", "TFSI"),
    ("metal", "EMIM"),
    ("metal", "TFSI"),
    ("EMIM", "TFSI"),
)

@dataclass(frozen=True)
class SystemIdentity:
    metal: str
    composition: str
    system: str

def detect_system_identity(path: Path) -> SystemIdentity:

    parts = [part.strip() for part in path.resolve().parts]
    allowed_compositions = {"20_5", "15_10", "15_15", "5_20"}
    matches = [part for part in parts if part in allowed_compositions]
    composition = matches[0] if len(matches) == 1 else "composition"
    return SystemIdentity(metal="metal", composition=composition, system=f"metal_{composition}")

def recursive_matches(patterns: tuple[str, ...]) -> list[Path]:
    matches: set[Path] = set()
    for pattern in patterns:
        matches.update(WORK_DIR.rglob(pattern))

    return sorted(
        path
        for path in matches
        if OUTPUT_DIR not in path.parents and path != OUTPUT_DIR
    )

def choose_single_file(
    explicit_file: Path | None,
    patterns: tuple[str, ...],
    description: str,
) -> Path:
    if explicit_file is not None:
        candidate = explicit_file if explicit_file.is_absolute() else WORK_DIR / explicit_file
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        return candidate

    matches = recursive_matches(patterns)

    if not matches:
        raise FileNotFoundError(
            f"No {description} was found below {WORK_DIR} using patterns {patterns}."
        )

    if len(matches) > 1:
        listing = "\n".join(f"  {path}" for path in matches)
        raise RuntimeError(
            f"Multiple {description} files were found:\n{listing}\n"
            "Set the corresponding explicit filename in USER SETTINGS."
        )

    return matches[0]

def discover_files() -> tuple[Path, Path]:
    topology = choose_single_file(
        TOPOLOGY_FILE,
        ("*.data",),
        "LAMMPS topology/data file",
    )
    trajectory = choose_single_file(
        TRAJECTORY_FILE,
        ("*.lammpsdump", "*.lammpstrj", "*.dump"),
        "LAMMPS trajectory",
    )
    return topology, trajectory

def detect_dump_coordinate_mode(trajectory_file: Path) -> str:

    with trajectory_file.open("rt", encoding="utf-8", errors="replace") as handle:
        for _ in range(200):
            line = handle.readline()
            if not line:
                break
            if line.startswith("ITEM: ATOMS"):
                fields = line.split()[2:]
                field_set = set(fields)
                if {"xu", "yu", "zu"}.issubset(field_set) or {
                    "xsu", "ysu", "zsu"
                }.issubset(field_set):
                    return "unwrapped"
                if {"x", "y", "z"}.issubset(field_set) or {
                    "xs", "ys", "zs"
                }.issubset(field_set):
                    return "wrapped"
                raise RuntimeError(
                    "Could not identify supported x/y/z coordinate columns in the "
                    f"LAMMPS dump header: {fields}"
                )

    raise RuntimeError("Could not find an 'ITEM: ATOMS' header in the trajectory.")

def validate_box(dimensions: np.ndarray) -> None:
    if dimensions is None or len(dimensions) < 6:
        raise RuntimeError("Trajectory frame does not contain valid box dimensions.")

    if not np.all(np.isfinite(dimensions[:6])):
        raise RuntimeError("Non-finite simulation-box dimensions were encountered.")

    if not np.allclose(dimensions[3:6], [90.0, 90.0, 90.0], atol=1.0e-3):
        raise NotImplementedError(
            "This implementation currently supports orthorhombic boxes only."
        )

def minimum_image(delta: np.ndarray, box_lengths: np.ndarray) -> np.ndarray:
    return delta - box_lengths * np.round(delta / box_lengths)

def extract_representative_positions(
    universe: mda.Universe,
    groups: dict[str, mda.core.groups.AtomGroup],
    coordinate_mode: str,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, float]]:

    n_frames = len(universe.trajectory)
    positions = {
        species: np.empty((n_frames, len(group), 3), dtype=COORDINATE_DTYPE)
        for species, group in groups.items()
    }
    volumes_a3 = np.empty(n_frames, dtype=np.float64)

    previous_wrapped: dict[str, np.ndarray] = {}
    previous_unwrapped: dict[str, np.ndarray] = {}
    maximum_raw_step_a = {species: 0.0 for species in SPECIES_ORDER}

    print(f"\nReading {n_frames} trajectory frames ({coordinate_mode} coordinates)...")

    for frame_index, ts in enumerate(universe.trajectory):
        dimensions = np.asarray(ts.dimensions, dtype=np.float64)
        validate_box(dimensions)
        box_lengths = dimensions[:3]
        volumes_a3[frame_index] = float(np.prod(box_lengths))

        for species, group in groups.items():
            current = group.positions.astype(np.float64, copy=True)

            if frame_index == 0 or coordinate_mode == "unwrapped":
                unwrapped = current.copy()
            else:
                raw_delta = current - previous_wrapped[species]
                maximum_raw_step_a[species] = max(
                    maximum_raw_step_a[species],
                    float(np.max(np.linalg.norm(raw_delta, axis=1))),
                )
                delta = minimum_image(raw_delta, box_lengths)
                unwrapped = previous_unwrapped[species] + delta

            positions[species][frame_index] = unwrapped.astype(COORDINATE_DTYPE)
            previous_wrapped[species] = current
            previous_unwrapped[species] = unwrapped

        if frame_index and frame_index % 1000 == 0:
            print(f"  processed {frame_index}/{n_frames} frames")

    return positions, volumes_a3, maximum_raw_step_a

def remove_ionic_barycenter_drift(
    positions: dict[str, np.ndarray],
    identity: SystemIdentity,
) -> tuple[dict[str, np.ndarray], np.ndarray]:

    masses = {
        "metal": MOLECULAR_MASSES_G_MOL["metal"],
        "EMIM": MOLECULAR_MASSES_G_MOL["EMIM"],
        "TFSI": MOLECULAR_MASSES_G_MOL["TFSI"],
    }

    total_mass = sum(masses[s] * positions[s].shape[1] for s in SPECIES_ORDER)
    barycenter = np.zeros((positions["metal"].shape[0], 3), dtype=np.float64)

    for species in SPECIES_ORDER:
        barycenter += masses[species] * positions[species].astype(np.float64).sum(axis=1)

    barycenter /= total_mass
    drift = barycenter - barycenter[0]

    corrected = {
        species: (positions[species].astype(np.float64) - drift[:, None, :]).astype(
            COORDINATE_DTYPE
        )
        for species in SPECIES_ORDER
    }

    return corrected, drift

def _ns_to_frames(time_ns: float, frame_interval_ps: float) -> int:
    return max(1, int(round(time_ns * 1000.0 / frame_interval_ps)))

def create_full_lag_frames(
    n_frames: int,
    frame_interval_ps: float,
    maximum_lag_ns: float,
) -> np.ndarray:

    maximum_lag_frames = min(
        n_frames - 1,
        _ns_to_frames(maximum_lag_ns, frame_interval_ps),
    )
    if maximum_lag_frames < 1:
        raise ValueError("Trajectory segment is too short for the requested lag range.")

    early = np.arange(
        1,
        min(N_EARLY_LINEAR_LAGS, maximum_lag_frames) + 1,
        dtype=int,
    )

    dense_end = min(
        maximum_lag_frames,
        _ns_to_frames(FULL_DENSE_LAG_END_NS, frame_interval_ps),
    )
    dense_step = _ns_to_frames(FULL_DENSE_LAG_STEP_NS, frame_interval_ps)
    dense = np.arange(dense_step, dense_end + 1, dense_step, dtype=int)

    late_step = _ns_to_frames(FULL_LATE_LAG_STEP_NS, frame_interval_ps)
    late_start = dense_end + late_step
    late = np.arange(late_start, maximum_lag_frames + 1, late_step, dtype=int)

    return np.unique(
        np.concatenate(
            [early, dense, late, np.asarray([maximum_lag_frames], dtype=int)]
        )
    )

def create_block_lag_frames(
    n_frames: int,
    frame_interval_ps: float,
    maximum_lag_ns: float,
) -> np.ndarray:

    maximum_lag_frames = min(
        n_frames - 1,
        _ns_to_frames(maximum_lag_ns, frame_interval_ps),
    )
    if maximum_lag_frames < 1:
        raise ValueError("Block is too short for the requested lag range.")

    early = np.arange(
        1,
        min(N_EARLY_LINEAR_LAGS, maximum_lag_frames) + 1,
        dtype=int,
    )
    step = _ns_to_frames(BLOCK_LAG_STEP_NS, frame_interval_ps)
    regular = np.arange(step, maximum_lag_frames + 1, step, dtype=int)
    return np.unique(
        np.concatenate([early, regular, np.asarray([maximum_lag_frames], dtype=int)])
    )

def calculate_one_lag(
    positions: dict[str, np.ndarray],
    lag: int,
    origin_stride: int,
    origin_chunk_size: int,
) -> dict[str, float]:

    n_frames = positions["metal"].shape[0]
    starts = np.arange(0, n_frames - lag, origin_stride, dtype=int)

    if len(starts) == 0:
        raise ValueError(f"No time origins are available for lag {lag}.")

    sums = {
        "C_metal_metal_A2": 0.0,
        "C_EMIM_EMIM_A2": 0.0,
        "C_TFSI_TFSI_A2": 0.0,
        "C_metal_EMIM_A2": 0.0,
        "C_metal_TFSI_A2": 0.0,
        "C_EMIM_TFSI_A2": 0.0,
        "Cself_metal_A2": 0.0,
        "Cself_EMIM_A2": 0.0,
        "Cself_TFSI_A2": 0.0,
    }

    n_origins_accumulated = 0

    for chunk_start in range(0, len(starts), origin_chunk_size):
        chunk = starts[chunk_start : chunk_start + origin_chunk_size]
        collective_vectors: dict[str, np.ndarray] = {}

        for species in SPECIES_ORDER:
            displacement = (
                positions[species][chunk + lag].astype(np.float64)
                - positions[species][chunk].astype(np.float64)
            )
            collective = displacement.sum(axis=1)
            collective_vectors[species] = collective

            sums[f"C_{species}_{species}_A2"] += float(
                np.sum(np.einsum("ij,ij->i", collective, collective))
            )
            sums[f"Cself_{species}_A2"] += float(
                np.sum(np.einsum("oij,oij->o", displacement, displacement))
            )

        sums["C_metal_EMIM_A2"] += float(
            np.sum(
                np.einsum(
                    "ij,ij->i",
                    collective_vectors["metal"],
                    collective_vectors["EMIM"],
                )
            )
        )
        sums["C_metal_TFSI_A2"] += float(
            np.sum(
                np.einsum(
                    "ij,ij->i",
                    collective_vectors["metal"],
                    collective_vectors["TFSI"],
                )
            )
        )
        sums["C_EMIM_TFSI_A2"] += float(
            np.sum(
                np.einsum(
                    "ij,ij->i",
                    collective_vectors["EMIM"],
                    collective_vectors["TFSI"],
                )
            )
        )

        n_origins_accumulated += len(chunk)

    result = {key: value / n_origins_accumulated for key, value in sums.items()}

    for species in SPECIES_ORDER:
        result[f"Cdistinct_{species}_A2"] = (
            result[f"C_{species}_{species}_A2"] - result[f"Cself_{species}_A2"]
        )

    result["n_time_origins"] = int(n_origins_accumulated)
    return result

def add_charge_combinations(
    table: pd.DataFrame,
    charge_numbers: dict[str, float],
) -> pd.DataFrame:
    z_m = charge_numbers["metal"]
    z_e = charge_numbers["EMIM"]
    z_t = charge_numbers["TFSI"]

    table = table.copy()

    table["charge_msd_formal_e2A2"] = (
        z_m**2 * table["C_metal_metal_A2"]
        + z_e**2 * table["C_EMIM_EMIM_A2"]
        + z_t**2 * table["C_TFSI_TFSI_A2"]
        + 2.0 * z_m * z_e * table["C_metal_EMIM_A2"]
        + 2.0 * z_m * z_t * table["C_metal_TFSI_A2"]
        + 2.0 * z_e * z_t * table["C_EMIM_TFSI_A2"]
    )

    table["charge_msd_self_formal_e2A2"] = (
        z_m**2 * table["Cself_metal_A2"]
        + z_e**2 * table["Cself_EMIM_A2"]
        + z_t**2 * table["Cself_TFSI_A2"]
    )

    table["charge_msd_correlation_formal_e2A2"] = (
        table["charge_msd_formal_e2A2"]
        - table["charge_msd_self_formal_e2A2"]
    )

    scale_squared = MODEL_CURRENT_CHARGE_SCALE**2
    table["charge_msd_model_e2A2"] = (
        scale_squared * table["charge_msd_formal_e2A2"]
    )
    table["charge_msd_self_model_e2A2"] = (
        scale_squared * table["charge_msd_self_formal_e2A2"]
    )

    return table

def calculate_correlation_table(
    positions: dict[str, np.ndarray],
    frame_interval_ps: float,
    lag_frames: np.ndarray,
    origin_stride: int,
    label: str,
) -> pd.DataFrame:

    rows: list[dict[str, float]] = []

    print(f"\nCalculating collective correlations for {label}...")
    for counter, lag in enumerate(lag_frames, start=1):
        row = calculate_one_lag(
            positions,
            int(lag),
            origin_stride,
            ORIGIN_CHUNK_SIZE,
        )
        row["lag_frames"] = int(lag)
        row["lag_time_ps"] = float(lag * frame_interval_ps)
        row["lag_time_ns"] = float(lag * frame_interval_ps / 1000.0)
        rows.append(row)

        if counter % 15 == 0 or counter == len(lag_frames):
            print(f"  completed {counter}/{len(lag_frames)} lag times")

    table = pd.DataFrame(rows).sort_values("lag_frames").reset_index(drop=True)
    return add_charge_combinations(table, FORMAL_CHARGE_NUMBERS)

def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residual = float(np.sum((y - fitted) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - residual / total if total > 0 else np.nan
    return float(slope), float(intercept), float(r2)

def loglog_exponent(x: np.ndarray, y: np.ndarray) -> float:
    valid = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) < MIN_FIT_POINTS:
        return np.nan
    slope, _, _ = linear_fit(np.log(x[valid]), np.log(y[valid]))
    return float(slope)

def conductivity_from_charge_slope(
    slope_e2_a2_per_ps: float,
    volume_a3: float,
    temperature_k: float,
) -> float:

    slope_si = (
        slope_e2_a2_per_ps
        * ELEMENTARY_CHARGE_C**2
        * ANGSTROM_TO_M**2
        / PS_TO_S
    )
    denominator = 6.0 * volume_a3 * A3_TO_M3 * BOLTZMANN_J_K * temperature_k
    sigma_s_m = slope_si / denominator
    return float(sigma_s_m * S_M_TO_MS_CM)

def fit_observable(
    table: pd.DataFrame,
    column: str,
    start_ns: float,
    end_ns: float,
) -> dict[str, float]:
    selected = table[
        (table["lag_time_ns"] >= start_ns)
        & (table["lag_time_ns"] <= end_ns)
        & np.isfinite(table[column])
    ]

    if len(selected) < MIN_FIT_POINTS:
        return {
            "slope": np.nan,
            "intercept": np.nan,
            "r2": np.nan,
            "beta": np.nan,
            "n_points": int(len(selected)),
        }

    x_ps = selected["lag_time_ps"].to_numpy(dtype=float)
    y = selected[column].to_numpy(dtype=float)
    slope, intercept, r2 = linear_fit(x_ps, y)
    beta = loglog_exponent(x_ps, y)

    return {
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "beta": beta,
        "n_points": int(len(selected)),
    }

def build_window_scan(
    table: pd.DataFrame,
    fit_windows_ns: tuple[tuple[float, float], ...],
    volume_a3: float,
    temperature_k: float,
    scope_label: str,
) -> pd.DataFrame:

    z = FORMAL_CHARGE_NUMBERS
    rows: list[dict[str, float | str]] = []

    observable_specs = {
        "total": ("charge_msd_formal_e2A2", 1.0),
        "self_total": ("charge_msd_self_formal_e2A2", 1.0),
        "correlation_total": ("charge_msd_correlation_formal_e2A2", 1.0),
        "metal_metal_total": ("C_metal_metal_A2", z["metal"] ** 2),
        "EMIM_EMIM_total": ("C_EMIM_EMIM_A2", z["EMIM"] ** 2),
        "TFSI_TFSI_total": ("C_TFSI_TFSI_A2", z["TFSI"] ** 2),
        "metal_self": ("Cself_metal_A2", z["metal"] ** 2),
        "EMIM_self": ("Cself_EMIM_A2", z["EMIM"] ** 2),
        "TFSI_self": ("Cself_TFSI_A2", z["TFSI"] ** 2),
        "metal_distinct": ("Cdistinct_metal_A2", z["metal"] ** 2),
        "EMIM_distinct": ("Cdistinct_EMIM_A2", z["EMIM"] ** 2),
        "TFSI_distinct": ("Cdistinct_TFSI_A2", z["TFSI"] ** 2),
        "metal_EMIM_cross": ("C_metal_EMIM_A2", 2.0 * z["metal"] * z["EMIM"]),
        "metal_TFSI_cross": ("C_metal_TFSI_A2", 2.0 * z["metal"] * z["TFSI"]),
        "EMIM_TFSI_cross": ("C_EMIM_TFSI_A2", 2.0 * z["EMIM"] * z["TFSI"]),
    }

    for start_ns, end_ns in fit_windows_ns:
        total_fit = fit_observable(
            table, "charge_msd_formal_e2A2", start_ns, end_ns
        )

        row: dict[str, float | str] = {
            "scope": scope_label,
            "fit_start_ns": start_ns,
            "fit_end_ns": end_ns,
            "fit_duration_ns": end_ns - start_ns,
            "n_fit_points": total_fit["n_points"],
            "total_charge_msd_beta": total_fit["beta"],
            "total_charge_msd_r2": total_fit["r2"],
        }

        fitted_slopes: dict[str, float] = {}

        for name, (column, charge_prefactor) in observable_specs.items():
            fit = fit_observable(table, column, start_ns, end_ns)
            weighted_slope = fit["slope"] * charge_prefactor
            fitted_slopes[name] = weighted_slope
            row[f"{name}_weighted_slope_e2A2_ps"] = weighted_slope
            row[f"sigma_{name}_formal_mS_cm"] = conductivity_from_charge_slope(
                weighted_slope,
                volume_a3,
                temperature_k,
            )

        row["sigma_total_model_mS_cm"] = (
            row["sigma_total_formal_mS_cm"] * MODEL_CURRENT_CHARGE_SCALE**2
        )

        decomposition_sum = (
            row["sigma_metal_metal_total_formal_mS_cm"]
            + row["sigma_EMIM_EMIM_total_formal_mS_cm"]
            + row["sigma_TFSI_TFSI_total_formal_mS_cm"]
            + row["sigma_metal_EMIM_cross_formal_mS_cm"]
            + row["sigma_metal_TFSI_cross_formal_mS_cm"]
            + row["sigma_EMIM_TFSI_cross_formal_mS_cm"]
        )
        self_distinct_cross_sum = (
            row["sigma_metal_self_formal_mS_cm"]
            + row["sigma_EMIM_self_formal_mS_cm"]
            + row["sigma_TFSI_self_formal_mS_cm"]
            + row["sigma_metal_distinct_formal_mS_cm"]
            + row["sigma_EMIM_distinct_formal_mS_cm"]
            + row["sigma_TFSI_distinct_formal_mS_cm"]
            + row["sigma_metal_EMIM_cross_formal_mS_cm"]
            + row["sigma_metal_TFSI_cross_formal_mS_cm"]
            + row["sigma_EMIM_TFSI_cross_formal_mS_cm"]
        )

        row["sigma_decomposition_sum_formal_mS_cm"] = decomposition_sum
        row["sigma_self_distinct_cross_sum_formal_mS_cm"] = self_distinct_cross_sum
        row["closure_error_decomposition_mS_cm"] = (
            decomposition_sum - row["sigma_total_formal_mS_cm"]
        )
        row["closure_error_self_distinct_mS_cm"] = (
            self_distinct_cross_sum - row["sigma_total_formal_mS_cm"]
        )

        rows.append(row)

    return pd.DataFrame(rows)

def select_diagnostic_window(window_scan: pd.DataFrame) -> pd.Series:

    candidates = window_scan.copy()
    candidates = candidates[
        np.isfinite(candidates["sigma_total_formal_mS_cm"])
        & np.isfinite(candidates["total_charge_msd_r2"])
        & (candidates["sigma_total_formal_mS_cm"] > 0)
    ]

    if candidates.empty:
        return window_scan.iloc[0]

    beta_penalty = np.abs(candidates["total_charge_msd_beta"] - 1.0).fillna(5.0)
    r2_penalty = (1.0 - candidates["total_charge_msd_r2"]).clip(lower=0).fillna(5.0)
    candidates = candidates.assign(score=beta_penalty + 2.0 * r2_penalty)
    return candidates.sort_values("score").iloc[0]

def generate_full_fit_windows(maximum_lag_ns: float) -> tuple[tuple[float, float], ...]:

    windows = set(FULL_FIT_WINDOWS_NS)
    for duration in SLIDING_WINDOW_LENGTHS_NS:
        start = 5.0
        while start + duration <= maximum_lag_ns + 1.0e-9:
            windows.add((round(start, 6), round(start + duration, 6)))
            start += SLIDING_WINDOW_START_STEP_NS
    return tuple(sorted(windows, key=lambda item: (item[0], item[1])))

def build_cumulative_scan(
    table: pd.DataFrame,
    volume_a3: float,
    temperature_k: float,
    sigma_ne_formal: float,
) -> pd.DataFrame:

    maximum_time_ns = float(table["lag_time_ns"].max())
    rows: list[dict[str, float]] = []

    for start_ns in CUMULATIVE_START_TIMES_NS:
        first_end = max(start_ns + 10.0, math.ceil((start_ns + 10.0) / CUMULATIVE_END_STEP_NS) * CUMULATIVE_END_STEP_NS)
        end_ns = first_end
        while end_ns <= maximum_time_ns + 1.0e-9:
            total = fit_observable(table, "charge_msd_formal_e2A2", start_ns, end_ns)
            self_fit = fit_observable(table, "charge_msd_self_formal_e2A2", start_ns, end_ns)
            sigma_total = conductivity_from_charge_slope(
                total["slope"], volume_a3, temperature_k
            )
            sigma_self = conductivity_from_charge_slope(
                self_fit["slope"], volume_a3, temperature_k
            )
            rows.append(
                {
                    "fit_start_ns": start_ns,
                    "fit_end_ns": end_ns,
                    "fit_duration_ns": end_ns - start_ns,
                    "n_fit_points": total["n_points"],
                    "total_charge_msd_beta": total["beta"],
                    "total_charge_msd_r2": total["r2"],
                    "sigma_total_formal_mS_cm": sigma_total,
                    "sigma_total_model_mS_cm": sigma_total * MODEL_CURRENT_CHARGE_SCALE**2,
                    "sigma_self_total_formal_mS_cm": sigma_self,
                    "ionicity_ratio_collective_over_NE": sigma_total / sigma_ne_formal,
                    "self_closure_ratio_collective_self_over_NE": sigma_self / sigma_ne_formal,
                }
            )
            end_ns += CUMULATIVE_END_STEP_NS

    return pd.DataFrame(rows)

def _window_overlap_fraction(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> float:
    overlap = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    denominator = min(end_a - start_a, end_b - start_b)
    return overlap / denominator if denominator > 0 else 0.0

def add_neighbor_stability_metrics(window_scan: pd.DataFrame) -> pd.DataFrame:

    result = window_scan.copy().reset_index(drop=True)
    local_counts: list[int] = []
    local_means: list[float] = []
    local_stds: list[float] = []
    local_cvs: list[float] = []
    local_relative_to_median: list[float] = []

    for _, row in result.iterrows():
        values: list[float] = []
        for _, other in result.iterrows():
            overlap = _window_overlap_fraction(
                float(row["fit_start_ns"]),
                float(row["fit_end_ns"]),
                float(other["fit_start_ns"]),
                float(other["fit_end_ns"]),
            )
            duration_ratio = float(other["fit_duration_ns"]) / float(row["fit_duration_ns"])
            if 0.67 <= duration_ratio <= 1.50 and overlap >= 0.50:
                value = float(other["sigma_total_formal_mS_cm"])
                if np.isfinite(value) and value > 0:
                    values.append(value)

        array = np.asarray(values, dtype=float)
        count = len(array)
        mean = float(np.mean(array)) if count else np.nan
        std = float(np.std(array, ddof=1)) if count > 1 else np.nan
        cv = std / abs(mean) if count > 1 and mean != 0 else np.nan
        median = float(np.median(array)) if count else np.nan
        current = float(row["sigma_total_formal_mS_cm"])
        relative = abs(current - median) / abs(median) if np.isfinite(median) and median != 0 else np.nan

        local_counts.append(count)
        local_means.append(mean)
        local_stds.append(std)
        local_cvs.append(cv)
        local_relative_to_median.append(relative)

    result["neighbor_window_count"] = local_counts
    result["neighbor_sigma_mean_mS_cm"] = local_means
    result["neighbor_sigma_std_mS_cm"] = local_stds
    result["neighbor_sigma_cv"] = local_cvs
    result["relative_difference_from_neighbor_median"] = local_relative_to_median
    return result

def evaluate_stable_collective_result(
    full_window_scan: pd.DataFrame,
    block_summary: pd.DataFrame,
    sigma_ne_formal: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    evaluated = add_neighbor_stability_metrics(full_window_scan)
    nearest_scheme: list[str | None] = []
    nearest_block_start: list[float] = []
    nearest_block_end: list[float] = []
    block_means: list[float] = []
    block_stds: list[float] = []
    block_sems: list[float] = []
    block_cvs: list[float] = []
    full_block_relative: list[float] = []
    block_counts: list[int] = []

    for _, row in evaluated.iterrows():
        best = None
        best_score = np.inf
        for _, block_row in block_summary.iterrows():
            overlap = _window_overlap_fraction(
                float(row["fit_start_ns"]),
                float(row["fit_end_ns"]),
                float(block_row["fit_start_ns"]),
                float(block_row["fit_end_ns"]),
            )
            if overlap <= 0:
                continue
            midpoint_difference = abs(
                0.5 * (float(row["fit_start_ns"]) + float(row["fit_end_ns"]))
                - 0.5 * (float(block_row["fit_start_ns"]) + float(block_row["fit_end_ns"]))
            )
            duration_difference = abs(
                float(row["fit_duration_ns"])
                - (float(block_row["fit_end_ns"]) - float(block_row["fit_start_ns"]))
            )
            score = midpoint_difference + 0.5 * duration_difference - 5.0 * overlap
            if score < best_score:
                best_score = score
                best = block_row

        if best is None:
            nearest_scheme.append(None)
            nearest_block_start.append(np.nan)
            nearest_block_end.append(np.nan)
            block_means.append(np.nan)
            block_stds.append(np.nan)
            block_sems.append(np.nan)
            block_cvs.append(np.nan)
            full_block_relative.append(np.nan)
            block_counts.append(0)
            continue

        mean = float(best.get("sigma_total_formal_mS_cm_block_mean", np.nan))
        std = float(best.get("sigma_total_formal_mS_cm_block_std", np.nan))
        sem = float(best.get("sigma_total_formal_mS_cm_block_sem", np.nan))
        cv = std / abs(mean) if np.isfinite(std) and np.isfinite(mean) and mean != 0 else np.nan
        full_value = float(row["sigma_total_formal_mS_cm"])
        relative = abs(full_value - mean) / abs(mean) if np.isfinite(mean) and mean != 0 else np.nan

        nearest_scheme.append(str(best["block_scheme"]))
        nearest_block_start.append(float(best["fit_start_ns"]))
        nearest_block_end.append(float(best["fit_end_ns"]))
        block_means.append(mean)
        block_stds.append(std)
        block_sems.append(sem)
        block_cvs.append(cv)
        full_block_relative.append(relative)
        block_counts.append(int(best["n_blocks"]))

    evaluated["nearest_block_scheme"] = nearest_scheme
    evaluated["nearest_block_fit_start_ns"] = nearest_block_start
    evaluated["nearest_block_fit_end_ns"] = nearest_block_end
    evaluated["nearest_block_sigma_mean_mS_cm"] = block_means
    evaluated["nearest_block_sigma_std_mS_cm"] = block_stds
    evaluated["nearest_block_sigma_sem_mS_cm"] = block_sems
    evaluated["nearest_block_sigma_cv"] = block_cvs
    evaluated["relative_difference_full_vs_block_mean"] = full_block_relative
    evaluated["nearest_block_count"] = block_counts

    evaluated["passes_beta"] = (
        evaluated["total_charge_msd_beta"].between(STABLE_BETA_MIN, STABLE_BETA_MAX)
    )
    evaluated["passes_r2"] = evaluated["total_charge_msd_r2"] >= STABLE_R2_MIN
    evaluated["passes_neighbor_stability"] = (
        (evaluated["neighbor_window_count"] >= STABLE_LOCAL_NEIGHBORS_MIN)
        & (evaluated["neighbor_sigma_cv"] <= STABLE_LOCAL_WINDOW_CV_MAX)
    )
    evaluated["passes_block_stability"] = (
        (evaluated["nearest_block_count"] >= 3)
        & (evaluated["nearest_block_sigma_cv"] <= STABLE_BLOCK_CV_MAX)
        & (
            evaluated["relative_difference_full_vs_block_mean"]
            <= STABLE_FULL_BLOCK_RELATIVE_DIFFERENCE_MAX
        )
    )
    evaluated["passes_self_NE_consistency"] = evaluated[
        "self_closure_ratio_collective_self_over_NE"
    ].between(STABLE_SELF_NE_MIN, STABLE_SELF_NE_MAX)
    evaluated["passes_positive_sigma"] = evaluated["sigma_total_formal_mS_cm"] > 0
    evaluated["is_stable_candidate"] = (
        evaluated["passes_beta"]
        & evaluated["passes_r2"]
        & evaluated["passes_neighbor_stability"]
        & evaluated["passes_block_stability"]
        & evaluated["passes_self_NE_consistency"]
        & evaluated["passes_positive_sigma"]
    )

    stable = evaluated[evaluated["is_stable_candidate"]].copy()
    if not stable.empty:
        stable["selection_score"] = (
            abs(stable["total_charge_msd_beta"] - 1.0)
            + 2.0 * (1.0 - stable["total_charge_msd_r2"]).clip(lower=0)
            + stable["neighbor_sigma_cv"].fillna(10.0)
            + stable["nearest_block_sigma_cv"].fillna(10.0)
            + stable["relative_difference_full_vs_block_mean"].fillna(10.0)
        )
        selected = stable.sort_values("selection_score").iloc[0]
        status = "stable_window_found"
    else:
        diagnostic = select_diagnostic_window(evaluated)
        selected = diagnostic
        status = "no_stable_window_found"

    summary = pd.DataFrame(
        [
            {
                "stability_status": status,
                "fit_start_ns": float(selected["fit_start_ns"]),
                "fit_end_ns": float(selected["fit_end_ns"]),
                "sigma_collective_formal_mS_cm": float(selected["sigma_total_formal_mS_cm"]),
                "sigma_collective_model_mS_cm": float(selected["sigma_total_model_mS_cm"]),
                "sigma_NE_formal_mS_cm": sigma_ne_formal,
                "ionicity_ratio_collective_over_NE": float(selected["sigma_total_formal_mS_cm"]) / sigma_ne_formal,
                "total_charge_msd_beta": float(selected["total_charge_msd_beta"]),
                "total_charge_msd_r2": float(selected["total_charge_msd_r2"]),
                "neighbor_sigma_cv": float(selected.get("neighbor_sigma_cv", np.nan)),
                "nearest_block_scheme": selected.get("nearest_block_scheme", None),
                "nearest_block_sigma_mean_mS_cm": float(selected.get("nearest_block_sigma_mean_mS_cm", np.nan)),
                "nearest_block_sigma_std_mS_cm": float(selected.get("nearest_block_sigma_std_mS_cm", np.nan)),
                "nearest_block_sigma_sem_mS_cm": float(selected.get("nearest_block_sigma_sem_mS_cm", np.nan)),
                "nearest_block_sigma_cv": float(selected.get("nearest_block_sigma_cv", np.nan)),
                "relative_difference_full_vs_block_mean": float(selected.get("relative_difference_full_vs_block_mean", np.nan)),
                "stable_candidate_count": int(evaluated["is_stable_candidate"].sum()),
                "interpretation": (
                    "Eligible as a final estimate only after visual confirmation."
                    if status == "stable_window_found"
                    else "Diagnostic only; no window met all convergence criteria."
                ),
            }
        ]
    )
    return evaluated, summary

def calculate_nernst_einstein(
    identity: SystemIdentity,
    counts: dict[str, int],
    volume_a3: float,
) -> pd.DataFrame:
    diffusion = DIFFUSION_COEFFICIENTS_NM2_NS.get(
        (identity.metal, identity.composition)
    )
    if diffusion is None:
        rows = []
        for species in SPECIES_ORDER:
            rows.append({
                "system": identity.system, "metal_species": identity.metal,
                "composition": identity.composition, "species": species,
                "number_of_ions": counts[species],
                "charge_number_formal": FORMAL_CHARGE_NUMBERS[species],
                "D_nm2_ns": np.nan, "sigma_NE_formal_mS_cm": np.nan,
                "sigma_NE_model_mS_cm": np.nan,
                "fraction_of_sigma_NE_formal": np.nan,
            })
        rows.append({
            "system": identity.system, "metal_species": identity.metal,
            "composition": identity.composition, "species": "TOTAL",
            "number_of_ions": sum(counts.values()), "charge_number_formal": np.nan,
            "D_nm2_ns": np.nan, "sigma_NE_formal_mS_cm": np.nan,
            "sigma_NE_model_mS_cm": np.nan, "fraction_of_sigma_NE_formal": np.nan,
        })
        return pd.DataFrame(rows)

    rows: list[dict[str, float | str]] = []
    total_formal = 0.0

    for species in SPECIES_ORDER:
        d_m2_s = diffusion[species] * NM2_NS_TO_M2_S
        z = FORMAL_CHARGE_NUMBERS[species]
        sigma_s_m = (
            counts[species]
            * (z * ELEMENTARY_CHARGE_C) ** 2
            * d_m2_s
            / (volume_a3 * A3_TO_M3 * BOLTZMANN_J_K * TEMPERATURE_K)
        )
        sigma_ms_cm = sigma_s_m * S_M_TO_MS_CM
        total_formal += sigma_ms_cm

        rows.append(
            {
                "system": identity.system,
                "metal_species": identity.metal,
                "composition": identity.composition,
                "species": species,
                "number_of_ions": counts[species],
                "charge_number_formal": z,
                "D_nm2_ns": diffusion[species],
                "sigma_NE_formal_mS_cm": sigma_ms_cm,
                "sigma_NE_model_mS_cm": sigma_ms_cm
                * MODEL_CURRENT_CHARGE_SCALE**2,
            }
        )

    total_row = {
        "system": identity.system,
        "metal_species": identity.metal,
        "composition": identity.composition,
        "species": "TOTAL",
        "number_of_ions": sum(counts.values()),
        "charge_number_formal": np.nan,
        "D_nm2_ns": np.nan,
        "sigma_NE_formal_mS_cm": total_formal,
        "sigma_NE_model_mS_cm": total_formal * MODEL_CURRENT_CHARGE_SCALE**2,
    }
    rows.append(total_row)

    table = pd.DataFrame(rows)
    table["fraction_of_sigma_NE_formal"] = (
        table["sigma_NE_formal_mS_cm"] / total_formal
    )
    return table

def run_block_analysis(
    positions: dict[str, np.ndarray],
    volumes_a3: np.ndarray,
    frame_interval_ps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    all_scans: list[pd.DataFrame] = []

    for scheme in BLOCK_SCHEMES:
        scheme_name = str(scheme["name"])
        block_length_ns = float(scheme["block_length_ns"])
        maximum_lag_ns = float(scheme["maximum_lag_ns"])
        expected_blocks = int(scheme["expected_blocks"])
        fit_windows_ns = tuple(scheme["fit_windows_ns"])

        intervals_per_block = int(round(block_length_ns * 1000.0 / frame_interval_ps))
        n_available_blocks = (positions["metal"].shape[0] - 1) // intervals_per_block
        n_blocks = min(expected_blocks, n_available_blocks)

        if n_blocks < 2:
            warnings.warn(f"Fewer than two complete blocks are available for {scheme_name}.")

        for block_index in range(n_blocks):
            start = block_index * intervals_per_block
            stop = start + intervals_per_block + 1
            block_positions = {
                species: positions[species][start:stop] for species in SPECIES_ORDER
            }
            block_volume_a3 = float(np.mean(volumes_a3[start:stop]))
            lag_frames = create_block_lag_frames(
                block_positions["metal"].shape[0],
                frame_interval_ps,
                maximum_lag_ns,
            )

            correlation = calculate_correlation_table(
                block_positions,
                frame_interval_ps,
                lag_frames=lag_frames,
                origin_stride=BLOCK_ORIGIN_STRIDE,
                label=f"{scheme_name}, block {block_index + 1}/{n_blocks}",
            )

            scan = build_window_scan(
                correlation,
                fit_windows_ns,
                block_volume_a3,
                TEMPERATURE_K,
                scope_label=f"{scheme_name}_block_{block_index + 1}",
            )
            scan.insert(0, "block_scheme", scheme_name)
            scan.insert(1, "block_index", block_index + 1)
            scan.insert(2, "block_start_ns", start * frame_interval_ps / 1000.0)
            scan.insert(3, "block_end_ns", (stop - 1) * frame_interval_ps / 1000.0)
            scan.insert(4, "block_average_volume_A3", block_volume_a3)
            all_scans.append(scan)

    block_scan = pd.concat(all_scans, ignore_index=True)

    numeric_columns = [
        column
        for column in block_scan.columns
        if column.startswith("sigma_")
        or column in {"total_charge_msd_beta", "total_charge_msd_r2"}
    ]

    summary_rows: list[dict[str, float | str]] = []
    for (scheme_name, start_ns, end_ns), group in block_scan.groupby(
        ["block_scheme", "fit_start_ns", "fit_end_ns"], sort=True
    ):
        row: dict[str, float | str] = {
            "block_scheme": str(scheme_name),
            "fit_start_ns": float(start_ns),
            "fit_end_ns": float(end_ns),
            "fit_duration_ns": float(end_ns - start_ns),
            "n_blocks": int(group["block_index"].nunique()),
        }
        for column in numeric_columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            row[f"{column}_block_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{column}_block_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else np.nan
            )
            row[f"{column}_block_sem"] = (
                float(values.std(ddof=1) / math.sqrt(len(values)))
                if len(values) > 1
                else np.nan
            )
        summary_rows.append(row)

    return block_scan, pd.DataFrame(summary_rows)

def add_identity_columns(table: pd.DataFrame, identity: SystemIdentity) -> pd.DataFrame:

    result = table.copy()

    result["system"] = identity.system
    result["metal_species"] = identity.metal
    result["composition"] = identity.composition

    identity_columns = ["system", "metal_species", "composition"]
    remaining_columns = [
        column for column in result.columns if column not in identity_columns
    ]
    return result[identity_columns + remaining_columns]

def save_plots(
    correlation: pd.DataFrame,
    window_scan: pd.DataFrame,
    cumulative_scan: pd.DataFrame,
    selected: pd.Series,
) -> None:
    figures_dir = OUTPUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(7.0, 5.0))
    axis = figure.add_subplot(111)
    axis.plot(
        correlation["lag_time_ns"],
        correlation["charge_msd_formal_e2A2"],
        marker="o",
        markersize=2,
        linewidth=1,
    )
    axis.set_xlabel("Lag time (ns)")
    axis.set_ylabel(r"Charge MSD ($e^2$ $\AA^2$)")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.grid(True, which="both", alpha=0.25)
    figure.tight_layout()
    figure.savefig(figures_dir / "collective_charge_msd.png", dpi=300)
    plt.close(figure)

    figure = plt.figure(figsize=(7.0, 5.0))
    axis = figure.add_subplot(111)
    midpoints = 0.5 * (
        window_scan["fit_start_ns"].to_numpy()
        + window_scan["fit_end_ns"].to_numpy()
    )
    axis.scatter(
        midpoints,
        window_scan["sigma_total_formal_mS_cm"],
        s=18,
        label="Collective",
    )
    axis.scatter(
        midpoints,
        window_scan["sigma_self_total_formal_mS_cm"],
        s=18,
        label="Self terms",
    )
    axis.set_xlabel("Fit-window midpoint (ns)")
    axis.set_ylabel("Conductivity (mS/cm)")
    axis.legend()
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(figures_dir / "conductivity_window_dependence.png", dpi=300)
    plt.close(figure)

    figure = plt.figure(figsize=(7.0, 5.0))
    axis = figure.add_subplot(111)
    for start_ns, group in cumulative_scan.groupby("fit_start_ns"):
        axis.plot(
            group["fit_end_ns"],
            group["sigma_total_formal_mS_cm"],
            marker="o",
            markersize=3,
            label=f"start {start_ns:g} ns",
        )
    axis.set_xlabel("Cumulative fit end time (ns)")
    axis.set_ylabel("Collective conductivity (mS/cm)")
    axis.legend()
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(figures_dir / "cumulative_conductivity_vs_fit_end.png", dpi=300)
    plt.close(figure)

    contribution_columns = [
        "sigma_metal_metal_total_formal_mS_cm",
        "sigma_EMIM_EMIM_total_formal_mS_cm",
        "sigma_TFSI_TFSI_total_formal_mS_cm",
        "sigma_metal_EMIM_cross_formal_mS_cm",
        "sigma_metal_TFSI_cross_formal_mS_cm",
        "sigma_EMIM_TFSI_cross_formal_mS_cm",
    ]
    labels = ["M-M", "E-E", "T-T", "M-E", "M-T", "E-T"]
    values = [float(selected[column]) for column in contribution_columns]

    figure = plt.figure(figsize=(7.0, 5.0))
    axis = figure.add_subplot(111)
    axis.bar(labels, values)
    axis.axhline(0.0, linewidth=1)
    axis.set_ylabel("Conductivity contribution (mS/cm)")
    axis.set_title(
        f"Diagnostic window: {selected['fit_start_ns']:.0f}-"
        f"{selected['fit_end_ns']:.0f} ns"
    )
    figure.tight_layout()
    figure.savefig(figures_dir / "conductivity_decomposition_diagnostic.png", dpi=300)
    plt.close(figure)

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MODULE 5A-D: COLLECTIVE IONIC TRANSPORT")
    print("=" * 80)

    identity = detect_system_identity(WORK_DIR)
    topology_file, trajectory_file = discover_files()
    coordinate_mode = detect_dump_coordinate_mode(trajectory_file)

    print(f"System       : {identity.system}")
    print(f"Topology     : {topology_file}")
    print(f"Trajectory   : {trajectory_file}")
    print(f"Coordinate mode: {coordinate_mode}")
    print(f"Frame interval : {FRAME_INTERVAL_PS} ps")

    universe = mda.Universe(
        str(topology_file),
        str(trajectory_file),
        format="LAMMPSDUMP",
        dt=FRAME_INTERVAL_PS,
    )

    groups = {
        species: universe.select_atoms(selection)
        for species, selection in SPECIES_SELECTIONS.items()
    }
    counts = {species: len(group) for species, group in groups.items()}

    for species, count in counts.items():
        if count == 0:
            raise RuntimeError(
                f"Selection '{SPECIES_SELECTIONS[species]}' returned zero {species} atoms."
            )

    formal_net_charge = sum(
        counts[species] * FORMAL_CHARGE_NUMBERS[species]
        for species in SPECIES_ORDER
    )

    print("\nSelected representative sites:")
    for species in SPECIES_ORDER:
        print(f"  {species:6s}: {counts[species]}")
    print(f"Formal net charge number: {formal_net_charge:+.6f} e")

    if abs(formal_net_charge) > 1.0e-8:
        raise RuntimeError(
            "The selected ionic species are not formally charge neutral. Check "
            "the representative atom types and molecule counts."
        )

    positions, volumes_a3, maximum_raw_step_a = extract_representative_positions(
        universe,
        groups,
        coordinate_mode,
    )

    average_volume_a3 = float(np.mean(volumes_a3))
    volume_std_a3 = float(np.std(volumes_a3, ddof=1)) if len(volumes_a3) > 1 else 0.0

    if REMOVE_IONIC_BARYCENTER_DRIFT:
        positions, barycenter_drift = remove_ionic_barycenter_drift(positions, identity)
    else:
        barycenter_drift = np.zeros((positions["metal"].shape[0], 3), dtype=float)

    if SAVE_UNWRAPPED_COORDINATES:
        np.savez_compressed(
            OUTPUT_DIR / "unwrapped_ion_representative_positions.npz",
            metal=positions["metal"],
            EMIM=positions["EMIM"],
            TFSI=positions["TFSI"],
            frame_interval_ps=FRAME_INTERVAL_PS,
            average_volume_a3=average_volume_a3,
        )

    nernst_einstein = calculate_nernst_einstein(
        identity,
        counts,
        average_volume_a3,
    )
    nernst_einstein.to_csv(
        OUTPUT_DIR / "nernst_einstein_conductivity.csv", index=False
    )
    sigma_ne_formal = float(
        nernst_einstein.loc[
            nernst_einstein["species"] == "TOTAL", "sigma_NE_formal_mS_cm"
        ].iloc[0]
    )

    full_lag_frames = create_full_lag_frames(
        positions["metal"].shape[0],
        FRAME_INTERVAL_PS,
        MAX_LAG_NS,
    )
    full_correlation = calculate_correlation_table(
        positions,
        FRAME_INTERVAL_PS,
        lag_frames=full_lag_frames,
        origin_stride=ORIGIN_STRIDE,
        label="full trajectory",
    )
    full_correlation = add_identity_columns(full_correlation, identity)
    full_correlation.to_csv(
        OUTPUT_DIR / "collective_correlation_functions.csv", index=False
    )

    full_fit_windows = generate_full_fit_windows(MAX_LAG_NS)
    full_window_scan = build_window_scan(
        full_correlation,
        full_fit_windows,
        average_volume_a3,
        TEMPERATURE_K,
        scope_label="full_trajectory",
    )
    full_window_scan["sigma_NE_input_formal_mS_cm"] = sigma_ne_formal
    full_window_scan["ionicity_ratio_collective_over_NE"] = (
        full_window_scan["sigma_total_formal_mS_cm"] / sigma_ne_formal
    )
    full_window_scan["self_closure_ratio_collective_self_over_NE"] = (
        full_window_scan["sigma_self_total_formal_mS_cm"] / sigma_ne_formal
    )
    full_window_scan = add_identity_columns(full_window_scan, identity)
    full_window_scan.to_csv(
        OUTPUT_DIR / "collective_conductivity_window_scan.csv", index=False
    )

    cumulative_scan = build_cumulative_scan(
        full_correlation,
        average_volume_a3,
        TEMPERATURE_K,
        sigma_ne_formal,
    )
    cumulative_scan = add_identity_columns(cumulative_scan, identity)
    cumulative_scan.to_csv(
        OUTPUT_DIR / "cumulative_collective_conductivity.csv", index=False
    )

    selected = select_diagnostic_window(full_window_scan)
    selected_summary = pd.DataFrame([selected.to_dict()])
    selected_summary["selection_note"] = (
        "Diagnostic automatic choice only; inspect block and neighboring-window "
        "stability before using as a manuscript value."
    )
    selected_summary.to_csv(
        OUTPUT_DIR / "selected_collective_conductivity_diagnostic.csv", index=False
    )

    block_scan, block_summary = run_block_analysis(
        positions,
        volumes_a3,
        FRAME_INTERVAL_PS,
    )
    block_scan["sigma_NE_input_formal_mS_cm"] = sigma_ne_formal
    block_scan["ionicity_ratio_collective_over_NE"] = (
        block_scan["sigma_total_formal_mS_cm"] / sigma_ne_formal
    )

    total_mean_column = "sigma_total_formal_mS_cm_block_mean"
    total_std_column = "sigma_total_formal_mS_cm_block_std"
    total_sem_column = "sigma_total_formal_mS_cm_block_sem"
    self_mean_column = "sigma_self_total_formal_mS_cm_block_mean"
    if total_mean_column in block_summary.columns:
        block_summary["sigma_NE_input_formal_mS_cm"] = sigma_ne_formal
        block_summary["ionicity_ratio_block_mean"] = (
            block_summary[total_mean_column] / sigma_ne_formal
        )
        block_summary["ionicity_ratio_block_std"] = (
            block_summary[total_std_column] / sigma_ne_formal
        )
        block_summary["ionicity_ratio_block_sem"] = (
            block_summary[total_sem_column] / sigma_ne_formal
        )
    if self_mean_column in block_summary.columns:
        block_summary["collective_self_over_NE_block_mean"] = (
            block_summary[self_mean_column] / sigma_ne_formal
        )

    block_scan = add_identity_columns(block_scan, identity)
    block_summary = add_identity_columns(block_summary, identity)
    block_scan.to_csv(
        OUTPUT_DIR / "block_collective_conductivity_window_scan.csv", index=False
    )
    block_summary.to_csv(
        OUTPUT_DIR / "block_collective_conductivity_summary.csv", index=False
    )

    stability_scan, stable_summary = evaluate_stable_collective_result(
        full_window_scan,
        block_summary,
        sigma_ne_formal,
    )
    stability_scan = add_identity_columns(stability_scan, identity)
    stable_summary = add_identity_columns(stable_summary, identity)
    stability_scan.to_csv(
        OUTPUT_DIR / "collective_conductivity_stability_scan.csv", index=False
    )
    stable_summary.to_csv(
        OUTPUT_DIR / "stable_collective_conductivity_summary.csv", index=False
    )

    closure_max = float(
        np.nanmax(np.abs(full_window_scan["closure_error_decomposition_mS_cm"]))
    )
    closure_self_distinct_max = float(
        np.nanmax(np.abs(full_window_scan["closure_error_self_distinct_mS_cm"]))
    )

    validation = {
        "system": identity.system,
        "metal_species": identity.metal,
        "composition": identity.composition,
        "temperature_K": TEMPERATURE_K,
        "frame_interval_ps": FRAME_INTERVAL_PS,
        "trajectory_frames": int(positions["metal"].shape[0]),
        "trajectory_duration_ns": float(
            (positions["metal"].shape[0] - 1) * FRAME_INTERVAL_PS / 1000.0
        ),
        "coordinate_mode": coordinate_mode,
        "species_counts": counts,
        "formal_net_charge_e": formal_net_charge,
        "average_volume_A3": average_volume_a3,
        "volume_std_A3": volume_std_a3,
        "remove_ionic_barycenter_drift": REMOVE_IONIC_BARYCENTER_DRIFT,
        "final_ionic_barycenter_drift_A": barycenter_drift[-1].tolist(),
        "maximum_raw_wrapped_step_A": maximum_raw_step_a,
        "model_current_charge_scale": MODEL_CURRENT_CHARGE_SCALE,
        "sigma_NE_input_formal_mS_cm": sigma_ne_formal,
        "maximum_decomposition_closure_error_mS_cm": closure_max,
        "maximum_self_distinct_closure_error_mS_cm": closure_self_distinct_max,
        "diagnostic_selected_window_ns": [
            float(selected["fit_start_ns"]),
            float(selected["fit_end_ns"]),
        ],
        "diagnostic_sigma_collective_formal_mS_cm": float(
            selected["sigma_total_formal_mS_cm"]
        ),
        "diagnostic_ionicity_ratio": float(
            selected["ionicity_ratio_collective_over_NE"]
        ),
        "stability_status": str(stable_summary["stability_status"].iloc[0]),
        "stable_candidate_count": int(stable_summary["stable_candidate_count"].iloc[0]),
        "full_lag_count": int(len(full_lag_frames)),
        "block_schemes": [str(scheme["name"]) for scheme in BLOCK_SCHEMES],
        "stability_criteria": {
            "beta_range": [STABLE_BETA_MIN, STABLE_BETA_MAX],
            "minimum_r2": STABLE_R2_MIN,
            "maximum_neighbor_window_cv": STABLE_LOCAL_WINDOW_CV_MAX,
            "minimum_neighbor_count": STABLE_LOCAL_NEIGHBORS_MIN,
            "maximum_block_cv": STABLE_BLOCK_CV_MAX,
            "maximum_full_block_relative_difference": STABLE_FULL_BLOCK_RELATIVE_DIFFERENCE_MAX,
            "self_over_NE_range": [STABLE_SELF_NE_MIN, STABLE_SELF_NE_MAX],
        },
        "notes": [
            "The automatically selected window is diagnostic, not automatically final.",
            "Final conductivity requires a stable_window_found status plus visual confirmation.",
            "Both 5x40 ns and 4x50 ns block schemes are reported.",
            "Negative cross contributions are physically allowed and indicate correlation-induced current cancellation for the corresponding charge signs.",
        ],
    }

    with (OUTPUT_DIR / "validation_report.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(validation, handle, indent=2)

    save_plots(full_correlation, full_window_scan, cumulative_scan, selected)

    print("\n" + "=" * 80)
    print("MODULE 5A-D COMPLETED")
    print("=" * 80)
    print(f"Results directory: {OUTPUT_DIR}")
    print(f"Nernst-Einstein conductivity: {sigma_ne_formal:.6g} mS/cm")
    print(
        "Diagnostic collective result: "
        f"{selected['sigma_total_formal_mS_cm']:.6g} mS/cm from "
        f"{selected['fit_start_ns']:.0f}-{selected['fit_end_ns']:.0f} ns"
    )
    print(
        "Diagnostic ionicity ratio: "
        f"{selected['ionicity_ratio_collective_over_NE']:.6g}"
    )
    print(
        "Convergence status: "
        f"{stable_summary['stability_status'].iloc[0]} "
        f"({int(stable_summary['stable_candidate_count'].iloc[0])} stable candidates)"
    )
    print("Inspect the stability scan, cumulative scan, and both block schemes.")

if __name__ == "__main__":
    main()