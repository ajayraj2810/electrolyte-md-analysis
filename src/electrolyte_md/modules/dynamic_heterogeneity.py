# -*- coding: utf-8 -*-
"""
Module 4: Dynamic Heterogeneity and State-Resolved Transport

Calculates:

1. MSD for metal, TFSI, and EMIM representative atoms
2. Local MSD exponent beta(t)
3. Non-Gaussian parameter alpha2(t)
4. Self Van Hove displacement distributions
5. Initial-state-conditioned metal MSD
6. Persistent-state metal MSD
7. Initially isolated versus clustered metal MSD
8. Persistently isolated versus clustered metal MSD

Run from inside one system folder.

Expected Module 1 input:
    master_coordination_*.csv.gz

Expected trajectory inputs:
    one LAMMPS data file
    one LAMMPS dump trajectory

Important:
    Coordinates are unwrapped by accumulating minimum-image
    displacements between consecutive trajectory frames.

Assumption:
    The simulation box is orthorhombic.

Author: Ajay Dwivedi
"""

from pathlib import Path
import json
import math

import MDAnalysis as mda
import numpy as np
import pandas as pd


# ============================================================
# USER SETTINGS
# ============================================================

WORK_DIR = Path.cwd()
OUTPUT_DIR = WORK_DIR / "dynamic_heterogeneity"

# Set explicit filenames when automatic discovery is ambiguous.
# Examples:
#
# TOPOLOGY_FILE = Path("system.data")
# TRAJECTORY_FILE = Path("system.lammpsdump")
#
# Leave as None for automatic recursive discovery.
TOPOLOGY_FILE = None
TRAJECTORY_FILE = None

# Atom selections.
METAL_SELECTION = "type 17"

# One representative atom per TFSI molecule.
# User mapping: TFSI representative/COM atom type = 7.
TFSI_REPRESENTATIVE_SELECTION = "type 7"

# One representative atom per EMIM molecule.
# User mapping: EMIM representative/COM atom type = 18.
EMIM_REPRESENTATIVE_SELECTION = "type 18"

# Maximum lag time used for MSD and NGP.
MAX_LAG_NS = 20.0

# Number of logarithmically distributed lag times.
N_LOG_LAGS = 70

# Include every short lag up to this number of saved frames.
N_EARLY_LINEAR_LAGS = 20

# Requested Van Hove lag times.
# The closest available trajectory lag will be used.
VAN_HOVE_LAG_TIMES_NS = [
    0.1,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
]

# Van Hove histogram settings.
VAN_HOVE_BIN_WIDTH_A = 0.10
VAN_HOVE_MAX_DISPLACEMENT_A = 30.0

# To reduce cost, use every Nth time origin for Van Hove.
VAN_HOVE_ORIGIN_STRIDE = 10

# Time-origin stride for MSD and NGP.
# Use 1 for maximum statistics.
MSD_ORIGIN_STRIDE = 1

# State categories.
STATE_ORDER = ["P", "PT", "T", "F"]

# Minimum number of displacement samples needed to report a value.
MIN_CONDITIONAL_SAMPLES = 100

# Save the unwrapped coordinates as compressed NumPy files.
SAVE_UNWRAPPED_COORDINATES = False

# Float32 greatly reduces memory and is sufficient for this analysis.
COORDINATE_DTYPE = np.float32


# ============================================================
# FILE DISCOVERY
# ============================================================

def find_single_file(pattern, excluded_directory=None):
    """Find exactly one file recursively below WORK_DIR."""

    matches = sorted(WORK_DIR.rglob(pattern))

    if excluded_directory is not None:
        matches = [
            path
            for path in matches
            if excluded_directory not in path.parents
        ]

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No file matching '{pattern}' was found below:\n"
            f"{WORK_DIR}"
        )

    if len(matches) > 1:
        print(f"\nMultiple files match '{pattern}':")

        for path in matches:
            print(f"  {path}")

        raise RuntimeError(
            "Specify the required file manually in USER SETTINGS."
        )

    return matches[0]


def discover_input_files():
    """Locate topology, trajectory, and master database."""

    master_file = find_single_file(
        "master_coordination_*.csv.gz",
        excluded_directory=OUTPUT_DIR,
    )

    if TOPOLOGY_FILE is None:
        topology_file = find_single_file(
            "*.data",
            excluded_directory=OUTPUT_DIR,
        )
    else:
        topology_file = WORK_DIR / TOPOLOGY_FILE

    if TRAJECTORY_FILE is None:
        dump_matches = []

        for pattern in [
            "*.lammpsdump",
            "*.lammpstrj",
            "*.dump",
        ]:
            dump_matches.extend(
                WORK_DIR.rglob(pattern)
            )

        dump_matches = sorted(
            set(dump_matches)
        )

        dump_matches = [
            path
            for path in dump_matches
            if OUTPUT_DIR not in path.parents
        ]

        if len(dump_matches) == 0:
            raise FileNotFoundError(
                "No LAMMPS trajectory file was found."
            )

        if len(dump_matches) > 1:
            print("\nMultiple trajectory files were found:")

            for path in dump_matches:
                print(f"  {path}")

            raise RuntimeError(
                "Set TRAJECTORY_FILE manually in USER SETTINGS."
            )

        trajectory_file = dump_matches[0]

    else:
        trajectory_file = WORK_DIR / TRAJECTORY_FILE

    for path in [
        master_file,
        topology_file,
        trajectory_file,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    return (
        topology_file,
        trajectory_file,
        master_file,
    )


# ============================================================
# MASTER DATABASE
# ============================================================

def read_master_database(master_file):
    """Read fields needed for state and cluster conditioning."""

    required_columns = [
        "system",
        "metal_species",
        "composition",
        "frame",
        "time_ps",
        "metal_local_index",
        "metal_atom_id",
        "coordination_state",
        "is_multi_metal_cluster",
    ]

    header = pd.read_csv(
        master_file,
        nrows=0,
    )

    missing = [
        column
        for column in required_columns
        if column not in header.columns
    ]

    if missing:
        raise KeyError(
            "Missing required master-database columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    master = pd.read_csv(
        master_file,
        usecols=required_columns,
        dtype={
            "system": "string",
            "metal_species": "string",
            "composition": "string",
            "coordination_state": "string",
        },
        low_memory=False,
    )

    master["frame"] = pd.to_numeric(
        master["frame"],
        errors="raise",
    ).astype(int)

    master["metal_local_index"] = pd.to_numeric(
        master["metal_local_index"],
        errors="raise",
    ).astype(int)

    master["metal_atom_id"] = pd.to_numeric(
        master["metal_atom_id"],
        errors="raise",
    ).astype(int)

    master["is_multi_metal_cluster"] = pd.to_numeric(
        master["is_multi_metal_cluster"],
        errors="raise",
    ).astype(int)

    master = master.sort_values(
        [
            "frame",
            "metal_local_index",
        ]
    ).reset_index(drop=True)

    return master


def create_state_and_cluster_matrices(master):
    """
    Convert the master table into frame x metal matrices.
    """

    frames = np.sort(
        master["frame"].unique()
    )

    metals = np.sort(
        master["metal_local_index"].unique()
    )

    frame_lookup = {
        int(frame): index
        for index, frame in enumerate(frames)
    }

    metal_lookup = {
        int(metal): index
        for index, metal in enumerate(metals)
    }

    states = np.empty(
        (len(frames), len(metals)),
        dtype="U2",
    )

    clusters = np.zeros(
        (len(frames), len(metals)),
        dtype=bool,
    )

    states[:] = ""

    for row in master[
        [
            "frame",
            "metal_local_index",
            "coordination_state",
            "is_multi_metal_cluster",
        ]
    ].itertuples(index=False):

        frame_index = frame_lookup[
            int(row.frame)
        ]

        metal_index = metal_lookup[
            int(row.metal_local_index)
        ]

        states[
            frame_index,
            metal_index
        ] = str(row.coordination_state)

        clusters[
            frame_index,
            metal_index
        ] = bool(
            row.is_multi_metal_cluster
        )

    if np.any(states == ""):
        raise RuntimeError(
            "State matrix contains missing frame-metal entries."
        )

    frame_time_table = (
        master[
            [
                "frame",
                "time_ps",
            ]
        ]
        .drop_duplicates("frame")
        .sort_values("frame")
    )

    times_ps = pd.to_numeric(
        frame_time_table["time_ps"],
        errors="coerce",
    ).to_numpy(dtype=float)

    if (
        len(times_ps) != len(frames)
        or not np.all(np.isfinite(times_ps))
    ):
        raise RuntimeError(
            "Could not construct a valid frame-time mapping."
        )

    time_differences = np.diff(times_ps)

    frame_interval_ps = float(
        np.median(time_differences)
    )

    if frame_interval_ps <= 0:
        raise RuntimeError(
            "Invalid frame interval in the master database."
        )

    return (
        frames,
        metals,
        times_ps,
        frame_interval_ps,
        states,
        clusters,
    )


# ============================================================
# TRAJECTORY COORDINATES
# ============================================================

def validate_orthorhombic_box(dimensions):
    """Confirm 90-degree box angles."""

    angles = np.asarray(
        dimensions[3:6],
        dtype=float,
    )

    if not np.allclose(
        angles,
        [90.0, 90.0, 90.0],
        atol=1.0e-3,
    ):
        raise ValueError(
            "This script currently supports orthorhombic boxes only."
        )


def minimum_image_displacement(delta, box_lengths):
    """Apply orthorhombic minimum-image convention."""

    return (
        delta
        - box_lengths
        * np.round(
            delta / box_lengths
        )
    )


def extract_unwrapped_positions(
    universe,
    atom_group,
    trajectory_frames,
    label,
):
    """
    Extract and unwrap selected atom coordinates.

    The unwrapped trajectory is constructed from consecutive
    minimum-image displacements.
    """

    n_frames = len(trajectory_frames)
    n_atoms = len(atom_group)

    if n_atoms == 0:
        raise ValueError(
            f"Selection for {label} returned zero atoms."
        )

    coordinates = np.empty(
        (
            n_frames,
            n_atoms,
            3,
        ),
        dtype=COORDINATE_DTYPE,
    )

    previous_wrapped = None
    previous_unwrapped = None

    print(
        f"Extracting {label}: "
        f"{n_atoms} atoms, {n_frames} frames"
    )

    for output_index, trajectory_frame in enumerate(
        trajectory_frames
    ):
        universe.trajectory[
            int(trajectory_frame)
        ]

        dimensions = universe.trajectory.ts.dimensions

        validate_orthorhombic_box(
            dimensions
        )

        box_lengths = np.asarray(
            dimensions[:3],
            dtype=np.float64,
        )

        wrapped = atom_group.positions.astype(
            np.float64,
            copy=True,
        )

        if output_index == 0:
            unwrapped = wrapped.copy()

        else:
            delta = (
                wrapped
                - previous_wrapped
            )

            delta = minimum_image_displacement(
                delta,
                box_lengths,
            )

            unwrapped = (
                previous_unwrapped
                + delta
            )

        coordinates[
            output_index
        ] = unwrapped.astype(
            COORDINATE_DTYPE
        )

        previous_wrapped = wrapped
        previous_unwrapped = unwrapped

        if (
            output_index > 0
            and output_index % 1000 == 0
        ):
            print(
                f"  {label}: "
                f"{output_index}/{n_frames} frames"
            )

    return coordinates


# ============================================================
# LAG-TIME SETUP
# ============================================================

def create_lag_frames(
    number_of_frames,
    frame_interval_ps,
):
    """Create mixed linear and logarithmic lag values."""

    maximum_lag_frames = int(
        min(
            number_of_frames - 1,
            round(
                MAX_LAG_NS
                * 1000.0
                / frame_interval_ps
            ),
        )
    )

    if maximum_lag_frames < 1:
        raise ValueError(
            "Trajectory is too short for the requested lag range."
        )

    early = np.arange(
        1,
        min(
            N_EARLY_LINEAR_LAGS,
            maximum_lag_frames,
        ) + 1,
        dtype=int,
    )

    logarithmic = np.unique(
        np.round(
            np.logspace(
                0,
                np.log10(
                    maximum_lag_frames
                ),
                N_LOG_LAGS,
            )
        ).astype(int)
    )

    lag_frames = np.unique(
        np.concatenate(
            [
                early,
                logarithmic,
                np.asarray(
                    [maximum_lag_frames],
                    dtype=int,
                ),
            ]
        )
    )

    lag_frames = lag_frames[
        lag_frames > 0
    ]

    lag_frames = lag_frames[
        lag_frames <= maximum_lag_frames
    ]

    return lag_frames


# ============================================================
# MSD AND NGP
# ============================================================

def displacement_squared_for_lag(
    positions,
    lag,
    origin_stride,
):
    """
    Return squared displacement for all origins and particles.

    Shape:
        number_of_origins x number_of_particles
    """

    starts = np.arange(
        0,
        positions.shape[0] - lag,
        origin_stride,
        dtype=int,
    )

    delta = (
        positions[starts + lag]
        - positions[starts]
    )

    squared = np.sum(
        delta.astype(np.float64) ** 2,
        axis=2,
    )

    return starts, squared


def calculate_msd_ngp(
    positions,
    lag_frames,
    frame_interval_ps,
    species_name,
):
    """Calculate conventional MSD, fourth moment, NGP, and beta."""

    rows = []

    for lag in lag_frames:

        _, squared = displacement_squared_for_lag(
            positions,
            lag,
            MSD_ORIGIN_STRIDE,
        )

        flattened = squared.ravel()

        mean_r2 = float(
            np.mean(flattened)
        )

        mean_r4 = float(
            np.mean(flattened ** 2)
        )

        if mean_r2 > 0:
            alpha2 = (
                3.0 * mean_r4
                / (
                    5.0
                    * mean_r2 ** 2
                )
                - 1.0
            )
        else:
            alpha2 = np.nan

        rows.append(
            {
                "species": species_name,
                "lag_frames": int(lag),
                "lag_time_ps": (
                    lag
                    * frame_interval_ps
                ),
                "lag_time_ns": (
                    lag
                    * frame_interval_ps
                    / 1000.0
                ),
                "msd_A2": mean_r2,
                "mean_r4_A4": mean_r4,
                "ngp_alpha2": alpha2,
                "n_samples": len(flattened),
            }
        )

    result = pd.DataFrame(rows)

    valid = (
        (result["lag_time_ps"] > 0)
        & (result["msd_A2"] > 0)
    )

    beta = np.full(
        len(result),
        np.nan,
        dtype=float,
    )

    valid_indices = np.where(
        valid.to_numpy()
    )[0]

    if len(valid_indices) >= 3:

        log_time = np.log(
            result.loc[
                valid,
                "lag_time_ps"
            ].to_numpy()
        )

        log_msd = np.log(
            result.loc[
                valid,
                "msd_A2"
            ].to_numpy()
        )

        beta_values = np.gradient(
            log_msd,
            log_time,
        )

        beta[
            valid_indices
        ] = beta_values

    result["local_msd_exponent_beta"] = beta

    return result


# ============================================================
# SELF VAN HOVE
# ============================================================

def nearest_available_lag(
    requested_ns,
    frame_interval_ps,
    number_of_frames,
):
    """Convert requested lag time to a valid frame lag."""

    lag = int(
        round(
            requested_ns
            * 1000.0
            / frame_interval_ps
        )
    )

    return max(
        1,
        min(
            lag,
            number_of_frames - 1,
        ),
    )


def calculate_van_hove_distributions(
    positions,
    requested_lag_times_ns,
    frame_interval_ps,
    species_name,
):
    """
    Calculate radial displacement probability densities.

    The saved quantity is P(r), normalized so that:
        integral P(r) dr = 1
    """

    edges = np.arange(
        0.0,
        VAN_HOVE_MAX_DISPLACEMENT_A
        + VAN_HOVE_BIN_WIDTH_A,
        VAN_HOVE_BIN_WIDTH_A,
    )

    centers = 0.5 * (
        edges[:-1]
        + edges[1:]
    )

    rows = []

    used_lags = set()

    for requested_ns in requested_lag_times_ns:

        lag = nearest_available_lag(
            requested_ns,
            frame_interval_ps,
            positions.shape[0],
        )

        if lag in used_lags:
            continue

        used_lags.add(lag)

        starts = np.arange(
            0,
            positions.shape[0] - lag,
            VAN_HOVE_ORIGIN_STRIDE,
            dtype=int,
        )

        delta = (
            positions[starts + lag]
            - positions[starts]
        )

        distances = np.linalg.norm(
            delta.astype(np.float64),
            axis=2,
        ).ravel()

        histogram, _ = np.histogram(
            distances,
            bins=edges,
            density=True,
        )

        for center, probability_density in zip(
            centers,
            histogram,
        ):
            rows.append(
                {
                    "species": species_name,
                    "lag_frames": lag,
                    "lag_time_ps": (
                        lag
                        * frame_interval_ps
                    ),
                    "lag_time_ns": (
                        lag
                        * frame_interval_ps
                        / 1000.0
                    ),
                    "r_A": center,
                    "probability_density_per_A":
                        probability_density,
                    "n_displacement_samples":
                        len(distances),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# CONDITIONAL METAL MSD
# ============================================================

def persistent_boolean_mask(
    condition_matrix,
    starts,
    lag,
):
    """
    Return origin x metal mask for particles satisfying the
    condition throughout the complete interval.

    This function processes one origin at a time to avoid a very
    large temporary array.
    """

    mask = np.zeros(
        (
            len(starts),
            condition_matrix.shape[1],
        ),
        dtype=bool,
    )

    for output_index, start in enumerate(starts):

        mask[
            output_index
        ] = np.all(
            condition_matrix[
                start:start + lag + 1
            ],
            axis=0,
        )

    return mask


def append_conditioned_result(
    rows,
    squared_displacements,
    mask,
    lag,
    frame_interval_ps,
    category_type,
    category_value,
    persistence_definition,
):
    """Summarize a conditioned displacement population."""

    selected = squared_displacements[
        mask
    ]

    n_samples = len(selected)

    if n_samples < MIN_CONDITIONAL_SAMPLES:
        mean_value = np.nan
        median_value = np.nan
    else:
        mean_value = float(
            np.mean(selected)
        )

        median_value = float(
            np.median(selected)
        )

    rows.append(
        {
            "category_type": category_type,
            "category_value": category_value,
            "persistence_definition":
                persistence_definition,
            "lag_frames": int(lag),
            "lag_time_ps": (
                lag
                * frame_interval_ps
            ),
            "lag_time_ns": (
                lag
                * frame_interval_ps
                / 1000.0
            ),
            "conditioned_msd_A2": mean_value,
            "median_squared_displacement_A2":
                median_value,
            "n_samples": n_samples,
        }
    )


def calculate_state_conditioned_msd(
    metal_positions,
    states,
    clusters,
    lag_frames,
    frame_interval_ps,
):
    """
    Calculate initial-state and persistent-state/cluster MSDs.
    """

    rows = []

    for lag in lag_frames:

        starts, squared = displacement_squared_for_lag(
            metal_positions,
            lag,
            MSD_ORIGIN_STRIDE,