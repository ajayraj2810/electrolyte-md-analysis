from pathlib import Path
import json
import math

import MDAnalysis as mda
import numpy as np
import pandas as pd

WORK_DIR = Path.cwd()
OUTPUT_DIR = WORK_DIR / "dynamic_heterogeneity"

TOPOLOGY_FILE = None
TRAJECTORY_FILE = None

METAL_SELECTION = "type 17"

TFSI_REPRESENTATIVE_SELECTION = "type 7"

EMIM_REPRESENTATIVE_SELECTION = "type 18"

MAX_LAG_NS = 20.0

N_LOG_LAGS = 70

N_EARLY_LINEAR_LAGS = 20

VAN_HOVE_LAG_TIMES_NS = [
    0.1,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
]

VAN_HOVE_BIN_WIDTH_A = 0.10
VAN_HOVE_MAX_DISPLACEMENT_A = 30.0

VAN_HOVE_ORIGIN_STRIDE = 10

MSD_ORIGIN_STRIDE = 1

STATE_ORDER = ["P", "PT", "T", "F"]

MIN_CONDITIONAL_SAMPLES = 100

SAVE_UNWRAPPED_COORDINATES = False

COORDINATE_DTYPE = np.float32

def find_single_file(pattern, excluded_directory=None):

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

def read_master_database(master_file):

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

def validate_orthorhombic_box(dimensions):

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

def create_lag_frames(
    number_of_frames,
    frame_interval_ps,
):

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

def displacement_squared_for_lag(
    positions,
    lag,
    origin_stride,
):

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

def nearest_available_lag(
    requested_ns,
    frame_interval_ps,
    number_of_frames,
):

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

def persistent_boolean_mask(
    condition_matrix,
    starts,
    lag,
):

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

    rows = []

    for lag in lag_frames:

        starts, squared = displacement_squared_for_lag(
            metal_positions,
            lag,
            MSD_ORIGIN_STRIDE,
        )

        initial_states = states[
            starts
        ]

        for state in STATE_ORDER:

            initial_mask = (
                initial_states == state
            )

            append_conditioned_result(
                rows=rows,
                squared_displacements=squared,
                mask=initial_mask,
                lag=lag,
                frame_interval_ps=frame_interval_ps,
                category_type="coordination_state",
                category_value=state,
                persistence_definition="initial_state",
            )

        initial_clustered = clusters[
            starts
        ]

        append_conditioned_result(
            rows=rows,
            squared_displacements=squared,
            mask=(~initial_clustered),
            lag=lag,
            frame_interval_ps=frame_interval_ps,
            category_type="cluster_status",
            category_value="isolated",
            persistence_definition="initial_status",
        )

        append_conditioned_result(
            rows=rows,
            squared_displacements=squared,
            mask=initial_clustered,
            lag=lag,
            frame_interval_ps=frame_interval_ps,
            category_type="cluster_status",
            category_value="clustered",
            persistence_definition="initial_status",
        )

        for state in STATE_ORDER:

            state_condition = (
                states == state
            )

            persistent_mask = persistent_boolean_mask(
                state_condition,
                starts,
                lag,
            )

            append_conditioned_result(
                rows=rows,
                squared_displacements=squared,
                mask=persistent_mask,
                lag=lag,
                frame_interval_ps=frame_interval_ps,
                category_type="coordination_state",
                category_value=state,
                persistence_definition="persistent_state",
            )

        persistent_isolated = persistent_boolean_mask(
            ~clusters,
            starts,
            lag,
        )

        persistent_clustered = persistent_boolean_mask(
            clusters,
            starts,
            lag,
        )

        append_conditioned_result(
            rows=rows,
            squared_displacements=squared,
            mask=persistent_isolated,
            lag=lag,
            frame_interval_ps=frame_interval_ps,
            category_type="cluster_status",
            category_value="isolated",
            persistence_definition="persistent_status",
        )

        append_conditioned_result(
            rows=rows,
            squared_displacements=squared,
            mask=persistent_clustered,
            lag=lag,
            frame_interval_ps=frame_interval_ps,
            category_type="cluster_status",
            category_value="clustered",
            persistence_definition="persistent_status",
        )

        print(
            f"Conditioned MSD completed for lag "
            f"{lag * frame_interval_ps / 1000.0:.4f} ns"
        )

    return pd.DataFrame(rows)

def summarize_ngp(msd_ngp_table):

    rows = []

    for species, data in msd_ngp_table.groupby(
        "species"
    ):

        valid = data[
            np.isfinite(
                data["ngp_alpha2"]
            )
        ]

        if valid.empty:
            continue

        maximum_index = valid[
            "ngp_alpha2"
        ].idxmax()

        maximum_row = valid.loc[
            maximum_index
        ]

        rows.append(
            {
                "species": species,
                "maximum_ngp_alpha2":
                    maximum_row["ngp_alpha2"],
                "time_of_maximum_ngp_ps":
                    maximum_row["lag_time_ps"],
                "time_of_maximum_ngp_ns":
                    maximum_row["lag_time_ns"],
                "msd_at_maximum_ngp_A2":
                    maximum_row["msd_A2"],
            }
        )

    return pd.DataFrame(rows)

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print("MODULE 4: DYNAMIC HETEROGENEITY")
    print("=" * 78)

    print("\nWorking directory:")
    print(WORK_DIR)

    (
        topology_file,
        trajectory_file,
        master_file,
    ) = discover_input_files()

    print("\nInput files:")
    print(f"Topology   : {topology_file}")
    print(f"Trajectory : {trajectory_file}")
    print(f"Database   : {master_file}")

    master = read_master_database(
        master_file
    )

    (
        database_frames,
        metal_indices,
        times_ps,
        frame_interval_ps,
        states,
        clusters,
    ) = create_state_and_cluster_matrices(
        master
    )

    system = str(
        master["system"].iloc[0]
    )

    metal_species = str(
        master["metal_species"].iloc[0]
    )

    composition = str(
        master["composition"].iloc[0]
    )

    print("\nSystem:")
    print(f"  Name          : {system}")
    print(f"  Metal         : {metal_species}")
    print(f"  Composition   : {composition}")
    print(f"  Frames        : {len(database_frames)}")
    print(f"  Metals        : {len(metal_indices)}")
    print(
        f"  Frame interval: "
        f"{frame_interval_ps:.6f} ps"
    )

    universe = mda.Universe(
        str(topology_file),
        str(trajectory_file),
        format="LAMMPSDUMP",
    )

    if database_frames.min() < 0:
        raise ValueError(
            "Negative database frame indices are not supported."
        )

    if database_frames.max() >= len(
        universe.trajectory
    ):
        raise IndexError(
            "The master database contains frame indices beyond "
            "the trajectory length."
        )

    metal_group = universe.select_atoms(
        METAL_SELECTION
    )

    tfsi_group = universe.select_atoms(
        TFSI_REPRESENTATIVE_SELECTION
    )

    emim_group = universe.select_atoms(
        EMIM_REPRESENTATIVE_SELECTION
    )

    if len(metal_group) != len(metal_indices):
        raise ValueError(
            "Metal selection count does not match the master database.\n"
            f"Trajectory selection: {len(metal_group)}\n"
            f"Database metals: {len(metal_indices)}"
        )

    trajectory_frames = database_frames.astype(int)

    metal_positions = extract_unwrapped_positions(
        universe=universe,
        atom_group=metal_group,
        trajectory_frames=trajectory_frames,
        label=metal_species,
    )

    tfsi_positions = extract_unwrapped_positions(
        universe=universe,
        atom_group=tfsi_group,
        trajectory_frames=trajectory_frames,
        label="TFSI",
    )

    emim_positions = extract_unwrapped_positions(
        universe=universe,
        atom_group=emim_group,
        trajectory_frames=trajectory_frames,
        label="EMIM",
    )

    if SAVE_UNWRAPPED_COORDINATES:

        np.savez_compressed(
            OUTPUT_DIR / "unwrapped_metal_positions.npz",
            positions=metal_positions,
            frames=trajectory_frames,
            times_ps=times_ps,
        )

        np.savez_compressed(
            OUTPUT_DIR / "unwrapped_tfsi_positions.npz",
            positions=tfsi_positions,
            frames=trajectory_frames,
            times_ps=times_ps,
        )

        np.savez_compressed(
            OUTPUT_DIR / "unwrapped_emim_positions.npz",
            positions=emim_positions,
            frames=trajectory_frames,
            times_ps=times_ps,
        )

    lag_frames = create_lag_frames(
        number_of_frames=len(
            trajectory_frames
        ),
        frame_interval_ps=frame_interval_ps,
    )

    pd.DataFrame(
        {
            "lag_frames": lag_frames,
            "lag_time_ps": (
                lag_frames
                * frame_interval_ps
            ),
            "lag_time_ns": (
                lag_frames
                * frame_interval_ps
                / 1000.0
            ),
        }
    ).to_csv(
        OUTPUT_DIR / "lag_times.csv",
        index=False,
    )

    species_tables = []

    for species_name, positions in [
        (metal_species, metal_positions),
        ("TFSI", tfsi_positions),
        ("EMIM", emim_positions),
    ]:

        print(
            f"\nCalculating MSD and NGP for {species_name}"
        )

        table = calculate_msd_ngp(
            positions=positions,
            lag_frames=lag_frames,
            frame_interval_ps=frame_interval_ps,
            species_name=species_name,
        )

        species_tables.append(
            table
        )

    msd_ngp = pd.concat(
        species_tables,
        ignore_index=True,
    )

    msd_ngp.insert(
        0,
        "composition",
        composition,
    )

    msd_ngp.insert(
        0,
        "metal_system",
        metal_species,
    )

    msd_ngp.insert(
        0,
        "system",
        system,
    )

    msd_ngp.to_csv(
        OUTPUT_DIR / "msd_ngp_all_species.csv",
        index=False,
    )

    ngp_summary = summarize_ngp(
        msd_ngp
    )

    ngp_summary.insert(
        0,
        "composition",
        composition,
    )

    ngp_summary.insert(
        0,
        "metal_system",
        metal_species,
    )

    ngp_summary.insert(
        0,
        "system",
        system,
    )

    ngp_summary.to_csv(
        OUTPUT_DIR / "ngp_peak_summary.csv",
        index=False,
    )

    van_hove_tables = []

    for species_name, positions in [
        (metal_species, metal_positions),
        ("TFSI", tfsi_positions),
        ("EMIM", emim_positions),
    ]:

        print(
            f"\nCalculating Van Hove distributions for "
            f"{species_name}"
        )

        van_hove = calculate_van_hove_distributions(
            positions=positions,
            requested_lag_times_ns=
                VAN_HOVE_LAG_TIMES_NS,
            frame_interval_ps=frame_interval_ps,
            species_name=species_name,
        )

        van_hove_tables.append(
            van_hove
        )

    van_hove_all = pd.concat(
        van_hove_tables,
        ignore_index=True,
    )

    van_hove_all.insert(
        0,
        "composition",
        composition,
    )

    van_hove_all.insert(
        0,
        "metal_system",
        metal_species,
    )

    van_hove_all.insert(
        0,
        "system",
        system,
    )

    van_hove_all.to_csv(
        OUTPUT_DIR
        / "self_van_hove_distributions.csv",
        index=False,
    )

    print(
        "\nCalculating state- and cluster-conditioned metal MSD"
    )

    conditioned_msd = calculate_state_conditioned_msd(
        metal_positions=metal_positions,
        states=states,
        clusters=clusters,
        lag_frames=lag_frames,
        frame_interval_ps=frame_interval_ps,
    )

    conditioned_msd.insert(
        0,
        "composition",
        composition,
    )

    conditioned_msd.insert(
        0,
        "metal_species",
        metal_species,
    )

    conditioned_msd.insert(
        0,
        "system",
        system,
    )

    conditioned_msd.to_csv(
        OUTPUT_DIR
        / "conditioned_metal_msd.csv",
        index=False,
    )

    metadata = {
        "system": system,
        "metal_species": metal_species,
        "composition": composition,
        "topology_file": str(topology_file),
        "trajectory_file": str(trajectory_file),
        "master_file": str(master_file),
        "frame_interval_ps": frame_interval_ps,
        "number_of_frames": len(
            trajectory_frames
        ),
        "number_of_metals": len(
            metal_group
        ),
        "number_of_tfsi_representatives": len(
            tfsi_group
        ),
        "number_of_emim_representatives": len(
            emim_group
        ),
        "maximum_lag_ns": MAX_LAG_NS,
        "number_of_lags": len(
            lag_frames
        ),
        "van_hove_lag_times_requested_ns":
            VAN_HOVE_LAG_TIMES_NS,
        "van_hove_origin_stride":
            VAN_HOVE_ORIGIN_STRIDE,
        "msd_origin_stride":
            MSD_ORIGIN_STRIDE,
        "metal_selection":
            METAL_SELECTION,
        "tfsi_representative_selection":
            TFSI_REPRESENTATIVE_SELECTION,
        "emim_representative_selection":
            EMIM_REPRESENTATIVE_SELECTION,
    }

    with (
        OUTPUT_DIR
        / "dynamic_heterogeneity_metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            metadata,
            handle,
            indent=2,
        )

    print("\n" + "=" * 78)
    print("MODULE 4 COMPLETED")
    print("=" * 78)

    print("\nResults saved in:")
    print(OUTPUT_DIR)

if __name__ == "__main__":
    main()