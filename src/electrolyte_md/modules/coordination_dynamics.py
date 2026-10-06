# -*- coding: utf-8 -*-
"""
Module 3: Coordination Dynamics

Run this script from inside one metal folder, for example:

    path/to/system/

The script automatically searches the current folder and all subfolders for:

    master_coordination_*.csv.gz
    tfsi_bridging_*.csv.gz

It calculates:

    - Continuous contact lifetimes
    - Intermittent contact lifetimes
    - PEO oxygen residence
    - PEO chain residence
    - TFSI oxygen residence
    - TFSI molecule residence
    - P/PT/T/F state residence
    - State transitions
    - Ligand exchange frequencies
    - TFSI bridge lifetimes
    - Block-averaged uncertainties
    - Survival functions

All source text is ASCII-compatible to avoid encoding errors.
"""

from pathlib import Path
from collections import Counter, defaultdict
import json

import numpy as np
import pandas as pd


# ============================================================
# USER SETTINGS
# ============================================================

# The script uses the folder from which it is executed.
WORK_DIR = Path.cwd()

# Output will be created inside the current system folder.
OUTPUT_DIR = WORK_DIR / "coordination_dynamics"

# Saved trajectory-frame interval.
FRAME_INTERVAL_PS = 10.0

# Number of blocks for uncertainty calculations.
N_BLOCKS = 5

# A contact may disappear for this many saved frames and still count
# as one intermittent contact.
#
# 2 frames x 10 ps/frame = 20 ps tolerance.
GAP_TOLERANCE_FRAMES = 2

# Remove an initial trajectory fraction when necessary.
# Use 0.0 for the complete trajectory.
DISCARD_INITIAL_FRACTION = 0.0

# Minimum number of events needed to generate a survival curve.
MIN_EVENTS_FOR_SURVIVAL = 5


# ============================================================
# CONTACT DEFINITIONS
# ============================================================

CONTACT_TYPES = {
    "peo_oxygen": "peo_oxygen_atom_ids",
    "peo_chain": "peo_chain_resids",
    "tfsi_oxygen": "tfsi_oxygen_atom_ids",
    "tfsi_molecule": "tfsi_resids",
}


MASTER_USE_COLUMNS = [
    "system",
    "metal_species",
    "composition",
    "frame",
    "time_ps",
    "metal_local_index",
    "metal_atom_id",
    "coordination_state",
    "peo_oxygen_atom_ids",
    "peo_chain_resids",
    "tfsi_oxygen_atom_ids",
    "tfsi_resids",
]


STRING_COLUMNS = {
    "system": "string",
    "metal_species": "string",
    "composition": "string",
    "coordination_state": "string",
    "peo_oxygen_atom_ids": "string",
    "peo_chain_resids": "string",
    "tfsi_oxygen_atom_ids": "string",
    "tfsi_resids": "string",
}


# ============================================================
# FILE DISCOVERY
# ============================================================

def find_single_file(pattern):
    """
    Search recursively below the current folder.

    Raises a clear error when no file or multiple files are found.
    """

    matches = sorted(WORK_DIR.rglob(pattern))

    # Do not accidentally read files created inside Module 3 output.
    matches = [
        path
        for path in matches
        if OUTPUT_DIR not in path.parents
    ]

    if len(matches) == 0:
        raise FileNotFoundError(
            f"\nCould not find a file matching:\n"
            f"    {pattern}\n\n"
            f"Search folder:\n"
            f"    {WORK_DIR}\n\n"
            f"Check that Module 1 outputs are located somewhere "
            f"inside the current system folder."
        )

    if len(matches) > 1:
        print("\nMultiple matching files were found:")

        for index, path in enumerate(matches, start=1):
            print(f"  {index}: {path}")

        raise RuntimeError(
            "\nMore than one matching database was found. "
            "Keep only one system database inside this system folder, "
            "or specify the path manually in find_input_files()."
        )

    return matches[0]


def find_optional_file(pattern):
    """
    Find one optional file. Return None if not found.
    """

    matches = sorted(WORK_DIR.rglob(pattern))

    matches = [
        path
        for path in matches
        if OUTPUT_DIR not in path.parents
    ]

    if len(matches) == 0:
        return None

    if len(matches) > 1:
        print(
            f"Warning: multiple files match {pattern}. "
            f"Using the first one:"
        )

        for path in matches:
            print(f"  {path}")

    return matches[0]


def find_input_files():
    """
    Locate the Module 1 files inside the current system folder.
    """

    master_file = find_single_file(
        "master_coordination_*.csv.gz"
    )

    bridge_file = find_optional_file(
        "tfsi_bridging_*.csv.gz"
    )

    return master_file, bridge_file


# ============================================================
# GENERAL HELPERS
# ============================================================

def parse_id_set(value):
    """
    Convert a semicolon-separated string into a set of integers.

    Examples
    --------
    "41;42;45" -> {41, 42, 45}
    empty value -> set()
    """

    if pd.isna(value):
        return set()

    text = str(value).strip()

    if text in {
        "",
        "nan",
        "None",
        "<NA>",
    }:
        return set()

    output = set()

    for token in text.split(";"):
        token = token.strip()

        if token == "":
            continue

        try:
            output.add(int(float(token)))
        except ValueError:
            continue

    return output


def discard_initial_frames(dataframe, fraction):
    """
    Remove an initial fraction of trajectory frames.
    """

    if fraction <= 0.0:
        return dataframe.copy()

    if fraction >= 1.0:
        raise ValueError(
            "DISCARD_INITIAL_FRACTION must be between 0 and 1."
        )

    unique_frames = np.sort(
        dataframe["frame"].unique()
    )

    number_to_discard = int(
        len(unique_frames) * fraction
    )

    retained_frames = set(
        unique_frames[number_to_discard:]
    )

    return dataframe[
        dataframe["frame"].isin(retained_frames)
    ].copy()


def create_frame_block_mapping(frames, number_of_blocks):
    """
    Divide unique frames into contiguous blocks.
    """

    unique_frames = np.sort(
        np.unique(frames)
    )

    if len(unique_frames) < number_of_blocks:
        raise ValueError(
            f"Only {len(unique_frames)} frames are present, "
            f"but N_BLOCKS is {number_of_blocks}."
        )

    frame_groups = np.array_split(
        unique_frames,
        number_of_blocks,
    )

    mapping = {}

    for block_number, group in enumerate(
        frame_groups,
        start=1,
    ):
        for frame in group:
            mapping[int(frame)] = block_number

    return mapping


def create_presence_runs(
    present_frames,
    gap_tolerance_frames,
):
    """
    Convert frames where a contact is present into residence events.

    Continuous:
        gap_tolerance_frames = 0

    Intermittent:
        gap_tolerance_frames > 0
    """

    frames = sorted(
        set(int(frame) for frame in present_frames)
    )

    if len(frames) == 0:
        return []

    maximum_difference = (
        gap_tolerance_frames + 1
    )

    runs = []

    start_frame = frames[0]
    previous_frame = frames[0]

    for current_frame in frames[1:]:

        frame_difference = (
            current_frame - previous_frame
        )

        if frame_difference <= maximum_difference:
            previous_frame = current_frame

        else:
            runs.append(
                (start_frame, previous_frame)
            )

            start_frame = current_frame
            previous_frame = current_frame

    runs.append(
        (start_frame, previous_frame)
    )

    return runs


def duration_frames(start_frame, end_frame):
    """
    Inclusive number of saved frames.
    """

    return int(
        end_frame - start_frame + 1
    )


def duration_ps(start_frame, end_frame):
    """
    Event duration based on saved-frame count.
    """

    return (
        duration_frames(
            start_frame,
            end_frame,
        )
        * FRAME_INTERVAL_PS
    )


def calculate_kaplan_meier(
    durations,
    right_censored,
):
    """
    Calculate a simple Kaplan-Meier survival function.
    """

    durations = np.asarray(
        durations,
        dtype=float,
    )

    right_censored = np.asarray(
        right_censored,
        dtype=bool,
    )

    valid = (
        np.isfinite(durations)
        & (durations > 0)
    )

    durations = durations[valid]
    right_censored = right_censored[valid]

    if len(durations) == 0:
        return pd.DataFrame(
            columns=[
                "time_ps",
                "survival_probability",
                "n_at_risk",
                "n_terminated",
                "n_censored",
            ]
        )

    unique_times = np.sort(
        np.unique(durations)
    )

    survival = 1.0

    rows = [
        {
            "time_ps": 0.0,
            "survival_probability": 1.0,
            "n_at_risk": len(durations),
            "n_terminated": 0,
            "n_censored": 0,
        }
    ]

    for current_time in unique_times:

        at_risk = int(
            np.sum(
                durations >= current_time
            )
        )

        at_current_time = (
            durations == current_time
        )

        number_terminated = int(
            np.sum(
                at_current_time
                & (~right_censored)
            )
        )

        number_censored = int(
            np.sum(
                at_current_time
                & right_censored
            )
        )

        if at_risk > 0:
            survival *= (
                1.0
                - number_terminated / at_risk
            )

        rows.append(
            {
                "time_ps": current_time,
                "survival_probability": survival,
                "n_at_risk": at_risk,
                "n_terminated": number_terminated,
                "n_censored": number_censored,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# LOAD MASTER DATABASE
# ============================================================

def read_master_database(master_file):
    """
    Read only columns needed for Module 3.
    """

    print("\nReading master database:")
    print(master_file)

    header = pd.read_csv(
        master_file,
        nrows=0,
    )

    available_columns = set(
        header.columns
    )

    missing_columns = [
        column
        for column in MASTER_USE_COLUMNS
        if column not in available_columns
    ]

    if missing_columns:
        raise KeyError(
            "The master database is missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_columns
            )
        )

    master = pd.read_csv(
        master_file,
        usecols=MASTER_USE_COLUMNS,
        dtype=STRING_COLUMNS,
        low_memory=False,
    )

    integer_columns = [
        "frame",
        "metal_local_index",
        "metal_atom_id",
    ]

    for column in integer_columns:
        master[column] = pd.to_numeric(
            master[column],
            errors="raise",
        ).astype(int)

    master = discard_initial_frames(
        master,
        DISCARD_INITIAL_FRACTION,
    )

    master = master.sort_values(
        [
            "metal_local_index",
            "frame",
        ]
    ).reset_index(drop=True)

    return master


# ============================================================
# CONTACT RESIDENCE EVENTS
# ============================================================

def build_contact_presence(master, partner_column):
    """
    Construct:
        (metal index, partner ID) -> list of presence frames
    """

    presence = defaultdict(list)

    for row in master[
        [
            "frame",
            "metal_local_index",
            partner_column,
        ]
    ].itertuples(index=False):

        frame = int(row.frame)
        metal_index = int(
            row.metal_local_index
        )

        partners = parse_id_set(
            getattr(row, partner_column)
        )

        for partner_id in partners:
            presence[
                (metal_index, partner_id)
            ].append(frame)

    return presence


def build_contact_events(
    master,
    contact_type,
    partner_column,
    event_definition,
    gap_tolerance_frames,
    first_frame,
    last_frame,
    frame_to_block,
    system,
    metal,
    composition,
    metal_atom_lookup,
):
    """
    Build continuous or intermittent contact-lifetime events.
    """

    presence = build_contact_presence(
        master,
        partner_column,
    )

    rows = []

    for (
        metal_index,
        partner_id,
    ), frames in presence.items():

        runs = create_presence_runs(
            frames,
            gap_tolerance_frames,
        )

        for start_frame, end_frame in runs:

            left_censored = (
                start_frame == first_frame
            )

            right_censored = (
                end_frame == last_frame
            )

            rows.append(
                {
                    "system": system,
                    "metal_species": metal,
                    "composition": composition,
                    "contact_type": contact_type,
                    "event_definition": event_definition,
                    "metal_local_index": metal_index,
                    "metal_atom_id": metal_atom_lookup[
                        metal_index
                    ],
                    "partner_id": partner_id,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start_time_ps": (
                        start_frame
                        * FRAME_INTERVAL_PS
                    ),
                    "end_time_ps": (
                        end_frame
                        * FRAME_INTERVAL_PS
                    ),
                    "duration_frames": duration_frames(
                        start_frame,
                        end_frame,
                    ),
                    "duration_ps": duration_ps(
                        start_frame,
                        end_frame,
                    ),
                    "left_censored": int(
                        left_censored
                    ),
                    "right_censored": int(
                        right_censored
                    ),
                    "censored": int(
                        left_censored
                        or right_censored
                    ),
                    "start_block": frame_to_block[
                        start_frame
                    ],
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# STATE RESIDENCE EVENTS
# ============================================================

def build_state_events(
    master,
    first_frame,
    last_frame,
    frame_to_block,
    system,
    metal,
    composition,
):
    """
    Build contiguous P, PT, T, and F state events.
    """

    rows = []

    for metal_index, metal_data in master.groupby(
        "metal_local_index",
        sort=False,
    ):

        metal_data = metal_data.sort_values(
            "frame"
        ).reset_index(drop=True)

        frames = metal_data[
            "frame"
        ].to_numpy(dtype=int)

        states = metal_data[
            "coordination_state"
        ].astype(str).to_numpy()

        metal_atom_id = int(
            metal_data[
                "metal_atom_id"
            ].iloc[0]
        )

        start_index = 0

        for index in range(
            1,
            len(metal_data) + 1,
        ):

            at_end = (
                index == len(metal_data)
            )

            if at_end:
                terminate_event = True

            else:
                state_changed = (
                    states[index]
                    != states[index - 1]
                )

                frame_gap = (
                    frames[index]
                    != frames[index - 1] + 1
                )

                terminate_event = (
                    state_changed
                    or frame_gap
                )

            if terminate_event:

                start_frame = int(
                    frames[start_index]
                )

                end_frame = int(
                    frames[index - 1]
                )

                state = states[start_index]

                left_censored = (
                    start_frame == first_frame
                )

                right_censored = (
                    end_frame == last_frame
                )

                rows.append(
                    {
                        "system": system,
                        "metal_species": metal,
                        "composition": composition,
                        "metal_local_index": int(
                            metal_index
                        ),
                        "metal_atom_id": metal_atom_id,
                        "state": state,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "start_time_ps": (
                            start_frame
                            * FRAME_INTERVAL_PS
                        ),
                        "end_time_ps": (
                            end_frame
                            * FRAME_INTERVAL_PS
                        ),
                        "duration_frames": duration_frames(
                            start_frame,
                            end_frame,
                        ),
                        "duration_ps": duration_ps(
                            start_frame,
                            end_frame,
                        ),
                        "left_censored": int(
                            left_censored
                        ),
                        "right_censored": int(
                            right_censored
                        ),
                        "censored": int(
                            left_censored
                            or right_censored
                        ),
                        "start_block": frame_to_block[
                            start_frame
                        ],
                    }
                )

                start_index = index

    return pd.DataFrame(rows)


# ============================================================
# STATE TRANSITIONS
# ============================================================

def build_state_transition_table(
    master,
    system,
    metal,
    composition,
):
    """
    Calculate transition counts between consecutive saved frames.

    Both self-transitions and state-changing transitions are stored.
    """

    state_order = [
        "P",
        "PT",
        "T",
        "F",
    ]

    counts = Counter()

    for _, metal_data in master.groupby(
        "metal_local_index",
        sort=False,
    ):

        metal_data = metal_data.sort_values(
            "frame"
        )

        frames = metal_data[
            "frame"
        ].to_numpy(dtype=int)

        states = metal_data[
            "coordination_state"
        ].astype(str).to_numpy()

        for index in range(
            1,
            len(frames),
        ):

            if frames[index] != frames[index - 1] + 1:
                continue

            counts[
                (
                    states[index - 1],
                    states[index],
                )
            ] += 1

    rows = []

    for from_state in state_order:

        outgoing_total = sum(
            counts[
                (from_state, to_state)
            ]
            for to_state in state_order
        )

        changing_total = sum(
            counts[
                (from_state, to_state)
            ]
            for to_state in state_order
            if to_state != from_state
        )

        for to_state in state_order:

            count = counts[
                (from_state, to_state)
            ]

            conditional_probability_all_steps = (
                count / outgoing_total
                if outgoing_total > 0
                else 0.0
            )

            conditional_probability_given_change = (
                count / changing_total
                if (
                    changing_total > 0
                    and to_state != from_state
                )
                else 0.0
            )

            rows.append(
                {
                    "system": system,
                    "metal_species": metal,
                    "composition": composition,
                    "from_state": from_state,
                    "to_state": to_state,
                    "transition_count": count,
                    "conditional_probability_all_steps":
                        conditional_probability_all_steps,
                    "conditional_probability_given_change":
                        conditional_probability_given_change,
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# LIGAND EXCHANGE
# ============================================================

def build_exchange_events(
    master,
    frame_to_block,
    system,
    metal,
    composition,
):
    """
    Compare coordination partners in consecutive frames.
    """

    rows = []

    for metal_index, metal_data in master.groupby(
        "metal_local_index",
        sort=False,
    ):

        metal_data = metal_data.sort_values(
            "frame"
        ).reset_index(drop=True)

        metal_atom_id = int(
            metal_data[
                "metal_atom_id"
            ].iloc[0]
        )

        for index in range(
            1,
            len(metal_data),
        ):

            previous = metal_data.iloc[
                index - 1
            ]

            current = metal_data.iloc[
                index
            ]

            previous_frame = int(
                previous["frame"]
            )

            current_frame = int(
                current["frame"]
            )

            if current_frame != previous_frame + 1:
                continue

            for contact_type, column in CONTACT_TYPES.items():

                previous_set = parse_id_set(
                    previous[column]
                )

                current_set = parse_id_set(
                    current[column]