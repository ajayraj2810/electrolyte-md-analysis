from pathlib import Path
from collections import Counter, defaultdict
import json

import numpy as np
import pandas as pd

WORK_DIR = Path.cwd()

OUTPUT_DIR = WORK_DIR / "coordination_dynamics"

FRAME_INTERVAL_PS = 10.0

N_BLOCKS = 5

GAP_TOLERANCE_FRAMES = 2

DISCARD_INITIAL_FRACTION = 0.0

MIN_EVENTS_FOR_SURVIVAL = 5

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

def find_single_file(pattern):

    matches = sorted(WORK_DIR.rglob(pattern))

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

    master_file = find_single_file(
        "master_coordination_*.csv.gz"
    )

    bridge_file = find_optional_file(
        "tfsi_bridging_*.csv.gz"
    )

    return master_file, bridge_file

def parse_id_set(value):

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

    return int(
        end_frame - start_frame + 1
    )

def duration_ps(start_frame, end_frame):

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

def read_master_database(master_file):

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

def build_contact_presence(master, partner_column):

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

def build_state_events(
    master,
    first_frame,
    last_frame,
    frame_to_block,
    system,
    metal,
    composition,
):

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

def build_state_transition_table(
    master,
    system,
    metal,
    composition,
):

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

def build_exchange_events(
    master,
    frame_to_block,
    system,
    metal,
    composition,
):

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
                )

                gained = (
                    current_set - previous_set
                )

                lost = (
                    previous_set - current_set
                )

                retained = (
                    previous_set
                    & current_set
                )

                rows.append(
                    {
                        "system": system,
                        "metal_species": metal,
                        "composition": composition,
                        "frame": current_frame,
                        "time_ps": (
                            current_frame
                            * FRAME_INTERVAL_PS
                        ),
                        "block": frame_to_block[
                            current_frame
                        ],
                        "metal_local_index": int(
                            metal_index
                        ),
                        "metal_atom_id": metal_atom_id,
                        "contact_type": contact_type,
                        "n_previous": len(
                            previous_set
                        ),
                        "n_current": len(
                            current_set
                        ),
                        "n_gained": len(gained),
                        "n_lost": len(lost),
                        "n_retained": len(
                            retained
                        ),
                        "gained_ids": ";".join(
                            str(value)
                            for value in sorted(gained)
                        ),
                        "lost_ids": ";".join(
                            str(value)
                            for value in sorted(lost)
                        ),
                        "exchange_occurred": int(
                            len(gained) > 0
                            or len(lost) > 0
                        ),
                    }
                )

    return pd.DataFrame(rows)

def build_tfsi_bridge_events(
    bridge_file,
    event_definition,
    gap_tolerance_frames,
    first_frame,
    last_frame,
    frame_to_block,
    system,
    metal,
    composition,
):

    if bridge_file is None:
        return pd.DataFrame()

    bridge_data = pd.read_csv(
        bridge_file,
        usecols=[
            "frame",
            "tfsi_resid",
            "is_bridging",
        ],
        low_memory=False,
    )

    bridge_data["frame"] = pd.to_numeric(
        bridge_data["frame"],
        errors="raise",
    ).astype(int)

    bridge_data = discard_initial_frames(
        bridge_data,
        DISCARD_INITIAL_FRACTION,
    )

    bridge_data = bridge_data[
        bridge_data["is_bridging"] == 1
    ].copy()

    presence = defaultdict(list)

    for row in bridge_data.itertuples(
        index=False
    ):
        presence[
            int(row.tfsi_resid)
        ].append(
            int(row.frame)
        )

    rows = []

    for tfsi_resid, frames in presence.items():

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
                    "event_definition": event_definition,
                    "tfsi_resid": tfsi_resid,
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

def summarize_events(
    events,
    grouping_columns,
):

    if events.empty:
        return pd.DataFrame()

    rows = []

    for group_key, group in events.groupby(
        grouping_columns,
        dropna=False,
    ):

        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        row = dict(
            zip(
                grouping_columns,
                group_key,
            )
        )

        all_durations = group[
            "duration_ps"
        ].astype(float)

        uncensored = group[
            group["censored"] == 0
        ]

        uncensored_durations = uncensored[
            "duration_ps"
        ].astype(float)

        row.update(
            {
                "n_events_total": len(group),
                "n_events_uncensored": len(
                    uncensored
                ),
                "fraction_censored": group[
                    "censored"
                ].mean(),
                "mean_duration_all_ps":
                    all_durations.mean(),
                "median_duration_all_ps":
                    all_durations.median(),
                "mean_duration_uncensored_ps": (
                    uncensored_durations.mean()
                    if len(uncensored) > 0
                    else np.nan
                ),
                "median_duration_uncensored_ps": (
                    uncensored_durations.median()
                    if len(uncensored) > 0
                    else np.nan
                ),
                "maximum_duration_ps":
                    all_durations.max(),
            }
        )

        rows.append(row)

    return pd.DataFrame(rows)

def calculate_block_summary(
    events,
    grouping_columns,
):

    if events.empty:
        return pd.DataFrame(), pd.DataFrame()

    uncensored = events[
        events["censored"] == 0
    ].copy()

    if uncensored.empty:
        return pd.DataFrame(), pd.DataFrame()

    block_grouping = (
        grouping_columns
        + ["start_block"]
    )

    block_means = (
        uncensored
        .groupby(
            block_grouping,
            dropna=False,
            as_index=False,
        )
        .agg(
            mean_lifetime_ps=(
                "duration_ps",
                "mean",
            ),
            median_lifetime_ps=(
                "duration_ps",
                "median",
            ),
            n_events=(
                "duration_ps",
                "size",
            ),
        )
    )

    summary_rows = []

    for group_key, group in block_means.groupby(
        grouping_columns,
        dropna=False,
    ):

        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        row = dict(
            zip(
                grouping_columns,
                group_key,
            )
        )

        values = group[
            "mean_lifetime_ps"
        ].astype(float)

        number_of_blocks = len(values)

        standard_deviation = (
            values.std(ddof=1)
            if number_of_blocks > 1
            else np.nan
        )

        row.update(
            {
                "mean_of_block_means_ps":
                    values.mean(),
                "block_std_ps":
                    standard_deviation,
                "block_sem_ps": (
                    standard_deviation
                    / np.sqrt(number_of_blocks)
                    if number_of_blocks > 1
                    else np.nan
                ),
                "n_blocks":
                    number_of_blocks,
                "total_uncensored_events":
                    group["n_events"].sum(),
            }
        )

        summary_rows.append(row)

    return (
        block_means,
        pd.DataFrame(summary_rows),
    )

def create_survival_tables(
    events,
    grouping_columns,
):

    if events.empty:
        return pd.DataFrame()

    outputs = []

    for group_key, group in events.groupby(
        grouping_columns,
        dropna=False,
    ):

        if len(group) < MIN_EVENTS_FOR_SURVIVAL:
            continue

        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        group_information = dict(
            zip(
                grouping_columns,
                group_key,
            )
        )

        survival = calculate_kaplan_meier(
            group["duration_ps"],
            group["right_censored"].astype(bool),
        )

        for column, value in reversed(
            list(group_information.items())
        ):
            survival.insert(
                0,
                column,
                value,
            )

        outputs.append(survival)

    if len(outputs) == 0:
        return pd.DataFrame()

    return pd.concat(
        outputs,
        ignore_index=True,
    )

def run_contact_lifetime_analysis(
    master,
    output_dir,
    first_frame,
    last_frame,
    frame_to_block,
    system,
    metal,
    composition,
    metal_atom_lookup,
):

    event_tables = []

    for contact_type, partner_column in CONTACT_TYPES.items():

        print(
            f"Calculating continuous {contact_type} lifetimes..."
        )

        continuous = build_contact_events(
            master=master,
            contact_type=contact_type,
            partner_column=partner_column,
            event_definition="continuous",
            gap_tolerance_frames=0,
            first_frame=first_frame,
            last_frame=last_frame,
            frame_to_block=frame_to_block,
            system=system,
            metal=metal,
            composition=composition,
            metal_atom_lookup=metal_atom_lookup,
        )

        print(
            f"Calculating intermittent {contact_type} lifetimes..."
        )

        intermittent = build_contact_events(
            master=master,
            contact_type=contact_type,
            partner_column=partner_column,
            event_definition=(
                f"intermittent_gap_"
                f"{GAP_TOLERANCE_FRAMES}_frames"
            ),
            gap_tolerance_frames=GAP_TOLERANCE_FRAMES,
            first_frame=first_frame,
            last_frame=last_frame,
            frame_to_block=frame_to_block,
            system=system,
            metal=metal,
            composition=composition,
            metal_atom_lookup=metal_atom_lookup,
        )

        event_tables.extend(
            [
                continuous,
                intermittent,
            ]
        )

    events = pd.concat(
        event_tables,
        ignore_index=True,
    )

    events.to_csv(
        output_dir
        / "contact_lifetime_events.csv.gz",
        index=False,
        compression="gzip",
    )

    grouping = [
        "system",
        "metal_species",
        "composition",
        "contact_type",
        "event_definition",
    ]

    summary = summarize_events(
        events,
        grouping,
    )

    summary.to_csv(
        output_dir
        / "contact_lifetime_summary.csv",
        index=False,
    )

    block_means, block_summary = (
        calculate_block_summary(
            events,
            grouping,
        )
    )

    block_means.to_csv(
        output_dir
        / "contact_lifetime_block_means.csv",
        index=False,
    )

    block_summary.to_csv(
        output_dir
        / "contact_lifetime_block_summary.csv",
        index=False,
    )

    survival = create_survival_tables(
        events,
        grouping,
    )

    survival.to_csv(
        output_dir
        / "contact_survival_functions.csv",
        index=False,
    )

def run_state_analysis(
    master,
    output_dir,
    first_frame,
    last_frame,
    frame_to_block,
    system,
    metal,
    composition,
):

    print(
        "Calculating coordination-state residence times..."
    )

    events = build_state_events(
        master=master,
        first_frame=first_frame,
        last_frame=last_frame,
        frame_to_block=frame_to_block,
        system=system,
        metal=metal,
        composition=composition,
    )

    events.to_csv(
        output_dir
        / "coordination_state_events.csv.gz",
        index=False,
        compression="gzip",
    )

    grouping = [
        "system",
        "metal_species",
        "composition",
        "state",
    ]

    summary = summarize_events(
        events,
        grouping,
    )

    summary.to_csv(
        output_dir
        / "coordination_state_lifetime_summary.csv",
        index=False,
    )

    block_means, block_summary = (
        calculate_block_summary(
            events,
            grouping,
        )
    )

    block_means.to_csv(
        output_dir
        / "coordination_state_block_means.csv",
        index=False,
    )

    block_summary.to_csv(
        output_dir
        / "coordination_state_block_summary.csv",
        index=False,
    )

    survival = create_survival_tables(
        events,
        grouping,
    )

    survival.to_csv(
        output_dir
        / "coordination_state_survival.csv",
        index=False,
    )

    transitions = build_state_transition_table(
        master,
        system,
        metal,
        composition,
    )

    transitions.to_csv(
        output_dir
        / "coordination_state_transition_matrix.csv",
        index=False,
    )

def run_exchange_analysis(
    master,
    output_dir,
    frame_to_block,
    system,
    metal,
    composition,
):

    print(
        "Calculating ligand-exchange statistics..."
    )

    exchange_events = build_exchange_events(
        master,
        frame_to_block,
        system,
        metal,
        composition,
    )

    exchange_events.to_csv(
        output_dir
        / "ligand_exchange_events.csv.gz",
        index=False,
        compression="gzip",
    )

    block_summary = (
        exchange_events
        .groupby(
            [
                "system",
                "metal_species",
                "composition",
                "contact_type",
                "block",
            ],
            as_index=False,
        )
        .agg(
            n_observations=(
                "exchange_occurred",
                "size",
            ),
            n_exchange_steps=(
                "exchange_occurred",
                "sum",
            ),
            total_gained=(
                "n_gained",
                "sum",
            ),
            total_lost=(
                "n_lost",
                "sum",
            ),
            mean_gained_per_saved_step=(
                "n_gained",
                "mean",
            ),
            mean_lost_per_saved_step=(
                "n_lost",
                "mean",
            ),
        )
    )

    block_summary[
        "exchange_probability_per_saved_step"
    ] = (
        block_summary["n_exchange_steps"]
        / block_summary["n_observations"]
    )

    block_summary[
        "exchange_steps_per_metal_ns"
    ] = (
        block_summary["n_exchange_steps"]
        / (
            block_summary["n_observations"]
            * FRAME_INTERVAL_PS
            / 1000.0
        )
    )

    block_summary.to_csv(
        output_dir
        / "ligand_exchange_block_summary.csv",
        index=False,
    )

    summary = (
        block_summary
        .groupby(
            [
                "system",
                "metal_species",
                "composition",
                "contact_type",
            ],
            as_index=False,
        )
        .agg(
            mean_exchange_probability=(
                "exchange_probability_per_saved_step",
                "mean",
            ),
            block_std_exchange_probability=(
                "exchange_probability_per_saved_step",
                "std",
            ),
            mean_exchange_steps_per_metal_ns=(
                "exchange_steps_per_metal_ns",
                "mean",
            ),
            block_std_exchange_steps_per_metal_ns=(
                "exchange_steps_per_metal_ns",
                "std",
            ),
            mean_gained_per_saved_step=(
                "mean_gained_per_saved_step",
                "mean",
            ),
            mean_lost_per_saved_step=(
                "mean_lost_per_saved_step",
                "mean",
            ),
            n_blocks=(
                "block",
                "nunique",
            ),
        )
    )

    summary[
        "block_sem_exchange_steps_per_metal_ns"
    ] = (
        summary[
            "block_std_exchange_steps_per_metal_ns"
        ]
        / np.sqrt(
            summary["n_blocks"]
        )
    )

    summary.to_csv(
        output_dir
        / "ligand_exchange_summary.csv",
        index=False,
    )

def run_bridge_analysis(
    bridge_file,
    output_dir,
    first_frame,
    last_frame,
    frame_to_block,
    system,
    metal,
    composition,
):

    if bridge_file is None:
        print(
            "No tfsi_bridging_*.csv.gz file was found. "
            "Skipping bridge lifetime analysis."
        )

        return

    print("\nReading TFSI bridging file:")
    print(bridge_file)

    continuous = build_tfsi_bridge_events(
        bridge_file=bridge_file,
        event_definition="continuous",
        gap_tolerance_frames=0,
        first_frame=first_frame,
        last_frame=last_frame,
        frame_to_block=frame_to_block,
        system=system,
        metal=metal,
        composition=composition,
    )

    intermittent = build_tfsi_bridge_events(
        bridge_file=bridge_file,
        event_definition=(
            f"intermittent_gap_"
            f"{GAP_TOLERANCE_FRAMES}_frames"
        ),
        gap_tolerance_frames=GAP_TOLERANCE_FRAMES,
        first_frame=first_frame,
        last_frame=last_frame,
        frame_to_block=frame_to_block,
        system=system,
        metal=metal,
        composition=composition,
    )

    bridge_events = pd.concat(
        [
            continuous,
            intermittent,
        ],
        ignore_index=True,
    )

    bridge_events.to_csv(
        output_dir
        / "tfsi_bridge_lifetime_events.csv.gz",
        index=False,
        compression="gzip",
    )

    grouping = [
        "system",
        "metal_species",
        "composition",
        "event_definition",
    ]

    summary = summarize_events(
        bridge_events,
        grouping,
    )

    summary.to_csv(
        output_dir
        / "tfsi_bridge_lifetime_summary.csv",
        index=False,
    )

    block_means, block_summary = (
        calculate_block_summary(
            bridge_events,
            grouping,
        )
    )

    block_means.to_csv(
        output_dir
        / "tfsi_bridge_lifetime_block_means.csv",
        index=False,
    )

    block_summary.to_csv(
        output_dir
        / "tfsi_bridge_lifetime_block_summary.csv",
        index=False,
    )

    survival = create_survival_tables(
        bridge_events,
        grouping,
    )

    survival.to_csv(
        output_dir
        / "tfsi_bridge_survival.csv",
        index=False,
    )

def main():

    print("=" * 78)
    print("MODULE 3: COORDINATION DYNAMICS")
    print("=" * 78)

    print("\nCurrent working directory:")
    print(WORK_DIR)

    master_file, bridge_file = find_input_files()

    print("\nInput files found:")
    print(f"Master database : {master_file}")
    print(
        f"TFSI bridge file: "
        f"{bridge_file if bridge_file else 'not found'}"
    )

    master = read_master_database(
        master_file
    )

    if master.empty:
        raise RuntimeError(
            "The master database is empty after filtering."
        )

    system = str(
        master["system"].iloc[0]
    )

    metal = str(
        master["metal_species"].iloc[0]
    )

    composition = str(
        master["composition"].iloc[0]
    )

    first_frame = int(
        master["frame"].min()
    )

    last_frame = int(
        master["frame"].max()
    )

    number_of_frames = int(
        master["frame"].nunique()
    )

    number_of_metals = int(
        master[
            "metal_local_index"
        ].nunique()
    )

    frame_to_block = create_frame_block_mapping(
        master["frame"].unique(),
        N_BLOCKS,
    )

    metal_atom_lookup = (
        master[
            [
                "metal_local_index",
                "metal_atom_id",
            ]
        ]
        .drop_duplicates(
            "metal_local_index"
        )
        .set_index(
            "metal_local_index"
        )[
            "metal_atom_id"
        ]
        .astype(int)
        .to_dict()
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nSystem information:")
    print(f"System            : {system}")
    print(f"Metal             : {metal}")
    print(f"Composition       : {composition}")
    print(f"Number of frames  : {number_of_frames}")
    print(f"Number of metals  : {number_of_metals}")
    print(f"First frame       : {first_frame}")
    print(f"Last frame        : {last_frame}")
    print(
        f"Nominal time span : "
        f"{number_of_frames * FRAME_INTERVAL_PS / 1000.0:.3f} ns"
    )

    run_contact_lifetime_analysis(
        master=master,
        output_dir=OUTPUT_DIR,
        first_frame=first_frame,
        last_frame=last_frame,
        frame_to_block=frame_to_block,
        system=system,
        metal=metal,
        composition=composition,
        metal_atom_lookup=metal_atom_lookup,
    )

    run_state_analysis(
        master=master,
        output_dir=OUTPUT_DIR,
        first_frame=first_frame,
        last_frame=last_frame,
        frame_to_block=frame_to_block,
        system=system,
        metal=metal,
        composition=composition,
    )

    run_exchange_analysis(
        master=master,
        output_dir=OUTPUT_DIR,
        frame_to_block=frame_to_block,
        system=system,
        metal=metal,
        composition=composition,
    )

    run_bridge_analysis(
        bridge_file=bridge_file,
        output_dir=OUTPUT_DIR,
        first_frame=first_frame,
        last_frame=last_frame,
        frame_to_block=frame_to_block,
        system=system,
        metal=metal,
        composition=composition,
    )

    metadata = {
        "working_directory": str(WORK_DIR),
        "master_file": str(master_file),
        "tfsi_bridge_file": (
            str(bridge_file)
            if bridge_file is not None
            else None
        ),
        "system": system,
        "metal_species": metal,
        "composition": composition,
        "frame_interval_ps": FRAME_INTERVAL_PS,
        "number_of_frames": number_of_frames,
        "number_of_metals": number_of_metals,
        "first_frame": first_frame,
        "last_frame": last_frame,
        "number_of_blocks": N_BLOCKS,
        "gap_tolerance_frames":
            GAP_TOLERANCE_FRAMES,
        "gap_tolerance_ps": (
            GAP_TOLERANCE_FRAMES
            * FRAME_INTERVAL_PS
        ),
        "discard_initial_fraction":
            DISCARD_INITIAL_FRACTION,
    }

    with (
        OUTPUT_DIR
        / "coordination_dynamics_metadata.json"
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
    print("MODULE 3 COMPLETED")
    print("=" * 78)

    print("\nResults saved in:")
    print(OUTPUT_DIR)

if __name__ == "__main__":
    main()