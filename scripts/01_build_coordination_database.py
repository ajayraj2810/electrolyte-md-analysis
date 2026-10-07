# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 19:03:37 2026

@author: Ajay R. Dwivedi
"""

"""
Master coordination and bridging database for:

    polymer / ionic-liquid / salt electrolyte containing generic metal ions

For every metal ion at every frame, this script stores:

    - Coordinating PEO oxygen atoms
    - Coordinating PEO chains
    - Local EO indices on each PEO chain
    - Coordinating TFSI oxygen atoms
    - Coordinating TFSI molecules
    - PEO and TFSI coordination numbers
    - TFSI denticity
    - P / PT / T / F coordination states
    - TFSI-mediated metal bridging
    - Shared PEO-chain occupancy
    - Local PEO-mediated bridging
    - Remote shared-chain occupancy
    - Metal-TFSI cluster membership

PEO definitions
---------------
Shared-chain occupancy:
    Two or more metals coordinate the same PEO chain.

Local PEO bridge:
    Two metals coordinate the same PEO chain and their nearest coordinating
    EO sites are separated by no more than LOCAL_PEO_BRIDGE_MAX_EO.

Remote shared-chain occupancy:
    Two metals coordinate the same PEO chain but their coordinating EO sites
    are farther apart than LOCAL_PEO_BRIDGE_MAX_EO.

Outputs
-------
1. master_coordination_<system>.csv.gz
2. frame_summary_<system>.csv
3. tfsi_bridging_<system>.csv.gz
4. peo_chain_bridging_<system>.csv.gz
5. cluster_summary_<system>.csv.gz
6. metadata_<system>.json

Author: Ajay Dwivedi
"""

import csv
import gzip
import json
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Set

import MDAnalysis as mda
import numpy as np
from MDAnalysis.lib.distances import capped_distance


# ============================================================
# USER SETTINGS
# ============================================================

SYSTEMS = {
    "example_system": {
        "topology": "system.data",
        "trajectory": "system.lammpsdump",
        "metal": "metal",
        "composition": "composition",
        "metal_selection": "type 17",
        "peo_oxygen_selection": "resid 1:40 and type 4",
        "tfsi_oxygen_selection": "type 9",
        # Set these from the first minimum of the corresponding RDFs.
        "peo_cutoff_A": 4.0,
        "tfsi_cutoff_A": 4.0,
    },
}


# Physical time between saved trajectory frames.
FRAME_INTERVAL_PS = 10.0

# Trajectory frame range.
START_FRAME = 0
STOP_FRAME = None
STRIDE = 1

# Maximum EO-index separation for local PEO-mediated bridging.
#
# Example:
# Metal A coordinates EO indices 5 and 6.
# Metal B coordinates EO indices 8 and 9.
# Minimum separation = 2.
# This pair is a local bridge when the threshold is 4.
LOCAL_PEO_BRIDGE_MAX_EO = 4

# Number of processed frames accumulated before writing.
WRITE_EVERY_N_FRAMES = 100

OUTPUT_ROOT = Path("coordination_database")

# Prefer LAMMPS atom IDs in output.
STORE_ATOM_IDS = True


# ============================================================
# UNION-FIND FOR METAL-TFSI CLUSTERS
# ============================================================

class UnionFind:
    """Disjoint-set data structure for connected components."""

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def add(self, item):
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0

    def find(self, item):
        self.add(item)

        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])

        return self.parent[item]

    def union(self, item_a, item_b):
        root_a = self.find(item_a)
        root_b = self.find(item_b)

        if root_a == root_b:
            return

        if self.rank[root_a] < self.rank[root_b]:
            self.parent[root_a] = root_b

        elif self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a

        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def encode_integer_list(values: Iterable[int]) -> str:
    """Encode integer values as a semicolon-separated string."""
    return ";".join(str(int(value)) for value in sorted(set(values)))


def encode_dictionary(values: Dict[int, int]) -> str:
    """Encode an integer dictionary as compact JSON."""
    clean = {
        str(int(key)): int(value)
        for key, value in sorted(values.items())
    }

    return json.dumps(clean, separators=(",", ":"))


def atom_output_id(atom) -> int:
    """Return LAMMPS atom ID when available; otherwise atom index."""
    if STORE_ATOM_IDS:
        try:
            return int(atom.id)
        except (AttributeError, TypeError, ValueError):
            pass

    return int(atom.index)


def determine_coordination_state(
    n_peo_oxygen: int,
    n_tfsi_molecules: int,
) -> str:
    """Assign P, PT, T, or F state."""

    if n_peo_oxygen > 0 and n_tfsi_molecules == 0:
        return "P"

    if n_peo_oxygen > 0 and n_tfsi_molecules > 0:
        return "PT"

    if n_peo_oxygen == 0 and n_tfsi_molecules > 0:
        return "T"

    return "F"


def build_peo_local_index_lookup(peo_oxygens):
    """
    Map each PEO oxygen atom index to its local EO number.

    EO numbering begins at 1 independently for each PEO chain.
    """

    lookup = {}

    unique_resids = sorted(
        set(int(resid) for resid in peo_oxygens.resids)
    )

    for resid in unique_resids:
        chain_oxygens = peo_oxygens[
            peo_oxygens.resids == resid
        ]

        sorted_atoms = sorted(
            chain_oxygens,
            key=lambda atom: atom_output_id(atom),
        )

        for local_eo_index, atom in enumerate(
            sorted_atoms,
            start=1,
        ):
            lookup[int(atom.index)] = int(local_eo_index)

    return lookup


def minimum_eo_separation(
    eo_indices_a: Set[int],
    eo_indices_b: Set[int],
):
    """
    Return minimum absolute EO-index separation between two metals.

    Both metals must coordinate the same PEO chain.
    """

    if not eo_indices_a or not eo_indices_b:
        return None

    return min(
        abs(int(eo_a) - int(eo_b))
        for eo_a in eo_indices_a
        for eo_b in eo_indices_b
    )


def write_csv_rows(
    output_file: Path,
    rows: List[dict],
    fieldnames: List[str],
    write_header: bool,
    compressed: bool,
):
    """Append rows to CSV or CSV.GZ."""

    if not rows:
        return

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if compressed:
        mode = "wt" if write_header else "at"

        with gzip.open(
            output_file,
            mode,
            newline="",
            encoding="utf-8",
        ) as handle:

            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
            )

            if write_header:
                writer.writeheader()

            writer.writerows(rows)

    else:
        mode = "w" if write_header else "a"

        with output_file.open(
            mode,
            newline="",
            encoding="utf-8",
        ) as handle:

            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
            )

            if write_header:
                writer.writeheader()

            writer.writerows(rows)


# ============================================================
# OUTPUT COLUMNS
# ============================================================

MASTER_COLUMNS = [
    "system",
    "metal_species",
    "composition",
    "frame",
    "time_ps",

    "metal_local_index",
    "metal_atom_id",
    "metal_resid",

    "n_peo_oxygen",
    "n_peo_chains",
    "peo_oxygen_atom_ids",
    "peo_chain_resids",
    "peo_local_eo_indices",
    "peo_chain_eo_mapping",

    "n_tfsi_oxygen",
    "n_tfsi_molecules",
    "tfsi_oxygen_atom_ids",
    "tfsi_resids",
    "tfsi_denticity",

    "total_oxygen_cn",
    "coordination_state",

    # TFSI bridging
    "is_tfsi_bridged",
    "n_bridging_tfsi",
    "bridging_tfsi_resids",

    # PEO shared-chain and bridging
    "is_shared_peo_chain",
    "n_shared_peo_chains",
    "shared_peo_chain_resids",

    "is_local_peo_bridged",
    "n_local_peo_bridge_chains",
    "local_peo_bridge_chain_resids",

    "is_remote_peo_shared",
    "n_remote_peo_shared_chains",
    "remote_peo_shared_chain_resids",

    # Metal-TFSI cluster
    "cluster_id",
    "cluster_n_metals",
    "cluster_n_tfsi",
    "cluster_total_nodes",
    "is_multi_metal_cluster",
]


FRAME_COLUMNS = [
    "system",
    "metal_species",
    "composition",
    "frame",
    "time_ps",
    "n_metals",

    # State populations
    "n_P",
    "n_PT",
    "n_T",
    "n_F",

    "frac_P",
    "frac_PT",
    "frac_T",
    "frac_F",

    # Coordination
    "mean_peo_oxygen_cn",
    "mean_peo_chain_cn",
    "mean_tfsi_oxygen_cn",
    "mean_tfsi_molecular_cn",
    "mean_total_oxygen_cn",

    # TFSI bridging
    "n_coordinated_tfsi",
    "n_bridging_tfsi",
    "fraction_coordinated_tfsi_bridging",
    "fraction_all_tfsi_bridging",
    "fraction_metals_tfsi_bridged",

    # PEO chain sharing and bridging
    "n_peo_chains_coordinating_metals",
    "n_peo_chains_shared_by_metals",
    "n_peo_chains_local_bridging",
    "n_peo_chains_remote_shared",

    "fraction_coordinating_peo_chains_shared",
    "fraction_all_peo_chains_shared",

    "fraction_coordinating_peo_chains_local_bridging",
    "fraction_all_peo_chains_local_bridging",

    "fraction_metals_shared_peo_chain",
    "fraction_metals_local_peo_bridged",
    "fraction_metals_remote_peo_shared",

    "mean_metals_per_coordinating_peo_chain",
    "max_metals_on_one_peo_chain",

    # Combined bridge populations
    "fraction_metals_any_bridge",
    "fraction_metals_both_peo_tfsi_bridged",
    "fraction_metals_peo_only_bridged",
    "fraction_metals_tfsi_only_bridged",

    # Metal-TFSI clusters
    "n_metal_containing_clusters",
    "mean_cluster_metals_number_average",
    "mean_cluster_metals_weight_average",
    "largest_cluster_n_metals",
    "largest_cluster_metal_fraction",
    "fraction_metals_in_multi_metal_clusters",
    "fraction_single_metal_clusters",
]


TFSI_BRIDGING_COLUMNS = [
    "system",
    "metal_species",
    "composition",
    "frame",
    "time_ps",
    "tfsi_resid",
    "n_coordinating_metals",
    "metal_local_indices",
    "metal_atom_ids",
    "is_bridging",
]


PEO_BRIDGING_COLUMNS = [
    "system",
    "metal_species",
    "composition",
    "frame",
    "time_ps",

    "peo_chain_resid",
    "n_coordinating_metals",
    "metal_local_indices",
    "metal_atom_ids",

    "is_shared_chain",
    "has_local_bridge",
    "has_remote_shared_pair",

    "n_metal_pairs",
    "n_local_bridge_pairs",
    "n_remote_shared_pairs",

    "minimum_pair_eo_separation",
    "pair_eo_separations",
]


CLUSTER_COLUMNS = [
    "system",
    "metal_species",
    "composition",
    "frame",
    "time_ps",

    "cluster_id",
    "n_metals",
    "n_tfsi",
    "total_nodes",

    "metal_local_indices",
    "metal_atom_ids",
    "tfsi_resids",

    "is_multi_metal_cluster",
]


# ============================================================
# CORE ANALYSIS
# ============================================================

def analyze_system(system_name: str, config: dict):
    """Build complete coordination database for one system."""

    start_clock = time.time()

    output_dir = OUTPUT_ROOT / system_name
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    master_file = (
        output_dir
        / f"master_coordination_{system_name}.csv.gz"
    )

    frame_file = (
        output_dir
        / f"frame_summary_{system_name}.csv"
    )

    tfsi_bridging_file = (
        output_dir
        / f"tfsi_bridging_{system_name}.csv.gz"
    )

    peo_bridging_file = (
        output_dir
        / f"peo_chain_bridging_{system_name}.csv.gz"
    )

    cluster_file = (
        output_dir
        / f"cluster_summary_{system_name}.csv.gz"
    )

    metadata_file = (
        output_dir
        / f"metadata_{system_name}.json"
    )

    # Prevent accidental appending to old results.
    for path in [
        master_file,
        frame_file,
        tfsi_bridging_file,
        peo_bridging_file,
        cluster_file,
        metadata_file,
    ]:
        if path.exists():
            path.unlink()

    print("\n" + "=" * 78)
    print(f"Building coordination database: {system_name}")
    print("=" * 78)

    topology = Path(config["topology"])
    trajectory = Path(config["trajectory"])

    if not topology.exists():
        raise FileNotFoundError(
            f"Topology file not found: {topology}"
        )

    if not trajectory.exists():
        raise FileNotFoundError(
            f"Trajectory file not found: {trajectory}"
        )

    universe = mda.Universe(
        str(topology),
        str(trajectory),
        format="LAMMPSDUMP",
    )

    metals = universe.select_atoms(
        config["metal_selection"]
    )

    peo_oxygens = universe.select_atoms(
        config["peo_oxygen_selection"]
    )

    tfsi_oxygens = universe.select_atoms(
        config["tfsi_oxygen_selection"]
    )

    if len(metals) == 0:
        raise ValueError(
            "Metal selection returned zero atoms."
        )

    if len(peo_oxygens) == 0:
        raise ValueError(
            "PEO oxygen selection returned zero atoms."
        )

    if len(tfsi_oxygens) == 0:
        raise ValueError(
            "TFSI oxygen selection returned zero atoms."
        )

    all_peo_chain_resids = sorted(
        set(int(value) for value in peo_oxygens.resids)
    )

    all_tfsi_resids = sorted(
        set(int(value) for value in tfsi_oxygens.resids)
    )

    n_total_peo_chains = len(all_peo_chain_resids)
    n_total_tfsi = len(all_tfsi_resids)

    print(f"Metal atoms             : {len(metals)}")
    print(f"PEO oxygen atoms        : {len(peo_oxygens)}")
    print(f"PEO chains              : {n_total_peo_chains}")
    print(f"TFSI oxygen atoms       : {len(tfsi_oxygens)}")
    print(f"TFSI molecules          : {n_total_tfsi}")
    print(
        f"PEO cutoff              : "
        f"{config['peo_cutoff_A']:.3f} Å"
    )
    print(
        f"TFSI cutoff             : "
        f"{config['tfsi_cutoff_A']:.3f} Å"
    )
    print(
        f"Local PEO bridge limit  : "
        f"ΔEO <= {LOCAL_PEO_BRIDGE_MAX_EO}"
    )
    print(
        f"Frame interval          : "
        f"{FRAME_INTERVAL_PS:.3f} ps"
    )

    peo_local_index_lookup = (
        build_peo_local_index_lookup(peo_oxygens)
    )

    master_buffer = []
    frame_buffer = []
    tfsi_bridging_buffer = []
    peo_bridging_buffer = []
    cluster_buffer = []

    master_header_needed = True
    frame_header_needed = True
    tfsi_header_needed = True
    peo_header_needed = True
    cluster_header_needed = True

    processed_frames = 0

    trajectory_slice = universe.trajectory[
        START_FRAME:STOP_FRAME:STRIDE
    ]

    for ts in trajectory_slice:

        frame = int(ts.frame)
        time_ps = frame * FRAME_INTERVAL_PS
        box = ts.dimensions

        # ====================================================
        # 1. METAL-PEO CONTACTS
        # ====================================================

        peo_pairs = capped_distance(
            metals.positions,
            peo_oxygens.positions,
            max_cutoff=float(config["peo_cutoff_A"]),
            box=box,
            return_distances=False,
        )

        metal_to_peo_atom_indices = defaultdict(set)
        metal_to_peo_chain_resids = defaultdict(set)
        metal_to_peo_local_indices = defaultdict(set)

        # Nested mapping:
        # metal -> PEO chain -> local EO indices
        metal_chain_to_eo_indices = defaultdict(
            lambda: defaultdict(set)
        )

        # Reverse map:
        # PEO chain -> metals
        peo_chain_to_metals = defaultdict(set)

        for metal_local_index, peo_group_index in peo_pairs:

            metal_local_index = int(
                metal_local_index
            )

            peo_atom = peo_oxygens[
                int(peo_group_index)
            ]

            peo_atom_index = int(peo_atom.index)
            chain_resid = int(peo_atom.resid)

            local_eo_index = int(
                peo_local_index_lookup[peo_atom_index]
            )

            metal_to_peo_atom_indices[
                metal_local_index
            ].add(peo_atom_index)

            metal_to_peo_chain_resids[
                metal_local_index
            ].add(chain_resid)

            metal_to_peo_local_indices[
                metal_local_index
            ].add(local_eo_index)

            metal_chain_to_eo_indices[
                metal_local_index
            ][chain_resid].add(local_eo_index)

            peo_chain_to_metals[
                chain_resid
            ].add(metal_local_index)

        # ====================================================
        # 2. PEO SHARED-CHAIN AND LOCAL-BRIDGE ANALYSIS
        # ====================================================

        shared_peo_chain_resids = set()
        local_bridging_peo_chain_resids = set()
        remote_shared_peo_chain_resids = set()

        metals_in_shared_peo_chains = set()
        metals_in_local_peo_bridges = set()
        metals_in_remote_peo_shared_chains = set()

        # Per-metal record of which chains have which motif.
        metal_to_shared_peo_chains = defaultdict(set)
        metal_to_local_peo_bridge_chains = defaultdict(set)
        metal_to_remote_peo_shared_chains = defaultdict(set)

        for chain_resid in all_peo_chain_resids:

            metal_set = peo_chain_to_metals.get(
                chain_resid,
                set(),
            )

            sorted_metals = sorted(metal_set)

            is_shared_chain = len(sorted_metals) >= 2

            pair_separations = []
            local_pairs = []
            remote_pairs = []

            if is_shared_chain:

                shared_peo_chain_resids.add(
                    chain_resid
                )

                metals_in_shared_peo_chains.update(
                    sorted_metals
                )

                for metal_index in sorted_metals:
                    metal_to_shared_peo_chains[
                        metal_index
                    ].add(chain_resid)

                for metal_a, metal_b in combinations(
                    sorted_metals,
                    2,
                ):

                    eo_indices_a = (
                        metal_chain_to_eo_indices[
                            metal_a
                        ][chain_resid]
                    )

                    eo_indices_b = (
                        metal_chain_to_eo_indices[
                            metal_b
                        ][chain_resid]
                    )

                    separation = minimum_eo_separation(
                        eo_indices_a,
                        eo_indices_b,
                    )

                    if separation is None:
                        continue

                    pair_separations.append(
                        int(separation)
                    )

                    if (
                        separation
                        <= LOCAL_PEO_BRIDGE_MAX_EO
                    ):
                        local_pairs.append(
                            (metal_a, metal_b, separation)
                        )

                        metals_in_local_peo_bridges.update(
                            [metal_a, metal_b]
                        )

                        metal_to_local_peo_bridge_chains[
                            metal_a
                        ].add(chain_resid)

                        metal_to_local_peo_bridge_chains[
                            metal_b
                        ].add(chain_resid)

                    else:
                        remote_pairs.append(
                            (metal_a, metal_b, separation)
                        )

                        metals_in_remote_peo_shared_chains.update(
                            [metal_a, metal_b]
                        )

                        metal_to_remote_peo_shared_chains[
                            metal_a
                        ].add(chain_resid)

                        metal_to_remote_peo_shared_chains[
                            metal_b
                        ].add(chain_resid)

                if local_pairs:
                    local_bridging_peo_chain_resids.add(
                        chain_resid
                    )

                if remote_pairs:
                    remote_shared_peo_chain_resids.add(
                        chain_resid
                    )

            metal_atom_ids = [
                atom_output_id(metals[index])
                for index in sorted_metals
            ]

            peo_bridging_buffer.append({
                "system": system_name,
                "metal_species": config["metal"],
                "composition": config["composition"],
                "frame": frame,
                "time_ps": time_ps,

                "peo_chain_resid": chain_resid,
                "n_coordinating_metals": len(
                    sorted_metals
                ),

                "metal_local_indices":
                    encode_integer_list(
                        sorted_metals
                    ),

                "metal_atom_ids":
                    encode_integer_list(
                        metal_atom_ids
                    ),

                "is_shared_chain": int(
                    is_shared_chain
                ),

                "has_local_bridge": int(
                    len(local_pairs) > 0
                ),

                "has_remote_shared_pair": int(
                    len(remote_pairs) > 0
                ),

                "n_metal_pairs": len(
                    pair_separations
                ),

                "n_local_bridge_pairs": len(
                    local_pairs
                ),

                "n_remote_shared_pairs": len(
                    remote_pairs
                ),

                "minimum_pair_eo_separation": (
                    min(pair_separations)
                    if pair_separations
                    else ""
                ),

                "pair_eo_separations":
                    encode_integer_list(
                        pair_separations
                    ),
            })

        # ====================================================
        # 3. METAL-TFSI CONTACTS AND DENTICITY
        # ====================================================

        tfsi_pairs = capped_distance(
            metals.positions,
            tfsi_oxygens.positions,
            max_cutoff=float(config["tfsi_cutoff_A"]),
            box=box,
            return_distances=False,
        )

        metal_to_tfsi_atom_indices = defaultdict(set)
        metal_to_tfsi_resids = defaultdict(set)

        metal_tfsi_denticity = defaultdict(
            lambda: defaultdict(int)
        )

        tfsi_to_metals = defaultdict(set)

        for metal_local_index, tfsi_group_index in tfsi_pairs:

            metal_local_index = int(
                metal_local_index
            )

            tfsi_atom = tfsi_oxygens[
                int(tfsi_group_index)
            ]

            tfsi_atom_index = int(tfsi_atom.index)
            tfsi_resid = int(tfsi_atom.resid)

            metal_to_tfsi_atom_indices[
                metal_local_index
            ].add(tfsi_atom_index)

            metal_to_tfsi_resids[
                metal_local_index
            ].add(tfsi_resid)

            metal_tfsi_denticity[
                metal_local_index
            ][tfsi_resid] += 1

            tfsi_to_metals[
                tfsi_resid
            ].add(metal_local_index)

        # ====================================================
        # 4. TFSI-MEDIATED BRIDGING
        # ====================================================

        bridging_tfsi_resids = {
            tfsi_resid
            for tfsi_resid, metal_set
            in tfsi_to_metals.items()
            if len(metal_set) >= 2
        }

        metals_in_tfsi_bridges = set()

        for tfsi_resid, metal_set in (
            tfsi_to_metals.items()
        ):

            is_bridging = len(metal_set) >= 2

            if is_bridging:
                metals_in_tfsi_bridges.update(
                    metal_set
                )

            metal_atom_ids = [
                atom_output_id(metals[index])
                for index in sorted(metal_set)
            ]

            tfsi_bridging_buffer.append({
                "system": system_name,
                "metal_species": config["metal"],
                "composition": config["composition"],
                "frame": frame,
                "time_ps": time_ps,                "tfsi_resid": tfsi_resid,

                "n_coordinating_metals":
                    len(metal_set),

                "metal_local_indices":
                    encode_integer_list(
                        metal_set
                    ),

                "metal_atom_ids":
                    encode_integer_list(
                        metal_atom_ids
                    ),

                "is_bridging":
                    int(is_bridging),
            })

        # ====================================================
        # 5. METAL-TFSI CLUSTER ANALYSIS
        # ====================================================

        union_find = UnionFind()

        # Include every metal, even without a TFSI contact.
        for metal_local_index in range(len(metals)):
            union_find.add(
                f"M:{metal_local_index}"
            )

        # Only coordinated TFSI molecules enter the graph.
        for tfsi_resid in tfsi_to_metals:
            union_find.add(
                f"T:{tfsi_resid}"
            )

        for (
            metal_local_index,
            tfsi_resid_set,
        ) in metal_to_tfsi_resids.items():

            for tfsi_resid in tfsi_resid_set:

                union_find.union(
                    f"M:{metal_local_index}",
                    f"T:{tfsi_resid}",
                )

        components = defaultdict(list)

        for node in union_find.parent:
            root = union_find.find(node)
            components[root].append(node)

        component_information = []

        for nodes in components.values():

            metal_members = sorted(
                int(node.split(":")[1])
                for node in nodes
                if node.startswith("M:")
            )

            tfsi_members = sorted(
                int(node.split(":")[1])
                for node in nodes
                if node.startswith("T:")
            )

            if metal_members:
                component_information.append(
                    (metal_members, tfsi_members)
                )

        component_information.sort(
            key=lambda item: (
                item[0][0],
                len(item[0]),
                len(item[1]),
            )
        )

        metal_cluster_lookup = {}
        frame_metal_cluster_sizes = []

        for cluster_number, (
            metal_members,
            tfsi_members,
        ) in enumerate(
            component_information,
            start=1,
        ):

            cluster_id = (
                f"{frame}_{cluster_number}"
            )

            n_metals_cluster = len(
                metal_members
            )

            n_tfsi_cluster = len(
                tfsi_members
            )

            total_nodes = (
                n_metals_cluster
                + n_tfsi_cluster
            )

            frame_metal_cluster_sizes.append(
                n_metals_cluster
            )

            metal_atom_ids = [
                atom_output_id(metals[index])
                for index in metal_members
            ]

            cluster_buffer.append({
                "system": system_name,
                "metal_species": config["metal"],
                "composition": config["composition"],
                "frame": frame,
                "time_ps": time_ps,

                "cluster_id": cluster_id,
                "n_metals": n_metals_cluster,
                "n_tfsi": n_tfsi_cluster,
                "total_nodes": total_nodes,

                "metal_local_indices":
                    encode_integer_list(
                        metal_members
                    ),

                "metal_atom_ids":
                    encode_integer_list(
                        metal_atom_ids
                    ),

                "tfsi_resids":
                    encode_integer_list(
                        tfsi_members
                    ),

                "is_multi_metal_cluster":
                    int(n_metals_cluster >= 2),
            })

            for metal_local_index in metal_members:

                metal_cluster_lookup[
                    metal_local_index
                ] = {
                    "cluster_id": cluster_id,
                    "n_metals": n_metals_cluster,
                    "n_tfsi": n_tfsi_cluster,
                    "total_nodes": total_nodes,
                }

        # ====================================================
        # 6. PER-METAL MASTER DATABASE
        # ====================================================

        state_counter = Counter()

        frame_peo_cn = []
        frame_peo_chain_cn = []
        frame_tfsi_o_cn = []
        frame_tfsi_molecular_cn = []
        frame_total_cn = []

        metals_in_multi_metal_clusters = 0

        metals_any_bridge = set(
            metals_in_local_peo_bridges
        ).union(
            metals_in_tfsi_bridges
        )

        metals_both_bridge_types = set(
            metals_in_local_peo_bridges
        ).intersection(
            metals_in_tfsi_bridges
        )

        metals_peo_only_bridge = set(
            metals_in_local_peo_bridges
        ).difference(
            metals_in_tfsi_bridges
        )

        metals_tfsi_only_bridge = set(
            metals_in_tfsi_bridges
        ).difference(
            metals_in_local_peo_bridges
        )

        for metal_local_index, metal_atom in enumerate(
            metals
        ):

            peo_atom_indices = (
                metal_to_peo_atom_indices[
                    metal_local_index
                ]
            )

            peo_chain_resids = (
                metal_to_peo_chain_resids[
                    metal_local_index
                ]
            )

            local_eo_indices = (
                metal_to_peo_local_indices[
                    metal_local_index
                ]
            )

            tfsi_atom_indices = (
                metal_to_tfsi_atom_indices[
                    metal_local_index
                ]
            )

            tfsi_resids = (
                metal_to_tfsi_resids[
                    metal_local_index
                ]
            )

            denticity = dict(
                metal_tfsi_denticity[
                    metal_local_index
                ]
            )

            n_peo_oxygen = len(
                peo_atom_indices
            )

            n_peo_chains = len(
                peo_chain_resids
            )

            n_tfsi_oxygen = len(
                tfsi_atom_indices
            )

            n_tfsi_molecules = len(
                tfsi_resids
            )

            total_oxygen_cn = (
                n_peo_oxygen
                + n_tfsi_oxygen
            )

            state = determine_coordination_state(
                n_peo_oxygen,
                n_tfsi_molecules,
            )

            state_counter[state] += 1

            # TFSI bridges for this metal.
            metal_bridging_tfsi = (
                tfsi_resids.intersection(
                    bridging_tfsi_resids
                )
            )

            # PEO shared/local/remote chains.
            shared_chains = (
                metal_to_shared_peo_chains[
                    metal_local_index
                ]
            )

            local_bridge_chains = (
                metal_to_local_peo_bridge_chains[
                    metal_local_index
                ]
            )

            remote_shared_chains = (
                metal_to_remote_peo_shared_chains[
                    metal_local_index
                ]
            )

            cluster_data = metal_cluster_lookup[
                metal_local_index
            ]

            is_multi_metal_cluster = (
                cluster_data["n_metals"] >= 2
            )

            if is_multi_metal_cluster:
                metals_in_multi_metal_clusters += 1

            peo_atom_ids = [
                atom_output_id(
                    universe.atoms[index]
                )
                for index in peo_atom_indices
            ]

            tfsi_atom_ids = [
                atom_output_id(
                    universe.atoms[index]
                )
                for index in tfsi_atom_indices
            ]

            # Store EO indices separately for each chain.
            peo_chain_eo_mapping = {
                int(chain_resid): sorted(
                    int(value)
                    for value in eo_indices
                )
                for (
                    chain_resid,
                    eo_indices,
                ) in metal_chain_to_eo_indices[
                    metal_local_index
                ].items()
            }

            master_buffer.append({
                "system": system_name,
                "metal_species": config["metal"],
                "composition": config["composition"],
                "frame": frame,
                "time_ps": time_ps,

                "metal_local_index":
                    metal_local_index,

                "metal_atom_id":
                    atom_output_id(metal_atom),

                "metal_resid":
                    int(metal_atom.resid),

                "n_peo_oxygen":
                    n_peo_oxygen,

                "n_peo_chains":
                    n_peo_chains,

                "peo_oxygen_atom_ids":
                    encode_integer_list(
                        peo_atom_ids
                    ),

                "peo_chain_resids":
                    encode_integer_list(
                        peo_chain_resids
                    ),

                "peo_local_eo_indices":
                    encode_integer_list(
                        local_eo_indices
                    ),

                "peo_chain_eo_mapping":
                    json.dumps(
                        peo_chain_eo_mapping,
                        separators=(",", ":"),
                    ),

                "n_tfsi_oxygen":
                    n_tfsi_oxygen,

                "n_tfsi_molecules":
                    n_tfsi_molecules,

                "tfsi_oxygen_atom_ids":
                    encode_integer_list(
                        tfsi_atom_ids
                    ),

                "tfsi_resids":
                    encode_integer_list(
                        tfsi_resids
                    ),

                "tfsi_denticity":
                    encode_dictionary(
                        denticity
                    ),

                "total_oxygen_cn":
                    total_oxygen_cn,

                "coordination_state":
                    state,

                "is_tfsi_bridged": int(
                    len(metal_bridging_tfsi) > 0
                ),

                "n_bridging_tfsi": len(
                    metal_bridging_tfsi
                ),

                "bridging_tfsi_resids":
                    encode_integer_list(
                        metal_bridging_tfsi
                    ),

                "is_shared_peo_chain": int(
                    len(shared_chains) > 0
                ),

                "n_shared_peo_chains": len(
                    shared_chains
                ),

                "shared_peo_chain_resids":
                    encode_integer_list(
                        shared_chains
                    ),

                "is_local_peo_bridged": int(
                    len(local_bridge_chains) > 0
                ),

                "n_local_peo_bridge_chains": len(
                    local_bridge_chains
                ),

                "local_peo_bridge_chain_resids":
                    encode_integer_list(
                        local_bridge_chains
                    ),

                "is_remote_peo_shared": int(
                    len(remote_shared_chains) > 0
                ),

                "n_remote_peo_shared_chains": len(
                    remote_shared_chains
                ),

                "remote_peo_shared_chain_resids":
                    encode_integer_list(
                        remote_shared_chains
                    ),

                "cluster_id":
                    cluster_data["cluster_id"],

                "cluster_n_metals":
                    cluster_data["n_metals"],

                "cluster_n_tfsi":
                    cluster_data["n_tfsi"],

                "cluster_total_nodes":
                    cluster_data["total_nodes"],

                "is_multi_metal_cluster":
                    int(is_multi_metal_cluster),
            })

            frame_peo_cn.append(
                n_peo_oxygen
            )

            frame_peo_chain_cn.append(
                n_peo_chains
            )

            frame_tfsi_o_cn.append(
                n_tfsi_oxygen
            )

            frame_tfsi_molecular_cn.append(
                n_tfsi_molecules
            )

            frame_total_cn.append(
                total_oxygen_cn
            )

        # ====================================================
        # 7. FRAME-LEVEL SUMMARY
        # ====================================================

        n_metals = len(metals)

        n_coordinated_tfsi = len(
            tfsi_to_metals
        )

        n_bridging_tfsi = len(
            bridging_tfsi_resids
        )

        n_peo_chains_coordinating_metals = len(
            peo_chain_to_metals
        )

        n_peo_chains_shared_by_metals = len(
            shared_peo_chain_resids
        )

        n_peo_chains_local_bridging = len(
            local_bridging_peo_chain_resids
        )

        n_peo_chains_remote_shared = len(
            remote_shared_peo_chain_resids
        )

        metals_per_coordinating_chain = [
            len(metal_set)
            for metal_set
            in peo_chain_to_metals.values()
        ]

        mean_metals_per_coordinating_chain = (
            float(
                np.mean(
                    metals_per_coordinating_chain
                )
            )
            if metals_per_coordinating_chain
            else 0.0
        )

        max_metals_on_one_chain = (
            int(
                max(
                    metals_per_coordinating_chain
                )
            )
            if metals_per_coordinating_chain
            else 0
        )

        number_average_cluster_size = float(
            np.mean(
                frame_metal_cluster_sizes
            )
        )

        weight_average_cluster_size = float(
            np.sum(
                np.square(
                    frame_metal_cluster_sizes
                )
            )
            / np.sum(
                frame_metal_cluster_sizes
            )
        )

        largest_cluster_n_metals = int(
            max(
                frame_metal_cluster_sizes
            )
        )

        n_single_metal_clusters = sum(
            cluster_size == 1
            for cluster_size
            in frame_metal_cluster_sizes
        )

        frame_buffer.append({
            "system": system_name,
            "metal_species": config["metal"],
            "composition": config["composition"],
            "frame": frame,
            "time_ps": time_ps,
            "n_metals": n_metals,

            "n_P": state_counter["P"],
            "n_PT": state_counter["PT"],
            "n_T": state_counter["T"],
            "n_F": state_counter["F"],

            "frac_P":
                state_counter["P"] / n_metals,

            "frac_PT":
                state_counter["PT"] / n_metals,

            "frac_T":
                state_counter["T"] / n_metals,

            "frac_F":
                state_counter["F"] / n_metals,

            "mean_peo_oxygen_cn":
                float(np.mean(frame_peo_cn)),

            "mean_peo_chain_cn":
                float(np.mean(frame_peo_chain_cn)),

            "mean_tfsi_oxygen_cn":
                float(np.mean(frame_tfsi_o_cn)),

            "mean_tfsi_molecular_cn":
                float(
                    np.mean(
                        frame_tfsi_molecular_cn
                    )
                ),

            "mean_total_oxygen_cn":
                float(np.mean(frame_total_cn)),

            "n_coordinated_tfsi":
                n_coordinated_tfsi,

            "n_bridging_tfsi":
                n_bridging_tfsi,

            "fraction_coordinated_tfsi_bridging": (
                n_bridging_tfsi
                / n_coordinated_tfsi
                if n_coordinated_tfsi > 0
                else 0.0
            ),

            "fraction_all_tfsi_bridging": (
                n_bridging_tfsi
                / n_total_tfsi
                if n_total_tfsi > 0
                else 0.0
            ),

            "fraction_metals_tfsi_bridged": (
                len(metals_in_tfsi_bridges)
                / n_metals
            ),

            "n_peo_chains_coordinating_metals":
                n_peo_chains_coordinating_metals,

            "n_peo_chains_shared_by_metals":
                n_peo_chains_shared_by_metals,

            "n_peo_chains_local_bridging":
                n_peo_chains_local_bridging,

            "n_peo_chains_remote_shared":
                n_peo_chains_remote_shared,

            "fraction_coordinating_peo_chains_shared": (
                n_peo_chains_shared_by_metals
                / n_peo_chains_coordinating_metals
                if n_peo_chains_coordinating_metals > 0
                else 0.0
            ),

            "fraction_all_peo_chains_shared": (
                n_peo_chains_shared_by_metals
                / n_total_peo_chains
                if n_total_peo_chains > 0
                else 0.0
            ),

            "fraction_coordinating_peo_chains_local_bridging": (
                n_peo_chains_local_bridging
                / n_peo_chains_coordinating_metals
                if n_peo_chains_coordinating_metals > 0
                else 0.0
            ),

            "fraction_all_peo_chains_local_bridging": (
                n_peo_chains_local_bridging
                / n_total_peo_chains
                if n_total_peo_chains > 0
                else 0.0
            ),

            "fraction_metals_shared_peo_chain": (
                len(metals_in_shared_peo_chains)
                / n_metals
            ),

            "fraction_metals_local_peo_bridged": (
                len(metals_in_local_peo_bridges)
                / n_metals
            ),

            "fraction_metals_remote_peo_shared": (
                len(
                    metals_in_remote_peo_shared_chains
                )
                / n_metals
            ),

            "mean_metals_per_coordinating_peo_chain":
                mean_metals_per_coordinating_chain,

            "max_metals_on_one_peo_chain":
                max_metals_on_one_chain,

            "fraction_metals_any_bridge": (
                len(metals_any_bridge)
                / n_metals
            ),

            "fraction_metals_both_peo_tfsi_bridged": (
                len(metals_both_bridge_types)
                / n_metals
            ),

            "fraction_metals_peo_only_bridged": (
                len(metals_peo_only_bridge)
                / n_metals
            ),

            "fraction_metals_tfsi_only_bridged": (
                len(metals_tfsi_only_bridge)
                / n_metals
            ),

            "n_metal_containing_clusters":
                len(frame_metal_cluster_sizes),

            "mean_cluster_metals_number_average":
                number_average_cluster_size,

            "mean_cluster_metals_weight_average":
                weight_average_cluster_size,

            "largest_cluster_n_metals":
                largest_cluster_n_metals,

            "largest_cluster_metal_fraction": (
                largest_cluster_n_metals
                / n_metals
            ),

            "fraction_metals_in_multi_metal_clusters": (
                metals_in_multi_metal_clusters
                / n_metals
            ),

            "fraction_single_metal_clusters": (
                n_single_metal_clusters
                / len(frame_metal_cluster_sizes)
            ),
        })

        processed_frames += 1

        # ====================================================
        # 8. PERIODIC DISK WRITE
        # ====================================================

        if (
            processed_frames
            % WRITE_EVERY_N_FRAMES
            == 0
        ):

            write_csv_rows(
                master_file,
                master_buffer,
                MASTER_COLUMNS,
                master_header_needed,
                compressed=True,
            )
            master_header_needed = False
            master_buffer.clear()

            write_csv_rows(
                frame_file,
                frame_buffer,
                FRAME_COLUMNS,
                frame_header_needed,
                compressed=False,
            )
            frame_header_needed = False
            frame_buffer.clear()

            write_csv_rows(
                tfsi_bridging_file,
                tfsi_bridging_buffer,
                TFSI_BRIDGING_COLUMNS,
                tfsi_header_needed,
                compressed=True,
            )
            tfsi_header_needed = False
            tfsi_bridging_buffer.clear()

            write_csv_rows(
                peo_bridging_file,
                peo_bridging_buffer,
                PEO_BRIDGING_COLUMNS,
                peo_header_needed,
                compressed=True,
            )
            peo_header_needed = False
            peo_bridging_buffer.clear()

            write_csv_rows(
                cluster_file,
                cluster_buffer,
                CLUSTER_COLUMNS,
                cluster_header_needed,
                compressed=True,
            )
            cluster_header_needed = False
            cluster_buffer.clear()

            elapsed_minutes = (
                time.time() - start_clock
            ) / 60.0

            print(
                f"Processed {processed_frames:6d} frames "
                f"| trajectory frame {frame:6d} "
                f"| elapsed {elapsed_minutes:8.2f} min"
            )

    # ========================================================
    # 9. WRITE REMAINING ROWS
    # ========================================================

    write_csv_rows(
        master_file,
        master_buffer,
        MASTER_COLUMNS,
        master_header_needed,
        compressed=True,
    )

    write_csv_rows(
        frame_file,
        frame_buffer,
        FRAME_COLUMNS,
        frame_header_needed,
        compressed=False,
    )

    write_csv_rows(
        tfsi_bridging_file,
        tfsi_bridging_buffer,
        TFSI_BRIDGING_COLUMNS,
        tfsi_header_needed,
        compressed=True,
    )

    write_csv_rows(
        peo_bridging_file,
        peo_bridging_buffer,
        PEO_BRIDGING_COLUMNS,
        peo_header_needed,
        compressed=True,
    )

    write_csv_rows(
        cluster_file,
        cluster_buffer,
        CLUSTER_COLUMNS,
        cluster_header_needed,
        compressed=True,
    )

    elapsed_seconds = time.time() - start_clock

    metadata = {
        "system": system_name,
        "metal_species": config["metal"],
        "composition": config["composition"],

        "topology": str(topology),
        "trajectory": str(trajectory),

        "metal_selection":
            config["metal_selection"],

        "peo_oxygen_selection":
            config["peo_oxygen_selection"],

        "tfsi_oxygen_selection":
            config["tfsi_oxygen_selection"],

        "peo_cutoff_A":
            config["peo_cutoff_A"],

        "tfsi_cutoff_A":
            config["tfsi_cutoff_A"],

        "local_peo_bridge_max_eo":
            LOCAL_PEO_BRIDGE_MAX_EO,

        "frame_interval_ps":
            FRAME_INTERVAL_PS,

        "start_frame":
            START_FRAME,

        "stop_frame":
            STOP_FRAME,

        "stride":
            STRIDE,

        "processed_frames":
            processed_frames,

        "n_metals":
            len(metals),

        "n_peo_oxygens":
            len(peo_oxygens),

        "n_peo_chains":
            n_total_peo_chains,

        "n_tfsi_oxygens":
            len(tfsi_oxygens),

        "n_tfsi_molecules":
            n_total_tfsi,

        "runtime_seconds":
            elapsed_seconds,
    }

    with metadata_file.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
        )

    print("\nCompleted successfully.")
    print(f"Processed frames : {processed_frames}")
    print(
        f"Runtime          : "
        f"{elapsed_seconds / 60.0:.2f} min"
    )
    print(
        f"Output directory : "
        f"{output_dir.resolve()}"
    )


# ============================================================
# MAIN DRIVER
# ============================================================

def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print(
        "polymer-electrolyte metal-ion "
        "Coordination and Bridging Database"
    )
    print("=" * 78)

    for system_name, configuration in SYSTEMS.items():
        analyze_system(
            system_name,
            configuration,
        )

    print("\n" + "=" * 78)
    print("ALL SYSTEMS COMPLETED")
    print("=" * 78)


if __name__ == "__main__":
    main()
    
    
    
    