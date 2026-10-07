#!/usr/bin/env bash
set -euo pipefail

# Install once from the repository root:
#   python -m pip install -e .
#
# Adapt topology/trajectory selections and cutoffs to your system before running.

python scripts/01_build_coordination_database.py
python scripts/02_structural_summary.py
python scripts/03_coordination_dynamics.py
python scripts/04_dynamic_heterogeneity.py
python scripts/05a_build_pair_database.py --topology system.data --trajectory system.lammpsdump --cutoff-A 4.0 --overwrite
python scripts/05b_finalize_pair_database.py --overwrite
python scripts/06_pair_comotion.py --metal-mass-g-mol 40.0 --overwrite
python scripts/07_aggregate_dynamics.py --metal-mass-g-mol 40.0 --metal-charge-number 1 --overwrite
python scripts/08_segmental_motion.py --topology system.data --trajectory system.lammpsdump --peo-cutoff 4.0 --tfsi-cutoff 4.0 --overwrite
python scripts/09_collective_transport.py
python scripts/10_denticity.py --overwrite
python scripts/11_ligand_sharing.py --overwrite
python scripts/12_network_topology.py --metal-charge-number 1 --overwrite
python scripts/13_exchange_translation.py --topology system.data --trajectory system.lammpsdump --overwrite
python scripts/14_radius_of_gyration.py --topology system.data --trajectory system.lammpsdump
