#!/usr/bin/env bash
set -euo pipefail

# Install once from the repository root:
#   python -m pip install -e .
#
# Adapt topology/trajectory selections and cutoffs to your system before running.

python scripts/build_coordination_database.py
python scripts/structural_summary.py
python scripts/coordination_dynamics.py
python scripts/dynamic_heterogeneity.py
python scripts/build_pair_database.py --topology system.data --trajectory system.lammpsdump --cutoff-A 4.0 --overwrite
python scripts/finalize_pair_database.py --overwrite
python scripts/pair_comotion.py --metal-mass-g-mol 40.0 --overwrite
python scripts/aggregate_dynamics.py --metal-mass-g-mol 40.0 --metal-charge-number 1 --overwrite
python scripts/segmental_motion.py --topology system.data --trajectory system.lammpsdump --peo-cutoff 4.0 --tfsi-cutoff 4.0 --overwrite
python scripts/collective_transport.py
python scripts/denticity.py --overwrite
python scripts/ligand_sharing.py --overwrite
python scripts/network_topology.py --metal-charge-number 1 --overwrite
python scripts/exchange_translation.py --topology system.data --trajectory system.lammpsdump --overwrite
python scripts/radius_of_gyration.py --topology system.data --trajectory system.lammpsdump
