# Analysis workflow

The modules are intentionally separated into a database-building stage and downstream analyses.

## Stage A — trajectory-derived coordination database

`build_coordination_database.py` is the central structural pass through the trajectory. It assigns polymer and anion contacts to every metal ion at every saved frame and writes compact databases that can be reused without repeatedly reconstructing the full coordination environment.

## Stage B — database-derived structure and dynamics

`structural_summary.py`, `coordination_dynamics.py`, `denticity.py`, `ligand_sharing.py`, and `network_topology.py` consume the master database and quantify state populations, contact residence, denticity, ligand sharing, and network structure.

## Stage C — trajectory-resolved mobility

`dynamic_heterogeneity.py`, `segmental_motion.py`, `collective_transport.py`, `exchange_translation.py`, and `radius_of_gyration.py` use coordinates directly, sometimes together with the master database, to relate structure to motion.

## Stage D — explicit metal–anion pair persistence

`build_pair_database.py` constructs continuous pair events. `finalize_pair_database.py` computes lifetime statistics and censoring-aware summaries. `pair_comotion.py` and `aggregate_dynamics.py` then quantify pair/bridge/aggregate transport using those event records.

The recommended order is therefore:

1. coordination database
2. structural summary
3. coordination dynamics
4. dynamic heterogeneity
5. pair database + finalization
6. pair co-motion
7. aggregate dynamics
8. segmental motion
9. collective transport
10. denticity
11. ligand sharing
12. network topology
13. exchange vs translation
14. radius of gyration
