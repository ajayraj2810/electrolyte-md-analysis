# Principal output files

The exact set of diagnostic plots and intermediate tables varies by module. Important outputs include:

| Module | Principal outputs |
|---|---|
| Coordination database | `master_coordination_*.csv.gz`, frame summary, anion-bridging table, polymer-bridging table, cluster summary, metadata JSON |
| Structural summary | overall/block structural summaries, state populations, coordination distributions, manuscript-ready figures |
| Coordination dynamics | contact-event summaries, survival functions, state transitions, ligand-exchange statistics |
| Dynamic heterogeneity | MSD, local exponent, NGP, van Hove distributions, state/cluster-conditioned mobility |
| Pair database | `pair_database.csv.gz`, frame pair summary, metadata/validation JSON |
| Pair finalization | pair-event and identity summaries, lifetime distributions, survival probabilities, convergence diagnostics |
| Pair co-motion | lag-resolved co-motion tables by lifetime/formation/coordination class |
| Aggregate dynamics | bridge/aggregate event summaries, lag-resolved transport and lineage metrics |
| Segmental motion | EO transition events, dwell statistics, local polymer/metal correlation functions |
| Collective transport | correlation functions, conductivity window scans, block uncertainty tables, decomposition diagnostics |
| Denticity | polymer/anion denticity distributions and block summaries |
| Ligand sharing | metals-per-polymer and metals-per-anion statistics |
| Network topology | component-size/topology summaries and giant-component proxy |
| Exchange/translation | shell-retention, turnover, retention-conditioned mobility and exchange-enhancement tables |
| Radius of gyration | chain-resolved and frame-averaged `Rg` tables plus distribution plot |
