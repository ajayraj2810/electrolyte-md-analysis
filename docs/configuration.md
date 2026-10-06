# Configuration guide

The public workflow intentionally avoids named metal species. Configure physical and topological properties explicitly.

## Required concepts

- **metal selection** — MDAnalysis selection identifying metal-ion atoms.
- **polymer oxygen selection** — oxygen sites used for metal–polymer coordination.
- **anion oxygen selection** — anion oxygen sites used for metal–anion coordination.
- **anion representative selection** — one representative site per anion for displacement analyses.
- **ionic-liquid cation representative selection** — one representative site per cation when collective transport is calculated.
- **coordination cutoffs** — system-specific distances, preferably taken from RDF first minima.
- **frame interval** — physical time represented by one stored trajectory frame.
- **metal charge number** — required for charge-current / conductivity calculations.
- **metal molar mass** — required only for mass-weighted pair/aggregate COM quantities and optional barycentric drift removal.

`configs/example.yaml` documents these values in one place. Several validated modules retain their original standalone settings/CLI style, so treat the YAML file as a shared schema/reference rather than an automatic universal driver.
