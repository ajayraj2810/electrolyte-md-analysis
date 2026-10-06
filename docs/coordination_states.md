# Coordination-state and association definitions

## P/PT/T/F states

For a given metal ion and frame:

- **P:** one or more polymer-oxygen contacts and no anion contact.
- **PT:** simultaneous polymer-oxygen and anion coordination.
- **T:** one or more anion contacts and no polymer-oxygen contact.
- **F:** no polymer or anion contact under the chosen cutoffs.

States are assigned only after distance-based contacts have been calculated under periodic boundary conditions.

## Denticity

Polymer denticity counts the number of coordinating polymer oxygen atoms associated with a polymer chain. Anion denticity counts the number of oxygen atoms from a single anion coordinating one metal ion.

## Shared-chain occupancy

Two or more metal ions occupy the same polymer chain when they coordinate oxygen sites belonging to that chain in the same frame.

## Local polymer bridge

A shared polymer chain is classified as a local bridge when the closest coordinating polymer sites of the two metal ions are separated by no more than the configured site-index threshold.

## Metal–anion bridge

An anion bridge occurs when one anion simultaneously coordinates more than one metal ion.

## Metal–anion aggregate/network

A bipartite graph is formed from metal-ion and anion nodes with coordination contacts as edges. Connected components define instantaneous aggregates. The topology module summarizes component sizes and giant-component proxies, while the aggregate-dynamics module additionally tracks exact membership and lineage persistence.
