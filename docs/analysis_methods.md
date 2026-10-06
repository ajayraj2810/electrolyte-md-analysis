# Analysis methods

## Continuous and intermittent residence

Continuous lifetimes require uninterrupted contact over consecutive analyzed frames. Intermittent lifetimes allow a configurable short gap before an event is considered terminated. Both are reported because fast boundary recrossings can otherwise dominate contact statistics.

## Mean-square displacement and dynamic heterogeneity

The dynamic-heterogeneity module computes unwrapped displacement statistics, including MSD, the local MSD power-law exponent, the non-Gaussian parameter, and self van Hove displacement distributions. Mobility can be conditioned on initial or persistent coordination states and aggregate membership.

## Pair co-motion

For persistent metal–anion pair events, the workflow compares metal and anion displacements, relative displacement, displacement dot products, directional correlation, and mass-weighted pair center-of-mass motion as a function of lag time and event lifetime.

## Bridge/aggregate dynamics

Multi-metal bridge events and exact-membership aggregates are tracked through time. Outputs include lifetimes, co-motion, relative-motion measures, translational fractions, and aggregate lineage persistence.

## Polymer segmental motion

Metal motion is compared with the displacement of the local polymer segment while the same dominant chain remains associated over the lag interval. EO-site trajectories are separately classified into same-chain neighbor sliding, larger same-chain jumps, interchain hops, attachment, and detachment.

## Collective transport

The collective-transport module evaluates Einstein-Helfand displacement correlations and decomposes the collective response into same-species self/distinct and cross-species Onsager terms. Multiple fit windows and block schemes are used to expose slope stability rather than relying on a single arbitrary fitting interval.

## Exchange versus translation

For a metal ion at time origin `t0` and lag `t`, the displacement is compared with retention of the original coordination shell. Endpoint retention, ligand loss/gain, cumulative shell turnover, retention-conditioned MSD, and exchange-enhancement ratios distinguish translation with a persistent shell from motion coupled to ligand exchange.
