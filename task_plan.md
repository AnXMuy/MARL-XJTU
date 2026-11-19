# Traffic MARL execution plan
## Goal
Implement the supplied SOP in a new branch, with real SUMO experiments and honest reproducibility evidence; preserve pre-existing worktree deletions.
## Phases
1. Environment, isolated package, single-junction smoke and safety tests — in_progress
2. Grid, deterministic demand, baselines and authoritative metrics — pending
3. MAPPO/IPPO/IDQN and checkpoint evaluation — pending
4. Paired experiments, statistics, figures and ablations — pending
5. Interactive replay demo and technical report — pending
6. Reproducibility checks, selective commit and GitHub branch update — pending
## Next Step
Install an isolated compatible SUMO/Python environment and record actual versions.
## Decisions
Use traffic-marl/ to isolate this project from pre-existing deleted files. Never stage unrelated deletions.
## Errors
None yet.

## Update 2026-11-19
- Phase 1: unit-level safety/config/algorithm tests pass; real SUMO smoke blocked by wheel download. Keep status partial.
- Phases 2–5: first implementation drafted; integration and experimental evidence are pending SUMO.
- Phase 6: selective commit and branch push remain.
## Next Step
Record current limitation, then commit only traffic-marl/, SOP, and planning files without staging pre-existing deletions.
