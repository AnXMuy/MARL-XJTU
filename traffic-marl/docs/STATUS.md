# Implementation status (2026-11-19)

## Implemented
- Generated 1x1/3x3/4x4 SUMO straight-through grids, pre-generated seeded Poisson route files, per-entry demand configuration and input hashes.
- TraCI environment adapter with local observations, action masks, team/local queue rewards, replay snapshots, and episode truncation.
- Yellow + all-red safety state machine and Fixed/Actuated/Max-Pressure/offset controllers.
- Authoritative vehicle/queue/fairness/throughput/teleport metrics and raw CSV exports.
- MAPPO, IPPO, shared/independent actor choice, IDQN, PPO logging/checkpoints, matched evaluation and bootstrap summaries.
- Streamlit controls and recorded replay view; report template and reproduction commands.

## Verified here
- Python unit tests: 7 passed.
- Python compileall: passed.
- PyTorch: 2.14.0+cu130, CUDA runtime 13.0, CUDA available.

## Not verified or not done
- SUMO wheel could not be downloaded: the 105 MB wheel remained below 40 KB/s over two attempts. There is no system `sumo` binary, so network conversion, TraCI stepping, safety integration, trip metrics, UI behavior, and all traffic experiments remain unverified.
- No trained checkpoint or traffic results exist. Do not describe any strategy as empirically effective.
- Only straight-through traffic on single lanes is modeled. Turning movements, pedestrians, calibrated city demand, dynamic routing, tidal lanes, and full cancellation/worker controls are pending.
- The first smoke command is `python traffic-marl/scripts/evaluate.py --config traffic-marl/configs/smoke.json --policy max_pressure --output traffic-marl/runs/smoke` after SUMO installation.
