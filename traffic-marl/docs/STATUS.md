# Implementation status (2026-11-19)

## Implemented
- SUMO 1.25 straight-through grids, fixed seeded Poisson demand, per-entry rates and demand hashes.
- TraCI observations, masks, shared/local queue reward, action logs, output replay.
- Yellow + all-red signal safety state machine; fixed, actuated, movement queue pressure and offset policies.
- Vehicle/queue/fairness/throughput/teleport metrics and CSV output.
- MAPPO/IPPO/shared or independent actors/IDQN implementations, checkpointing and paired evaluation code.
- Streamlit UI, report template, dependency lock and setup scripts.

## Verified here
- `eclipse-sumo==1.25.0`, `traci==1.25.0`, `sumolib==1.25.0`; PyTorch 2.14.0+cu130 with CUDA 13.0 runtime. The current lock is a direct-dependency snapshot; generate a full transitive lock in a clean venv because this container's global site-packages add machine-local conda URLs.
- Unit and SUMO integration tests: 10 passed, including repeatable single-junction and paired 3x3 demand.
- Real 1x1 SUMO smoke: 1000 s, 320 scheduled/departed, 299 arrived, 21 remain, 0 teleports, 0 illegal requests, ~269 simulation steps/s.
- Real 3x3 quick runs: 600 scheduled vehicles for fixed/actuated/max-pressure, identical demand XML hash, repeated pressure result identical, 0 teleports and 0 illegal requests. Values are short smoke evidence only, not statistically meaningful comparison.
- Streamlit starts on port 8765 and its health endpoint returns `ok`.
- Playwright browser check loaded the demo, clicked Run SUMO on 1x1, and found KPI charts with no page errors.
- One-episode MAPPO, IPPO and IDQN training plus MAPPO checkpoint inference all complete. A 3x3, 120-decision MAPPO sanity run has 1,080 agent steps; its one held-out run is poor (queue mean 165.7 vs 19.7 Max-Pressure, 153 vs 427 arrivals). This is a negative smoke result, not an efficacy estimate.

## Not verified or not done
- No multi-seed study, confidence interval, or supported performance claim exists; training sanity runs are far below a meaningful budget.
- Browser replay slider/control and replay-on browser rendering have not been exercised; replay export itself was generated (600 frames, about 11 MB).
- Only straight-through single-lane traffic is modeled; turns, pedestrians, calibrated city traffic, route guidance and reversible lanes are pending.
- SUMO Python wheel installs CLI tools under `.venv/bin`; call commands through `traffic-marl/scripts/sumo_env.sh` or add `.venv/bin` to PATH.

## Smoke command

```bash
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python traffic-marl/scripts/evaluate.py --config traffic-marl/configs/smoke.json --policy max_pressure --output traffic-marl/runs/smoke
```
