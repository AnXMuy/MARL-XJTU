# XJTU Traffic MARL

Reproducible SUMO signal-control experiments. This repository starts with a small synthetic grid: straight-through movements, one lane per approach, two non-conflicting phases. It does not model turning traffic, pedestrians, buses, or calibrated city behavior. Do not interpret simulation outcomes as field deployment evidence.

## Install

Requires Python 3.11+ and a working SUMO CLI (`sumo`, `netconvert`). Use the supplied `pyproject.toml`; freeze the resolved environment before a study:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
python -m pip freeze > requirements.lock
sumo --version
python -m pytest traffic-marl/tests
```

`SUMO_HOME` is not required when using the PyPI wheel. The CLI must be on PATH.

## Smoke and baselines

```bash
python traffic-marl/scripts/evaluate.py --config traffic-marl/configs/smoke.json --policy max_pressure --output traffic-marl/runs/smoke
python traffic-marl/scripts/evaluate.py --config traffic-marl/configs/quick.json --policy fixed --output traffic-marl/runs/fixed
```

Each run saves its exact generated network/demand inputs, config, provenance, per-second timeseries, completed/incomplete vehicle rows, executed signal states and summary. Seeded Poisson arrivals are generated once and hashed. Teleport is disabled and counted if SUMO still reports one. High-demand episodes retain unfinished vehicles.

## Train and evaluate

```bash
python traffic-marl/scripts/train.py --algorithm mappo --seed 0 --episodes 100 --output traffic-marl/runs/mappo0
python traffic-marl/scripts/train.py --algorithm ippo --seed 0 --episodes 100 --output traffic-marl/runs/ippo0
python traffic-marl/scripts/train.py --algorithm idqn --seed 0 --episodes 100 --output traffic-marl/runs/idqn0
python traffic-marl/scripts/evaluate.py --policy mappo --checkpoint traffic-marl/runs/mappo0/best.pt --output traffic-marl/runs/mappo-test
```

Checkpoints carry observation/safety config and training/validation/test seed metadata. Validation seeds select checkpoints; test seeds are held for final evaluation. Initial schedules use five training seeds as course-scale minimum, with 20 demand seeds per scenario. Increase with resources, report every seed and failed episode.

Run paired matrices with `scripts/run_matrix.py`. It saves raw episodes, mean/std and bootstrap confidence intervals plus paired differences against Fixed and Max-Pressure. Start with `--limit 4` to check throughput and schemas. Every strategy on a demand condition receives the same seed and input hash.

## Demo

```bash
streamlit run traffic-marl/src/traffic_marl/demo/app.py
```

The page runs SUMO synchronously and shows queues, completed vehicles, KPIs, downloads and optional recorded replay. Learnable policies appear in the selector only after a checkpoint is added in a future UI update; the current visible set contains rules only. Large episodes block the Streamlit worker; stop the process from the server if needed.

## Status

No performance claims have been made. See `evidence/` for environment snapshots, actual SUMO smoke results, test logs, and reproducible limitations. Results and figures must be regenerated from raw episode CSV.
