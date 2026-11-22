from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from pettingzoo.test import parallel_api_test

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import traffic_signal_marl
from sumo_rl import parallel_env


def main() -> None:
    grid_dir = Path(__import__("sumo_rl").__file__).resolve().parent / "nets/RESCO/grid4x4"
    network = grid_dir / "grid4x4.net.xml"
    routes = grid_dir / "grid4x4_1.rou.xml"
    if not network.is_file() or not routes.is_file():
        raise FileNotFoundError(f"SUMO-RL bundled smoke scenario missing: {grid_dir}")

    version = subprocess.run(["sumo", "--version"], check=True, capture_output=True, text=True).stdout.splitlines()[0]
    env = parallel_env(
        net_file=str(network),
        route_file=str(routes),
        use_gui=False,
        num_seconds=60,
        delta_time=5,
        yellow_time=3,
        min_green=5,
        sumo_seed=42,
    )
    try:
        parallel_api_test(env, num_cycles=3)
    finally:
        env.close()
    print(json.dumps({"sumo": version, "scenario": "SUMO-RL RESCO grid4x4", "seed": 42, "api_cycles": 3, "status": "passed"}))


if __name__ == "__main__":
    main()
