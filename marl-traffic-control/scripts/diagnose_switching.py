"""Replay a post-hoc 10-second rotation on one paired seed per profile."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traffic_signal_marl.baselines.controllers import FixedTimeController
from traffic_signal_marl.evaluation.episode import run_episode


def main():
    seed = 1000
    studio = ROOT / "results/study-v1"
    output = studio / "diagnostics/fixed-10s"
    policy = FixedTimeController(phase_duration_seconds=10)
    policy.name = "fixed-10s"
    for profile in ("C1", "C2", "C3"):
        folder = output / profile
        path = folder / "result.json"
        if not path.exists():
            result = run_episode(network_file=ROOT / "scenarios/network/grid4x4.net.xml",
                                 route_file=ROOT / f"scenarios/routes/test_{profile}_{seed}.rou.xml",
                                 controller=policy, seed=seed, episode_seconds=600, result_dir=folder)
            path.write_text(json.dumps(result, indent=2))
        result = json.loads(path.read_text())
        print(profile, seed, result["metrics"]["phase_switches"],
              result["metrics"]["restricted_wait_seconds_per_scheduled_vehicle"], flush=True)


if __name__ == "__main__":
    main()
