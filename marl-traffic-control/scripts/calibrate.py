import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from generate_scenario import generate_routes
from traffic_signal_marl.evaluation.episode import controller_from_name, run_episode


def main():
    output = ROOT / "results/calibration"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for period in [3.0, 1.0, 0.5, 0.25, 0.125]:
        for seed in [71, 72]:
            route = ROOT / f"scenarios/routes/calibration_period{period}_seed{seed}.rou.xml"
            generate_routes(ROOT / "scenarios/network/grid4x4.net.xml", route, seed, period, 300)
            for name in ["fixed-time", "max-pressure"]:
                result = run_episode(network_file=ROOT / "scenarios/network/grid4x4.net.xml", route_file=route,
                                     controller=controller_from_name(name), seed=seed, episode_seconds=600,
                                     result_dir=output / f"period{period}_{seed}_{name}")
                row = {"period": period, "seed": seed, "method": name, **result["metrics"]}
                rows.append(row)
                (output / "rows.json").write_text(json.dumps(rows, indent=2))
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
