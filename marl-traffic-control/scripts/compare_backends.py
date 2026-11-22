from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from extended import ROOT, atomic_json, evaluation_task


def examples(study):
    for profile in ("C1", "C2", "C3"):
        yield profile, "fixed-10", None, None
        yield profile, "max-pressure", None, None
        yield profile, "mappo", 11, study / "models/mappo/11/checkpoint_24.pt"


def run(study, output, backend):
    for profile, method, init, checkpoint in examples(study):
        evaluation_task((output / backend, study / f"routes/test_{profile}_2000.rou.xml", method, init, checkpoint,
                         600, "pilot", profile, 2000))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, default=ROOT / "results/study-v2")
    parser.add_argument("--output", type=Path, default=ROOT / "results/backend-pilot")
    parser.add_argument("--worker", choices=["traci", "libsumo"])
    args = parser.parse_args()
    if args.worker:
        run(args.study, args.output, args.worker)
        return
    processes = []
    for backend in ("traci", "libsumo"):
        environment = dict(os.environ)
        environment.pop("LIBSUMO_AS_TRACI", None)
        environment.pop("LIBTRACI_AS_TRACI", None)
        if backend == "libsumo":
            environment["LIBSUMO_AS_TRACI"] = "quiet"
        processes.append(subprocess.Popen([sys.executable, __file__, "--study", str(args.study), "--output",
                                           str(args.output), "--worker", backend], env=environment))
    statuses = [process.wait() for process in processes]
    if any(statuses):
        raise RuntimeError(f"Backend comparison failed: {statuses}")
    comparisons = []
    for profile, method, init, _ in examples(args.study):
        folders = [args.output / backend / f"evaluation/pilot/{profile}/2000/{method}/{init}" for backend in ("traci", "libsumo")]
        traci, libsumo = [json.loads((folder / "result.json").read_text()) for folder in folders]

        def equal_artifact(field):
            return (traci[field] is None and libsumo[field] is None) or (
                (folders[0] / Path(traci[field]).name).read_bytes() == (folders[1] / Path(libsumo[field]).name).read_bytes())

        comparisons.append(dict(profile=profile, method=method, init=init, traci_s=traci["runtime_seconds"],
                                libsumo_s=libsumo["runtime_seconds"], same_metrics=traci["metrics"] == libsumo["metrics"],
                                same_trace=equal_artifact("trace_file"), same_actions=equal_artifact("action_trace_file")))
    atomic_json(args.output / "comparison.json", comparisons)
    print(json.dumps(comparisons, indent=2))
    if not all(row["same_metrics"] and row["same_trace"] and row["same_actions"] for row in comparisons):
        raise ValueError("Backend equivalence failed on this pilot; preserve original backend")


if __name__ == "__main__":
    main()
