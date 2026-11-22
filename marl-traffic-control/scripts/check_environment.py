from __future__ import annotations

import importlib.metadata
import os
import shutil
import subprocess
import sys
import importlib
from pathlib import Path


def main() -> int:
    if not os.environ.get("SUMO_HOME"):
        try:
            sumo_module = importlib.import_module("sumo")
            os.environ["SUMO_HOME"] = str(Path(sumo_module.__file__).resolve().parent)
        except (ImportError, TypeError):
            pass
    print(f"Python: {sys.version.split()[0]}")
    for package in ("sumo-rl", "gymnasium", "pettingzoo", "torch", "numpy"):
        try:
            print(f"{package}: {importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{package}: MISSING")
    for module in ("sumo_rl", "gymnasium", "pettingzoo", "traci", "sumolib"):
        try:
            importlib.import_module(module)
            print(f"import {module}: OK")
        except ImportError as error:
            print(f"import {module}: FAILED ({error})")

    sumo_binary = shutil.which("sumo")
    print(f"SUMO_HOME: {os.environ.get('SUMO_HOME', 'UNSET')}")
    if not sumo_binary:
        print("SUMO binary: MISSING (install SUMO before simulator smoke tests)")
        return 1
    result = subprocess.run([sumo_binary, "--version"], check=False, text=True, capture_output=True)
    print(result.stdout.strip() or result.stderr.strip())
    if result.returncode:
        return result.returncode
    if not os.environ.get("SUMO_HOME"):
        print("SUMO_HOME is unset; SUMO-RL's Python tools may not be importable.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
