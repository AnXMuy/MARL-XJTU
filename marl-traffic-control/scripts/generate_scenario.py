from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import sumo
import sumolib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NETWORK_CONFIG = PROJECT_ROOT / "configs/network.json"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import traffic_signal_marl


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_network(output: Path) -> None:
    config = json.loads(NETWORK_CONFIG.read_text())
    sumo_home = Path(sumo.__file__).resolve().parent
    os.environ["SUMO_HOME"] = str(sumo_home)
    command = [
        "netgenerate", "--grid",
        "--grid.number", str(config["grid_junction_count_per_axis"]),
        "--grid.length", str(config["grid_length_m"]),
        "--default.lanenumber", str(config["lanes_per_edge"]),
        "--default.speed", str(config["speed_limit_mps"]),
        "--tls.guess", "--tls.guess.threshold", str(config["traffic_light_guess_threshold"]),
        "--output-file", str(output),
    ]
    if not config["turnarounds"]:
        command.append("--no-turnarounds")
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)
    traffic_light_ids = sorted(light.getID() for light in sumolib.net.readNet(str(output)).getTrafficLights())
    if len(traffic_light_ids) != 16:
        raise RuntimeError(f"Expected 16 controlled intersections; generated {len(traffic_light_ids)}")
    manifest = {
        **config,
        "traffic_light_ids": traffic_light_ids,
        "network_sha256": sha256(output),
        "sumo_version": subprocess.run(["sumo", "--version"], check=True, capture_output=True, text=True).stdout.splitlines()[0],
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def generate_routes(network: Path, output: Path, seed: int, period: float, duration: int) -> None:
    if period <= 0 or duration <= 0:
        raise ValueError("period and duration must be positive")
    sumo_home = Path(sumo.__file__).resolve().parent
    random_trips = sumo_home / "tools/randomTrips.py"
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(random_trips),
        "--net-file", str(network),
        "--route-file", str(output),
        "--begin", "0", "--end", str(duration), "--period", str(period),
        "--seed", str(seed), "--fringe-factor", "10", "--validate", "--threads", "1",
    ]
    with tempfile.TemporaryDirectory() as temporary:
        command.extend(["--output-trip-file", str(Path(temporary) / "trips.xml")])
        subprocess.run(command, check=True, cwd=temporary)
    root = ET.parse(output).getroot()
    ET.indent(root)
    ET.ElementTree(root).write(output, encoding="UTF-8", xml_declaration=True)
    net = sumolib.net.readNet(str(network))
    traffic_lights = {signal.getID() for signal in net.getTrafficLights()}
    vehicles = root.findall("vehicle")
    bypass_vehicles = sum(not any(net.getEdge(edge).getToNode().getID() in traffic_lights
                                 for edge in vehicle.find("route").get("edges").split()[:-1])
                          for vehicle in vehicles)
    manifest = {
        "network": os.path.relpath(network, PROJECT_ROOT),
        "network_sha256": sha256(network),
        "route_file": os.path.relpath(output, PROJECT_ROOT),
        "route_sha256": sha256(output),
        "seed": seed,
        "period_seconds": period,
        "demand_duration_seconds": duration,
        "scheduled_vehicles": len(vehicles),
        "vehicles_bypassing_controlled_junctions": bypass_vehicles,
        "sumo_version": subprocess.run(["sumo", "--version"], check=True, capture_output=True, text=True).stdout.splitlines()[0],
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the fixed 4x4 signal network and deterministic trip files")
    subparsers = parser.add_subparsers(dest="command", required=True)
    network_parser = subparsers.add_parser("network")
    network_parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "scenarios/network/grid4x4.net.xml")
    route_parser = subparsers.add_parser("routes")
    route_parser.add_argument("--network", type=Path, default=PROJECT_ROOT / "scenarios/network/grid4x4.net.xml")
    route_parser.add_argument("--output", type=Path, required=True)
    route_parser.add_argument("--seed", type=int, required=True)
    route_parser.add_argument("--period", type=float, required=True)
    route_parser.add_argument("--duration", type=int, default=900)
    args = parser.parse_args()
    if args.command == "network":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        generate_network(args.output.resolve())
    else:
        generate_routes(args.network.resolve(), args.output.resolve(), args.seed, args.period, args.duration)


if __name__ == "__main__":
    main()
