import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path


spec = importlib.util.spec_from_file_location("generate_scenario", Path(__file__).parents[1] / "scripts/generate_scenario.py")
generation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generation)


def test_deterministic_routes(tmp_path):
    network = Path(__file__).parents[1] / "scenarios/network/grid4x4.net.xml"
    first, second = tmp_path / "a.rou.xml", tmp_path / "b.rou.xml"
    generation.generate_routes(network, first, 19, 5, 60)
    generation.generate_routes(network, second, 19, 5, 60)
    assert first.read_bytes() == second.read_bytes()
    assert len(ET.parse(first).getroot().findall("vehicle")) == 12
