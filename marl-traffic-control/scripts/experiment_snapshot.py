"""Record the environment and source used for an experiment."""
import hashlib
import importlib.metadata
import json
import platform
import zipfile

from traffic_signal_marl.provenance import git_metadata

def snapshot(output, root):
    output.mkdir(parents=True, exist_ok=True)
    revision, status = git_metadata(root)
    manifest = {"python": platform.python_version(), "packages": {name: importlib.metadata.version(name)
                for name in ["numpy", "torch", "sumo-rl", "pettingzoo", "gymnasium", "traci", "sumolib"]},
                "git_commit": revision, "git_status": status,
                "source_hashes": {}}
    with zipfile.ZipFile(output / "source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for directory in ["src", "scripts", "configs", "tests"]:
            for path in sorted((root / directory).rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    relative = str(path.relative_to(root))
                    manifest["source_hashes"][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
                    archive.write(path, relative)
    (output / "environment.json").write_text(json.dumps(manifest, indent=2))
