import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def metadata():
    versions = {}
    for name in ('torch','numpy','eclipse-sumo','traci','gymnasium','streamlit','pandas'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    def command(args):
        p = subprocess.run(args, capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else p.stderr.strip()
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'packages': versions, 'commit': command(['git','rev-parse','HEAD']),
            'dirty': bool(command(['git','status','--porcelain'])),
            'sumo': command(['sumo','--version']),
            'gpu': command(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv'])}

def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
