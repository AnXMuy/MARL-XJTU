import os
from pathlib import Path

import sumo


def setup_sumo():
    home = Path(sumo.__file__).resolve().parent
    os.environ["SUMO_HOME"] = str(home)
    os.environ["PATH"] = str(home / "bin") + os.pathsep + os.environ.get("PATH", "")
    return home
