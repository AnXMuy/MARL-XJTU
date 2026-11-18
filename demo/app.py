"""Main UI entrypoint: BenchMARL checkpoint, no EPyMARL dependency."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from demo.redesign import main
if __name__=='__main__':main()
