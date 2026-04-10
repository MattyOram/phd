import sys
import glob
from pathlib import Path

inp_root = Path(sys.argv[1])
inps = list(inp_root.glob('**/*.inp'))
print(len(inps))

subjects = None
poses = None
run_ids = None
run_ids_mesh = None
