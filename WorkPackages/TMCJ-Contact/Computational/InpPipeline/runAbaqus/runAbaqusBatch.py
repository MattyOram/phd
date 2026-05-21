import sys
from pathlib import Path
import numpy as np
import subprocess




inp_root = Path(sys.argv[1]) # output_root of InpPipeline parameters
inps = np.sort(list(inp_root.glob('**/*.inp')))

# ------ SELECT INPUT FILES (if None runs all) ------------------------------------------------ #

subjects     = None # e.g. ['14548R', '50000R', ...]
poses        = None # e.g. ['neutral', 'flexion', ...]
run_ids      = ['00', '01', '02', '04', '06', '08', '10', '12', '14'] # strings e.g. ['0', '1', ...]
run_ids_mesh = None # strings e.g. ['0-0-0', '0-0-1', ...]

# --------------------------------------------------------------------------------------------- #





def get_sub(path: Path)->str:
    return path.parents[2].name
def get_pose(path: Path)->str:
    return path.name.split('-')[-2]
def get_id(path: Path)->str:
    return path.with_suffix('').name.split('-')[-1]
def get_id_mesh(path: Path)->str:
    return ('-').join(path.with_suffix('').name.split('-')[:-2])

# get mask of inputs if selected above
sub_mask = np.ones(len(inps), dtype=bool)
pose_mask = np.ones(len(inps), dtype=bool)
id_mask = np.ones(len(inps), dtype=bool)
id_mesh_mask = np.ones(len(inps), dtype=bool)

if subjects is not None:
    sub_mask = np.array([get_sub(x) in subjects for x in inps])
if poses is not None:
    pose_mask = np.array([get_pose(x) in poses for x in inps])
if run_ids is not None:
    id_mask = np.array([get_id(x) in run_ids for x in inps])
if run_ids_mesh is not None:
    id_mesh_mask = np.array([get_id_mesh(x) in run_ids_mesh for x in inps])

mask = sub_mask & pose_mask & id_mask & id_mesh_mask

inps = inps[mask]

print('\n Running input files:')
print(inps)
print('\n')



file_dir = Path(__file__).resolve().parent
run_script = file_dir / "runAbaqus.py"
abaqus_cmd = Path(r"C:\SIMULIA\Commands\abaqus.BAT")
for inp in inps:
    cmd = [
        "cmd",
        "/c",
        str(abaqus_cmd),
        'python',
        run_script,
        get_sub(inp),
        get_pose(inp),
        get_id(inp),
        get_id_mesh(inp),
        str(inp_root)
    ]
    subprocess.run(cmd, check=True)