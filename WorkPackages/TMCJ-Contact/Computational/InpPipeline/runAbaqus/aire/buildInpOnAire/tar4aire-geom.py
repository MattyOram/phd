from pathlib import Path
import tarfile
import sys
import re
import shutil
import numpy as np


# ------ SELECT INPUT FILES (if None runs all) ------------------------------------------------ #

subjects     = None # e.g. ['14548R', '50000R', ...]
poses        = None # e.g. ['neutral', 'flexion', ...]
run_ids      = None # strings e.g. ['0', '1', ...]
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



def create_job_script(filepath: str, savepath: str, n: int) -> None:
    """
    create copy of slurm array script and fill N value
    """
    src = Path(filepath)
    dst = Path(savepath)

    shutil.copyfile(src, dst)

    text = dst.read_text()

    text = re.sub(
        r"^(#SBATCH\s+--array=1-)N(%\d+\s*)$",
        rf"\g<1>{n}\2",
        text,
        flags=re.MULTILINE,
    )

    dst.write_text(text)

path = Path(sys.argv[1]) # path to out_dir for .inp files
out_dir = path / 'aire/input'
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir.parent/'output').mkdir(parents=True, exist_ok=True) # also make dir for aire results

# get param files
params = sorted(list(path.glob('params/loop_params/*.json')))
# get input files
inps = sorted(list(path.glob('**/*-geom.inp')))
inps = np.array([x.parent / x.name.replace('-geom', '') for x in inps]) # remove -geom so that the rest can be the same as before

# MASK input files #
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
# MASK input files #


# get final input files list
inps = inps[mask]
inps_geom = [x.parent / x.name.replace('.inp', '-geom.inp') for x in inps]
tar_names = [x.parents[2].name + '-' + x.name for x in inps] # append sub to filename 
geom_names = [x.replace('.inp', '-geom.inp') for x in tar_names]
param_names = [x.split('-')[-1].replace('.inp', '.json') for x in tar_names]

# create text file of .inp file names for Aire slurm array
txt_file = out_dir / "inpFiles.txt"
txt_file.write_text(
    "\n".join(tar_names),
    encoding="utf-8"
)

# create text file of -geom.inp file names for Aire slurm array
txt_file_geom = out_dir / "inpFiles-geom.txt"
txt_file_geom.write_text(
    "\n".join(geom_names),
    encoding="utf-8"
)
# create text file of run_id.json file names for Aire slurm array
txt_file_param = out_dir / "paramFiles.txt"
txt_file_param.write_text(
    "\n".join(param_names),
    encoding="utf-8"
)

# create .sh file of slurm array Aire job script
template_path = Path(__file__).resolve().parent / "../AbaqusBatchTemplate.sh"
job_file = out_dir / "runAbaqus.sh"
create_job_script(
    filepath=template_path,
    savepath=job_file,
    n=len(tar_names),
)

# create .sh file of slurm array Aire job script for input file build
template_path_geom = Path(__file__).resolve().parent / "AbaqusBatchTemplate-geom.sh"
job_file_geom = out_dir / "runAbaqus-geom.sh"
create_job_script(
    filepath=template_path_geom,
    savepath=job_file_geom,
    n=len(geom_names),
)

# get build_abaqus_inp.py
build_inp_file = Path(__file__).resolve().parent / "build_abaqus_inp.py"

# tar .inp files and inpFiles.txt ready for transfer to Aire
output_tar = out_dir / "inpFiles.tar.gz"
with tarfile.open(output_tar, "w:gz") as tar:
    for inp_geom, geom_name in zip(inps_geom, geom_names):
        tar.add(inp_geom, arcname=geom_name) # add .inp files
    for param_file in params:
        tar.add(param_file, arcname=param_file.name) # add run_id.json param files
    tar.add(txt_file, arcname=txt_file.name) # add inpFiles.txt
    tar.add(job_file, arcname=job_file.name) # add rubAbaqus.sh
    tar.add(txt_file_geom, arcname=txt_file_geom.name) # add inpFiles-geom.txt
    tar.add(txt_file_param, arcname=txt_file_param.name) # ass paramFiles.txt
    tar.add(job_file_geom, arcname=job_file_geom.name) # add rubAbaqus-geom.sh
    tar.add(build_inp_file, arcname=build_inp_file.name) # add build_abaqus_inp.py