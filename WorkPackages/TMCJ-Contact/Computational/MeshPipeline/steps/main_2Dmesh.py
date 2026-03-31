import numpy as np
import pandas as pd
import pyvista as pv
from pathlib import Path
import shutil
import json
import sys
import time
import itertools


from smooth import smooth
from remesh import remesh_surface

from phd_helpers.paths import get_mesh, get_subject_stl_path
from phd_helpers.CartilageGeneration import get_min_df_fast

from MeshPrep import get_run_id

t0 = time.perf_counter()
print(f'\n\n---------------- REMESHING SURFACE ----------------\n\n')

##########################################################
# --------------------- PARAMETERS --------------------- #

print(f'LOADING PARAMETERS...')

argvs = sys.argv

# get param file
param_path = sys.argv[1] # path to json file of parameters
with open(param_path, "r") as f:
    params_2D = json.load(f)

output_dir = Path(argvs[2])

# get run id if triggered by main.py or passed run id args manually
if len(argvs) > 3:
    id_args = argvs[3:]
    run_id = get_run_id(id_args)
    print(run_id)
else: run_id = ''

overwrite = params_2D['overwrite']

input_bone_mesh = params_2D['input_bone_mesh']
input_arbone_mesh = params_2D['input_arbone_mesh']

output_filename = params_2D['output_filename']
output_filename_smooth = params_2D['output_filename_smooth']
cgal_input_name = params_2D['cgal_input_name']
cgal_path = Path(params_2D['cgal_path'])


subject_sideL = output_dir.parent.name
subject, sideL = subject_sideL[:-1], subject_sideL[-1]
stl_path = get_subject_stl_path(subject, sideL)

bone_arbone = output_dir.name
bone1, bone2 = bone_arbone.split('-')[0], bone_arbone.split('-')[1]
poses = params_2D['poses']

remesh_arbone = params_2D['remesh_arbone']

taubin_iters = params_2D['taubin_iters']
save_smoothed_mesh = params_2D['save_smoothed_mesh']

max_gap_remesh = params_2D['max_gap_remesh']
adjacent_cells = params_2D['adjacent_cells']

fine_edge_length = float(params_2D['fine_edge_length'])
coarse_edge_length = float(params_2D['coarse_edge_length'])
grad_width = float(params_2D['grad_width'])
remesh_iters = params_2D['remesh_iters']

print('Complete\n')

# --------------------- PARAMETERS --------------------- #
##########################################################

print(f'LOADING BONE MESHES ({bone1}, {bone2})...')

# target bone (neutral)
if    input_bone_mesh: bone1_mesh = pv.read(input_bone_mesh)
else: bone1_mesh = get_mesh(stl_path, bone1)

# articulating bone (neutral)
if    input_arbone_mesh: bone2_mesh = pv.read(input_arbone_mesh)
else: bone2_mesh = get_mesh(stl_path, bone2)


if    output_filename: remesh_file = output_filename
else: remesh_file = f'bone_remesh{run_id}.obj'

if    output_filename_smooth: smooth_file = output_filename_smooth
else: smooth_file = f'bone_smooth{run_id}.obj'

print('Complete\n')

# logic for remeshing either just bone or both - (for cartilage generation better to have both)
i = 1
if remesh_arbone: i = 2

bones = list(itertools.permutations([bone1, bone2], 2))
meshes = list(itertools.permutations([bone1_mesh, bone2_mesh], 2))

for (bone, ar_bone), (bone_mesh, arbone_mesh) in zip(bones[:i], meshes[:i]):

    #################################################################
    # --------------------- CREATE OUTPUT DIRs --------------------- #

    output_path = output_dir.parent / f'{bone}-{ar_bone}/2Dmesh'
    output_path.mkdir(parents=True, exist_ok=True)

    # --------------------- CREATE OUTPUT DIRs --------------------- #
    #################################################################

    # check if mesh already exists due to run with other bone pair
    mesh_exists = False
    out_file = output_path / remesh_file
    if out_file.is_file():
        mesh_exists = True
        print("Remesh already exists")
        print("Overwrite", overwrite, '\n')

    if not mesh_exists or overwrite:

        print('\n•••••••••••••••••••• ', bone.upper(), ' ••••••••••••••••••••\n')

        #########################################################
        # --------------------- SMOOTHING --------------------- #

        print('SMOOTHING MESH...')

        bone_mesh_smooth = smooth(bone_mesh, taubin_iters)
        arbone_mesh_smooth = smooth(arbone_mesh, taubin_iters)
        if save_smoothed_mesh:
            bone_mesh_smooth.save(output_path / smooth_file)

        print('Complete\n')

        # --------------------- SMOOTHING --------------------- #
        #########################################################

        ####################################################################################
        # --------------------- COPMUTING REMESH ARTICULATION REGION --------------------- #

        print('COMPUTING REMESH ARTICULATION REGION...')
        min_df = get_min_df_fast(
            stl_path, bone, ar_bone, bone_mesh_smooth, arbone_mesh_smooth, poses, max_gap_remesh
        )
        print('Complete\n')

        # --------------------- COPMUTING REMESH ARTICULATION REGION --------------------- #
        ####################################################################################


        #########################################################
        # --------------------- REMESHING --------------------- #

        if    cgal_input_name: cgal_name = cgal_input_name
        else: cgal_name = 'bone_remesh.obj'
        remesh_input_path = cgal_path / f'inputs/sf_input/{cgal_name}' # path to CGAL C++ input dir

        print(f'REMESHING... ({bone})')
        remesh_surface(bone_mesh_smooth, min_df, fine_edge_length, coarse_edge_length, 
                grad_width, remesh_input_path, n_iters=remesh_iters, adjacent_cells=adjacent_cells)

        # copy from remesh output dir to output path
        remesh_output_path = cgal_path / f'outputs/sf_output/{cgal_name}'
        shutil.copy(remesh_output_path, output_path / remesh_file)
        bone_remesh = pv.read(output_path / remesh_file)

        print('Complete\n')

        # --------------------- REMESHING --------------------- #
        #########################################################


dt = time.perf_counter() - t0
print(f"Step time: {dt:.3f}s")