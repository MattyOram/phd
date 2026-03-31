import numpy as np
import pandas as pd
import pyvista as pv
from pathlib import Path
import json
import sys
import time


from cartilage import articular_gap

from phd_helpers.paths import get_subject_stl_path, get_mesh
from phd_helpers.CartilageGeneration import get_min_df_fast

from MeshPrep import get_run_id


t0 = time.perf_counter()
print(f'\n\n\n---------------- GENERATING CARTILAGE ----------------\n\n')

##########################################################
# --------------------- PARAMETERS --------------------- #

print(f'LOADING PARAMETERS...')

argvs = sys.argv

# get param file
param_path = sys.argv[1] # path to json file of parameters
with open(param_path, "r") as f:
    params_cart = json.load(f)

output_dir = Path(argvs[2])

# get run id if triggered by main.py or passed run id args manually
if len(argvs) > 3:
    id_args = argvs[3:]
    run_id = get_run_id(id_args)
    print(run_id)

    mesh_id = get_run_id(id_args[:-1])
else: run_id = ''

overwrite = params_cart['overwrite'] 

input_bone_mesh = params_cart['input_bone_mesh']
input_arbone_mesh = params_cart['input_arbone_mesh']

output_filename = params_cart['output_filename']
cgal_input_name = params_cart['cgal_input_name']
cgal_path = Path(params_cart['cgal_path'])


subject_sideL = output_dir.parent.name
subject, sideL = subject_sideL[:-1], subject_sideL[-1]
stl_path = get_subject_stl_path(subject, sideL)

bone_arbone = output_dir.name
bone, ar_bone = bone_arbone.split('-')[0], bone_arbone.split('-')[1]
poses = params_cart['poses']

remesh_cartilage = params_cart['remesh_cartilage']

save_orig_smooth = params_cart['save_orig_smooth']

use_remeshed_arbone = params_cart['use_remeshed_arbone']

max_gap_cartilage = params_cart['max_gap_cartilage']

taper_width = params_cart['taper_width']
#max_height = params_cart['max_height']
p_h = params_cart['p_h'] 
p_v = params_cart['p_v']
cartilage_smooth_iters = params_cart['smooth_iters']
n_iters = params_cart['n_iters']

edge_length = params_cart['edge_length']
if not edge_length: # set edge length from parent 2D mesh params
    mesh2D_id = id_args[-2]
    with open(Path(param_path).parent.parent / f'2Dmesh/{mesh2D_id}.json', "r") as f:
        params_2D = json.load(f)
    edge_length = params_2D['fine_edge_length']

print('Complete\n')

# --------------------- PARAMETERS --------------------- #
##########################################################


####################################################
# --------------------- DIRS --------------------- #

output_path = output_dir / '2Dmesh'
output_path.mkdir(parents=True, exist_ok=True)

if    output_filename: mesh_name = output_filename
else: mesh_name = f'bone_cartilage_mesh{run_id}.vtp'

# --------------------- DIRS --------------------- #
####################################################

# check if mesh already exists
mesh_exists = False
out_file = output_path / mesh_name
if out_file.is_file():
    mesh_exists = True
    print("Cartilage mesh already exists")
    print("Overwrite", overwrite, '\n')

if not mesh_exists or overwrite:

    ###########################################################
    # --------------------- LOAD MESHES --------------------- #

    print('LOADING MESHES...')

    # target bone (neutral)
    if    input_bone_mesh: bone_mesh = pv.read(input_bone_mesh)
    else: 
        try:
            bone_mesh = pv.read(output_path / f'bone_remesh{mesh_id}.obj')
        except:
            raise FileNotFoundError('No input mesh')

    # articulating bone (neutral)
    if    input_arbone_mesh: arbone_mesh = pv.read(input_arbone_mesh)
    else: 
        if use_remeshed_arbone:
            ar_path = output_dir.parent / f"{ar_bone}-{bone}/2Dmesh/bone_remesh{mesh_id}.obj"
            try:
                arbone_mesh = pv.read(ar_path)
            except:
                raise FileNotFoundError('No input mesh')
        else:
            arbone_mesh = get_mesh(stl_path, ar_bone)

    print('Complete\n')

    # --------------------- LOAD MESHES --------------------- #
    ###########################################################


    ########################################################################
    # --------------------- COMPUTE CARTILAGE REGION --------------------- #

    print('COMPUTING CARTILAGE REGION...')
    ################# GET CLOSEST POINTS ACCROSS ALL POSES #################
    # recomputed for new surface meshes
    min_df = get_min_df_fast(stl_path, bone, ar_bone, bone_mesh, arbone_mesh, poses, max_gap_cartilage)
    print('Complete\n')

    # --------------------- COMPUTE CARTILAGE REGION --------------------- #
    ########################################################################


    ###################################################################
    # --------------------- DETECT INTERFERENCE --------------------- #

    ###### can't detect interference at this point because some taper region point are inside as they don't have line of sight

    # --------------------- DETECT INTERFERENCE --------------------- #
    ###################################################################



    ##################################################################
    # --------------------- CREATING CARTILAGE --------------------- #

    # path to CGAL C++ imput dir
    if    cgal_input_name: cartilage_remesh_name = cgal_input_name
    else: cartilage_remesh_name = f'CartilageCap.obj' # cartilage cap mesh file name
    cgal_input_path = cgal_path / f'inputs/fb_input/{cartilage_remesh_name}'

    print('CREATING CARTILAGE...')
    orig_mesh, smooth_mesh, combined_mesh = articular_gap(
        bone_mesh,
        min_df,
        remesh_cartilage,
        cgal_input_path, # input path for CGAL C++ remeshing
        taper_width, # width of cartilage taper region 
        #max_height, # max height of cartilage in taper region
        p_h, # shape of taper height (1 = linear , higher = steeper taper)
        p_v, # shape of vector ratio (1 = linear)
        cartilage_smooth_iters, # need to look at this, currently uses laplacian  ##################### check this #############
        edge_length, # target edge length of cartilage remesh
        n_iters # n isotropic remeshing iterations for cartilage remesh
        )
    print('Complete\n')
    # --------------------- CREATING CARTILAGE --------------------- #
    ################################################################## 


    #########################################################
    # --------------------- SAVE MESH --------------------- #
    print(f"SAVING MESH... ({mesh_name})")
    if save_orig_smooth:
        orig_mesh.save(output_path / 'orig_cart_surf.vtp')
        smooth_mesh.save(output_path / 'smooth_cart_surf.vtp')
    combined_mesh.save(output_path / mesh_name, recompute_normals=False)
    print("Complete\n")
    # --------------------- SAVE MESH --------------------- #
    #########################################################

dt = time.perf_counter() - t0
print(f"Step time: {dt:.3f}s")