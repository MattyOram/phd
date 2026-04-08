
# #################### add option for const cartilage cell size ###################### #

import numpy as np
import pandas as pd
import pyvista as pv
from pathlib import Path
import json
import subprocess
import shutil
import meshio
import time
import sys

from phd_helpers.paths import find_shared_cells, identical_points_count, get_subject_stl_path, get_mesh, get_boundary
from MeshPrep import get_run_id

from ad_hoc_mesh_fix import ad_hoc_mesh_fix

def write_regions_off(mesh: pv.PolyData, region2filename: dict[int, str]):

    region_ids = mesh['region_id']
    tris = mesh.faces.reshape(-1, 4)[:, 1:]
    pts = np.asarray(mesh.points)

    for rid, out_path in region2filename.items():
        region_mask = region_ids == rid
        if region_mask.size == 0:
            raise ValueError(f"No faces found for region id {rid}.")

        region = tris[region_mask]
        used = np.unique(region.reshape(-1))
        old_to_new = -np.ones(pts.shape[0], dtype=np.int64)
        old_to_new[used] = np.arange(used.size, dtype=np.int64)
        pts_local = pts[used]
        tri_local = old_to_new[region]

        m = meshio.Mesh(points=pts_local, cells=[("triangle", tri_local)])
        meshio.write(out_path, m, file_format="off")


print(f'\n\n---------------- GENERATING VOLUMETRIC MESH ----------------\n\n')
t0 = time.perf_counter()

##########################################################
# --------------------- PARAMETERS --------------------- #

print(f'LOADING PARAMETERS...')

argvs = sys.argv

# get param file
param_path = sys.argv[1] # path to json file of parameters
with open(param_path, "r") as f:
    params_3D = json.load(f)

output_dir = Path(argvs[2])

# get run id if triggered by main.py or passed run id args manually
if len(argvs) > 3:
    id_args = argvs[3:]
    run_id = get_run_id(id_args)
    print(run_id)

    mesh_id = get_run_id(id_args[:-1])
else: run_id = ''

overwrite = params_3D['overwrite']

input_mesh = params_3D['input_mesh']

output_filename = params_3D['output_filename']


subject_sideL = output_dir.parent.name
subject, sideL = subject_sideL[:-1], subject_sideL[-1]
stl_path = get_subject_stl_path(subject, sideL)

bone_arbone = output_dir.name
bone, ar_bone = bone_arbone.split('-')[0], bone_arbone.split('-')[1]

keep_cgal_copy = params_3D['keep_cgal_copy']
postprocess = params_3D['postprocess']

save_cgal_inputs = params_3D['save_cgal_inputs']

cgal_params = params_3D['cgal_params']

if cgal_params["facet_distance"]["fd_edge_loop"] is None:
    cgal_params["facet_distance"]["fd_edge_loop"] = cgal_params["facet_distance"]["fd_cart_near"]

if cgal_params["facet_distance"]["fd_cart_far"] is None:
    cgal_params["facet_distance"]["fd_cart_far"] = cgal_params["facet_distance"]["fd_cart_near"]

if cgal_params["lloyd"]["convergence"] is None:
    cgal_params["lloyd"]["convergence"] = cgal_params["lloyd"]["freeze_bound"]

cgal_input_name = params_3D['cgal_input_name']
cgal_path = Path(params_3D['cgal_path'])
cgal_input_path = cgal_path / 'inputs/tr_input'
mesh_out_path = cgal_path / f"outputs/tr_output/mesh_{cgal_input_name}.mesh"


output_path = output_dir / '3Dmesh'
output_path.mkdir(parents=True, exist_ok=True)

if    output_filename: remesh_name = output_filename
else: remesh_name = f'mesh{run_id}.vtu'

print('Complete\n')

# --------------------- PARAMETERS --------------------- #
##########################################################

# check if mesh already exists
mesh_exists = False
out_file = output_path / remesh_name
if out_file.is_file():
    mesh_exists = True
    print("Mesh already exists")
    print("Overwrite", overwrite, '\n')

if not mesh_exists or overwrite:

    #########################################################
    # --------------------- LOAD MESH --------------------- #

    print('\nLOADING MESH...')
    if    input_mesh: mesh = pv.read(input_mesh)
    else: 
        try:
            mesh = pv.read(output_path.parent / f"2Dmesh/bone_cartilage_mesh{mesh_id}.vtp")
        except:
            raise FileNotFoundError('No input mesh')
    print('Complete\n')

    # --------------------- LOAD MESH --------------------- #
    #########################################################


    #################################################################
    # --------------------- WRITE CGAL INPUTS --------------------- #

    print("WRITING CGAL INPUTS...")

    # mesh surface patch names and filepaths
    bone_patch,      bone_id      = "bone_surf",      1
    cartilage_patch, cartilage_id = "cartilage_surf", 2
    interface_patch, interface_id = "interface_surf", 3

    bone_path = cgal_input_path / f"{bone_patch}_{cgal_input_name}.off"
    cartilage_path = cgal_input_path / f"{cartilage_patch}_{cgal_input_name}.off"
    interface_path = cgal_input_path / f"{interface_patch}_{cgal_input_name}.off"

    # remeshing parameters filenames
    cgal_params_filepath = cgal_input_path / f"params_{cgal_input_name}.json"
    patches_filepath = cgal_input_path / f"patches_{cgal_input_name}.json"

    data = {
        "subdomains": {"outside": 0, "bone": 1, "cartilage": 2},
        "patches": [
            {
                "name": bone_patch,
                "file": bone_path.name,
                "incident_subdomains": ["bone", "outside"], # normals point bone->outside
            },
            {
                "name": cartilage_patch,
                "file": cartilage_path.name,
                "incident_subdomains": ["cartilage", "outside"], # normals point cartilage->outside
            },
            {
                "name": interface_patch,
                "file": interface_path.name,
                "incident_subdomains": ["bone", "cartilage"], # normals point bone->cartilage
            }
        ]
    }

    print ("Writing mesh surface patches")
    with open(patches_filepath, "w") as f:
        json.dump(data, f, indent=2)

    # map region ids in mesh to cgal input surface patches
    region2filename = {
        bone_id: bone_path,
        cartilage_id: cartilage_path,
        interface_id: interface_path,
    }
    write_regions_off(mesh, region2filename)

    print("Writing remeshing parameters")
    with open(cgal_params_filepath, "w") as f:
        json.dump(cgal_params, f, indent=2)


    if save_cgal_inputs:
        # keep copy of meshing input files
        inputs_copy_path = output_path / f'cgal-inputs{run_id}' 
        inputs_copy_path.mkdir(parents=True, exist_ok=True)

        # patches json
        with open(inputs_copy_path/f'patches.json', "w") as f:
            json.dump(data, f, indent=2)
        # .off patch files
        region2filename_inputs = {
            bone_id: inputs_copy_path / f'{bone_patch}.off',
            cartilage_id: inputs_copy_path / f'{cartilage_patch}.off',
            interface_id: inputs_copy_path / f'{interface_patch}.off',
        }
        write_regions_off(mesh, region2filename_inputs)

        with open(inputs_copy_path/f'params.json', "w") as f:
            json.dump(cgal_params, f, indent=2)

    print('Complete\n')

    # --------------------- WRITE CGAL INPUTS --------------------- #
    #################################################################





    #########################################################
    # --------------------- REMESHING --------------------- #

    print('REMESHING...')

    exe = cgal_path / "bin/mesh_tr"

    args = [
        str(exe),
        str(patches_filepath),   # input mesh patches .json path
        str(cgal_params_filepath),  # mesh params .json path
        str(mesh_out_path),      # mesh output .mesh path
    ]

    t0_mesh = time.perf_counter()
    result = subprocess.run(args, text=True)
    dt_mesh = time.perf_counter() - t0_mesh

    #print(f"returncode: {result.returncode}")
    print(f"Mesh time: {dt_mesh:.3f}s")
    result.check_returncode()

    # copy from remesh output dir to output path
    if keep_cgal_copy:
        shutil.copy(mesh_out_path, output_path / remesh_name.replace('.vtu', '.mesh'))

    print('Complete\n')

    # --------------------- REMESHING --------------------- #
    #########################################################

    if not keep_cgal_copy and not postprocess:
        print("WARNING - both: keep_cgal_copy & postprocess = FALSE\nNo mesh saved to output path\n")


    ##############################################################
    # --------------------- POSTPROCESSING --------------------- #

    #••• THIS RELIES ON ALL BONE TETS BEING CONTAINED WITHIN ORIGINAL BONE VOLUME - PROBABLY TRUE BUT NOT GURANTEED •••#
    # - updated to find any cells not within bone volume as long as they aren't connected to any cartilage cells (better, not perfect)
    # - updated to again, to address most cases for when cells are not in bone volume but are connected to cartilage cells

    print('POSTPROCESSING...')

    if postprocess:
        # MAIN

        # load in cgal 3D mesh
        cgal_mesh = pv.read(mesh_out_path)

        tet = cgal_mesh.extract_cells_by_type(10) # extract tetrahedral mesh
        tet['cell_id_tet'] = np.arange(tet.n_cells)
        # mask of tets in original bone volume
        bone_mesh_shell = mesh.extract_cells(mesh['region_id']==cartilage_id, invert=True).extract_surface(algorithm=None)
        tet_bone_ids = tet['cell_id_tet'][tet.cell_centers().compute_implicit_distance(bone_mesh_shell)['implicit_distance'] <= 0]
        cartilage_tet = tet.extract_cells(tet_bone_ids, invert=True)

        # check if any bone cells were not contained within input bone volume (won't catch cells that are connected to cartilage!)
        # - and assign them to bone
        cartilage_conn = cartilage_tet.connectivity()
        cart_island_ids, island_counts = np.unique(cartilage_tet.connectivity()["RegionId"], return_counts=True)
        if cart_island_ids.shape[0] > 1:
            bone_islands = []
            for island_id in cart_island_ids[island_counts < island_counts.max()]: # region ids of islands that should be part of bone
                bone_islands.append(cartilage_conn['cell_id_tet'][cartilage_conn['RegionId']==island_id])
            tet_bone_ids = np.hstack((np.hstack(bone_islands), tet['cell_id_tet'][tet_bone_ids]))

            cartilage_tet = tet.extract_cells(tet_bone_ids, invert=True)
            bone_tet = tet.extract_cells(tet_bone_ids)
        else:
            bone_tet = tet.extract_cells(tet_bone_ids)

        # assing region ids to tet mesh
        tet_bone_mask = np.isin(tet['cell_id_tet'], tet_bone_ids)
        tet['region_id'] = np.ones(tet.n_cells, dtype=int) # 1 bone volume
        tet['region_id'][~tet_bone_mask] = 2               # 2 cartilage volume

        # check for islands in volume meshes incase assinging region ids to cells went wrong
        if bone_tet.connectivity()["RegionId"].max() + 1 != 1:
            raise RuntimeError("Cell islands in bone volume mesh!") 
        if cartilage_tet.connectivity()["RegionId"].max() + 1 != 1:
            raise RuntimeError("Cell islands in cartilage volume mesh!")

        # extract volume shells for surface meshes
        cartilage_shell = cartilage_tet.extract_surface(algorithm=None) # cartilage shell
        bone_shell = bone_tet.extract_surface(algorithm=None) # bone shell
        #tet_shell = tet.extract_surface(algorithm=None) # tet shell

        # find interface surfaces on cartilage and bone
        interface_mask_bone = find_shared_cells(bone_shell, cartilage_shell)
        interface_mask_cartilage = find_shared_cells(cartilage_shell, bone_shell)
        interface_bone = bone_shell.extract_cells(interface_mask_bone)
        interface_cartilage = cartilage_shell.extract_cells(interface_mask_cartilage)
        # check they are identical
        n_shared = identical_points_count(interface_cartilage.points, interface_bone.points)
        if n_shared != interface_bone.n_points and n_shared == interface_cartilage.n_points:
            raise AssertionError('Interfaces are not identical')
        # assign bone interface to interface surf for consistency (normals point outwards relative to bone)
        interface_surf = interface_bone
        interface_surf['region_id'] = np.full(interface_surf.n_cells, -3)

        # extract bone and cartilage surfs
        bone_surf = bone_shell.extract_cells(~interface_mask_bone)
        bone_surf['region_id'] = np.full(bone_surf.n_cells, -1)
        cartilage_surf = cartilage_shell.extract_cells(~interface_mask_cartilage)
        cartilage_surf['region_id'] = np.full(cartilage_surf.n_cells, -2)

        # combined mesh of tets and tris with region ids - ••• Needs to be tets then tris for abaqus input script to work •••
        combined = tet + interface_surf + bone_surf + cartilage_surf

        # CHECKS #
        #print("\nFINAL MESH...\n")
        if not tet.n_points == combined.n_points:
            raise AssertionError("All points merged in final mesh:")

        # These checks are probs trivial...
        ids, counts = np.unique(combined['region_id'], return_counts=True)
        checks = {
        "Region ID assigned to all cells:         ": np.isin(ids, [-3, -2, -1, 1, 2]).all(),
        "All bone cells present (1):              ": counts[ids==1][0] == bone_tet.n_cells,
        "All cartilage cells present (2):         ": counts[ids==2][0] == cartilage_tet.n_cells,
        "All bone surface cells present (-1):     ": counts[ids==-1][0] == bone_surf.n_cells,
        "All cartilage surface cells present (-2):": counts[ids==-2][0] == cartilage_surf.n_cells,
        "All interface surface cells present (-3):": counts[ids==-3][0] == interface_surf.n_cells
        }

        if not np.all(list(checks.values())):
            for key, value in checks.items():
                print(key, value)
                raise AssertionError("Postprocessing failed")


        # QUICK FIX #
        # quick, slow fix to address " as long as they aren't connected to any cartilage cells"
        # - from study1-analyseMetrics-box.ipynb ; still probably some edge cases it doesn't work for
        combined = ad_hoc_mesh_fix(combined)


        print("Saving mesh")
        combined.save(output_path / remesh_name)

        print('Complete\n')

    # --------------------- POSTPROCESSING --------------------- #
    ##############################################################

dt = time.perf_counter() - t0
print(f"Step time: {dt:.3f}s")