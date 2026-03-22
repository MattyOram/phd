# this takes the volumes outputted from the 3D meshing and ensures that the volume shells are manifold (not guaranteed)
# - will be watertight but it is easy to get edges the are "pinched" that are shared by 4 faces
# - also this might not be necessary ; watertight might be good enough for 3D printing software - haven't checked


# •••••••••••••••• Assumes the interface isn't the problem - maybe major floor, but difficult to do it more sophisticated


import numpy as np
import pyvista as pv
import pymeshfix
import gdist
from pathlib import Path
import sys
import json

from phd_helpers.paths import find_corresponding_cells, get_boundary
from phd_helpers.CartilageGeneration import remove_normals

from MeshPrep import get_run_id

##########################################################
# --------------------- PARAMETERS --------------------- #
print(f'LOADING PARAMETERS...')

argvs = sys.argv

# get param file
param_path = sys.argv[1] # path to json file of parameters
with open(param_path, "r") as f:
    params_man = json.load(f)

output_dir = Path(argvs[2])

# get run id if triggered by main.py or passed run id args manually
if len(argvs) > 3:
    id_args = argvs[3:]
    run_id = get_run_id(id_args)
    print(run_id)

    mesh_id = get_run_id(id_args[:-1])
else: run_id = ''

input_mesh = params_man['input_mesh']

output_filename = params_man['output_filename']

subject_sideL = output_dir.parent.name
subject, sideL = subject_sideL[:-1], subject_sideL[-1]

bone_arbone = output_dir.name
bone, ar_bone = bone_arbone.split('-')[0], bone_arbone.split('-')[1]

require_manifold = params_man['require_manifold']
max_area = params_man['max_area']
max_loc = params_man['max_loc']


output_path = output_dir / '3Dmesh/3Dprint'
output_path.mkdir(parents=True, exist_ok=True)





print('Complete\n')

# --------------------- PARAMETERS --------------------- #
##########################################################


#########################################################
# --------------------- LOAD MESH --------------------- #

print('\nLOADING MESH...')

if    input_mesh: mesh = pv.read(input_mesh)
else: 
    try:
        mesh = pv.read(output_path.parent / f'mesh{mesh_id}.vtu') # mesh file name
    except:
        raise FileNotFoundError('No input mesh')

print('Complete\n')

# --------------------- LOAD MESH --------------------- #
#########################################################

mesh_tri = mesh.extract_cells_by_type(5)

cartilage_shell = mesh_tri.extract_cells(np.where(mesh_tri['region_id']==-1)[0], invert=True).extract_geometry()
bone_shell = mesh_tri.extract_cells(np.where(mesh_tri['region_id']==-2)[0], invert=True).extract_geometry()

shells = {
    -1: bone_shell,
    -2: cartilage_shell
}
shell_map = {
    -1: 'Bone',
    -2: 'Cartilage'
}

# orient normals to point outwards
for shell in shells.values():
    remove_normals(shell)
    shell.compute_normals(auto_orient_normals=True, inplace=True)
    shell['repaired'] = np.zeros(shell.n_cells)

# identify non-manifold shells
bad_shells = [region_id for region_id, shell in shells.items() if not shell.is_manifold]
if len(bad_shells)==0:
    print('\nBone & Cartilage regions are manifold')

for region_id in bad_shells:
    shell = shells[region_id]

    print(f'\nFIXING NON-MANIFOLD REGION ({region_id})...\n')

    shell['shell_point_id'] = np.arange(shell.n_points)
    shell['shell_cell_id'] = np.arange(shell.n_cells)
    surf = shell.extract_cells(np.where(shell['region_id']==region_id)[0]).extract_geometry()
    surf_faces = surf.faces.reshape(-1, 4)[:, 1:]


    # find non-manifold edges / points and remove the faces connected to those points
    bad_edges = shell.extract_feature_edges(manifold_edges=False,boundary_edges=False,feature_edges=False,non_manifold_edges=True)
    bad_points = np.arange(surf.n_points)[np.isin(
                                                        surf['shell_point_id'], 
                                                        bad_edges['shell_point_id'])]
    bad_faces = surf['shell_cell_id'][np.isin(surf_faces, bad_points).sum(axis=1) >= 1]
    shell_holes = shell.extract_cells(bad_faces, invert=True).extract_geometry()


    v = np.asarray(shell_holes.points, dtype=np.float64)

    f = np.asarray(shell_holes.faces.reshape(-1, 4)[:, 1:], dtype=np.int32)  # triangle faces
    v2, f2 = pymeshfix.clean_from_arrays(v, f)

    repaired = pv.PolyData(v2, np.hstack([np.full((len(f2), 1), 3), f2]).ravel())
    #print(f'Is manifold ({region_id}):', repaired.is_manifold)

    # check all surf cells that were not repaired are still there
    print("\nCheck manifold cells have not moved")
    orig_cells = find_corresponding_cells(repaired, shell_holes, raise_error=True)
    repaired['repaired'] = np.ones(repaired.n_cells)
    repaired['repaired'][orig_cells] = 0

    # check all interface cells are still there
    print("Check interface cells have not moved")
    inter_cells = find_corresponding_cells(repaired, mesh_tri.extract_cells(np.where(mesh_tri['region_id']==-3)), raise_error=True)
    repaired['region_id'] = np.full(repaired.n_cells, region_id)
    repaired['region_id'][inter_cells] = -3

    shells[region_id] = repaired

    # EVALUATE REPAIR SIZE AND LOCATION
    print('\nEvaluate repair size and proximity:')
    repaired_cartilage_surf = repaired.extract_cells(np.where(repaired['region_id']==-2)[0]).extract_geometry()
    repaired_cartilage_surf['repaired_cartilage_surf_id'] = np.arange(repaired_cartilage_surf.n_points)
    repaired_patch = repaired_cartilage_surf.extract_cells(np.where(repaired_cartilage_surf['repaired']==1)[0]).extract_geometry()

    # measure proximity to cartilage boundary
    repaired_cartilage_surf_edge = get_boundary(repaired_cartilage_surf)
    cartilage_edge_dists = gdist.compute_gdist(
        repaired_cartilage_surf.points.astype(np.float64),
        repaired_cartilage_surf.faces.reshape(-1, 4)[:, 1:].astype(np.int32),
        source_indices=repaired_cartilage_surf_edge['repaired_cartilage_surf_id'].astype(np.int32), 
    )
    dist_max = cartilage_edge_dists.max()
    repaired_dist_max = cartilage_edge_dists[repaired_patch['repaired_cartilage_surf_id']].max()
    repaired_proxmimity = repaired_dist_max / dist_max
    print(f'\tProximity of repaired cartilage surface area to boundary: {repaired_dist_max:.2f} mm   ({repaired_proxmimity:.4f} ; 0: boundary, 1:centre)')
    if max_loc and repaired_proxmimity > max_loc:
        raise AssertionError(f"{shell_map[region_id]} repair location too far from boundary")

    # measure size of repair region
    repaired_A = repaired_patch.area / repaired_cartilage_surf.area
    print(f'\tArea of cartilage surface repaired:                       {repaired_patch.area:.2f} mm^2 ({repaired_A*100:.2f}%)')
    if max_area and repaired_A > max_area:
        raise AssertionError(f"{shell_map[region_id]} repair area is too large")

    print('\nComplete')

print('\nCHECK AND SAVE...\n')
# check if they are now manifold
for region_id, shell in shells.items():
    print(f'{shell_map[region_id]} is manifold: ', shell.is_manifold)
    if require_manifold and not shell.is_manifold:
        raise KeyError(f"{shell_map[region_id]} is not manifold")
    print('Saving...')

    if    output_filename: savename = output_filename
    else: savename = f'{shell_map[region_id].lower()}{run_id}.vtp'
    shell.save(output_path / savename)

print('\nComplete\n')