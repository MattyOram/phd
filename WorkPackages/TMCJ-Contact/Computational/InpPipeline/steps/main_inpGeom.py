"""Create mesh-only Abaqus input files for later setup through Abaqus/CAE."""

import numpy as np
import pyvista as pv
from pathlib import Path
import json
import sys

from phd_helpers.paths import(
get_subject_stl_path, get_bone_inertia, transform_mesh, get_relative_transform_new_basis, get_bone_transforms,
pose2idCMC, linear_to_quadratic_mesh, transform_points
)

from phd_helpers.AbaqusPreprocessing2 import (
position_mc1_tpm, bone_surface_patch_nodes, AbaqusInpBuilder, rotate_mesh, translate_mesh
)

#####################################################
# --------------------- PATHS --------------------- #

args = sys.argv

output_dir = Path(args[1])
param_path = args[2] # path to param file in loop params
sub_path = Path(args[3]) # path to subject folder in output dir
tpm_path = args[4]
mc1_path = args[5]
run_id = sys.argv[6]
run_id_mesh = sys.argv[7]

with open(param_path, "r") as f:
    params = json.load(f)

sub = sub_path.name
subject, sideL = sub[:-1], sub[-1]
stl_path = get_subject_stl_path(subject, sideL)


# save dir
savepath_inp = output_dir / f'inpFiles/{sub}/inp' 
savepath_inp.mkdir(parents=True, exist_ok=True)

overwrite = params['overwrite']
# --------------------- PATHS --------------------- #
#####################################################

##########################################################
# --------------------- PARAMETERS --------------------- #

misalign_t = params['misalign_t'] 
misalign_R = params['misalign_R']

target_dist = params['target_dist'] # gap between cartilage at start of simulation

tpm_patch_params = params['tpm_patch_params']
mc1_patch_params = params['mc1_patch_params']


# ELEMENT TYPES
bone_element_type = params['element_type']
if  'C3D10' in bone_element_type:
    element_order = 'quad'
elif 'C3D4' in bone_element_type:
    element_order = 'linear'
else:
    raise AssertionError(f'Element type not recognised: {bone_element_type}')

cartilage_element_type = params['element_type'] + params['cartilage_element_suffix']


# REGION IDs
bone_vol_id = params['bone_vol_id']
cartilage_vol_id = params['cartilage_vol_id']
cartilage_surf_id = params['cartilage_surf_id']

# --------------------- PARAMETERS --------------------- #
##########################################################


###########################################################
# --------------------- LOAD MESHES --------------------- #

print('\nLOADING MESHES...\n')

poses = params['poses']
use_neutral11 = params['use_neutral11']
mc1_centroid, _, mc1_axes = get_bone_inertia(stl_path, 'mc1') # mc1 centroid and inertial axes for alignment

# load meshes and align with mc1 inertial axes
tpm_mesh_neu = pv.read(tpm_path)
mc1_mesh = pv.read(mc1_path)

tpm_mesh_neu = transform_mesh(tpm_mesh_neu, mc1_axes, mc1_centroid, inverse=True)
mc1_mesh = transform_mesh(mc1_mesh, mc1_axes, mc1_centroid, inverse=True)

# for centre of rotation for re-alignment of joint to match instron misalignment
tpm_centroid, _, _ = get_bone_inertia(stl_path, 'tpm')
tpm_centroid_neu = transform_points(tpm_centroid, mc1_axes, mc1_centroid, inverse=True)[0]

if element_order == 'quad':
    print("Converting to quadratic elements")
    tpm_mesh_neu = linear_to_quadratic_mesh(tpm_mesh_neu)
    mc1_mesh = linear_to_quadratic_mesh(mc1_mesh)

print('Complete\n')

# --------------------- LOAD MESHES --------------------- #
###########################################################

print(f'\n\n---------------- PREPARING INPUT FILES... ----------------\n\n')

#############################################################
# --------------------- PREPROCESSING --------------------- #

# get mesh centroid for reference point for boundary condition
print("Computing mc1 RP location (mean of points)")
mc1_RP_loc = mc1_mesh.points.mean(axis=0)

# get nodes to form surface for coupling with reference point
# - nodes aren't used anymore, use mesh.cell_data['bc_patch] to directly create surface 
# (requires: only_full_face_nodes=True)
print("Computing mc1 bone surface patch for RP coupling")
bone_surface_patch_nodes(mc1_mesh, mc1_patch_params[1], mc1_patch_params[0], only_full_face_nodes=True)


for pose in poses:

    print(f"\nMESH PREPROCESSING...   ({pose.upper()})\n")

    # position trapezium
    print('Positioning trapezium')
    pose_id = pose2idCMC(pose)
    try: # if subject has alternate neutral that will be used otherwise Exception and use default neutral
        if pose == 'neutral' and not use_neutral11:
            raise
        transforms = get_bone_transforms(pose_id, stl_path)
        R, t = get_relative_transform_new_basis(transforms, 'tpm', 'mc1', mc1_centroid, mc1_axes)
    except:
        R, t = np.eye(3), np.zeros(3)

    tpm_mesh = transform_mesh(tpm_mesh_neu, R, t)

    # re-alignment of joint to match instron misalignment
    # dont think order of t and R matters just need centre of rotation to be constant
    #tpm_centroid = transform_points(tpm_centroid_neu, R, t)[0] # keep centre of rotation at tpm_centroid_neu
    tpm_mesh = rotate_mesh(tpm_mesh, tpm_centroid_neu, misalign_R[0], misalign_R[1], misalign_R[2])
    tpm_mesh = translate_mesh(tpm_mesh, misalign_t[0], misalign_t[1], misalign_t[2])

    # position tpm and metacarpal chosen distance apart (moves tpm)
    position_mc1_tpm(mc1_mesh, tpm_mesh, target_dist, raise_error=True)

    # get mesh centroid for reference point for boundary conditions
    print("Computing tpm RP location (mean of points)")
    tpm_RP_loc = tpm_mesh.points.mean(axis=0)

    # get nodes to form surface for coupling with reference point
    # - nodes aren't used anymore, use mesh.cell_data['bc_patch] to directly create surface 
    # (requires: only_full_face_nodes=True)
    print("Computing tpm bone surface patch for RP coupling")
    bone_surface_patch_nodes(tpm_mesh, tpm_patch_params[1], tpm_patch_params[0], only_full_face_nodes=True)

    print('\nComplete\n')

# --------------------- PREPROCESSING --------------------- #
#############################################################


###################################################################
# --------------------- BUILDING INPUT FILE --------------------- #

    print("BUILDING INPUT...")

    b = AbaqusInpBuilder()

    # SET ELEMENT TYPE AND REGION IDS
    b.set_element_types(bone_element_type, cartilage_element_type)
    b.set_region_ids(bone_vol_id, cartilage_vol_id)

    # PARTS
    b.add_part_from_vtu("tpm", tpm_mesh, instance_name="tpm_INST", mode='mesh')
    b.add_part_from_vtu("mc1", mc1_mesh, instance_name="mc1_INST", mode='mesh')

    # SURFACES FOR BONE CONSTRAINTS
    b.add_surface_from_cell_data("tpm", "bc_patch", 1, "tpm_PATCH_SURF")
    b.add_surface_from_cell_data("mc1", "bc_patch", 1, "mc1_PATCH_SURF")

    # RPs FOR SURFACE PATCH COUPLING
    b.create_reference_point("RP_tpm", node_id=9000001, xyz=tpm_RP_loc)
    b.create_reference_point("RP_mc1", node_id=9000002, xyz=mc1_RP_loc)

    # CARTILAGE SURFACES FOR CONTACT
    b.add_surface_from_cell_data("tpm", "region_id", cartilage_surf_id, "tpm_CART_SURF")
    b.add_surface_from_cell_data("mc1", "region_id", cartilage_surf_id, "mc1_CART_SURF")

    # WRITE INPUT FILE
    print(f"Writing input file - ({pose.upper()})")
    inp_name = f'{run_id_mesh}-{pose}-{run_id}'
    inp_dir = savepath_inp / inp_name
    inp_dir.mkdir(parents=True, exist_ok=True)
    inp_file = inp_dir / (inp_name+'-geom.inp')
    if inp_file.exists() and not overwrite:
        raise FileExistsError(f'{inp_file} already exists and overwrite is false')
    else:
        b.write_input_file(inp_file)

    print("\nComplete")

# --------------------- BUILDING INPUT FILE --------------------- #
###################################################################
