import numpy as np
import pyvista as pv
from pathlib import Path
import json
import sys

from phd_helpers.paths import(
get_subject_stl_path, get_bone_inertia, transform_mesh, get_relative_transform_new_basis, get_bone_transforms,
pose2idCMC, quadratic_to_linear_mesh, linear_to_quadratic_mesh
)

from phd_helpers.AbaqusPreprocessing import position_mc1_tpm, bone_surface_patch_nodes, AbaqusInpBuilder


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

# main #
# ABAQUS CLI INPUTS
#inp_file = inp_path # path to input file i.e inputs/input1.inp
#job_name = inp_path.name.split('.')[0]   # saves files to this path i.e. inputs/input1
#CPUs = 8 # 4(=8tokens) performance won't get much better after 8(=12tokens)
#setting = 'interactive'

# PRE-PROCESSING #
#save_mesh = params['save_meshes']
#if save_mesh:
#    savepath_mesh = output_dir / f'inpFiles/{sub}/mesh' 
#    savepath_mesh.mkdir(parents=True, exist_ok=True)

target_dist = params['target_dist'] # gap between cartilage at start of simulation

tpm_patch_params = params['tpm_patch_params']
mc1_patch_params = params['mc1_patch_params']


# ELEMENT TYPES
bone_element_type = params['element_type']
if bone_element_type == 'C3D10':
    element_order = 'quad'
elif bone_element_type == 'C3D4':
    element_order = 'linear'
else:
    raise AssertionError(f'Element type not recognised: {bone_element_type}')

cartilage_element_type = params['element_type'] + params['cartilage_element_suffix']


# BONE PROPERTIES
bone_material = params['bone_material']
bone_density = params['bone_density']

# CARTILAGE PROPERTIES
cartilage_material = params['cartilage_material']
cartilage_density = params['cartilage_density']

cartilage_friction = params['cartilage_friction']

# REGION IDs
bone_vol_id = params['bone_vol_id']
cartilage_vol_id = params['cartilage_vol_id']
cartilage_surf_id = params['cartilage_surf_id']

# DISPLACEMENT / FORCE LIMITS
mc1_disp_x  = params['mc1_disp_x']
#Forces = [10.0, 20.0]   # refine step time to hit these forces 
                            # - would need to set user defined DT REFINEMENT - not worth it right now
max_force = params['max_force']

# STEP PARAMS
total_step_time = params['total_step_time']
initial_increment = params['initial_increment']
min_increment = params['min_increment']
max_increment = params['max_increment']

step_type = params['step_type']
nlgeom = params['nlgeom']
unsymm = params['unsymm']
convert_sdi = params['convert_sdi']
extrapolation = params['extrapolation']

equil_iters = params['equil_iters']
sdi_iters = params['sdi_iters']
increment_attemps = params['increment_attemps']

# --------------------- PARAMETERS --------------------- #
##########################################################


###########################################################
# --------------------- LOAD MESHES --------------------- #

print('\nLOADING MESHES...\n')

poses = params['poses']
mc1_centroid, _, mc1_axes = get_bone_inertia(stl_path, 'mc1') # mc1 centroid and inertial axes for alignment

# load meshes and align with mc1 inertial axes
tpm_mesh_neu = pv.read(tpm_path)
mc1_mesh = pv.read(mc1_path)

tpm_mesh_neu = transform_mesh(tpm_mesh_neu, mc1_axes, mc1_centroid, inverse=True)
mc1_mesh = transform_mesh(mc1_mesh, mc1_axes, mc1_centroid, inverse=True)

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
mc1_patch_nodes = bone_surface_patch_nodes(mc1_mesh, mc1_patch_params[1], mc1_patch_params[0], only_full_face_nodes=True)

#if save_mesh:
#    # save input mesh (with bc_patch array)(linear versions) # wastes memory but makes life easier?
#    print('Saving input mesh ("mc1-positioned.vtu) (with linear elements)')
#    quadratic_to_linear_mesh(mc1_mesh).save(savepath_mesh / f"{run_id_mesh}-mc1_positioned.vtu")

for pose in poses:

    print(f"\nMESH PREPROCESSING...   ({pose.upper()})\n")

    # position trapezium
    print('Positioning trapezium')
    pose_id = pose2idCMC(pose)
    try: # if subject has alternate neutral that will be used otherwise Exception and use default neutral
        transforms = get_bone_transforms(pose_id, stl_path)
        R, t = get_relative_transform_new_basis(transforms, 'tpm', 'mc1', mc1_centroid, mc1_axes)
    except:
        R, t = np.eye(3), np.zeros(3)

    tpm_mesh = transform_mesh(tpm_mesh_neu, R, t)

    # position tpm and metacarpal chosen distance apart (moves tpm)
    position_mc1_tpm(mc1_mesh, tpm_mesh, target_dist, raise_error=True)

    # get mesh centroid for reference point for boundary conditions
    print("Computing tpm RP location (mean of points)")
    tpm_RP_loc = tpm_mesh.points.mean(axis=0)

    # get nodes to form surface for coupling with reference point
    # - nodes aren't used anymore, use mesh.cell_data['bc_patch] to directly create surface 
    # (requires: only_full_face_nodes=True)
    print("Computing tpm bone surface patch for RP coupling")
    tpm_patch_nodes = bone_surface_patch_nodes(tpm_mesh, tpm_patch_params[1], tpm_patch_params[0], only_full_face_nodes=True)

    #if save_mesh:
    #    # save input mesh (positioned with bc_patch array)(linear versions) # wastes memory but makes life easier?
    #    print(f'Saving input mesh ("tpm-{pose}.vtu") (with linear elements)')
    #    quadratic_to_linear_mesh(tpm_mesh).save(savepath_mesh / f"{run_id_mesh}-tpm_{pose}.vtu") 

    print('\nComplete\n')

# --------------------- PREPROCESSING --------------------- #
#############################################################


###################################################################
# --------------------- BUILDING INPUT FILE --------------------- #

    print("BUILDING INPUT...")

    b = AbaqusInpBuilder()

    # SET ELEMENT TYPE AND REGION IDS
    b.set_element_types(bone_element_type, cartilage_element_type)
    b.set_region_ids(bone_vol_id, cartilage_vol_id, cartilage_surf_id)

    # MATERIALS
    b.create_material("BONE", "elastic", bone_material, density=bone_density)
    b.create_material("CARTILAGE", "neo_hooke", cartilage_material, density=cartilage_density)

    # PARTS
    b.add_part_from_vtu("tpm", tpm_mesh, instance_name="tpm_INST", mode='mesh')
    b.add_part_from_vtu("mc1", mc1_mesh, instance_name="mc1_INST", mode='mesh')

    # ELSETS
    b.create_elset_from_region("tpm", bone_vol_id, "tpm_BONE")
    b.create_elset_from_region("tpm", cartilage_vol_id, "tpm_CARTILAGE")
    b.create_elset_from_region("mc1", bone_vol_id, "mc1_BONE")
    b.create_elset_from_region("mc1", cartilage_vol_id, "mc1_CARTILAGE")

    # SOLID SECTIONS
    b.create_solid_section("tpm", "tpm_BONE", "BONE")
    b.create_solid_section("tpm", "tpm_CARTILAGE", "CARTILAGE")
    b.create_solid_section("mc1", "mc1_BONE", "BONE")
    b.create_solid_section("mc1", "mc1_CARTILAGE", "CARTILAGE")

    # SURFACES FOR BONE CONSTRAINTS
    #b.add_surface_from_nodes("tpm", tpm_patch_nodes, "tpm_PATCH_NODES", "tpm_PATCH_SURF")
    #b.add_surface_from_nodes("mc1", mc1_patch_nodes, "mc1_PATCH_NODES", "mc1_PATCH_SURF")
    # SURFACES FOR BONE CONSTRAINTS
    b.add_surface_from_cell_data("tpm", "bc_patch", 1, "tpm_PATCH_SURF")
    b.add_surface_from_cell_data("mc1", "bc_patch", 1, "mc1_PATCH_SURF")

    # RPs FOR SURFACE PATCH COUPLING
    b.create_reference_point("RP_tpm", node_id=9000001, xyz=tpm_RP_loc)
    b.create_reference_point("RP_mc1", node_id=9000002, xyz=mc1_RP_loc)

    # RP-SURFACE COUPLING
    b.add_rp_surface_coupling("CP_tpm", "RP_tpm", "tpm", "tpm_PATCH_SURF", coupling_type="KINEMATIC")
    b.add_rp_surface_coupling("CP_mc1", "RP_mc1", "mc1", "mc1_PATCH_SURF", coupling_type="KINEMATIC")

    # CARTILAGE SURFACES FOR CONTACT
    #b.add_surface_from_region_id("tpm", cartilage_surf_id, "tpm_CART_SURF")
    #b.add_surface_from_region_id("mc1", cartilage_surf_id, "mc1_CART_SURF")
    b.add_surface_from_cell_data("tpm", "region_id", cartilage_surf_id, "tpm_CART_SURF")
    b.add_surface_from_cell_data("mc1", "region_id", cartilage_surf_id, "mc1_CART_SURF")


    # CONTACT
    b.set_contact(
        interaction_name = "CART_CONTACT",
        friction = cartilage_friction,
        surfaces = [("tpm", "tpm_CART_SURF", "mc1", "mc1_CART_SURF")], # [(part1, surface1, part2, surface2)]
    )

    # STEP
    step_name = "MOVE"
    b.create_step(
        step_name = step_name,
        step_type = step_type,
        initial_increment_size = initial_increment,     # starting point is 0.01mm from contact
        total_step_size = total_step_time, # set to total displacement so that increment params don't have to change with displacement
        min_increment_size = min_increment,
        max_increment_size = max_increment, 
        nlgeom = nlgeom,
        convert_sdi = convert_sdi,
        unsymm=unsymm,
        extrapolation=extrapolation
    )

    # CONTROLS
    b.add_control_lines(
        step_name,
        [
        '*CONTROLS, PARAMETERS=TIME INCREMENTATION',
        f',,,{equil_iters},,,{sdi_iters},{increment_attemps},,,,,'
    ]
    )

    # BOUNDARY CONDITIONS
    b.set_bc(
        step_name,
        node_set='RP_tpm',
        op='MOD', # MOD-change current BCs, NEW-replace all current BCs
        U1=0, U2=0, U3=0, UR1=0, UR2=0, UR3=0
    )
    b.set_bc(
        step_name,
        op='MOD',
        node_set='RP_mc1',
        U1=mc1_disp_x, U2=0, U3=0, UR1=0, UR2=0, UR3=0
    )

    # HISTORY OUTPUTS
    b.add_history_output_lines(
        step_name,
        [
        "*OUTPUT, HISTORY, SENSOR, NAME=mc1_RF1",
        "*NODE OUTPUT, NSET=RP_mc1",
        "RF1"
    ]
    )
    b.add_history_output_lines(
        step_name,
        [
        "*OUTPUT, HISTORY, OP=ADD",
        "*CONTACT OUTPUT",
        "CAREA"
        ]
    )

    # FIELD OUTPUTS
    b.add_field_output_lines(
        step_name,
        [
        "*OUTPUT, FIELD",
        "**",
        "*NODE OUTPUT",
        "U, COORD",
        "**",
        "*ELEMENT OUTPUT, POSITION=INTEGRATION POINTS",
        "S, LE, COORD",
        "**",
        "*CONTACT OUTPUT",
        "CSTRESS, CDISP, CSTATUS, CNAREA"
    ])

    # STEP CONTROLS
    #for F in Forces:
    #    b.add_step_control_lines(
    #        step_name,
    #        [
    #        f"*STEP CONTROL, NAME=REFINE_ON_RF{F:.0f}, ACTION=CONTINUE, DT REFINEMENT=YES", # need to set user defined DT REFINEMENT - not worth it right now
    #        f"mc1_RF1, AbsMax, {float(F)}",
    #    ])


    b.add_step_control_lines(
        step_name,
        [
        f"*STEP CONTROL, NAME=STOP_ON_RF{max_force:.0f}, ACTION=END ANALYSIS, DT REFINEMENT=AUTO",
        f"mc1_RF1, AbsMax, {float(max_force)}",
    ])

    # WRITE INPUT FILE
    print(f"Writing input file - ({pose.upper()})")
    inp_name = f'{run_id_mesh}-{pose}-{run_id}'
    inp_dir = savepath_inp / inp_name
    inp_dir.mkdir(parents=True, exist_ok=True)
    inp_file = inp_dir / (inp_name+'.inp')
    if inp_file.exists() and not overwrite:
        raise(f'{inp_file} already exists and overwrite is false')
    else:
        b.write_input_file(inp_file)

    print("\nComplete")

# --------------------- BUILDING INPUT FILE --------------------- #
###################################################################