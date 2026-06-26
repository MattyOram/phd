import numpy as np
import pyvista as pv
from pathlib import Path
from trimesh.interfaces import blender
import sys
import json

from phd_helpers.paths import(
get_subject_stl_path, get_bone_inertia, transform_mesh, transform_points, get_relative_transform_new_basis, get_bone_transforms,
pose2idCMC
)
from phd_helpers.CartilageGeneration import get_trimesh, get_pvmesh, get_outward_normal_mask, flip_faces
from makemanifold import make_manifold
from multipart3mf import mesh_files_to_multi_part_3mf

##############################################################
# --------------------- SELECT SUBJECT --------------------- #

args = sys.argv

output_dir = Path(args[1])
param_path = args[2] # path to param file in loop params
sub_path = Path(args[3]) # path to subject folder in output dir
tpm_neu_path = args[4]
mc1_path = args[5]
mesh_id = sys.argv[6]

with open(param_path, "r") as f:
    params = json.load(f)
params_man = params['manifold']
params_print = params['printJig']

sub = sub_path.name
subject, sideL = sub[:-1], sub[-1]

savepath = output_dir / f'meshes/{subject+sideL}'
savepath.mkdir(parents=True, exist_ok=True)


# Jigs
jig_path = params_print['jig_path']
jig_mc1 = pv.read(jig_path)
jig_tpm_neu = pv.read(jig_path)

# --------------------- SELECT SUBJECT --------------------- #
##############################################################


##########################################################
# --------------------- PARAMETERS --------------------- #

poses = params_print['poses']

# make manifold #
max_area = params_man['max_area']
max_loc = params_man['max_loc']

# tpm #
tpm_disk_depth = params_print['tpm_disk_depth']    # depth of jig disk that sits on steel jig (mm)
tpm_offset = params_print['tpm_offset']        # offset of top of jig from tpm centroid (+ive = more overlap)

# mc1 #
mc1_disk_depth = params_print['mc1_disk_depth']    # depth of jig disk that sits on steel jig (mm)
#mc1_halffraction = 0.5 # proportion from centroid to bottom of mc1 to cut off - ••••• CHANGE TO CONSTANT MC1 TIP X HEIGHT?
mc1_potrusion = params_print['mc1_potrusion'] # how far mc1 should potrude out of jig

# --------------------- PARAMETERS --------------------- #
##########################################################


###########################################################
# --------------------- LOAD MESHES --------------------- #

print('\nLOADING MESHES...')

# get subject transform data
stl_path = get_subject_stl_path(subject, sideL)
mc1_centroid, _, mc1_axes = get_bone_inertia(stl_path, 'mc1') # mc1 centroid and inertial axes for alignment
tpm_centroid_neu, _, _ = get_bone_inertia(stl_path, 'tpm') # centroid for alignment
tpm_centroid_neu = transform_points(tpm_centroid_neu, mc1_axes, mc1_centroid, inverse=True)[0]

# load meshes - extract trangle surface mesh
mesh_mc1 = pv.read(mc1_path).extract_cells_by_type(5)
mesh_tpm_neu = pv.read(tpm_neu_path).extract_cells_by_type(5)

# transform basis
mesh_mc1 = transform_mesh(mesh_mc1, mc1_axes, mc1_centroid, inverse=True)
mesh_tpm_neu = transform_mesh(mesh_tpm_neu, mc1_axes, mc1_centroid, inverse=True)

# extract bone and cartilage
bone_mc1 = mesh_mc1.extract_cells(mesh_mc1['region_id'] != -2).extract_surface(algorithm=None)
bone_tpm_neu = mesh_tpm_neu.extract_cells(mesh_tpm_neu['region_id'] != -2).extract_surface(algorithm=None)
cartilage_mc1 = mesh_mc1.extract_cells(mesh_mc1['region_id'] != -1).extract_surface(algorithm=None)
cartilage_tpm_neu = mesh_tpm_neu.extract_cells(mesh_tpm_neu['region_id'] != -1).extract_surface(algorithm=None)

bone_mc1 = make_manifold(bone_mc1, -1, max_area, max_loc, f'bone-mc1-{mesh_id}')
bone_tpm_neu = make_manifold(bone_tpm_neu, -1, max_area, max_loc, f'bone-tpm-{mesh_id}')
cartilage_mc1 = make_manifold(cartilage_mc1, -2, max_area, max_loc, f'cart-mc1-{mesh_id}')
cartilage_tpm_neu = make_manifold(cartilage_tpm_neu, -2, max_area, max_loc, f'cart-tpm-{mesh_id}')

# This next step of orienting normals might not be needed, pymeshfix probs takes care of it
def orient_normals(mesh):
    mesh.compute_normals(auto_orient_normals=True, inplace=True)
    # check if normals point outwards or inwards and if not flip them
    if not get_outward_normal_mask(mesh.cell_centers().points, mesh.cell_normals, mesh).any():
        mesh = flip_faces(mesh, np.arange(mesh.n_cells))
        print('flipped faces')
    return mesh

bone_mc1 = orient_normals(bone_mc1)
bone_tpm_neu = orient_normals(bone_tpm_neu)
cartilage_mc1 = orient_normals(cartilage_mc1)
cartilage_tpm_neu = orient_normals(cartilage_tpm_neu)



print('Complete\n')

# --------------------- LOAD MESHES --------------------- #
###########################################################



print('\n\n\n---------------- JOINING mc1 to jig ----------------\n\n')

##############################################################
# --------------------- CHOP & CAP mc1 --------------------- #
# chop mc1 and cap open end so that it can be used for boolean union
#mc1_halfoffset = np.min(bone_mc1.points[:, 0]) * mc1_halffraction # offset of centroid(origin) (-ive)
mc1_halfoffset = np.min(bone_mc1.points[:, 0]) + mc1_potrusion
mc1_offset = mc1_disk_depth + mc1_halfoffset # for steel attachment

cube_xlength, cube_ylength, cube_zlength = 50, 20, 20 # big enough to engulf mc1 section to be chopped
cube_centre = ((mc1_disk_depth/2)+(mc1_halfoffset)+(cube_xlength/2), 0, 0)
cube = pv.Cube(center=cube_centre, x_length=cube_xlength, y_length=cube_ylength, z_length=cube_zlength).triangulate()

# cap mc1 mesh
mc1_tmesh = blender.boolean([get_trimesh(bone_mc1), get_trimesh(cube)], operation='difference')

# --------------------- CHOP & CAP mc1 --------------------- #
##############################################################

##########################################################################
# --------------------- ORIENT & POSITION JIG MESH --------------------- #

jig_mc1.points *= np.array([-1, 1, 1])
jig_mc1.flip_faces(inplace=True)
jig_mc1.points += np.array([mc1_halfoffset, 0, 0])
jig_tmesh_mc1 = get_trimesh(jig_mc1)

# --------------------- ORIENT & POSITION JIG MESH --------------------- #
##########################################################################

################################################################################
# --------------------- CHECK FOR CARTILAGE INTERFERENCE --------------------- #

# check meshes
if not jig_mc1.is_manifold: raise AssertionError('Jig mc1 is not manifold')
if not jig_tmesh_mc1.is_watertight: raise AssertionError('Jig mc1 is not watertight')
if not bone_mc1.is_manifold: raise AssertionError('mc1 bone is not manifold')
if not mc1_tmesh.is_watertight: raise AssertionError('mc1 bone is not watertight')

# check intersection with jig
intersection_jig = get_trimesh(cartilage_mc1).intersection(jig_tmesh_mc1) # Check intersection
if intersection_jig.volume:# or intersection_steel_jig.volume:
    raise ValueError(f'mc1 cartilage intersects jig')

# --------------------- CHECK FOR CARTILAGE INTERFERENCE --------------------- #
################################################################################

###############################################################
# --------------------- JOIN BONE & JIG --------------------- #

# boolean union
print_tmesh_mc1 = blender.boolean([jig_tmesh_mc1, mc1_tmesh], operation='union')
if not print_tmesh_mc1.is_watertight: raise AssertionError('mc1 union is not watertight')
print_mesh_mc1 = get_pvmesh(print_tmesh_mc1)
if not print_mesh_mc1.is_manifold: raise AssertionError('mc1 union is not manifold')

# saving mesh
bone_savepath = savepath / f'mc1_boneJig-{mesh_id}.vtp'
cartilage_savepath = savepath / f'mc1_cartilage-{mesh_id}.vtp'
print_mesh_mc1.save(bone_savepath)
cartilage_mc1.save(cartilage_savepath)

# save to .3mf file
print("\nWriting to .3mf file...")
mesh_list = [
    (bone_savepath, 'boneJig'),
    (cartilage_savepath, 'cartilage')
]
mesh_files_to_multi_part_3mf(mesh_list, savepath / f'mc1Print-{mesh_id}.3mf')

print('Complete\n')

# --------------------- JOIN BONE & JIG --------------------- #
###############################################################





print('\n\n---------------- JOINING tpm to jig ----------------\n\n')

for pose in poses:
    print(pose.upper(), '\n')

    #############################################################
    # --------------------- TRANSFORM tpm --------------------- #

    pose_id = pose2idCMC(pose)
    try: # if subject has alternate neutral that will be used otherwise Exception and use default neutral
        transforms = get_bone_transforms(pose_id, stl_path)
        R, t = get_relative_transform_new_basis(transforms, 'tpm', 'mc1', mc1_centroid, mc1_axes)
    except:
        R, t = np.eye(3), np.zeros(3)

    bone_tpm = transform_mesh(bone_tpm_neu, R, t)
    cartilage_tpm = transform_mesh(cartilage_tpm_neu, R, t)
    tpm_centroid = transform_points(tpm_centroid_neu, R, t)[0]

    tpm_tmesh = get_trimesh(bone_tpm) # convert to trimesh for booleans

    # --------------------- TRANSFORM tpm --------------------- #
    #############################################################

    ##########################################################################
    # --------------------- ORIENT & POSITION JIG MESH --------------------- #

    jig_tpm = jig_tpm_neu.copy(deep=True)
    jig_tpm.points += np.array([tpm_centroid[0]+tpm_offset, 0, 0])
    jig_tmesh_tpm = get_trimesh(jig_tpm)

    # --------------------- ORIENT & POSITION JIG MESH --------------------- #
    ##########################################################################

    ################################################################################
    # --------------------- CHECK FOR CARTILAGE INTERFERENCE --------------------- #

    # check meshes
    if not jig_tpm.is_manifold: raise AssertionError('Jig tpm is not manifold')
    if not jig_tmesh_tpm.is_watertight: raise AssertionError('Jig tpm is not watertight')
    if not bone_tpm.is_manifold: raise AssertionError('tpm bone is not manifold')
    if not tpm_tmesh.is_watertight: raise AssertionError('tpm bone is not watertight')

    # check intersection with jig
    intersection_jig = get_trimesh(cartilage_tpm).intersection(jig_tmesh_tpm) # Check intersection

    if intersection_jig.volume: #or intersection_steel_jig.volume:
        print("\tSlicing cartilage...")
        cartilage_tpm_tmesh = blender.boolean([get_trimesh(cartilage_tpm), jig_tmesh_tpm], operation='difference')
        cartilage_tpm = get_pvmesh(cartilage_tpm_tmesh)

        if not cartilage_tpm_tmesh.is_watertight: raise AssertionError('tpm cartilage is not watertight')
        if not cartilage_tpm.is_manifold: raise AssertionError('tpm cartilage is not manifold')

    # --------------------- CHECK FOR CARTILAGE INTERFERENCE --------------------- #
    ################################################################################

    ###############################################################
    # --------------------- JOIN BONE & JIG --------------------- #

    # boolean union
    print_tmesh_tpm = blender.boolean([jig_tmesh_tpm, tpm_tmesh], operation='union')
    print_mesh_tpm = get_pvmesh(print_tmesh_tpm)
    if not print_tmesh_tpm.is_watertight: raise AssertionError(f'\ntpm BoneJig union is not watertight')
    if not print_mesh_tpm.is_manifold: raise AssertionError(f'\ntpm BoneJig union is not manifold')

    # saving mesh
    bone_savepath = savepath / f'tpm_boneJig_{pose}-{mesh_id}.vtp'
    cartilage_savepath = savepath / f'tpm_cartilage_{pose}-{mesh_id}.vtp'
    print_mesh_tpm.save(bone_savepath)
    cartilage_tpm.save(cartilage_savepath)

    # save to .3mf file
    print("\nWriting to .3mf file...")
    mesh_list = [
        (bone_savepath, 'boneJig'),
        (cartilage_savepath, 'cartilage')
    ]
    mesh_files_to_multi_part_3mf(mesh_list, savepath / f'tpmPrint_{pose}-{mesh_id}.3mf')

    print('Complete\n')

    # --------------------- JOIN BONE & JIG --------------------- #
    ###############################################################


