import numpy as np
import pandas as pd
import pyvista as pv
import trimesh
from tqdm import tqdm
from scipy.spatial.distance import cdist
import pymeshfix

from phd_helpers.paths import pose2idCMC, get_bone_transforms, get_relative_transform_new_basis, transform_mesh, get_mc_lower_cell_ids

def get_volume_seed(mesh, instep=0.01):
    """Returns point contained within mesh volume (steps in from central point by instep value)"""
    central_node = mesh.find_closest_point(np.array(mesh.center))
    central_point = mesh.points[central_node]
    central_normal = mesh.point_normals[central_node]

    seed = central_point - instep*central_normal
    # check its inside
    probe = pv.PolyData(seed)
    if not bool(probe.select_interior_points(mesh, check_surface=True)['selected_points'][0]):
        print('Warning: seed not inside. Can try reducing offset')
    return seed

def get_trimesh(pv_mesh: pv.PolyData, n_verts=3, process=False):
    """convert pyvista mesh to trimesh"""
    vertices = pv_mesh.points
    faces = pv_mesh.faces.reshape(-1, n_verts+1)[:, 1:]
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=process)

def get_pvmesh(tmesh: trimesh.Trimesh, n_verts=3):
    vertices = tmesh.vertices
    faces = tmesh.faces
    faces_pv = np.hstack(
        np.c_[np.full(len(faces), n_verts), faces]
    )
    return pv.PolyData(vertices, faces_pv)


################# MESH CHECKS #################
def outward_normals(mesh, eps=1e-5, return_check=False):
    """Checks that all face normals on a watertight mesh point outwards"""
    normals = mesh.compute_normals().cell_data['Normals'] # face normals
    centres = mesh.cell_centers().points
    point_cloud = pv.PolyData(centres - normals * eps)
    check = point_cloud.select_interior_points(mesh, check_surface=True)['selected_points'].all()
    print(
        'All normals point outwards:     ', 
        check
        )
    if return_check:
        return check

def mesh_checks(mesh, raise_error=False):

    # trimesh checks
    tm_mesh = get_trimesh(mesh)
    tm_watertight = tm_mesh.is_watertight
    tm_winding = tm_mesh.is_winding_consistent
    print('Trimesh checks:')
    print('Mesh is watertight              ', tm_watertight)
    print('Mesh is winding consistent      ', tm_winding)

    print('\nPyVista checks:')
    pv_manifold = mesh.is_manifold
    # verify that there are no open edges
    print('Mesh is manifold (no open edges)', pv_manifold)

    # verify that all normals point outwards
    out_normals = outward_normals(mesh, return_check=True)

    # check for duplicate faces
    sorted_faces = np.sort(mesh.faces.reshape(-1, 4)[:, 1:], axis=1) # sort so that [i, j, k] == [k, j, i]
    _, unique_idxs = np.unique(sorted_faces, axis=0, return_index=True) # get unique faces
    dupe_faces = len(unique_idxs) == mesh.n_cells
    dupe_points = mesh.clean(inplace=False).n_points == mesh.n_points
    print('No duplicate faces              ', dupe_faces)
    # check for duplicate points
    print('No duplicate points             ', dupe_points)

    if raise_error:
        if not np.all([tm_watertight, tm_winding, pv_manifold, out_normals, dupe_faces, dupe_points]):
            raise AssertionError("Failed mesh checks")

# bone and cartilage mesh checks

def bone_cartilage_checks(bone_mesh, cartilage_mesh, inerface_points, raise_error=False, check_intersection=True, eta=1e-9):

    # check proximity of boundary nodes
    n_interface_nodes = len(inerface_points)
    def touching_count(A, B):
        # (N,3) float arrays
        A_view = np.ascontiguousarray(A).view([('', A.dtype)] * A.shape[1])
        B_view = np.ascontiguousarray(B).view([('', B.dtype)] * B.shape[1])
        return np.intersect1d(A_view.ravel(), B_view.ravel()).size

    touching_nodes = touching_count(bone_mesh.points, cartilage_mesh.points)
    touching = touching_nodes == n_interface_nodes
    print(
        'All interface nodes present and identical    ', 
        touching
        )

    
    # trimesh
    bone_trimesh = get_trimesh(bone_mesh)
    cartilage_trimesh = get_trimesh(cartilage_mesh)
    if check_intersection:
        intersection = bone_trimesh.intersection(cartilage_trimesh) # Check intersection
        intersection_check = intersection.volume < eta
        if intersection_check: 
            print('Warning: meshes overlap at interface')
            print('Volume of intersection:         ', intersection.volume)
    else: intersection_check = True


    if raise_error:
        if not np.all([touching, intersection_check]):
            raise AssertionError("Failed bone-cartilage checks")
################# MESH CHECKS #################

# stich cartilage cap (upper) surface to base
def stitch_cartilage(cartilage_cap_mesh, base_mesh):
    # combine cartilage base and cap
    combined_mesh = base_mesh + cartilage_cap_mesh

    # get cartilage base and cap edges
    cartilage_edges = combined_mesh.extract_feature_edges(
        boundary_edges=True, non_manifold_edges=False,
        feature_edges=False, manifold_edges=False
    )

    # sort edge lines
    cartilage_edge_lines = cartilage_edges.lines.reshape(-1, 3)[:, 1:] # lines are not in any order
    cap_edge_lines = cartilage_edge_lines[cartilage_edges['mesh_id']<0] # edge lines of cartilage cap

    cap_edge_lines_idx = np.zeros(len(cap_edge_lines), int)
    idx = 0
    for i in range(len(cap_edge_lines)):
        cap_edge_lines_idx[i] = idx
        idx = np.argmax(cap_edge_lines[:, 0] == cap_edge_lines[idx][1])

    cap_edge_lines_sorted = cartilage_edge_lines[cap_edge_lines_idx]

    if len(np.unique(cap_edge_lines_sorted[:,0]))!=len(cap_edge_lines_sorted) or cap_edge_lines_sorted[0, 0]!=cap_edge_lines_sorted[-1,1]:
        print('cap edge not complete loop')

    # create faces to connect upper and lower cartilage edges
    edge_faces = np.zeros((len(cap_edge_lines_sorted)*2, 4), int)
    edge_faces[:, 0] = [3] * len(edge_faces)
    for i, line in enumerate(cap_edge_lines_sorted):
        mesh_id1 = cartilage_edges['mesh_id'][line[0]]
        mesh_id2 = cartilage_edges['mesh_id'][line[1]]

        # adjacent nodes on cartilage cap edge (v1, v2)
        v1 = np.argmax(combined_mesh['mesh_id'] == mesh_id1)
        v2 = np.argmax(combined_mesh['mesh_id'] == mesh_id2) 
        # corresponding adjacent nodes on cartilage base edge (v3, v4)
        v3 = np.argmax(combined_mesh['mesh_id'] == -mesh_id1) 
        v4 = np.argmax(combined_mesh['mesh_id'] == -mesh_id2)

        edge_faces[i*2, 1:] = [v2, v1, v3]
        edge_faces[i*2+1, 1:] = [v2, v3, v4]

    # create complete cartilage mesh
    stitched_mesh = pv.PolyData(
        combined_mesh.points, 
        faces=np.vstack((combined_mesh.faces.reshape(-1, 4), edge_faces))
        ).compute_normals(consistent_normals=True, auto_orient_normals=True)

    return stitched_mesh


################# FLIP NORMALS #################
def remove_normals(mesh):
    for name in list(mesh.point_data.keys()):
        if name.lower() == "normals":
            mesh.point_data.pop(name)
    for name in list(mesh.cell_data.keys()):
        if name.lower() == "normals":
            mesh.cell_data.pop(name)

def get_outward_normal_mask(centres, normals, mesh, eps=1e-5): 
    point_cloud = pv.PolyData(centres - normals * eps)
    return point_cloud.select_interior_points(mesh, check_surface=True)['selected_points'].astype(bool)


def flip_faces(mesh, flip_ids):
    """for face [v1, v2, v3], flips v2 and v3 - creates new mesh and recomputes normals"""
    #mesh = mesh.copy(deep=True)

    # Swap vertex 1 and 2 for those faces
    faces = mesh.faces.reshape(-1, 4).copy()
    faces[flip_ids, 2:4] = faces[flip_ids, 2:4][:, ::-1]
    #mesh.faces = faces.flatten()
    out = pv.PolyData(mesh.points.copy(), faces.flatten())


    for name in mesh.point_data:
        out.point_data[name] = mesh.point_data[name].copy()

    for name in mesh.cell_data:
        out.cell_data[name] = mesh.cell_data[name].copy()

    remove_normals(out)
    return out.compute_normals(
        cell_normals=True,
        point_normals=False,
        consistent_normals=False,
        auto_orient_normals=False,
        split_vertices=False,
        non_manifold_traversal = False,
        feature_angle=180.0,
        inplace=False)
################# FLIP NORMALS #################

################# ARTICULAR GAP STUFF #################
def get_closest_points(bone_df, max_gap):
    """
    Input: df of min dists in each pose for that bone
    Output: df of min_dists, midpoints, bone_ids, ar_bone_ids, poses; for closest points (< max_gap) accross all poses.
    """
    min_dists_df = bone_df[[x for x in bone_df.columns if 'min_dist' in x]].values
    midpoints_df = bone_df[[x for x in bone_df.columns if 'midpoint' in x]].values
    ar_points_df = bone_df[[x for x in bone_df.columns if 'ar_coord' in x]].values
    ar_bone_ids_df = bone_df[[x for x in bone_df.columns if 'point_id' in x]].values

    min_dists = min_dists_df.min(axis=1) # min distance accross all poses
    min_dists_idx = min_dists_df.argmin(axis=1) # idx of min distance accross all poses
    ar_mask = min_dists < max_gap # mask of points within contour

    bone_ids = np.arange(len(bone_df)) 
    ar_bone_ids = ar_bone_ids_df[bone_ids, min_dists_idx]
    midpoints = midpoints_df[bone_ids, min_dists_idx]
    ar_points = ar_points_df[bone_ids, min_dists_idx]

    data = {
        'dist': min_dists[ar_mask],
        'midpoint': midpoints[ar_mask],
        'ar_point': ar_points[ar_mask],
        'bone_id': bone_ids[ar_mask], 
        'ar_bone_id': ar_bone_ids[ar_mask], 
        'pose_id': np.array([int(x.split('-')[-1]) for x in bone_df.columns if 'min_dist' in x])[min_dists_idx[ar_mask]]
    }

    return pd.DataFrame(data)

def get_min_dfs(stl_path, bone, ar_bone, bone_mesh, ar_mesh_neu, poses, max_gap=2):
    bone_dfs = []
    ar_bone_dfs = []
    for pose in tqdm(poses):
        ################# TRANSFORM AR MESH INTO CURRENT POSE #################
        pose_id = pose2idCMC(pose)
        try: # works for all except if neutral-2 doesn't exists - in which case use default neutral
            transforms = get_bone_transforms(pose_id, stl_path)
            R, t = get_relative_transform_new_basis(transforms, ar_bone, bone)
            ar_mesh_posed = transform_mesh(ar_mesh_neu.copy(deep=True), R, t)
        except:
            ar_mesh_posed = ar_mesh_neu.copy(deep=True)
        ################# TRANSFORM AR MESH INTO CURRENT POSE #################

        ################# COMPUTE CLOSEST DISTANCE FOR EACH VERTEX IN EACH POSE #################
        dists = cdist(bone_mesh.points, ar_mesh_posed.points) # euclidean distance between all point pairs
        min_dists_bone = np.min(dists, axis=1) # get minimum distance for each bone vertex
        min_dists_idxs_bone = np.argmin(dists, axis=1) # get corresponding point id of closest vertex on ar bone
        min_dists_ar_bone = np.min(dists, axis=0) # get minimum distance for each ar_bone vertex
        min_dists_idxs_ar_bone = np.argmin(dists, axis=0) # get corresponding point id of closest vertex on bone
        ################# COMPUTE CLOSEST DISTANCE FOR EACH VERTEX IN EACH POSE #################

        ################# GET MINIMUM DISTANCE DFS #################
        bone_dists = {
            f'min_dist-{pose_id}': min_dists_bone, 
            f'midpoint-{pose_id}': list((bone_mesh.points + ar_mesh_posed.points[min_dists_idxs_bone]) / 2),
            f'ar_coord-{pose_id}': list(ar_mesh_posed.points[min_dists_idxs_bone]),
            f'ar_point_id-{pose_id}': min_dists_idxs_bone,     # point id on the other bone
            }
        ar_bone_dists = {
            f'min_dist-{pose_id}': min_dists_ar_bone, 
            f'midpoint-{pose_id}': list((ar_mesh_posed.points + bone_mesh.points[min_dists_idxs_ar_bone]) / 2), 
            f'ar_coord-{pose_id}': list(bone_mesh.points[min_dists_idxs_ar_bone]),
            f'ar_point_id-{pose_id}': min_dists_idxs_ar_bone,  # point id on the other bone (index gives bone point id)
            }
        bone_dfs.append(pd.DataFrame(bone_dists))
        ar_bone_dfs.append(pd.DataFrame(ar_bone_dists))
        ################# GET MINIMUM DISTANCE DFS #################


    ################# COMBINE MINIMUM DISTANCE DFS #################
    bone_df = pd.concat(bone_dfs, axis=1)
    ar_bone_df = pd.concat(ar_bone_dfs, axis=1)
    ################# COMBINE MINIMUM DISTANCE DFS #################

    bone_min_df = get_closest_points(bone_df, max_gap)
    ar_bone_min_df = get_closest_points(ar_bone_df, max_gap)

    return bone_min_df, ar_bone_min_df

def get_min_dfs_mc(stl_path, bone, ar_bone, bone_mesh, ar_mesh_neu, poses, max_gap=2):
    """
    Returns: DF for each bone with point ids of points within max gap (accross all poses) 
            of ar bone and corresponding point ids, coords, and dists.
    Same as min_dfs except it only computes cdist for lower half of mc bones - 2x faster\n
    Still works for any bone pair
    """
    bone_dfs = []
    ar_bone_dfs = []

    # if one of the bones is a metacarpal, speed up computation by only computing cdist for cells close to ar region
    bone_mask, arbone_mask = np.arange(bone_mesh.n_points), np.arange(ar_mesh_neu.n_points)
    if bone[:-1] == 'mc':
        _, bone_mask = get_mc_lower_cell_ids(stl_path, bone_mesh, bone, p=0.5)
    if ar_bone[:-1] == 'mc':
        _, arbone_mask = get_mc_lower_cell_ids(stl_path, ar_mesh_neu, ar_bone, p=0.5)

    for pose in tqdm(poses):
        ################# TRANSFORM AR MESH INTO CURRENT POSE #################
        pose_id = pose2idCMC(pose)
        try: # works for all except if neutral-2 doesn't exists - in which case use default neutral
            transforms = get_bone_transforms(pose_id, stl_path)
            R, t = get_relative_transform_new_basis(transforms, ar_bone, bone)
            ar_mesh_posed = transform_mesh(ar_mesh_neu.copy(deep=True), R, t)
        except:
            ar_mesh_posed = ar_mesh_neu.copy(deep=True)
        ################# TRANSFORM AR MESH INTO CURRENT POSE #################

        ################# COMPUTE CLOSEST DISTANCE FOR EACH VERTEX IN EACH POSE #################
        dists = cdist(bone_mesh.points[bone_mask], ar_mesh_posed.points[arbone_mask]) # euclidean distance between all point pairs

        min_dists_bone = np.min(dists, axis=1) # get minimum distance for each bone vertex
        min_dists_idxs_bone = arbone_mask[np.argmin(dists, axis=1)] # get corresponding point id of closest vertex on ar bone

        min_dists_ar_bone = np.min(dists, axis=0) # get minimum distance for each ar_bone vertex
        min_dists_idxs_ar_bone = bone_mask[np.argmin(dists, axis=0)] # get corresponding point id of closest vertex on bone
        ################# COMPUTE CLOSEST DISTANCE FOR EACH VERTEX IN EACH POSE #################

        ################# GET MINIMUM DISTANCE DFS #################
        bone_dists = {
            f'min_dist-{pose_id}': min_dists_bone, 
            f'midpoint-{pose_id}': list((bone_mesh.points[bone_mask] + ar_mesh_posed.points[min_dists_idxs_bone]) / 2),
            f'ar_coord-{pose_id}': list(ar_mesh_posed.points[min_dists_idxs_bone]),
            f'ar_point_id-{pose_id}': min_dists_idxs_bone,     # point id on the other bone
            }
        ar_bone_dists = {
            f'min_dist-{pose_id}': min_dists_ar_bone, 
            f'midpoint-{pose_id}': list((ar_mesh_posed.points[arbone_mask] + bone_mesh.points[min_dists_idxs_ar_bone]) / 2), 
            f'ar_coord-{pose_id}': list(bone_mesh.points[min_dists_idxs_ar_bone]),
            f'ar_point_id-{pose_id}': min_dists_idxs_ar_bone,  # point id on the other bone (index gives bone point id)
            }
        bone_dfs.append(pd.DataFrame(bone_dists))
        ar_bone_dfs.append(pd.DataFrame(ar_bone_dists))
        ################# GET MINIMUM DISTANCE DFS #################


    ################# COMBINE MINIMUM DISTANCE DFS #################
    bone_df = pd.concat(bone_dfs, axis=1)
    ar_bone_df = pd.concat(ar_bone_dfs, axis=1)
    ################# COMBINE MINIMUM DISTANCE DFS #################

    ################# ACTUALLY GET MIN_DFS #################
    def get_closest_points_mc(bone_df, max_gap, mask):
        """
        Input: df of min dists in each pose for that bone
        Output: df of min_dists, midpoints, bone_ids, ar_bone_ids, poses; for closest points (< max_gap) accross all poses.
        """
        min_dists_df = bone_df[[x for x in bone_df.columns if 'min_dist' in x]].values
        midpoints_df = bone_df[[x for x in bone_df.columns if 'midpoint' in x]].values
        ar_points_df = bone_df[[x for x in bone_df.columns if 'ar_coord' in x]].values
        ar_bone_ids_df = bone_df[[x for x in bone_df.columns if 'point_id' in x]].values

        min_dists = min_dists_df.min(axis=1) # min distance accross all poses
        min_dists_idx = min_dists_df.argmin(axis=1) # idx of min distance accross all poses
        ar_mask = min_dists < max_gap # mask of points within contour

        bone_ids = np.arange(len(bone_df))
        ar_bone_ids = ar_bone_ids_df[bone_ids, min_dists_idx]
        midpoints = midpoints_df[bone_ids, min_dists_idx]
        ar_points = ar_points_df[bone_ids, min_dists_idx]

        data = {
            'dist': min_dists[ar_mask],
            'midpoint': midpoints[ar_mask],
            'ar_point': ar_points[ar_mask],
            'bone_id': mask[ar_mask], 
            'ar_bone_id': ar_bone_ids[ar_mask], 
            'pose_id': np.array([int(x.split('-')[-1]) for x in bone_df.columns if 'min_dist' in x])[min_dists_idx[ar_mask]]
        }

        return pd.DataFrame(data)

    bone_min_df = get_closest_points_mc(bone_df, max_gap, bone_mask)
    ar_bone_min_df = get_closest_points_mc(ar_bone_df, max_gap, arbone_mask)
    ################# ACTUALLY GET MIN_DFS #################

    return bone_min_df, ar_bone_min_df

def get_min_df(stl_path, bone, ar_bone, bone_mesh, ar_mesh_neu, poses, max_gap=2):
    """
    Returns: DF for each bone with point ids of points within max gap (accross all poses) 
            of ar bone and corresponding point ids, coords, and dists.
    """
    bone_dfs = []

    # if one of the bones is a metacarpal, speed up computation by only computing cdist for cells close to ar region
    #bone_mask, arbone_mask = np.arange(bone_mesh.n_points), np.arange(ar_mesh_neu.n_points)
    #if bone[:-1] == 'mc':
    #    _, bone_mask = get_mc_lower_cell_ids(stl_path, bone_mesh, bone, p=0.5)
    #if ar_bone[:-1] == 'mc':
    #    _, arbone_mask = get_mc_lower_cell_ids(stl_path, ar_mesh_neu, ar_bone, p=0.5)

    for pose in tqdm(poses):
        ################# TRANSFORM AR MESH INTO CURRENT POSE #################
        pose_id = pose2idCMC(pose)
        try: # works for all except if neutral-2 doesn't exists - in which case use default neutral
            transforms = get_bone_transforms(pose_id, stl_path)
            R, t = get_relative_transform_new_basis(transforms, ar_bone, bone)
            ar_mesh_posed = transform_mesh(ar_mesh_neu, R, t)
        except:
            ar_mesh_posed = ar_mesh_neu
        ################# TRANSFORM AR MESH INTO CURRENT POSE #################

        ################# COMPUTE CLOSEST DISTANCE FOR EACH VERTEX IN EACH POSE #################
        #dists = cdist(bone_mesh.points[bone_mask], ar_mesh_posed.points[arbone_mask]) # euclidean distance between all point pairs
        #points, dists, _ = get_trimesh(ar_mesh_posed).nearest.on_surface(bone_mesh.points)
        #dists = bone_mesh.compute_implicit_distance(ar_mesh_posed)['implicit_distance']
        cells, points = ar_mesh_posed.find_closest_cell(bone_mesh.points, return_closest_point=True)
        dists = np.linalg.norm(points - bone_mesh.points, axis=1)
        ################# COMPUTE CLOSEST DISTANCE FOR EACH VERTEX IN EACH POSE #################

        ################# GET MINIMUM DISTANCE DFS #################
        bone_dists = {
            f'min_dist-{pose_id}': dists, 
            f'midpoint-{pose_id}': list((bone_mesh.points + points) / 2),
            f'ar_coord-{pose_id}': list(points), # closest point on ar bone (not necessarily a vertex)
            }
        bone_dfs.append(pd.DataFrame(bone_dists))
        ################# GET MINIMUM DISTANCE DFS #################


    ################# COMBINE MINIMUM DISTANCE DFS #################
    bone_df = pd.concat(bone_dfs, axis=1)
    ################# COMBINE MINIMUM DISTANCE DFS #################

    ################# ACTUALLY GET MIN_DFS #################
    def get_closest_points_mc(bone_df, max_gap):
        """
        Input: df of min dists in each pose for that bone
        Output: df of min_dists, midpoints, bone_ids, ar_bone_ids, poses; for closest points (< max_gap) accross all poses.
        """
        min_dists_df = bone_df[[x for x in bone_df.columns if 'min_dist' in x]].values
        midpoints_df = bone_df[[x for x in bone_df.columns if 'midpoint' in x]].values
        ar_points_df = bone_df[[x for x in bone_df.columns if 'ar_coord' in x]].values

        min_dists = min_dists_df.min(axis=1) # min distance accross all poses
        min_dists_idx = min_dists_df.argmin(axis=1) # idx of min distance accross all poses
        ar_mask = min_dists < max_gap # mask of points within contour

        bone_ids = np.arange(len(bone_df))
        midpoints = midpoints_df[bone_ids, min_dists_idx]
        ar_points = ar_points_df[bone_ids, min_dists_idx]

        data = {
            'dist': min_dists[ar_mask],
            'midpoint': midpoints[ar_mask],
            'ar_point': ar_points[ar_mask],
            #'bone_id': mask[ar_mask], 
            'bone_id': bone_ids[ar_mask],
            'pose_id': np.array([int(x.split('-')[-1]) for x in bone_df.columns if 'min_dist' in x])[min_dists_idx[ar_mask]]
        }

        return pd.DataFrame(data)

    bone_min_df = get_closest_points_mc(bone_df, max_gap)
    ################# ACTUALLY GET MIN_DFS #################

    return bone_min_df

def get_min_df_fast(stl_path, bone, ar_bone, bone_mesh, ar_mesh_neu, poses, max_gap=2):
    """
    For each point on the bone mesh that comes within the given max_gap of the ar_mesh, finds the closest ar mesh point.

    Output columns:
        - dist
        - midpoint
        - ar_point
        - bone_id
        - pose_id
    """
    bone_points = np.asarray(bone_mesh.points)
    n_points = bone_points.shape[0]

    # Running best values across poses
    best_sqdist = np.full(n_points, np.inf, dtype=np.float64)
    best_midpoints = np.empty((n_points, 3), dtype=np.float64)
    best_ar_points = np.empty((n_points, 3), dtype=np.float64)
    best_pose_ids = np.empty(n_points, dtype=np.int64)

    pose_ids = [pose2idCMC(pose) for pose in poses]
    for pose_id in pose_ids:

        try:
            transforms = get_bone_transforms(pose_id, stl_path)
            R, t = get_relative_transform_new_basis(transforms, ar_bone, bone)
            ar_mesh_posed = transform_mesh(ar_mesh_neu.copy(deep=True), R, t)
        except:
            ar_mesh_posed = ar_mesh_neu.copy(deep=True)

        # Closest point on posed AR mesh for every bone point
        _, ar_points = ar_mesh_posed.find_closest_cell(
            bone_points,
            return_closest_point=True
        )

        diff = ar_points - bone_points
        sqdist = np.einsum("ij,ij->i", diff, diff)   # faster

        update_mask = sqdist < best_sqdist

        if np.any(update_mask):
            best_sqdist[update_mask] = sqdist[update_mask]
            best_ar_points[update_mask] = ar_points[update_mask]
            best_midpoints[update_mask] = 0.5 * (bone_points[update_mask] + ar_points[update_mask])
            best_pose_ids[update_mask] = pose_id

    best_dist = np.sqrt(best_sqdist)
    keep_mask = best_dist < max_gap
    bone_ids = np.arange(n_points, dtype=np.int64)

    return pd.DataFrame({
        "dist": best_dist[keep_mask],
        "midpoint": list(best_midpoints[keep_mask]),
        "ar_point": list(best_ar_points[keep_mask]),
        "bone_id": bone_ids[keep_mask],
        "pose_id": best_pose_ids[keep_mask],
    })










# check edges haven't moved
def check_points_still_there(mesh, points, eps=1e-5, raise_error=False):
    """Checks that new points still have corresponding unique point on previous mesh
    and returns the corresponding mesh point ids"""
    n_points = len(points)
    mesh_ids = np.zeros(n_points, dtype=int) # on bone_mesh
    for i, mesh_edge_point in enumerate(points):
        point_id = mesh.find_closest_point(mesh_edge_point)
        mesh_ids[i] = point_id
        if np.linalg.norm(mesh.points[point_id] - mesh_edge_point) > eps:
            print('Point not close!') # make sure it finds identical point on other mesh
            if raise_error:
                raise AssertionError("Point not close")
    if len(np.unique(mesh_ids)) != n_points:
        print('Not all vertices are unique!') # make sure it finds a unique point for each point
    return mesh_ids

def interp_vecs(A, B, f, eps=1e-8):
    """Returns: unit vectors that lie fraction f (0 to 1) on the smaller angle from A to B"""

    a = A / np.linalg.norm(A, axis=1, keepdims=True)
    b = B / np.linalg.norm(B, axis=1, keepdims=True)

    # angles
    cos_theta = np.clip(np.sum(a * b, axis=1))
    theta = np.arccos(cos_theta)
    sin_theta = np.sin(theta)

    # mask for non-zero angles (incase divide by zero)
    mask = sin_theta > eps

    v = np.empty_like(a)
    if np.any(mask):
        st = sin_theta[mask]
        th = theta[mask]
        ff = f[mask]
        s1 = np.sin((1.0 - ff) * th) / st
        s2 = np.sin(ff * th) / st
        v[mask] = s1[:, None] * a[mask] + s2[:, None] * b[mask]
    if np.any(~mask):
        ff = f[~mask]
        v0 = (1.0 - ff)[:, None] * a[~mask] + ff[:, None] * b[~mask]
        v[~mask] = v0


    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v

def get_nearest_boundary(taper_boundary_ids, geo_dists_matrix):
    """Returns taper_mesh ids and geo_dists for nearest id in taper_boundary_ids to each point in taper mesh"""
    D_boundary = geo_dists_matrix[:, taper_boundary_ids].toarray() # dists of all_points(i) x outer_points(j)
    D_boundary = np.where(D_boundary==0, np.inf, D_boundary) # dists of all_points x outer_points (0=>inf)
    D_boundary[taper_boundary_ids, np.arange(len(taper_boundary_ids))] = 0 # set outer_point dists back to 0
    near_boundary_ids = np.argmin(D_boundary, axis=1) # idx(of outer_points) of nearest outer_point for each point(inc. outers)
    near_boundary_D = np.min(D_boundary, axis=1) # dist(of outer_points) of nearest outer_point for each point(inc. outers)
    near_boundary_taper_ids = taper_boundary_ids[near_boundary_ids] # idxs of nearest outer_point on taper mesh for each point 
    #boundary_vecs = taper_mesh.points[near_boundary_taper_ids] - taper_mesh.points
    return near_boundary_taper_ids, near_boundary_D

def taper_f(x, max_x, max_y, p=2):
    a, b = 0, max_x # min x, max x (dist from cartilage edge)
    c, d = max_y, 0 # max y, min y (height above surface)
    return c + (1 - (x - a) / (b - a))**p * (d - c)

def get_triangle_adjacency(mesh):
    """ 
    returns:
    edge_map - cells that use each edge
    adjacency - neighbouring cell ids for each cell
    """
    tris = mesh.faces.reshape(-1, 4)[:, 1:]
    n_tris = tris.shape[0]

    # Map edges -> triangles
    edge_map = {}
    for fi, tri in enumerate(tris):
        e0 = tuple(sorted((tri[0], tri[1])))
        e1 = tuple(sorted((tri[1], tri[2])))
        e2 = tuple(sorted((tri[2], tri[0])))
        for e in (e0, e1, e2):
            edge_map.setdefault(e, []).append(fi)

    # Build adjacency list
    adjacency = [[] for _ in range(n_tris)]
    for e, flist in edge_map.items():
        if len(flist) == 2:
            a, b = flist
            adjacency[a].append(b)
            adjacency[b].append(a)

    return edge_map, adjacency

from collections import deque

def flood_fill_cells(mesh, start_face, boundary_edges, adjacency):
    """
    Flood-fill triangles starting from start_face, including those that have edges
    on the boundary. Only prevents crossing the boundary to the outside.
    """
    tris = mesh.faces.reshape(-1, 4)[:, 1:]
    visited = set()
    queue = deque([start_face])
    
    # Convert boundary edges to a set of tuples for faster lookup
    boundary_set = {tuple(sorted(edge)) for edge in boundary_edges}

    while queue:
        f = queue.popleft()
        if f in visited:
            continue
        visited.add(f)

        for nbr in adjacency[f]:
            if nbr in visited:
                continue

            # Find the shared edge between f and nbr
            shared_edge = tuple(sorted(frozenset(tris[f]).intersection(tris[nbr])))
            
            # If the shared edge is a boundary edge, don't cross it
            if shared_edge in boundary_set:
                continue

            queue.append(nbr)

    return np.array(list(visited))
################# ARTICULAR GAP STUFF #################


def fill_holes_pmf(mesh, nbe=20, refine=False):
    """fill holes using pymeshfix\n
    nbe: max number of boundary edges for hole to be filled\n
    refine: refine filled mesh to match surrounding cell sizes\n
    returns: filled mesh"""
    v = np.asarray(mesh.points, dtype=np.float64)
    f = np.asarray(mesh.faces.reshape(-1, 4)[:, 1:], dtype=np.int32)

    mfix = pymeshfix.PyTMesh()
    mfix.load_array(v, f)

    # Fill only holes whose boundary has <= 50 edges
    mfix.fill_small_boundaries(nbe=nbe, refine=refine)

    v2, f2 = mfix.return_arrays()

    return pv.PolyData(v2,np.hstack([np.full((len(f2), 1), 3), f2]).ravel())