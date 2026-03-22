import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cdist


from phd_helpers.helpers2 import (
    get_bone_motions, get_relative_motion_new_coords, get_motion_mesh, pose2idCMC
)

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
    ar_bone_ids = ar_bone_ids_df[bone_ids, min_dists_idx] # unessesary to idx bone_ids (== :)
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

def get_min_dfs(bone, ar_bone, bone_mesh, ar_mesh_neu, motion_paths, poses, max_gap=2):
    bone_dfs = []
    ar_bone_dfs = []
    for pose in tqdm(poses):
        ################# TRANSFORM AR MESH INTO CURRENT POSE #################
        pose_id = pose2idCMC(pose)
        #if pose == 'neutral': # for neutral pose
            #R, t = np.eye(3), np.zeros(3)
        #else:
            #motions = get_bone_motions(pose_id, motion_paths)
            #R, t = get_relative_motion_new_coords(motions, ar_bone, bone)
        #ar_mesh_posed = get_motion_mesh(ar_mesh_neu.copy(deep=True), R, t)
        
        try: # works for all except if neutral-2 doesn't exists - in which case use default neutral
            motions = get_bone_motions(pose_id, motion_paths)
            R, t = get_relative_motion_new_coords(motions, ar_bone, bone)
            ar_mesh_posed = get_motion_mesh(ar_mesh_neu.copy(deep=True), R, t)
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