# Other cartilage-old.py in this folder has some small improvements over previous MeshPrep cartilage.py
# But this one has some more major changes - mainly using identical points counts to replace closest point

import numpy as np
import pandas as pd
import pyvista as pv
import gdist
from scipy.spatial.distance import cdist
from pathlib import Path
import subprocess

from phd_helpers.CartilageGeneration import (
    mesh_checks, bone_cartilage_checks, get_outward_normal_mask, flip_faces, taper_f, get_nearest_boundary, 
    interp_vecs, get_triangle_adjacency, flood_fill_cells, remove_normals, fill_holes_pmf
)
from phd_helpers.paths import find_corresponding_cells, identical_points_count, get_boundary

def articular_gap(
    bone_mesh,
    min_df,
    compute_quality,
    quality_path, # where to save cartilage mesh distance data (to measure effect of smoothing/remeshing)
    remesh_cartilage,
    cgal_input_path, # path to c++ fixed boundary input
    taper_width = 2, # width of cartilage taper region (limit - only tapers if above taper curve)
    max_height = 1, # max height of cartilage in taper region
    p_h = 2, # shape of taper height (1 = linear , higher = steeper taper)
    p_v = 1, # shape of vector ratio (1 = linear)
    cartilage_smooth_iters = 200, # need to look at this, currently uses laplacian 
    edge_length = 0.2, # target edge length of cartilage remesh
    n_iters = 5 # n isotropic remeshing iterations for cartilage remesh
    ):
    """
    Returns:
    combined_mesh_flipped - bone and cartilage combined mesh with shared non-manifold interface
    """

    ################# MESH STUFF #################
    bone_mesh['Normals'] = bone_mesh.compute_normals(point_normals=True, cell_normals=False)['Normals']
    bone_mesh['bone_id'] = np.arange(bone_mesh.n_points)
    bone_mesh['bone_cell_id'] = np.arange(bone_mesh.n_cells)
    ################# MESH STUFF #################


    ################# COMPUTE TAPER REGION #################
    # extract mesh of cartilage points on bone mesh (makes gdist computation faster) - (bone-cartilage interface mesh)
    inter_mesh = bone_mesh.extract_points(min_df['bone_id'], adjacent_cells=False).extract_geometry()
    inter_mesh['inter_cell_ids'] = np.arange(inter_mesh.n_cells)
    # remove any missing points due to extract geometry (so remove any points not part of a complete triangle)
    missing_mask = ~np.isin(min_df['bone_id'], inter_mesh['bone_id'])
    min_df.drop(min_df['bone_id'][missing_mask].index.values, inplace=True)

    # useful values from min_df (min_df does not change after this point - final change was at min_df.drop(missing))
    midpoint_dist = min_df['dist'] / 2 # distance of midpoint between two bones for each pair of closest points
    midpoints = np.array(min_df['midpoint'].tolist()) # midpoint coordinates
    #ar_points = np.array(min_df['ar_point'].tolist()) # closest points on ar bone to each bone point


    # get boundary of cartilage on inter mesh
    inter_boundary = get_boundary(inter_mesh)
    boundary_mask_inter = np.isin(inter_mesh['bone_id'], inter_boundary['bone_id']) 
    boundary_ids = np.arange(inter_mesh.n_points)[boundary_mask_inter] # on inter_mesh

    # minimum geo dist of every node from closest source_idx
    geo_dists = gdist.compute_gdist(
        inter_mesh.points.astype(np.float64),
        inter_mesh.faces.reshape(-1, 4)[:, 1:].astype(np.int32),
        source_indices=boundary_ids.astype(np.int32), 
    ) 

    # get mask of nodes within taper width and below taper function
    taper_heights = taper_f(geo_dists, taper_width, max_height, p=p_h)
    taper_mask = (taper_heights <= min_df['dist'] / 2) & (geo_dists<=taper_width)

    # get taper points mesh (makes computation much faster) - mesh of only taper region
    taper_mask_inter = np.isin(inter_mesh['bone_id'], min_df['bone_id'][taper_mask])
    taper_mesh = inter_mesh.extract_points(taper_mask_inter, adjacent_cells=False).extract_geometry()
        # assign non-taper cells that lie on "pinched" islands to taper region
    not_taper_mesh = inter_mesh.extract_cells(taper_mesh['inter_cell_ids'], invert=True).extract_geometry()
    edge_map, adjacency = get_triangle_adjacency(not_taper_mesh)
    start_face = not_taper_mesh.find_closest_cell(np.mean(not_taper_mesh.points, axis=0)) # not best way of doing this!
    inner_cells = flood_fill_cells(not_taper_mesh, start_face, get_boundary(not_taper_mesh).lines.reshape(-1, 3)[:, 1:], adjacency)
        # final taper mesh #
    taper_mesh = inter_mesh.extract_cells(not_taper_mesh['inter_cell_ids'][inner_cells], invert=True).extract_geometry()
        # remove any missing points from taper_mask after extracting geometry and islands - not ideal but quicker than using inter mesh
    taper_mask = np.isin(min_df['bone_id'], taper_mesh['bone_id'])
    #taper_geo_dists = geo_dists[taper_mask]

    # geo_dist between all points within max_distance
    geo_dists_matrix = gdist.local_gdist_matrix(
        taper_mesh.points.astype(np.float64),
        taper_mesh.faces.reshape(-1, 4)[:, 1:].astype(np.int32),
        max_distance=taper_width+1e-3
    ) 

    # get taper mesh boundaries
    taper_boundary = get_boundary(taper_mesh)
    boundary_outer_mask_tb = np.isin(taper_boundary['bone_id'], inter_boundary['bone_id']) # on taper_boundary

    # get taper boundary innner and outer nodes on taper mesh
    taper_outer_mask = np.isin(taper_mesh['bone_id'], taper_boundary['bone_id'][boundary_outer_mask_tb]) # on taper_mesh
    taper_inner_mask = np.isin(taper_mesh['bone_id'], taper_boundary['bone_id'][~boundary_outer_mask_tb]) # on taper_mesh
    taper_outer_ids = np.arange(taper_mesh.n_points)[taper_outer_mask] # on taper_mesh
    taper_inner_ids = np.arange(taper_mesh.n_points)[taper_inner_mask] # on taper_mesh

    # ids and distances of boundary nodes that are closest to each taper node
    _, near_taper_outer_D = get_nearest_boundary(taper_outer_ids, geo_dists_matrix) # ids on taper_mesh
    near_taper_inner_ids, near_taper_inner_D = get_nearest_boundary(taper_inner_ids, geo_dists_matrix) # ids on taper_mesh
    # Distance fraction of each taper_mesh point from closest outer_node to closest_inner node
    taper_Df = (near_taper_outer_D) / (near_taper_outer_D + near_taper_inner_D)

    # get nearest inner node midpoint heights
    taper_inner_mask_c = np.isin(min_df['bone_id'], taper_mesh['bone_id'][taper_inner_ids]) # on c_mesh
    inner_node_midpoint_heights = midpoint_dist.values[taper_inner_mask_c]
    near_inner_node_midpoint_heights = inner_node_midpoint_heights[np.searchsorted(taper_inner_ids, near_taper_inner_ids)]

    # set taper node heights
    taper_heights = taper_f(taper_Df, 1, near_inner_node_midpoint_heights, p=p_h) # non-linear

    # set vector directions
    midpoint_vecs = midpoints - bone_mesh.points[min_df['bone_id']] # vector from bone point to midpoint
    taper_vecs_mid = midpoint_vecs[taper_mask] # midpoint_vecs
    taper_vecs_norm = bone_mesh['Normals'][min_df['bone_id']][taper_mask] # normal vecs
    vec_dirs = taper_f(taper_Df, 1, 1, p=p_v)
    taper_vecs = interp_vecs(taper_vecs_norm, taper_vecs_mid, vec_dirs)

    # get taper coords
    taper_points = bone_mesh.points[min_df['bone_id']][taper_mask]
    taper_points = taper_points + taper_heights.reshape(-1, 1)*taper_vecs

    # get taper points
    points_tapered = midpoints.copy()
    points_tapered[taper_mask] = taper_points

    #closest_points, distances, cell_ids = get_trimesh(bone_mesh).nearest.on_surface(points_tapered)
    ################# COMPUTE TAPER REGION #################

    ################# MESH TAPER REGION #################
    # map faces from bone mesh to extruded taper region points
    # mesh curves down in taper region and delaunay doesn't like that
    tapered_mesh = pv.PolyData(taper_points, taper_mesh.faces)
    ################# MESH TAPER REGION #################

    ################# MESH INNER REGION #################
    tapered_edge = get_boundary(tapered_mesh)

    inner_edge = tapered_edge.extract_points(~boundary_outer_mask_tb).extract_geometry()
    #inner_points = pv.PolyData(midpoints[~taper_mask])
    inner_mesh = pv.PolyData(np.vstack( (inner_edge.points, midpoints[~taper_mask]) ), lines=inner_edge.lines)

    inner_mesh = inner_mesh.delaunay_2d(edge_source=inner_edge, alpha=0.9).triangulate()
    inner_mesh = inner_mesh.fill_holes(inner_mesh.area/20)
    #inner_mesh.lines = inner_edge.lines # reset edge lines to remove delaunay leftover lines
    inner_mesh.lines = np.empty(0, dtype='int64') # remove all lines for now cos they show up in mesh.faces

    # remove cells that lie outside of the inner region boundary
    edge_map, adjacency = get_triangle_adjacency(inner_mesh)
    start_face = inner_mesh.find_closest_cell(np.mean(inner_mesh.points, axis=0))
    inner_cells = flood_fill_cells(inner_mesh, start_face, inner_edge.lines.reshape(-1, 3)[:, 1:], adjacency)
    inner_mesh_clean = pv.PolyData(inner_mesh.points, inner_mesh.faces.reshape(-1, 4)[inner_cells])
    # should maybe be calling remove unused points here, lines bring their own points that are left behind?
    ################# MESH INNER REGION #################

    ################# COMBINE INNER MESH AND TAPER MESH #################
    #edge_check1 = np.isin(inner_edge.points, tapered_mesh.points).all()
    edge_check1 = identical_points_count(inner_edge.points, tapered_mesh.points) == inner_edge.n_points #*** new
    #edge_check2 = np.isin(inner_edge.points, inner_mesh_clean.points).all()
    edge_check2 = identical_points_count(inner_edge.points, inner_mesh_clean.points) == inner_edge.n_points #*** new

    inner_mesh_clean['inner_cells'] = np.full(inner_mesh_clean.n_cells, 1)
    tapered_mesh['inner_cells'] = np.full(tapered_mesh.n_cells, 0)
    mesh = inner_mesh_clean + tapered_mesh # full cartilage cap mesh
        # check if there are any holes at boundary between inner mesh and tapered mesh and try to fill them
    if get_boundary(mesh).connectivity()['RegionId'].any():
        mesh = fill_holes_pmf(mesh, nbe=20)
        mesh['inner_cells'] = np.ones(mesh.n_cells, dtype=int)
        mesh['inner_cells'][find_corresponding_cells(mesh, tapered_mesh, raise_error=True)] = 0
    else:
        if not mesh.n_points == midpoints.shape[0]: # had to move to else cos pymeshfix messes with stuff sometimes
            raise AssertionError('Not all midpoints in mesh (& no dupes)')

    if not edge_check1:
        raise AssertionError('Not all boundary points in tapered mesh')
    if not edge_check2:
        raise AssertionError('Not all boundary points in inner mesh')

    # check for flat faces
    #closest_points, implicit_distances, cell_ids = get_trimesh(bone_mesh).nearest.on_surface(mesh.points)
    _, ps = bone_mesh.find_closest_cell(mesh.points, return_closest_point=True)
    implicit_distances = np.linalg.norm(mesh.points - ps, axis=1) #*** changed from line above

    mesh_faces = mesh.faces.reshape(-1, 4)[:, 1:]
    flat_face_mask = (implicit_distances[mesh_faces] <= 1e-12).all(axis=1) #*** changed from ==0 to <=1e-12
    #flat_face_ids = np.where(flat_face_mask)[0]

    # remove flat faces and leftover points - occur when all 3 vertices lie on the boundary
    mesh_clean = pv.PolyData(mesh.points, mesh.faces.reshape(-1, 4)[~flat_face_mask]).compute_normals(auto_orient_normals=True)
    mesh_clean.remove_unused_points(inplace=True)
    mesh_clean['mesh_clean_id'] = np.arange(mesh_clean.n_points)
    # get array of inner points/cells on mesh_clean 
    mesh_clean['inner_cells'] = mesh['inner_cells'][find_corresponding_cells(mesh, mesh_clean)]
    mesh_clean['inner_points'] = np.full(mesh_clean.n_points, 1)
    mesh_clean['inner_points'][np.unique(mesh_clean.faces.reshape(-1, 4)[:, 1:][np.where(mesh_clean['inner_cells']==0)[0]])] = 0
    #mesh_clean['inner_points'] = np.zeros(mesh_clean.n_points, dtype=int)
    #for p in inner_mesh_clean.points:
    #    mesh_clean['inner_points'][mesh_clean.find_closest_point(p)] = 1

    # get edge points on mesh and bone_mesh
    mesh_clean_edge = get_boundary(mesh_clean)
    mesh_clean_edge_mask = np.isin(mesh_clean['mesh_clean_id'], mesh_clean_edge['mesh_clean_id']) # on mesh_clean
    mesh_edge_ids = mesh_clean['mesh_clean_id'][mesh_clean_edge_mask] # on mesh_clean
    mesh_edge_points = mesh_clean.points[mesh_clean_edge_mask] # on mesh_clean

    #print('Checking mesh boundary has not changed')
    #bone_mesh_edge_ids = check_points_still_there(bone_mesh, mesh_edge_points)
    _, bone_mesh_edge_ids, _ = identical_points_count(bone_mesh.points, mesh_edge_points, return_indices=True) #*** new
    bone_mesh_edge_ids = np.sort(bone_mesh_edge_ids)
    if len(bone_mesh_edge_ids) != len(mesh_edge_points):
        raise AssertionError("Mesh boundary has changed")
    

    if (mesh.n_cells - mesh_clean.n_cells) != flat_face_mask.sum():
        raise AssertionError("Not all flat faces removed")
    ################# COMBINE INNER MESH AND TAPER MESH #################
    print('Created cartilage cap')

    print('Smoothing cartilage cap')
    ################# SMOOTH CARTILAGE MESH CAP #################
    cartilage_cap = mesh_clean.smooth(
    n_iter=cartilage_smooth_iters,
    feature_angle=180, # prevent anything from being feature cos I don't understant feature_smoothing arg...
    boundary_smoothing = False, # keeps boundary fixed-ish
    feature_smoothing = False # prevents feature edges from being identified - so they get smoothed?... (idk stupid)
    )
    cartilage_cap_edge = get_boundary(cartilage_cap)

    if cartilage_cap_edge.n_points != mesh_clean_edge.n_points:
        raise AssertionError(f'{cartilage_cap_edge.n_points - mesh_clean_edge.n_points} Boundary points lost during smoothing')
    #_ = check_points_still_there(mesh_clean_edge, cartilage_cap_edge.points)

    # put edge points back so they are numerically identical
    cartilage_cap.points[mesh_edge_ids] = bone_mesh.points[bone_mesh_edge_ids]

    if compute_quality:
        # measure change due to smoothing and save to file - compute_implicit_distance is fastest algo
        cartilage_cap['implicit_distance_orig'] = np.asarray(
            cartilage_cap.compute_implicit_distance(mesh_clean)['implicit_distance'],
            dtype=np.float64
            ).copy()
        quality_path.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            'dist_orig':cartilage_cap['implicit_distance_orig'], 
            'inner_point':cartilage_cap['inner_points']
            }).to_csv(quality_path / 'smoothing_dists.csv', index=False)
    ################# SMOOTH CARTILAGE MESH CAP #################

    if remesh_cartilage:
        ################# REMESH CARTILAGE #################
        # remesh cartilage with CGAL fixed boundary - check boundary hasn't moved (and put points back anyway)
        # this is the final step in creating the cartilage cap mesh (just need to attach to bone after this - trivial)

        print('Writing mesh to remeshing input directory')
        cartilage_cap.save(cgal_input_path)

        print('Remeshing cartilage cap')

        cgal_path = cgal_input_path.parent.parent.parent
        cgal_output_path = cgal_path / 'outputs/fb_output' / cgal_input_path.name
        args = [
            str(cgal_path / "bin/fixed_boundary"),
            str(cgal_input_path),  # path to input mesh
            str(cgal_output_path), # path to output mesh
            str(edge_length), # target edge length
            str(n_iters), # number of CGAL isotropic remeshing iterations
            ]

        result = subprocess.run(args, text=True)
        result.check_returncode()  # raise after printing




        # load remeshed cartilage cap
        cartilage_remesh = pv.read(cgal_output_path)
        cartilage_remesh['remesh_ids'] = np.arange(cartilage_remesh.n_points)

        # check boudnary hasn't moved
        #print('Checking mesh boundary has not changed')
        cartilage_remesh_edge = get_boundary(cartilage_remesh)
        if cartilage_remesh_edge.n_points != cartilage_cap_edge.n_points:
            raise ValueError(f'{cartilage_remesh_edge.n_points - cartilage_cap_edge.n_points} Boundary points lost in remeshing')
        
        #cartilage_cap_edge_ids = check_points_still_there(cartilage_cap_edge, cartilage_remesh_edge.points)
        remesh_edge_dists = cdist(cartilage_cap_edge.points, cartilage_remesh_edge.points) #*** new
        cartilage_cap_edge_ids = np.argmin(remesh_edge_dists, axis=0)
        closest_dists = np.min(remesh_edge_dists, axis=0)
        if (closest_dists > 1e-5).any():
            raise AssertionError(f"Cartilage boundary points moved during remeshing: {max(closest_dists):.5f} mm")

        # put edge points back so they are numerically identical
        cartilage_remesh.points[cartilage_remesh_edge['remesh_ids']] = cartilage_cap_edge.points[cartilage_cap_edge_ids]

        # measure cartilage height and store in mesh
        cartilage_remesh_height = cartilage_remesh.compute_implicit_distance(bone_mesh)['implicit_distance']
        if (cartilage_remesh_height < 0).any():
            raise AssertionError('Not all cartilage points above bone surface') # detect interference

        # add cartilge taper/inner region array
        cartilage_remesh['inner_cells'] = cartilage_cap['inner_cells'][cartilage_cap.find_closest_cell(cartilage_remesh.cell_centers().points)]
        
        if compute_quality:
            # measure change due to remeshing and save to file
            cartilage_remesh['implicit_distance_orig'] = np.asarray(
                cartilage_remesh.compute_implicit_distance(mesh_clean)['implicit_distance'],
                dtype=np.float64
                ).copy()
            cartilage_remesh['implicit_distance_smooth'] = np.asarray(
                cartilage_remesh.compute_implicit_distance(cartilage_cap)['implicit_distance'],
                dtype=np.float64
                ).copy()
            pd.DataFrame({
                'dist_orig':cartilage_remesh['implicit_distance_orig'], 
                'dist_smooth':cartilage_remesh['implicit_distance_smooth'],
                }).to_csv(quality_path / 'remesh_dists.csv', index=False)
        ################# REMESH CARTILAGE #################
    else:
        cartilage_remesh = cartilage_cap.copy(deep=True)
        cartilage_remesh['remesh_ids'] = np.arange(cartilage_remesh.n_points)
        cartilage_remesh_edge = get_boundary(cartilage_remesh)

    print('\nAttaching cartilage cap')
    ################# GET FINAL INTERFACE MESH BETWEEN BONE AND CARTILAGE #################
    # find which inter_boundary points are still in mesh boundary - after unused point removal
    #inter_boundary_ids = check_points_still_there(inter_boundary, mesh_edge_points)
    _, inter_boundary_ids, _ = identical_points_count(inter_boundary.points, mesh_edge_points, return_indices=True) #*** new

    # find which inter_mesh points are still in mesh and their bone_ids
    unused_inter_ids = inter_boundary['bone_id'][~np.isin(np.arange(inter_boundary.n_points), inter_boundary_ids)]
    interface_bone_mesh_ids = inter_mesh['bone_id'][~np.isin(inter_mesh['bone_id'], unused_inter_ids)]

    # extract bone_cartilage interface mesh
    interface_mesh = bone_mesh.extract_points(interface_bone_mesh_ids, adjacent_cells=False).extract_geometry()
    ################# GET FINAL INTERFACE MESH BETWEEN BONE AND CARTILAGE #################


    ################# GET FINAL COMBINED MESH WITH REGION IDs #################

    # assign scalar data - region_id
    bone_mesh['region_id'] = np.ones(bone_mesh.n_cells, dtype=int)
    bone_mesh['region_id'][find_corresponding_cells(bone_mesh, interface_mesh, raise_error=True)] = 3
    cartilage_remesh['region_id'] = np.full(cartilage_remesh.n_cells, 2)
    # assign scalar data - cartilage regions
    #bone_mesh['cartilage_region']        = np.full(bone_mesh.n_points,       -1).astype(int)
    #cartilage_remesh['cartilage_region'] = np.full(cartilage_remesh.n_points, 1).astype(int)

    #cartilage_edge_dists = gdist.compute_gdist(
    #    cartilage_remesh.points.astype(np.float64),
    #    cartilage_remesh.faces.reshape(-1, 4)[:, 1:].astype(np.int32),
    #    source_indices=cartilage_remesh_edge['remesh_ids'].astype(np.int32), 
    #)
    #remesh_taper_mask = cartilage_edge_dists <= taper_width
    #cartilage_remesh['cartilage_region'][remesh_taper_mask] = 0

    # create combined mesh
    bone_mesh['inner_cells'] = np.full(bone_mesh.n_cells, -1)
    combined_mesh = bone_mesh + cartilage_remesh
    combined_mesh.compute_normals(inplace=True, consistent_normals=False)

    combined_mesh['inner_points'] = np.full(combined_mesh.n_points, -1)
    combined_mesh['inner_points'][np.unique(combined_mesh.faces.reshape(-1, 4)[:, 1:][np.where(combined_mesh['inner_cells']==1)[0]])] = 1
    combined_mesh['inner_points'][np.unique(combined_mesh.faces.reshape(-1, 4)[:, 1:][np.where(combined_mesh['inner_cells']==0)[0]])] = 0

    dupe_check = combined_mesh.n_points == (cartilage_remesh.n_points+bone_mesh.n_points) - mesh_edge_ids.shape[0]
    if not dupe_check:
        raise AssertionError('Not all duplicate points removed')
    ################# GET FINAL COMBINED MESH WITH REGION IDs #################

    ################# MESH CHECKS #################
    # get cells ids of each region on the combined mesh
    combined_mesh_cell_ids = np.arange(combined_mesh.n_cells)
    bone_surf_ids = combined_mesh_cell_ids[np.where(combined_mesh['region_id']==1)[0]]
    bone_shell_ids = combined_mesh_cell_ids[np.where(combined_mesh['region_id']!=2)[0]]
    cartilage_surf_ids = combined_mesh_cell_ids[np.where(combined_mesh['region_id']==2)[0]]
    #interface_surf_ids = combined_mesh_cell_ids[np.where(combined_mesh['region_id']==3)[0]]

    # extract enclosed cartilage volume to do checks
    cartilage_mesh = combined_mesh.extract_cells(bone_surf_ids, invert=True).extract_geometry()
    remove_normals(cartilage_mesh)  # have to remove them before recomputing 
                            #- cos inherited from parent mesh and doesn't recompute something about them for some reason
    cartilage_mesh.compute_normals(auto_orient_normals=True, inplace=True)
    # check if normals point outwards or inwards and if not flip them
    if not get_outward_normal_mask(cartilage_mesh.cell_centers().points, cartilage_mesh.cell_normals, cartilage_mesh).any():
        cartilage_mesh = flip_faces(cartilage_mesh, np.arange(cartilage_mesh.n_cells))
        print('flipped faces')

    print('\nMESH CHECKS...')
    # mesh checks
    print('\n----- CARTILAGE -----')
    mesh_checks(cartilage_mesh, raise_error=True)

    print('\n----- BONE -----')
    mesh_checks(bone_mesh, raise_error=True)

    print('\n----- BONE-CARTILAGE -----')
    bone_cartilage_checks(bone_mesh, cartilage_mesh, interface_mesh.points, raise_error=True, check_intersection=False)

    combined_mesh_centres = combined_mesh.cell_centers().points
    combined_mesh_normals = combined_mesh.cell_normals

    # check if normals point outwards
    bone_normal_mask = get_outward_normal_mask( # including boundary
        combined_mesh_centres[bone_shell_ids], 
        combined_mesh_normals[bone_shell_ids], 
        bone_mesh
        )
    cartilage_normal_mask = get_outward_normal_mask(
        combined_mesh_centres[cartilage_surf_ids], 
        combined_mesh_normals[cartilage_surf_ids], 
        cartilage_mesh
        )

    # check if all normals point outwards, fix if not
    flip_ids = np.hstack((cartilage_surf_ids[~cartilage_normal_mask], bone_shell_ids[~bone_normal_mask]))
    if len(flip_ids):
        combined_mesh_flipped = flip_faces(combined_mesh, flip_ids)
        # changed flip faces to carry over point/cell arrays 2026-03-11
    else:
        combined_mesh_flipped = combined_mesh

    # check normal direction
    if not get_outward_normal_mask( 
        combined_mesh_centres[bone_shell_ids], 
        combined_mesh_flipped.cell_normals[bone_shell_ids], 
        bone_mesh
        ).all():
        raise AssertionError('Not all bone normals point outward')
    if not get_outward_normal_mask( 
        combined_mesh_centres[cartilage_surf_ids], 
        combined_mesh_flipped.cell_normals[cartilage_surf_ids], 
        cartilage_mesh
        ).all():
        raise AssertionError('Not all cartilage surface normals point outward')

    # manually check normals
    faces = combined_mesh_flipped.faces.reshape(-1, 4)[:, 1:]
    points = combined_mesh_flipped.points

    v1 = points[faces[:, 1]] - points[faces[:, 0]]
    v2 = points[faces[:, 2]] - points[faces[:, 0]]
    geom_normals = np.cross(v1, v2)
    geom_normals /= np.linalg.norm(geom_normals, axis=1)[:, None]

    dots = np.einsum("ij,ij->i", geom_normals, combined_mesh_flipped['Normals'])
    if not np.min(dots)>0.999:
        raise AssertionError("['Normals'] don't match edge winding")
    # mesh['Normals']doesn't update with edge winding but .cell_normals does
    ################# MESH CHECKS #################

    ################# RETURN MESH #################
    return combined_mesh_flipped #  bone cartilage shared interface
    ################# RETURN MESH #################