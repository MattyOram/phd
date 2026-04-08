from pathlib import Path
import pyvista as pv
import numpy as np
import pandas as pd
import gdist
from scipy.spatial.distance import cdist
import json
from tqdm import tqdm

from phd_helpers.MeshQuality import (
        compute_d_metrics, compute_dists, compute_rmsd, sample_surface,
    )
from phd_helpers.paths import get_mesh, get_subject_stl_path, get_boundary

import sys

def build_ids(ids: list):
    id_1, id_2, id_3 = ids

    id_2d = f'-{id_1}'
    id_cart = f'-{id_1}-{id_2}'
    id_3d = f'-{id_1}-{id_2}-{id_3}'
    return id_2d, id_cart, id_3d

def assign_inner(mesh, mesh_with_inner):
    mesh['inner_cells'] = mesh_with_inner['inner_cells'][mesh_with_inner.find_closest_cell(mesh.cell_centers().points)]
    mesh['inner_points'] = np.full(mesh.n_points, 1)
    mesh['inner_points'][np.unique(mesh.faces.reshape(-1, 4)[:, 1:][mesh['inner_cells']==0])] = 0

def build_mesh_dict(path_mesh, ids, bone, stl_path, taper_width):
    id_2d, id_cart, id_3d = build_ids(ids)
    
    path2d = path_mesh / '2Dmesh'
    path3d = path_mesh / '3Dmesh'

    # -------- COMBINED --------------------------------------------- #
    mesh2d = pv.read(path2d / f'bone_cartilage_mesh{id_cart}.vtp')

    mesh3d = pv.read(path3d / f'mesh{id_3d}.vtu')
    mesh3d['mesh3d_point_id'] = np.arange(mesh3d.n_points)
    shell_mesh = mesh3d.extract_cells_by_type(5)

    # -------- BONE --------------------------------------------- #
    bone_orig = get_mesh(stl_path, bone) # no inner cells
    bone_smooth = pv.read(path2d / f'bone_smooth{id_2d}.obj') # no inner cells
    bone_remesh2d = mesh2d.extract_cells(mesh2d['region_id']==2, invert=True).extract_surface(algorithm=None) # no inner cells
    bone_remesh3d = shell_mesh.extract_cells(shell_mesh['region_id']==-2, invert=True).extract_surface(algorithm=None) # no inner cells

    # assign region ids to bone_orig and bone_smooth
    #bone_orig['region_id'] = bone_remesh3d['region_id'][bone_remesh3d.find_closest_cell(bone_orig.cell_centers().points)] * -1
    #bone_smooth['region_id'] = bone_remesh3d['region_id'][bone_remesh3d.find_closest_cell(bone_smooth.cell_centers().points)] * -1

    # compute inner region for bone_remesh2d
    bone_remesh2d['bone_remesh2d_point_id'] = np.arange(bone_remesh2d.n_points)
    bone_remesh2d['bone_remesh2d_cell_id'] = np.arange(bone_remesh2d.n_cells)
    inter_remesh2d = bone_remesh2d.extract_cells(bone_remesh2d['region_id']==3).extract_surface(algorithm=None)
    inter_remesh2d['inter_remesh2d_point_id'] = np.arange(inter_remesh2d.n_points)
    inter_bound2d_ids = get_boundary(inter_remesh2d)['inter_remesh2d_point_id']
    geo_dists = gdist.compute_gdist(
        inter_remesh2d.points.astype(np.float64),
        inter_remesh2d.faces.reshape(-1, 4)[:, 1:].astype(np.int32),
        source_indices=inter_bound2d_ids.astype(np.int32), 
    ) 
    inner_mask = geo_dists > taper_width

    # assign inner cells and points arrays to bone_remesh2d
    bone_remesh2d['inner_points'] = np.zeros(bone_remesh2d.n_points, dtype=int)
    bone_remesh2d['inner_points'][inter_remesh2d['bone_remesh2d_point_id'][inner_mask]] = 1
    bone_remesh2d['inner_cells'] = np.zeros(bone_remesh2d.n_cells, dtype=int)
    bone_remesh2d['inner_cells'][bone_remesh2d.extract_points(bone_remesh2d['inner_points'].astype(bool), adjacent_cells=False)['bone_remesh2d_cell_id']] = 1
    # assign to others
    assign_inner(bone_remesh3d, bone_remesh2d)
    assign_inner(bone_smooth, bone_remesh2d)
    assign_inner(bone_orig, bone_remesh2d)

    # -------- CARTILAGE --------------------------------------------- #
    cart_orig = pv.read(path2d / f'orig_cart_surf{id_cart}.vtp') # inner cells
    cart_smooth = pv.read(path2d / f'smooth_cart_surf{id_cart}.vtp') # inner cells
    cart_remesh2d = mesh2d.extract_cells(mesh2d['region_id']==2) # inner cells
    cart_remesh3d = shell_mesh.extract_cells(shell_mesh['region_id']==-2).extract_surface(algorithm=None) # no inner cells

    # assign inner cells and points arrays to cart_remesh3d
    assign_inner(cart_remesh3d, cart_orig)


    # -------- TETRAHEDRAL --------------------------------------------- #
    cart_tet = mesh3d.extract_cells(mesh3d['region_id']==2)
    bone_tet = mesh3d.extract_cells(mesh3d['region_id']==1)

    # compute inner region of cart_tet #
    # mesh3d ids of points that are part of bone_remesh inner region
    bone_mesh3d_inner_ids = bone_remesh3d['mesh3d_point_id'][np.unique(bone_remesh3d.regular_faces[bone_remesh3d['inner_cells']==1])]
    # mesh3d ids of points that are part of cart_remesh inner region
    cart_mesh3d_inner_ids = cart_remesh3d['mesh3d_point_id'][np.unique(cart_remesh3d.regular_faces[cart_remesh3d['inner_cells']==1])]

    cart_tet_cells = cart_tet['mesh3d_point_id'][cart_tet.cells.reshape(-1, 5)[:, 1:]]
    # mask of cart_tet cells that form the bone_remesh3d inner region faces
    bone_mask = np.isin(cart_tet_cells, bone_mesh3d_inner_ids).sum(axis=1) >= 3
    # mask of cart_tet cells that form the cart_remesh3d inner region faces
    cart_mask = np.isin(cart_tet_cells, cart_mesh3d_inner_ids).sum(axis=1) >= 3

    # get mask of cart_tet cells that are > taper width from the boundary
    boundary_points = inter_remesh2d.points[inter_bound2d_ids]
    d = cdist(cart_tet.cell_centers().points, boundary_points).min(axis=1) # min dist of each tet to boundary 
    tet_mask = d > taper_width
    # get mask of cart_tet cells that don't have a face on the boundary
    cart_shell = cart_tet.extract_surface(algorithm=None)
    cart_shell_mesh3d_ids = cart_shell['mesh3d_point_id'][np.unique(cart_shell.regular_faces)]
    cart_interior_tet_mask = ~(np.isin(cart_tet_cells, cart_shell_mesh3d_ids).sum(axis=1) >= 3)
    # get mask of cart_tet cells that are > taper width from the boundary and don't have a face on the surface
    inner_tet_mask = tet_mask & cart_interior_tet_mask

    # assign inner cells to cart_tet
    cart_tet['inner_cells'] = np.zeros(cart_tet.n_cells, dtype=int)
    cart_tet['inner_cells'][bone_mask | cart_mask | inner_tet_mask] = 1

    # extract inner region
    cart_tet_inner = cart_tet.extract_cells(cart_tet['inner_cells']==1)


    # organise into dictionary

    parts = ['bone', 'cart']

    mesh_dict = {
        'bone':{
            'orig': bone_orig,
            'smooth': bone_smooth,
            'remesh2d': bone_remesh2d,
            'remesh3d': bone_remesh3d
        },
        'cart':{
            'orig': cart_orig,
            'smooth': cart_smooth,
            'remesh2d': cart_remesh2d,
            'remesh3d': cart_remesh3d
        }
    }

    for part in parts:
        part_dict = mesh_dict[part]
        for key, mesh in part_dict.items():
            part_dict[key] = {}
            part_dict[key]['full'] = mesh
            part_dict[key]['inner'] = mesh.extract_points(mesh['inner_points']==1, adjacent_cells=False).extract_surface(algorithm=None)
            part_dict[key]['outer'] = mesh.extract_points(mesh['inner_points']!=1, adjacent_cells=False).extract_surface(algorithm=None)

    mesh_dict['tet'] = {
        'full': mesh3d.extract_cells_by_type(10),
        'bone': bone_tet,
        'cart': cart_tet,
        'cart_inner': cart_tet_inner
    }

    return mesh_dict


# QUALITY #
METRICS = ['min_angle', 'radius_ratio', 'aspect_ratio', 'scaled_jacobian']
ACCEPTABLE_RANGE = {
    'min_angle': (10, 70.53),
    'radius_ratio': (1.0, 5.0),
    'aspect_ratio': (1.0, 5.0),
    'scaled_jacobian': (0.2, 1.0)
}
PREFERRED_RANGE = {
    'min_angle': (15, 70.53),
    'radius_ratio': (1.0, 3.0),
    'aspect_ratio': (1.0, 3.0),
    'scaled_jacobian': (0.4, 1.0)
}
IDEAL_VALUE = {
    'min_angle': 70.53,
    'radius_ratio': 1.0,
    'aspect_ratio': 1.0,
    'scaled_jacobian': 1.0
}
BAD_SIDE = {
    'min_angle': 'low',
    'radius_ratio': 'high',
    'aspect_ratio': 'high',
    'scaled_jacobian': 'low',
}

def compute_quality_metrics(mesh, dic=None, label=''):
    quality = mesh.cell_quality(METRICS)

    if dic is None:
        dic = {}

    dic |= {
    f"{label}n_cells": mesh.n_cells,
    }

    for metric in METRICS:
        # handle NAN values
        vals = np.asarray(quality[metric], dtype=float)
        finite = np.isfinite(vals)
        n_invalid = np.sum(~finite)
        vals = vals[finite]
        if len(vals) == 0:
            dic |= {
                f"{label}{metric}_n_nan": int(n_invalid),
                f"{label}{metric}_bad_cells": mesh.n_cells,
            }
            continue


        ideal = IDEAL_VALUE[metric]

        # distance from ideal
        dists = np.abs(vals - ideal)
        # closest to ideal
        best_val = vals[np.argmin(dists)]
        # furthest from ideal
        worst_val = vals[np.argmax(dists)]

        # % within preferred range
        vmin_p, vmax_p = PREFERRED_RANGE[metric]
        within_p = np.logical_and(vals >= vmin_p, vals <= vmax_p)
        pct_within_p = 100 * np.sum(within_p) / mesh.n_cells

        # % within acceptable range
        vmin_a, vmax_a = ACCEPTABLE_RANGE[metric]
        within_a = np.logical_and(vals >= vmin_a, vals <= vmax_a)
        pct_within_a = 100 * np.sum(within_a) / mesh.n_cells

        # count outside the acceptable range
        outside_count = (len(vals) - np.sum(within_a)) + n_invalid

        if BAD_SIDE[metric] == 'low':
            # worst below ideal = 5th pct
            pct_95 = np.percentile(vals, 5)
            pct_99 = np.percentile(vals, 1)
            pct_999 = np.percentile(vals, 0.1)
        else:
            # worst above ideal  = 95th percentile
            pct_95 = np.percentile(vals, 95)
            pct_99 = np.percentile(vals, 99)
            pct_999 = np.percentile(vals, 99.9)

# percentages are over all mesh cells
# but summary stats (mean, median, std, percentiles) are over valid cells only
# often nan values so need to keep eye on n_nan count
        data = ({
            f"{label}{metric}_n_nan": int(n_invalid),
            f"{label}{metric}_mean": vals.mean(),
            f"{label}{metric}_median": np.median(vals),
            f"{label}{metric}_std": np.std(vals),
            f"{label}{metric}_best": best_val,
            f"{label}{metric}_worst": worst_val,
            f"{label}{metric}_preferred_range_pct": pct_within_p,
            f"{label}{metric}_acceptable_range_pct": pct_within_a,
            f"{label}{metric}_bad_cells": outside_count,
            f"{label}{metric}_95%": pct_95,
            f"{label}{metric}_99%": pct_99,
            f"{label}{metric}_99.9%": pct_999,
        })

        dic |= data

    return dic



# MAIN #
if __name__ == "__main__":
# -------- PATHS --------------------------------------------- #
    root_dir = Path(sys.argv[1]) # path to output_root in set_parameters
    study_name = root_dir.name 

    out_dir = Path(sys.argv[2])  # path dir to save outputs in
    out_dir.mkdir(parents=True, exist_ok=True)

    param_file = sys.argv[3] # name of full params file to use - full_params.json
    pf_id = '-0' if '-' not in param_file else '-'+param_file.split('.')[0].split('-')[-1]
    
    u_id = '' # e.g. a
    if len(sys.argv) > 4:
        u_id = sys.argv[4]


    # -------- PARAMS --------------------------------------------- #
    params_path = root_dir / f'params/{param_file}'
    with open(params_path, 'r') as f:
        params = json.load(f)

    taper_width = params['cartilage']['taper_width']
    id_2d, id_cart = 0, 0
    subs = params['subjects']['subject_sideL']
    bone_pairs = params['subjects']['bone_arbone']
    parts = ['bone', 'cart']
    n_samples = 20000

    # -------- MAIN --------------------------------------------- #
    data = {
        'bone': [],
        'cart': [],
        'qual': []
    }
    for sub in subs:

        subject, sideL = sub[:-1], sub[-1]
        stl_path = get_subject_stl_path(subject, sideL)

        for bone_pair in bone_pairs:

            bone, arbone = bone_pair.split('-')
            path_mesh = root_dir / f'meshes/{subject}{sideL}/{bone_pair}'
            run_ids = np.sort([int(p.name.split('-')[-1][:-4]) for p in (path_mesh / '3Dmesh').iterdir() if p.suffix == '.vtu'])

            for run_id in tqdm(run_ids):

                mesh_dict = build_mesh_dict(path_mesh, [id_2d, id_cart, run_id], bone, stl_path, taper_width)

                for part in parts: # bone, cart
                    
                    # create data row
                    mets = {
                        'sub': sub,
                        'bone': bone,
                        'run_id': str(run_id)+pf_id+u_id
                    }
                    
                    mesh = mesh_dict[part]
                    remesh3d = mesh['remesh3d']
                    full_3d = remesh3d['full']
                    inner_3d = remesh3d['inner']
                    outer_3d = remesh3d['outer']

                    # distance from remesh3d
                    for name, regions in mesh.items(): # orig, smooth, remesh2d ,remesh3d
                        if name != 'remesh3d':
                            full = regions['full']
                            inner = regions['inner']
                            outer = regions['outer']
                            
                            di_ab = compute_dists(sample_surface(inner_3d, n_samples), full)
                            di_ba = compute_dists(sample_surface(inner, n_samples), full_3d)
                            di = np.hstack( (di_ab, di_ba) )
                        
                            do_ab = compute_dists(sample_surface(outer_3d, n_samples), full)
                            do_ba = compute_dists(sample_surface(outer, n_samples), full_3d)
                            do = np.hstack( (do_ab, do_ba) )

                            label = f'{name}_'
                            mets[label + 'rmsdi'] = compute_rmsd(di)
                            mets = compute_d_metrics(di, mets, label + 'di_')
                            mets[label + 'rmsdo'] = compute_rmsd(do)
                            mets = compute_d_metrics(do, mets, label + 'do_')

                            # height of inner cartilage for: orig smooth remesh2d
                            if part == 'cart':
                                bone_full = mesh_dict['bone']['smooth']['full']
                                h = compute_dists(sample_surface(inner, n_samples), bone_full)
                                mets[label + 'h_rmsd'] = compute_rmsd(h)
                                mets = compute_d_metrics(h, mets, label + 'h_')

                    # height of cartilage for remesh3d
                    if part == 'cart':
                        bone_full = mesh_dict['bone']['remesh3d']['full']
                        h = compute_dists(sample_surface(inner_3d, n_samples), bone_full)
                        mets['remesh3d_h_rmsd'] = compute_rmsd(h)
                        mets = compute_d_metrics(h, mets, 'remesh3d_h_')

                    # only store volume of bone remesh3d
                    if part == 'bone':
                        mets['remesh3d_vol'] = full_3d.volume

                    data[part].append(mets)

                # compute quality of tetrahedral mesh
                row = {
                    'sub': sub,
                    'bone': bone,
                    'run_id': str(run_id)+pf_id+u_id,
                    'total_vol': sum([x.volume for x in mesh_dict['tet'].values()])
                }
                for name, tet in mesh_dict['tet'].items(): # bone cart cart_inner
                    row = compute_quality_metrics(tet, row, name + '_')
                data['qual'].append(row)

    # write to file
    for key, value in data.items():
        pd.DataFrame(data[key]).to_csv(out_dir / f'{study_name+pf_id+u_id}-{key}Metrics.csv', index=False)
                    