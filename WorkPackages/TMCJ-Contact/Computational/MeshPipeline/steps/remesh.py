import numpy as np
import pandas as pd
import gdist
from scipy.spatial.distance import cdist
import subprocess

from phd_helpers.paths import get_boundary

def remesh_surface(bone_mesh, min_df, fine_edge_length, coarse_edge_length, grad_width, remesh_input_path, n_iters=5, adjacent_cells=False):

    print('Computing target edge lengths')
    bone_mesh.point_data['bone_id'] = np.arange(bone_mesh.n_points)
    ar_mesh = bone_mesh.extract_points(min_df['bone_id'], adjacent_cells=adjacent_cells).extract_geometry()
    source_ids = get_boundary(ar_mesh).point_data['bone_id']

    #dists = gdist.compute_gdist(
    #    bone_mesh.points.astype(np.float64),
    #    bone_mesh.faces.reshape(-1, 4)[:, 1:].astype(np.int32),
    #    source_indices=source_ids.astype(np.int32),
    #)

    # -------- swtiching to euclidean distance so that grad region covers 3Dmesh grad region -------- #
    dists = cdist(bone_mesh.points, bone_mesh.points[source_ids]).min(axis=1)


    not_ar_mask = ~np.isin(bone_mesh.point_data['bone_id'], min_df['bone_id'])
    grad_mask = dists < grad_width
    grad_ids = bone_mesh.point_data['bone_id'][not_ar_mask & grad_mask]
    grad_dists = dists[not_ar_mask & grad_mask]

    grad_edge_length = fine_edge_length + (grad_dists / grad_width) * (coarse_edge_length - fine_edge_length)


    # Output csv of per vertex edge lengths
    df = pd.Series(np.full(bone_mesh.n_points, coarse_edge_length, dtype=float))
    df.iloc[grad_ids] = float(grad_edge_length)
    df.iloc[min_df['bone_id']] = float(fine_edge_length)

    print('Writing mesh and target edge lengths to remeshing input directory')
    csv_path = remesh_input_path.with_suffix('.csv')
    df.to_csv(csv_path, index=False, header=False)
    bone_mesh.save(remesh_input_path)

    print('Remeshing')
    
    cgal_path = remesh_input_path.parent.parent.parent
    exe = cgal_path / "bin/sizing_field"

    args = [
        str(exe),
        str(remesh_input_path),  # path to input mesh
        str(csv_path), # path to csv of target edge lengths
        str(fine_edge_length), # edge length of articulation region
        str(coarse_edge_length), # edge length away from articulation region
        str(cgal_path / 'outputs/sf_output' / remesh_input_path.name), # path to output mesh
        str(n_iters), # number of CGAL isotropic remeshing iterations
        ]

    result = subprocess.run(args, text=True)
    result.check_returncode()  # raise after printing