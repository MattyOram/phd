import numpy as np
import pandas as pd
import gdist
import subprocess
from pathlib import Path

def remesh_surface(bone_mesh, min_df, fine_edge_length, coarse_edge_length, grad_width, remesh_input_path, n_iters=5, adjacent_cells=False):

    print('Computing target edge lengths')
    bone_mesh.point_data['bone_id'] = np.arange(bone_mesh.n_points)
    ar_mesh = bone_mesh.extract_points(min_df['bone_id'], adjacent_cells=adjacent_cells).extract_geometry()
    source_ids = ar_mesh.extract_feature_edges(
                            boundary_edges=True, 
                            non_manifold_edges=False, 
                            feature_edges=False, 
                            manifold_edges=False
                            ).point_data['bone_id']

    dists = gdist.compute_gdist(
        bone_mesh.points.astype(np.float64),
        bone_mesh.faces.reshape(-1, 4)[:, 1:].astype(np.int32),
        source_indices=source_ids.astype(np.int32),
    )
    not_ar_mask = ~np.isin(bone_mesh.point_data['bone_id'], min_df['bone_id'])
    grad_mask = dists < grad_width
    grad_ids = bone_mesh.point_data['bone_id'][not_ar_mask & grad_mask]
    grad_dists = dists[not_ar_mask & grad_mask]

    grad_edge_length = fine_edge_length + (grad_dists / grad_width) * (coarse_edge_length - fine_edge_length)


    # Output csv of per vertex edge lengths
    df = pd.Series(np.full(bone_mesh.n_points, coarse_edge_length))
    df.iloc[grad_ids] = grad_edge_length
    df.iloc[min_df['bone_id']] = fine_edge_length

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