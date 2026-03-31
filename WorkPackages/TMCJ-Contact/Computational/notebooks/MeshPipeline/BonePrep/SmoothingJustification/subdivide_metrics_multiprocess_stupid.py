from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from phd_helpers.paths import get_mesh, avg_edge_length, get_task_stl_paths, get_info


def compute_smoothing_metrics(og_mesh, mesh_ss, n_iters, return_max_idx=False):
    """
    og_mesh: Original DB mesh
    mesh_ss: mesh to start smoothing from
    n_iters: array of smoothing iterations
    """
    vol = og_mesh.volume
    area = og_mesh.area

    metrics = {
        'Vsmooth': [],
        'Vchange': [],
        'Vchange_pct': [],
        'Asmooth': [],
        'Achange': [],
        'Achange_pct': [],
        'RMSD': [],
        'AbsMaxD': [],
        'MeanDist': [],

        'Vchange_self': [],
        'Vchange_pct_self': [],
        'Achange_self': [],
        'Achange_pct_self': [],
        'RMSD_self': [],
        'AbsMaxD_self': [],
        'MeanDist_self': [],

        'p2p_RMSD_self': [],
        'p2p_AbsMaxD_self': [],
        
    }

    mesh_b = mesh_ss
    for n_iter in n_iters:
        if n_iter == 0:
            mesh_smooth = mesh_ss.copy(deep=True)
        else:
            mesh_smooth = mesh_ss.smooth_taubin(n_iter=n_iter)

        # change relative to og_mesh
        metrics['Vsmooth'].append(mesh_smooth.volume)
        metrics['Vchange'].append(metrics['Vsmooth'][-1] - vol)
        metrics['Vchange_pct'].append(metrics['Vchange'][-1] / vol * 100)

        metrics['Asmooth'].append(mesh_smooth.area)
        metrics['Achange'].append(metrics['Asmooth'][-1] - area)
        metrics['Achange_pct'].append(metrics['Achange'][-1] / area * 100)

        #dists = og_mesh.compute_implicit_distance(mesh_smooth)['implicit_distance']
        _, ps = mesh_smooth.find_closest_cell(og_mesh.points, return_closest_point=True)
        dists = np.linalg.norm(og_mesh.points - ps, axis=1)
        metrics['RMSD'].append(np.sqrt(np.mean(dists**2)))
        metrics['AbsMaxD'].append(np.max(np.abs(dists)))
        max_dist_idx = np.argmax(np.abs(dists))
        metrics['MeanDist'].append(np.mean(dists))

        # change relative to mesh from previous smoothing step
        #dists_self = mesh_smooth.compute_implicit_distance(mesh_b)['implicit_distance']
        _, ps_self = mesh_b.find_closest_cell(mesh_smooth.points, return_closest_point=True)
        dists_self = np.linalg.norm(mesh_smooth.points - ps_self, axis=1)
        metrics['RMSD_self'].append(np.sqrt(np.mean(dists_self**2)))
        metrics['AbsMaxD_self'].append(np.max(np.abs(dists_self)))

        metrics['Vchange_self'].append(metrics['Vsmooth'][-1] - mesh_b.volume)
        metrics['Vchange_pct_self'].append(metrics['Vchange_self'][-1] / mesh_b.volume * 100)

        metrics['Achange_self'].append(metrics['Asmooth'][-1] - mesh_b.area)
        metrics['Achange_pct_self'].append(metrics['Achange_self'][-1] / mesh_b.area * 100)

        metrics['MeanDist_self'].append(np.mean(dists_self))

        # compute distance metrics between each corresponding point for each smoothin step
        points_a = mesh_smooth.points
        points_b = mesh_b.points
        dists_p2p = np.linalg.norm(points_b - points_a, axis=1)
        metrics['p2p_RMSD_self'].append(np.sqrt(np.mean(dists_p2p**2)))
        metrics['p2p_AbsMaxD_self'].append(np.max(dists_p2p))

        mesh_b = mesh_smooth.copy(deep=True)
        del mesh_smooth


    if return_max_idx:
        return metrics, max_dist_idx
    return metrics


stl_paths = get_task_stl_paths('CMC')[1:]

bones = np.array([
    'rad', 'uln', 'sca', 'lun', 'trq', 'pis', 'tpd',
    'tpm', 'cap', 'ham', 'mc1', 'mc2', 'mc3', 'mc4', 'mc5'
])

ledges = [ # max_edge_length
    0.3,  # avg_edge_length ~0.20mm    ~55,000  cells
    0.15, # avg_edge_length ~0.10mm    ~220,000 cells
    0.075 # avg_edge_length ~0.05mm    ~850,000 cells
]

its = 50
n_iters = np.array([50, 60, 70, 80, 90, 100, 150, 200]) - its


def process_stl_path(sp):
    sp = Path(sp)  # ensure it is a Path inside worker
    print(sp.parent.name)
    
    sbjt, sdL = get_info(sp)

    metrics_list = []

    for bn in bones:
        msh = get_mesh(sp, bn)  # load once per bone, not once per Ledge
        for Ledge in ledges:
            msh_ss = msh.smooth_taubin(n_iter=its).subdivide_adaptive(max_edge_len=Ledge)
            mets = compute_smoothing_metrics(msh, msh_ss, n_iters)


            mets['n_iter'] = n_iters + its
            mets['max_edge_length'] = [Ledge] * len(n_iters)
            mets['edge_length'] = [avg_edge_length(msh_ss)] * len(n_iters)
            mets['subject'] = [sbjt] * len(n_iters)
            mets['sideL'] = [sdL] * len(n_iters)
            mets['bone'] = [bn] * len(n_iters)

            metrics_list.append(pd.DataFrame(mets))

    out_df = pd.concat(metrics_list, ignore_index=True)

    sbjt, sdL = get_info(sp)
    out_path = Path("metric_dfs/subdivide") / f"{sbjt}{sdL}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    return str(sp), str(out_path)


if __name__ == "__main__":
    stl_paths_list = [str(p) for p in stl_paths]  # safer for multiprocessing

    with ProcessPoolExecutor(max_workers=6) as ex:
        futures = [ex.submit(process_stl_path, sp) for sp in stl_paths_list]

        for fut in tqdm(as_completed(futures), total=len(futures)):
            sp, out_path = fut.result()
            print(f"done: {sp} -> {out_path}")