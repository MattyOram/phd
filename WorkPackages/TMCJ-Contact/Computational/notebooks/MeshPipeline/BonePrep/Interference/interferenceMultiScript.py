from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from phd_helpers.paths import (
    pose2idCMC, get_bone_transform, transform_mesh, get_task_stl_paths, get_mesh, get_info_df, get_trimesh, get_info
)

def process_stl_path(stl_path, savedir, poses, bones, bone_pairs, smooth_iters):
    subject, sideL = get_info(stl_path)
    key = f"{subject}-{sideL}"
    savepath = Path(savedir) / f"{subject}{sideL}.csv"

    if savepath.is_file():
        return f"skip {key}"

    print(f"processing {key}")

    # Neutral bone meshes
    meshes_neu = [
        get_mesh(stl_path, bone).smooth_taubin(n_iter=smooth_iters)
        for bone in bones
    ]

    # Avoid repeated np.where in inner loop
    bone_idx = {bone: i for i, bone in enumerate(bones)}

    intersections = {pose: [] for pose in poses}

    for pose in poses:
        pose_id = pose2idCMC(pose)

        # Transform bones
        meshes_posed = []
        for mesh, bone in zip(meshes_neu, bones):
            try:
                R, t = get_bone_transform(stl_path, bone, pose_id)
            except Exception:
                R, t = np.eye(3), np.zeros(3)

            meshes_posed.append(transform_mesh(mesh, R, t))

        # Check interference
        for bone1, bone2 in bone_pairs:
            mesh1 = meshes_posed[bone_idx[bone1]]
            mesh2 = meshes_posed[bone_idx[bone2]]

            trimesh1 = get_trimesh(mesh1)
            trimesh2 = get_trimesh(mesh2)

            intersection = trimesh1.intersection(trimesh2)
            intersections[pose].append(intersection.volume > 0)

    # Save per-subject CSV
    subject_df = pd.DataFrame(intersections)
    subject_df["bone_pairs"] = [f"{a}-{b}" for a, b in bone_pairs]
    cols = ["bone_pairs"] + [c for c in subject_df.columns if c != "bone_pairs"]
    subject_df[cols].to_csv(savepath, index=False)

    return f"done {key}"


################# DATA #################
info = get_info_df()
cmc_info = info[info.group=='CMC']
stl_paths = get_task_stl_paths('CMC')
################# DATA #################

bones = np.array(['rad', 'uln', 'sca', 'lun', 'trq', 'pis', 'tpd',
                        'tpm', 'cap', 'ham', 'mc1', 'mc2', 'mc3', 'mc4', 'mc5'])
poses = ['adduction','abduction','flexion','extension','pinch','grasp',
                            'jar','pinch_load','grasp_load','jar_load','neutral']

smooth_iters = 50

bone_pairs = [ # articulating bone pairs
    ['tpm', 'mc1'],
    ['tpm', 'sca'],
    ['tpm', 'mc2'],
    ['tpm', 'tpd'],
    ['rad', 'uln'],
    ['rad', 'sca'],
    ['rad', 'lun'],
    ['sca', 'tpd'],
    ['sca', 'lun'],
    ['sca', 'cap'],
    ['lun', 'cap'],
    ['lun', 'ham'],
    ['lun', 'trq'],
    ['tpd', 'mc2'],
    ['tpd', 'cap'],
    ['cap', 'ham'],
    ['cap', 'mc2'],
    ['cap', 'mc3'],
    ['cap', 'mc4'],
    ['trq', 'ham'],
    ['trq', 'pis'],
    ['ham', 'mc4'],
    ['ham', 'mc5'],
    ['mc1', 'mc2'],
    ['mc2', 'mc3'],
    ['mc3', 'mc4'],
    ['mc4', 'mc5']
]


def main():
    savedir = Path("Interference-smooth")
    savedir.mkdir(parents=True, exist_ok=True)

    max_workers = max(1, mp.cpu_count() - 1)

    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp.get_context("spawn"),  # safer for mesh libs
    ) as ex:
        futures = [
            ex.submit(
                process_stl_path,
                stl_path,
                savedir,
                poses,
                bones,
                bone_pairs,
                smooth_iters,
            )
            for stl_path in stl_paths
        ]

        for fut in tqdm(as_completed(futures), total=len(futures)):
            print(fut.result())


if __name__ == "__main__":
    main()