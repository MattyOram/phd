from pathlib import Path
import numpy as np
import pandas as pd
import json


#root_dir = Path('../../../../MeshPipeline/outputs/ParamOptimisation/criteria3D') # path to output_root in set_parameters
#out_path = Path('outputs/study1-meshFix') # path dir to save outputs in

#study_prefix = 'study1' # start of dir name of output_root in set_parameters
#studies = ['a', 'b', 'c'] # individual study identifier (end of dir name of output_root in set_parameters)

def get_params3d(root_dir, study_prefix='study1', studies=('a', 'b', 'c'), extra_params=None, full_param_file='full_params.json'):
    """Now also works for 2d and cartilage steps"""
    root_dir = Path(root_dir)
    extra_params = extra_params or []

    def get_nested(d, path, default=None):
        value = d
        for key in path.split('.'):
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        return value

    rows = []
    for study in studies:
        study_name = study_prefix + study
        study_dir = root_dir / study_name

        params_path = study_dir / f'params/{full_param_file}'
        pf_id = '-0' if '-' not in full_param_file else '-'+full_param_file.split('.')[0].split('-')[-1]
        with open(params_path, 'r') as f:
            params = json.load(f)

        subs = params['subjects']['subject_sideL']
        bone_pairs = params['subjects']['bone_arbone']

        for sub in subs:
            subject, sideL = sub[:-1], sub[-1]

            for bone_pair in bone_pairs:
                bone, arbone = bone_pair.split('-')
                path_mesh = study_dir / f'meshes/{subject}{sideL}/{bone_pair}'
                run_ids = np.sort([
                    p.name.lstrip('mesh-').rstrip('.vtu')
                    for p in (path_mesh / '3Dmesh').iterdir()
                    if p.suffix == '.vtu'
                ])

                for run_id in run_ids:
                    id1, id2, id3 = run_id.split('-')
                    param3d_path = root_dir / f'{study_name}/params/3Dmesh/{id3}.json'
                    with open(param3d_path, 'r') as f:
                        param3d = json.load(f)
                    param_cart_path = root_dir / f'{study_name}/params/cartilage/{id2}.json'
                    with open(param_cart_path, 'r') as f:
                        param_cart = json.load(f)
                    param2d_path = root_dir / f'{study_name}/params/cartilage/{id1}.json'
                    with open(param2d_path, 'r') as f:
                        param2d = json.load(f)

                    row = {
                        'sub': sub,
                        'bone': bone,
                        'run_id': run_id+pf_id+study,
                    }

                    for path in extra_params:
                        row[path] = get_nested(param3d, path)
                    if '_loop' in param3d:
                        for param, val in param3d['_loop'].items():
                            row[param] = val
                    if '_loop' in param_cart:
                        for param, val in param_cart['_loop'].items():
                            row[param] = val
                    if '_loop' in param2d:
                        for param, val in param2d['_loop'].items():
                            row[param] = val

                    rows.append(row)

    return pd.DataFrame(rows)


def get_runtimes(root_dir, study_prefix='study1', studies=['a', 'b', 'c'], full_param_file=None):
    root_dir = Path(root_dir)
    
    run_dfs = []
    for study in studies:
        study_name = study_prefix + study
        study_dir = root_dir / study_name
        runtimes = pd.read_json(study_dir / 'reports/runtimes.jsonl', lines=True)
        if full_param_file is not None:
            runtimes = runtimes[runtimes['full_params']==full_param_file]

        runtimes = runtimes[runtimes['step']=='3Dmesh'].copy()
        runtimes['bone'] = runtimes['bones'].apply(lambda x: x.split('-')[0])
        runtimes['run_id'] = runtimes['run_ids'].apply(lambda x: '-'.join([str(y) for y in x]))
        runtimes['pf_id'] = runtimes['full_params'].apply(lambda x: '-0' if '-' not in x else '-'+x.split('.')[0].split('-')[-1])
        runtimes['run_id'] = runtimes['run_id'] + runtimes['pf_id'] + study
        run_dfs.append(runtimes[['subject', 'bone', 'run_id', 'runtime']].copy())
    return pd.concat(run_dfs)


def combine_metric_dfs(out_dir, prefix = ['study1a', 'study1b', 'study1c']):
    df_bone, df_cart, df_qual = [], [], []
    for p in prefix:
        df_bone.append(pd.read_csv(out_dir / f'{p}-boneMetrics.csv'))
        df_cart.append(pd.read_csv(out_dir / f'{p}-cartMetrics.csv'))
        df_qual.append(pd.read_csv(out_dir / f'{p}-qualMetrics.csv'))
    df_bone = pd.concat(df_bone).copy()
    df_cart = pd.concat(df_cart).copy()
    df_qual = pd.concat(df_qual).copy()
    #df_qual['total_tets'] = df_qual['bone_n_cells'] + df_qual['cart_n_cells']

    return df_bone, df_cart, df_qual

def score_value(x, ideal, acceptable, linear_floor=0.5, decay=5.0):
    """
    Score values in [0, 1] with:
      - score = 1 at and beyond the ideal region
      - linear scaling between acceptable and ideal
      - exponential drop-off beyond acceptable toward 0"""

    x = np.asarray(x, dtype=float)
    score = np.empty_like(x, dtype=float)

    if acceptable <= ideal: # bigger better
        # ideal or better -> 1
        mask_ideal = x >= ideal
        score[mask_ideal] = 1.0

        # acceptable to ideal -> linear from linear_floor to 1
        mask_linear = (x >= acceptable) & (x < ideal)
        score[mask_linear] = linear_floor + (
            (x[mask_linear] - acceptable) / (ideal - acceptable)
        ) * (1.0 - linear_floor)

        # below acceptable -> exponential decay from linear_floor toward 0
        mask_exp = x < acceptable
        score[mask_exp] = linear_floor * np.exp(
            decay * (x[mask_exp] - acceptable) / (ideal - acceptable)
        )

    else:
        # ideal or better -> 1
        mask_ideal = x <= ideal
        score[mask_ideal] = 1.0

        # ideal to acceptable -> linear from 1 to linear_floor
        mask_linear = (x > ideal) & (x <= acceptable)
        score[mask_linear] = linear_floor + (
            (acceptable - x[mask_linear]) / (acceptable - ideal)
        ) * (1.0 - linear_floor)

        # above acceptable -> exponential decay from linear_floor toward 0
        mask_exp = x > acceptable
        score[mask_exp] = linear_floor * np.exp(
            -decay * (x[mask_exp] - acceptable) / (acceptable - ideal)
        )

    score = np.clip(score, 0.0, 1.0)

    if score.ndim == 0:
        return float(score)
    return score