import numpy as np
from scipy.optimize import minimize_scalar
import json
import pandas as pd

from phd_helpers.experiments import (
    parse_tekscan, force_per_frame, get_sensor_loc, build_sensor_mesh, project_sensor_new
)
from phd_helpers.AbaqusPostprocessing import inp2pv, get_field_path, get_field_df, add_field_to_mesh, get_history_path


def compute_power_law_coeffs(
    cal_frames,
    sensel_area=1.6129,
    return_fit_points=False,
    mask=0,
    weight_by_patch_size=True
):
    """
    cal_frames: {F1: frame1, ...}
    P = a * r^b

    If weight_by_patch_size=True, minimizes force error.
    If False, minimizes pressure error equally across patches.
    """

    patches = list(cal_frames.values())
    Fs = np.array(list(cal_frames.keys()), dtype=float)

    xs = [
        np.asarray(p, dtype=float)[np.asarray(p, dtype=float) > mask]
        for p in patches
    ]

    n = np.array([len(x) for x in xs], dtype=float)
    area = n * sensel_area
    p = Fs / area

    def S(b):
        return np.array([np.mean(x ** b) for x in xs])

    def a_of_b(b):
        s = S(b)

        if weight_by_patch_size:
            w = area ** 2
            return np.dot(w * p, s) / np.dot(w * s, s)

        return np.dot(p, s) / np.dot(s, s)

    def J(b):
        s = S(b)
        a = a_of_b(b)
        err = a * s - p

        if weight_by_patch_size:
            err = area * err

        return np.sum(err ** 2)

    res = minimize_scalar(J, bounds=(0, 3), method="bounded")

    b = res.x
    a = a_of_b(b)

    if not return_fit_points:
        return a, b

    s = S(b)
    p_fit = a * s
    r_eff = s ** (1 / b)

    fit_points = {
        "raw_eff": r_eff,
        "pressure_actual": p,
        "pressure_fit": p_fit,
        "force": Fs,
        "area": area,
        "n_sensels": n,
    }

    return a, b, fit_points

def compute_polynomial_coeffs(
    cal_frames,
    sensel_area=1.6129,
    return_fit_points=False,
    mask=0,
    degree=3,
    zero_intercept=True,
    weight_by_patch_size=True
):
    """
    cal_frames: {F1: frame1, ...}

    P(r) = c0 + c1*r + c2*r^2 + ... + cn*r^n

    Fit is done at patch level:
        p_i ≈ sum_k c_k * mean(x_i ** k)

    If weight_by_patch_size=True, minimizes force error.
    If False, minimizes pressure error equally across patches.

    Coefficients are returned in ascending order:
        [c0, c1, c2, ..., cn]
    """

    patches = list(cal_frames.values())
    Fs = np.array(list(cal_frames.keys()), dtype=float)

    xs = [
        np.asarray(p, dtype=float)[np.asarray(p, dtype=float) > mask]
        for p in patches
    ]

    n = np.array([len(x) for x in xs], dtype=float)
    area = n * sensel_area
    p = Fs / area

    powers = np.arange(1 if zero_intercept else 0, degree + 1)

    X = np.array([
        [np.mean(x ** k) for k in powers]
        for x in xs
    ])

    if weight_by_patch_size:
        X_fit = X * area[:, None]
        y_fit = p * area
    else:
        X_fit = X
        y_fit = p

    fitted, *_ = np.linalg.lstsq(X_fit, y_fit, rcond=None)

    coeffs = np.zeros(degree + 1)
    coeffs[powers] = fitted

    if not return_fit_points:
        return coeffs

    p_fit = X @ fitted

    fit_points = {
        "raw_mean": np.array([np.mean(x) for x in xs]),
        "pressure_actual": p,
        "pressure_fit": p_fit,
        "force": Fs,
        "area": area,
        "n_sensels": n,
    }

    return coeffs, fit_points

def apply_ab_fit(a, b, frames):
    "returns frames with fit a*frame**b applied"
    return {f: a * frame**b for f, frame in frames.items()}

def apply_poly_fit(coeffs, frames):
    """Returns frames with polynomial fit applied.
    Raw zero sensels remain zero even if the fit has an intercept.
    """
    return {
        f: np.where(
            frame == 0,
            0,
            np.polynomial.polynomial.polyval(frame, coeffs)
        )
        for f, frame in frames.items()
    }

def apply_fit(coeffs, fit_type, frames):
    if fit_type == 'power_law':
        return apply_ab_fit(coeffs[0], coeffs[1], frames)
    if fit_type == 'poly':
        return apply_poly_fit(coeffs, frames)
    else: raise

def compute_hold_end_idx(frames):
    y = frames.sum(axis=(1, 2))
    return  np.argmin(np.gradient(np.gradient(y))) - 1
def compute_hold_start_idx(end_idx, hold_time, spf):
    n_frames = round(hold_time / spf)
    return max(0, end_idx - n_frames)

def get_hold_frames(frames, hold_time, spf, start_idx=None, end_idx=None):
    """Requires that tekscan sensor recording started before and ended after the hold period (with rapid-ish unloading)"""
    if end_idx is None:
        end_idx = compute_hold_end_idx(frames)
    if start_idx is None:
        start_idx = compute_hold_start_idx(end_idx, hold_time, spf)
    return frames[start_idx: end_idx+1]

def avg_frames(frames, raw_mask=0):
    """raw mask sets values <= to raw mask to zero"""
    frame = np.mean(frames, axis=0)
    frame[frame <= raw_mask] = 0
    return frame

def get_avg_hold_frame(hold_frames, start_time, avg_window_time, spf, raw_mask=0):
    """
    hold_frames: tekscan frames during force hold
    start_time: seconds into hold_frames to start the hold window
    avg_window_time: time window over which to average the frames
    """
    n_frames = round(avg_window_time / spf)
    start_frame = round(start_time / spf)
    end_frame = start_frame + n_frames
    frames = hold_frames[start_frame:end_frame+1]
    return avg_frames(frames, raw_mask)

def F_from_path(path):
    return int(path.name.split('N')[0].split('_')[-1])
def F_from_path2(path):
    return int(path.name[3:].split('.')[0].split('_')[0])

def avg_frames_from_paths(paths, Fs, sensor_id, hold_time, start_time, avg_window, custom_masks=[], raw_mask=0, use_F2=False):
    avg_frames = {}
    for path in paths:
        if use_F2:
            F = F_from_path2(path)
        else:
            F = F_from_path(path)

        if F not in Fs:
            continue

        frames, spf = parse_tekscan(path, sensor_id, return_spf=True)
        #•••••••••••• custom mask for noisy cells ••••••••••••#
        for custom_mask in custom_masks:
            mask_value, j, k = custom_mask
            mask = frames[:, j, k] <= mask_value
            frames[mask, j, k] = 0
        #•••••••••••• custom mask for noisy cells ••••••••••••#
        hold_frames = get_hold_frames(frames, hold_time, spf)
        frame = get_avg_hold_frame(hold_frames, start_time, avg_window, spf, raw_mask)

        avg_frames[F] = frame
    return avg_frames


def get_step_ids(csv_dir):
    history_files = list(csv_dir.glob('history*.csv'))
    step_ids = np.sort(np.unique([int(x.with_suffix('').name.replace('history_step-', '')) for x in history_files]))
    return step_ids

def fe_mets_from_mesh(mesh):
    return {
        'CA': mesh.field_data['CA'][0],
        'P_max': mesh.field_data['P_max'][0],
        'P_avg': mesh.field_data['P_avg'][0],
        'COP': mesh.field_data['COP'],
    }

def find_boundary_cells(a):
    """
    input: NxM array of cells
    Returns: NxM (1, 0) array of non-zero cells that border a zero"""
    a = np.asarray(a)
    m = a > 0
    out = np.zeros(a.shape, dtype=bool)

    out[1:, :]  |= m[1:, :]  & (a[:-1, :] == 0)
    out[:-1, :] |= m[:-1, :] & (a[1:, :] == 0)
    out[:, 1:]  |= m[:, 1:]  & (a[:, :-1] == 0)
    out[:, :-1] |= m[:, :-1] & (a[:, 1:] == 0)

    return out

def compute_img_metrics(img, sensel_area=1.6129):

    CA = len(img[img>0]) * sensel_area
    # CA error - between 25% and 125% of the area of the boundary cells 
    # - plenty of cases that could exceed these limits but still reasonable limits I think...
    # - Given the nature of the contacting surfaces, it would be impossible for every boundary cell to be an edge case 
    # - so error bars based on those would give unreasonably wide range
    CA_e_low = find_boundary_cells(img).sum() * sensel_area * 0.75 # lower bound error bar for CA
    CA_e_high = find_boundary_cells(img).sum() * sensel_area * 0.25 # upper bound error bar for CA

    pred_F = force_per_frame(img.reshape(1, 11, 11))[0]
    P_max = img.max()
    P_avg = img[img>0].mean()
    #loc_Pmax = np.array(np.where(img==img.max())).ravel() # idx - (i, j)
    loc_Pmax = np.argwhere(img == img.max()).mean(axis=0) # solves the multiple maximums problem

    return {
        'F_pred': pred_F,
        'CA': CA,
        'CA_e_low': CA_e_low,
        'CA_e_high': CA_e_high,
        'P_max': P_max,
        'P_avg': P_avg,
        'COP': loc_Pmax
    }

def compute_CA_overlap(img_fea, img_tek):
    mask_fea = (img_fea > 0)
    mask_tek = (img_tek > 0)
    CA_mean = (mask_fea.sum() + mask_tek.sum()) / 2
    CA_overlap = (mask_fea & mask_tek).sum()

    return 100-(100 * (CA_mean - CA_overlap) / CA_mean)

def compute_Pavg(mesh):
    """Converts CPRESS to cell data and computes area weighted average pressure"""
    tet_surf = mesh.extract_cells_by_type(10).extract_surface(algorithm=None)
    P = tet_surf.point_data_to_cell_data().cell_data['CPRESS'] 
        # CPRESS comes as point data
        # this assigns to each elements CPRESS value, the average of the CPRESS of the 3 points of the element
        # Edge case: elements has point CPRESS = [0, 0, 1.5], element given CPRESS = 0.5, but entire area considered in contact for avg
        # This slightly overestimates area and therefore reduces average pressure
    A = tet_surf.compute_cell_sizes()["Area"]
    mask = P > 0

    return np.sum(P[mask] * A[mask]) / np.sum(A[mask])


def build_fe_data(inp_file, csv_dir=None):
    param_path = inp_file.parents[4]/ 'params/full_params.json'
    with open(param_path, 'r') as f:
        Fs = json.load(f)['inp']['force_steps']

    if csv_dir is None:
        csv_dir = inp_file.parent / 'resultCSVs' 

    steps = get_step_ids(csv_dir)
    bone = 'tpm'

    frame = -1 # final frame of each step
    field_metrics = ["CPRESS", "U"]

    fe_data = {} # {F1: {'tpm':mesh1, 'mc1':mesh1}, ...} - each mesh contains all fea data

    meshes_orig = inp2pv(inp_file)
    for step in steps[1:]:
        meshes = {
            bone: mesh.copy(deep=True)
            for bone, mesh in meshes_orig.items()
        }
        #meshes = inp2pv(inp_file)
        for bone, mesh in meshes.items():
            instance = f"{bone.upper()}_INST"
            
            # Field data
            for metric in field_metrics:
                field_path = get_field_path(csv_dir, metric, step, frame, instance)
                field_df = get_field_df(field_path)
                add_field_to_mesh(mesh, field_df)

            # History data
            history_data = pd.read_csv(get_history_path(csv_dir, step))
            # A
            CAREA_data = history_data[history_data['historyOutputDescription']=='Total area in contact']
            CA = CAREA_data['value'].iloc[frame]

            mesh.field_data['RF'] = Fs[step+1]
            mesh.field_data['CA'] = CA

            #P_avg = np.mean(mesh['CPRESS'][mesh['CPRESS']>0])
            #Summary data
            mesh.field_data['P_max'] = mesh['CPRESS'].max()
            mesh.field_data['P_avg'] = compute_Pavg(mesh)
            mesh.field_data['COP'] = np.array(mesh.points[np.argmax(mesh['CPRESS'])])

        fe_data[Fs[step+1]] = meshes
    return fe_data

def compute_fe_grid_data(Fs, fe_data, tek_data, bone, guide_wall_z=10, sensor_offset=-2, sensor_offset_y=0):
    fe_grid_data = {}
    for F in Fs:

        surf = fe_data[F][bone].extract_surface(algorithm=None)
        mesh = surf.extract_cells(surf[f'{bone}_CART_SURF']==1).extract_surface(algorithm=None)

        sensor_loc = get_sensor_loc(fe_data[F]['mc1'], guide_wall_z=guide_wall_z, sensor_offset_z=sensor_offset, sensor_size=13.97, sensor_offset_y=sensor_offset_y) # skinny guide ledge
        sensor = build_sensor_mesh(sensor_loc, normal=(1, 0, 0), ncells=11, size=13.97)
        
        fe_grid_new = project_sensor_new(
            mesh=mesh,
            sensor=sensor,
            sensor_vals=tek_data[F],
            data_loc="cells",
            downscale_fea=True,
            return_fea_grid=True,
            active_sensel_width=0.635,
            pressure_name="CPRESS",
        )
        
        
        fe_grid_data[F] = fe_grid_new

    return fe_grid_data



import matplotlib.pyplot as plt
def plot_comparison(test_mets, fe_grid_mets, fe_mets, common_Fs, fe_grid_data, test_frames_mpa):

    nrows, ncols = 2, 2
    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols*6, nrows*5), dpi=200)
    ax = ax.flatten()


    fs = 13
    colors = {
        "tek": "#0097b2",
            #"#009e73"
        "fe down": "#e67a14",
        'fe raw': "#cc79a7"
            #"#cc79a7"
            #"#bfb200"
        }
    markers = {
        "tek": "o", 
        "fe down": "^",
        'fe raw': "^"
    }

    mets = ['P_max', 'P_avg', 'CA']
    mets_datas = [test_mets, fe_grid_mets, fe_mets]
    mets_labels = ['tek', 'fe down', 'fe raw']

    x = list(common_Fs)
    for mets_data, mets_label in zip(mets_datas, mets_labels):
        for i, met in enumerate(mets):
            y = [mets_data[f][met] for f in x]
            label = mets_label if mets_label == 'tek' else None
            ax[i].scatter(x, y, c=colors[mets_label], marker=markers[mets_label], label=mets_label, zorder=2)
            #ax[2].errorbar(F, tek_mets['CA'], yerr=[[tek_mets['CA_e_low']], [tek_mets['CA_e_high']]] , c=colors['tek'], marker=markers['tek'], capsize=5, elinewidth=0.8)

    for f in x:
        ax[3].scatter(f, compute_CA_overlap(fe_grid_data[f], test_frames_mpa[f]), c="#009e73", marker="D")



    ax[0].set_ylabel('Max pressure (MPa)', fontsize=fs)
    ax[1].set_ylabel('Average contact area pressure (MPa)', fontsize=fs)
    ax[2].set_ylabel('Contact area (mm$^2$)', fontsize=fs)
    ax[3].set_ylabel('Contact area overlap %', fontsize=fs)
    for ax_i in ax:
        ax_i.set_xlabel('Reaction force (N)', fontsize=fs)
        ax_i.grid()
    ax[0].legend(fontsize=fs)
    #ax[3].legend(fontsize=fs)
    plt.show()

def plot_sensor_grids(F, fe_grid_data, test_frames_mpa):

    nrows, ncols = 1, 2
    fig, ax = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 5*nrows), dpi=200)

    P_fea, P_tek = fe_grid_data[F][::-1], test_frames_mpa[F][::-1]

    press_min = np.min( [np.min(P_fea), np.min(P_tek)] )
    press_max = np.max( [np.max(P_fea), np.max(P_tek)] )
    clims = (press_min, press_max)

    # imshows
    im = ax[0].imshow(P_fea, vmin=clims[0], vmax=clims[1], cmap="Blues")
    ax[1].imshow(P_tek, vmin=clims[0], vmax=clims[1], cmap='Blues')

    # peak pressure loc
    y, x = compute_img_metrics(P_fea)['COP']
    ax[0].scatter(x, y, color="#cc79a7", s=10, marker='x')
    y, x = compute_img_metrics(P_tek)['COP']
    ax[1].scatter(x, y, color="#cc79a7", s=10, marker='x')

    ax[0].set_title('FE')
    ax[1].set_title('Tekscan')
    fig.suptitle(f'Contact Pressure at {F} N Reaction Force')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.08, orientation='horizontal')
    cbar.set_label('Contact pressure (MPa)')

    plt.show()


#•••••••••••••••••••••••••••••••••• ERRORs ••••••••••••••••••••••••••••••••••#
def compute_mean_error(a, b):
    return np.mean(b - a)

def compute_mean_abs_error(a, b):
    return np.mean(np.abs(b - a))

def compute_mean_pct_error(a, b):
    return 100 * np.mean((b - a) / a)

def compute_mean_abs_pct_error(a, b):
    return 100 * np.mean(np.abs(b - a) / a)

def compute_max_abs_pct_error(a, b):
    ape = 100 * np.abs((b - a) / a)
    i = np.argmax(ape)
    return a[i], ape[i]

def compute_rmse(a, b):
    return np.sqrt(np.mean((b - a)**2))

def compute_nrmse_mean(a, b):
    return 100 * (compute_rmse(a, b) / np.mean(a))


def compute_error(a, b):
    """
    a: experimental results (or instron force)\n
    b: fe results (or tekscan force)
    """
    a, b = np.asarray(a), np.asarray(b)
    return {
        'bias': compute_mean_error(a, b),
        'mae': compute_mean_abs_error(a, b),
        'mpe': compute_mean_pct_error(a, b), 
        'mape': compute_mean_abs_pct_error(a, b),
        'max_ape': compute_max_abs_pct_error(a, b),
        'rmse': compute_rmse(a, b),
        'nrmse_mean_%': compute_nrmse_mean(a, b),
    }

def compute_test_fe_error(metrics, common_Fs, test_mets, fe_mets):
    fe_error = {}
    for metric in metrics:
        a, b = [], []
        for f in common_Fs:
            a.append(test_mets[f][metric])
            b.append(fe_mets[f][metric])
        fe_error[metric] = compute_error(a, b)
    return fe_error

def compute_cop_error(test_mets, fe_mets, common_Fs, tpm_vol, mode=None):
    """
    a: experimental results (or instron force)\n
    b: fe results (or tekscan force)
    """
    a, b = [], []
    for f in common_Fs:
        a.append(test_mets[f]['COP'])
        if mode == 'raw':
            b.append(fe_mets[f]['COP'][1:])
        else:
            b.append(fe_mets[f]['COP'])

    a, b = np.asarray(a), np.asarray(b)
    d_i = np.linalg.norm(a - b, axis=1) / (tpm_vol**(1/3)) 
    return {
        'max': (list(common_Fs)[np.argmax(d_i)], np.max(d_i)),
        'abs_mean': np.mean(d_i),
        'rmse': np.sqrt(np.mean(d_i**2)),
    }
#•••••••••••••••••••••••••••••••••• ERRORs ••••••••••••••••••••••••••••••••••#
def json_converter(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Cannot serialize {type(obj)}")

def save_errors(combinations, cols, errors, run_id):
    pd.DataFrame(combinations, columns=cols).to_csv(f"errors_backups/errors-{run_id}.csv", index=False)
    with open(f"errors_backups/errors-{run_id}.json", "w") as f:
        json.dump(errors, f, default=json_converter)

import re
from itertools import combinations
import pandas as pd
def natural_level_key(x):
    """Order categorical levels according to embedded numeric values."""
    nums = re.findall(r'\d+(?:\.\d+)?', str(x))
    return tuple(float(n) for n in nums)


def paired_parameter_sensitivity(data, parameters, metric):
    results = []

    for param in parameters:
        df = data.copy()

        matching_cols = [p for p in parameters if p != param]

        # Special case
        if param == 'cartilage_material.model':
            matching_cols = [
                p for p in matching_cols
                if p != 'cartilage_material.C10'
            ]
            df = df[df['cartilage_material.C10'] != 0.085]

        # Natural numeric ordering of categorical labels
        levels = sorted(
            df[param].dropna().unique(),
            key=natural_level_key
        )

        for level1, level2 in combinations(levels, 2):

            df1 = df[df[param] == level1]
            df2 = df[df[param] == level2]

            merged = pd.merge(
                df1,
                df2,
                on=matching_cols,
                suffixes=("_1", "_2"),
            )

            changes = (
                merged[f"{metric}_2"]
                - merged[f"{metric}_1"]
            )

            results.append({
                "parameter": param,
                "level_1": level1,
                "level_2": level2,
                "n_pairs": len(changes),
                "max_change": changes.max(),
                "mean_change": changes.mean(),
                "min_change": changes.min(),
            })

    return pd.DataFrame(results)