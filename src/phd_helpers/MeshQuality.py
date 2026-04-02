import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt
import trimesh

from phd_helpers.paths import avg_edge_length

METRICS = [
    "min_angle",
    "max_angle",
    "radius_ratio",
    "shape",
    "aspect_ratio",
    #"relative_size_squared",
    "scaled_jacobian",
    "aspect_frobenius",
    "condition",
    #"distortion"
    ]

def get_acceptable_ranges(cell_type):
    ACCEPTABLE_RANGES_TETS = {
        #'min_angle': (30, 60),
        'min_angle': (10, 70.53),
        'max_angle': (70.53, 170),
        'radius_ratio': (0.4, 1.0),
        'shape': (0.4, 1.0),
        'aspect_ratio': (1.0, 3.0),
        #'relative_size_squared': (0.4, 1),
        'scaled_jacobian': (0.2, 1.0),
        'equiangle_skew': (0.4, 1.0),
        "aspect_frobenius": (1.0, 2.0),
        "condition": (1.0, 3.0),
        #"distortion": (0.5, 1.0)
    }

    ACCEPTABLE_RANGES_TRIS = {
        #'min_angle': (30, 60),
        'min_angle': (30, 60),
        'max_angle': (60, 120),
        'radius_ratio': (0.4, 1.0),
        'shape': (0.4, 1.0),
        'aspect_ratio': (1.0, 3.0),
        #'relative_size_squared': (0.4, 1),
        'scaled_jacobian': (0.4, 1.0),
        'equiangle_skew': (0.4, 1.0),
        "aspect_frobenius": (1.0, 2.0),
        "condition": (1.0, 3.0),
        #"distortion": (0.5, 1.0)
    }

    if cell_type == 'tri':
        return ACCEPTABLE_RANGES_TRIS
    elif cell_type =='tet':
        return ACCEPTABLE_RANGES_TETS
    else: raise KeyError('Cell type values: "tri", "tet"')

def get_ideal_values(cell_type):
    ACCEPTABLE_RANGES = get_acceptable_ranges(cell_type)

    IDEAL_VALUES = {
        "min_angle": ACCEPTABLE_RANGES['min_angle'][1],
        "max_angle": ACCEPTABLE_RANGES['max_angle'][0],
        "radius_ratio": ACCEPTABLE_RANGES['radius_ratio'][1],
        "shape": ACCEPTABLE_RANGES['shape'][1],
        "aspect_ratio": ACCEPTABLE_RANGES['aspect_ratio'][0],
        #"relative_size_squared":ACCEPTABLE_RANGES['relative_size_squared'][1],
        "scaled_jacobian": ACCEPTABLE_RANGES['scaled_jacobian'][1],
        "equiangle_skew": ACCEPTABLE_RANGES['equiangle_skew'][1],
        "aspect_frobenius": ACCEPTABLE_RANGES['aspect_frobenius'][0],
        "condition": ACCEPTABLE_RANGES['condition'][0],
        #"distortion": ACCEPTABLE_RANGES['distortion'][1]
    }
    return IDEAL_VALUES

def get_worst_acceptable(cell_type):
    ACCEPTABLE_RANGES = get_acceptable_ranges(cell_type)

    WORST_ACCEPTABLE = {
        "min_angle": ACCEPTABLE_RANGES['min_angle'][0],
        "max_angle": ACCEPTABLE_RANGES['max_angle'][1],
        "radius_ratio": ACCEPTABLE_RANGES['radius_ratio'][0],
        "shape": ACCEPTABLE_RANGES['shape'][0],
        "aspect_ratio": ACCEPTABLE_RANGES['aspect_ratio'][1],
        #"relative_size_squared":ACCEPTABLE_RANGES['relative_size_squared'][0],
        "scaled_jacobian": ACCEPTABLE_RANGES['scaled_jacobian'][0],
        "equiangle_skew": ACCEPTABLE_RANGES['equiangle_skew'][0],
        "aspect_frobenius": ACCEPTABLE_RANGES['aspect_frobenius'][1],
        "condition": ACCEPTABLE_RANGES['condition'][1],
        #"distortion": ACCEPTABLE_RANGES['distortion'][0]
    }
    return WORST_ACCEPTABLE

# missing ratio of traingle to equilateral triangle with same ascribed circle - but maybe shape does same thing

def check_mesh_quality(mesh: pv.PolyData, cell_type='tri'):

    quality = mesh.cell_quality(METRICS)
    if cell_type == 'tet': # pyvsita dihedral angle is weird gives values > 70.53
        quality['min_angle'], quality['max_angle'] = dihedral_angles(mesh.points, mesh.cells_dict[10])
    if cell_type == 'tri':
        # equiangle skew
        alphas = quality['min_angle']
        betas = quality['max_angle']
        quality['equiangle_skew'] = np.minimum(
            alphas / 60,
            (180 - betas) / (180 - 60)
        )
    quality['radius_ratio'] = 1 / quality['radius_ratio']

    # remove any unwanted cell_data arrays
    for key in np.array(quality.cell_data.keys())[~np.isin(quality.cell_data.keys(), METRICS + ['equiangle_skew'])]:
        quality.cell_data.remove(key)

    return quality

def plot_mesh_quality(quality, bins=40, return_fig=False, cell_type='tri'):
    ACCEPTABLE_RANGES = get_acceptable_ranges(cell_type)
    WORST_ACCEPTABLE = get_worst_acceptable(cell_type)

    metrics = quality.cell_data.keys()

    n_rows = np.ceil(len(metrics) / 4).astype(int)
    
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, n_rows*5))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        vals = quality[metric]
        ax = axes[i]

        # histogram
        ax.hist(vals, bins=bins, edgecolor='black', alpha=0.7)
        ax.set_title(metric)
        ax.set_xlabel(metric)
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

        vmin, vmax = ACCEPTABLE_RANGES[metric]
        ax.axvspan(vmin, vmax, color='green', alpha=0.2, label='Acceptable range')
        ax.legend()

        # Plot worst acceptable threshold as red vertical line
        w = WORST_ACCEPTABLE[metric]
        ax.axvline(w, color='red', linestyle='--', linewidth=2, label="Worst acceptable")

    # Hide any empty axs
    for i in range(len(metrics), 8):
        axes[i].axis('off')

    plt.tight_layout()

    if return_fig:
        return fig
    else:
        plt.show()

def mesh_quality_summary(quality, cell_type='tri'):
    ACCEPTABLE_RANGES = get_acceptable_ranges(cell_type)
    IDEAL_VALUES = get_ideal_values(cell_type)

    metrics = quality.cell_data.keys()

    rows = []
    for metric in metrics:
        vals = quality[metric]
        ideal = IDEAL_VALUES[metric]

        # distance from ideal
        dists = np.abs(vals - ideal)
        # closest to ideal
        best_val = vals[np.argmin(dists)]
        # furthest from ideal
        worst_val = vals[np.argmax(dists)]

        # % within threshold
        vmin, vmax = ACCEPTABLE_RANGES[metric]
        within = np.logical_and(vals >= vmin, vals <= vmax)
        pct_within = 100 * np.sum(within) / len(vals)

        # count outside the ideal range
        outside_count = len(vals) - np.sum(within)
        if outside_count:
            print(metric, f'{outside_count}/{len(vals)} cells outside of acceptable range ({vmin}, {vmax})')

        if worst_val < ideal:
            # worst below ideal = 5th pct
            pct_95 = np.round((np.percentile(vals, 5), best_val), 2)
        else:
            # worst above ideal  = 95th percentile
            pct_95 = np.round((best_val, np.percentile(vals, 95)), 2)

        rows.append({
            "metric": metric,
            "mean": vals.mean(),
            "best": best_val,
            "worst": worst_val,
            "acceptable_range": ACCEPTABLE_RANGES[metric],
            "acceptable_range_pct": pct_within,
            "bad_cells": outside_count,
            "95%": ( float(pct_95[0]), float(pct_95[1]) )
        })

    df = pd.DataFrame(rows)
    return df.set_index('metric')

def plot_bad_cells(mesh, quality, metric, cell_type='tri'):
    ACCEPTABLE_RANGES = get_acceptable_ranges(cell_type)
    unacceptable = quality.extract_values(scalars=metric, ranges=ACCEPTABLE_RANGES[metric], invert=True)
    print(metric)
    print('Bad cells = ', unacceptable.n_cells)

    if unacceptable.n_cells:
        pl = pv.Plotter()
        pl.add_mesh(mesh, style='wireframe', color='light gray')
        pl.add_mesh(unacceptable, color='lime')
        pl.view_xy()
        pl.camera.zoom(1.5)
        pl.show()

def plot_bad_cells2(mesh, quality, metric, range=(0, 0.4)):
    """choose own range for plotted cells"""
    unacceptable = quality.extract_values(scalars=metric, ranges=range, invert=False)
    print(metric)
    print('Bad cells = ', unacceptable.n_cells)

    if unacceptable.n_cells:
        pl = pv.Plotter()
        pl.add_mesh(mesh, style='wireframe', color='light gray')
        pl.add_mesh(unacceptable, color='lime')
        pl.view_xy()
        pl.camera.zoom(1.5)
        pl.show()


from matplotlib.backends.backend_pdf import PdfPages
def export_mesh_quality_report(quality_plot, quality_summary, pdf_path="mesh_quality_report.pdf", 
                                title='Mesh Quality Summary', figsize=(20, 10)):
    # Build summary table
    df = quality_summary.copy()

    # Make the plot figure
    fig_plot = quality_plot

    # Make a table figure
    fig_table, ax = plt.subplots(figsize=figsize)  # ~A4 landscape-ish in inches
    ax.axis("off")

    # Format values for nicer display
    df_disp = df.copy()
    df_disp["mean"] = df_disp["mean"].map(lambda x: f"{x:.4g}")
    df_disp["best"] = df_disp["best"].map(lambda x: f"{x:.4g}")
    df_disp["worst"] = df_disp["worst"].map(lambda x: f"{x:.4g}")
    df_disp["acceptable_range_pct"] = df_disp["acceptable_range_pct"].map(lambda x: f"{x:.1f}%")
    df_disp["acceptable_range"] = df_disp["acceptable_range"].map(lambda r: f"({r[0]}, {r[1]})")
    df_disp["95%"] = df_disp["95%"].map(lambda r: f"({r[0]}, {r[1]})")

    table = ax.table(
        cellText=df_disp.reset_index().values,
        colLabels=["metric"] + list(df_disp.columns),
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(15)
    table.scale(1, 2)

    fig_table.suptitle(title, fontsize=20, y=0.98)

    # Write PDF
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig_table, bbox_inches="tight")
        pdf.savefig(fig_plot, bbox_inches="tight")

    plt.close(fig_table)
    plt.close(fig_plot)

    return pdf_path


######## MIN DIHEDRAL ANGLE ##########

import numpy as np

def angle_between(n1, n2):
    dot = np.einsum("ij,ij->i", n1, n2)
    n1n = np.linalg.norm(n1, axis=1)
    n2n = np.linalg.norm(n2, axis=1)

    denom = n1n * n2n
    cosang = np.zeros_like(dot, dtype=np.float64)
    mask = denom > 0
    cosang[mask] = dot[mask] / denom[mask]

    cosang = np.clip(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

def dihedral_angles(points, tets):
    """Returns (min_dihedral, max_dihedral) per tetrahedron in degrees, each in [0, 180]."""
    p = points[tets]  # (M,4,3)
    p0, p1, p2, p3 = p[:,0], p[:,1], p[:,2], p[:,3]

    angles = np.empty((len(tets), 6), dtype=np.float64)

    def dihedral_about_edge(pi, pj, pk, pl):
        e = pj - pi
        n1 = np.cross(e, pk - pi)  # ⟂ edge, lies in face (i,j,k)
        n2 = np.cross(e, pl - pi)  # ⟂ edge, lies in face (i,j,l)
        return angle_between(n1, n2)  # interior dihedral in [0,180]

    angles[:,0] = dihedral_about_edge(p0, p1, p2, p3)  # edge (0,1)
    angles[:,1] = dihedral_about_edge(p0, p2, p3, p1)  # edge (0,2)
    angles[:,2] = dihedral_about_edge(p0, p3, p1, p2)  # edge (0,3)
    angles[:,3] = dihedral_about_edge(p1, p2, p0, p3)  # edge (1,2)
    angles[:,4] = dihedral_about_edge(p1, p3, p2, p0)  # edge (1,3)
    angles[:,5] = dihedral_about_edge(p2, p3, p0, p1)  # edge (2,3)

    return angles.min(axis=1), angles.max(axis=1)



def plot_cross_section_through_center(
    grid: pv.UnstructuredGrid,
    normal: str | tuple[float, float, float] = "x",
    origin: tuple[float, float, float] | None = None,
    show_surface: bool = False,
    show_edges: bool = True,
    scalars: str | None = "region_id",
):
    if origin is None:
        origin = tuple(grid.center)

    # Slice the volume with a plane
    slc = grid.slice(normal=normal, origin=origin)

    # Choose scalars only if they exist
    if scalars is not None and scalars not in slc.cell_data and scalars not in slc.point_data:
        scalars = None

    p = pv.Plotter()
    if show_surface:
        surf = grid.extract_surface().triangulate()
        p.add_mesh(surf, opacity=0.05)

    p.add_mesh(slc, scalars=scalars, show_edges=show_edges)
    p.show()


# Mesh deviation metrics after smoothing / remeshing ... #

def compute_dists(points, surf):
    """computes the distance of points from surface"""
    _, ps = surf.find_closest_cell(points, return_closest_point=True)
    return np.linalg.norm(points - ps, axis=1)

def compute_rmsd(dists):
    return np.sqrt(np.mean(dists**2))

def compute_d_metrics(dists, dic=None, label=''):
    if dic is None:
        dic = {}
    dic[f'{label}mean'] = np.mean(dists)
    dic[f'{label}median'] = np.median(dists)
    dic[f'{label}std'] = np.std(dists)
    dic[f'{label}min'] = np.min(dists)
    dic[f'{label}max'] = np.max(dists)
    dic[f'{label}99'] = np.percentile(dists, 99)
    dic[f'{label}95'] = np.percentile(dists, 95)
    return dic

def compute_mesh_metrics(surf, dic=None, vol=True, edge_length=False, label=''):
    if dic is None:
        dic = {}
    dic[f'{label}points'] = surf.n_points
    dic[f'{label}cells'] = surf.n_cells
    dic[f'{label}A'] = surf.area
    if vol:
        dic[f'{label}V'] = surf.volume
    if edge_length:
        dic[f'{label}L_edge'] = avg_edge_length(surf)
    return dic

def sample_surface(surf: pv.PolyData, n_samples: int):
    """Area weighted sampling of a mesh, returns sampled points\n
    probability of a face being picked for sampling is proportinal to its area - so gives approximately evenly spaced sample"""
    ps, face_ids = trimesh.sample.sample_surface(pv.to_trimesh(surf), count=n_samples)
    return ps

def compute_curv_metrics(mesh, dic, mean=True, gauss=True, maxi=True, mini=True, label=''):
    if mean:
        dic = compute_d_metrics(mesh.curvature(curv_type='mean'), dic, f'{label}Kmean_')
    if gauss:
        dic = compute_d_metrics(mesh.curvature(curv_type='gaussian'), dic, f'{label}Kgauss_')
    if maxi:
        dic = compute_d_metrics(mesh.curvature(curv_type='maximum'), dic, f'{label}Kmax_')
    if mini:
        dic = compute_d_metrics(mesh.curvature(curv_type='minimum'), dic, f'{label}Kmin_')
    return dic