import numpy as np
import pandas as pd

import os
from pathlib import Path
import pyvista as pv
import trimesh
from vtk import vtkLinearToQuadraticCellsFilter

from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def get_project_root() -> Path:
    return PROJECT_ROOT


def get_db_path() -> Path:
    """get BrownUniCarpalDataset path"""
    return PROJECT_ROOT / 'data' / 'BrownUniCarpalDataset' / 'Database'

def get_info_path()  -> Path:
    return PROJECT_ROOT / 'data' / 'BrownUniCarpalDataset' / 'subject_info.csv'

def get_info_df() -> pd.DataFrame:
    return pd.read_csv(get_info_path())

def get_stl_paths() -> list[Path]:
    """Returns list of Paths to STL folders for all subjects and sides"""
    return list(Path(get_db_path()).glob(f'**/*STL/'))

def get_task_stl_paths(task='CMC') -> list[Path]:
    """Returns list of Paths to STL folders for all subjects and sides for given task"""
    return list(Path(get_db_path()).glob(f'**/{task}*/**/*STL/'))

def get_subject_stl_path(subject, sideL) -> Path:
    """Returns Path to STL folder for given subject and side (e.g. subject='50009', sideL='R')"""
    return list(Path(get_db_path()).glob(f'**/{subject}/{sideL}*STL/'))[0]

def get_info(stl_path):
    """pathlib Path\n
    -> subject, sideL"""
    subject = stl_path.parent.name
    sideL = stl_path.name[0]
    return subject, sideL

def get_mesh(stl_path, bone):
    bone_stl_path = list(stl_path.glob(f'*_{bone}_*.stl'))[0]
    mesh = pv.read(bone_stl_path)
    return mesh

def transform_points(points, R, t, inverse=False):
    """Transform points, use inverse to change basis"""
    points1 = points.copy()
    points1 = points1.reshape(-1, 3)
    if inverse:
        points1 -= t
        points1 = (R.T @ points1.T).T
    else:
        points1 = (R @ points1.T).T
        points1 += t
    return points1

def transform_mesh(mesh: pv.PolyData, R, t, inverse=False):
    """Transform mesh, use inverse to change basis"""
    mesh1 = mesh.copy(deep=True)
    if inverse:
        mesh1.points -= t
        mesh1.points = (R.T @ mesh1.points.T).T
    else:
        mesh1.points = (R @ mesh1.points.T).T
        mesh1.points += t
    return mesh1


def pose2idCMC(pose):
    poses = {
        'adduction': '01',
        'abduction': '02',
        'flexion': '03',
        'extension': '04',
        'pinch': '05',
        'grasp': '06',
        'jar': '07',
        'pinch_load': '08',
        'grasp_load': '09',
        'jar_load': '10',
        'neutral': '11'
    }
    return poses[pose]

def bone2code(bone):
    bones = np.array(['rad', 'uln', 'sca', 'lun', 'trq', 'pis', 'tpd',
                        'tpm', 'cap', 'ham', 'mc1', 'mc2', 'mc3', 'mc4', 'mc5'])
    return int(np.where(bones==bone)[0][0])

def inertia_data(bone, inertia):
    """Returns: centroid, magnitudes of the principal inertia axes, unit vectors for the principal inertial axes"""
    bone_inertia = inertia[bone2code(bone)]
    return bone_inertia[0].copy(), bone_inertia[1].copy(), bone_inertia[2:].copy()

def get_inertia(stl_path):
    """returns inertia data for all bones for given wrist"""
    subject, sideL= get_info(stl_path)
    inertia_path = list(stl_path.parent.glob(f'{sideL}*Info/*_inertia_{sideL}.dat'))[0]
    inertia = np.loadtxt(inertia_path)   #75x3
    inertia = inertia.reshape(inertia.shape[0] // 5, 5, 3) #15x5x3 (centroid, magnitudes, coordinates)
    return inertia

def get_bone_inertia(stl_path, bone):
    """Returns: centroid, magnitudes of the principal inertia axes, unit vectors for the principal inertial axes"""
    return inertia_data(bone, get_inertia(stl_path))

def get_transform_data(bone, transforms):
    """Returns: 3x3 rotation matrix and 1x3 translation vector"""
    bone_motion = transforms[bone2code(bone)]
    return bone_motion[:-1].copy(), bone_motion[-1].copy()

def get_bone_transforms(pose_id: str, stl_path: Path):
    """return transformation matrices for all bones for that motion (15x4x3) (ACTUAL ID i.e. '01' - NOT integer idx)"""
    subject, sideL = get_info(stl_path)
    path = list(stl_path.parent.glob(f'**/*_Motion{pose_id}{sideL}.dat'))
    #if len(path) != 1:
        #print(len(path), 'paths found!') # commented out cos it prints when doing try: neutral-2 excpet default neutral
    return np.loadtxt(path[0]).reshape(-1, 4, 3) #15x4x3

def get_bone_transform(stl_path, bone, pose_id):
    """Returns: 3x3 rotation matrix and 1x3 translation vector"""
    return get_transform_data(bone, get_bone_transforms(pose_id, stl_path))

def get_relative_transform_new_basis(transforms, bone, ref_bone, origin=[0, 0, 0], axes=np.eye(3)):
    """
    transforms are the transforms returned by get_bone_transforms\n
    Calculate transformations relative to reference bone in chosen coordinate basis - defaults to global basis\n
    Returns: R, t
    """
    # Global transformations
    R_i, t_i = get_transform_data(bone, transforms)
    R_ref, t_ref = get_transform_data(ref_bone, transforms)
    
    # Transformation to change coordinate basis
    T_basis = np.eye(4)
    T_basis[0:3, 0:3] = axes.T  # global to new
    T_basis[0:3, 3] = -axes.T @ origin
    
    # Inverse transformation (new to global)
    T_basis_inv = np.linalg.inv(T_basis)
    
    # Transformation matrices for both bones
    T_i = np.eye(4)
    T_i[0:3, 0:3] = R_i
    T_i[0:3, 3] = t_i
    
    T_ref = np.eye(4)
    T_ref[0:3, 0:3] = R_ref
    T_ref[0:3, 3] = t_ref
    
    # Relative transformation in new coordinate system
    T_rel = T_basis @ np.linalg.inv(T_ref) @ T_i @ T_basis_inv
    
    # rotation and translation parts
    R_rel = T_rel[0:3, 0:3]
    t_rel = T_rel[0:3, 3]
    
    return R_rel, t_rel



def furthest_points(mesh_points):
    """Get coordinates of 2 furthest points on mesh"""

    hull = ConvexHull(mesh_points) #.vertices gives idxs of hull points
    hullpoints = mesh_points[hull.vertices,:] # points on hull
    hdist = cdist(hullpoints, hullpoints, metric='euclidean') # distance of every point from every point (NxN)
    bestpair = np.unravel_index(hdist.argmax(), hdist.shape) # points greatest distance from each other

    return [hullpoints[bestpair[0]], hullpoints[bestpair[1]]]

def avg_edge_length(mesh):
    edges = mesh.extract_all_edges()
    line_points = edges.points[edges.lines.reshape(-1, 3)[:, 1:]] # (N, 2) coords
    return np.mean(np.linalg.norm(line_points[:, 1] - line_points[:, 0], axis=1))

def compute_edge_lengths(mesh):
    edges = mesh.extract_all_edges()
    line_points = edges.points[edges.lines.reshape(-1, 3)[:, 1:]] # (N, 2) coords
    return np.linalg.norm(line_points[:, 1] - line_points[:, 0], axis=1)

def get_mc_lower_cell_ids(stl_path, mc_mesh, mc_bone='mc1', p=0.6):
    """
        Get cell ids and point ids of mc1 cells after removing cells whose centres are above 'p' fraction down the bone \n
        Mesh must be in global coordinate basis
    """
    mesh = mc_mesh.copy(deep=True)
    mesh['mesh_id'] = np.arange(mesh.n_points)
    t, _, R = get_bone_inertia(stl_path, mc_bone)
    mesh.points = transform_points(mesh.points, R, t, inverse=True)

    xs = mesh.cell_centers().points[:, 0]

    top, bot = np.max(xs), np.min(xs)
    cutoff = top - (top-bot)*p
    cell_ids = np.where(xs<cutoff)[0]
    point_ids = mesh.extract_cells(cell_ids)['mesh_id']
    return cell_ids, point_ids

def find_shared_cells(mesh1, mesh2, eta=1e-10):
    """Returns mask of cells on mesh1 with identical centers to mesh 2"""
    close_ids = mesh2.find_closest_cell(mesh1.cell_centers().points)
    interface_mask = np.linalg.norm(
        mesh2.cell_centers().points[close_ids] - mesh1.cell_centers().points, axis=1
        ) <= eta
    return interface_mask

def find_corresponding_cells(mesh, feature_mesh, eta=1e-6, raise_error=False):
    """
    Given a mesh of mutiple features/pathches and a mesh of one of those features/patches\n
    returns indexes of the corresponding feature/patch cells on mesh
    """
    cell_ids = mesh.find_closest_cell(feature_mesh.cell_centers().points)
    #if np.sum(np.abs(mesh.cell_centers().points[cell_ids] - feature_mesh.cell_centers().points)) > eta:
        #print('Points not close!') # lazy check

    # check how close they are
    msh_centres = mesh.cell_centers().points[cell_ids]
    cell_dists = np.linalg.norm(msh_centres - feature_mesh.cell_centers().points, axis=1)
    if not (cell_dists < eta).all() and raise_error:
        print(f'Max cell centre distance from original mesh: {np.max(cell_dists):.6f}')
        raise AssertionError(f"Mesh cell centres have moved too far (<={eta})")

    return cell_ids

def find_corresponding_points(mesh, points, eta=1e-10):
    ids = []
    for p in points:
        ids.append(mesh.find_closest_point(p))
    if np.sum(np.abs(mesh.points[ids] - points)) > eta:
        print('Points not close!')
    return np.array(ids)

def identical_points_count(A, B, return_indices=False):
    """takes two lists of points and returns the number of unique identical points ( where: (XA, YA, ZA) == (XB, YB, ZB) )\n
    if return_indices, returns: array of tupes of idetical points, array of point ids on A, array of point ids on B"""
    # (N,3) float arrays
    A_view = np.ascontiguousarray(A).view([('', A.dtype)] * A.shape[1])
    B_view = np.ascontiguousarray(B).view([('', B.dtype)] * B.shape[1])
    if return_indices:
        return np.intersect1d(A_view.ravel(), B_view.ravel(), return_indices=True)
    else:
        return np.intersect1d(A_view.ravel(), B_view.ravel()).size

def get_volume_seed(mesh, instep=0.01):
    """Returns point contained within mesh volume (steps in from central point by instep value)"""
    central_node = mesh.find_closest_point(np.array(mesh.center))
    central_point = mesh.points[central_node]
    central_normal = mesh.point_normals[central_node]

    seed = central_point - instep*central_normal
    # check its inside
    probe = pv.PolyData(seed)
    if not bool(probe.select_enclosed_points(mesh, tolerance=0, check_surface=True)['SelectedPoints'][0]):
        print('Warning: seed not inside. Can try reducing offset')
    return seed

def get_trimesh(pv_mesh: pv.PolyData, n_verts=3, process=False):
    """convert pyvista mesh to trimesh"""
    vertices = pv_mesh.points
    faces = pv_mesh.faces.reshape(-1, n_verts+1)[:, 1:]
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=process)

def get_pvmesh(tmesh: trimesh.Trimesh, n_verts=3):
    vertices = tmesh.vertices
    faces = tmesh.faces
    faces_pv = np.hstack(
        np.c_[np.full(len(faces), n_verts), faces]
    )
    return pv.PolyData(vertices, faces_pv)

def linear_to_quadratic_mesh(linear_mesh: pv.UnstructuredGrid):
    """
    Convert a linear mesh to a quadratic mesh\n 
    Inserts nodes at midpoints on cell edges\n
    Preserves cell_array data (n_cells before == n_cells after)\n
    (to visualise actual mesh surface do quad_mesh.linear_copy().plot() 
    - otherwise pyvista connects midpoint nodes with new surface edge lines for visual)
    """
    mesh = linear_mesh.copy(deep=True)
    filt = vtkLinearToQuadraticCellsFilter()
    filt.SetInputData(mesh)
    filt.Update()

    return  pv.wrap(filt.GetOutput())

import numpy as np
import pyvista as pv

def quadratic_to_linear_mesh(mesh: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    celltypes = np.unique(mesh.celltypes)

    quads = {
        pv.CellType.QUADRATIC_TETRA,
        pv.CellType.QUADRATIC_TRIANGLE,
    }
    lins = {
        pv.CellType.TETRA,
        pv.CellType.TRIANGLE,
    }

    if set(celltypes).issubset(lins):
        return mesh

    if not set(celltypes).issubset(quads):
        raise ValueError(
            "Only works for tets and tris"
        )

    cells = mesh.cells
    out_cells = []
    out_types = []

    i = 0
    for ctype in mesh.celltypes:
        npts = cells[i]
        ids = cells[i + 1 : i + 1 + npts]

        if ctype == pv.CellType.QUADRATIC_TRIANGLE:
            out_cells.extend([3, ids[0], ids[1], ids[2]])
            out_types.append(pv.CellType.TRIANGLE)

        elif ctype == pv.CellType.QUADRATIC_TETRA:
            out_cells.extend([4, ids[0], ids[1], ids[2], ids[3]])
            out_types.append(pv.CellType.TETRA)

        i += npts + 1

    out = pv.UnstructuredGrid(
        np.array(out_cells, dtype=np.int64),
        np.array(out_types, dtype=np.uint8),
        mesh.points.copy(),
    )

    for name in mesh.point_data:
        out.point_data[name] = mesh.point_data[name].copy()

    for name in mesh.cell_data:
        out.cell_data[name] = mesh.cell_data[name].copy()

    return out.remove_unused_points()

def outward_normals(mesh, eps=1e-5):
    """Checks that all face normals on a watertight mesh point outwards"""
    normals = mesh.compute_normals()['Normals'] # face normals
    centres = mesh.cell_centers().points
    point_cloud = pv.PolyData(centres - normals * eps)
    print(
        'All normals point outwards      ', 
        point_cloud.select_enclosed_points(mesh, tolerance=0, check_surface=True)['SelectedPoints'].all()
        )
    
def get_boundary(mesh):
    return mesh.extract_feature_edges(
    boundary_edges=True, 
    non_manifold_edges=False, 
    feature_edges=False, 
    manifold_edges=False
    )

def sort_edge_lines(line_pd: pv.PolyData, start_idx=0):
    """Given a pv.PolyData object containing connnected lines,\n 
    Returns the sorted idx of the lines (in walkable order) (starting at start_idx)"""
    # sort edge lines
    lines = line_pd.lines.reshape(-1, 3)[:, 1:] # lines are not in any order

    lines_idx = np.zeros(line_pd.n_lines, int)
    idx = start_idx
    for i in range(line_pd.n_lines):
        lines_idx[i] = idx
        idx = np.argmax(lines[:, 0] == lines[idx][1])

    return lines_idx

def get_intercepts(surface, start_points, vectors, ray_length=100, offset=0):
    """
    Find where lines extended from start_points in the direction of the vectors of length ray_length intercept the surface\n
    Returns surface intercepts, corresponding start_points, and the mask of which start_points had intercepts
    """

    surface_intercepts = np.zeros_like(start_points)
    intercept_mask = np.zeros(len(start_points))
    for idx in range(start_points.shape[0]-1):
        ray_start = start_points[idx] + vectors[idx]*offset
        ray_end = ray_start + vectors[idx] * ray_length
        point, face = surface.ray_trace(ray_start, ray_end)
        if point.shape[0] > 0:
            surface_intercepts[idx] = point.reshape(-1, 3)[0]
            intercept_mask[idx] = 1

    intercept_mask = intercept_mask.astype('bool')
    return surface_intercepts[intercept_mask], start_points[intercept_mask], intercept_mask

########################################  intersection ##########################################
def bone_pair_intersection(bone_pairs, bone_pair_counts, n=0):
    """Return subjects that have interferences in n or less poses for the given bone pair"""
    mask = np.ones(len(bone_pair_counts.columns),dtype=bool)
    for bone_pair in bone_pairs:
        mask = mask & (bone_pair_counts[np.isin(bone_pair_counts.reset_index()['bone_pairs'], bone_pair)].values[0] <= n)
    return pd.Series(bone_pair_counts.columns)[mask].reset_index(drop=True)

def subject_intersection_poses(intersection_path, subjectL, bone_pair='tpm-mc1'):
    """returns list of poses that have intersections for the given bone pair and subject """
    subject_intersections = pd.read_csv(os.path.join(intersection_path, f'{subjectL}.csv'), index_col=0)
    return subject_intersections.columns[subject_intersections.loc[bone_pair].values].to_list()

def bone_pair_pose_intersection(bone_pairs, poses, intersection_path):
    """returns subjects with no interfernce for list of bone pairs and poses"""
    bone_pairs = np.array(bone_pairs) 
    poses = np.array(poses)
    paths = [os.path.join(intersection_path, x) for x in os.listdir(intersection_path) if x[:-5].isnumeric()]

    no_intersection = []
    for path in paths:
        subject_df = pd.read_csv(path, index_col=0)
        if (subject_df.loc[bone_pairs, poses] == False).values.all():
            no_intersection.append(path.split(os.path.sep)[-1].replace('.csv', ''))
    return pd.Series(no_intersection)
########################################  intersection ##########################################