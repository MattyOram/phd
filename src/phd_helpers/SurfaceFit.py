import pyvista as pv
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import sympy as sp

from phd_helpers.SaddleAnalysis import get_centroid, get_norm
from phd_helpers.CartilageGeneration import get_trimesh, get_pvmesh


def get_edge_shell(mesh, n_vertices=3, d1=10, d2=2):
    """create a double pointed pyramid shell with mesh shaped edge, for cutting/isolating mesh/surfaces that lie inside of mesh edge bounds"""

    # get centre coord and centre norm
    saddle_centre = mesh.find_closest_point(np.sum(mesh.points, axis=0) / mesh.n_points)
    saddle_centre_coord = mesh.points[saddle_centre]
    centre_norm = mesh.compute_normals(point_normals=True, cell_normals=False)['Normals'][saddle_centre]

    # get min and max distance of mesh_surface points from centre norm plane (to make sure the tent points are above the mesh)
    mesh_to_centre_vectors = mesh.points - saddle_centre_coord # mesh based surface points to saddle centre vectors
    mesh_normal_dists = mesh_to_centre_vectors @ centre_norm # normal component of mesh based surface points to saddle centre vectors
    min_normal_dist = min(mesh_normal_dists)
    max_normal_dist = max(mesh_normal_dists)

    # get tent points
    point1 = saddle_centre_coord + centre_norm*(max_normal_dist*d1) # upper 
    point2 = saddle_centre_coord + centre_norm*(min_normal_dist*d2) # lower

    # edge
    edge_lines = mesh.extract_feature_edges(
        boundary_edges=True,
        non_manifold_edges=False,
        feature_edges=False,
        manifold_edges=False
    )

    # create tent mesh
    tent_points = np.vstack((edge_lines.points, point1, point2)) # list of coords of connected edge_points, point1, point2,

    faces1 = np.hstack(
        (np.array([3]*edge_lines.n_lines).reshape(-1, 1), 
        edge_lines.lines.reshape(-1, 3)[:, 1:], 
        np.array([edge_lines.n_lines]*edge_lines.n_lines).reshape(-1, 1))
                )

    faces2 = np.hstack(
        (np.array([3]*edge_lines.n_lines).reshape(-1, 1), 
        edge_lines.lines.reshape(-1, 3)[:, 1:], 
        np.array([edge_lines.n_lines+1]*edge_lines.n_lines).reshape(-1, 1))
                )

    tent_faces = np.vstack((faces1, faces2))

    tent = pv.PolyData(tent_points, tent_faces)
    return tent.triangulate().clean().compute_normals(auto_orient_normals=True)

def cut_surface(surface, shell):
    """will return the surface region and point mask that is engulfed by the shell (includes cells partially engulfed)"""
    point_cloud = pv.PolyData(surface.points)
    selected = point_cloud.select_enclosed_points(shell, tolerance=0.00, check_surface=True, )
    mask = selected["SelectedPoints"].astype(bool)
    return surface.extract_points(mask), mask

def cut_grid_surface(grid_surface, shell):
    """will return the surface region that is engulfed by the shell (includes cells partially engulfed)"""
    point_cloud = pv.PolyData(grid_surface.points)
    selected = point_cloud.select_enclosed_points(shell, tolerance=0.00, check_surface=True, )
    mask = selected["SelectedPoints"].astype(bool)
    return grid_surface.extract_points(mask)

def get_grid(mesh_points, n_points=200, buffer=0.05):
    """get x, y points (n_points*n_points, 2) from meshgrid of x, y points that cover mesh + buffer"""
    xs, ys = mesh_points[:, 0], mesh_points[:, 1]
    xmin, xmax = np.min(xs), np.max(xs)
    xdiff = xmax - xmin
    ymin, ymax = np.min(ys), np.max(ys)
    ydiff = ymax - ymin

    x_lin = np.linspace(xmin-(buffer*xdiff), xmax+(buffer*xdiff), n_points)
    y_lin = np.linspace(ymin-(buffer*ydiff), ymax+(buffer*ydiff), n_points)
    x_grid, y_grid = np.meshgrid(x_lin, y_lin)
    return np.vstack((x_grid.ravel(), y_grid.ravel())).T

def get_grid2(mesh_points, point_density=200, buffer=0.05):
    """get x, y points (n_points*n_points, 2) from meshgrid of x, y points that cover mesh + buffer"""
    xs, ys = mesh_points[:, 0], mesh_points[:, 1]
    xmin, xmax = np.min(xs), np.max(xs)
    xdiff = xmax - xmin
    ymin, ymax = np.min(ys), np.max(ys)
    ydiff = ymax - ymin

    xmin_buffered = xmin - buffer * xdiff
    xmax_buffered = xmax + buffer * xdiff
    xdiff_buffered = xmax_buffered - xmin_buffered
    ymin_buffered = ymin - buffer * ydiff
    ymax_buffered = ymax + buffer * ydiff
    ydiff_buffered = ymax_buffered - ymin_buffered

    nx = round(xdiff_buffered * np.sqrt(point_density))
    ny = round(ydiff_buffered * np.sqrt(point_density))

    x_lin = np.linspace(xmin_buffered, xmax_buffered, nx)
    y_lin = np.linspace(ymin-(buffer*ydiff), ymax+(buffer*ydiff), ny)
    x_grid, y_grid = np.meshgrid(x_lin, y_lin, indexing='ij')
    return x_grid, y_grid, nx, ny

def get_grid_surface(mesh, model, poly, grid_points):
    xy_grid = get_grid(mesh.points, n_points=grid_points, buffer=0.05)
    z_grid_pred = model.predict(poly.transform(xy_grid)).reshape((grid_points, grid_points))
    grid_surface = pv.StructuredGrid(xy_grid[:, 0].reshape(z_grid_pred.shape), xy_grid[:, 1].reshape(z_grid_pred.shape), z_grid_pred)
    grid_surface = grid_surface.extract_surface()
    shell = get_edge_shell(mesh)
    return cut_grid_surface(grid_surface, shell).extract_geometry()

def cut_grid_surface2(grid_surface, shell):
    """will return the surface region that is engulfed by the shell (only cells completely engulfed)"""
    point_cloud = pv.PolyData(grid_surface.points)
    selected = point_cloud.select_enclosed_points(shell, tolerance=0.00, check_surface=True)
    mask = selected["SelectedPoints"].astype(bool)
    
    # only get cells where all vertices are inside the shell
    cells_to_keep = []
    for i in range(grid_surface.n_cells):
        cell_points = grid_surface.get_cell(i).point_ids
        if all(mask[pid] for pid in cell_points):
            cells_to_keep.append(i)
    
    return grid_surface.extract_cells(cells_to_keep)

def get_grid_surface2(mesh, model, poly, grid_points):
    x_grid, y_grid, nx, ny = get_grid2(mesh.points, point_density=grid_points, buffer=0.05)
    xy_flat = np.vstack((x_grid.ravel(), y_grid.ravel())).T
    
    z_grid_pred = model.predict(poly.transform(xy_flat)).reshape((nx, ny))
    grid_surface = pv.StructuredGrid(x_grid, y_grid, z_grid_pred)
    grid_surface = grid_surface.extract_surface()

    shell = get_edge_shell(mesh)
    return cut_grid_surface2(grid_surface, shell).extract_geometry()

def plot_principal_curvatures(mesh, scalar='kmax', vector='kmax_dir', step=1, show_edges=True, factor=1.0):
    indices = np.arange(0, mesh.n_points, step)

    # Extract points directly (creates a new mesh with only selected points)
    points_subset = mesh.points[indices]
    subset = pv.PolyData(points_subset)

    # Add the corresponding vectors
    subset[vector] = mesh[vector][indices]
    subset[scalar] = mesh[scalar][indices]
    subset['neg'] = -subset[vector]

    # Create glyphs
    arrows = subset.glyph(orient=vector, scale=scalar, factor=factor)
    arrows_neg = subset.glyph(orient="neg", scale=scalar, factor=factor)

    # Plot
    p = pv.Plotter()
    p.add_mesh(mesh, scalars=scalar, show_edges=False, opacity=1.)
    if show_edges:
        p.add_mesh(mesh, scalars=scalar, opacity=1., show_edges=True, edge_opacity=0.5)
    p.add_mesh(arrows, color='black')
    p.add_mesh(arrows_neg, color="black")
    p.add_axes()
    p.show()

    # get polynomial
def get_polynomial_model_no_scaling(mesh, degree=5):
    points = mesh.points
    xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
    XY = np.vstack((xs, ys)).T

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    XY_poly = poly.fit_transform(XY) # 21 coefs for 5th degree polynomial
    model = LinearRegression(fit_intercept=True)
    model.fit(XY_poly, zs)

    # fit quality
    z_pred = model.predict(XY_poly)
    r2 = r2_score(zs, z_pred)
    rmse = np.sqrt(mean_squared_error(zs, z_pred))
    #print(f"Polynomial fit quality: R² = {r2:.4f}, RMSE = {rmse:.4f}")
    return model, poly

def get_polynomial_model(points, degree=5):
    # also no scaling but takes points and returns r2 and rmse
    xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
    XY = np.vstack((xs, ys)).T

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    XY_poly = poly.fit_transform(XY) # 21 coefs for 5th degree polynomial
    model = LinearRegression(fit_intercept=True)
    model.fit(XY_poly, zs)

    # fit quality
    z_pred = model.predict(XY_poly)
    r2 = r2_score(zs, z_pred)
    rmse = np.sqrt(mean_squared_error(zs, z_pred))
    #print(f"Polynomial fit quality: R² = {r2:.4f}, RMSE = {rmse:.4f}")
    return model, poly, r2, rmse

# get f
def get_sp_terms(poly):
    x, y = sp.symbols('x y')
    poly_terms = poly.get_feature_names_out(input_features=['x', 'y'])
    return np.array([eval(term.replace('^', '**').replace(' ', '*'), {'x': x, 'y': y}) for term in poly_terms])

def get_f(model, terms):
    return sum(c * m for c, m in zip(model.coef_, terms)) + model.intercept_

# get principal curvatures
def get_ks_HK(f, points):
    xs, ys = points[:, 0], points[:, 1]

    x, y = sp.symbols('x y')
    fx, fy = sp.diff(f, x), sp.diff(f, y)
    fxx, fxy, fyy = sp.diff(fx, x), sp.diff(fx, y), sp.diff(fy, y)

    # First and second fundamental forms
    E = 1 + fx**2
    F = fx * fy
    G = 1 + fy**2

    w = sp.sqrt(1 + fx**2 + fy**2)
    L = fxx / w
    M = fxy / w
    N = fyy / w

    # Curvatures
    H = (E * N - 2 * F * M + G * L) / (2 * (E * G - F**2)) # mean curvature
    K = (L * N - M**2) / (E * G - F**2) # gaussian curvature 

    k1 = H + sp.sqrt(H**2 - K)
    k2 = H - sp.sqrt(H**2 - K)

    k1_func = sp.lambdify((x, y), k1, modules='numpy')
    k2_func = sp.lambdify((x, y), k2, modules='numpy')

    k1s = k1_func(xs, ys)
    k2s = k2_func(xs, ys)
    ks = np.sort(np.column_stack((k1s, k2s)), axis=1)


    return {'kmin': ks[:, 0], 'kmax': ks[:, 1]}

# get principal curvatures and directions
def get_ks_eig(f, points):
    xs, ys = points[:, 0], points[:, 1]

    x, y = sp.symbols('x y')
    fx, fy = sp.diff(f, x), sp.diff(f, y)
    fxx, fxy, fyy = sp.diff(fx, x), sp.diff(fx, y), sp.diff(fy, y)

    # First and second fundamental forms
    E = 1 + fx**2
    F = fx*fy
    G = 1 + fy**2

    w = sp.sqrt(1 + fx**2 + fy**2)
    L = fxx / w
    M = fxy / w
    N = fyy / w

    E_func = sp.lambdify((x, y), E, 'numpy')
    F_func = sp.lambdify((x, y), F, 'numpy')
    G_func = sp.lambdify((x, y), G, 'numpy')
    L_func = sp.lambdify((x, y), L, 'numpy')
    M_func = sp.lambdify((x, y), M, 'numpy')
    N_func = sp.lambdify((x, y), N, 'numpy')

    E_vals = E_func(xs, ys)
    F_vals = F_func(xs, ys)
    G_vals = G_func(xs, ys)
    L_vals = L_func(xs, ys)
    M_vals = M_func(xs, ys)
    N_vals = N_func(xs, ys)

    kmin_vecs2d = np.zeros((len(xs), 2))
    kmax_vecs2d = np.zeros((len(xs), 2))
    ks = np.zeros((len(xs), 2))
    for i in range(len(xs)):
        I = np.array([[E_vals[i], F_vals[i]], [F_vals[i], G_vals[i]]])
        II = np.array([[L_vals[i], M_vals[i]], [M_vals[i], N_vals[i]]])
        
        S = np.linalg.inv(I) @ II
        eigvals, eigvecs = np.linalg.eigh(S)  # ensures real symmetric handling
        
        idx = np.argsort(eigvals)
        kmin_vecs2d[i], kmax_vecs2d[i] = eigvecs[:, idx[0]], eigvecs[:, idx[1]]
        ks[i] = np.sort(eigvals)

    # 3D vectors
    fx_func = sp.lambdify((x, y), fx, 'numpy')
    fy_func = sp.lambdify((x, y), fy, 'numpy')

    fx_vals = fx_func(xs, ys)
    fy_vals = fy_func(xs, ys)

    tx = np.stack([np.ones_like(fx_vals), np.zeros_like(fx_vals), fx_vals], axis=1)  # [1, 0, fx]
    ty = np.stack([np.zeros_like(fy_vals), np.ones_like(fy_vals), fy_vals], axis=1)  # [0, 1, fy]

    kmin_vecs3d = kmin_vecs2d[:, 0].reshape(-1, 1) * tx + kmin_vecs2d[:, 1].reshape(-1, 1) * ty
    kmax_vecs3d = kmax_vecs2d[:, 0].reshape(-1, 1) * tx + kmax_vecs2d[:, 1].reshape(-1, 1) * ty

    kmin_vecs3d /= np.linalg.norm(kmin_vecs3d, axis=1, keepdims=True)
    kmax_vecs3d /= np.linalg.norm(kmax_vecs3d, axis=1, keepdims=True)

    return {'kmin_dir': kmin_vecs3d, 'kmin': ks[:, 0], 'kmax_dir': kmax_vecs3d, 'kmax': ks[:, 1]}

def get_saddle_points(f, points):
    xs, ys = points[:, 0], points[:, 1]
    x, y = sp.symbols('x y')

    # Gradient and Hessian
    fx, fy = sp.diff(f, x), sp.diff(f, y)
    H = sp.hessian(f, (x, y)) # Hessian

    # lambdify
    f_func = sp.lambdify((x, y), f, 'numpy')
    fx_func, fy_func = sp.lambdify((x, y), fx, 'numpy'), sp.lambdify((x, y), fy, 'numpy')
    fxx_func = sp.lambdify((x, y), H[0, 0], 'numpy')
    fxy_func = sp.lambdify((x, y), H[0, 1], 'numpy')
    fyy_func = sp.lambdify((x, y), H[1, 1], 'numpy')

    # find points where gradient == 0
    gx = fx_func(xs, ys)
    gy = fy_func(xs, ys)
    tol = 1e-6
    zero_mask = (np.abs(gx) < tol) & (np.abs(gy) < tol)

    if np.any(zero_mask): # check if there are any 
        # find critical points with det(H) < 0
        Xc = xs[zero_mask]
        Yc = ys[zero_mask]

        Hxx = fxx_func(Xc, Yc)
        Hxy = fxy_func(Xc, Yc)
        Hyy = fyy_func(Xc, Yc)
        D = Hxx * Hyy - Hxy**2 # det(H)

        # get saddle points
        saddle_mask = D < 0
        Zc = f_func(Xc[saddle_mask], Yc[saddle_mask])
        return np.array(list(zip(Xc[saddle_mask], Yc[saddle_mask], Zc))).reshape(-1, 3)
    else:
        print('No saddle points :(')
        return []
    
def get_intercepts(surface, start_points, vectors, ray_length=100, offset=1):
    """
    Find where lines extended from start_points in the direction of the vectors of length ray_length intercept the surface
    Returns surface intercepts, corresponding start_points, and the mask of which start_points had intercepts
    Assumes vectors point away from surface (because z points proximally in tpm intertial coords)
    """

    surface_intercepts = np.zeros_like(start_points)
    intercept_mask = np.zeros(len(start_points))
    for idx in range(start_points.shape[0]-1):
        ray_start = start_points[idx] + vectors[idx]*offset
        ray_end = ray_start + vectors[idx] * ray_length * (-1) # *(-1) because normals point proximally
        point, face = surface.ray_trace(ray_start, ray_end)
        if point.shape[0] > 0:
            surface_intercepts[idx] = point.reshape(-1, 3)[0]
            intercept_mask[idx] = 1

    intercept_mask = intercept_mask.astype('bool')
    return surface_intercepts[intercept_mask], start_points[intercept_mask], intercept_mask

def get_intercepts_trimesh(surface, start_points, rays, batch_size=500):
    """
    Find where lines extended from start_points in the direction of the rays intercept the surface
    Returns surface intercepts, corresponding start_points, and the mask of which start_points had intercepts
    """

    mesh = get_trimesh(surface, n_verts=4)
    starts = start_points - 10*rays

    all_intercepts = []
    all_ray_idx = []
    for i in range(0, len(start_points), batch_size): # batches cos memory use doesn't scale well

        intercepts, ray_idx, _ = mesh.ray.intersects_location(
                ray_origins=starts[i:i+batch_size],
                ray_directions=rays[i:i+batch_size],
                multiple_hits=False
            )
        if len(intercepts):
            all_intercepts.append(intercepts)
            all_ray_idx.extend(ray_idx+i)
    
    all_intercepts = np.vstack(all_intercepts)
    surface_intercepts = np.zeros_like(start_points)
    surface_intercepts[all_ray_idx] = all_intercepts
    intercept_mask = np.isin(np.arange(len(start_points)), all_ray_idx)
    return surface_intercepts[intercept_mask], start_points[intercept_mask], intercept_mask

def get_poly_normals(f, points):
    x, y = sp.symbols('x y')
    fx, fy = sp.diff(f, x), sp.diff(f, y)
    fx_func, fy_func = sp.lambdify((x, y), fx, 'numpy'), sp.lambdify((x, y), fy, 'numpy')
    
    normals = np.column_stack((
        fx_func(points[:, 0], points[:, 1]) * (-1), 
        fy_func(points[:, 0], points[:, 1]) * (-1), 
        np.ones(len(points))
        ))
    normals /= np.linalg.norm(normals, axis=1).reshape(-1, 1)
    return normals