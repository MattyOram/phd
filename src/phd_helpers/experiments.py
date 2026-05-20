import numpy as np
import pandas as pd
import pyvista as pv
from sklearn.linear_model import LinearRegression as lr


def get_instron_data(file_path, zero_displacement=True):
    data = pd.read_csv(file_path)
    data.columns = [x.lower() for x in data.columns]
    #print(data.iloc[0])
    data = data.copy().iloc[1:]
    data = data.astype('float32')
    if zero_displacement:
        data['displacement'] -= data['displacement'].min()
    return data

def get_stress_strain(data, r=5, l=20):
    A = np.pi*r**2
    strain = (data.displacement / l)
    stress = data.force*1e3 / A #MPa - if F is kN and A is mm^2

    data['stress'] = stress
    data['strain'] = strain

    return data

def get_ym(data, mask1=0, mask2=None, return_XYs=False, n=1000):
    if not mask2:
        mask2 = data.stress.values.max()
    mask = (data['stress'].values >= mask1) & (data['stress'].values <= mask2)
    reg = lr(data['strain'][mask], data['stress'][mask])
    m, c = reg.slope, reg.intercept
    if return_XYs:
        Xs = np.linspace(data.stress.values.min(), data.stress.values.max(), n)
        Ys = m*Xs + c
        return m, c, Xs, Ys
    else:
        return m, c

def parse_tekscan(path, sensor=None):
    header = {}
    frames = []

    with open(path, "r") as f:
        lines = f.readlines()

    # ---- header ----
    i = 0
    while not lines[i].startswith("ASCII_DATA"):
        key, *val = lines[i].strip().split(maxsplit=1)
        header[key] = val[0] if val else None
        i += 1
    i += 1  # skip "ASCII_DATA @@"

    rows = int(header["ROWS"])
    cols = int(header["COLS"])

    # ---- frames ----
    while i < len(lines):
        if not lines[i].startswith("Frame"):
            i += 1
            continue

        i += 1  # first grid row
        raw = np.array([lines[i + r].strip().split(",") for r in range(rows)], dtype=object)
        i += rows

        # Keep only rows/cols that are NOT all 'B'
        keep_r = ~(raw == "B").all(axis=1)
        keep_c = ~(raw == "B").all(axis=0)

        clean = raw[keep_r][:, keep_c].astype(float)  # now no 'B' cells remain
        frames.append(clean)

    data = np.stack(frames)  # (T, H, W)

    # ---- split into 4 sensors (quadrants) ----
    H, W = data.shape[1:]
    h2, w2 = H // 2, W // 2
    s1 = data[:, :h2, :w2]
    s2 = data[:, :h2, w2:]
    s3 = data[:, h2:, :w2]
    s4 = data[:, h2:, w2:]

    if sensor:
        return (s1, s2, s3, s4)[sensor-1]
    else:
        return header, (s1, s2, s3, s4)

def get_frame_at_F(F, instron_data, header, return_t: bool=False):
    """Get which frame of a tekscan sensor movie corresponds to a given instron Force F (N) from a synchronised instron test"""
    t = instron_data[instron_data['force']>=F/1000].iloc[0]['time']
    frame = np.ceil(t / float(header['SECONDS_PER_FRAME'])).astype(int)
    if return_t:
        return frame, t
    else:
        return frame

def force_per_frame(frames, sensor_area=1.6129):
    """
    Total tekscan force for each tekscan frame\n
    sensor area in mm^2 and tekscan sensel values in MPa gives Newtons
    """
    return frames.sum(axis=(1, 2)) * sensor_area

def F2P(raw_frames, Forces, sensel_area=1.6129, i=11, j=11):
    """
    Convert raw tekscan frames to pressure frames by distributing force evenly over the raw values\n
    raw_frames: (N,i,j) array OR list of (i,j) arrays\n
    sensel area: (13.97/11)**2 mm^2\n
    sensel_area in mm and F in N gives MPa
    """

    raw = np.asarray(raw_frames, float).reshape(-1, i, j)
    R = raw.sum(axis=(1, 2)) # total raw value per frame
    F = np.asarray(Forces, dtype=float).ravel() # toal force per fram
    scale = (F / np.where(R == 0, 1.0, R)).reshape(F.shape[0], 1, 1) # Force per raw unit
                                                                    # avoid divide by zero but still outputs zeros cos raw=all 0s
    return (raw * scale) / sensel_area # raw units per sensel * force per raw unit / sensel area = Pressure per sensel

def get_sensor_loc(mc1_mesh, guide_wall_z=10, sensor_offset_z=-1, sensor_size=13.97):
    """
    mc1_mesh: should be aligned with x-axis with cartilage toward negative end\n
    guide_wall_z: z offset of guide inner wall (was 10(mm) for skinny ledge and -6.9(mm) for big ledge)\n
    sensor_offset_z: z offset of sensor from guide_wall (~ -1 mm) -ive if guide_wall_z is +ive and vice verse\n
    \n
    returns: sensor centre coord (normal is (1, 0, 0))
    """

    sign = np.sign(sensor_offset_z)
    x = mc1_mesh.points[:, 0].min()
    z = guide_wall_z + (sign*(sensor_size/2)) + sensor_offset_z
    return np.array([x, 0, z])

def project_sensor(mesh: pv.PolyData, sensor: pv.Plane, sensor_vals, data_loc='cells', downscale_fea=True, return_fea_grid=False, downscale_mode='mean'):
    """
    Projects (planar) sensor values onto mesh and if downscale_fea: downscale mesh['CPRESS'] values onto sensor grid\n
    Assigns array of values to mesh['tek_press'] - point_data if data_loc='points' else cell_data\n
    Assigns downscaled fea values to mesh['fea_press']\n
    if return_fea_grid: return nci, ncj grid of downscaled fea values (if downscale_fea=True)\n
    dowscale_mode: I think mean over the active sensel area (0.635x0.635mm RWxRW) is more representative of what tekscan does
    """

    # n sensor cells
    nci = sensor_vals.shape[0]
    ncj = sensor_vals.shape[1]

    # create orthonormal basis from sensor normal
    n = np.asarray(sensor.compute_normals()['Normals'][0])
    n = n / np.linalg.norm(n)

    a = np.array((0.0, 0.0, 1.0))
    if abs(np.dot(a, n)) > 0.99:
        a = np.array((0.0, 1.0, 0.0))

    u = np.cross(a, n); u /= np.linalg.norm(u)
    v = np.cross(n, u); v /= np.linalg.norm(v)

    # sensor bounds in u v coords
    P0 = np.asarray(sensor.center)
    pp = sensor.points - P0 # vector of each point from sensor centre (the points are the corners of each sensor cell)
    su = pp @ u             # vectors values in u direction
    sv = pp @ v             # vectors values in v direction
    u_min, u_max = su.min(), su.max() # outer bounds of sensor
    v_min, v_max = sv.min(), sv.max() # outer bounds of sensor

    du = (u_max - u_min) / ncj # size of each cell in u (column spacing - pitch)
    dv = (v_max - v_min) / nci # size of each cell in v (row spacing - pitch)

    # bin mesh cell centres in u v sensor cell bins
    centres = mesh.cell_centers().points
    if data_loc == 'points':
        centres = mesh.points

    cc = centres - P0 # vector of each cell centre from sensor centre
    cu = cc @ u # vectors values in u direction
    cv = cc @ v # vectors values in v direction

    iu = np.floor((cu - u_min) / du).astype(int) # j of sensor cell that the cell centre falls in (not bounded by sensor yet)
    iv = np.floor((cv - v_min) / dv).astype(int) # i of sensor cell that the cell centre falls in (not bounded by sensor yet)

    out = np.zeros(len(centres), dtype=float)
    inside = (iu >= 0) & (iu < ncj) & (iv >= 0) & (iv < nci) # mask of cell centres that are within bounds of sensor
    out[inside] = sensor_vals[iv[inside], iu[inside]]  # assign sensor values to mesh cells

    mesh['sensor_cell'] = list(zip(iv, iu))
    mesh["tek_press"] = out

    if downscale_fea:
        fea_vals = mesh.point_data_to_cell_data()['CPRESS']
        if data_loc == 'points':
            fea_vals = mesh['CPRESS']

        if downscale_mode == 'mean':
            # accumulate mean in sensor grid
            fea_sum = np.zeros((nci, ncj))
            fea_count = np.zeros((nci, ncj), dtype=int)
            np.add.at(fea_sum, (iv[inside], iu[inside]), fea_vals[inside])
            np.add.at(fea_count, (iv[inside], iu[inside]), 1)
            fea_grid = np.zeros((nci, ncj))
            np.divide(fea_sum, fea_count, out=fea_grid, where=fea_count > 0)
        elif downscale_mode == 'max':
            fea_grid = np.full((nci, ncj), -np.inf, dtype=float)
            fea_count = np.zeros((nci, ncj), dtype=int)
            np.maximum.at(fea_grid, (iv[inside], iu[inside]), fea_vals[inside])
            np.add.at(fea_count, (iv[inside], iu[inside]), 1)
            fea_grid[fea_count == 0] = 0.0

        # reproject onto mesh
        fea_ds = np.zeros(len(centres))
        valid = inside.copy()
        valid[inside] &= (fea_count[iv[inside], iu[inside]] > 0)
        fea_ds[valid] = fea_grid[iv[valid], iu[valid]]
        mesh["fea_press"] = fea_ds
        if return_fea_grid:
            return fea_grid
    

#---------------------------------------- project_sensor_new ----------------------------------------#
#-------------------- takes average pressure of nodes within active sensel area ---------------------#


def _sensor_basis(sensor: pv.PolyData):
    """
    Returns sensor centre P0 and orthonormal basis vectors n, u, v.

    u and v span the sensor plane.
    n is the sensor normal.
    """
    normals = sensor.compute_normals(cell_normals=False, point_normals=True)
    n = np.asarray(normals["Normals"][0], dtype=float)
    n /= np.linalg.norm(n)

    a = np.array((0.0, 0.0, 1.0))
    if abs(np.dot(a, n)) > 0.99:
        a = np.array((0.0, 1.0, 0.0))

    u = np.cross(a, n)
    u /= np.linalg.norm(u)

    v = np.cross(n, u)
    v /= np.linalg.norm(v)

    P0 = np.asarray(sensor.center, dtype=float)

    return P0, n, u, v


def _sensor_grid_geometry(sensor: pv.PolyData, nci: int, ncj: int):
    """
    Returns sensor bounds and pitch in local u-v coordinates.
    """
    P0, n, u, v = _sensor_basis(sensor)

    pp = sensor.points - P0
    su = pp @ u
    sv = pp @ v

    u_min, u_max = su.min(), su.max()
    v_min, v_max = sv.min(), sv.max()

    du = (u_max - u_min) / ncj
    dv = (v_max - v_min) / nci

    return P0, n, u, v, u_min, u_max, v_min, v_max, du, dv


def downscale_fea_nodes_to_active_sensel_grid(
    mesh: pv.PolyData,
    sensor: pv.PolyData,
    nci: int,
    ncj: int,
    pressure_name: str = "CPRESS",
    active_sensel_width: float | None = 0.635, # 1.27 / 2 (mm)
):
    """
    For each active sensel square, finds FE mesh nodes whose projected positions
    fall inside that active area and takes the arithmetic mean of their nodal
    pressure values.

    Parameters
    ----------
    mesh:
        FE surface mesh with nodal pressure values.
    sensor:
        Tekscan sensor grid as pv.PolyData.
    nci, ncj:
        Number of sensor rows and columns.
    pressure_name:
        Name of nodal pressure array in mesh.point_data.
    active_sensel_width:
        Width of the square active sensel region centred in each sensor cell.
        Uses the same length units as mesh/sensor coordinates.
        If None, uses the full pitch min(du, dv).

    Returns
    -------
    fea_grid:
        Mean nodal FE pressure per active sensel area.
    fea_count:
        Number of FE nodes contributing to each sensel.
    """
    if pressure_name not in mesh.point_data:
        raise KeyError(f"'{pressure_name}' must exist in mesh.point_data.")

    (
        P0, n, u, v,
        u_min, u_max,
        v_min, v_max,
        du, dv,
    ) = _sensor_grid_geometry(sensor, nci, ncj)

    if active_sensel_width is None:
        active_sensel_width = min(du, dv)

    active_sensel_width = float(active_sensel_width)

    if active_sensel_width <= 0:
        raise ValueError("active_sensel_width must be positive.")

    if active_sensel_width > min(du, dv):
        raise ValueError(
            "active_sensel_width should not exceed the sensor pitch. "
            f"Got active_sensel_width={active_sensel_width}, du={du}, dv={dv}."
        )

    half_w = 0.5 * active_sensel_width

    p = np.asarray(mesh.point_data[pressure_name], dtype=float)

    pp = mesh.points - P0
    pu = pp @ u
    pv_ = pp @ v

    # Pitch-cell indices containing each node.
    iu = np.floor((pu - u_min) / du).astype(int)
    iv = np.floor((pv_ - v_min) / dv).astype(int)

    inside_pitch = (
        (iu >= 0) & (iu < ncj) &
        (iv >= 0) & (iv < nci)
    )

    # Centres of each node's assigned sensor pitch cell.
    uc = u_min + (iu + 0.5) * du
    vc = v_min + (iv + 0.5) * dv

    # Node must also be inside the central active square.
    inside_active = (
        inside_pitch &
        (np.abs(pu - uc) <= half_w) &
        (np.abs(pv_ - vc) <= half_w)
    )

    fea_sum = np.zeros((nci, ncj), dtype=float)
    fea_count = np.zeros((nci, ncj), dtype=int)

    np.add.at(
        fea_sum,
        (iv[inside_active], iu[inside_active]),
        p[inside_active],
    )

    np.add.at(
        fea_count,
        (iv[inside_active], iu[inside_active]),
        1,
    )

    fea_grid = np.zeros((nci, ncj), dtype=float)
    np.divide(
        fea_sum,
        fea_count,
        out=fea_grid,
        where=fea_count > 0,
    )

    return fea_grid, fea_count


def project_sensor_new(
    mesh: pv.PolyData,
    sensor: pv.PolyData,
    sensor_vals,
    data_loc: str = "cells",
    downscale_fea: bool = True,
    return_fea_grid: bool = False,
    active_sensel_width: float | None = None,
    pressure_name: str = "CPRESS",
):
    """
    Projects Tekscan values onto the FE mesh.

    Optionally downscales FE nodal pressure onto the sensor grid by averaging
    FE nodes that fall inside each central active sensel area.
    """
    if data_loc not in {"cells", "points"}:
        raise ValueError("data_loc must be either 'cells' or 'points'.")

    sensor_vals = np.asarray(sensor_vals, dtype=float)

    nci = sensor_vals.shape[0]
    ncj = sensor_vals.shape[1]

    (
        P0, n, u, v,
        u_min, u_max,
        v_min, v_max,
        du, dv,
    ) = _sensor_grid_geometry(sensor, nci, ncj)

    if data_loc == "points":
        locs = mesh.points
    else:
        locs = mesh.cell_centers().points

    pp = locs - P0
    pu = pp @ u
    pv_ = pp @ v

    iu = np.floor((pu - u_min) / du).astype(int)
    iv = np.floor((pv_ - v_min) / dv).astype(int)

    inside = (
        (iu >= 0) & (iu < ncj) &
        (iv >= 0) & (iv < nci)
    )

    tek_press = np.zeros(len(locs), dtype=float)
    tek_press[inside] = sensor_vals[iv[inside], iu[inside]]

    sensor_cell = np.full((len(locs), 2), -1, dtype=int)
    sensor_cell[inside, 0] = iv[inside]
    sensor_cell[inside, 1] = iu[inside]

    if data_loc == "points":
        mesh.point_data["sensor_cell"] = sensor_cell
        mesh.point_data["tek_press"] = tek_press
    else:
        mesh.cell_data["sensor_cell"] = sensor_cell
        mesh.cell_data["tek_press"] = tek_press

    if downscale_fea:
        fea_grid, fea_count = downscale_fea_nodes_to_active_sensel_grid(
            mesh=mesh,
            sensor=sensor,
            nci=nci,
            ncj=ncj,
            pressure_name=pressure_name,
            active_sensel_width=active_sensel_width,
        )

        # Reproject downscaled FE grid back onto mesh locations.
        fea_press = np.zeros(len(locs), dtype=float)

        valid = inside.copy()
        valid[inside] &= fea_count[iv[inside], iu[inside]] > 0

        fea_press[valid] = fea_grid[iv[valid], iu[valid]]

        if data_loc == "points":
            mesh.point_data["fea_press"] = fea_press
        else:
            mesh.cell_data["fea_press"] = fea_press

        if return_fea_grid:
            return fea_grid



#---------------------------------------- project_sensor_complex ----------------------------------------#
#---------------------------------- see last function for description -----------------------------------#


def _sensor_basis(sensor: pv.PolyData):
    """
    Returns sensor centre P0 and orthonormal basis vectors n, u, v.

    u and v span the sensor plane.
    n is the sensor normal.
    """
    normals = sensor.compute_normals(cell_normals=False, point_normals=True)
    n = np.asarray(normals["Normals"][0], dtype=float)
    n /= np.linalg.norm(n)

    a = np.array((0.0, 0.0, 1.0))
    if abs(np.dot(a, n)) > 0.99:
        a = np.array((0.0, 1.0, 0.0))

    u = np.cross(a, n)
    u /= np.linalg.norm(u)

    v = np.cross(n, u)
    v /= np.linalg.norm(v)

    P0 = np.asarray(sensor.center, dtype=float)

    return P0, n, u, v


def _sensor_grid_geometry(sensor: pv.PolyData, nci: int, ncj: int):
    """
    Returns sensor bounds and pitch in local u-v coordinates.
    """
    P0, n, u, v = _sensor_basis(sensor)

    pp = sensor.points - P0
    su = pp @ u
    sv = pp @ v

    u_min, u_max = su.min(), su.max()
    v_min, v_max = sv.min(), sv.max()

    du = (u_max - u_min) / ncj
    dv = (v_max - v_min) / nci

    return P0, n, u, v, u_min, u_max, v_min, v_max, du, dv


def _iter_triangle_faces(mesh: pv.PolyData):
    """
    Yields triangle point-id triplets from a PolyData surface.

    Assumes the mesh is triangulated or triangle-only.
    """
    faces = mesh.faces
    idx = 0

    while idx < len(faces):
        npts = faces[idx]
        ids = faces[idx + 1: idx + 1 + npts]

        if npts != 3:
            raise ValueError(
                "Mesh contains non-triangle faces. "
                "Call mesh = mesh.triangulate() before using this function."
            )

        yield ids

        idx += npts + 1


def interpolate_nodal_pressure_over_element_faces(
    mesh: pv.PolyData,
    pressure_name: str = "CPRESS",
    out_name: str = "CPRESS_FACE",
    inplace: bool = False,
):
    """
    Converts nodal pressure on a triangular PolyData surface to element-face pressure.

    For a linear triangular element, the area-average of a linearly interpolated
    nodal pressure field is the arithmetic mean of the three nodal values.

    Parameters
    ----------
    mesh:
        Triangle-only pyvista.PolyData surface.
    pressure_name:
        Name of nodal pressure array in mesh.point_data.
    out_name:
        Name of output cell_data array.
    inplace:
        If True, modifies mesh. If False, returns a copied mesh.

    Returns
    -------
    pv.PolyData
        Mesh with cell_data[out_name].
    """
    if pressure_name not in mesh.point_data:
        raise KeyError(f"'{pressure_name}' must exist in mesh.point_data.")

    out_mesh = mesh if inplace else mesh.copy()

    p_node = np.asarray(out_mesh.point_data[pressure_name], dtype=float)
    p_face = np.empty(out_mesh.n_cells, dtype=float)

    for c, tri_ids in enumerate(_iter_triangle_faces(out_mesh)):
        p_face[c] = np.mean(p_node[tri_ids])

    out_mesh.cell_data[out_name] = p_face

    return out_mesh


def _clip_polygon_against_halfspace(poly, inside_fn, intersect_fn):
    """
    Sutherland-Hodgman polygon clipping helper.
    """
    if len(poly) == 0:
        return []

    clipped = []
    prev = poly[-1]
    prev_inside = inside_fn(prev)

    for curr in poly:
        curr_inside = inside_fn(curr)

        if curr_inside:
            if not prev_inside:
                clipped.append(intersect_fn(prev, curr))
            clipped.append(curr)
        elif prev_inside:
            clipped.append(intersect_fn(prev, curr))

        prev = curr
        prev_inside = curr_inside

    return clipped


def _clip_polygon_to_rect(poly, u0, u1, v0, v1):
    """
    Clips a 2D polygon to an axis-aligned rectangle in u-v coordinates.

    poly is a list of [u, v] points.
    """

    def interp_to_u(a, b, u_clip):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        t = (u_clip - a[0]) / (b[0] - a[0])
        return (a + t * (b - a)).tolist()

    def interp_to_v(a, b, v_clip):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        t = (v_clip - a[1]) / (b[1] - a[1])
        return (a + t * (b - a)).tolist()

    poly = _clip_polygon_against_halfspace(
        poly,
        inside_fn=lambda p: p[0] >= u0,
        intersect_fn=lambda a, b: interp_to_u(a, b, u0),
    )

    poly = _clip_polygon_against_halfspace(
        poly,
        inside_fn=lambda p: p[0] <= u1,
        intersect_fn=lambda a, b: interp_to_u(a, b, u1),
    )

    poly = _clip_polygon_against_halfspace(
        poly,
        inside_fn=lambda p: p[1] >= v0,
        intersect_fn=lambda a, b: interp_to_v(a, b, v0),
    )

    poly = _clip_polygon_against_halfspace(
        poly,
        inside_fn=lambda p: p[1] <= v1,
        intersect_fn=lambda a, b: interp_to_v(a, b, v1),
    )

    return poly


def _polygon_area_centroid(poly):
    """
    Returns signed-area magnitude and centroid of a 2D polygon.

    Parameters
    ----------
    poly:
        Sequence of [u, v] points.

    Returns
    -------
    area, centroid
    """
    pts = np.asarray(poly, dtype=float)

    if len(pts) < 3:
        return 0.0, np.array([np.nan, np.nan])

    x = pts[:, 0]
    y = pts[:, 1]

    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)

    cross = x * y_next - x_next * y
    signed_area = 0.5 * np.sum(cross)

    if abs(signed_area) < 1e-18:
        return 0.0, np.array([np.nan, np.nan])

    cx = np.sum((x + x_next) * cross) / (6.0 * signed_area)
    cy = np.sum((y + y_next) * cross) / (6.0 * signed_area)

    return abs(signed_area), np.array([cx, cy])


def _triangle_pressure_affine_coeffs(uv_tri, p_tri):
    """
    Fits p(u, v) = c0 + c1*u + c2*v exactly through triangle nodal values.
    """
    A = np.column_stack([
        np.ones(3),
        uv_tri[:, 0],
        uv_tri[:, 1],
    ])

    return np.linalg.solve(A, p_tri)

def build_sensor_mesh(centre_coord, normal = (1, 0, 0), ncells=11, size=13.97):
    return pv.Plane(
        center=centre_coord,
        direction=normal,
        i_size=size, j_size=size,
        i_resolution=ncells, j_resolution=ncells, 
    )


def downscale_fea_to_active_sensel_grid(
    mesh: pv.PolyData,
    sensor: pv.PolyData,
    nci: int,
    ncj: int,
    pressure_name: str = "CPRESS",
    active_sensel_width: float | None = None,
    return_force_grid: bool = False,
):
    """
    Downscales FE nodal contact pressure to the Tekscan sensor grid.

    The FE pressure is linearly interpolated over each triangular element face.
    Each triangle is clipped against the central active area of each sensel.
    Pressure is then area-integrated over the clipped polygon.

    Parameters
    ----------
    mesh:
        Triangle-only pyvista.PolyData surface with nodal pressure.
    sensor:
        11x11 sensor grid as a pv.PolyData mesh.
    nci, ncj:
        Number of sensor rows and columns.
    pressure_name:
        Name of nodal pressure array in mesh.point_data.
    active_sensel_width:
        Width of the square active area centred in each pitch cell.
        Uses the same length units as mesh/sensor coordinates.
        If None, the full cell pitch min(du, dv) is used.
    return_force_grid:
        If True, also returns integrated force per sensel.

    Returns
    -------
    fea_press_grid:
        Area-weighted mean FE pressure over each active sensel area.
    fea_force_grid:
        Optional. Integrated FE force over each active sensel area.
    active_area_grid:
        Actual FE-covered active area contributing to each sensel.
    """
    if pressure_name not in mesh.point_data:
        raise KeyError(f"'{pressure_name}' must exist in mesh.point_data.")

    (
        P0, n, u, v,
        u_min, u_max,
        v_min, v_max,
        du, dv,
    ) = _sensor_grid_geometry(sensor, nci, ncj)

    if active_sensel_width is None:
        active_sensel_width = min(du, dv)

    active_sensel_width = float(active_sensel_width)

    if active_sensel_width <= 0:
        raise ValueError("active_sensel_width must be positive.")

    if active_sensel_width > min(du, dv):
        raise ValueError(
            "active_sensel_width should not exceed the sensor pitch. "
            f"Got active_sensel_width={active_sensel_width}, "
            f"du={du}, dv={dv}."
        )

    half_w = 0.5 * active_sensel_width

    p_node = np.asarray(mesh.point_data[pressure_name], dtype=float)

    fea_force_grid = np.zeros((nci, ncj), dtype=float)
    active_area_grid = np.zeros((nci, ncj), dtype=float)

    mesh_uv = np.column_stack([
        (mesh.points - P0) @ u,
        (mesh.points - P0) @ v,
    ])

    for tri_ids in _iter_triangle_faces(mesh):
        tri_ids = np.asarray(tri_ids, dtype=int)

        uv_tri = mesh_uv[tri_ids]
        p_tri = p_node[tri_ids]

        # Skip degenerate triangles in sensor projection.
        try:
            coeffs = _triangle_pressure_affine_coeffs(uv_tri, p_tri)
        except np.linalg.LinAlgError:
            continue

        tri_poly = uv_tri.tolist()

        tri_u_min = uv_tri[:, 0].min()
        tri_u_max = uv_tri[:, 0].max()
        tri_v_min = uv_tri[:, 1].min()
        tri_v_max = uv_tri[:, 1].max()

        # Candidate sensel indices whose active squares may overlap this triangle.
        j0 = int(np.floor((tri_u_min - half_w - u_min) / du))
        j1 = int(np.floor((tri_u_max + half_w - u_min) / du))

        i0 = int(np.floor((tri_v_min - half_w - v_min) / dv))
        i1 = int(np.floor((tri_v_max + half_w - v_min) / dv))

        j0 = max(j0, 0)
        j1 = min(j1, ncj - 1)

        i0 = max(i0, 0)
        i1 = min(i1, nci - 1)

        if j1 < j0 or i1 < i0:
            continue

        for i in range(i0, i1 + 1):
            vc = v_min + (i + 0.5) * dv
            rect_v0 = vc - half_w
            rect_v1 = vc + half_w

            for j in range(j0, j1 + 1):
                uc = u_min + (j + 0.5) * du
                rect_u0 = uc - half_w
                rect_u1 = uc + half_w

                clipped = _clip_polygon_to_rect(
                    tri_poly,
                    rect_u0,
                    rect_u1,
                    rect_v0,
                    rect_v1,
                )

                area, centroid = _polygon_area_centroid(clipped)

                if area <= 0:
                    continue

                # For affine pressure over the triangle, the area-average pressure
                # over the clipped polygon is pressure evaluated at its centroid.
                p_avg = coeffs[0] + coeffs[1] * centroid[0] + coeffs[2] * centroid[1]

                fea_force_grid[i, j] += p_avg * area
                active_area_grid[i, j] += area

    fea_press_grid = np.zeros((nci, ncj), dtype=float)
    np.divide(
        fea_force_grid,
        active_area_grid,
        out=fea_press_grid,
        where=active_area_grid > 0,
    )

    if return_force_grid:
        return fea_press_grid, fea_force_grid, active_area_grid

    return fea_press_grid, active_area_grid


def project_sensor_complex(
    mesh: pv.PolyData,
    sensor: pv.PolyData,
    sensor_vals,
    data_loc: str = "cells",
    downscale_fea: bool = True,
    return_fea_grid: bool = False,
    active_sensel_width: float | None = None,
    pressure_name: str = "CPRESS",
):
    """
    Projects planar Tekscan sensor values onto an FE mesh.

    Also optionally downscales FE nodal contact pressure onto the sensor grid
    using active sensel areas rather than full pitch cells. 
    - Interpolates the nodal pressures over the element faces
    - Clips the elements intersecting the active sensel region 
    - Integrates pressure over the clipped areas and divides by active area

    Parameters
    ----------
    mesh:
        FE contact surface as pyvista.PolyData.
    sensor:
        11x11 Tekscan sensor grid as pyvista.PolyData.
    sensor_vals:
        2D sensor array with shape (n_rows, n_cols).
    data_loc:
        'cells' or 'points'. Controls where tek_press and fea_press are assigned.
    downscale_fea:
        If True, downscale mesh.point_data[pressure_name] to sensor grid.
    return_fea_grid:
        If True, returns downscaled FE grid.
    active_sensel_width:
        Width of square active sensel region centred in each sensor pitch cell.
        Same units as mesh and sensor coordinates.
        For the 6900 row/column width, this would typically be about 0.635 mm
        if your model units are mm.
    pressure_name:
        Nodal FE pressure array name.

    Returns
    -------
    Optional[np.ndarray]
        If return_fea_grid=True and downscale_fea=True, returns fea_grid.
    """
    if data_loc not in {"cells", "points"}:
        raise ValueError("data_loc must be either 'cells' or 'points'.")

    sensor_vals = np.asarray(sensor_vals, dtype=float)

    nci = sensor_vals.shape[0]
    ncj = sensor_vals.shape[1]

    (
        P0, n, u, v,
        u_min, u_max,
        v_min, v_max,
        du, dv,
    ) = _sensor_grid_geometry(sensor, nci, ncj)

    # Select mesh locations for Tekscan projection.
    if data_loc == "points":
        locs = mesh.points
    else:
        locs = mesh.cell_centers().points

    cc = locs - P0
    cu = cc @ u
    cv = cc @ v

    iu = np.floor((cu - u_min) / du).astype(int)
    iv = np.floor((cv - v_min) / dv).astype(int)

    inside = (
        (iu >= 0) & (iu < ncj) &
        (iv >= 0) & (iv < nci)
    )

    tek_press = np.zeros(len(locs), dtype=float)
    tek_press[inside] = sensor_vals[iv[inside], iu[inside]]

    sensor_cell = np.full((len(locs), 2), -1, dtype=int)
    sensor_cell[inside, 0] = iv[inside]
    sensor_cell[inside, 1] = iu[inside]

    if data_loc == "points":
        mesh.point_data["sensor_cell"] = sensor_cell
        mesh.point_data["tek_press"] = tek_press
    else:
        mesh.cell_data["sensor_cell"] = sensor_cell
        mesh.cell_data["tek_press"] = tek_press

    if downscale_fea:
        # Use triangulated copy for geometric integration.
        tri_mesh = mesh.triangulate()

        fea_grid, active_area_grid = downscale_fea_to_active_sensel_grid(
            tri_mesh,
            sensor=sensor,
            nci=nci,
            ncj=ncj,
            pressure_name=pressure_name,
            active_sensel_width=active_sensel_width,
            return_force_grid=False,
        )

        # Reproject downscaled FE values back onto the requested mesh locations.
        fea_press = np.zeros(len(locs), dtype=float)
        valid = inside.copy()

        valid[inside] &= active_area_grid[iv[inside], iu[inside]] > 0
        fea_press[valid] = fea_grid[iv[valid], iu[valid]]

        if data_loc == "points":
            mesh.point_data["fea_press"] = fea_press
        else:
            mesh.cell_data["fea_press"] = fea_press

        if return_fea_grid:
            return fea_grid
