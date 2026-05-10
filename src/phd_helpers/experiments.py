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

def project_sensor(mesh, sensor: pv.Plane, sensor_vals, data_loc='cells', downscale_fea=True, return_fea_grid=False, downscale_mode='mean'):
    """Projects (planar) sensor values onto mesh and if downscale_fea: downscale mesh['CPRESS'] values onto sensor grid\n
    Assigns array of values to mesh['tek_press'] - point_data if data_loc='points' else cell_data\n
    Assigns downscaled fea values to mesh['fea_press']\n
    if return_fea_grid: return nci, ncj grid of downscaled fea values (if downscale_fea=True)"""

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
    
def build_sensor_mesh(centre_coord, normal = (1, 0, 0), ncells=11, size=13.97):
    return pv.Plane(
        center=centre_coord,
        direction=normal,
        i_size=size, j_size=size,
        i_resolution=ncells, j_resolution=ncells, 
    )