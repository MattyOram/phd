import numpy as np
import pandas as pd
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
    sensor area in mm^2
    """
    return frames.sum(axis=(1, 2)) * sensor_area

def get_sensor_loc(mc1_mesh, guide_wall_z=10, sensor_offset_z=-1, sensor_size=14):
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

