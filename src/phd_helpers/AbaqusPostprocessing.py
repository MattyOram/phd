import numpy as np
import pandas as pd
import re
from pathlib import Path

def parse_wallclock_time(dat_file):
    text = Path(dat_file).read_text(errors="ignore")
    matches = re.findall(r"WALLCLOCK TIME \(SEC\)\s*=\s*([0-9]+(?:\.[0-9]+)?)", text)
    if not matches:
        raise ValueError("Could not find WALLCLOCK TIME in .dat file")
    return float(matches[-1])

def get_history_path(DIR, step=0):
    return DIR / (f"history_step-{step}.csv")

def get_field_path(DIR, metric, step, frame, instance):
    csvs = list(DIR.glob("*.csv"))
    return [x for x in csvs if f'{instance}-' in x.name and f'-{metric}-' in x.name and f'-{str(step)}-' in x.name and f'-{str(frame)}' in x.name][0]

def get_field_df(field_path):
    metric = field_path.name.split('-')[1]
    headers = []
    with open (field_path, 'r') as f:
        for line in f:
            headers = np.array(line.split(','))
            break

    df = pd.read_csv(field_path, skiprows=[0], header=None)
    cc_idx = np.where(np.array(headers)=='componentCount')[0] # idx of componentCount header
    cc = df.iloc[:1, cc_idx].values[0][0] # component count
    if cc == 1:
        df.columns = list(headers[:-1]) + [metric]
    else:
        df.columns = list(headers[:-1]) + [f'{metric}{x}' for x in range(1, cc+1)]

    df['nodeLabel'] -= 1 # Abaqus starts at 1
    return df

def add_field_to_mesh(mesh, field_df):
    cc = field_df.loc[:0, ['componentCount']].values[0][0] # component count

    # *************** NEED TO DECIDE ON INTERPOLATION METHOD IF NOT NODAL *************** #
    pos = field_df.loc[:0, ['position']].values[0][0] # NODAL, ELEMENT, ...
    # *************** NEED TO DECIDE ON INTERPOLATION METHOD IF NOT NODAL *************** #
    if pos != 'NODAL':
        raise RuntimeError("NEED TO DECIDE ON, AND IMPLEMENT, INTERPOLATION METHOD IF NOT NODAL")

    for x in range(1, cc+1):
        mesh[field_df.columns[-x]] = np.zeros(mesh.n_points)
        mesh[field_df.columns[-x]][field_df['nodeLabel']] = field_df.iloc[:, -x]

def get_deformed_mesh(mesh):
    mesh_def = mesh.copy(deep=True)
    mesh_def.points += np.column_stack((mesh['U1'], mesh['U2'], mesh['U3']))
    return mesh_def


def extract_bone_cartilage(mesh):
    mesh_shell = mesh.extract_cells_by_type(5)
    cartilage_surf_mask = np.where(mesh_shell['region_id']==-2)[0]
    cartilage_surf = mesh_shell.extract_cells(cartilage_surf_mask)
    bone_shell = mesh_shell.extract_cells(cartilage_surf_mask, invert=True)
    return bone_shell, cartilage_surf

def compute_cartilage_height_change(mesh, mesh_def):
    """
    Adds values to mesh and mesh def point data
    h_diff: h_def - h\n
    h_ratio: h_diff / before\n
    i.e. diff and ratio are -ive if compressed\n
    ratio * 100 = % change
    """

    mesh['point_id'] = np.arange(mesh.n_points)

    bone_shell, cartilage_surf = extract_bone_cartilage(mesh)
    bone_shell_def, cartilage_surf_def = extract_bone_cartilage(mesh_def)
    cartilage_h = cartilage_surf.compute_implicit_distance(bone_shell.extract_surface(algorithm=None))['implicit_distance']
    cartilage_h_def = cartilage_surf_def.compute_implicit_distance(bone_shell_def.extract_surface(algorithm=None))['implicit_distance']

    cartilage_h_diff = cartilage_h_def - cartilage_h
    #cartilage_h_ratio = np.where(cartilage_h > 1e-10, cartilage_h_diff / cartilage_h, 0)
    cartilage_h_ratio = np.zeros(cartilage_surf.n_points)
    np.divide(cartilage_h_diff, cartilage_h, out=cartilage_h_ratio, where=cartilage_h > 1e-6)

    values = [cartilage_h, cartilage_h_def, cartilage_h_diff, cartilage_h_ratio]
    names = ['cartilage_h', 'cartilage_h_def', 'cartilage_h_diff', 'cartilage_h_ratio']
    for n, v in zip(names, values):
        mesh[n] = np.zeros(mesh.n_points)
        mesh[n][cartilage_surf['point_id']] = v
        mesh_def[n] = np.zeros(mesh.n_points)
        mesh_def[n][cartilage_surf['point_id']] = v






from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional

import numpy as np
import pyvista as pv

from phd_helpers.paths import find_shared_cells, quadratic_to_linear_mesh

# ============================================================
# Abaqus INP parser
# ============================================================

@dataclass
class SurfaceDef:
    name: str
    surface_type: str  # "NODE" or "ELEMENT"
    node_set_name: Optional[str] = None
    element_faces: List[Tuple[int, str]] = field(default_factory=list)  # [(eid, "S1"), ...]


@dataclass
class PartDef:
    name: str
    nodes: Dict[int, List[float]] = field(default_factory=dict)          # nid -> [x,y,z]
    elements: Dict[int, List[int]] = field(default_factory=dict)         # eid -> [nid,...]
    element_types: Dict[int, str] = field(default_factory=dict)          # eid -> "C3D4"/"C3D10"
    elsets: Dict[str, List[int]] = field(default_factory=dict)           # name -> [eid,...]
    nsets: Dict[str, List[int]] = field(default_factory=dict)            # name -> [nid,...]
    surfaces: Dict[str, SurfaceDef] = field(default_factory=dict)        # name -> SurfaceDef

class AbaqusInpParser:
    """
    parses:
      *PART, NAME=...
      *NODE
      *ELEMENT, TYPE=..., ELSET=...
      *ELSET, ELSET=...
      *NSET, NSET=...
      *SURFACE, TYPE=NODE, NAME=...
      *SURFACE, TYPE=ELEMENT, NAME=...
      *END PART

    """

    _re_keyval = re.compile(r"([A-Z0-9_ ]+)\s*=\s*([^,]+)", re.IGNORECASE)

    def __init__(self, inp_path: str | Path):
        self.inp_path = Path(inp_path)
        self.parts: Dict[str, PartDef] = {}

    @staticmethod
    def _clean_line(line: str) -> str:
        return line.strip()

    @classmethod
    def _parse_keyword(cls, line: str) -> Tuple[str, Dict[str, str]]:
        """
        e.g.:
            *ELEMENT, TYPE=C3D4, ELSET=mc1_BONE
        -> ("ELEMENT", {"TYPE": "C3D4", "ELSET": "mc1_BONE"})
        """
        assert line.startswith("*")
        items = [x.strip() for x in line[1:].split(",")]
        keyword = items[0].upper()
        params: Dict[str, str] = {}

        for item in items[1:]:
            if "=" in item:
                k, v = item.split("=", 1)
                params[k.strip().upper()] = v.strip()
        return keyword, params

    @staticmethod
    def _parse_int_list_lines(lines: List[str]) -> List[int]:
        out: List[int] = []
        for line in lines:
            if not line or line.startswith("**"):
                continue
            parts = [x.strip() for x in line.split(",") if x.strip()]
            out.extend(int(x) for x in parts)
        return out

    def parse(self) -> Dict[str, PartDef]:
        lines = self.inp_path.read_text().splitlines()
        i = 0
        current_part: Optional[PartDef] = None

        while i < len(lines):
            raw = lines[i]
            line = self._clean_line(raw)

            if not line or line.startswith("**"):
                i += 1
                continue

            if not line.startswith("*"):
                i += 1
                continue

            keyword, params = self._parse_keyword(line)

            # ------------------------------------------------
            # PART
            # ------------------------------------------------
            if keyword == "PART":
                part_name = params["NAME"]
                current_part = PartDef(name=part_name)
                self.parts[part_name] = current_part
                i += 1
                continue

            if keyword == "END PART":
                current_part = None
                i += 1
                continue

            # ignore everything outside PART blocks
            if current_part is None:
                i += 1
                continue

            # ------------------------------------------------
            # NODE
            # ------------------------------------------------
            if keyword == "NODE":
                i += 1
                while i < len(lines):
                    line = self._clean_line(lines[i])
                    if not line or line.startswith("**"):
                        i += 1
                        continue
                    if line.startswith("*"):
                        break

                    vals = [x.strip() for x in line.split(",")]
                    nid = int(vals[0])
                    xyz = [float(vals[1]), float(vals[2]), float(vals[3])]
                    current_part.nodes[nid] = xyz
                    i += 1
                continue

            # ------------------------------------------------
            # ELEMENT
            # ------------------------------------------------
            if keyword == "ELEMENT":
                elem_type = params["TYPE"].upper()
                # ELSET name is not required for parsing geometry, but present in writer
                i += 1
                while i < len(lines):
                    line = self._clean_line(lines[i])
                    if not line or line.startswith("**"):
                        i += 1
                        continue
                    if line.startswith("*"):
                        break

                    vals = [x.strip() for x in line.split(",") if x.strip()]
                    eid = int(vals[0])
                    conn = [int(x) for x in vals[1:]]
                    current_part.elements[eid] = conn
                    current_part.element_types[eid] = elem_type
                    i += 1
                continue

            # ------------------------------------------------
            # ELSET
            # ------------------------------------------------
            if keyword == "ELSET":
                elset_name = params["ELSET"]
                block_lines: List[str] = []
                i += 1
                while i < len(lines):
                    line = self._clean_line(lines[i])
                    if not line or line.startswith("**"):
                        i += 1
                        continue
                    if line.startswith("*"):
                        break
                    block_lines.append(line)
                    i += 1

                current_part.elsets[elset_name] = self._parse_int_list_lines(block_lines)
                continue

            # ------------------------------------------------
            # NSET
            # ------------------------------------------------
            if keyword == "NSET":
                nset_name = params["NSET"]
                block_lines: List[str] = []
                i += 1
                while i < len(lines):
                    line = self._clean_line(lines[i])
                    if not line or line.startswith("**"):
                        i += 1
                        continue
                    if line.startswith("*"):
                        break
                    block_lines.append(line)
                    i += 1

                current_part.nsets[nset_name] = self._parse_int_list_lines(block_lines)
                continue

            # ------------------------------------------------
            # SURFACE
            # ------------------------------------------------
            if keyword == "SURFACE":
                surface_name = params["NAME"]
                surface_type = params["TYPE"].upper()

                surf = SurfaceDef(name=surface_name, surface_type=surface_type)

                i += 1
                while i < len(lines):
                    line = self._clean_line(lines[i])
                    if not line or line.startswith("**"):
                        i += 1
                        continue
                    if line.startswith("*"):
                        break

                    vals = [x.strip() for x in line.split(",") if x.strip()]

                    if surface_type == "NODE":
                        # writer emits exactly one nset name on the next line
                        if len(vals) != 1:
                            raise ValueError(
                                f"{current_part.name}:{surface_name}: expected single NSET name for TYPE=NODE surface"
                            )
                        surf.node_set_name = vals[0]

                    elif surface_type == "ELEMENT":
                        # lines like: 123, S4
                        if len(vals) != 2:
                            raise ValueError(
                                f"{current_part.name}:{surface_name}: expected 'eid, Sx' for TYPE=ELEMENT surface"
                            )
                        surf.element_faces.append((int(vals[0]), vals[1].upper()))

                    else:
                        raise ValueError(
                            f"{current_part.name}:{surface_name}: unsupported surface type '{surface_type}'"
                        )

                    i += 1

                current_part.surfaces[surface_name] = surf
                continue

            # everything else inside PART is ignored
            i += 1

        return self.parts


# ============================================================
# Convert to PyVista
# ============================================================

ABAQUS_ELEM_TO_VTK = {
    "C3D4": pv.CellType.TETRA,
    "C3D10": pv.CellType.QUADRATIC_TETRA,
    "C3D4H": pv.CellType.TETRA,
    "C3D10H": pv.CellType.QUADRATIC_TETRA,
}


FACE_LABEL_TO_ID = {
    "S1": 1,
    "S2": 2,
    "S3": 3,
    "S4": 4,
}


def part_to_pyvista(part: PartDef) -> pv.UnstructuredGrid:
    """
    Build a PyVista UnstructuredGrid for one part.

    Point data:
      - one 0/1 array per NSET
      - one 0/1 array per NODE surface

    Cell data:
      - one 0/1 array per ELSET
      - one 0/1 array per ELEMENT surface
      - one face-id array per ELEMENT surface (SURFNAME__FACE_ID)
    """
    if not part.nodes:
        raise ValueError(f"{part.name}: no nodes found")
    if not part.elements:
        raise ValueError(f"{part.name}: no elements found")

    # ---- nodes: Abaqus node ids are 1-based, keep stable sorted order
    node_ids = np.array(sorted(part.nodes.keys()), dtype=int)
    nid_to_pid = {nid: i for i, nid in enumerate(node_ids)}
    points = np.array([part.nodes[nid] for nid in node_ids], dtype=float)

    # ---- elements: Abaqus element ids are 1-based, keep stable sorted order
    eids = np.array(sorted(part.elements.keys()), dtype=int)

    cell_types = []
    cell_conn_flat = []
    eid_to_cid = {}

    for cid, eid in enumerate(eids):
        etype = part.element_types[eid]
        if etype not in ABAQUS_ELEM_TO_VTK:
            raise ValueError(f"{part.name}: unsupported element type '{etype}'")

        conn_nids = part.elements[eid]
        conn_pids = [nid_to_pid[nid] for nid in conn_nids]

        cell_conn_flat.append(len(conn_pids))
        cell_conn_flat.extend(conn_pids)
        cell_types.append(int(ABAQUS_ELEM_TO_VTK[etype]))
        eid_to_cid[eid] = cid

    grid = pv.UnstructuredGrid(
        np.asarray(cell_conn_flat, dtype=np.int64),
        np.asarray(cell_types, dtype=np.uint8),
        points,
    )

    # --------------------------------------------------------
    # Point data from NSETs
    # --------------------------------------------------------
    for nset_name, ids in part.nsets.items():
        arr = np.zeros(grid.n_points, dtype=np.uint8)
        valid = [nid_to_pid[nid] for nid in ids if nid in nid_to_pid]
        arr[valid] = 1
        grid.point_data[nset_name] = arr

    # --------------------------------------------------------
    # Point data from node surfaces
    # --------------------------------------------------------
    for surf_name, surf in part.surfaces.items():
        if surf.surface_type != "NODE":
            continue

        arr = np.zeros(grid.n_points, dtype=np.uint8)
        if surf.node_set_name is None:
            raise ValueError(f"{part.name}:{surf_name}: NODE surface has no node_set_name")
        nids = part.nsets.get(surf.node_set_name, [])
        valid = [nid_to_pid[nid] for nid in nids if nid in nid_to_pid]
        arr[valid] = 1
        grid.point_data[surf_name] = arr

    # --------------------------------------------------------
    # Cell data from ELSETs
    # --------------------------------------------------------
    for elset_name, ids in part.elsets.items():
        arr = np.zeros(grid.n_cells, dtype=np.uint8)
        valid = [eid_to_cid[eid] for eid in ids if eid in eid_to_cid]
        arr[valid] = 1
        grid.cell_data[elset_name] = arr

    # --------------------------------------------------------
    # Cell data from element surfaces
    # --------------------------------------------------------
    for surf_name, surf in part.surfaces.items():
        if surf.surface_type != "ELEMENT":
            continue

        surf_mask = np.zeros(grid.n_cells, dtype=np.uint8)
        face_id = np.zeros(grid.n_cells, dtype=np.uint8)

        for eid, face_label in surf.element_faces:
            cid = eid_to_cid.get(eid)
            if cid is None:
                continue
            surf_mask[cid] = 1
            face_id[cid] = FACE_LABEL_TO_ID.get(face_label, 0)

        grid.cell_data[surf_name] = surf_mask
        grid.cell_data[f"{surf_name}__FACE_ID"] = face_id

    grid.point_data["abaqus_node_id"] = node_ids
    grid.cell_data["abaqus_element_id"] = eids

    return grid


def inp2pv(inp_path: str | Path) -> Dict[str, pv.UnstructuredGrid]:
    parser = AbaqusInpParser(inp_path)
    parts = parser.parse()
    return {part_name: part_to_pyvista(part_def) for part_name, part_def in parts.items()}

def build_tri_tet_mesh(mesh, bone):
    """Convert output of inp1pv back into mesh3D format"""
    mesh1 = mesh.copy(deep=True)
    mesh1 = quadratic_to_linear_mesh(mesh1)
    mesh1['region_id'] = np.ones(mesh1.n_cells, dtype=int)
    mesh1['region_id'][mesh1[f'{bone}_CARTILAGE']==1] = 2
    bone_shell = mesh1.extract_cells(mesh1['region_id']==1).extract_surface(algorithm=None)
    cart_shell = mesh1.extract_cells(mesh1['region_id']==2).extract_surface(algorithm=None)


    # find interface surfaces on cartilage and bone
    interface_mask_bone = find_shared_cells(bone_shell, cart_shell)
    interface_mask_cartilage = find_shared_cells(cart_shell, bone_shell)

    interface_surf = bone_shell.extract_cells(interface_mask_bone)
    interface_surf['region_id'] = np.full(interface_surf.n_cells, -3)

    bone_surf = bone_shell.extract_cells(~interface_mask_bone)
    bone_surf['region_id'] = np.full(bone_surf.n_cells, -1)
    cartilage_surf = cart_shell.extract_cells(~interface_mask_cartilage)
    cartilage_surf['region_id'] = np.full(cartilage_surf.n_cells, -2)

    return mesh1 + interface_surf + bone_surf + cartilage_surf