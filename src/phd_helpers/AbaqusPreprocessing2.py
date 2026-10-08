import numpy as np
import gdist
from scipy.spatial.distance import cdist
from pathlib import Path
import re

import pyvista as pv

from phd_helpers.paths import get_boundary, get_intercepts, get_intercepts_multi
from phd_helpers.MeshQuality import sample_surface

#•••••••••••••••••••••••• For changing alignment of joint to match instron misalignment ••••••••••••••••••••••••#
def rotate_mesh(mesh1, cor, Rx=0.0, Ry=0.0, Rz=0.0):
    """Centre of Rotation (cor)"""
    mesh = mesh1.copy()
    mesh.rotate_x(Rx, point=cor, inplace=True, transform_all_input_vectors=True)
    mesh.rotate_y(Ry, point=cor, inplace=True, transform_all_input_vectors=True)
    mesh.rotate_z(Rz, point=cor, inplace=True, transform_all_input_vectors=True)
    return mesh

def translate_mesh(mesh1, tx=0.0, ty=0.0, tz=0.0):
    mesh = mesh1.copy()
    mesh.translate([tx, ty, tz], inplace=True, transform_all_input_vectors=True)
    return mesh
#•••••••••••••••••••••••• For changing alignment of joint to match instron misalignment ••••••••••••••••••••••••#

def compute_x_dist(tpm, mc1, return_points=False, cartilage_id=-2, n_samples=20000):
    """Compute the minimum distance in the x direction between the trapezium and metacarpal cartilage surfaces"""
    # extract cartilage surfaces to speed up computation
    mc1_cartilage_surf = mc1.extract_cells(mc1['region_id']==cartilage_id).extract_surface(algorithm=None)
    tpm_cartilage_surf = tpm.extract_cells(tpm['region_id']==cartilage_id).extract_surface(algorithm=None)
    # compute distance in direction of metacarpal principal axis (x) between cartilage surfaces
    vecs_x = np.zeros_like(mc1_cartilage_surf.points)+np.array([-1, 0, 0])
    mc1_points = mc1_cartilage_surf.points
    if mc1_cartilage_surf.n_points > n_samples:
        print(f'sampling surface ({n_samples})')
        mc1_points = sample_surface(mc1_cartilage_surf, n_samples)
    points_tpm, points_mc1, mask_points_mc1 = get_intercepts(tpm_cartilage_surf, mc1_points, vecs_x)
    dists_x = np.linalg.norm(points_mc1 - points_tpm, axis=1)
    if not return_points:
        return min(dists_x)
    else:
        return min(dists_x), points_tpm, points_mc1
    
def position_mc1_tpm(mc1, tpm, target_dist=0.01, raise_error=False):
    """Given mc1 and tpm (with cartilage) in mc1 inertial basis 
    - moves tpm so that closest points on cartilage surfaces are target_dist apart in x direction\n
    Doesn't copy meshes - moves them inplace"""

    # move tpm into position
    tpm.points -= np.array([10, 0, 0]) # ensure no interference before computing x-distance
    min_dist_x = compute_x_dist(tpm, mc1)
    move_dist = min_dist_x - target_dist
    tpm.points += np.array([move_dist, 0, 0])

    # checks
    final_dist = abs(compute_x_dist(tpm, mc1))
    print(f"Distance between cartilage surfaces (x->): {final_dist:.4f}")
    interference_check = (tpm.extract_surface(algorithm=None)
                          .compute_implicit_distance(mc1.extract_surface(algorithm=None))['implicit_distance'] >= 0).all()
    print("No interference: ", interference_check)

    if raise_error:
        if not interference_check:
            raise RuntimeError(f"Bones not positioned correctly - Interference found")
        if abs(final_dist - target_dist) > 0.005:
            raise RuntimeError(f"Bones not positioned correctly - final dist = {final_dist:.4f}")


def bone_surface_patch_nodes(mesh, patch_dist, distance_measure="euclidean", only_full_face_nodes=True):
    """
    mesh: FE ready (positioned with region_id)\n
    patch_dist: min distance of nodes from cartilage edge e.g. float/int ; or within given x coordinates e.g. [-100, -5] if xlims\n
    distance_measure: "geodesic" or "euclidean" or "xlims"\n
    only_full_face_nodes: if True, remove nodes that do not form part of a complete mesh surface face\n\n
    Returns the node ids on the given mesh 
    """
    mesh['node_id'] = np.arange(mesh.n_points)
    mesh['cell_id'] = np.arange(mesh.n_cells)
    bone_surf = mesh.extract_cells(np.where(mesh['region_id']==-1)[0]).extract_surface(algorithm=None)
    bone_boudnary = get_boundary(bone_surf)
    boundary_mask = np.isin(bone_surf['node_id'], bone_boudnary['node_id']) # on bone surf

    if distance_measure == "geodesic":
        # min distance for every point from boundary points (inc. boundary points)
        dists = gdist.compute_gdist( 
            bone_surf.points.astype(np.float64),
            bone_surf.faces.reshape(-1, 4)[:, 1:].astype(np.int32),
            source_indices = np.arange(bone_surf.n_points)[boundary_mask].astype(np.int32)
        )
        bone_patch_mask = dists >= patch_dist # mask of nodes on bone surf 

    elif distance_measure == "euclidean":
        # min distance for every point from boundary points (inc. boundary points)
        dists = cdist(bone_surf.points[boundary_mask],  bone_surf.points).min(axis=0)
        bone_patch_mask = dists >= patch_dist # mask of nodes on bone surf 

    elif distance_measure == "xlims":
        Xs = bone_surf.points[:, 0]
        bone_patch_mask = (Xs>=patch_dist[0]) & (Xs<=patch_dist[1])

    else:
        raise ValueError("Invalid distance measure. Choose from: 'geodesic' or 'euclidean'")


    if only_full_face_nodes:
        extracted_surf = bone_surf.extract_points(bone_patch_mask, adjacent_cells=False).extract_surface(algorithm=None)
        bone_patch_mask = np.isin(bone_surf['node_id'], extracted_surf['node_id'])
        mesh['bc_patch'] = np.zeros(mesh.n_cells)
        mesh.cell_data['bc_patch'][extracted_surf['cell_id']] = 1

    # output node ids
    bone_patch_nodes = bone_surf['node_id'][bone_patch_mask] # nodes on full mesh

    # label output nodes on original mesh
    mesh.point_data['bc_patch'] = np.zeros(mesh.n_points)
    mesh.point_data['bc_patch'][bone_patch_nodes] = 1

    return bone_patch_nodes


def parse_memory_estimate(dat_file):
    """
    Parse Abaqus .dat file MEMORY ESTIMATE table.

    Returns:
        {
            "process": 1,
            "minimum_memory_required_gb": 0.354,
            "memory_to_minimize_io_gb": 3.380,
        }
    """
    text = Path(dat_file).read_text(errors="ignore")

    pattern = re.compile(
        r"""
        M\s*E\s*M\s*O\s*R\s*Y\s+E\s*S\s*T\s*I\s*M\s*A\s*T\s*E   # heading
        .*?                                                    # table header
        ^\s*(\d+)\s+                                           # process
        [0-9.E+-]+\s+                                          # floating point ops
        (\d+)\s+                                               # minimum memory (MB)
        (\d+)                                                  # memory to minimize I/O (MB)
        \s*$
        """,
        re.IGNORECASE | re.DOTALL | re.MULTILINE | re.VERBOSE,
    )

    m = pattern.search(text)
    if not m:
        raise ValueError("Could not find Abaqus MEMORY ESTIMATE table")

    return {
        "process": int(m.group(1)),
        "minimum_memory_required_gb": int(m.group(2))/1e3,
        "memory_to_minimize_io_gb": int(m.group(3))/1e3,
    }



# •••••••••••••••••••••••••••••••••••• INPUT FILE BUILDER •••••••••••••••••••••••••••••••••••• #

class AbaqusInpBuilder:
    """Write geometry only input file imported by the Abaqus/CAE python script.

    The generated file contains parts, nodes, tetrahedral elements, named
    element sets/surfaces, assembly instances, and reference-point node sets.
    """
    TET_FACES = {
        "S1": (0, 1, 2),
        "S2": (0, 3, 1),
        "S3": (1, 3, 2),
        "S4": (2, 3, 0),
    }

    ELEMENT_TYPE_MAP = {
        'C3D4': 10,
        'C3D10': 24,
        'C3D4H': 10,
        'C3D10H': 24,
        'C3D10HS': 24,
        'C3D10M': 24,
        'C3D10MH': 24
    }

    def __init__(self):
        # set abaqus elements types e.g. C3D4
        self.bone_element_type = None
        self.cartilage_element_type = None

        # cell type ids in pyvista e.g. 24 (tet10)
        self.tri_celltype = None 
        self.tet_celltype = None

        # region ids + which tet regions are written (controls Abaqus element-id numbering)
        self.region = {"bone": 1, "cartilage": 2}
        self.write_tet_regions = [self.region["bone"], self.region["cartilage"]]

        self.parts = {}
        self.surfaces = {}

        # assembly
        self.assembly_name = "ASSEMBLY"
        self.reference_points = {}

    # ----------------------------
    # helpers
    # ----------------------------
    @staticmethod
    def _get_cells_of_type(mesh: pv.UnstructuredGrid, cell_type: int):
        cell_ids = np.where(mesh.celltypes == cell_type)[0]
        conn = np.asarray(mesh.cells_dict[cell_type], dtype=int)
        return cell_ids, conn

    @classmethod
    def _map_tris2faces(
        cls,
        tet_full_ids,
        tet_conn_full,
        tri_conn_selected,
        fullcell_to_eid,
        part_name,
    ):
        face_map = {}
        for full_cid, conn in zip(tet_full_ids, tet_conn_full):
            corners = np.asarray(conn, dtype=int)[:4]  # tet4/tet10
            for face_label, loc in cls.TET_FACES.items():
                key = tuple(sorted(corners[list(loc)]))
                face_map.setdefault(key, (int(full_cid), face_label))

        tri_conn_selected = np.asarray(tri_conn_selected, dtype=int)
        if tri_conn_selected.size == 0:
            raise ValueError(f"{part_name}: no selected TRIANGLE cells found")

        abaqus_faces = set()
        for tri in tri_conn_selected:
            tri_corners = np.asarray(tri, dtype=int)[:3]  # tri3/tri6
            key = tuple(sorted(tri_corners))
            hit = face_map.get(key)
            if hit is None:
                continue
            full_cid, face = hit
            eid = fullcell_to_eid.get(full_cid)
            if eid is not None:
                abaqus_faces.add((eid, face))

        if not abaqus_faces:
            raise ValueError(f"{part_name}: no selected surface faces matched to tet faces")
        return abaqus_faces

    def _determine_elem_types(self, mesh):
        self.tri_celltype, self.tet_celltype = sorted(list(mesh.cells_dict.keys()))
        for element_type in (self.bone_element_type, self.cartilage_element_type):
            if element_type not in self.ELEMENT_TYPE_MAP:
                raise ValueError(f"Unsupported Abaqus element type: {element_type}")
            if self.ELEMENT_TYPE_MAP[element_type] != self.tet_celltype:
                raise ValueError(
                    f"Abaqus element type ({element_type}) does not match "
                    f"PyVista cell type ({self.tet_celltype})"
                )

    # ----------------------------
    # element preprocessing
    # ----------------------------
    def _preprocess_part_elements(self, part_name: str):
        mesh = self.parts[part_name]["mesh"]
        if "region_id" not in mesh.cell_data:
            raise KeyError(f"{part_name}: mesh.cell_data['region_id'] is missing")
        region_ids = np.asarray(mesh.cell_data["region_id"]).astype(int)

        tet_full_ids, tet_conn_full = self._get_cells_of_type(mesh, self.tet_celltype)
        tri_full_ids, tri_conn_full = self._get_cells_of_type(mesh, self.tri_celltype)

        tet_regions = region_ids[tet_full_ids]
        tri_regions = region_ids[tri_full_ids]

        write_mask = np.isin(tet_regions, self.write_tet_regions)
        if not np.any(write_mask):
            raise ValueError(f"{part_name}: no tet elements found in write_tet_regions={self.write_tet_regions}")

        fullcell_to_eid = {}
        eid = 1
        for full_cid, keep in zip(tet_full_ids, write_mask):
            if not keep:
                continue
            fullcell_to_eid[int(full_cid)] = eid
            eid += 1

        self.parts[part_name]["_elem_cache"] = {
            "tet_full_ids": tet_full_ids,
            "tet_conn_full": tet_conn_full,
            "tet_regions": tet_regions,
            "tri_conn_full": tri_conn_full,
            "tri_regions": tri_regions,
            "write_mask": write_mask,
            "fullcell_to_eid": fullcell_to_eid,
        }

    # ----------------------------
    # configuration
    # ----------------------------
    def set_region_ids(self, bone_region_id=1, cartilage_region_id=2):
        self.region["bone"] = int(bone_region_id)
        self.region["cartilage"] = int(cartilage_region_id)
        self.write_tet_regions = [self.region["bone"], self.region["cartilage"]]

    def set_assembly_name(self, name: str):
        self.assembly_name = str(name)

    def set_element_types(self, bone_element_type, cartilage_element_type):
        self.bone_element_type = bone_element_type
        self.cartilage_element_type = cartilage_element_type

    def add_part_from_vtu(self, part_name: str, part_mesh: str|pv.UnstructuredGrid, mode: str ='path', instance_name: str|None = None):
        if mode == 'path':
            mesh = pv.read(part_mesh)
        elif mode == 'mesh':
            mesh = part_mesh.copy(deep=True)
        else:
            raise ValueError("mode must be 'path' or 'mesh'")
        self.parts[part_name] = {
            "mesh": mesh,
            "instance_name": instance_name or f"{part_name}_INST",
        }
        self._determine_elem_types(mesh)
        self.surfaces[part_name] = {}
        self._preprocess_part_elements(part_name)

    def create_reference_point(self, rp_set_name: str, node_id: int, xyz):
        self.reference_points[rp_set_name] = {
            "node_id": int(node_id),
            "xyz": np.asarray(xyz, dtype=float),
        }

    def create_surface_from_element_faces(self, part_name: str, surface_name: str, face_pairs):
        if part_name not in self.parts:
            raise KeyError(f"Unknown part: {part_name}")
        self.surfaces[part_name][surface_name] = {
            "type": "element",
            "faces": [(int(e), str(face)) for (e, face) in face_pairs],
        }

    def add_surface_from_cell_data(self, part_name: str, cell_data_name: str, value, surface_name: str):
        if part_name not in self.parts:
            raise KeyError(f"Unknown part: {part_name}")
        c = self.parts[part_name]["_elem_cache"]
        mesh = self.parts[part_name]["mesh"]

        if cell_data_name not in mesh.cell_data:
            raise KeyError(f"{part_name}: mesh.cell_data['{cell_data_name}'] is missing")

        tri_full_ids, tri_conn_full = self._get_cells_of_type(mesh, self.tri_celltype)
        tri_values = np.asarray(mesh.cell_data[cell_data_name])[tri_full_ids]
        tri_conn_selected = tri_conn_full[tri_values == value]

        abaqus_faces = self._map_tris2faces(
            c["tet_full_ids"],
            c["tet_conn_full"],
            tri_conn_selected,
            c["fullcell_to_eid"],
            part_name,
        )
        self.create_surface_from_element_faces(part_name, surface_name, sorted(abaqus_faces))

    # ----------------------------
    # writer 
    # ----------------------------
    def write_input_file(self, output_inp: str):
        if not self.parts:
            raise ValueError("No parts defined.")

        bone_reg = self.region["bone"]
        cart_reg = self.region["cartilage"]

        with open(output_inp, "w") as f:
            f.write("** Mesh and topology for Abaqus/CAE import\n**\n")
            for part_name, pinfo in self.parts.items():
                mesh = pinfo["mesh"]
                f.write(f"*PART, NAME={part_name}\n")

                # NODES
                f.write("*NODE\n")
                for nid, xyz in enumerate(mesh.points, start=1):
                    f.write(f"{nid}, {xyz[0]}, {xyz[1]}, {xyz[2]}\n")

                c = pinfo["_elem_cache"]

                def write_elements_for_region(region_value, elem_type, elset_name):
                    f.write(f"*ELEMENT, TYPE={elem_type}, ELSET={elset_name}\n")
                    eid = 1
                    for conn, reg, keep in zip(c["tet_conn_full"], c["tet_regions"], c["write_mask"]):
                        if not keep:
                            continue
                        if int(reg) == int(region_value):
                            nodes = (np.asarray(conn, dtype=int) + 1).tolist()
                            f.write(f"{eid}, {', '.join(map(str, nodes))}\n")
                        eid += 1

                write_elements_for_region(bone_reg, self.bone_element_type, f"{part_name}_BONE")
                write_elements_for_region(cart_reg, self.cartilage_element_type, f"{part_name}_CARTILAGE")

                # Element sets are created directly by the ELSET parameter on
                # each *ELEMENT block above.
                for surf_name, sdef in self.surfaces.get(part_name, {}).items():
                    if sdef["type"] != "element":
                        raise ValueError(f"Unknown surface type '{sdef['type']}' for {part_name}:{surf_name}")
                    f.write(f"*SURFACE, NAME={surf_name}, TYPE=ELEMENT\n")
                    for eid, face in sdef["faces"]:
                        f.write(f"{eid}, {face}\n")

                f.write("*END PART\n")
                f.write("**\n")

            f.write(f"*ASSEMBLY, NAME={self.assembly_name}\n")

            for part_name, pinfo in self.parts.items():
                inst = pinfo["instance_name"]
                f.write(f"*INSTANCE, NAME={inst}, PART={part_name}\n")
                f.write("*END INSTANCE\n")
            f.write("**\n")

            for rp_set_name, rp in self.reference_points.items():
                xyz = rp["xyz"]
                f.write(f"*NODE, NSET={rp_set_name}\n")
                f.write(f"{rp['node_id']}, {xyz[0]}, {xyz[1]}, {xyz[2]}\n")

            f.write("*END ASSEMBLY\n")

        return output_inp
