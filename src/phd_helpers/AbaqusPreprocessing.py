import numpy as np
import gdist
from scipy.spatial.distance import cdist
from pathlib import Path
import re

import pyvista as pv
from collections import OrderedDict

from phd_helpers.paths import get_boundary, get_intercepts, get_intercepts_multi
from phd_helpers.MeshQuality import sample_surface

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
            "minimum_memory_required_mb": 354,
            "memory_to_minimize_io_mb": 3380,
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
        raise ValueError("Could not find Abaqus MEMORY ESTIMATE table in .dat file")

    return {
        "process": int(m.group(1)),
        "minimum_memory_required_gb": int(m.group(2))/1e3,
        "memory_to_minimize_io_gb": int(m.group(3))/1e3,
    }



# •••••••••••••••••••••••••••••••••••• INPUT FILE BUILDER •••••••••••••••••••••••••••••••••••• #

class AbaqusInpBuilder:
    TET_FACES = {
        "S1": (0, 1, 2),
        "S2": (0, 3, 1),
        "S3": (1, 3, 2),
        "S4": (2, 3, 0),
    }

    ELEMENT_TYPE_MAP = {
        'C3D4': 10,
        'C3D10': 24,
        'C3D4M': 10,
        'C3D10M': 24
    }

    def __init__(self):
        # set abaqus elements types e.g. C3D4
        self.bone_element_type = None
        self.cartilage_element_type = None

        # cell type ids in pyvista e.g. 24 (tet10)
        self.tri_celltype = None 
        self.tet_celltype = None

        # region ids + which tet regions are written (controls Abaqus element-id numbering)
        self.region = {"bone": 1, "cartilage": 2, "cartilage_outer_tri": -2}
        self.write_tet_regions = [self.region["bone"], self.region["cartilage"]]

        # model stores
        self.materials = {}         # mat_name -> dict(model, props, density)
        self.parts = {}             # part_name -> dict(mesh, instance_name, element cache...)
        self.elsets = {}            # part_name -> dict(elset_name -> list[int])
        self.nsets = {}             # part_name -> dict(nset_name  -> list[int])
        self.solid_sections = {}    # part_name -> list[dict(elset, material)]
        self.surfaces = {}          # part_name -> dict(surface_name -> dict(type, ...))

        # assembly
        self.assembly_name = "ASSEMBLY"
        self.reference_points = {}  # rp_set_name -> dict(node_id, xyz)
        self.couplings = []         # list[dict(constraint_name, rp_set_name, surface_ref, coupling_type)]

        # contact
        self.contacts = {}         # dict(interaction_name, friction, pairs)

        # steps
        self.steps = OrderedDict()        # {step_name:{step_name, nlgeom, step params, ...}, ...}

    # ----------------------------
    # helpers
    # ----------------------------
    @staticmethod
    def _write_id_list(f, ids, per_line=16):
        ids = list(map(int, ids))
        for i in range(0, len(ids), per_line):
            chunk = ids[i : i + per_line]
            f.write(", ".join(map(str, chunk)) + "\n")

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

    def _ensure_part_buckets(self, part_name: str):
        self.elsets.setdefault(part_name, {})
        self.nsets.setdefault(part_name, {})
        self.solid_sections.setdefault(part_name, [])
        self.surfaces.setdefault(part_name, {})

    # This is a lazy check - assumes bone and cartilage are same pyvista celltypes
    # - the error check also only checks bone element types and assumes it is one of the two in ELEMENT_TYPE_MAP
    def _determine_elem_types(self, mesh):
        self.tri_celltype, self.tet_celltype = sorted(list(mesh.cells_dict.keys()))
        if self.ELEMENT_TYPE_MAP[self.bone_element_type] != self.tet_celltype:
            raise ValueError(f"Abaqus element type ({self.bone_element_type}) does not match pyvista celltype ({self.tet_celltype})")

    def _require_elem_cache(self, part_name: str):
        if "_elem_cache" not in self.parts[part_name]:
            self._preprocess_part_elements(part_name)

    # ----------------------------
    # element preprocessing + ELSET from region
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
    def set_region_ids(self, bone_region_id=1, cartilage_region_id=2, cartilage_outer_tri_region_id=-2):
        self.region["bone"] = int(bone_region_id)
        self.region["cartilage"] = int(cartilage_region_id)
        self.region["cartilage_outer_tri"] = int(cartilage_outer_tri_region_id)
        self.write_tet_regions = [self.region["bone"], self.region["cartilage"]]

    def set_written_tet_regions(self, region_ids):
        self.write_tet_regions = [int(r) for r in region_ids]

    def set_assembly_name(self, name: str):
        self.assembly_name = str(name)

    def set_element_types(self, bone_element_type, cartilage_element_type):
        self.bone_element_type = bone_element_type
        self.cartilage_element_type = cartilage_element_type

    # ----------------------------
    # store-only creation methods
    # ----------------------------
    def create_material(self, name: str, material_model: str, material_props: dict, density=None):
        self.materials[name] = {
            "model": material_model,
            "props": dict(material_props),
            "density": None if density is None else float(density),
        }

    def add_part_from_vtu(self, part_name: str, part_mesh: str|pv.UnstructuredGrid, mode: str ='path', instance_name: str|None = None):
        if mode=='path': mesh = pv.read(part_mesh)
        elif mode=='mesh': mesh = part_mesh.copy(deep=True)
        self.parts[part_name] = {
            "mesh": mesh,
            "instance_name": instance_name or f"{part_name}_INST",
        }
        self._determine_elem_types(mesh)
        self._ensure_part_buckets(part_name)
        self._preprocess_part_elements(part_name)


    def create_reference_point(self, rp_set_name: str, node_id: int, xyz):
        self.reference_points[rp_set_name] = {
            "node_id": int(node_id),
            "xyz": np.asarray(xyz, dtype=float),
        }

    def create_nset(self, part_name: str, nset_name: str, node_ids_abaqus_1based):
        self._ensure_part_buckets(part_name)
        self.nsets[part_name][nset_name] = list(map(int, node_ids_abaqus_1based))

    def create_surface_from_nset(self, part_name: str, surface_name: str, nset_name: str):
        self._ensure_part_buckets(part_name)
        self.surfaces[part_name][surface_name] = {"type": "node", "nset": nset_name}

    def create_solid_section(self, part_name: str, elset_name: str, material_name: str):
        self._ensure_part_buckets(part_name)
        self.solid_sections[part_name].append({"elset": elset_name, "material": material_name})

    def create_surface_from_element_faces(self, part_name: str, surface_name: str, face_pairs):
        self._ensure_part_buckets(part_name)
        self.surfaces[part_name][surface_name] = {
            "type": "element",
            "faces": [(int(e), str(face)) for (e, face) in face_pairs],
        }

    def create_elset_from_region(self, part_name: str, region_id: int, elset_name: str):
        self._ensure_part_buckets(part_name)
        self._require_elem_cache(part_name)
        c = self.parts[part_name]["_elem_cache"]

        region_id = int(region_id)
        eids = []
        eid_local = 1
        for reg, keep in zip(c["tet_regions"], c["write_mask"]):
            if not keep:
                continue
            if int(reg) == region_id:
                eids.append(eid_local)
            eid_local += 1

        self.elsets[part_name][elset_name] = eids

    # ----------------------------
    # patch nodes + cartilage surface
    # ----------------------------
    def add_surface_from_nodes(
        self,
        part_name: str,
        patch_nodes,
        nset_name: str,
        surface_name: str,
        patch_nodes_are_1based: bool = False,
    ):
        nodes = np.asarray(patch_nodes, dtype=int)
        if not patch_nodes_are_1based:
            nodes = nodes + 1
        self.create_nset(part_name, nset_name, nodes.tolist())
        self.create_surface_from_nset(part_name, surface_name, nset_name)

    #def add_surface_from_region_id(self, part_name: str, tri_region_id: int, surface_name: str):
    #    self._require_elem_cache(part_name)
    #    c = self.parts[part_name]["_elem_cache"]
    #    abaqus_faces = self._map_tris2faces(
    #        c["tet_full_ids"], c["tet_conn_full"],
    #        c["tri_conn_full"], c["tri_regions"],
    #        int(tri_region_id),
    #        c["fullcell_to_eid"],
    #        part_name,
    #    )
    #    self.create_surface_from_element_faces(part_name, surface_name, sorted(abaqus_faces))

    def add_surface_from_cell_data(self, part_name: str, cell_data_name: str, value, surface_name: str):
        self._require_elem_cache(part_name)
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
    # assembly methods
    # ----------------------------
    def add_rp_surface_coupling(
        self,
        constraint_name: str,
        rp_set_name: str,
        part_name: str,
        patch_surface_name: str,
        coupling_type: str = "KINEMATIC",
    ):
        """
        coupling_type: e.g. "KINEMATIC" (written as '*KINEMATIC' next line)
        """
        inst = self.parts[part_name]["instance_name"]
        self.couplings.append(
            {
                "constraint_name": str(constraint_name),
                "rp_set_name": str(rp_set_name),
                "surface_ref": f"{inst}.{patch_surface_name}",
                "coupling_type": str(coupling_type).upper(),
            }
        )

    # ----------------------------
    # contact methods
    # ----------------------------
    def set_contact(self, interaction_name, surfaces, friction=0.0):
        """
        pairs: list of (part_a, surf_a, part_b, surf_b)
        """
        self.contacts[interaction_name] = {
            "interaction_name": str(interaction_name),
            "friction": float(friction),
            "pairs": list(surfaces or []),
        }

    # ----------------------------
    # step, boundary, outputs methods
    # ----------------------------
    def create_step(
        self,
        step_name: str,
        step_type: str,
        initial_increment_size: float,
        total_step_size: float,
        min_increment_size: float,
        max_increment_size: float,
        nlgeom: str = "YES",
        convert_sdi: str = "NO",
        unsymm: str = "YES",
        extrapolation: str | None = None,
        stabilize: bool = False,
        stabilize_factor: float | None = None,
        allsdtol: float | None = None,
    ):
        self.steps[step_name] = {
            "step_name": str(step_name),
            "nlgeom": str(nlgeom),
            "convert_sdi": str(convert_sdi),
            "unsymm": str(unsymm),
            "extrapolation": None if extrapolation is None else str(extrapolation).upper(),
            "stabilize": bool(stabilize),
            "stabilize_factor": stabilize_factor,
            "allsdtol": allsdtol,
            "step_type": str(step_type).upper(),
            "step_params": f"{initial_increment_size}, {total_step_size}, {min_increment_size}, {max_increment_size}",
            "step_blocks": {
                "step_blocks": [],
                "control_blocks": [],
                "bc_blocks": [],
                "history_blocks": [],
                "field_blocks": [],
                "step_control_blocks": [],
            },
        }

    def set_bc(self, step_name, node_set, op='MOD', U1=None, U2=None, U3=None, UR1=None, UR2=None, UR3=None):
        dofs = [U1, U2, U3, UR1, UR2, UR3]
        dofs = [(i, x) for i, x in enumerate(dofs, 1) if x is not None]

        bc_lines = []
        if len(dofs) and not len(self.steps[step_name]['step_blocks']["bc_blocks"]):
            bc_lines.append(f"*BOUNDARY, OP={op}")

        for i, dof in dofs:
            bc_lines.append(f'{node_set}, {i}, {i}, {float(dof)}')

        self.add_bc_lines(step_name, bc_lines)

    # Add raw blocks of text lines - list of text lines - no trailing \n
    def add_step_lines(self, step_name, lines):
        self.steps[step_name]['step_blocks']["step_blocks"].append({"lines": [str(x) for x in lines]})

    def add_control_lines(self, step_name, lines):
        self.steps[step_name]['step_blocks']["control_blocks"].append({"lines": [str(x) for x in lines]})

    def add_bc_lines(self, step_name, lines):
        self.steps[step_name]['step_blocks']["bc_blocks"].append({"lines": [str(x) for x in lines]})

    def add_history_output_lines(self, step_name, lines):
        self.steps[step_name]['step_blocks']["history_blocks"].append({"lines": [str(x) for x in lines]})

    def add_field_output_lines(self, step_name, lines):
        self.steps[step_name]['step_blocks']["field_blocks"].append({"lines": [str(x) for x in lines]})

    def add_step_control_lines(self, step_name, lines):
        self.steps[step_name]['step_blocks']["step_control_blocks"].append({"lines": [str(x) for x in lines]})

    # ----------------------------
    # writer (only place that writes)
    # ----------------------------
    def write_input_file(self, output_inp: str):
        if not self.parts:
            raise ValueError("No parts defined.")

        bone_reg = self.region["bone"]
        cart_reg = self.region["cartilage"]

        with open(output_inp, "w") as f:
            # ==================================================
            # MATERIALS
            # ==================================================
            f.write("**\n** Model Definition\n**\n")
            for mat_name, mat in self.materials.items():
                f.write(f"*MATERIAL, NAME={mat_name}\n")
                model = mat["model"].lower()

                if model == "elastic":
                    f.write("*ELASTIC\n")
                    f.write(f"{float(mat['props']['E'])}, {float(mat['props']['nu'])}\n")
                elif model in ("neo_hooke", "neo_hookean", "neohooke"):
                    f.write("*HYPERELASTIC, NEO HOOKE\n")
                    f.write(f"{float(mat['props']['C10'])}, {float(mat['props']['D1'])}\n")
                else:
                    raise ValueError(f"Unknown material_model '{mat['model']}' for '{mat_name}'")

                if mat["density"] is not None:
                    f.write("*DENSITY\n")
                    f.write(f"{float(mat['density'])}\n")

                f.write("**\n")

            # ==================================================
            # PARTS
            # ==================================================
            for part_name, pinfo in self.parts.items():
                mesh = pinfo["mesh"]
                f.write(f"*PART, NAME={part_name}\n")

                # NODES
                f.write("*NODE\n")
                for nid, xyz in enumerate(mesh.points, start=1):
                    f.write(f"{nid}, {xyz[0]}, {xyz[1]}, {xyz[2]}\n")

                # ELEMENT CONNECTIVITY (only if elem cache exists)
                if "_elem_cache" in pinfo:
                    c = pinfo["_elem_cache"]
                    tet_conn_full = c["tet_conn_full"]
                    tet_regions = c["tet_regions"]
                    write_mask = c["write_mask"]

                    def write_elements_for_region(region_value, elem_type, elset_name):
                        f.write(f"*ELEMENT, TYPE={elem_type}, ELSET={elset_name}\n")
                        eid_local = 1
                        for conn, reg, keep in zip(tet_conn_full, tet_regions, write_mask):
                            if not keep:
                                continue
                            if int(reg) == int(region_value):
                                conn_abaqus = (np.asarray(conn, dtype=int) + 1).tolist()
                                f.write(f"{eid_local}, {', '.join(map(str, conn_abaqus))}\n")
                            eid_local += 1

                    # keep your original intent: write bone and cartilage separately
                    write_elements_for_region(bone_reg, self.bone_element_type, f"{part_name}_BONE")
                    write_elements_for_region(cart_reg, self.cartilage_element_type, f"{part_name}_CARTILAGE")

                # ELSETs
                for elset_name, ids in self.elsets.get(part_name, {}).items():
                    f.write(f"*ELSET, ELSET={elset_name}\n")
                    self._write_id_list(f, ids)

                # SOLID SECTIONs
                for ss in self.solid_sections.get(part_name, []):
                    f.write(f"*SOLID SECTION, ELSET={ss['elset']}, MATERIAL={ss['material']}\n")
                    f.write("**\n")

                # NSETs
                for nset_name, ids in self.nsets.get(part_name, {}).items():
                    f.write(f"*NSET, NSET={nset_name}\n")
                    self._write_id_list(f, ids)

                # SURFACEs
                for surf_name, sdef in self.surfaces.get(part_name, {}).items():
                    if sdef["type"] == "node":
                        f.write(f"*SURFACE, TYPE=NODE, NAME={surf_name}\n")
                        f.write(f"{sdef['nset']}\n")
                        f.write("**\n")
                    elif sdef["type"] == "element":
                        f.write(f"*SURFACE, NAME={surf_name}, TYPE=ELEMENT\n")
                        for eid, face in sdef["faces"]:
                            f.write(f"{eid}, {face}\n")
                    else:
                        raise ValueError(f"Unknown surface type '{sdef['type']}' for {part_name}:{surf_name}")

                f.write("*END PART\n")
                f.write("**\n")

            # ==================================================
            # ASSEMBLY + RPs + COUPLING
            # ==================================================
            f.write("**\n")
            f.write(f"*ASSEMBLY, NAME={self.assembly_name}\n")

            for part_name, pinfo in self.parts.items():
                inst = pinfo["instance_name"]
                f.write(f"*INSTANCE, NAME={inst}, PART={part_name}\n")
                f.write("*END INSTANCE\n")
            f.write("**\n")

            f.write("** Reference points\n")
            for rp_set_name, rp in self.reference_points.items():
                xyz = rp["xyz"]
                f.write(f"*NODE, NSET={rp_set_name}\n")
                f.write(f"{rp['node_id']}, {xyz[0]}, {xyz[1]}, {xyz[2]}\n")
            f.write("**\n")

            f.write("** Couplings\n")
            for c in self.couplings:
                f.write(
                    f"*COUPLING, CONSTRAINT NAME={c['constraint_name']}, "
                    f"REF NODE={c['rp_set_name']}, SURFACE={c['surface_ref']}\n"
                )
                # coupling type line
                f.write(f"*{c['coupling_type']}\n")
            f.write("**\n")

            f.write("*END ASSEMBLY\n")
            f.write("**\n")

            # ==================================================
            # CONTACT
            # ==================================================
            for contact in self.contacts.values():
                f.write("**\n** Contact definition\n**\n")
                f.write(f"*SURFACE INTERACTION, NAME={contact['interaction_name']}\n")
                f.write("*FRICTION\n")
                f.write(f"{contact['friction']}\n")
                f.write("**\n")

                f.write("*CONTACT\n")
                f.write("*CONTACT INCLUSIONS\n")
                for (pa, sa, pb, sb) in contact["pairs"]:
                    inst_a = self.parts[pa]["instance_name"]
                    inst_b = self.parts[pb]["instance_name"]
                    f.write(f"{inst_a}.{sa}, {inst_b}.{sb}\n")

                f.write("*CONTACT PROPERTY ASSIGNMENT\n")
                for (pa, sa, pb, sb) in contact["pairs"]:
                    inst_a = self.parts[pa]["instance_name"]
                    inst_b = self.parts[pb]["instance_name"]
                    f.write(f"{inst_a}.{sa}, {inst_b}.{sb}, {contact['interaction_name']}\n")
                f.write("**\n")

            # ==================================================
            # STEP + BCs + OUTPUTS
            # ==================================================

            # step blocks
            for step in self.steps.values():
                step_line = (
                    f"*STEP, NAME={step['step_name']}, NLGEOM={step['nlgeom']}, "
                    f"CONVERT SDI={step['convert_sdi']}, UNSYMM={step['unsymm']}"
                )
                if step.get("extrapolation"):
                    step_line += f", EXTRAPOLATION={step['extrapolation']}"
                f.write(step_line + "\n")

                if step["step_type"] == "STATIC" and step.get("stabilize", False):
                    static_line = "*STATIC"

                    if step.get("stabilize_factor") is not None:
                        static_line += f", STABILIZE={step['stabilize_factor']}"
                    else:
                        static_line += ", STABILIZE"

                    if step.get("allsdtol") is not None:
                        static_line += f", ALLSDTOL={step['allsdtol']}"

                    f.write(static_line + "\n")
                else:
                    f.write(f"*{step['step_type']}\n")

                f.write(step["step_params"].rstrip("\n") + "\n")
                f.write("**\n")

                for block in step['step_blocks']["step_blocks"]:
                    for line in block["lines"]:
                        f.write(line.rstrip("\n") + "\n")
                    f.write("**\n")

                # control blocks
                for block in step['step_blocks']["control_blocks"]:
                    for line in block["lines"]:
                        f.write(line.rstrip("\n") + "\n")
                    f.write("**\n")

                # boundary blocks
                for block in step['step_blocks']["bc_blocks"]:
                    for line in block["lines"]:
                        f.write(line.rstrip("\n") + "\n")
                    f.write("**\n")

                # history output blocks
                for block in step['step_blocks']["history_blocks"]:
                    for line in block["lines"]:
                        f.write(line.rstrip("\n") + "\n")
                    f.write("**\n")

                # field output blocks
                for block in step['step_blocks']["field_blocks"]:
                    for line in block["lines"]:
                        f.write(line.rstrip("\n") + "\n")
                    f.write("**\n")

                # step control blocks
                for block in step['step_blocks']["step_control_blocks"]:
                    for line in block["lines"]:
                        f.write(line.rstrip("\n") + "\n")
                    f.write("**\n")

                f.write("*END STEP\n")
                f.write("**\n")

        return output_inp