import numpy as np
import pyvista as pv

def get_bad_face_cell_ids(mesh):
    """takes in mesh after full postprocessing and gets cell ids of any tet faces attached to cartilage that should be part of bone"""

    mesh['mesh_point_id'] = np.arange(mesh.n_points)
    mesh['mesh_cell_id'] = np.arange(mesh.n_cells)
    mesh_shell = mesh.extract_cells_by_type(5)
    mesh_shell['shell_cell_id'] = np.arange(mesh_shell.n_cells)

    bad_ids = []
    while True:
        mesh_cart_shell = mesh_shell.extract_cells(mesh_shell['region_id']==-1, invert=True)
        mesh_cart_surf = mesh_shell.extract_cells(mesh_shell['region_id']==-2).extract_surface(algorithm=None)
        # this gets edges between cart and bad cells if they connect by > 1 point (doesn't get 1 pointers)
        edges_nm = mesh_cart_shell.extract_feature_edges(non_manifold_edges=True, boundary_edges=True, feature_edges=False, manifold_edges=False)
        edges_surf = mesh_cart_surf.extract_feature_edges(non_manifold_edges=False, boundary_edges=True, feature_edges=False, manifold_edges=False)
        edges = edges_nm + edges_surf

        boundary_ids = edges['mesh_point_id'][edges.lines.reshape(-1, 3)[:, 1:]] # on surf
        bad_mask = np.isin(mesh_cart_surf['mesh_point_id'][mesh_cart_surf.regular_faces], boundary_ids).sum(axis=1) >= 3 # on surf

        if bad_mask.any():
            bad = np.where(bad_mask)[0]
            #print(len(bad))
            bad_ids.extend(mesh_cart_surf['mesh_cell_id'][bad])
            mesh_shell = mesh_shell.extract_cells(~np.isin(mesh_shell['shell_cell_id'], mesh_cart_surf['shell_cell_id'][bad]))
        else:
            break

    return bad_ids

def get_bad_tet_cell_mask(mesh, bad_face_cell_ids):
    """takes in mesh after full postprocessing and gets cell mask of any tets attached to cartilage that should be part of bone"""

    mask = np.zeros(mesh.n_cells, dtype=bool)
    #if len(bad_face_cell_ids) == 0:
    #    return mask

    bad_tris = np.array([mesh.get_cell(cid).point_ids for cid in bad_face_cell_ids]) # on mesh

    bad_tris_sorted = np.sort(bad_tris, axis=1)
    bad_tri_set = {tuple(row) for row in bad_tris_sorted}

    tet_cell_ids = np.where(mesh.celltypes == pv.CellType.TETRA)[0]
    tets = mesh.cells_dict[pv.CellType.TETRA]   # shape (n_tets, 4)

    tet_faces = np.vstack([
        tets[:, [0, 1, 2]],
        tets[:, [0, 1, 3]],
        tets[:, [0, 2, 3]],
        tets[:, [1, 2, 3]],
    ])
    tet_faces = np.sort(tet_faces, axis=1)

    face_matches = np.array([tuple(face) in bad_tri_set for face in tet_faces])
    tet_mask = face_matches.reshape(4, -1).any(axis=0)

    mask[tet_cell_ids] = tet_mask

    return mask

from phd_helpers.paths import find_shared_cells, identical_points_count

def rebuild_combined_mesh(mesh):
    """takes in mesh with reassigned edge case cells and outputs the final combined mesh"""

    tet = mesh.extract_cells_by_type(10)
    cartilage_tet = tet.extract_cells(tet['region_id']==2)
    bone_tet = tet.extract_cells(tet['region_id']==1)


    # extract volume shells for surface meshes
    cartilage_shell = cartilage_tet.extract_surface(algorithm=None) # cartilage shell
    bone_shell = bone_tet.extract_surface(algorithm=None) # bone shell
    #tet_shell = tet.extract_surface(algorithm=None) # tet shell

    # find interface surfaces on cartilage and bone
    interface_mask_bone = find_shared_cells(bone_shell, cartilage_shell)
    interface_mask_cartilage = find_shared_cells(cartilage_shell, bone_shell)
    interface_bone = bone_shell.extract_cells(interface_mask_bone)
    interface_cartilage = cartilage_shell.extract_cells(interface_mask_cartilage)
    # check they are identical
    n_shared = identical_points_count(interface_cartilage.points, interface_bone.points)
    if n_shared != interface_bone.n_points and n_shared == interface_cartilage.n_points:
        raise AssertionError('Interfaces are not identical')
    # assign bone interface to interface surf for consistency (normals point outwards relative to bone)
    interface_surf = interface_bone
    interface_surf['region_id'] = np.full(interface_surf.n_cells, -3)

    # extract bone and cartilage surfs
    bone_surf = bone_shell.extract_cells(~interface_mask_bone)
    bone_surf['region_id'] = np.full(bone_surf.n_cells, -1)
    cartilage_surf = cartilage_shell.extract_cells(~interface_mask_cartilage)
    cartilage_surf['region_id'] = np.full(cartilage_surf.n_cells, -2)

    # combined mesh of tets and tris with region ids - ••• Needs to be tets then tris for abaqus input script to work •••
    combined = tet + interface_surf + bone_surf + cartilage_surf
    return combined

def ad_hoc_mesh_fix(mesh):
    """takes in final postprocessed mesh and postprocesses again..."""
    # assign tets that were outside of input bone volume and attached to cartilage to bone (defo some edge cases that it still won't work for...)
    bad_ids = get_bad_face_cell_ids(mesh)
    if len(bad_ids) == 0:
        return mesh
    tet_mask = get_bad_tet_cell_mask(mesh, bad_ids)
    mesh['region_id'][(mesh['region_id']==2) & tet_mask] = 1
    return rebuild_combined_mesh(mesh)
    