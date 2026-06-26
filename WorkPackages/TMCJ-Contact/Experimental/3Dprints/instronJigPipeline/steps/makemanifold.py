# this takes the volumes outputted from the 3D meshing and ensures that the volume shells are manifold (not guaranteed)
# - will be watertight but it is easy to get edges the are "pinched" that are shared by 4 faces
# - also this might not be necessary ; watertight might be good enough for 3D printing software - haven't checked


# •••••••••••••••• Assumes the interface isn't the problem - maybe major floor, but difficult to do it more sophisticated


import numpy as np
import pyvista as pv
import pymeshfix
import gdist

from phd_helpers.paths import find_corresponding_cells, get_boundary


def make_manifold(shell, region_id, max_area, max_loc, name):
    """
    shell: polydata watertight mesh
    region_id: -1 if bone, -2 if cartilage
    max_area: max size of repair region
    max_loc: max distance of repair region from cartilage boundary (0 (boundary -> 1 (centre))
    require_manifold: throw error is result is not manifold
    """


    # check if already manifold
    manifold = shell.is_manifold
    if manifold:
        print(f'[[capture]]Mesh ({name}) is already manifold')
        return shell

    else:
        print(f'\nFIXING NON-MANIFOLD REGION ({region_id})...\n')

        shell['shell_point_id'] = np.arange(shell.n_points)
        shell['shell_cell_id'] = np.arange(shell.n_cells)
        surf = shell.extract_cells(np.where(shell['region_id']==region_id)[0]).extract_geometry()
        surf_faces = surf.faces.reshape(-1, 4)[:, 1:]


        # find non-manifold edges / points and remove the faces connected to those points
        bad_edges = shell.extract_feature_edges(manifold_edges=False,boundary_edges=False,feature_edges=False,non_manifold_edges=True)
        bad_points = np.arange(surf.n_points)[np.isin(
                                                            surf['shell_point_id'], 
                                                            bad_edges['shell_point_id'])]
        bad_faces = surf['shell_cell_id'][np.isin(surf_faces, bad_points).sum(axis=1) >= 1]
        shell_holes = shell.extract_cells(bad_faces, invert=True).extract_geometry()


        v = np.asarray(shell_holes.points, dtype=np.float64)

        f = np.asarray(shell_holes.faces.reshape(-1, 4)[:, 1:], dtype=np.int32)  # triangle faces
        v2, f2 = pymeshfix.clean_from_arrays(v, f)

        repaired = pv.PolyData(v2, np.hstack([np.full((len(f2), 1), 3), f2]).ravel())
        #print(f'Is manifold ({region_id}):', repaired.is_manifold)

        # check all surf cells that were not repaired are still there
        print("\nCheck manifold cells have not moved")
        orig_cells = find_corresponding_cells(repaired, shell_holes, raise_error=True)
        repaired['repaired'] = np.ones(repaired.n_cells)
        repaired['repaired'][orig_cells] = 0

        # check all interface cells are still there
        print("Check interface cells have not moved")
        inter_cells = find_corresponding_cells(repaired, shell.extract_cells(shell['region_id']==-3), raise_error=True)
        repaired['region_id'] = np.full(repaired.n_cells, region_id)
        repaired['region_id'][inter_cells] = -3


        # EVALUATE REPAIR SIZE AND LOCATION
        print('\nEvaluate repair size and proximity:')
        repaired_cartilage_surf = repaired.extract_cells(repaired['region_id']==-2).extract_geometry()
        repaired_cartilage_surf['repaired_cartilage_surf_id'] = np.arange(repaired_cartilage_surf.n_points)
        repaired_patch = repaired_cartilage_surf.extract_cells(repaired_cartilage_surf['repaired']==1).extract_geometry()

        # measure proximity to cartilage boundary
        repaired_cartilage_surf_edge = get_boundary(repaired_cartilage_surf)
        cartilage_edge_dists = gdist.compute_gdist(
            repaired_cartilage_surf.points.astype(np.float64),
            repaired_cartilage_surf.faces.reshape(-1, 4)[:, 1:].astype(np.int32),
            source_indices=repaired_cartilage_surf_edge['repaired_cartilage_surf_id'].astype(np.int32), 
        )
        dist_max = cartilage_edge_dists.max()
        repaired_dist_max = cartilage_edge_dists[repaired_patch['repaired_cartilage_surf_id']].max()
        repaired_proxmimity = repaired_dist_max / dist_max
        print(f'[[capture]] ({name}) Repair loc: {repaired_dist_max:.2f} mm   ({repaired_proxmimity:.4f} ; 0: boundary, 1:centre)')
        if max_loc and repaired_proxmimity > max_loc:
            raise AssertionError(f"{region_id} repair location too far from boundary")

        # measure size of repair region
        repaired_A = repaired_patch.area / repaired_cartilage_surf.area
        print(f'[[capture]] ({name}) Repair area: {repaired_patch.area:.2f} mm^2 ({repaired_A*100:.2f}%)')
        if max_area and repaired_A > max_area:
            raise AssertionError(f"{region_id} repair area is too large")

        print('\nComplete')

        print('\nCHECK RESULT...\n')
        # check if it is now manifold
        print(f'[[capture]]({name}) is manifold: {shell.is_manifold}')
        #if not shell.is_manifold:
            #raise KeyError(f"{region_id} is not manifold")

        print('\nComplete\n')

        return repaired