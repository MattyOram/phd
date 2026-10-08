from __future__ import print_function

import csv
import sys

from odbAccess import openOdb
from abaqusConstants import NODAL, INTEGRATION_POINT, ELEMENT_NODAL, ELEMENT_CENTROID

POS_MAP = {
    "NODAL": NODAL,
    "INTEGRATION_POINT": INTEGRATION_POINT,
    "ELEMENT_NODAL": ELEMENT_NODAL,
    "ELEMENT_CENTROID": ELEMENT_CENTROID,
}

DEFAULT_POSITIONS = [NODAL, INTEGRATION_POINT, ELEMENT_NODAL, ELEMENT_CENTROID]


def _open_csv(path):
    """Open a CSV correctly in Abaqus 2022/Python 2.7 and Python 3."""
    if sys.version_info[0] < 3:
        return open(path, "wb")
    return open(path, "w", newline="")


def to_list(data):
    try:
        return list(data)
    except TypeError:
        return [float(data)]


def position_name(pos):
    for key, value in POS_MAP.items():
        if value == pos:
            return key
    return str(pos)


def odbField2csv(odb_path, step_idx_s, frame_idx_s, inst_name,
                 field_name, out_csv, positions=DEFAULT_POSITIONS):
    step_idx = int(step_idx_s)
    frame_idx = int(frame_idx_s)

    odb = openOdb(odb_path, readOnly=True)
    try:
        step_name = list(odb.steps.keys())[step_idx]
        frame = odb.steps[step_name].frames[frame_idx]

        asm = odb.rootAssembly
        if inst_name not in asm.instances:
            raise RuntimeError(
                "Instance '{0}' not found. Available: {1}".format(
                    inst_name, ", ".join(sorted(asm.instances.keys()))
                )
            )

        inst = asm.instances[inst_name]

        # Nodal coordinates lookup: nodeLabel -> (x, y, z)
        node_coord_map = {}
        if "COORD" in frame.fieldOutputs:
            try:
                ncoord_fld = frame.fieldOutputs["COORD"].getSubset(
                    region=inst, position=NODAL
                )
                for nv in ncoord_fld.values:
                    coords = to_list(nv.data)
                    if len(coords) == 2:
                        coords = [coords[0], coords[1], 0.0]
                    node_coord_map[nv.nodeLabel] = coords[:3]
            except Exception:
                node_coord_map = {}

        # Integration-point coordinates lookup: (elementLabel, integrationPoint) -> xyz
        coord_map = {}
        if "COORD" in frame.fieldOutputs:
            try:
                coord_fld = frame.fieldOutputs["COORD"].getSubset(
                    region=inst, position=INTEGRATION_POINT
                )
                for cv in coord_fld.values:
                    coords = to_list(cv.data)
                    if len(coords) == 2:
                        coords = [coords[0], coords[1], 0.0]
                    coord_map[(cv.elementLabel, cv.integrationPoint)] = coords[:3]
            except Exception:
                coord_map = {}

        if field_name not in frame.fieldOutputs:
            raise RuntimeError(
                "Field '{0}' not found in this frame. Available: {1}".format(
                    field_name, ", ".join(sorted(frame.fieldOutputs.keys()))
                )
            )

        fld = frame.fieldOutputs[field_name]
        wrote_any = False

        csv_file = _open_csv(out_csv)
        try:
            writer = csv.writer(csv_file)
            writer.writerow([
                "odb", "step", "frameIndex", "stepTime", "instance",
                "field", "position",
                "nodeLabel", "elementLabel", "integrationPoint",
                "x", "y", "z",
                "ipX", "ipY", "ipZ",
                "componentCount", "components..."
            ])

            for pos in positions:
                try:
                    sub = fld.getSubset(region=inst, position=pos)
                except Exception:
                    continue

                vals = sub.values
                if not vals:
                    continue

                wrote_any = True
                pos_name = position_name(pos)

                for value in vals:
                    ipx = ipy = ipz = ""
                    if pos == INTEGRATION_POINT:
                        key = (
                            getattr(value, "elementLabel", ""),
                            getattr(value, "integrationPoint", "")
                        )
                        coords = coord_map.get(key)
                        if coords is not None and len(coords) >= 3:
                            ipx, ipy, ipz = coords[0], coords[1], coords[2]

                    x = y = z = ""
                    if pos == NODAL:
                        coords = node_coord_map.get(getattr(value, "nodeLabel", ""))
                        if coords is not None and len(coords) >= 3:
                            x, y, z = coords[0], coords[1], coords[2]

                    comps = to_list(value.data)
                    writer.writerow([
                        odb_path, step_name, frame_idx,
                        getattr(frame, "frameValue", ""),
                        inst_name, field_name, pos_name,
                        getattr(value, "nodeLabel", ""),
                        getattr(value, "elementLabel", ""),
                        getattr(value, "integrationPoint", ""),
                        x, y, z,
                        ipx, ipy, ipz,
                        len(comps)
                    ] + comps)
        finally:
            csv_file.close()

        if not wrote_any:
            raise RuntimeError(
                "No values exported. The field exists, but none were found "
                "for the requested position(s) in this instance/frame."
            )

        print("Wrote:", out_csv)
    finally:
        odb.close()


if __name__ == "__main__":
    if len(sys.argv) not in (7, 8):
        print(
            "Usage: abaqus python odbField2csv_aire.py <odbPath> <stepIndex> "
            "<frameIndex> <instanceName> <fieldName> <outCsv> [position]"
        )
        sys.exit(2)

    if len(sys.argv) == 8:
        pos_str = sys.argv[7].upper()
        if pos_str not in POS_MAP:
            raise RuntimeError(
                "Unknown position '{0}'. Use one of: {1}".format(
                    pos_str, ", ".join(sorted(POS_MAP.keys()))
                )
            )
        positions = [POS_MAP[pos_str]]
    else:
        positions = DEFAULT_POSITIONS

    args = sys.argv[1:7]
    odbField2csv(args[0], args[1], args[2], args[3], args[4], args[5], positions)
