# export_field_csv.py
# Usage:
#   abaqus python export_field_csv.py <odbPath> <stepName> <frameIndex> <instanceName> <fieldName> <outCsv> [position]
#
# Exports ONE chosen field from one frame for one instance, writing raw FieldValue rows (no averaging).
# If [position] is omitted, it will try: NODAL, INTEGRATION_POINT, ELEMENT_NODAL, ELEMENT_CENTROID
# If [position] is provided, it must be one of:
#   NODAL | INTEGRATION_POINT | ELEMENT_NODAL | ELEMENT_CENTROID

from __future__ import print_function
import sys, csv
from odbAccess import openOdb
from abaqusConstants import NODAL, INTEGRATION_POINT, ELEMENT_NODAL, ELEMENT_CENTROID

POS_MAP = {
    "NODAL": NODAL,
    "INTEGRATION_POINT": INTEGRATION_POINT,
    "ELEMENT_NODAL": ELEMENT_NODAL,
    "ELEMENT_CENTROID": ELEMENT_CENTROID,
}

DEFAULT_POSITIONS = [NODAL, INTEGRATION_POINT, ELEMENT_NODAL, ELEMENT_CENTROID]

def to_list(data):
    try:
        return list(data)
    except TypeError:
        return [float(data)]

def position_name(pos):
    for k, v in POS_MAP.items():
        if v == pos:
            return k
    return str(pos) 

def odbField2csv(odb_path, step_idx_s, frame_idx_s, inst_name, field_name, out_csv, positions=DEFAULT_POSITIONS):
    step_idx = int(step_idx_s)
    frame_idx = int(frame_idx_s)

    odb = openOdb(odb_path, readOnly=True)

    step_name = list(odb.steps.keys())[step_idx]
    frame = odb.steps[step_name].frames[frame_idx]

    asm = odb.rootAssembly
    if inst_name not in asm.instances:
        odb.close()
        raise RuntimeError(f"Instance '{inst_name}' not found. Available: {', '.join(sorted(asm.instances.keys()))}")

    inst = asm.instances[inst_name]


    # Nodal coords (deformed/current) lookup: nodeLabel -> (x,y,z)
    node_coord_map = {}
    if "COORD" in frame.fieldOutputs:
        try:
            ncoord_fld = frame.fieldOutputs["COORD"].getSubset(region=inst, position=NODAL)
            for nv in ncoord_fld.values:
                c = to_list(nv.data)
                if len(c) == 2:
                    c = [c[0], c[1], 0.0]
                node_coord_map[nv.nodeLabel] = c[:3]
        except:
            node_coord_map = {}

    # IP coords
    coord_map = {}
    if "COORD" in frame.fieldOutputs:
        try:
            coord_fld = frame.fieldOutputs["COORD"].getSubset(region=inst, position=INTEGRATION_POINT)
            for cv in coord_fld.values:
                c = to_list(cv.data)
                if len(c) == 2:
                    c = [c[0], c[1], 0.0]
                coord_map[(cv.elementLabel, cv.integrationPoint)] = c[:3]
        except:
            coord_map = {}


    if field_name not in frame.fieldOutputs:
        odb.close()
        raise RuntimeError(f"Field '{field_name}' not found in this frame. Available: {', '.join(sorted(frame.fieldOutputs.keys()))}")


    fld = frame.fieldOutputs[field_name]

    with open(out_csv, "w", newline='') as f:
        w = csv.writer(f)
        w.writerow([
            "odb", "step", "frameIndex", "stepTime", "instance",
            "field", "position",
            "nodeLabel", "elementLabel", "integrationPoint",
            "x", "y", "z",
            "ipX", "ipY", "ipZ",
            "componentCount", "components..."
        ])


        wrote_any = False 

        for pos in positions:
            try:
                sub = fld.getSubset(region=inst, position=pos)
            except:
                continue

            vals = sub.values
            if not vals:
                continue

            wrote_any = True
            pos_name = position_name(pos)

            for v in vals:
                ipx = ipy = ipz = ""
                if pos == INTEGRATION_POINT:
                    key = (getattr(v, "elementLabel", ""), getattr(v, "integrationPoint", ""))
                    c = coord_map.get(key, None)
                    if c is not None and len(c) >= 3:
                        ipx, ipy, ipz = c[0], c[1], c[2]

                x = y = z = ""
                if pos == NODAL:
                    c = node_coord_map.get(getattr(v, "nodeLabel", ""), None)
                    if c is not None and len(c) >= 3:
                        x, y, z = c[0], c[1], c[2]

                comps = to_list(v.data)

                w.writerow([
                    odb_path, step_name, frame_idx, getattr(frame, "frameValue", ""),
                    inst_name, field_name, pos_name,
                    getattr(v, "nodeLabel", ""),
                    getattr(v, "elementLabel", ""),
                    getattr(v, "integrationPoint", ""),
                    x, y, z,
                    ipx, ipy, ipz,
                    len(comps)
                ] + comps)

    odb.close()

    if not wrote_any:
        raise RuntimeError("No values exported. The field exists, but none were found for the requested position(s) in this instance/frame.")

    print("Wrote:", out_csv)

if __name__ == "__main__":
    if len(sys.argv) not in (7, 8):
        print("Usage: abaqus python odbField2csv.py <odbPath> <stepIndex> <frameIndex> <instanceName> <fieldName> <outCsv> [position]")
        sys.exit(2)
    # Optional position argument
    if len(sys.argv) == 8:
        pos_str = sys.argv[7].upper()
        if pos_str not in POS_MAP:
            raise RuntimeError("Unknown position '%s'. Use one of: %s"
                               % (pos_str, ", ".join(sorted(POS_MAP.keys()))))
        positions = [POS_MAP[pos_str]]
    else:
        positions = DEFAULT_POSITIONS

    odb_path, step_idx_s, frame_idx_s, inst_name, field_name, out_csv = sys.argv[1:7]
    odbField2csv(odb_path, step_idx_s, frame_idx_s, inst_name, field_name, out_csv, positions)
