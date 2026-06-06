from odbField2csv import odbField2csv
from odbHistory2csv import odbHistory2csv
import os
import sys
from abaqusConstants import NODAL, INTEGRATION_POINT, ELEMENT_NODAL, ELEMENT_CENTROID
from odbAccess import openOdb

def get_step_idx_range(odb_path):
    odb = openOdb(odb_path, readOnly=True)
    try:
        n_steps = len(odb.steps)
    finally:
        odb.close()

    return range(n_steps)

#odb_path = "testing/test8V2/test.odb"
odb_path = sys.argv[1]

out_dir = os.path.join(os.path.dirname(odb_path), "resultCSVs")
os.makedirs(out_dir, exist_ok=True)

step_list = None # None to export all, list like [0, 2, -1] to export specific steps
frame_list = [-1]
instance_list = ["TPM_INST", "MC1_INST"]

# Get step idxs if step_list == None
if step_list is None:
    step_list = get_step_idx_range(odb_path)

field_list = [
    #"CNAREA",
    "CPRESS",
    #"CSTATUS",
    "U",
    #"VF",
    #"S",
    #"LE"
]
position_list = [
    #NODAL,
    NODAL,
    #NODAL,
    NODAL,
    #NODAL,
    #INTEGRATION_POINT,
    #INTEGRATION_POINT
]

for step_idx_s in step_list:

    out_csv_hist = os.path.join(out_dir, f"history_step-{step_idx_s}.csv")
    odbHistory2csv(odb_path, step_idx_s, out_csv_hist)

    for inst_name in instance_list:
        for frame_idx_s in frame_list:
            for field_name, position in zip(field_list, position_list):

                out_csv_field = os.path.join(out_dir, f"{inst_name}-{field_name}-{step_idx_s}-{frame_idx_s}.csv")

                positions = [position]

                odbField2csv(odb_path, step_idx_s, frame_idx_s, inst_name, field_name, out_csv_field, positions) 