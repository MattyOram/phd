# export_all_history_csv.py
# Usage:
#   abaqus python export_all_history_csv.py <odbPath> <stepName> <outCsv>
#
# Exports ALL history outputs in the given step:
# - Every historyRegion
# - Every historyOutput in that region
# - Every (time, value) pair
#
# Output is "long/tidy" format: one row per data point.

from __future__ import print_function
import sys, csv
from odbAccess import openOdb

def odbHistory2csv(odb_path, step_idx, out_csv):

    odb = openOdb(odb_path, readOnly=True)

    step_idx = int(step_idx)
    step_name = list(odb.steps.keys())[step_idx]
    step = odb.steps[step_name]
    hrs = step.historyRegions

    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow([
            "odb", "step",
            "historyRegionKey", "historyRegionDescription",
            "historyOutputKey", "historyOutputDescription",
            "time", "value"
        ])

        # Deterministic order
        for region_key in sorted(hrs.keys()):
            region = hrs[region_key]
            region_desc = getattr(region, "description", "")

            for out_key in sorted(region.historyOutputs.keys()):
                ho = region.historyOutputs[out_key]
                out_desc = getattr(ho, "description", "")

                # ho.data is a sequence of (time, value). value can be float or tuple.
                for (t, val) in ho.data:
                    if isinstance(val, (tuple, list)):
                        # flatten vector/tensor history outputs into multiple columns
                        w.writerow([odb_path, step_name, region_key, region_desc, out_key, out_desc, t] + list(val))
                    else:
                        w.writerow([odb_path, step_name, region_key, region_desc, out_key, out_desc, t, val])

    odb.close()
    print("Wrote:", out_csv)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: abaqus python obdHistory2csv.py <odbPath> <stepIndex> <outCsv>")
        sys.exit(2)

    odb_path, step_idx, out_csv = sys.argv[1:]
    odbHistory2csv(odb_path, step_idx, out_csv)

