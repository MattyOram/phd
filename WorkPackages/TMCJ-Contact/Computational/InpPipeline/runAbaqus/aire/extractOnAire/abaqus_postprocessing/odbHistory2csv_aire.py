from __future__ import print_function

import csv
import sys

from odbAccess import openOdb


def _open_csv(path):
    if sys.version_info[0] < 3:
        return open(path, "wb")
    return open(path, "w", newline="")


def odbHistory2csv(odb_path, step_idx, out_csv):
    odb = openOdb(odb_path, readOnly=True)
    try:
        step_idx = int(step_idx)
        step_name = list(odb.steps.keys())[step_idx]
        step = odb.steps[step_name]
        history_regions = step.historyRegions

        csv_file = _open_csv(out_csv)
        try:
            writer = csv.writer(csv_file)
            writer.writerow([
                "odb", "step",
                "historyRegionKey", "historyRegionDescription",
                "historyOutputKey", "historyOutputDescription",
                "time", "value"
            ])

            for region_key in sorted(history_regions.keys()):
                region = history_regions[region_key]
                region_desc = getattr(region, "description", "")

                for out_key in sorted(region.historyOutputs.keys()):
                    history_output = region.historyOutputs[out_key]
                    out_desc = getattr(history_output, "description", "")

                    for time_value, value in history_output.data:
                        if isinstance(value, (tuple, list)):
                            writer.writerow([
                                odb_path, step_name,
                                region_key, region_desc,
                                out_key, out_desc,
                                time_value
                            ] + list(value))
                        else:
                            writer.writerow([
                                odb_path, step_name,
                                region_key, region_desc,
                                out_key, out_desc,
                                time_value, value
                            ])
        finally:
            csv_file.close()

        print("Wrote:", out_csv)
    finally:
        odb.close()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: abaqus python odbHistory2csv_aire.py "
            "<odbPath> <stepIndex> <outCsv>"
        )
        sys.exit(2)

    odbHistory2csv(sys.argv[1], sys.argv[2], sys.argv[3])
