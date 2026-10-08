from __future__ import print_function

import os
import sys

from odbAccess import openOdb
from abaqusConstants import NODAL

from odbField2csv_aire import odbField2csv
from odbHistory2csv_aire import odbHistory2csv


def _mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def get_step_idx_range(odb_path):
    odb = openOdb(odb_path, readOnly=True)
    try:
        n_steps = len(odb.steps)
    finally:
        odb.close()
    return range(n_steps)


def process_odb(odb_path, out_dir=None):

    odb_path = os.path.abspath(odb_path)

    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(odb_path), "resultCSVs")

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    step_list = get_step_idx_range(odb_path)

    frame_list = [-1]
    instance_list = ["TPM_INST", "MC1_INST"]

    field_list = [
        "CPRESS",
        "CSTATUS",
        "U",
    ]

    position_list = [
        NODAL,
        NODAL,
        NODAL,
    ]

    completed_steps = []

    for step_idx in step_list:

        print("Processing step {0}".format(step_idx))

        # Keep track of files created for this step only
        step_files = []

        try:

            # History
            out_csv_hist = os.path.join(
                out_dir,
                "history_step-{0}.csv".format(step_idx)
            )

            step_files.append(out_csv_hist)

            odbHistory2csv(
                odb_path,
                step_idx,
                out_csv_hist
            )

            # Fields
            for inst_name in instance_list:

                for frame_idx in frame_list:

                    for field_name, position in zip(
                        field_list,
                        position_list
                    ):

                        out_csv_field = os.path.join(
                            out_dir,
                            "{0}-{1}-{2}-{3}.csv".format(
                                inst_name,
                                field_name,
                                step_idx,
                                frame_idx
                            )
                        )

                        step_files.append(out_csv_field)

                        odbField2csv(
                            odb_path,
                            step_idx,
                            frame_idx,
                            inst_name,
                            field_name,
                            out_csv_field,
                            [position]
                        )

            completed_steps.append(step_idx)

        except Exception as exc:

            print(
                "Step {0} could not be extracted: {1}".format(
                    step_idx,
                    exc
                )
            )

            print(
                "Reached incomplete step. "
                "Stopping extraction for this ODB."
            )

            # Remove anything partially written for the failed step
            for path in step_files:
                if os.path.isfile(path):
                    os.remove(path)

            break

    if not completed_steps:
        raise RuntimeError(
            "No completed steps could be extracted from {0}".format(
                odb_path
            )
        )

    print(
        "Successfully extracted {0} completed step(s).".format(
            len(completed_steps)
        )
    )


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print(
            "Usage: abaqus python main_odb2csv_aire.py "
            "<odbPath> [outputDirectory]"
        )
        sys.exit(2)

    odb_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) == 3 else None
    process_odb(odb_path, output_dir)
