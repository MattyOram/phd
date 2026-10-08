from __future__ import print_function

import os
import shutil
import sys
import tarfile
import traceback

from main_odb2csv_aire import process_odb


def find_odb_files(output_dir):
    odb_paths = []
    result_root = os.path.abspath(os.path.join(output_dir, "resultCSVs"))

    for root, dirs, files in os.walk(output_dir):
        root_abs = os.path.abspath(root)

        # Never descend into generated result directories.
        dirs[:] = [
            d for d in dirs
            if os.path.abspath(os.path.join(root, d)) != result_root
        ]

        for filename in files:
            if filename.lower().endswith(".odb"):
                odb_paths.append(os.path.join(root_abs, filename))

    return sorted(odb_paths)


def safe_rmtree(path):
    if os.path.isdir(path):
        shutil.rmtree(path)


def make_results_tar(output_dir, successful_jobs):
    tar_path = os.path.join(output_dir, "resultCSVs.tar.gz")
    result_root = os.path.join(output_dir, "resultCSVs")

    if os.path.exists(tar_path):
        os.remove(tar_path)

    tar = tarfile.open(tar_path, "w:gz")
    try:
        for job_name in successful_jobs:
            job_dir = os.path.join(result_root, job_name)
            tar.add(
                job_dir,
                arcname=os.path.join("resultCSVs", job_name)
            )
    finally:
        tar.close()

    return tar_path


def process_output_folder(output_dir):
    output_dir = os.path.abspath(output_dir)
    if not os.path.isdir(output_dir):
        raise RuntimeError("Output directory does not exist: {0}".format(output_dir))

    odb_paths = find_odb_files(output_dir)
    if not odb_paths:
        raise RuntimeError("No .odb files found under: {0}".format(output_dir))

    result_root = os.path.join(output_dir, "resultCSVs")
    if not os.path.isdir(result_root):
        os.makedirs(result_root)

    print("Found {0} ODB file(s).".format(len(odb_paths)))

    successful_jobs = []
    failed_jobs = []

    for index, odb_path in enumerate(odb_paths, start=1):
        job_name = os.path.splitext(os.path.basename(odb_path))[0]
        job_out_dir = os.path.join(result_root, job_name)

        # Remove only this job's previously generated CSV directory so stale
        # files cannot be mixed into a new extraction.
        safe_rmtree(job_out_dir)
        os.makedirs(job_out_dir)

        print("[{0}/{1}] Processing {2}".format(index, len(odb_paths), odb_path))
        try:
            process_odb(odb_path, job_out_dir)
            successful_jobs.append(job_name)
        except Exception as exc:
            failed_jobs.append((job_name, str(exc)))
            safe_rmtree(job_out_dir)
            print("FAILED: {0}".format(job_name))
            traceback.print_exc()

    if not successful_jobs:
        raise RuntimeError("All ODB extractions failed; no tar file was created.")

    tar_path = make_results_tar(output_dir, successful_jobs)

    print("")
    print("Successfully processed {0}/{1} ODB file(s).".format(
        len(successful_jobs), len(odb_paths)
    ))
    print("Results archive: {0}".format(tar_path))

    if failed_jobs:
        print("")
        print("The following ODBs failed and were NOT included in the tar file:")
        for job_name, error in failed_jobs:
            print("  {0}: {1}".format(job_name, error))
        return 1

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: abaqus python resultsFromAire.py <simulation_output_folder>")
        sys.exit(2)

    sys.exit(process_output_folder(sys.argv[1]))
