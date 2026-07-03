from pathlib import Path
import tarfile
import shutil
import sys

def extract_tar(tar_path, output_dir):
    tar_path = Path(tar_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, "r:*") as tar:
        tar.extractall(path=output_dir)


def extract_outputs(path):
    """extract tar file of results from aire, ran with AbaqusBatchTemplate"""

    savepath = path.parent
    extract_tar(path, savepath)

    dir_name = path.name.removesuffix('.tar.gz')
    outdir = savepath / dir_name

    run_dict = {} # job_name: run_id (for .out, .err)
    with open(outdir / 'inpFiles.txt', 'r') as f:
        for i, line in enumerate(f, start=1):
            run_dict[line.strip().removesuffix('.inp')] = i



    for job_name, run_id in run_dict.items():

        job_dir = outdir / job_name
        job_dir.mkdir(parents=True, exist_ok=True)

        all_files = (
            list(outdir.glob(f"{job_name}.*")) +
            list(outdir.glob(f"abaqus_*_{run_id}.*"))
        )

        for file in all_files:
            if file.is_file() and file.parent != job_dir:
                shutil.move(str(file), str(job_dir / file.name))

if __name__ == '__main__':
    path = Path(sys.argv[1])
    extract_outputs(path)