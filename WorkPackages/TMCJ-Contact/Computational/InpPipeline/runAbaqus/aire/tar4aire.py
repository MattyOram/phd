from pathlib import Path
import tarfile
import sys
import re
import shutil


def create_job_script(filepath: str, savepath: str, n: int) -> None:
    """
    create copy of slurm array script and fill N value
    """
    src = Path(filepath)
    dst = Path(savepath)

    shutil.copyfile(src, dst)

    text = dst.read_text()

    text = re.sub(
        r"^(#SBATCH\s+--array=1-)N(%1\s*)$",
        rf"\g<1>{n}\2",
        text,
        flags=re.MULTILINE,
    )

    dst.write_text(text)

path = Path(sys.argv[1]) # path to out_dir for .inp files
out_dir = path / 'aire/input'
out_dir.mkdir(parents=True, exist_ok=True)

inps = sorted(list(path.glob('**/*.inp')))
tar_names = [x.parents[2].name + '-' + x.name for x in inps] # append sub to filename 

# create text file of .inp file names for Aire slurm array
txt_file = out_dir / "inpFiles.txt"
txt_file.write_text(
    "\n".join(tar_names),
    encoding="utf-8"
)

# create .sh file of slurm array Aire job script
template_path = Path(__file__).resolve().parent / "AbaqusBatchTemplate.sh"
job_file = out_dir / "runAbaqus.sh"
create_job_script(
    filepath=template_path,
    savepath=job_file,
    n=len(tar_names),
)

# tar .inp files and inpFiles.txt ready for transfer to Aire
output_tar = out_dir / "inpFiles.tar.gz"
with tarfile.open(output_tar, "w:gz") as tar:
    for inp, tar_name in zip(inps, tar_names):
        tar.add(inp, arcname=tar_name) # add .inp files
    tar.add(txt_file, arcname=txt_file.name) # add inpFiles.txt
    tar.add(job_file, arcname=job_file.name) # add rubAbaqus.sh