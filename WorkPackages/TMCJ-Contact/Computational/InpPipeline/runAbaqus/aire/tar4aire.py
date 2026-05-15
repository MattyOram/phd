from pathlib import Path
import tarfile
import sys

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

# tar .inp files and inpFiles.txt ready for transfer to Aire
output_tar = out_dir / "inpFiles.tar.gz"
with tarfile.open(output_tar, "w:gz") as tar:
    for inp, tar_name in zip(inps, tar_names):
        tar.add(inp, arcname=tar_name) # add .inp files
    tar.add(txt_file, arcname=txt_file.name) # add inpFiles.txt