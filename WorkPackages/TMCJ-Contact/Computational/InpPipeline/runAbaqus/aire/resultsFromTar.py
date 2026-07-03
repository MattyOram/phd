from extractOutputs import extract_outputs
from pathlib import Path
import sys
import subprocess


# extract output files
tar_path = Path(sys.argv[1])
extract_outputs(tar_path)

out_dir = tar_path.parent / tar_path.name.removesuffix('.tar.gz')
odb_paths = [
    p for p in out_dir.glob("**/*.odb")
    if not p.name.startswith("upgraded-")
]

# upgrade odbs
abaqus_cmd = Path(r"C:\SIMULIA\Commands\abaqus.BAT")

upgraded_odb_paths = []    
for odb_path in odb_paths:
    odb_path = odb_path.resolve()

    upgrade_stem = f"upgraded-{odb_path.stem}"
    upgraded_odb_path = odb_path.parent / f"{upgrade_stem}.odb"

    cmd = [
        "cmd",
        "/c",
        str(abaqus_cmd),
        "-upgrade",
        "-job",
        upgrade_stem,
        "-odb",
        odb_path.stem,
    ]

    subprocess.run(cmd, check=True, cwd=odb_path.parent)

    if upgraded_odb_path.exists():
        odb_path.unlink()
        upgraded_odb_paths.append(upgraded_odb_path)
    else:
        raise FileNotFoundError("Upgrade problem")


# extract results from odbs
file_dir = Path(__file__).resolve().parent
postprocess_file = (file_dir / "../AbaqusPostProcessing/main_odb2csv.py").resolve()

for odb_path in upgraded_odb_paths:
    try:
        cmd = [
            "cmd",
            "/c",
            str(abaqus_cmd),
            "python",
            str(postprocess_file),
            str(odb_path.name),
        ]
        subprocess.run(cmd, check=True, cwd=odb_path.parent)
    except subprocess.CalledProcessError:
        print("Failed to postprocess", odb_path.name)