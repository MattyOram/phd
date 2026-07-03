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
    upgrade_stem = f"upgraded-{odb_path.stem}"
    upgrade_path = odb_path.parent / upgrade_stem
    upgraded_odb_path = upgrade_path.with_suffix(".odb")

    cmd = [
        "cmd",
        "/c",
        str(abaqus_cmd),
        "-upgrade",
        "-job",
        str(upgrade_path),
        "-odb",
        str(odb_path.with_suffix('')),
    ]

    subprocess.run(cmd, check=True)

    # delete original ODB if upgrade worked
    if upgraded_odb_path.exists():
        odb_path.unlink()
        upgraded_odb_paths.append(upgraded_odb_path)
    else:
        raise FileNotFoundError(f"Upgrade problem")


# extract results from odbs
file_dir = Path(__file__).resolve().parent
postprocess_file = file_dir / "../AbaqusPostProcessing/main_odb2csv.py"

for odb_path in upgraded_odb_paths:
    try:
        cmd = [
            "cmd",
            "/c",
            str(abaqus_cmd),
            "python",
            str(postprocess_file),
            str(odb_path),
        ]
        subprocess.run(cmd, check=True)
    except:
        print("Failed to postprocess", odb_path.name)