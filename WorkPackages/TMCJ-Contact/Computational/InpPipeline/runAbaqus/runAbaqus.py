import subprocess
import os
import shutil
import tempfile
import sys
from pathlib import Path

datacheck = False # just run pre.exe run datacheck 

args = sys.argv
if len(args) == 1:
    sub = "14548R"
    pose = "neutral"
    run_id = "0"
    run_id_mesh = "0-0-0"
    inp_root = "outputs/testing/test2"  # output_root of InpPipeline parameters
elif len(args) == 6:
    sub = args[1]
    pose = args[2]
    run_id = args[3]
    run_id_mesh = args[4]
    inp_root = args[5]
else:
    raise ValueError(
        "Pass all 5 args. Or 0 args and set them in the script: "
        "sub pose run_id run_id_mesh inp_root"
    )

job_name = f"{run_id_mesh}-{pose}-{run_id}"
inp_root = Path(inp_root)

inp_dir = (inp_root / "inpFiles" / sub / "inp" / job_name).resolve()
inp_file = f"{job_name}.inp"
src_inp_path = inp_dir / inp_file

file_dir = Path(__file__).resolve().parent
postprocess_file = file_dir / "AbaqusPostProcessing" / "main_odb2csv.py"

#env = os.environ.copy()

if not src_inp_path.is_file():
    raise FileNotFoundError(f"Input file not found: {src_inp_path}")

documents_dir = Path.home() / "Documents"
abaqus_cmd = Path(r"C:\SIMULIA\Commands\abaqus.BAT")

with tempfile.TemporaryDirectory(prefix=f"abaqus_{job_name}_", dir=documents_dir) as tmp_dir_str:
    tmp_dir = Path(tmp_dir_str)

    # Copy input file into temp working directory
    tmp_inp_path = tmp_dir / inp_file
    shutil.copy2(src_inp_path, tmp_inp_path)


    # ------ RUN ABAQUS ------------------------------------------------ #
    cmd = [
        "cmd",
        "/c",
        str(abaqus_cmd),
        f"job={job_name}",
        f"input={inp_file}",   # relative to cwd=tmp_dir
        "interactive",
        "ask_delete=OFF",
        "cpus=8",
        'memory="28gb"'
    ]
    if datacheck:
        cmd = [
            "cmd",
            "/c",
            str(abaqus_cmd),
            f"job={job_name}",
            f"input={inp_file}",   # relative to cwd=tmp_dir
            "datacheck",
            "interactive"
        ]
    subprocess.run(cmd, cwd=tmp_dir, check=True)


    # ------ POSTPROCESS ------------------------------------------------ #
    if not datacheck:
        try:
            tmp_odb_path = tmp_dir / f"{job_name}.odb"

            cmd = [
                "cmd",
                "/c",
                str(abaqus_cmd),
                "python",
                str(postprocess_file),
                str(tmp_odb_path),
            ]
            subprocess.run(cmd, cwd=tmp_dir, check=True)
        except:
            print("Failed to postprocess")

    # ------ COPY RESULTS BACK ------------------------------------------ #
    for src in tmp_dir.iterdir():
        dst = inp_dir / src.name

        if src.is_file():
            shutil.copy2(src, dst)
        elif src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)

    # temp dir deleted automatically