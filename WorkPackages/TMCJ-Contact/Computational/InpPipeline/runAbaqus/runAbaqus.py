import subprocess
import os
import shutil
import tempfile
import sys

args = sys.argv
if len(args) == 1:
    sub = '14548R'
    pose = 'neutral'
    run_id = '0'
    run_id_mesh = '0-0-0'
    inp_root = 'outputs/testing/test2' # output_root of InpPipeline parameters
elif len(args) == 6:
    sub = args[1]
    pose = args[2]
    run_id = args[3]
    run_id_mesh = args[4]
    inp_root = args[5]
else:
    raise ValueError(
        'Pass all 5 args. Or 0 args and set them in the script: '
        'sub pose run_id run_id_mesh inp_root'
    )



job_name = f'{run_id_mesh}-{pose}-{run_id}'
relative_path = os.path.join(inp_root, 'inpFiles', sub, 'inp', job_name)
inp_file = job_name + '.inp'


inp_dir = os.path.abspath(relative_path)
src_inp_path = os.path.join(inp_dir, inp_file)

file_dir = os.path.dirname(os.path.abspath(__file__))
postprocess_file = os.path.join(file_dir, 'AbaqusPostProcessing', 'main_odb2csv.py')

env = os.environ.copy()

if not os.path.isfile(src_inp_path):
    raise FileNotFoundError(f'Input file not found: {src_inp_path}')

documents_dir = os.path.join(os.path.expanduser('~'), 'Documents')

abaqus_cmd = r"C:\SIMULIA\Commands\abaqus.BAT"

with tempfile.TemporaryDirectory(prefix=f'abaqus_{job_name}_', dir=documents_dir) as tmp_dir:
    # Copy input file into temp working directory
    tmp_inp_path = os.path.join(tmp_dir, inp_file)
    shutil.copy2(src_inp_path, tmp_inp_path)



    # ------ RUN ABAQUS ------------------------------------------------ #
    cmd = [
        "cmd",
        "/c",
        abaqus_cmd,
        f"job={job_name}",
        f'input={inp_file}',   # relative to cwd=tmp_dir
        'interactive',
        'ask_delete=OFF',
        'cpus=8'
    ]
    subprocess.run(cmd, cwd=tmp_dir, env=env, check=True)



    # ------ POSTPROCESS ------------------------------------------------ #
    tmp_odb_path = os.path.join(tmp_dir, job_name + '.odb')

    cmd = [
        "cmd",
        "/c",
        abaqus_cmd,
        "python",
        postprocess_file,
        tmp_odb_path
    ]
    subprocess.run(cmd, cwd=tmp_dir, env=env, check=True)



    # ------ COPY RESULTS BACK ------------------------------------------ #
    for name in os.listdir(tmp_dir):
        src = os.path.join(tmp_dir, name)
        dst = os.path.join(inp_dir, name)

        if os.path.isfile(src):
            shutil.copy2(src, dst)
        elif os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    # temp dir deleted automatically