import subprocess
import os
import shutil
import tempfile


sub = '50034R'
pose = 'neutral'
run_id = '0'
run_id_mesh = '0-0-0'

job_name = f'{run_id_mesh}-{pose}-{run_id}'
relative_path = f'outputs/testing/SML-1/inpFiles/{sub}/inp/{job_name}'
inp_file = job_name + '.inp'


inp_dir = os.path.abspath(relative_path)
src_inp_path = os.path.join(inp_dir, inp_file)

file_dir = os.path.dirname(os.path.abspath(__file__))
postprocess_file = os.path.join(file_dir, 'abaqusPostProcessing', 'main_odb2csv.py')

env = os.environ.copy()

if not os.path.isfile(src_inp_path):
    raise FileNotFoundError(f'Input file not found: {src_inp_path}')

documents_dir = os.path.join(os.path.expanduser('~'), 'Documents')


with tempfile.TemporaryDirectory(prefix=f'abaqus_{job_name}_', dir=documents_dir) as tmp_dir:
    # Copy input file into temp working directory
    tmp_inp_path = os.path.join(tmp_dir, inp_file)
    shutil.copy2(src_inp_path, tmp_inp_path)


    # ------ RUN ABAQUS ------------------------------------------------ #
    cmd = [
        'abaqus',
        f'job={job_name}',
        f'input={inp_file}',   # relative to cwd=tmp_dir
        'interactive',
        'ask_delete=OFF',
        'cpus=8'
    ]
    subprocess.run(cmd, cwd=tmp_dir, env=env, check=True)



    # ------ POSTPROCESS ------------------------------------------------ #
    tmp_odb_path = os.path.join(tmp_dir, job_name + '.odb')

    cmd = [
        'abaqus',
        'python',
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