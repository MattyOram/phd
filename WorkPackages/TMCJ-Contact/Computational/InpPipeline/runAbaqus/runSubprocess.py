import subprocess
import os


sub = '50000R'
pose = 'neutral'
run_id = '0'
run_id_mesh = '0-0-0'

job_name = f'{run_id_mesh}-{pose}-{run_id}'
inp_file = job_name + '.inp'   
out_dir = f'outputs/testing/SML-1/inpFiles/{sub}/inp/{job_name}' # relative to wherever this script is run from
print(out_dir)
# ------ RUN ABAQUS ------------------------------------------------ #

cmd = [
    'abaqus',
    f"job={job_name}",
    f'input={inp_file}', # relative to out_dir
    'interactive', # keep subprocess active for whole runtime, so python script doesn't continue
    "ask_delete=OFF",
]
subprocess.run(cmd, cwd=out_dir, check=True) # set current working directory as out_dir


# ------ POSTPROCESS ------------------------------------------------#

file_dir = os.path.dirname(os.path.abspath(__file__))
postprocess_file = os.path.join(file_dir , 'abaqusPostProcessing/main_odb2csv.py')
odb_path = f'{out_dir}/{job_name}.odb'

cmd = [
    'abaqus',
    'python',
    postprocess_file,
    odb_path
]
subprocess.run(cmd, check=True) 
