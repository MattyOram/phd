import json
import subprocess
from datetime import datetime
import os
import signal

def to_text(x):
    if x is None:
        return ""
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    return str(x)

def write_failure_log(log_dir, filename, subject, bones, stdout, stderr, step, input_json, run_ids, full_params_file):
    info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "full_params": full_params_file,
        "subject": subject,
        "bones": bones,
        "step": step,
        "input_json": str(input_json), 
        "run_ids": run_ids,  
        "stdout": to_text(stdout),
        "stderr": to_text(stderr)
    }

    with open(log_dir / filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(info, ensure_ascii=False))
        f.write("\n")

def run_subprocess(args, timeout=60):
    out_dir = args[6] # Path
    subject = str(out_dir.parent.name)
    bones = str(out_dir.name)
    log_dir = args[0] # Path
    step = str(args[5].parent.name)

    full_params_file = str(args[1])
    args_str = [str(a) for a in args]
    # args_str starts at 1 so index from there
    input_json = args_str[3]
    run_ids = args_str[5:] 

    proc = subprocess.Popen(
        args_str,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )

    try:
        stdout, stderr = proc.communicate(timeout=timeout)

        if proc.returncode != 0:
            stderr_text = to_text(stderr)

            # check if an input mesh from the previous step exists
            previous_step_check = "FileNotFoundError: No input mesh"
            if previous_step_check in stderr_text:
                return 'no input'

        return 'ok'

    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        stdout, stderr = proc.communicate()

        write_failure_log(
            log_dir,
            "timeouts.jsonl",
            subject=subject,
            bones=bones,
            stdout=stdout,
            stderr=stderr,
            step=step,
            input_json=input_json,
            run_ids=run_ids,
            full_params_file=full_params_file
        )
        return 'timeout'
    
def get_list(value):
    if isinstance(value, list): return value
    else: return [value]


def load_parameters(param_path):
    # run parameters.py to update parameters.json with any changes in parameters.py, then load parameters.json
    result = subprocess.run(
        ["python", param_path.with_suffix(".py"), param_path],
        capture_output=True, 
        text=True
        )
    with open(param_path, "r") as f:
        params = json.load(f)
    return params

def write_full_params_copy(param_dir):
    i = 0 # if file alread exists, do not overwrite, save with -i suffix
    while (full_param_path := param_dir / f"full_params{'' if i == 0 else f'-{i}'}.json").exists():
        i += 1
    with open(full_param_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f'Full parameter file saved to {full_param_path}')
    
def get_subs(mesh_root):
    mesh_path = mesh_root / 'meshes'
    return [x.name for x in mesh_path.iterdir() if x.is_dir()]

from pathlib import Path

from phd_helpers.paths import get_project_root

MeshPipeline_root = get_project_root() / 'WorkPackages/TMCJ-Contact/Computational/InputFilePipeline'

# LOAD PARAMETERS #
print('\nUpdating parameters.json')
param_path = MeshPipeline_root / 'set_parameters/parameters.json'
params = load_parameters(param_path)

# -------- GENERAL PARAMETERS ---------------------------- #
params_glob = params['general']
params_pre = params['preprocessing']

# create output dir
root_dir = Path(params_glob["output_root"])
root_dir.mkdir(parents=True, exist_ok=True)

# create param dir
param_dir = root_dir / "params"
param_dir.mkdir(parents=True, exist_ok=True)

# create log dir
log_dir = root_dir / 'reports'
log_dir.mkdir(parents=True, exist_ok=True)

# save copy of full parameters in root directory 
write_full_params_copy(param_dir)

# input meshes root dir
mesh_root = Path(params_glob['mesh_root'])
mesh_glob = '**/mesh*.vtu'

# subject selection
subs_sides = params_pre['subjects']
if subs_sides is not None:
    subs = get_list(subs_sides)
else:
    subs = get_subs(mesh_root)





for sub in subs: 
    



