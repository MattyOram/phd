import json
import subprocess
from datetime import datetime
import os
import signal
import copy
from pathlib import Path
import time
from phd_helpers.paths import PROJECT_ROOT

def to_text(x):
    if x is None:
        return ""
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    return str(x)

def write_failure_log(log_dir, filename, subject, stdout, stderr, input_json, run_id_mesh, full_params_file):
    info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "full_params": full_params_file,
        "subject": subject,
        "input_json": str(input_json), 
        'run_id_mesh': run_id_mesh, 
        "stdout": to_text(stdout),
        "stderr": to_text(stderr)
    }

    with open(log_dir / filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(info, ensure_ascii=False))
        f.write("\n")

def write_captured_lines(log_dir, subject, stdout, full_params_file):
    out_file = log_dir / "manifold.jsonl"

    info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "full_params": full_params_file,
        "subject": subject,
        "outputs": {}
    }
    i = 1
    for line in stdout.splitlines():
        if line.startswith("[[capture]]"):
            info['outputs'][f'line{i}'] = line.removeprefix("[[capture]]").lstrip()
            i+=1

    with open(out_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(info, ensure_ascii=False))
        f.write("\n")

def run_subprocess(args, timeout=60):

    param_path = args[2]
    log_dir = param_path.parent.parent / 'reports'
    subject = args[3].name
    full_params_file = param_path.name

    args_str = ['python', '-u'] + [str(a) for a in args]
    proc = subprocess.Popen(
        args_str,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )

    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        write_captured_lines(log_dir, subject, stdout, full_params_file)

        if proc.returncode != 0:

            write_failure_log(
                log_dir,
                "errors.jsonl",
                subject=subject,
                stdout=stdout,
                stderr=stderr,
                input_json=param_path,
                run_id_mesh=run_id_mesh,
                full_params_file=full_params_file
            )
            return 'error'

        return 'ok'

    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        stdout, stderr = proc.communicate()

        write_failure_log(
            log_dir,
            "timeouts.jsonl",
            subject=subject,
            stdout=stdout,
            stderr=stderr,
            input_json=param_path,
            run_id_mesh=run_id_mesh,
            full_params_file=full_params_file
        )
        return 'timeout'


def load_parameters(param_path):
    # run parameters.py to update parameters.json with any changes in parameters.py, then load parameters.json
    subprocess.run(
        ["python", param_path.with_suffix(".py"), param_path],
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
    return full_param_path
    
def get_subs(mesh_root):
    return [x.name for x in mesh_root.iterdir() if x.is_dir()]

def extract_mesh_run_id(mesh_path: Path):
    return mesh_path.name.removeprefix("mesh-").removesuffix(".vtu")

def get_list(value):
    if isinstance(value, list): return value
    else: return [value]








JigPipeline_root = PROJECT_ROOT / 'WorkPackages/TMCJ-Contact/Experimental/3Dprints/instronJigPipeline'

# LOAD PARAMETERS #
print('\nUpdating parameters.json')
param_path = JigPipeline_root / 'set_parameters/parameters.json'
params = load_parameters(param_path)

# -------- GENERAL PARAMETERS ---------------------------- #
params_gen = params['global']
timeout = params_gen['timeout']

# create output dir
root_dir = Path(params_gen["output_root"])
root_dir.mkdir(parents=True, exist_ok=True)

# create param dir
param_dir = root_dir / "params"
param_dir.mkdir(parents=True, exist_ok=True)

# create log dir
log_dir = root_dir / 'reports'
log_dir.mkdir(parents=True, exist_ok=True)

# save copy of full parameters in root directory 
full_param_path = write_full_params_copy(param_dir)


# input meshes root dir
mesh_root = Path(params_gen['mesh_root']) / 'meshes'
mesh_glob = '**/mesh*.vtu'

# subjects
subs_sides = params_gen['subjects']
if subs_sides is not None:
    subs = get_list(subs_sides)
else:
    subs = get_subs(mesh_root)

mesh_ids = params_gen['mesh_ids']
if mesh_ids is not None:
    mesh_ids = get_list(mesh_ids)

print(f"\n POSES: {params['printJig']['poses']}")

run = True
for sub in subs: 
    sub_path = mesh_root / sub
    mesh_paths_tpm = list(sub_path.glob('tpm-mc1/3Dmesh/mesh*.vtu'))
    mesh_paths_mc1 = [Path(str(x).replace('tpm-mc1', 'mc1-tpm')) for x in mesh_paths_tpm]

    # dumb check that's repeated in next loop so that SUBJECT isn't printed out if none of their meshes are in id list
    if mesh_ids is not None:
        mesh_run_ids = [extract_mesh_run_id(tpm_path) for tpm_path in mesh_paths_tpm]
        if len([x for x in mesh_run_ids if x in mesh_ids]):
            print(f"\nSUBJECT: {sub}")

    
    for tpm_path, mc1_path in zip(mesh_paths_tpm, mesh_paths_mc1):
        run_id_mesh = extract_mesh_run_id(tpm_path) #str: 0-0-0
        if mesh_ids is not None:
            if run_id_mesh not in mesh_ids:
                run = False

        if run:
            print(f"\tMESH: {run_id_mesh}")
            t0 = time.perf_counter()

            args = [
                JigPipeline_root / f'steps/main_boneJig.py',
                root_dir,
                full_param_path,
                sub_path,
                tpm_path,
                mc1_path,
                run_id_mesh
            ]
            ok = run_subprocess(args, timeout=timeout)

            dt = time.perf_counter() - t0
            print(f"\t\t\tRuntime: {dt:.3f}s - {ok}")



    




