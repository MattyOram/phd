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

def write_failure_log(log_dir, filename, subject, stdout, stderr, input_json, run_id, run_id_mesh, full_params_file):
    info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "full_params": full_params_file,
        "subject": subject,
        "input_json": str(input_json), 
        'run_id_mesh': run_id_mesh,
        "run_id": run_id,  
        "stdout": to_text(stdout),
        "stderr": to_text(stderr)
    }

    with open(log_dir / filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(info, ensure_ascii=False))
        f.write("\n")

def run_subprocess(args, timeout=60):

    param_path = args[2]
    log_dir = param_path.parent.parent.parent / 'reports'
    subject = args[3].name
    full_params_file = args[-1]

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

        if proc.returncode != 0:

            write_failure_log(
                log_dir,
                "errors.jsonl",
                subject=subject,
                stdout=stdout,
                stderr=stderr,
                input_json=param_path,
                run_id=run_id,
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
            run_id=run_id,
            run_id_mesh=run_id_mesh,
            full_params_file=full_params_file
        )
        return 'timeout'
    
def get_list(value):
    if isinstance(value, list): return value
    else: return [value]


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



def is_list_of_lists(x):
    return (
        isinstance(x, list)
        and len(x) > 0
        and all(isinstance(item, list) for item in x)
    )

always_list = ['poses', 'tpm_patch_params', 'mc1_patch_params']

def find_loops(d, path=()):
    loops = []

    if not isinstance(d, dict):
        return loops

    for key, value in d.items():
        new_path = path + (key,)

        if isinstance(value, dict):
            loops.extend(find_loops(value, new_path))

        if isinstance(value, tuple):
            print(value)
            pass

        elif isinstance(value, list):
            # poses is always a list or list of lists
            if key in always_list:
                if is_list_of_lists(value) and len(value) > 1:
                    loops.append((new_path, value))
            else:
                if len(value) > 1:
                    loops.append((new_path, value))

    return loops

def set_nested(d, path, value):
    for key in path[:-1]:
        d = d[key]
    d[path[-1]] = value
    
def unwrap_singles(d, parent_key=None):
    if isinstance(d, dict):
        return {k: unwrap_singles(v, k) for k, v in d.items()}

    elif isinstance(d, list):
        # poses=[[...]] -> poses=[...]
        if parent_key == 'poses' and is_list_of_lists(d) and len(d) == 1:
            return unwrap_singles(d[0])

        # poses=[x] should stay a list
        if parent_key == 'poses':
            return [unwrap_singles(v) for v in d]

        # any other single-item list -> scalar/item
        if len(d) == 1:
            return unwrap_singles(d[0])

        return [unwrap_singles(v) for v in d]

    else:
        return d

def expand_params(params):
    loops = find_loops(params)

    if not loops:
        param = copy.deepcopy(params)
        param = unwrap_singles(param)
        param["run_id"] = 0
        yield param
        return

    results = []

    def recurse(i, param, chosen):
        if i == len(loops):
            out = unwrap_singles(copy.deepcopy(param))
            out["_loop"] = chosen.copy()
            results.append(out)
            return

        path, values = loops[i]
        for value in values:
            set_nested(param, path, value)
            chosen[".".join(path)] = value
            recurse(i + 1, param, chosen)
            chosen.pop(".".join(path), None)

    recurse(0, copy.deepcopy(params), {})

    for i, param in enumerate(results):
        param["run_id"] = i
        yield param


def write_param_files(params, output_dir):

    last_run_id = -1
    for param in expand_params(params):
        out_path = output_dir / f"{param['run_id']:02}.json"
        with open(out_path, "w") as f:
            json.dump(param, f, indent=2)
        last_run_id = param["run_id"]

    return last_run_id



InpPipeline_root = PROJECT_ROOT / 'WorkPackages/TMCJ-Contact/Computational/inpPipeline'

# LOAD PARAMETERS #
print('\nUpdating parameters.json')
param_path = InpPipeline_root / 'set_parameters/parameters.json'
params = load_parameters(param_path)

# -------- GENERAL PARAMETERS ---------------------------- #
params_gen = params['general']
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

# generate json param files for each combination of parameters -clear any from previous runs``
loop_param_dir = param_dir / 'loop_params'
loop_param_dir.mkdir(parents=True, exist_ok=True)
for p in loop_param_dir.iterdir():
    if p.is_file():
        Path.unlink(p)
run_count = write_param_files(params['inp'], loop_param_dir)


# input meshes root dir
mesh_root = Path(params_gen['mesh_root']) / 'meshes'
mesh_glob = '**/mesh*.vtu'

# subjects
subs_sides = params_gen['subjects']
if subs_sides is not None:
    subs = get_list(subs_sides)
else:
    subs = get_subs(mesh_root)

#subs = [x for x in subs if x == '14548R']
for sub in subs: 
    print(f"\nSUBJECT: {sub}")
    sub_path = mesh_root / sub
    mesh_paths_tpm = list(sub_path.glob('tpm-mc1/3Dmesh/mesh*.vtu'))
    mesh_paths_mc1 = [Path(str(x).replace('tpm-mc1', 'mc1-tpm')) for x in mesh_paths_tpm]
    
    for tpm_path, mc1_path in zip(mesh_paths_tpm, mesh_paths_mc1):
        run_id_mesh = extract_mesh_run_id(tpm_path) #str: 0-0-0
        print(f"\tMESH: {run_id_mesh}")
        

        for run_id in range(run_count+1):
            run_id = f'{run_id:02}'
            print(f"\t\tRUN ID: {run_id}")
            t0 = time.perf_counter()

            param_path = loop_param_dir / f'{run_id}.json'
            args = [
                InpPipeline_root / 'steps/main_inp.py',
                root_dir,
                param_path,
                sub_path,
                tpm_path,
                mc1_path,
                run_id,
                run_id_mesh,
                full_param_path.name
            ]
            ok = run_subprocess(args, timeout=timeout)

            dt = time.perf_counter() - t0
            print(f"\t\t\tRuntime: {dt:.3f}s - {ok}")



    




