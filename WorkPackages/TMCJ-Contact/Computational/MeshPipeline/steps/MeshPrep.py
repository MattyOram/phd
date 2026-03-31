from pathlib import Path
import json
import subprocess
import copy
from datetime import datetime
import os
import signal

import copy
import itertools
import json

def get_run_id(argvs):
    run_id = ''
    for arg in argvs:
        run_id += f"-{arg}"
    return run_id

# Map conditional optimisation blocks to controlling flags 
# - so that if False it doesn't loop over disabled params
OPTIM_STEP_FLAGS = {
    ("cgal_params", "odt"): ("cgal_params", "optimisation", "odt"),
    ("cgal_params", "lloyd"): ("cgal_params", "optimisation", "lloyd"),
    ("cgal_params", "perturb"): ("cgal_params", "optimisation", "perturb"),
    ("cgal_params", "exude"): ("cgal_params", "optimisation", "exude"),
}


def is_list_of_lists(x):
    return (
        isinstance(x, list)
        and len(x) > 0
        and all(isinstance(item, list) for item in x)
    )


def find_loops(d, path=()):
    loops = []

    if not isinstance(d, dict):
        return loops

    for key, value in d.items():
        new_path = path + (key,)

        if isinstance(value, dict):
            loops.extend(find_loops(value, new_path))

        elif isinstance(value, list):
            # poses is always a list or list of lists
            if key == "poses":
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


def get_nested(d, path):
    for key in path:
        d = d[key]
    return d


def unwrap_singles(d, parent_key=None):
    if isinstance(d, dict):
        return {k: unwrap_singles(v, k) for k, v in d.items()}

    elif isinstance(d, list):
        # poses=[[...]] -> poses=[...]
        if parent_key == "poses" and is_list_of_lists(d) and len(d) == 1:
            return unwrap_singles(d[0])

        # any single-item list -> scalar/item
        if len(d) == 1:
            return unwrap_singles(d[0])

        return [unwrap_singles(v) for v in d]

    else:
        return d


def first_unwrapped(value, key=None):
    if key == "poses" and is_list_of_lists(value):
        if len(value) == 1:
            return value[0]
        return value

    if isinstance(value, list) and len(value) == 1:
        return first_unwrapped(value[0])

    return value


def conditional_flag_path(path):
    """
    Return the controlling optimisation flag path for this parameter path,
    or None if the parameter is not conditionally controlled.
    """
    for block_path, flag_path in OPTIM_STEP_FLAGS.items():
        if path[:len(block_path)] == block_path:
            return flag_path
    return None


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
        key = path[-1]

        flag_path = conditional_flag_path(path)

        if flag_path is not None:
            flag_value = get_nested(param, flag_path)

            if flag_value is False:
                # Step disabled: do not sweep this parameter block
                set_nested(param, path, first_unwrapped(values[0], key))
                recurse(i + 1, param, chosen)
                return

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
    output_dir.mkdir(parents=True, exist_ok=True)

    last_run_id = -1
    for param in expand_params(params):
        out_path = output_dir / f"{param['run_id']}.json"
        with open(out_path, "w") as f:
            json.dump(param, f, indent=2)
        last_run_id = param["run_id"]

    return last_run_id


def to_text(x):
    if x is None:
        return ""
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    return str(x)

def write_runtime_log(log_dir, filename, runtime, subject, bones, step, input_json, run_ids, full_params_file):
    info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "full_params": full_params_file,
        "subject": subject,
        "bones": bones,
        "step": step,
        "input_json": str(input_json), 
        "run_ids": run_ids,  
        "runtime": runtime
    }

    with open(log_dir / filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(info, ensure_ascii=False))
        f.write("\n")

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

def write_captured_lines(log_dir, subject, bones, step, input_json, run_ids, stdout, full_params_file):
    out_file = log_dir / "info_3Dmesh.jsonl"

    info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "full_params": full_params_file,
        "subject": subject,
        "bones": bones,
        "step": step,
        "input_json": str(input_json), 
        "run_ids": run_ids, 
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



def run_subprocess(args, timeout=300):
    out_dir = args[6] # Path
    subject = str(out_dir.parent.name)
    bones = str(out_dir.name)
    log_dir = args[0] # Path
    step = str(args[5].parent.name)

    full_params_file = str(args[1])
    args_str = [str(a) for a in args][2:] # exclude full_params_file from args passed to subprocess
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
        if step == '3Dmesh':
            write_captured_lines(log_dir, subject, bones, step, input_json, run_ids, stdout, full_params_file)

        if proc.returncode != 0:
            stderr_text = to_text(stderr)

            # check if an input mesh from the previous step exists
            previous_step_check = "FileNotFoundError: No input mesh"
            if previous_step_check in stderr_text:
                return 'no input'

            if step == 'cartilage':
                cartilage_height_check = "AssertionError: Not all cartilage points above bone surface"
                if cartilage_height_check in stderr_text:
                    write_failure_log(
                        log_dir,
                        "interference.jsonl",
                        subject=subject,
                        bones=bones,
                        stdout='',
                        stderr=cartilage_height_check,
                        step=step,
                        input_json=input_json,
                        run_ids=run_ids,
                        full_params_file=full_params_file
                    )
                    return "interference"


            write_failure_log(
                log_dir,
                "errors.jsonl",
                subject=subject,
                bones=bones,
                stdout=stdout,
                stderr=stderr,
                step=step,
                input_json=input_json,
                run_ids=run_ids,
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
            bones=bones,
            stdout=stdout,
            stderr=stderr,
            step=step,
            input_json=input_json,
            run_ids=run_ids,
            full_params_file=full_params_file
        )
        return 'timeout'

#def run_subprocess(args):
#    result = subprocess.run(args, capture_output=True, text=True)
#    print(result.stdout)
#    print(result.stderr)
#    #print("returncode:", result.returncode)
#    result.check_returncode()  # raise after printing


def get_list(value):
    if isinstance(value, list): return value
    else: return [value]