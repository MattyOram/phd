"""

root_dir
|_full_params.json
|_params
    |_step1params
    |_step2params
    |_...
|_meshes
    |_subjectsideL
        |_bone-arbone
            |_2Dmesh
            |_3Dmesh

"""

from pathlib import Path
import json
import shutil
import subprocess
import itertools
import time

import itertools
import json

from steps.MeshPrep import get_list, write_param_files, run_subprocess, write_runtime_log

from phd_helpers.paths import PROJECT_ROOT


MeshPipeline_root = PROJECT_ROOT / 'WorkPackages/TMCJ-Contact/Computational/MeshPipeline'

# LOAD PARAMETERS #
print('\nUpdating parameters.json')

# run parameters.py to update parameters.json with any changes in parameters.py, then load parameters.json
param_path = MeshPipeline_root / 'set_parameters/parameters.json'
result = subprocess.run(
    ["python", param_path.with_suffix(".py"), param_path],
    capture_output=True, 
    text=True
    )
print(result.stdout)
print(result.stderr)
result.check_returncode()  # raise after printing
with open(param_path, "r") as f:
    params = json.load(f)

# global params
params_glob = params['global']

step_names = ['2Dmesh', 'cartilage', '3Dmesh', 'manifold'] # in order
steps = []
for step_name in step_names:
    if params_glob['steps'][step_name]:
        steps.append(step_name)


# subject selection
params_sub = params['subjects']
subject_sideLs = get_list(params_sub['subject_sideL'])
bone_pairs = get_list(params_sub['bone_arbone'])

# create root dir
root_dir = Path(params_glob["output_root"])
root_dir.mkdir(parents=True, exist_ok=True)

# create step param dir and clear contents if exists - except for full params files from previous runs
step_param_dir = root_dir / "params"
step_param_dir.mkdir(parents=True, exist_ok=True)

for p in step_param_dir.iterdir():
    if p.is_dir():
        shutil.rmtree(p)

# save copy of full parameters in root directory 
i = 0 # if file alread exists, do not overwrite, save with -i suffix
while (full_param_path := step_param_dir / f"full_params{'' if i == 0 else f'-{i}'}.json").exists():
    i += 1
with open(full_param_path, "w") as f:
    json.dump(params, f, indent=2)
print(f'Full parameter file saved to {full_param_path}')

# GENERATE .JSON PARAMETER FILES FOR EACH STEP #
step_counts = []
for step in steps:
    if not params_glob['allow_overwrite']:
        params[step]['overwrite'] = False
    step_count = write_param_files(params[step], step_param_dir / step)
    step_counts.append(step_count)



from steps.MeshPrep import get_run_id
# YIELD RUNNING ORDER #
# for each subject - bone:
#[(0,), (1,)] - run all step 1 run_ids
#[(0, 0), (0, 1), (1, 0), (1, 1)] - run all step 2 run_ids
#[(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3), ... ] - run all step 3 run_ids
# ...

log_dir = root_dir / 'reports'
log_dir.mkdir(parents=True, exist_ok=True)
for subject_sideL in subject_sideLs:
    print(f"\nSUBJECT: {subject_sideL}")
    
    for bone_pair in bone_pairs:
        print(f"\tBONES: {bone_pair}")

        for i, (step_count, step) in enumerate(zip(step_counts, steps)):
            print(f"\t\tSTEP: {step}")
            run_ids = itertools.product(*(range(v + 1) for v in step_counts[:i+1]))

            for run_id in run_ids:
                t0 = time.perf_counter()
                print(f"\t\t\tRUN ID: {get_run_id(run_id)}")

                out_dir = root_dir / f"meshes/{subject_sideL}/{bone_pair}"
                out_dir.mkdir(parents=True, exist_ok=True)
                input_json = step_param_dir/step/f'{run_id[-1]}.json'
                ok = run_subprocess([
                    log_dir, # dir for reports
                    full_param_path.name, # pass full params filename for reports
                    'python', 
                    '-u',
                    MeshPipeline_root / f'steps/main_{step}.py', 
                    input_json, 
                    out_dir, # used to determine subject-sideL and bone-arbone in each step
                    *[str(x) for x in run_id]
                ], timeout=params_glob['step_timeout'])

                dt = time.perf_counter() - t0
                write_runtime_log(log_dir, "runtimes.jsonl", dt, subject_sideL, bone_pair, step, input_json, run_id, full_param_path.name)
                print(f"\t\t\tRuntime: {dt:.3f}s - {ok}")



# might be good to keep track of pymeshfix usage in cartilage creation
# only apply timeout to cgal subprocesses not parent subprocess
# shouldn't write all combos to file, should just pass full_params_id and combo id to each run
#   - would solve problem of not being able to run mutiple processes on same output_dir

