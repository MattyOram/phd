import sys
import json

# ------------------------------- ALWAYS KEEP THIS FILE IN THIS DIRECTORY ------------------------------ #
# - even if setting output_root somewhere else (a json copy will be saved to the output_root)
# - (could change it so that params_path is a cli arg for main.py)

params = {
    'general': {},
    'preprocessing': {},
    'inp': {}
}

# ••••••••••••••••••••• GLOBAL ••••••••••••••••••••• # - no lists!
params_glob = params['general']

# root directory for outputs and save loc of params file - if relative will be relative to your current directory!
params_glob['mesh_root']       = '../MeshPipeline/outputs/ParamOptimisation/fullRuns/study4' # output_root in MeshPipeline

params_glob['output_root']     = 'outputs/testing/test1'  # output dir for input files and meshes # -------- *** -------- #

params_glob['allow_overwrite'] = True 
# - keeps all copies of full params for each output_root, with -i suffix for each new file

params_glob['step_timeout']    = 1200 # (s) overall time limit just in case



# ••••••••••••••••••••• PREPROCESSING ••••••••••••••••••••• #
params_pre = params['preprocessing']

params_pre['subjects']  = None # provide list of subjects or set to None for all available subjects 
                                                                # (assumes Meshpipeline dir layout)
params_pre['poses']         = ['adduction','abduction','flexion','extension','pinch','grasp','jar','neutral']



##########################################################
# --------------------- PARAMETERS --------------------- #

# main #
# ABAQUS CLI INPUTS
#inp_file = inp_path # path to input file i.e inputs/input1.inp
#job_name = inp_path.name.split('.')[0]   # saves files to this path i.e. inputs/input1
#CPUs = 8 # 4(=8tokens) performance won't get much better after 8(=12tokens)
#setting = 'interactive'

"""# PRE-PROCESSING #
target_dist = 0.01 # gap between cartilage at start of simulation

tpm_patch_params = None # params for bone_surface_patch_nodes() - defaults to ("xlims", [tpm_centroid[0]-100, tpm_centroid[0]])
mc1_patch_params = None #- i.e. ('xlims', [mc1_offset, 100]) - defaults to ("xlims", [mc1_centroid[0], mc1_centroid[0]+100])

# ELEMENT ORDER
element_order = 'linear' # or 'quad'

# ELEMENT TYPES
bone_element_type = "C3D4"
cartilage_element_type = "C3D4H"

# BONE PROPERTIES
bone_material = {
    "E": 1629, # MPa
    "nu": 0.4
}
bone_density=None

# CARTILAGE PROPERTIES
cartilage_material = {
    "C10": 0.091,
    "D1": 0.0         
}
cartilage_density=None

cartilage_friction = 0.0

# REGION IDs
bone_vol_id=1
cartilage_vol_id=2
cartilage_surf_id=-2

# DISPLACEMENT / FORCE LIMITS
mc1_disp_x  = -0.80 # end analysis at this displacement     - starting point is 0.01mm from contact
#Forces = [10.0, 20.0]   # refine step time to hit these forces - would need to set user defined DT REFINEMENT - not worth it right now
max_force = 50.0    # end analysis at this force

# STEP PARAMS
total_step_time = abs(mc1_disp_x) # set to total displacement so that increment params don't have to change with displacement
initial_increment = target_dist          # starting point is 0.01mm from contact
min_increment = 1e-10
max_increment = 0.025

step_type = "STATIC"
nlgeom = "YES" # non-linear geometry
unsymm = "YES" # store unsymmetric matrix
convert_sdi = "NO" #force a new iteration if severe discontinuities occur during an iteration, regardless of the magnitude of the penetration and force errors

equil_iters = 16 # default=16 - upper limit on the number of consecutive equilibrium iterations (without severe discontinuities) (4)
sdi_iters = 15 # deafult=12 - maximum number of severe discontinuity iterations allowed in an increment if CONVERT SDI=NO (7)
increment_attemps = 5 # default=5 - maximum number of attempts allowed for an increment (8)

# --------------------- PARAMETERS --------------------- #
##########################################################"""



# WRITE TO FILE #
param_path = sys.argv[1]
with open (param_path, 'w') as f:
    json.dump(params, f, indent=2)
print(f"\tWrote {param_path}")