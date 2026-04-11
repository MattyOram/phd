import sys
import json

# ------------------------------- ALWAYS KEEP THIS FILE IN THIS DIRECTORY ------------------------------ #
# - even if setting output_root somewhere else (a json copy will be saved to the output_root)
# - (could change it so that params_path is a cli arg for main.py)

params = {
    'general': {},
    'inp': {}
}

# ------- GLOBAL ---------------------------------------------------------------------------------------- # - no lists!
params_gen = params['general']

# root directory for outputs and save loc of params file - if relative will be relative to your current directory!
#params_gen['mesh_root']    = '../MeshPipeline/outputs/ParamOptimisation/fullRuns/study4' # output_root in MeshPipeline
params_gen['mesh_root']    = 'outputs/StMmLt'

#params_gen['subjects']  = ['14548R']
params_gen['subjects']     = ['22306R', '50037L'] # provide list of subjects or set to None for all available subjects 
                                                                # (assumes Meshpipeline dir layout)

params_gen['output_root']  = 'outputs/StMmLt-inps'  # output dir for input files and meshes        # -------- *** -------- #

params_gen['timeout'] = 600 # (s) overall time limit just in case



# main #
# ABAQUS CLI INPUTS
#inp_file = inp_path # path to input file i.e inputs/input1.inp
#job_name = inp_path.name.split('.')[0]   # saves files to this path i.e. inputs/input1
#CPUs = 8 # 4(=8tokens) performance won't get much better after 8(=12tokens)
#setting = 'interactive'

# ------- INPUT FILE ---------------------------------------------------------------------------------------- #
params_inp = params['inp']

params_inp['poses'] = [
            #'adduction','abduction','flexion','extension',
            #'pinch','grasp','jar','jar_load','pinch_load','grasp_load',
            'neutral'
            ]
#params_inp['poses'] = ['neutral']

params_inp['save_meshes'] = False # can parse from inp files - also will currently overwrite for each run_id

# PRE-PROCESSING #
params_inp['target_dist'] = 0.01 # gap between cartilage at start of simulation

params_inp['tpm_patch_params'] = ("euclidean", 3) # distance of BC patch from cartilage boundary
params_inp['mc1_patch_params'] = ("euclidean", 5) # distance of BC patch from cartilage boundary

# ELEMENT ORDER - now inferred from element type
#params_inp['element_order'] = 'quad' # 'linear' (4 node) or 'quad' (10 node (~8x linear node count))

# ELEMENT TYPES
params_inp['element_type']      = ["C3D4", "C3D10"]
params_inp['cartilage_hybrid']  = True # e.g. C3D10H
#params_inp['cartilage_element_type'] = "C3D10H"

# BONE PROPERTIES
params_inp['bone_material'] = {
                        "E": 1629, # MPa
                        "nu": 0.4
                    }
params_inp['bone_density'] = None

# CARTILAGE PROPERTIES
params_inp['cartilage_material'] = {
                        "C10": 0.091,
                        "D1": 0.0         
                    }
params_inp['cartilage_density']  = None
params_inp['cartilage_friction'] = 0.0

# REGION IDs
params_inp['bone_vol_id']       = 1
params_inp['cartilage_vol_id']  = 2
params_inp['cartilage_surf_id'] = -2

# DISPLACEMENT / FORCE LIMITS
params_inp['mc1_disp_x']  = -0.80 # end analysis at this displacement     - starting point is 0.01mm from contact
#Forces = [10.0, 20.0]   # refine step time to hit these forces - would need to set user defined DT REFINEMENT - not worth it right now
params_inp['max_force'] = 50.0    # end analysis at this force

# STEP PARAMS
params_inp['total_step_time'] = abs(params_inp['mc1_disp_x']) # set to total displacement so that increment params don't have to change with displacement
params_inp['initial_increment'] = params_inp['target_dist']         # starting point is 0.01mm from contact
params_inp['min_increment'] = 1e-5
params_inp['max_increment'] = 0.025

params_inp['step_type']   = "STATIC"
params_inp['nlgeom']      = "YES" # non-linear geometry
params_inp['unsymm']      = "YES" # store unsymmetric matrix
params_inp['convert_sdi'] = ["NO"] #force a new iteration if severe discontinuities occur during an iteration, regardless of the magnitude of the penetration and force errors

params_inp['equil_iters']       = 16 # default=16 - upper limit on the number of consecutive equilibrium iterations (without severe discontinuities) (4)
params_inp['sdi_iters']         = 15 # deafult=12 - maximum number of severe discontinuity iterations allowed in an increment if CONVERT SDI=NO (7)
params_inp['increment_attemps'] = 5 # default=5 - maximum number of attempts allowed for an increment (8)




# ------- WRITE TO FILE ---------------------------------------------------------------------------------------- #
param_path = sys.argv[1]
with open (param_path, 'w') as f:
    json.dump(params, f, indent=2)
print(f"\tWrote {param_path}")