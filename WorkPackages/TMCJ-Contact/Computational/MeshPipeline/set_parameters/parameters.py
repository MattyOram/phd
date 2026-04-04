"""
DO STUDY TO DETERMINE OPTIMAL PARAMS
 - evaluate quality based on mesh quality and accuracy of surface of interest (cartilage contact region) 
 - check taper region proportion
    - need to to inspect meshes without taper to better understand reasonable taper width (can delaunay then stitch them)
 - for cartilage rmse. compare to dense cartilage height field. choose dense cartilage height field rmse convergence?

beware if using subprocesses to parrallelise - need to give cgal input files per subprocess names to ensure no overwriting issues
"""


from phd_helpers.paths import get_project_root
import pandas as pd
import json
import sys


#------------------------------- ALWAYS KEEP THIS FILE IN THIS DIRECTORY ------------------------------ #
# - even if setting output_root somewhere else (a json copy will be saved to the output_root)
# - (could change it so that params_path is a cli arg for main.py)


params = {
    'global': {},
    'subjects': {},

    '2Dmesh': {},
    'cartilage': {},
    '3Dmesh': {},
    'manifold': {}
}

# ••••••••••••••••••••• GLOBAL ••••••••••••••••••••• # - no lists!
params_glob = params['global']

# root directory for outputs and save loc of params file - if relative will be relative to your current directory!
#params_glob['output_root']     = 'outputs/ParamOptimisation/study1' 
params_glob['output_root']     = 'outputs/3Dmesh_postprocess_test' 

params_glob['allow_overwrite'] = True # If False, ignores per step overwrite flags
# - Will always overwite step specific param directories!
#     - but not full_params.json - keeps all copies of full params for each output_root, with -i suffix for each new file


params_glob['step_timeout']    = 180 # (s) time limit per step (3D meshing can hang if criteria too strict)


# switches for the steps to carry out
# - for now, must do all previous steps unless passing input mesh in params here
params_glob['steps'] = {
    '2Dmesh':    True,
    'cartilage': True,
    '3Dmesh':    True,
    'manifold':  False # only might be needed if planning to 3D print (haven't checked...)
} 


# ••••••••••••••••••••• SUBJECTS ••••••••••••••••••••• #
params_sub = params['subjects']



params_sub['subject_sideL'] = ['14548R'] # subject id and wrist side 
#params_sub['subject_sideL'] = ['50014R'] # subject id and wrist side 


# all CMC subjects that pass both bone and cartilage interference checks for final params (TMCJ-Contact 2Dmesh->cartilage)
#  - see: InterferenceCheckFinal/interference-box.ipynb
#  - 36 total
#params_sub['subject_sideL'] = pd.read_csv(
#                                get_project_root() / 'WorkPackages/TMCJ-Contact/Computational/MeshPipeline/subs_ok.csv'
#                            ).subs_ok.to_list()

# ALL CMC SUBJECTS
#  - 46 total
"""params_sub['subject_sideL'] = ['50037L', '50090R', '15294R', '50053R', '50049R', '15737R', 
                               '50061R', '14726R', '50016L', '14613R', '15358R', '50008L', 
                               '16276L', '15441R', '50024R', '14874R', '22306R', '14727R', 
                               '50033L', '15284R', '50017L', '50029R', '50027L', '50018L', 
                               '15357R', '50001R', '15006R', '14819R', '14873R', '50034R', 
                               '15283R', '50021R', '50019R', '15285R', '50020R', '50006R', 
                               '50000R', '50007L', '50014R', '14827L', '14818R', '14548R', 
                               '15882R', '15282R', '50045R', '14685R']"""


params_sub['bone_arbone']   = ['tpm-mc1'] # target_bone - articulating_bone



# ••••••••••••••••••••• 2DMESH ••••••••••••••••••••• #
params_2D = params['2Dmesh']

params_2D['overwrite']          = False # overwrite output meshes if they already exist (if params_glob['allow_overwrite'])

params_2D['input_bone_mesh']    = None # filepath
params_2D['input_arbone_mesh']  = None # filepath

params_2D['output_filename']    = None # remesh filename (.vpt/.obj ...)
params_2D['cgal_input_name']    = None # filename for cgal input mesh (assign unique per subprocess name to avoid issues!)

# path to dir containing bin, inputs, outputs folders
params_2D['cgal_path']          = str(get_project_root() / 'WorkPackages/TMCJ-Contact/Computational/MeshPipeline/cpp/2Dmesh')

        # ACTUAL PARAMETERS #
params_2D['poses']              = [
                            ['adduction','abduction','flexion','extension','pinch','grasp','jar','neutral']
                            ]

params_2D['taubin_iters']       = 50  # n smoothing iterations
params_2D['save_smoothed_mesh'] = False # by default, only the final remeshed output of 2Dmesh is saves
                                        # if taubin iters is not list but other 2Dmesh param is then it will 
                                        # save the identical smooth mesh every time with different run-id...
params_2D['output_filename_smooth'] = None # smooth mesh filename (.vtp/.obj ...)

params_2D['remesh_arbone']      = True # results in smoother cartilage surface and better 3Dmesh quality

# max gap remesh must be ≥ params_cart['max_gap_cartilage'] to ensure entire cartilage is within fine_edge_length region (if remeshing cartilage)
params_2D['max_gap_remesh']     = 2.5   # max distance of point on mesh1 from mesh2 to be part of fine mesh region
params_2D['adjacent_cells']     = True # include any cells with ≥1 node in region - True should mean can set max_gap_remesh = max_gap_cartilage

params_2D['fine_edge_length']   = 0.2 # edge length in articulation region
params_2D['coarse_edge_length'] = 0.4 # edge length away from articulation region
params_2D['grad_width']         = 8 # width of edge length gradient region from fine to coarse
params_2D['remesh_iters']       = 5  # n isotropic remeshing iterations
        # ACTUAL PARAMETERS #






# ••••••••••••••••••••• CARTILAGE ••••••••••••••••••••• #
params_cart = params['cartilage']

params_cart['overwrite']          = False # overwrite output mesh it already exists (if params_glob['allow_overwrite'])

params_cart['input_bone_mesh']    = None # filepath
params_cart['input_arbone_mesh']  = None # filepath

params_cart['output_filename']    = None # mesh filename (.vtp) 
params_cart['cgal_input_name']    = None # filename for cgal input mesh (assign unique per subprocess name to avoid issues!)

# path to dir containing bin, inputs, outputs folders
params_cart['cgal_path']          = str(get_project_root() / 'WorkPackages/TMCJ-Contact/Computational/MeshPipeline/cpp/2Dmesh')

params_cart['save_orig_smooth']    = False

        # ACTUAL PARAMETERS #
params_cart['remesh_cartilage']   = True # after creating cartilage cap remesh to high quality mesh (not needed if mesh3D but maybe makes 3D mesh better quality)
params_cart['edge_length']        = None # target edge length of cartilage remesh (if None read from params_2D['fine_edge_length'])

params_cart['poses']              = [
                            ['adduction','abduction','flexion','extension','pinch','grasp','jar','neutral']
                            ]

params_cart['max_gap_cartilage']  = 2 # maximum articular gap for cartilage presence - justified by BU paper

# if params_2D['remesh_arbone']
params_cart['use_remeshed_arbone']= True # results in smoother cartilage surface and better 3Dmesh quality

params_cart['taper_width']        = 1.5 # width of cartilage taper region
params_cart['p_h']                = 8.5 # shape of taper height (1=linear , higher = steeper taper)
params_cart['p_v']                = 1 # shape of vector ratio (1=linear) - normal to midpoint vector ratio for taper region extrusion
params_cart['smooth_iters']       = 100 # looked at this in ArticularGap4-box - might be different for different tri density?
params_cart['n_iters']            = 5 # n isotropic remeshing iterations for cartilage remesh
        # ACTUAL PARAMETERS #





# ••••••••••••••••••••• 3Dmesh ••••••••••••••••••••• #
params_3D = params['3Dmesh']

params_3D['overwrite']          = True # overwrite postprocessed output mesh if it already exist (if params_glob['allow_overwrite'])

params_3D['input_mesh']         = None # filepath

params_3D['output_filename']    = None # mesh filename (.vtu) (if keep_cgal_copy=True, cgal copy is auto given .mesh)
params_3D['cgal_input_name']    = '3'   # filename add on for cgal inputs (assign unique per subprocess name to avoid issues!)
params_3D['save_cgal_inputs']   = False

# path to dir containing bin, inputs, outputs folders
params_3D['cgal_path']          = str(get_project_root() / 'WorkPackages/TMCJ-Contact/Computational/MeshPipeline/cpp/3Dmesh')

# One of these must be true for a copy to be saved to the output path - otherwise cgal copy only exists in cgal output
params_3D['postprocess']        = True # Assign region_id scalar
params_3D['keep_cgal_copy']     = False # keep copy of cgals ouput mesh - (pre postprocessing mesh)

        # VOLUMETRIC MESHING PARAMETERS #
params_3D['cgal_params'] = { 
    # Sizing field params
    "sizing_field": {
        "n_tets": 3,        # number of tetrahedrons accross thickness of cartilage
        "min_size": 0.02,   # min target edge length (or circumradius?) within main cartilage region
        "max_size": 0.5,   # max target edge length (or circumradius?) within main cartilage region

        # edge size linearly increases from d_taper to cartilage boundary
        "d_taper": params_cart['taper_width'], # width of cartilage taper region that doesn't use height based size
        "taper_size": 0.2, # target max edge length (or circumradius?) at cartilage boundary

        # bone ramp - bone surface/volume mesh grows with distance from interface
        "h_bone_max": 2.0,  # max edge length (or circumradius?) - bone surface/volumetric mesh
        "d0": [1.0]          # distance of growth region from interface edge length (or circumradius?) to h_bone_max                 
    },                      # - ~8->10 mm covers whole tpm

    # Surface facet distance params - max deviation from origial mesh surface
    # - distance between centre of circumscribed circle of candidate facet and centre of delaunay ball 
    # - delaunay ball passes through 3 the vertices of the candidate facet and it's centre lies on the input mesh
    "facet_distance": {
        "fd_cart_near": 0.10, # target max facet distance - at cartilage boundary (==fd_edge_loop)
        "fd_cart_far": 0.05,  # target max facet distance - at d0 from cartilage boundary
        "d0_cart": params_cart['taper_width'],        # distance over which cartilage fd grows

        "fd_bone": 1.0,      # target max facet distance - bone
        "fd_edge_loop": None, # target max facet distance - edge loop (if None==fd_cart_near)
    },

    # CGAL Mesh criteria
    #mesh code hanging can happen in the initial make_mesh_3 step due to too strict criteria - probs facet distance and quality combo
    "criteria": {
        # these two only affect surface mesh and the input is already high quality 
        #  - plus higher is best and any higher and the code starts hanging so leave as they are
        "facet_angle": 30,            # target min dihedral(?) angle - hangs at higher values
        "cell_radius_edge_ratio": 3,  # target max radius ratio
        "manifold_with_boundary": False # Should ensure that volume shells of returned mesh are manifold (default=False)
                                        # - found that it can make remeshing either hang or take forever probs cos of criteria
    },

    # above - facet angle, cell ratio, fd bone, fd interface, d0, h_bone_max, taper_size?, n_tets
        # - fd could be based on what is acceptable difference (given specifics of study)
        # - ((4*3**7) / 7) / 60 = 21 ; 7 is doable
    # COULD SPLIT UP MESH CREATION AND OPTIMISATION - for optimisation STUDIES #


    # ---- OPTIMISATION STEPS ---- #
    # if flags are set to [True, False] then corresponding params are only looped over when flag==True 
    # - see main.py
    "optimisation": { # This determines which of the following optimisation steps are used
        "odt": False,    # not had good results with odt - makes worst worse to make avg better?
        "lloyd": False,   # does good stuff - smoothes mesh improves qaulity
        "perturb": False, # does good stuff - improves dihedral angles of worst elements
        "exude": False    # doesn't seem to do much  - removes slivers
    },

    "odt": { # these were the original args for lloyd when I first got it working, except iter=0
        "time_limit": 60, # 0 means that there is no limit (default)
        "max_iteration_number": 3, # 0 means that there is no limit (default)
        "convergence": 0.002, # default = 0.02 
        "do_freeze": True, # default True
        "freeze_bound": 0.002 # default 0.01
    },

    "lloyd": {
        "time_limit": 120, # 0 means that there is no limit (default)
        "max_iteration_number": 0, # 0 means that there is no limit (default) - hits this if 50 and con 0.005, f_b 0.01, t_l 120
        "convergence": 0.002, # default = 0.02
        "do_freeze": True, # default True
        "freeze_bound": 0.002 # default 0.01
    },

    # if sliver bound set for peturb and exude, it may sacrifice other metrics to hit sliver target but also somtimes helps a lot.
    "perturb": {
        "time_limit": 60,   # 0 means that there is no time limit (default)
        "sliver_bound": 15, # targeted lower bound on dihedral angles (0=default)
    },

    "exude": {
        "time_limit": 60,   # 0 means that there is no time limit (default)
        "sliver_bound": 0, # targeted lower bound on dihedral angles (0=default)
    },
}
        # VOLUMETRIC MESHING PARAMETERS #







# ••••••••••••••••••••• manifold ••••••••••••••••••••• #
params_man = params['manifold']

params_man['input_mesh']       = None # filepath
params_man['output_filename']  = None # mesh filename (.vtp)

# if any of the following are set then error will be raised if repaired mesh doesn't pass (can set to None or False)
params_man['require_manifold'] = True # repairs must work
params_man['max_area']         = 0.05 # max porportion of total cartilage surface area repaired (0 -> 1)
params_man['max_loc']          = 0.20 # location of repairs as proportion of way from boundary to centre (0 -> 1)








# WRITE TO FILE #
param_path = sys.argv[1]
with open (param_path, 'w') as f:
    json.dump(params, f, indent=2)
print(f"\tWrote {param_path}")
