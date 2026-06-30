import sys
import json




params = {
    'global': {},

    'manifold': {},
    'printJig': {}
}

# ••••••••••••••••••••• global ••••••••••••••••••••• #
params_glob = params['global']

params_glob['timeout'] = 300
params_glob['mesh_root'] = '../../../Computational/MeshPipeline/outputs/initialFEAstuff/35T/35Tbest'
params_glob['output_root'] = 'outputs/CA-35T'

params_glob['subjects'] = None    # None if want all subjects
params_glob['mesh_ids'] = ['35T']       # None if want all meshes




# ••••••••••••••••••••• manifold ••••••••••••••••••••• #
params_man = params['manifold']

# if any of the following are set then error will be raised if repaired mesh doesn't pass (can set to None or False)
params_man['max_area']         = 0.05 # max porportion of total cartilage surface area repaired (0 -> 1)
params_man['max_loc']          = 0.20 # location of repairs as proportion of way from boundary to centre (0 -> 1)


# ••••••••••••••••••••• printJig ••••••••••••••••••••• #
params_print = params['printJig']

#params_print['poses'] = ['adduction','abduction','flexion','extension','pinch','grasp','jar','neutral']
params_print['poses'] = ['flexion', 'extension', 'abduction', 'adduction', 'pinch_load', 'neutral']

params_print['jig_path'] = '../CADmodels/instronJigs/solidWorks/Adaptor.ply'

# tpm #
params_print['tpm_disk_depth'] = 10    # depth of jig disk that sits on steel jig (mm)
params_print['tpm_offset'] = 0         # offset of top of jig from tpm centroid (+ive = more overlap)

# mc1 #
params_print['mc1_disk_depth'] = 10    # depth of jig disk that sits on steel jig (mm)
params_print['mc1_potrusion'] = 10 # how far mc1 should potrude out of jig







# WRITE TO FILE #
param_path = sys.argv[1]
with open (param_path, 'w') as f:
    json.dump(params, f, indent=2)
print(f"\tWrote {param_path}")