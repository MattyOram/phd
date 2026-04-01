# MeshPipeline parameter optimisation study initial plan

## Relevant parameters

### global:
* step_timeout = 300

### subjects:
* subeject_sideL = 14548R
    * run for one subject first? Or check mesh quality for current params and pick subjects that span the range?

    * Or choose subjects that span cartilage thickness? or something else

* bone_arbone = tpm-mc1
    * just run for tpm for now

### 2Dmesh:
* taubin_iters = 50
    * Justified in smooth-box.ipynb

* max_gap_remesh ≥ params_cart['max_gap_cartilage']
    * ensures entire cartilage region is within fine_edge_length region
    * params_cart['max_gap_cartilage'] is already justified at 2mm so don't need to investigate either of these params

* fine_edge_length = 0.2
    * justified in remesh-box.ipyb

* coarse_egde_length < min(mesh3D[h_bone_max])
    * Provide smooth surface as input for 3Dmesh to coarsen

* grad_width = < max(mesh3D[d0])
    * Provide smooth surface as input for 3Dmesh to coarsen
    * Changed to Euclidean to reflect 3Dmesh beahviour

* remesh_iters = 10
    * Justified in remesh-box.ipynb


### cartilage:
* poses = ['adduction','abduction','flexion','extension','pinch','grasp','jar','neutral']
    * need to do all relevant unloaded poses

* max_gap_cartilage = 2
    * justified by the Brown University paper

* taper_width = 1.5
    * Justified in taper-box.ipynb

* max_height = N/A
    * No longer needed as taper width is now constant (taper-box.ipynb)

* p_h = 8.5
    * Justified in taper-box.ipynb

* p_v = 1
    * Justified by the fact it works and only affects part of cartilage that is manually created and away from contact.

* smooth_iters = 100
    * Justified in taper-box.ipynb

### 3Dmesh

#### Criteria
* n_tets = 3 
    * this will be dictated by FEA convergence study

* taper_size = 0.2 
    * average is ~0.6mm => ~3 accross thickness

* h_bone_max = [0.5, 1, 2]

* d0 = [2, 4, 8]

* fd_cart_near = [0.02, 0.04, 0.08]

* fd_cart_far = [0.01, 0.02, 0.04]

* fd_bone = [0.2, 0.4, 0.8]

* facet_angle = [7.5, 15, 30]

* cell_radius_edge_ratio = [3, 6, 12]


#### Optmisation
     - Set time limits and iterations so that so hopefully arn't hit unless it is making very slow progress
     - check exit codes after
     - lloyd give 2 minutes because it can be slower - global regularisation
     - peturb and exude target sliver triangles - local repair

* lloyd = True
    * time_limit = 120
    * max_iterations = 0 (no limit)
    * convergence = []
    * freeze_bound = []
    * do_freeze = True

* peturb = True
    * time_limit = 60
    * sliver_bound = []

* exude = True
    * time_limit = 60
    * sliver_bound = []
