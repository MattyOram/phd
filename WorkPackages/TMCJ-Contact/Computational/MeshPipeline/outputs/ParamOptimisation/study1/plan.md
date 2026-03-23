# MeshPipeline parameter optimisation study initial plan

## Relevant parameters

#### global:
* step_timeout = 330?

#### subjects:
* subeject_sideL = 14548R
    * run for one subject first? Or check mesh quality for current params and pick subjects that span the range?

* bone_arbone = tpm-mc1
    * just run for tpm for now

#### 2Dmesh:
* taubin_iters = !!!!!!! pre justify !!!!!!!
    * affects computed cartilage height field and 3Dmesh input
    * Justify based on volume change and surface rmse and element quality

* max_gap_remesh ≥ params_cart['max_gap_cartilage']
    * ensures entire cartilage region is within fine_edge_length region
    * but params_cart['max_gap_cartilage'] is already justified at 2mm so don't need to investigate either of these params

* fine_edge_length = !!!!!!! pre justify !!!!!!!
    * affects computed cartilage height field and edge length of cartilage remesh and 3Dmesh input
    * justify based on convergence of height field

* coarse_egde_length < min(mesh3D[h_bone_max])
    * Provide smooth surface as input for 3Dmesh to coarsen

* grad_width = < max(mesh3D[d0])
    * Provide smooth surface as input for 3Dmesh to coarsen
    * Although this is geodesic distance and 3Dmesh is based on euclidean
        * change to euclidean???????????????????????????????????????????????????????

* remesh_iters = maybe !!!!!!! pre justify !!!!!!! - see if it converges
    * affects computed cartilage height field and 3Dmesh input


#### cartilage:
* remesh_cartilage = True
    * along with smoothing, gives smoother surface for 3Dmesh input

* poses = ['adduction','abduction','flexion','extension','pinch','grasp','jar','neutral']
    * need to do all relevant unloaded poses

* max_gap_cartilage = 2
    * justified by the Brown University paper

* taper_width = !!!!!!! pre justify !!!!!!!
    * ideally justify this by looking at the height field and picking a value that limits unnatural flaring and avoids main contact region

* max_height = !!!!!!! pre justify !!!!!!!
    * same as for taper_width

* p_h = []
    * could help improve element quality at boundary

* p_v = !!!!!!! pre justify !!!!!!!
    * have a look at effect of favouring normal vector earlier, might give smoother taper region

* smooth_iters = []
    * effects cartilage height field and 3Dmesh input

* n_iters = maybe !!!!!!! pre justify !!!!!!! - see if it converges
    * effects cartilage height field and 3Dmesh input

#### 3Dmesh
* 
