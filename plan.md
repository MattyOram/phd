
NEW PARAM STUDY 
- optimiseV2
   - initial runs with the params chosen pick optimal ones
      - justification for using perturb and exude - they target bad cells and make them better
   - then another run where look at combos of lloyd odt perturb exude to see if others help
   - then can do final run with all subjects, but still investigate final most important params
      - for final more general refinement incase of overfit to those 3 subjects.
      
   - choose 1 set of geometry parameters for all subjects because that is more future proof?
      - check how much difference it makes, if not much use same for everyone
         - check biggest % decrease in quality resulting from using avg best params.
      - or use their individual favourites and then in future use average best?
          - but what about n_tets and min thickness? if that changes it will change their preference so better to use avg?




FIGURE OUT HOW MANY NODES ABAQUS ON LAPTOP CAN HANDLE 
 - need this to be sure of final paramters

CHOOSE WHICH POSES TO USE

PLAN FEA SENSITIVITY STUDY