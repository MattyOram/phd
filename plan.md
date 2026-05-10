PLAN FEA SENSITIVITY STUDY


### d0 study
 - Redo d0 study for 3.5T4 for SMT cartilage
 - Then based on those determine minimum, then see if it works for all subs

### robustness study - done for 35T4-d5 
 - generate meshes for the 3 contact patch subjects for both bones
 - generate input files for flexion, extension, abduction, adduction, pinch_load
 - Run all input files

### Accuracy study for 14548neutral35T-d5 
 - Compare current setup with experimental results
 - Do FE param sensitivity study for this subject (try elastico material models that have different behaviour in tension)
 - Make the results match

### Aire






### Change mc1 position logic
 - make position_cylinder_mc1() function
     - can then easily use on guide to visually verify guide suitability for each subject.
     - 10mm between top of cyclinder and tip of mc1 cartilage