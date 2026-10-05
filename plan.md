
### d0 study
 - Redo d0 study for 3.5T4 for SMT cartilage
 - Then based on those determine minimum, then see if it works for all subs

### Change mc1 position logic
 - make position_cylinder_mc1() function
     - can then easily use on guide to visually verify guide suitability for each subject.
 - whats stopping me from just placing the guide slightly further back so the contact is more central?
    - Add bigger cylinder or extension platform?
    - SHOULD DO THIS - could also only have one guide arm (check which side is best, or print one of each)
        - also check the dimentions for the wiggle room around the base



### Jobs
- (finalise mesh d0)
- look at literature on accuracy of pressure distribution of thin layers of hyperelastic material under large deformation
and on accuracy of tekscan pressure distribution

- run a simulation directly set up in abaqus with experimental materials data
    - take the study3 inp file and strip everything except models surface and BCs

- redo sensitivity with known compressive and tensile behaviour and just look at stiffening vs soffening behaviour
    - DO ALL WITH 50017L ? - in loaded pinch (if possible do other two subs in ext and abd to offer range of poses)
    - also do it for all forces...
    - also do more twist translation combinations
    - also look at BC patch effect - this could include removal of bone and just constrain cartilage base
        - first show patch makes no difference then show rigid bone/removal of bone makes no difference.   
            - Show speed up due to removal and that results are identical with rigid vs removed.
        - can just do softest possible vero vs rigid 

- tie bone and cartilage together see if it matters

- try removing constraints and adding tensioned springs (to reflect compliance of parts) to see what it changes about fe results


- plot yeoh elastico fit against input data just to verify

### Random
- MAYBE DO PROPER TESTS WITH MULTIPLE SENSORS TO SEE IF IT EFFECTS RESULTS AND MIGHT HELP JUSTIFY NOT equilibrating
    - each repeat with a different sensor

- 15006 has dodgy jar_load transforms

- iteravtive solver for memory issues?

- what effect does compliance have on the compression tests?
    - did the standards mention this?


### FE DOWNSACLING
maybe don't project the sensor at all
 - we know angle and y and z
 - set x at best fit to cartilage surface
 - find sensel corners based on closest point 
 - use geodesic lines to connect corners and average elements within each 
OR 
find closest point in x between cart surfaces
 - use that and known y/z/orientation to map the sensor onto the surface 
 - could refine sensor mesh and just piecwise linear it from the sensor mesh element at initial contact
    - piecewise linear mesh can be much finer than sensor, break each sensel down into smaller squares.
        - each of those squares inherits that pressure of its most overlapping element
        - can do mesh refinement to determine how refined sensor mesh needs to be 
    - also do error based on how much sensor may have slipped while being pressed onto surface
I LIKE THIS ONE

#### shape and motion
- grid independence of shape and motion
- evenly distributed points (random sample like in mesh prep?), even density
- sort out Cv and Cm distributions