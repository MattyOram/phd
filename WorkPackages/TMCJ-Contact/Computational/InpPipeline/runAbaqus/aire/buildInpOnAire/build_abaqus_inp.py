"""
Build complete .inp starting with the inpGeom.inp
abaqus cae noGUI=build_abaqus_model.py -- inpGeom.inp 00.json complete.inp
"""
from __future__ import print_function

#------------------------ IMPORTS ------------------------#
import json
import os
import sys
import shutil



#------------------------ STEP INCREMENT SETTINGS ------------------------#
def step_settings(p, duration, suffix=''):
    return dict(timePeriod=duration,
                initialInc=p['initial_increment' + suffix],
                minInc=p['min_increment' + suffix],
                maxInc=min(p['max_increment' + suffix], duration))


def build(input_path, p, output_path):
    from abaqus import mdb
    import abaqusConstants as C
    import mesh
    # Load the CAE modules that register the corresponding model methods.
    import part
    import material
    import section
    import assembly
    import interaction
    import step
    import load
    import job


    #------------------------ IMPORT GEOMETRY ------------------------#
    model_name = 'TMC'
    model = mdb.ModelFromInputFile(name=model_name, inputFileName=input_path)
    a = model.rootAssembly

    #------------------------ BONE MATERIAL ------------------------#
    bm = p['bone_material']
    mat = model.Material(name='BONE')
    mat.Elastic(table=((bm['E'], bm['nu']),))

    #------------------------ CARTILAGE MATERIAL ------------------------#
    cm = p['cartilage_material']
    n = int(cm['n'])
    coefficients = tuple(cm['C{}0'.format(i)]
                         for i in range(1, n + 1))
    coefficients += tuple(cm['D{}'.format(i)]
                          for i in range(1, n + 1))
    mat = model.Material(name='CARTILAGE')
    mat.Hyperelastic(materialType=C.ISOTROPIC, testData=C.OFF,
                     type=C.REDUCED_POLYNOMIAL, n=n,
                     volumetricResponse=C.VOLUMETRIC_DATA, table=(coefficients,))

    #------------------------ DENSITIES AND SOLID SECTIONS ------------------------#
    for region, density_key in (('BONE', 'bone_density'), ('CARTILAGE', 'cartilage_density')):
        density = p[density_key]
        if density is not None:
            model.materials[region].Density(table=((density,),))
        model.HomogeneousSolidSection(name=region + '_SECTION', material=region)


    #------------------------ ELEMENT TYPES AND SECTION ASSIGNMENTS ------------------------#
    codes = {'BONE': p['element_type'],
             'CARTILAGE': p['element_type'] + p['cartilage_element_suffix']}
    for bone in ('TPM', 'MC1'):
        pt = model.parts[bone]
        for region in ('BONE', 'CARTILAGE'):
            region_set = pt.sets[bone + '_' + region]
            code = codes[region]
            pt.setElementType(regions=(region_set.elements,),
                              elemTypes=(mesh.ElemType(elemCode=getattr(C, code),
                                                       elemLibrary=C.STANDARD),))
            pt.SectionAssignment(region=region_set, sectionName=region + '_SECTION')

        #------------------------ REFERENCE POINT COUPLINGS ------------------------#
        model.Coupling(name='CP_' + bone, controlPoint=a.sets['RP_' + bone],
                       surface=a.instances[bone + '_INST'].surfaces[bone + '_PATCH_SURF'],
                       influenceRadius=C.WHOLE_SURFACE, couplingType=C.KINEMATIC,
                       localCsys=None, u1=C.ON, u2=C.ON, u3=C.ON,
                       ur1=C.ON, ur2=C.ON, ur3=C.ON)
    a.regenerate()


    #------------------------ CONTACT PROPERTIES ------------------------#
    prop = model.ContactProperty('CART_CONTACT_PROPERTY')
    prop.NormalBehavior(pressureOverclosure=C.HARD, allowSeparation=C.ON)
    friction = p['cartilage_friction']
    if friction == 0:
        prop.TangentialBehavior(formulation=C.FRICTIONLESS)
    else:
        prop.TangentialBehavior(formulation=C.PENALTY, directionality=C.ISOTROPIC,
                                table=((friction,),), maximumElasticSlip=C.FRACTION,
                                fraction=0.005)

    #------------------------ CONTACT INTERACTION ------------------------#
    main_surface = a.instances['TPM_INST'].surfaces['TPM_CART_SURF']
    secondary_surface = a.instances['MC1_INST'].surfaces['MC1_CART_SURF']
    if p['contact_type'] == 'general':
        contact = model.ContactStd(name='CART_CONTACT', createStepName='Initial')
        contact.includedPairs.setValuesInStep(stepName='Initial', useAllstar=C.OFF,
                                              addPairs=((main_surface, secondary_surface),))
        contact.contactPropertyAssignments.appendInStep(
            stepName='Initial', assignments=((main_surface, secondary_surface,
                                               'CART_CONTACT_PROPERTY'),))
    else:
        primary = str(p['primary']).upper()
        secondary = str(p['secondary']).upper()
        main_surface = a.instances[primary + '_INST'].surfaces[primary + '_CART_SURF']
        secondary_surface = a.instances[secondary + '_INST'].surfaces[secondary + '_CART_SURF']
        model.SurfaceToSurfaceContactStd(
            'CART_CONTACT', 'Initial', main_surface, secondary_surface,
            sliding=C.FINITE, enforcement=C.SURFACE_TO_SURFACE,
            interactionProperty='CART_CONTACT_PROPERTY', thickness=C.ON,
            adjustMethod=C.NONE)


    #------------------------ STEP SETTINGS ------------------------#
    common = dict(nlgeom=C.ON if p['nlgeom'] == 'YES' else C.OFF,
                  matrixStorage=C.UNSYMMETRIC if p['unsymm'] == 'YES' else C.SYMMETRIC,
                  convertSDI=C.CONVERT_SDI_ON if p['convert_sdi'] == 'YES' else C.CONVERT_SDI_OFF)

    #------------------------ INITIAL DISPLACEMENT STEP ------------------------#
    model.StaticStep(name='MOVE', previous='Initial',
                     **dict(common, **step_settings(p, p['total_step_time'])))

    #------------------------ BOUNDARY CONDITIONS ------------------------#
    model.DisplacementBC(name='TPM_FIXED', createStepName='Initial', region=a.sets['RP_TPM'],
                         u1=0.0, u2=0.0, u3=0.0, ur1=0.0, ur2=0.0, ur3=0.0)
    model.DisplacementBC(name='MC1_GUIDE', createStepName='Initial', region=a.sets['RP_MC1'],
                         u2=0.0, u3=0.0, ur1=0.0, ur2=0.0, ur3=0.0)
    axial_bc = model.DisplacementBC(name='MC1_MOVE', createStepName='MOVE',
                                    region=a.sets['RP_MC1'], u1=float(p['mc1_disp_x']))

    #------------------------ FORCE STEPS AND LOADS ------------------------#
    previous = 'MOVE'
    previous_force = 0.0
    direction = 1.0 if float(p['mc1_disp_x']) > 0 else -1.0
    forces = p['force_steps']

    if not isinstance(forces, list):
        forces = [forces]
    for i, force in enumerate(forces):
        name = 'F{:g}'.format(force)
        duration = force - (3.0 if i == 0 else previous_force)
        settings = step_settings(p, duration, '_F1' if i == 0 else '_Fn')
        model.StaticStep(name=name, previous=previous, **dict(common, **settings))
        if i == 0:
            axial_bc.deactivate(name)
            force_load = model.ConcentratedForce(name='MC1_FORCE', createStepName=name,
                                                 region=a.sets['RP_MC1'], cf1=direction * force)
        else:
            force_load.setValuesInStep(stepName=name, cf1=direction * force)
        previous, previous_force = name, force


    #------------------------ OUTPUTS ------------------------#
    for repository in (model.fieldOutputRequests, model.historyOutputRequests):
        for name in list(repository.keys()):
            del repository[name]
    model.FieldOutputRequest(name='FIELDS', createStepName='MOVE', frequency=1,
                             variables=('U', 'COORD', 'S', 'LE', 'CSTRESS', 'CDISP', 'CSTATUS', 'CNAREA'))
    model.HistoryOutputRequest(name='MC1_HISTORY', createStepName='MOVE', frequency=1,
                               region=a.sets['RP_MC1'], variables=('U1', 'RF1'))
    model.HistoryOutputRequest(name='TPM_HISTORY', createStepName='MOVE', frequency=1,
                               region=a.sets['RP_TPM'], variables=('RF1',))
    model.HistoryOutputRequest(name='ENERGY', createStepName='MOVE', frequency=1,
                               variables=('ALLIE', 'ALLSD'))
    contact_history = dict(name='CONTACT_AREA', createStepName='MOVE',
                           frequency=1, variables=('CAREA',))
    if p['contact_type'] == 'explicit':
        contact_history['interactions'] = ('CART_CONTACT',)
    model.HistoryOutputRequest(**contact_history)


    #------------------------ WRITE INPUT FILE ------------------------#
    job_name = os.path.splitext(os.path.basename(output_path))[0]

    mdb.Job(name=job_name, model=model_name).writeInput(
        consistencyChecking=C.ON
    )

    shutil.move(job_name + '.inp', output_path)
    print('Wrote: ' + output_path)



#------------------------ READ PER-RUN PARAMETERS AND BUILD MODEL ------------------------#

def convert_strings(value): # cos python 2 loads strings as unicode
    if isinstance(value, dict):
        return {str(k): convert_strings(v) for k, v in value.items()}
    if isinstance(value, list):
        return [convert_strings(v) for v in value]
    if isinstance(value, unicode):
        return value.encode('utf-8')
    return value


if __name__ == '__main__':
    args = sys.argv[-2:]
    input_path, param_path = [os.path.abspath(path) for path in args]
    output_path = input_path.replace('-geom', '')
    with open(param_path) as stream:
        params = convert_strings(json.load(stream))
    build(input_path, params, output_path)
