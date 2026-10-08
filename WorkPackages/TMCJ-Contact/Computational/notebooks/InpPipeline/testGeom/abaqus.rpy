# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 2022 replay file
# Internal Version: 2021_09_15-18.57.30 176069
# Run by scmo on Mon Sep 28 17:55:58 2026
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(1.36719, 1.36719), width=201.25, 
    height=135.625)
session.viewports['Viewport: 1'].makeCurrent()
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
execfile('test_import.py', __main__.__dict__)
#: ==============================================================================
#: Abaqus input-file import test
#: Input file: /mnt/scratch/scmo/abaqus/test_inpGeom/35T-flexion-00.inp
#: ==============================================================================
#: 
#: Importing input file with mdb.ModelFromInputFile() ...
#: The model "ImportedGeometry" has been created.
#: The part "TPM" has been imported from the input file.
#: The part "MC1" has been imported from the input file.
#: The model "ImportedGeometry" has been imported from an input file. 
#: Please scroll up to check for error and warning messages.
#: Import completed successfully.
#: 
#: MODEL
#: Parts (2): ['MC1', 'TPM']
#: Materials (0): []
#: Sections (0): []
#: Steps (1): ['Initial']
#: Interactions (0): []
#: Interaction properties (0): []
#: Field output requests (0): []
#: History output requests (0): []
#: 
#: PART CONTENTS
#: 
#:   Part: MC1
#:     Nodes: 63682
#:     Elements: 342189
#:     Sets (2): ['MC1_BONE', 'MC1_CARTILAGE']
#:     Surfaces (2): ['MC1_CART_SURF', 'MC1_PATCH_SURF']
#:     Reference points (0): []
#: 
#:   Part: TPM
#:     Nodes: 60331
#:     Elements: 329970
#:     Sets (2): ['TPM_BONE', 'TPM_CARTILAGE']
#:     Surfaces (2): ['TPM_CART_SURF', 'TPM_PATCH_SURF']
#:     Reference points (0): []
#: 
#: ASSEMBLY
#:   Instances (2): ['MC1_INST', 'TPM_INST']
#:   Sets (2): ['RP_MC1', 'RP_TPM']
#:   Surfaces (0): []
#:   Reference points (2): [5, 8]
#: 
#: INSTANCE CONTENTS
#: 
#:   Instance: MC1_INST
#:     Nodes: 63682
#:     Elements: 342189
#:     Sets (2): ['MC1_BONE', 'MC1_CARTILAGE']
#:     Surfaces (2): ['MC1_CART_SURF', 'MC1_PATCH_SURF']
#:     Reference points (0): []
#: 
#:   Instance: TPM_INST
#:     Nodes: 60331
#:     Elements: 329970
#:     Sets (2): ['TPM_BONE', 'TPM_CARTILAGE']
#:     Surfaces (2): ['TPM_CART_SURF', 'TPM_PATCH_SURF']
#:     Reference points (0): []
#: The model database has been saved to "/mnt/scratch/scmo/abaqus/test_inpGeom/imported_testGeom.cae".
#: 
#: Saved imported model database:
#:   /mnt/scratch/scmo/abaqus/test_inpGeom/imported_testGeom.cae
#: 
#: ==============================================================================
#: TEST COMPLETED SUCCESSFULLY
#: ==============================================================================
print 'RT script done'
#: RT script done
