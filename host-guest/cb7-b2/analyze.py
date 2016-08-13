#!/usr/bin/env python
"""
OpenPathSampling simulation of ligand unbinding for CB7:B2 host-guest complex.

"""
################################################################################
# IMPORTS
################################################################################

import numpy as np

from simtk import openmm, unit
from simtk.openmm import app

import openpathsampling as paths
import openpathsampling.engines.openmm as engine

import mdtraj as md

# Import mojo for using units
paths.netcdfplus.dictify.ObjectJSON.safe_modules += ['simtk', 'unit']

################################################################################
# PARAMETERS
################################################################################

receptor_atoms = np.arange(0, 126)
ligand_atoms = np.arange(126, 156)

################################################################################
# MAIN
################################################################################

# Create storage
print('Opening storage...')
storage = paths.Storage("host-guest.nc", 'r')

distance = storage.cvs['distance']
thin = 100
last_index = -thin
for (index, traj) in enumerate(storage.trajectories[1:]):
    x = distance(traj).flatten()
    if np.any(x < 0.35) and np.any(x > 0.65) and ((index-last_index) > thin):
        print(index, x)
        filename = 'trajectory-%05d.pdb' % index
        storage.trajectories[0].md().save_pdb(filename)
        last_index = index
