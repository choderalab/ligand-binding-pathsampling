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

################################################################################
# PARAMETERS
################################################################################

receptor_atoms = np.arange(0, 126)
ligand_atoms = np.arange(126, 156)

################################################################################
# MAIN
################################################################################

# Create host-guest test system
print('Creating host-guest system...')
from openmmtools.testsystems import HostGuestVacuum
testsystem = HostGuestVacuum()

# Generate an OpenPathSampling template.
print('Creating template...')
template = engine.snapshot_from_testsystem(testsystem)

print('Creating an integrator...')
integrator = openmm.LangevinIntegrator(300*unit.kelvin, 1.0/unit.picoseconds, 2.0*unit.femtoseconds)
integrator.setConstraintTolerance(1.0e-6)

print("Selecting a platform...")
platform = openmm.Platform.getPlatformByName('CPU')
properties = {'OpenCLPrecision': 'mixed'}

# Create an engine
print('Creating engine...')
engine_options = {
    'n_frames_max': 2000,
    'platform': 'CPU',
    'n_steps_per_frame': 10
}
engine = engine.Engine(
    template.topology,
    testsystem.system,
    integrator,
    properties=properties,
    options=engine_options
)
engine.name = 'default'

# Create a hot engine for generating an initial unbinding path
print('Creating a "hot" engine...')
integrator_hot = openmm.LangevinIntegrator(900*unit.kelvin, 1.0/unit.picoseconds, 2.0*unit.femtoseconds)
integrator_hot.setConstraintTolerance(1.0e-6)
engine_hot = engine.from_new_options(integrator=integrator_hot)
engine_hot.name = 'hot'

# Select engine mover
paths.EngineMover.engine = engine

# Create storage
print('Initializing storage...')
storage = paths.Storage("host-guest.nc", 'w', template=template)
print storage.save(engine)
print storage.save(engine_hot)

#
platform = engine.simulation.context.getPlatform()
print platform.getName()
platform = engine_hot.simulation.context.getPlatform()
print platform.getName()

# Equilibrate
engine.current_snapshot = template
engine.minimize()
initial_snapshot_cool = engine.current_snapshot

engine_hot.current_snapshot = template
engine_hot.minimize()
initial_snapshot_hot = engine_hot.current_snapshot

storage.tag['cool_template'] = engine.current_snapshot
storage.tag['hot_template'] = engine_hot.current_snapshot

# Define order parameters
# Compute the closest heavy atom distances between receptor and ligand
# TODO: How do we restrict this to only residues 0 and 1?
print(template.topology)
compute_contacts = paths.MDTrajFunctionCV(
    name="compute_contacts",
    cv_scalarize_numpy_singletons=False, # needed because compute_contacts() does not return a single numpy array
    contacts=[[0,1]],
    f=md.compute_contacts,
    topology=template.topology,
    scheme='closest-heavy',
    ignore_nonprotein=False,
    periodic=True
).with_diskcache()
storage.save([compute_contacts])

# Create CV states for bound and unbound
def compute_cv(snapshot, center, compute_contacts):
    [distances, residue_pairs] = compute_contacts(snapshot)
    return distances[0] / unit.angstroms

# State definitions
states = ['bound', 'unbound']

state_centers = {
    'bound' : 0.0,
    'unbound' : 10.0,
}

interface_levels = {
    'bound' : np.array([0.0, 10.0]),
    'unbound' : np.array([0.0, 10.0])
}

cv_state = dict()
for state in state_centers:
    op = paths.FunctionCV(
        name = 'op' + state,
        f=compute_cv,
        center=state_centers[state],
        compute_contacts=compute_contacts
    )
    cv_state[state] = op

# Create interfaces
interface_list = {}
for state in interface_levels:
    levels = interface_levels[state]
    interface_list[state] = [None] * len(levels)
    for idx, level in enumerate(levels):
        interface_list[state][idx] = paths.CVRangeVolume(cv_state[state], lambda_max=level)
        interface_list[state][idx].name = state + str(idx)

vol_state = {state : interface_list[state][0] for state in interface_list}
stAll = reduce(lambda x, y: x | y, [vol_state[state] for state in states])
stAll.name = 'all'
storage.save(stAll)

# Initial core
paths.EngineMover.engine = engine_hot

reach_core = paths.SequentialEnsemble([
        paths.OptionalEnsemble(paths.AllOutXEnsemble(stAll)),
        paths.SingleFrameEnsemble(
            paths.AllInXEnsemble(stAll)
        )
    ])

core2core = paths.SequentialEnsemble(
    [
        paths.OptionalEnsemble(paths.AllInXEnsemble(stAll)),
        paths.AllOutXEnsemble(stAll),
        paths.SingleFrameEnsemble(
            paths.AllInXEnsemble(stAll)
        )
    ]
)

# Generate initial trajectory
init_traj = engine_hot.generate(template, [reach_core.can_append])
state_information(init_traj[-1])
