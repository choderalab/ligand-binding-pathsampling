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
    'n_frames_max': 1000,
    'platform': 'CPU',
    'n_steps_per_frame': 50
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
integrator_hot = openmm.LangevinIntegrator(1800*unit.kelvin, 10.0/unit.picoseconds, 2.0*unit.femtoseconds)
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
cv = paths.MDTrajFunctionCV(
    name="compute_contacts",
    cv_scalarize_numpy_singletons=False, # needed because compute_contacts() does not return a single numpy array
    contacts=[[0,1]],
    f=md.compute_contacts,
    topology=template.topology,
    scheme='closest-heavy',
    ignore_nonprotein=False,
    periodic=True
).with_diskcache()
storage.save([cv])

# Create CV states for bound and unbound
def compute_cv(snapshot, center, compute_contacts):
    from simtk import unit
    distances = cv(snapshot)
    distance = distances[0][0] * 10 # convert from nanometers to angstroms
    #print('%8.3f' % distance)
    return distance

# State definitions
states = [
    'bound  ',
    'unbound']

max_bound   = 3.0
min_unbound = 7.0

print('Creating interfaces...')
ninterfaces = 30
# note that we just changed CVRangeVolume => CVDefinedVolume, but you haven't updated that yet
bound = paths.CVRangeVolume(cv, lambda_min=0.0, lambda_max=max_bound)
unbound = paths.CVRangeVolume(cv, lambda_min=min_unbound, lambda_max=float("inf"))
interfaces = paths.VolumeInterfaceSet(cv, minvals=0.0, maxvals=np.linspace(3.1, 6.9, ninterfaces))

# Create a network to study unbinding paths
# note the list/tuple structure: that's because this is normally a list of tuples,
# each tuple representing a transition to study
print('Creating network...')
network = paths.MISTISNetwork([(bound, interfaces, unbound)])

# generate high-temperature trajectory
print('Generating high-temperature trajectory...')
ensemble = paths.AllOutXEnsemble(bound) | paths.AllOutXEnsemble(unbound)
long_trajectory = engine_hot.generate(initial_snapshot_hot, [ensemble])

# split out the subtrajectory of interest
tmp_network = paths.TPSNetwork(bound, unbound)
short_trajectory = tmp_network.ensembles[0].split(long_trajectory)

bootstrap = FullBootstrapping(
    transition=network.transitions[(bound, unbound)],
    snapshot=initial_snapshot,
    engine=engine
)

#
# OLD
#

total_sample_set = paths.SampleSet.relabel_replicas_per_ensemble(
    initial_bootstrap.values()
)

total_sample_set.sanity_check()

storage.save(total_sample_set)
storage.sync_all()
