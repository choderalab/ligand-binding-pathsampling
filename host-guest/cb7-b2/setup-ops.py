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
platform_name = 'CPU'
platform = openmm.Platform.getPlatformByName(platform_name)
properties = {'OpenCLPrecision': 'mixed'}

# Create an engine
print('Creating engine...')
engine_options = {
    'n_frames_max': 100,
    'platform': platform_name,
    'n_steps_per_frame': 250
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
integrator_hot = openmm.LangevinIntegrator(2000*unit.kelvin, 10.0/unit.picoseconds, 1.0*unit.femtoseconds)
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
    periodic=False
).with_diskcache()
storage.save([cv])

# State definitions
states = [
    'bound  ',
    'unbound']

max_bound   = 0.35 # nanometers, maximum bound state separation distance
min_unbound = 0.70 # nanometers, minimum unbound state separation distance

print('Creating interfaces...')
ninterfaces = 30
# CVDefinedVolume?
bound = paths.CVRangeVolume(cv, lambda_min=0.0, lambda_max=max_bound)
unbound = paths.CVRangeVolume(cv, lambda_min=min_unbound, lambda_max=float("inf"))
interfaces = paths.VolumeInterfaceSet(cv, minvals=0.0, maxvals=np.linspace(max_bound+0.01, min_unbound-0.01, ninterfaces))

print('Creating network...')
mistis = paths.MISTISNetwork([(bound, interfaces, unbound)])

initial_trajectory_method = 'high-temperature'
if initial_trajectory_method == 'high-temperature':
    # We are starting in the bound state, so
    # generate high-temperature trajectory that reaches the unbound state
    print('Generating high-temperature trajectory...')
    #ensemble = not (paths.ExitsXEnsemble(bound) & paths.EntersXEnsemble(unbound))
    unbinding_ensemble = paths.AllOutXEnsemble(unbound)
    bridging_ensemble = paths.AllOutXEnsemble(bound) & paths.AllOutXEnsemble(unbound)
    initial_trajectories = list()
    tmp_network = paths.TPSNetwork(bound, unbound)
    attempt = 0
    while len(initial_trajectories) == 0:
        print('Attempt %d' % attempt)
        long_trajectory = engine_hot.generate(initial_snapshot_hot, [unbinding_ensemble])
        print('long trajectory:')
        print(long_trajectory)
        distances = np.array([ cv(snapshot)[0][0] for snapshot in long_trajectory ])
        print(distances)
        # split out the subtrajectory of interest
        initial_trajectories = tmp_network.all_ensembles[0].split(long_trajectory)
        print('initial trajectories:')
        print(initial_trajectories)
        distances = np.array([ cv(snapshot)[0][0] for snapshot in initial_trajectories[0] ])
        print(distances)
        attempt += 1
        print('')

elif initial_trajectory_method == 'bootstrap':
    print('Bootstrapping initial trajectory...')
    bootstrap = paths.FullBootstrapping(
        transition=mistis.transitions[(bound, unbound)],
        snapshot=initial_snapshot_hot,
        engine=engine_hot
    )
    initial_sample_set = bootstrap.run()
    initial_trajectories = [s.trajectory for s in initial_sample_set]
    print(initial_trajectories)
else:
    raise Exception('initial trajectory method "%s" unknown' % initial_trajectory_method)

# Create a network to study unbinding paths
# note the list/tuple structure: that's because this is normally a list of tuples,
# each tuple representing a transition to study
scheme = paths.DefaultScheme(mistis, engine=engine)
sset = scheme.initial_conditions_from_trajectories(initial_trajectories)
print scheme.initial_conditions_report(sset)

# Populate minus ensemble
print('Populating the minus ensemble')
minus_samples = []
for minus in mistis.minus_ensembles:
    samp = minus.populate_minus_ensemble_from_set(
        samples=sset,
        minus_replica_id=-mistis.minus_ensembles.index(minus)-1,
        engine=engine
    )
    minus_samples.append(samp)

sset = sset.apply_samples(minus_samples)

print('Running MISTIS')
mistis_calc = paths.PathSampling(
    storage=storage,
    move_scheme=scheme,
    sample_set=sset
)
mistis_calc.save_frequency = 100

import logging.config
logging.config.fileConfig("logging.conf", disable_existing_loggers=False)
mistis_calc.run(400)
