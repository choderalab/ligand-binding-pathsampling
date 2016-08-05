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
# TODO: How do we restrict this to only residues 0 and 1?
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
    from simtk import unit
    distances = compute_contacts(snapshot)
    distance = distances[0][0] * 10 # convert from nanometers to angstroms
    #print('%8.3f' % distance)
    return distance

# State definitions
states = [
    'bound  ',
    'unbound']

state_centers = {
    'bound  ' : 3.0,
    'unbound' : 7.0,
}

ninterfaces = 30
print('There are %d interfaces per state')
interface_levels = {
    'bound  ' : np.linspace(3.1, 5.0, ninterfaces),
    'unbound' : np.linspace(5.1, 6.9, ninterfaces)
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
print('Creating interface list...')
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
print('Define initial core...')
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
print('Generate initial trajectory...')
init_traj = engine_hot.generate(template, [reach_core.can_append])

# State information utility functions
def get_state(snapshot):
    for state in states:
        if interface_list[state][-1](snapshot):
            return state

    return '-'

def get_core(snapshot):
    for state in states:
        if interface_list[state][0](snapshot):
            return state

    return '-'

def get_interface(snapshot):
    for state in states:
        for idx, interface in enumerate(interface_list[state]):
            if interface(snapshot):
                return state, idx

    return '-', 0

def state_information(snapshot):
    for state in states:
        cv = cv_state[state]
        print '%s: %7.2f' % (state, cv(snapshot)),
        for idx, interface in enumerate(interface_list[state]):
            if interface(snapshot):
                print '+',
                current_state = state
            else:
                print '-',

        print

    print 'The initial configuration is in state', current_state
state_information(init_traj[-1])

# Generate intercore trajectories
print('Generate intercore trajectories...')
tt = init_traj
found_states = set()
missing_states = set(states)
data = np.array([[cv_state[state](frame) for state in states] for frame in tt])
found_states.update([state for state in missing_states if any(map(interface_list[state][0], tt))])
missing_states = missing_states - found_states
storage.save(tt)
initials = {state : list() for state in states}

chunksize = 5
first = True
count = 0
while len(missing_states) > 0 or first:
    first = False
    while True:
        oldtt = tt
        try:
            count += 1
            try:
                tt = engine_hot.generate_forward(tt[-1].reversed, core2core)
                storage.save(tt)

                for state in states:
                    if vol_state[state](tt[0]):
                        initials[state].append(tt[0])
                    if vol_state[state](tt[-1]):
                        initials[state].append(tt[-1].reversed)

                found_states.update([state for state in missing_states if any(map(vol_state[state], tt[[0,-1]]))])
                missing_states = missing_states - found_states
                paths.tools.refresh_output(
                    '[%4d] %4d  %s < %4d > %s   still missing states %s\n' % (
                        count,
                        len(storage.trajectories),
                        get_state(tt[0]),
                        len(tt) - 2,
                        get_state(tt[-1]),
                        ''.join(missing_states)
                    )
                )

                if len(storage.trajectories) % chunksize == 0:
                    break
            except ValueError:
                paths.tools.refresh_output(
                    '[%4d] %4d  %s < ERROR > ??   still missing states %s\n' % (
                        count,
                        len(storage.trajectories),
                        get_state(tt[0]),
                        ''.join(missing_states)
                    )
                )

        except Exception as e:
            # NaN exception?
            if str(e) == 'Particle coordinate is nan':
                print(e)
                tt = oldtt
            else:
                raise(e)

paths.tools.refresh_output('DONE!\n')

# Set up MSTIS
print('Set up MSTIS')
mstis = paths.MSTISNetwork([
    (
        vol_state[state],
        interface_list[state][0:],
        cv_state[state]
    ) for state in states
])

storage.save(mstis)

initial_bootstrap = dict()

for state in states:
    while state not in initial_bootstrap:
        try:
            paths.tools.refresh_output(
                'Bootstrapping state ' + state
            )
            from_state = vol_state[state]
            bootstrapA = paths.FullBootstrapping(
                transition=mstis.from_state[from_state],
                snapshot=random.choice(initials[state]),
                engine=engine,
                extra_interfaces=[interface_list[state][-1]],
                initial_max_length=20
            )
            paths.tools.refresh_output(
                'Bootstrapping state ' + state + " - let's go"
            )
            final = bootstrapA.run(max_ensemble_rounds=2, build_attempts=5)
            if all([s.ensemble(s) for s in final]):
                initial_bootstrap[state] = final
        except ValueError:
            paths.tools.refresh_output(
                'Encountered NaN'
            )
        except RuntimeError:
            paths.tools.refresh_output(
                'Too many attempts. Start retry.'
            )

paths.tools.refresh_output(
    'DONE!'
)

total_sample_set = paths.SampleSet.relabel_replicas_per_ensemble(
    initial_bootstrap.values()
)

total_sample_set.sanity_check()

storage.save(total_sample_set)
storage.sync_all()
