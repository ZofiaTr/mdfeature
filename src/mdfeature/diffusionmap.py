"""
.. module:: diffusionmap
This module allows computation of the diffusionmaps.
.. moduleauthor:: ZofiaTr
"""

from simtk.openmm import *
from simtk.openmm.app import *
from simtk.unit import *
from matplotlib.pyplot import cm
import mdtraj as md
import time
import numpy as np
from openmmtools.constants import kB
from sklearn.neighbors.kde import KernelDensity

import pydiffmap.diffusion_map as dfm

def energy(state, simulation):
    """
    Evaluate energy from the current state.

    :param openmm.positions state: the current state
    :param openmm.Simulation simulation: simulation object
    """
    simulation.context.setPositions(state)
    return simulation.context.getState(getEnergy=True).getPotentialEnergy()

def compute_energy(xyz, simulation, positions_unit, energy_unit):
    """
    Compute energy from the trajectory.

    :param ndrarray xyz: trajectory
    :param openmm.Simulation simulation: simulation object
    :param positions_unit: simtk.unit of the positions
    :param energy_unit: simtk.unit of the energy

    :return: numpy ndarray of energy for every trajectory frame.
    """

    Erecompute = np.zeros(len(xyz))

    for i in range(0,len(X_FT)):
                Erecompute[i]=energy(X_xyzFT[i]*positions_unit, simulation).value_in_unit(energy_unit)

    return  Erecompute

def compute_target_measure(energy, kT, energy_unit):
    """
    Helper function to compute Boltzman density from energy and temperature.

    :param numpy.ndarray energy: energy value,  shape number of frames
    :param numpy.ndarray kT: kT factor, computed as kT = openmmtools.constants.kB * simtk.unit.kelvin * temperature, where temperature is a double
    """

    qTargetDistribution = np.zeros(len(energy))

    for i in range(0,len(xyz)):
                betatimesH_unitless = np.abs(energy[i]) / kT.value_in_unit(energy_unit)
                qTargetDistribution[i] = np.exp(-(betatimesH_unitless))

    return qTargetDistribution

def compute_diffusionmaps(traj_orig, nrpoints=None, epsilon='bgh', nrneigh=64, weights=None, weight_params={}):
    """
    Compute diffusionmaps using pydiffmap.

    :param mdtraj.Trajectory traj_orig: trajectory for diffusionmap analysis
    :param nrpoints: if None, keep all trajectory, if integer, subsample the trajectory leaving nrpoints of datapoints.
    :param epsilon: epsilon parameter in diffusionmap construction.
    :param int nrneigh: number of neighbors in diffusionmap construction.
    :param weights: weights for TMDmap. If None, no TMDmap but vanilla diffusionmap, if 'compute', weights will be computed using the energy and it requires the dictionary weight_params to be nonempty

    :rtype: pydiffmap.diffusion_map.DiffusionMap, mdtraj.Trajectory
    """

    # computation of weights for tmdmap
    if weights == 'compute':

        raise NotImplemented

        simulation = weight_params['simulation']

        positions = simulation.context.getState(getPositions=True).getPositions()
        energy_unit = energy(positions, simulation).unit
        positions_unit = positions.unit

        T = weight_params['temperature']
        kT = kB * T * kelvin

        print('TMD map not implemented yet, weights wont be used')
        E = compute_energy(traj.xyz, simulation, positions_unit, energy_unit)
        print('Energy has shape')
        print(E.shape)

        qTargetDistribution= compute_target_measure(E, kT, energy_unit)
        weights =  qTargetDistribution

    # subsampling
    landmark_indices = np.random.choice(np.arange(len(traj_orig.xyz)), size=nrpoints)

    traj = md.Trajectory(traj_orig.xyz[landmark_indices], traj_orig.topology)
    traj = traj.superpose(traj[0])

    print(traj_orig)
    print('Subsampled to')
    print(traj)

    Xresh = traj.xyz.reshape(traj.xyz.shape[0], traj.xyz.shape[1]*traj.xyz.shape[2])

    if weights is None:
        mydmap = dfm.DiffusionMap.from_sklearn(epsilon = epsilon, alpha = 0.5, k=nrneigh, kernel_type='gaussian', n_evecs=5, neighbor_params=None,
                     metric='euclidean', metric_params=None, weight_fxn=None, density_fxn=None, bandwidth_type="-1/(d+2)",
                     bandwidth_normalize=False, oos='nystroem')
    else:
        raise NotImplemented

    mydmap.fit(Xresh)
    traj = md.Trajectory(Xresh.reshape(traj.xyz.shape[0], traj.xyz.shape[1], traj.xyz.shape[2]), traj_orig.topology)

    return mydmap, traj
