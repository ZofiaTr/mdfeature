"""
.. module:: features
This module contains functions facilitating feature selection.

.. moduleauthor:: ZofiaTr
"""

from simtk.openmm import *
from simtk.openmm.app import *
from simtk.unit import *
from matplotlib.pyplot import cm
import mdtraj as md
import time
import numpy as np
from scipy import stats
from itertools import combinations
from ipywidgets import IntProgress

def compute_torsion(coordinates, i, j, k, l):
    """
    Compute torsion angle defined by four atoms.

    ARGUMENTS

    coordinates (simtk.unit.Quantity wrapping numpy natoms x 3) - atomic coordinates
    i, j, k, l - four atoms defining torsion angle

    NOTES

    Algorithm of Swope and Ferguson [1] is used.

    [1] Swope WC and Ferguson DM. Alternative expressions for energies and forces due to angle bending and torsional energy.
    J. Comput. Chem. 13:585, 1992.

    """
    # Swope and Ferguson, Eq. 26
    rij = (coordinates[i] - coordinates[j]) / nanometers
    rkj = (coordinates[k] - coordinates[j]) / nanometers
    rlk = (coordinates[l] - coordinates[k]) / nanometers
    rjk = (coordinates[j] - coordinates[k]) / nanometers # added

    # Swope and Ferguson, Eq. 27
    t = np.cross(rij, rkj)
    u = np.cross(rjk, rlk) # fixed because this didn't seem to match diagram in equation in paper

    # Swope and Ferguson, Eq. 28
    t_norm = np.sqrt(numpy.dot(t, t))
    u_norm = np.sqrt(numpy.dot(u, u))

    cos_theta = np.dot(t, u) / (t_norm * u_norm)

    theta = np.arccos(cos_theta) * np.sign(np.dot(rkj, np.cross(t, u)).value_in_unit(nanometer**-5))
    theta = math.degrees(theta)*degree
    return theta


def compute_torsion_mdraj(traj, angle_index):
    """
    Compute torsion angle defined by four atoms.

    ARGUMENTS

    traj (mdtraj trajectory frame) - atomic coordinates
    i, j, k, l - four atoms defining torsion angle

    NOTES

    Algorithm of Swope and Ferguson [1] is used.

    [1] Swope WC and Ferguson DM. Alternative expressions for energies and forces due to angle bending and torsional energy.
    J. Comput. Chem. 13:585, 1992.

    """

    torsions = []
    i, j, k, l = angle_index

    for x in traj:
        coordinates = x.xyz.squeeze()


        # Swope and Ferguson, Eq. 26
        rij = (coordinates[i] - coordinates[j])
        rkj = (coordinates[k] - coordinates[j])
        rlk = (coordinates[l] - coordinates[k])
        rjk = (coordinates[j] - coordinates[k])

        # Swope and Ferguson, Eq. 27
        t = np.cross(rij, rkj)
        u = np.cross(rjk, rlk) # fixed because this didn't seem to match diagram in equation in paper

        # Swope and Ferguson, Eq. 28
        t_norm = np.sqrt(numpy.dot(t, t))
        u_norm = np.sqrt(numpy.dot(u, u))

        cos_theta = np.dot(t, u)/ (t_norm * u_norm)
        theta = np.arccos(cos_theta) * np.sign(np.dot(rkj, np.cross(t, u)))

        torsions.append(theta)

    return torsions

def compute_cos_torsion_mdraj(traj, angle_index):
    """
    Return cos of torsions.
    """
    return np.cos(compute_torsion_mdraj(traj, angle_index))



def compute_free_energy(cv1, cv2, bins=40):
    """
    Free energy in 2D.
    """
    FE, xx, yy = np.histogram2d(cv1, cv2, bins=bins)
    FE =  - np.log(FE).T

    xx = 0.5*(xx[1:] + xx[:-1])
    yy = 0.5*(yy[1:] + yy[:-1])

    return FE, xx, yy

def get_name(element_idx, table=None, pdb_file=None, format='full'):
    """
    Given index of one atom, return string of the element name in the required format

    :param  int element_idx: index of one element
    :param dict table (optional): table returned by table, bonds = mdtrajectory.topology.to_dataframe()
    :param str pdb_file (optional, required for table): string containing path to the topology file (pdb)
    :param str format: format of the name string - 'full' default is resName + resSeq + element, 'short'is resSeq + element
    """
    if table is None:
        topology = md.load(pdb_file).topology
        table, bonds = topology.to_dataframe()

    if format == 'full':
        return table.values[element_idx][4] + str(table.values[element_idx][3]) + table.values[element_idx][2]
    elif format == 'short':
        return str(table.values[element_idx][3]) + table.values[element_idx][2]
    else:
        raise

def get_name_torsion(indices, table=None, pdb_file=None, format='full'):
    """
    Given indices of the atom, return string of atom names in the format resName + resSeq + element

    :param  list indices: list of ints indices of 4 atoms used for torsions
    :param dict table: table returned by table, bonds = mdtrajectory.topology.to_dataframe()
    :param str pdb_file: string containing path to the topology file (pdb)
    :param str format: format of the name string - 'full' default is resName + resSeq + element, 'short'is resSeq + element
    """

    if format == 'full':
        return get_name(indices[0], pdb_file=pdb_file, table=table) + '-'\
        +get_name(indices[1], pdb_file=pdb_file, table=table)+'-'\
        +get_name(indices[2], pdb_file=pdb_file, table=table)+'-'\
        +get_name(indices[3], pdb_file=pdb_file, table=table)

    elif format == 'short':
        return get_name(indices[0], pdb_file=pdb_file, table=table, format='short')+ '-'\
        +get_name(indices[1], pdb_file=pdb_file, table=table, format='short') + '-'\
        +get_name(indices[2], pdb_file=pdb_file, table=table, format='short') + '-'\
        +get_name(indices[3], pdb_file=pdb_file, table=table, format='short')
    else:
        raise

def compute_cv(X_FT, top, cv_indices_chosen):
    traj_or = md.Trajectory(X_FT, top)
    torsion_max = compute_torsion_mdraj(traj_or, cv_indices_chosen[0])
    torsion_max_1 = compute_torsion_mdraj(traj_or, cv_indices_chosen[1])
    return np.vstack([torsion_max, torsion_max_1]).T

def compute_correlation(v, cv, coeff='pearson', minmax=False):
    """
    Correlation coefftients.

    :param ndarray v:  eigenvector size(n, 1)
    :param ndarray cv: feature size(n, 1)
    :param str coeff: pearson or spearmanr, see scipy.stats
    :param minmax: renormalize the values to be in [0,1]
    """
    if minmax:
        v = (v - np.min(v))/(np.max(v)-np.min(v))
        cv = (cv - np.min(cv))/(np.max(cv)-np.min(cv))

    if coeff == 'pearson':
        return stats.pearsonr(v, cv)[0]
    elif coeff == 'spearmanr':
        return stats.spearmanr(v, cv)[0]
    else:
        raise

def correlations_features(evec, traj, list_of_functions, list_of_params=None, progress_bar=True):
    """
    Compute correlations between the eigenvector and a list of functions evaluated on the trajectory, optionally with some parameters taken from list_of_params.

    :param evec: diffmap eigenvector
    :param traj: trajectory (mdtraj)
    :param list_of_functions: list of functions
    :param list_of_params (optional): list of list_of_params. If None, then functions will be called without parameters. If list_of_params is list, then the length of list_of_functions must equal to list_of_params.

    :return: numpy array of correlations of length of the function list.
    """
    correlations = []

    if progress_bar:
        max_count = len(list_of_functions)
        bar = IntProgress(min=0, max=max_count) # instantiate the bar
        display(bar) # display the bar

    if list_of_params is None:
        for funcs in list_of_functions:
            correlations.append(compute_correlation(evec, eval(funcs +'(traj)')))

            if progress_bar:
                bar.value += 1
    else:
        for funcs, pars in zip(list_of_functions, list_of_params):
            correlations.append(compute_correlation(evec, eval(funcs +'(traj, pars)')))

            if progress_bar:
                bar.value += 1

    return np.array(correlations)


def correlations_cv_torsion_index(evec, traj, list_of_indices):
    """
    evec: diffmap eigenvector
    traj: trajectory (mdtraj)
    list_of_indices: list of indices to compute torsions from
    """

    max_count = len(list_of_indices)
    bar = IntProgress(min=0, max=max_count) # instantiate the bar
    display(bar) # display the bar

    correlations = []
    indices_cv = []
    for indx in list_of_indices:

        bar.value += 1

        #print('Torsion for '+str(indx))
        corr_cos = compute_correlation(evec, np.cos(compute_torsion_mdraj(traj, indx)))

        #if np.sum(~np.isnan(corr_cos)):
        indices_cv.append(indx)
        correlations.append(corr_cos)

    return np.array(correlations)#, list_of_indices[indices_cv]

def create_torsions_list(atoms, size=None, print_list=True, append_to=None):
    """
    Create list of indices for torsion angles.

    :param atoms: if np.int - number of atoms to create the combinations from. If list of ints, combinations of these indeces will be generated.
    :param size: number of randomly selected quadruples. If size is None, select all, if int, randomly choose int many combinations.
    :param bool print_list: print the generated list.
    :param list append_to: append to an existing list.
    """
    if type(atoms) == np.int:
        atoms_indices = range(0, atoms)
    else:
        atoms_indices = atoms

    all_possible_combinations = np.array([c for c in combinations(atoms_indices, 4)])

    if size is None:
        chosen_indices = range(len(all_possible_combinations))
    else:
        chosen_indices = np.random.choice(range(len(all_possible_combinations)), size=size)

    if append_to is not None:
        all_combinations = np.vstack((append_to, all_possible_combinations[chosen_indices]))
    else:
        all_combinations = all_possible_combinations[chosen_indices]

    if print_list:
        print(all_combinations)

    return all_combinations

def get_index_combinations(all_combinations, index):
    """
    assume unique
    """
    for i in range(len(all_combinations)):
        if (index == all_combinations[i]).all():
            return i

    #print(get_index_combinations(all_combinations, all_combinations[5]))

def compute_all_correlations(traj, mydmap, dimension, list_of_functions, nevery=1, list_of_params=None, progress_bar=True):
    """
    Calls features.correlations_features for first 'dimension' eigenvectors.

    :param mdtraj.Trajectory traj: trajectory to evaluate the features from
    :param pydiffmap.diffusion_map.Diffusionmap mydmap: diffusionmap object
    :param list_of_functions: list of functions computing the features
    :param nevery: subsample the trajectory and eigenvectors as x[::nevery]
    :param dimension: number of eigenvectors
    :param int dimension: number of diffusionmap eigenvectors
    :param nevery: subsample the trajctory as traj[::nevery] to accelerate the correlation computations

    :return: dict correlations with the dimension as keys for return of features.correlations_features
    """

    correlations = {}

    for ev_idx in range(dimension):
        print('Dimension '+repr(ev_idx + 1))

        evec = mydmap.evecs[::nevery, ev_idx]
        correlations[ev_idx] = correlations_features(evec, traj[::nevery], list_of_functions, list_of_params=list_of_params, progress_bar=progress_bar)

    return correlations


def identify_features(correlations, all_combinations, remove_nans=False):
    """
    Choose two most correlated features.

    :param correlations: returned by features.compute_all_correlations
    :param all_combinations: parameters of the features (for example indices for the torsions)
    :param remove_nans: search only non nans features

    :return: features_indices - index of the best feature from the function list, correlations_features - the corresponding correlation, features_indices_2 - second best feature, correlations_features_2
    """

    cv_indices = []
    cv_indices_2 = []
    correlations_cv = []
    correlations_cv_2 = []

    corr = {}
    dimension = len(correlations)

    for ev_idx in range(dimension):

        if remove_nans:
            correlations_clean_idx = np.array([i for i in range(len(correlations[ev_idx])) if str(correlations[ev_idx][i]) != 'nan'])
        else:
            correlations_clean_idx = range(len(correlations[ev_idx]))

        correlations_clean = correlations[ev_idx][correlations_clean_idx]

        print('Number of cvs:')
        print(len(correlations_clean))
        all_combinations_clean = all_combinations[correlations_clean_idx]

        assert (len(correlations_clean)) == len(all_combinations_clean)

        sorted_indeces_clean = np.argsort(np.abs(correlations_clean))[::-1]
        #print(all_combinations_clean[sorted_indeces_clean])

        print('Maximal cv:')
        print('index:')
        print(all_combinations_clean[sorted_indeces_clean[0]])
        print('corr:')
        print(correlations_clean[sorted_indeces_clean[0]])

        print('Second naximal cv:')
        print('index:')
        print(all_combinations_clean[sorted_indeces_clean[1]])
        print('corr:')
        print(correlations_clean[sorted_indeces_clean[1]])

        correlations_cv.append(correlations_clean[sorted_indeces_clean[0]])
        cv_indices.append(all_combinations_clean[sorted_indeces_clean][0])

        cv_indices_2.append(all_combinations_clean[sorted_indeces_clean][1])
        correlations_cv_2.append(correlations_clean[sorted_indeces_clean[1]])

        corr[ev_idx] = correlations_clean

    return cv_indices, correlations_cv, cv_indices_2, correlations_cv_2


def identify_worst_features(corr, all_combinations, dimension):
    """
    Least correlated features, see identify_features for input and output.
    """

    corr_arr = np.array((corr[0], corr[1])).T
    print(corr_arr.shape)
    cv_indices_minimal = []

    eps = 0
    inter = []
    while len(inter) < dimension and eps < 1:
        inter = np.intersect1d(np.where(np.abs(corr_arr[:,0]) < eps)[0], np.where(np.abs(corr_arr[:,1]) < eps)[0])
        print(inter)
        eps += 0.01

    cv_indices_minimal = [all_combinations[inter[0]], all_combinations[inter[1]]]

    correlations_cv_minimal= np.array((corr[0], corr[1])).T[inter]#np.array((correlations[0], correlations[1])).T[inter]
    print('\n\n\Minimal correlated collective variables:')
    print(cv_indices_minimal)
    print(cv_indices_minimal[1].tolist())
    print('Corr:')
    print(correlations_cv_minimal)

    return cv_indices_minimal, correlations_cv_minimal


def get_angle_name(phi_adapt, psi_adapt):

    angle_name_1 = str(phi_adapt[0])+'_'+str(phi_adapt[1])+'_'+str(phi_adapt[2])+'_'+str(phi_adapt[3])
    angle_name_2 = str(psi_adapt[0])+'_'+str(psi_adapt[1])+'_'+str(psi_adapt[2])+'_'+str(psi_adapt[3])

    return angle_name_1 +'-'+ angle_name_2
