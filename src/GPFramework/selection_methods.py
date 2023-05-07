"""
Programmed by Jason Zutty
Modified by VIP Team
Implements a number of population selection methods for use with deap
"""
import numpy as np
import sys
import random
import sklearn.covariance
import scipy.special
import gzip
import time
import multiprocessing
import functools
from deap import tools
import networkx as nx
from networkx.algorithms import approximation
from itertools import chain
from heapq import heappush, heappop
import math
import sklearn.neighbors
sys.path.insert(1, '/home/cameron/Desktop/emade/src')
import GPFramework.constants

def selTournamentDCD(individuals, k, acceptable, goal, penalty_factor=1.):
    """
    Taken from DEAP as a starting point
    Tournament selection based on dominance (D) between two individuals, if
    the two individuals do not interdominate the selection is made
    based on crowding distance (CD). The *individuals* sequence length has to
    be a multiple of 4. Starting from the beginning of the selected
    individuals, two consecutive individuals will be different (assuming all
    individuals in the input list are unique). Each individual from the input
    list won't be selected more than twice.

    This selection requires the individuals to have a :attr:`crowding_dist`
    attribute, which can be set by the :func:`assignCrowdingDist` function.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :returns: A list of selected individuals.
    """
    def tourn(ind1, ind2):
        if ind1.fitness.dominates(ind2.fitness):
            return ind1
        elif ind2.fitness.dominates(ind1.fitness):
            return ind2

        if ind1.fitness.crowding_dist < ind2.fitness.crowding_dist:
            return ind2
        elif ind1.fitness.crowding_dist > ind2.fitness.crowding_dist:
            return ind1

        if random.random() <= 0.5:
            return ind1
        return ind2

    # Let's start by extracting our objective scores from the population of individuals
    population = np.array([individual.fitness.values for individual in individuals])
    # Now we will create a weights object which will eventually correspond to a probability of mating
    weights = np.ones((len(population,)))
    # Iterate through population to compute weights for each individual
    for i in np.arange(len(population)):
        difference_to_acceptable = population[i] - acceptable
        difference_to_goal = population[i] - goal
        # Let's now add a penalty for all objectives that are greater than the acceptable level
        weights[i] += penalty_factor*np.linalg.norm(difference_to_acceptable[difference_to_acceptable > 0])
        # And a similar penalty for all objectives that are greater than a goal level without the relative scale factor
        weights[i] += np.linalg.norm(difference_to_goal[difference_to_goal > 0])
    # Invert so higher values are now worse
    weights = 1.0/weights
    # If all of the weights are zero, the normalization will return nans
    # We will just reset everything to one
    if np.all(weights == 0):
        weights = np.ones((len(population,)))
    # Now let's get a relative performance by normalizing
    weights /= np.sum(weights)
    # Set weight zero to a very small value instead, this occurs when objective scores are infs
    weights[weights == 0] = np.finfo(float).eps

    chosen = []
    for i in range(0, k, 4):
        # Let's now take two random samples of the data, from a biased selection method
        try:
            individuals_1_index = np.random.choice(np.arange(len(individuals)), 4, replace=False, p=weights)
        except Exception as e:
            print(np.sum(weights), weights)
            raise e
        individuals_2_index = np.random.choice(np.arange(len(individuals)), 4, replace=False, p=weights)

        # We need to operate this way due to the fact that individuals is a list of lists. (Because of ADFs)
        individuals_1 = list(individuals[j] for j in individuals_1_index)
        individuals_2 = list(individuals[j] for j in individuals_2_index)

        chosen.append(tourn(individuals_1[0],   individuals_1[1]))
        chosen.append(tourn(individuals_1[2], individuals_1[3]))
        chosen.append(tourn(individuals_2[0],   individuals_2[1]))
        chosen.append(tourn(individuals_2[2], individuals_2[3]))

    return chosen

def sel_nsga2(individuals, k):
    # NSGA2 algorithm first calls for an assigning of crowding distances handled by selNSGA2
    # This has actually already been done in the main loop
    #sorted_pop = tools.selNSGA2(individuals, k)
    # Next use a binary tournament with ties broken by crowding distance to select the pop
    selected_pop = tools.selTournamentDCD(individuals, k)
    return selected_pop

def sel_nsga2_weighted(individuals, k, penalty_factor, dynamic_args):
    """
    Uses modified selTournamentDCD which scales prob of selection using thresholds
    """
    # NSGA2 algorithm first calls for an assigning of crowding distances handled by selNSGA2
    # This has actually already been done in the main loop
    #sorted_pop = tools.selNSGA2(individuals, k)
    # Next use a binary tournament with ties broken by crowding distance to select the pop
    acceptable = dynamic_args['acceptable']
    goal = dynamic_args['goal']
    selected_pop = selTournamentDCD(individuals, k, acceptable, goal, penalty_factor)
    return selected_pop

def sel_nsga2_novelty(individuals, k, n, dynamic_args):
    """
    This function computes a novelty score.
    It then recomputes nsga2 crowding distances.
    Finally it performs a binary tournament on included objectives + novelty.
    The dynamic_args expects a pset
    """
    # Psets will match the number of trees in each individual.
    psets = dynamic_args['psets']
    datasetDict = dynamic_args['datasetDict']
    primitive_lists = []
    for pset in psets:
        primitive_list = []
        for key in pset.primitives:
            # Iterates by output types through the primitives an builds one large vector
            primitive_list += [primitive.name for primitive in pset.primitives[key]]
        primitive_lists.append(np.array(primitive_list))

    histograms = []
    for individual in individuals:
        ind_hist = []
        for tree, primitive_list in zip(individual, primitive_lists):
            hist = np.sum([elem.name == primitive_list for elem in tree], axis=0)
            ind_hist.append(hist)
        # Now we need to concatenate together the adf hists
        ind_hist = np.hstack(ind_hist)
        #print('Histogram shape', ind_hist.shape)
        histograms.append(ind_hist)
    histograms = np.array(histograms)
    # I am happy with this print so I am commenting it out
    # print('All histograms shape', histograms.shape)
    # Now that we have these large sparse vectors as a behavior space, we need to compute a novelty score
    # as the distance to the nth nearest neighbor
    neighbors = sklearn.neighbors.BallTree(histograms)
    # We use n+1 for the neighbors, because the original point is included
    distances, indices = neighbors.query(histograms, n+1)
    # Because this is sorted we can now just take the last distance for each individual
    # as the novelty score
    novelties = distances[:,-1]
    # I am happy with this print, so I am commenting it out
    # print('NOVELTIES', novelties, novelties.shape)
    offset = GPFramework.constants.TIERED_MULTIPLIER*(len(datasetDict.keys()) - 1)
    # Now we go through and replace the values and weights (note we maximize novelty) for each individual
    for ind, novelty in zip(individuals, novelties):
        # add a novelty score to the individual for analysis later
        print('NOVELTY', novelty)
        if ind.novelties is None:
            ind.novelties = [novelty]
        else:
            ind.novelties.append(novelty)
        print('IND NOVELTIES', ind.novelties)
        # Adjusting for age here. we maximize here so the older something is, the higher its value should be.
        for i in range(int(np.floor(ind.age))):
            novelty += offset
        ind.fitness.weights = ind.fitness.weights + (1,)
        ind.fitness.values = np.hstack((ind.fitness.values, [novelty])) 
        print('NEW FITNESS', ind.fitness.values)
        print('NEW WEIGHTS', ind.fitness.weights)
    # Now we perform an nsga2 sort
    individuals = tools.selNSGA2(individuals, len(individuals))

    # And now the binary tournament
    if len(individuals)-k <=2:
        # k is too big; index error will happen during binary tournament
        k = len(individuals) - 3
    selected_pop = tools.selTournamentDCD(individuals, k)

    # Before we return, we need to put all the original fitnesses and weights back!
    for ind in individuals:
        # Removing the weight removes the fitness value as well!!!!
        ind.fitness.weights = ind.fitness.weights[:-1]
        print('CORRECTED FITNESS', ind.fitness.values)
        print('CORRECTED WEIGHTS', ind.fitness.weights)

    print('NOVELTIES', [ind.novelties for ind in individuals])
    # Now we return the selected pop
    return selected_pop





def distance_graph(pts, d):
    """
    Generates a d-connectivity graph from the set of given points.

    Creates a graph with vertices from {0,...,len(pts) - 1}, where there
    is an edge between two points iff their Euclidian distance is <= d.
    """
    g = nx.Graph()
    g.add_nodes_from(range(len(pts)))
    for i in range(len(pts)):
        for j in range(i):
            dist = np.linalg.norm(np.subtract(pts[i], pts[j]))
            if dist <= d:
                g.add_edge(i, j)
    return g

def maximize_minimum_distance_is(pts, k):
    """
    Finds the subset of pts of length k that maximizes the minimum Euclidian
    distance between any two points in the returned subset. Returns a set of
    the indices of the chosen points.

    May not exactly maximize the distance, as an approximation algorithm
    for maximum independent set is used.
    """
    if k <= 0 or k > len(pts):
        return []
    h = [0.0]
    for i in range(len(pts)):
        for j in range(i):
            heappush(h, np.linalg.norm(np.subtract(pts[i], pts[j])))
    cur = 0.0
    h2 = [heappop(h) for i in range(len(h))]
    distances = [0.0]
    for x in h2:
        if x != cur:
            distances.append(x)
            cur = x
    l = 0
    r = len(distances) - 1
    while l <= r:
        m = math.floor((l + r) / 2)
        ind_set = approximation.independent_set.maximum_independent_set(distance_graph(pts, distances[m]))
        if len(ind_set) < k:
            r = m - 1
        else:
            l = m + 1
    m = m - 1
    ind_set = approximation.independent_set.maximum_independent_set(distance_graph(pts, distances[m]))
    if len(ind_set) >= k:
        return list(ind_set)[:k]
    else:
        return []

def my_max_distance_nsga(individuals, k):
    """
    NSGA variant that chooses individuals in the contested front by maximizing
    the minimum distance in fitness between any two pairs of chosen individuals.
    """
    pareto_fronts = tools.sortNondominated(individuals, k)

    chosen = list(chain(*pareto_fronts[:-1]))
    k = k - len(chosen)
    if k > 0:
        last_front = pareto_fronts[-1]
        if k >= 500:
            tools.assignCrowdingDist(last_front)
            sorted_front = sorted(pareto_fronts[-1], key=attrgetter("fitness.crowding_dist"), reverse=True)
            chosen.extend(sorted_front[:k])
        else:
            pts = [np.array(ind.fitness.values) for ind in last_front]
            if pts[0][0] == np.inf:
                chosen.extend(last_front[:k])
            else:
                inds = maximize_minimum_distance_is(pts, k)
                chosen.extend([last_front[i] for i in inds])
    return chosen

def sel_lexicase(individuals, k):
    return tools.selLexicase(individuals, k)

def sample(means, cov, trials, a=0, b=1):
    """
    This is a method for performing a truncated normal multivaraite sampling
    :param means: list of expected values of length N
    :param cov: covariance matrix size NxN
    :param trials: number of samples to draw
    :param a: vector of lower bounds length N
    :param b: vector of upper bounds length N
    :return: list of samples NxTrials
    """
    dimensions = len(means)



    # Get the precision matrix
    H = np.linalg.inv(cov)
    samples = np.empty((trials, dimensions))
    # Let's pick a start value for our chain as the mean (Look in to a better one?)
    samples[0, :] = means

    i_s = np.arange(dimensions)
    not_i_s = [i_s[i_s != i] for i in np.arange(dimensions)]
    H_invs = 1./np.diag(H)
    sigmas = np.sqrt(H_invs)
    sigmas_inv = 1./sigmas

    for j in np.arange(1, trials):
        for i in np.arange(0, dimensions):
            # Index not i
            not_i = not_i_s[i]
            x_not_i = np.hstack((samples[j, :i], samples[j-1, i+1:]))

            # Get the conditional mean for dimension i
            mu = means[i] - H_invs[i]*np.dot(H[i, not_i], x_not_i-means[not_i])
            # Get the conditional standard deviation
            sigma_inv = sigmas_inv[i]
            sigma = sigmas[i]
            # Let's sample the conditional univariate truncated normal
            # Start with a uniform random number
            u = np.random.rand()
            # Next transform to truncated normal
            # ndtri is inverse norm
            cdf_a = scipy.special.ndtr((a[i]-mu)*sigma_inv)
            cdf_b = scipy.special.ndtr((b[i]-mu)*sigma_inv)
            trunc_normal = scipy.special.ndtri(u*(cdf_b - cdf_a) + cdf_a)
            x = mu+sigma*trunc_normal
            if not np.isfinite(x):
                # My hunch is currently that algorithms that operate outside the bounds of their variables will need to
                # be forced to their bounds i.e. if means[i] < a or means[i] > b
                print('X', x, 'mu', mu, 'sigma', sigma, 'a', a[i], 'b', b[i], 'mean', means[i])
                print('cdfa', cdf_a, 'cdfb', cdf_b)
                print('all means', [mean for mean in means])
                raise Exception('Should not have gotten here')
            samples[j,i] = x

    return samples

def dominates(a, b):
    """
    Test if a dominates b
    :param a:
    :param b:
    :return: boolean
    """
    return np.all(a <= b)

def my_pareto(data):
    pareto_points = []
    for i, data_point in enumerate(data):
        pareto_status = True
        pareto_points_to_remove = []
        for j in pareto_points:
            # If a pareto point dominates our point, we already know we are not pareto
            if dominates(data[j, :], data_point):
                pareto_status = False
                break
            # If our point dominates a Pareto point, that point is no longer Pareto
            if dominates(data_point, data[j, :]):
                pareto_points_to_remove.append(j)
        # Remove elements by value
        for non_pareto_point in pareto_points_to_remove:
            pareto_points.remove(non_pareto_point)
        if pareto_status:
            pareto_points.append(i)
    return pareto_points

def find_pareto(sample_row, num_objectives):
    samples = sample_row.reshape((-1, num_objectives))
    p_front_inds = my_pareto(samples)
    return p_front_inds

def fuzzy_select(population, k, gen=-1, tests=int(1e4), output_filename=None, objectives=['tp_vec', 'fp_vec', 'n_elems'],
                 a=[0, 0, 0], b=[1, 1, 280]):
    """
    This method implements a fuzzy selection based upon a probability of dominance for each individual.

    :param population: list of individuals
    :param k: number of individuals to select
    :param gen: generation number, used for logging
    :param tests: number of Monte Carlo samples to draw
    :param output_filename: name of output file, also used for logging
    :param objectives: names of objectives to be used for extracting test case vectors
    :param a: lower bounds per objective
    :param b: upper bounds per objective
    :return: list of parents
    """

    # Now let's try to monte carlo the whole population
    # First we need to build a covariance matrix from a matrix m
    # m will have each row represent a solution vector (variable) and each column
    # will be an observation

    # Let's begin by collection our test case vectors for each objective
    objective_mats ={}
    for objective in objectives:
        objective_array = np.array([getattr(individual, objective) for individual in population])
        objective_mats[objective] = objective_array

    # At this point we have a matrix for both objectives, but we want an interleaved
    # matrix so that we can compare the covariances of each individual on each Objective
    # To do this, we need to regularize the behavioral vectors in to an equal number of bins
    BINS = min((objective_mats[objective].shape[1] for objective in objectives))
    # Compute pad sizes
    pads = {}
    for objective in objectives:
        pad_size = np.int(np.ceil(float(objective_mats[objective].shape[1]) / BINS) * BINS - objective_mats[objective].shape[1])
        pads[objective] = np.empty(pad_size) * np.NaN

    observation_matrix = []

    # Iterate through individuals
    for i in np.arange(len(population)):
        for objective in objectives:
            # First pad so that our row is evenly divisible
            objective_row = np.append(objective_mats[objective][i, :], pads[objective])
            objective_row = np.reshape(objective_row, (-1, BINS))
            objective_row = np.nanmean(objective_row, axis=0)
            observation_matrix.append(objective_row)

    observation_matrix = np.array(observation_matrix)
    sys.stdout.flush()
    start_time = time.time()
    # Let's divide the observation matrix by the square root of the number of samples so that the variance will be
    # on the mean not the samples.  This should be the same as dividing the resultant matrix by the number of samples
    scaled_observation_matrix = observation_matrix / np.sqrt(float(BINS))

    # Ledoit wolf expects samples, features
    observation_covariance, shrinkage = sklearn.covariance.ledoit_wolf(scaled_observation_matrix.T)

    # Let's back out the means and covariance of the underlying normal
    means_vector = np.mean(observation_matrix, axis=1)
    sys.stdout.flush()
    start_time = time.time()

    random_sampling = sample(means_vector, observation_covariance, tests, a=a*len(population), b=b*len(population))
    sys.stdout.flush()

    start_time = time.time()

    totals_bin = np.zeros((len(population), 1))
    pool = multiprocessing.Pool(12)
    find_pareto_partial = functools.partial(find_pareto, num_objectives=len(objectives))
    pareto_inds_list = pool.map(find_pareto_partial, random_sampling)
    pool.close()

    for p_front_inds in pareto_inds_list:
        totals_bin[p_front_inds, 0] += 1

    sys.stdout.flush()
    totals_bin /= float(len(random_sampling))
    totals_bin /= float(np.sum(totals_bin))
    # Now we will use these probabilities to randomly select k individuals
    next_gen = np.random.choice(np.arange(len(population)), k, replace=True, p=totals_bin[:, 0])
    # if DEBUG:
    # If debug is defined to be true, we are going to write out the current population selection
    # is being performed on
    if gen >= 0:
        fuzzy_pop_log = gzip.open(output_filename, 'a')
        output_matrix = np.hstack((gen * np.ones((len(population), 1)),
                                   # FP
                                   means_vector[1::3].reshape((-1, 1)),
                                   # TP
                                   means_vector[::3].reshape((-1, 1)),
                                   # Len
                                   means_vector[2::3].reshape((-1,1)),
                                   # Prob number of selections
                                   totals_bin * k))
        np.savetxt(fuzzy_pop_log, output_matrix, delimiter=',')

    return [population[i] for i in next_gen]
