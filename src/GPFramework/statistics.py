"""
Programmed by Jason Zutty
Modified by VIP Team
Implements statistics functions
"""
import numpy as np
import re
from gp_framework_helper import get_sm_list, get_sp_list, get_op_list, get_select_list, get_learner_list, get_modify_list

def min_objective_distance(individuals):
    """Minimum distance between two individuals in the objective space"""
    min_dist = None
    for i in range(len(individuals)):
        if individuals[i].fitness.values[0] == np.inf:
            continue
        for j in range(i):
            if individuals[j].fitness.values[0] == np.inf:
                continue
            dist = np.linalg.norm(np.array(individuals[i].fitness.values) - np.array(individuals[j].fitness.values))
            if min_dist is None or dist < min_dist:
                min_dist = dist
    return min_dist if min_dist is not None else 0

def min_nonzero_objective_distance(individuals):
    """Minimum nonzero distance between two individuals in the objective space"""
    min_dist = None
    for i in range(len(individuals)):
        if individuals[i].fitness.values[0] == np.inf:
            continue
        for j in range(i):
            if individuals[j].fitness.values[0] == np.inf:
                continue
            dist = np.linalg.norm(np.array(individuals[i].fitness.values) - np.array(individuals[j].fitness.values))
            if dist != 0 and (min_dist is None or dist < min_dist):
                min_dist = dist
    return min_dist if min_dist is not None else 0

def objectives_z_score(individuals):
    """
    Average z-score of individuals in the objective space
    Calculates the mean μ and sample covariance matrix Σ of the objectives,
    and returns the mean score (x - μ)' inv(Σ) (x - μ) across all individuals x
    """
    objectives = [np.array(ind.fitness.values) for ind in individuals if np.all(np.isfinite(ind.fitness.values))]
    if len(objectives) <= 2:
        return 0
    objectives_T = np.transpose(objectives)
    covariance = np.cov(objectives_T, ddof=1)
    covariance_inv = np.linalg.inv(covariance)
    mean = np.mean(objectives_T, axis=1)
    z_scores = [np.matmul(np.transpose(x - mean), np.matmul(covariance_inv, x - mean)) for x in objectives]
    return np.mean(z_scores)

def tree_size(individuals):
    """Average tree size"""
    if len(individuals) == 0:
        return 0
    num_elements = 0
    for individual in individuals:
        for tree in individual:
            num_elements += len(tree)
    return num_elements / len(individuals)

'''Return a tuple of statistics (#, #, #, ...)'''
def primitive_parse(individuals):
    total_sum, sm_sum, sp_sum, op_sum, select_sum, learner_sum, modify_sum = (0,)*7
    j = 0;
    for individual in individuals:
        '''grab main tree, get list of primitives, if there is an mod,
        then append its primitives and check the mod for more mods until there
        are none left.'''
        my_string = str(individual[0])

        # Next check to see if that tree uses any ADF's
        mod_matches = re.findall(r"mod_(\d+)", my_string)

        # Get the int representing the index from the back of the array e.g. [main, 2, 1, 0]
        mod_matches = np.unique([int(mod_match) for mod_match in mod_matches])
        while len(mod_matches) > 0:
            for mod_match_num in mod_matches:
                my_string += '\nmod_' + str(mod_match_num) + ': ' + str(individual[-1-mod_match_num])
            mod_matches = re.findall(r"mod_(\d+)", str(individual[-1-mod_match_num]))
            mod_matches = np.unique([int(mod_match) for mod_match in mod_matches if 'mod_' + mod_match + ': ' not in my_string])

        # split string by mod lines
        mod_list = my_string.split("\n")
        for s in mod_list:
            s = s[7:]
            re.sub(r'[ \d,ARG)]', '', s)
            p_list = s.split("(")
            '''This would be simple if we could just get the primitive name from the
            deap primitive objects :('''
            for p in p_list:
                total_sum += 1
                if p in get_sm_list():
                    sm_sum += 1
                if p in get_sp_list():
                    sp_sum += 1
                if p in get_op_list():
                    op_sum += 1
                if p in get_select_list():
                    select_sum += 1
                if p in get_learner_list():
                    learner_sum += 1
                if p in get_modify_list():
                    modify_sum += 1
    return (total_sum, sm_sum, sp_sum, op_sum, select_sum, learner_sum, modify_sum)
