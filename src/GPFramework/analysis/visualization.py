import numpy as np
import os
from GPFramework import EMADE as emade
from GPFramework.sql_connection_orm_master import SQLConnectionMaster
from deap import creator, base
import multiprocess
import pickle
import argparse as ap
import sys
import itertools
import matplotlib.pyplot as plt
import imageio
import moviepy.editor as mpy
import glob
from operator import itemgetter

# Get pickled input as argument
parser = ap.ArgumentParser()
parser.add_argument('filename', help='Pickled input from parsed xml')
parser.add_argument('database', default='sqlite', type=str, help='SQLAlchemy connection string in the form of dialect[+driver]://user:password@host/dbname. See http://docs.sqlalchemy.org/en/latest/core/engines.html#sqlalchemy.create_engine')

args = parser.parse_args()
inputFile = args.filename
database_str = args.database

# Load pickled dictionaries
with open(inputFile,'rb') as pickleFile:
    # Dictionaries must be loaded in the same order they were written
    evolution_params_dict = pickle.load(pickleFile)
    objectives_dict = pickle.load(pickleFile)
    dataset_dict = pickle.load(pickleFile)
    stats_dict = pickle.load(pickleFile)
    misc_dict = pickle.load(pickleFile)
    cache_dict = pickle.load(pickleFile)

if 'regression' in evolution_params_dict:
    regression = int(evolution_params_dict['regression'])
else:
    regression = False

fitness_names = [objectives_dict[objective]['name'] for objective in objectives_dict]
dataset_names = [dataset_dict[dataset]['name'] for dataset in dataset_dict]
objectives_dict_names = {objectives_dict[x]['name']: objectives_dict[x] for x in objectives_dict}

"""
weights = tuple([objectives_dict[objective]['weight'] for objective in objectives_dict])
creator.create("FitnessMin", base.Fitness, weights=weights)
fitness_attr = dict(zip(fitness_names, itertools.repeat(creator.FitnessMin)))
creator.create("Individual", list, fitness=creator.FitnessMin,
       pset=[], age=0, hash_val=None, **fitness_attr)
"""
emade.create_representation(mods=3, regression=regression)
emade.setObjectives(objectives_dict)
emade.setDatasets(dataset_dict)
emade.setMemoryLimit(misc_dict['memoryLimit'])
emade.setCacheInfo(cache_dict)
emade.set_statistics(stats_dict)
emade.buildClassifier()

db = SQLConnectionMaster(connection_str=database_str,
        reuse=True, fitness_names=fitness_names, dataset_names=dataset_names,
        statistics_dict=stats_dict, cache_dict=cache_dict, is_worker=True)

def dominates(a, b):
    """
    Test if a dominates b
    :param a:
    :param b:
    :return: boolean
    """
    return np.all(a <= b)

def my_pareto(data):
    pareto_points = set()
    for i, x in enumerate(data):
        pareto_status = True
        pareto_points_to_remove = []
        for j in pareto_points:
            # If a pareto point dominates our point, we already know we are not pareto
            if dominates(data[j], x):
                pareto_status = False
                break
            # If our point dominates a Pareto point, that point is no longer Pareto
            if dominates(x, data[j]):
                pareto_points_to_remove.append(j)
        # Remove elements by value
        for non_pareto_point in pareto_points_to_remove:
            pareto_points.remove(non_pareto_point)
        if pareto_status:
            pareto_points.add(i)
    return pareto_points

def statistic(name: str, directory: str, title: str=None, range_gen=None,
        range_y=None):
    """
    Generates image of the given statistic over time

    Args:
        name: statistic
        directory: directory to save image into
        title: name of the file
        range_gen: tuple consisting of minimum and maximum generation to plot
        range_y: tuple consisting of minimum and maximum for y axis
    """
    stats = db.select_statistics()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    if range_y is not None:
        plt.ylim(range_y[0], range_y[1])

    plt.xlabel('Generation')
    plt.ylabel(name)
    if range_gen is not None:
        gens = db.num_gens()
        a = 0 if range_gen[0] < 0 else range_gen[0]
        b = gens if range_gen[1] > gens else gens
        plt.plot(range(a, b), stats[name][a:b])
    else:
        plt.plot(stats[name])
    if title is not None:
        plt.title(title)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    filename = '{}.png'.format(name)
    plt.savefig(os.path.join(directory, filename))
    return filename

def pareto_front(objective1: str, objective2: str, gen: int, directory: str,
        dataset: str=dataset_names[0], boundary: bool=False, range1=None,
        range2=None, force_pareto: bool=True):
    """
    Generates image of the Pareto front on a given generation

    Args:
        objective1: horizontal axis objective
        objective2: vertical axis objective
        gen: generation to retrieve front of
        directory: directory to save image into
        boundary: whether to display Pareto front boundary, force_pareto must also be true
        range1: tuple of minimum and maximum value on horizontal axis plot
        range2: tuple of minimum and maximum value on vertical axis plot
        force_pareto: if True selects only the Pareto individuals with respect
                      to the given objectives, otherwise plots all Pareto individuals
    """
    inds = db.get_pareto(gen)

    fitnesses = [np.array([getattr(ind, dataset + ' ' + objective1),
        getattr(ind, dataset + ' ' + objective2)]) for ind in inds]
    print(fitnesses)
    fitnesses = [x for x in fitnesses if not np.any(np.equal(x, None)) and np.all(np.isfinite(x))]
    pt1 = np.array([objectives_dict_names[objective1]['achievable'], 0])
    if not np.any([dominates(x, pt1) for x in fitnesses]):
        fitnesses.append(pt1)
    pt2 = np.array([0, objectives_dict_names[objective2]['achievable']])
    if not np.any([dominates(x, pt2) for x in fitnesses]):
        fitnesses.append(pt2)
    if force_pareto:
        fitnesses = [fitnesses[i] for i in my_pareto(fitnesses)]
    list.sort(fitnesses, key=itemgetter(0))

    fig = plt.figure()
    # fig.set_size_inches(5,5) # can set an arbitrary
    # plt.axis('equal')
    ax = fig.add_subplot(111)

    if range1 is not None:
        plt.xlim(range1[0], range1[1])
        ax.set_autoscale_on(False)
    if range2 is not None:
        plt.ylim(range2[0], range2[1])
        ax.set_autoscale_on(False)

    ax.scatter([fitness[0] for fitness in fitnesses], [fitness[1] for fitness in fitnesses], color='r')
    if boundary and force_pareto:
        ax.plot([fitness[0] for fitness in fitnesses], [fitness[1] for fitness in fitnesses], color='r', drawstyle='steps-post')
    plt.xlabel(objective1)
    plt.ylabel(objective2)
    plt.title("Non-dominated Front, Generation " + str(gen))
    # plt.ion() # uncomment if running concurrently with something else
    if not os.path.isdir(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, "pareto_front_gen_{}.png".format(gen))
    plt.savefig(filename)
    return filename
    # plt.pause(.001) # uncomment if running concurrently with something else

def pareto_front_anim(objective1: str, objective2: str, directory: str,
    dataset: str=dataset_names[0], boundary: bool=False, range1=None,
    range2=None, force_pareto: bool=True):
    """
    Generates video of the Pareto front over all generations

    Args:
        objective1: horizontal axis objective
        objective2: vertical axis objective
        directory: directory to save images and video into
        dataset: dataset of the objective scores
        boundary: whether to display Pareto front boundary,
                  force_pareto must also be true
        range1: tuple of minimum and maximum value on horizontal axis plot
        range2: tuple of minimum and maximum value on vertical axis plot
        force_pareto: if true selects only the Pareto individuals
                      with respect to the given objectives, otherwise
                      plots all Pareto individuals
    """
    fig = plt.figure()
    generations = db.num_gens()
    if range1 is None:
        range1 = [0, objectives_dict_names[objective1]['achievable'] * 1.1]
    if range2 is None:
        range2 = [0, objectives_dict_names[objective2]['achievable'] * 1.1]
    images = [pareto_front(objective1, objective2, gen, directory, dataset,
                boundary, range1, range2, force_pareto)
                    for gen in range(generations)]
    clip = mpy.ImageSequenceClip(images, fps=1)
    filename = os.path.join(directory, 'pareto_front_anim.mp4')
    clip.write_videofile(filename)
    return filename

if __name__ == '__main__':
    """
    pareto_front_anim('False Positives', 'False Negatives', 'visualization',
        boundary=True, force_pareto=True)
    statistic('min_nonzero_objective_distance', 'visualization')
    """
    # statistic('tree_size', 'visualization2', range_y=[0,60])
    pareto_front_anim('FalsePositives', 'FalseNegatives', 'visualization',
        boundary=True, force_pareto=True)
