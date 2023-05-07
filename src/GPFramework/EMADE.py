"""
Georgia Tech Multiple Objective Evolutionary Programming Module
Coded by Jason Zutty
Modified by VIP Team
"""
from deap import gp
from deap import base
from deap import creator
from deap import tools
from deap.benchmarks.tools import hypervolume
import re
import GPFramework.gp_framework_helper as gpFrameworkHelper
from GPFramework.gp_framework_helper import LearnerType
import numpy as np
from GPFramework.data import EmadeDataPair, load_many_to_one_from_file, EmadeDataPairNN, EmadeDataPairNNF
from GPFramework.data import load_feature_data_from_file, load_many_to_many_from_file, load_text_data_from_file, load_text_data_for_summary, load_pickle_from_file, load_feature_pickle_from_file
from GPFramework.data import load_images_from_file, load_many_to_some_from_file, load_image_csv_from_file
from GPFramework import emade_operators
from GPFramework.emade_operators import clean_individual, simulateMateMut, simulateMateMutNN
from GPFramework import selection_methods
from GPFramework.sql_connection_orm_master import SQLConnectionMaster
from GPFramework.sql_connection_orm_base import IndividualStatus, ConnectionSetup
#from GPFramework.general_methods import pingGPU
from GPFramework.cache_methods import cleanup_trash
from GPFramework import neural_network_methods as nnm
from GPFramework.neural_network_methods import DataDims, preprocessDataPair, genNNLearner, genADF, get_individual_layer_frequencies
import GPFramework.constants
import random
import copy as cp
import pickle
import dill
import sys
import os
import traceback
import time
import multiprocess as mp
import multiprocess.pool
from multiprocess import Process, Manager, Pool
from multiprocess.queues import Queue
import socket
from pathlib import Path
from functools import reduce
from datetime import datetime
import concurrent.futures

import itertools
import operator
import hashlib

# For probability based selection
import math
from psutil import virtual_memory
import psutil
import signal

if sys.platform != 'win32':
    import resource
    

class EMADE:

    def __init__(self):
        """Initializes the EMADE object. Creates the toolbox that will be used to store all of the GP functions.
        """
        self.toolbox = base.Toolbox()
        gp.genNNLearner = genNNLearner
        gp.genADF = genADF


    def create_representation(self, mods = 0, regression = False, datatype = None):
        """Creates a GP tree object with primitives and automatically defined functions (ADFs)

        Args:
            mods: Numbers of automatically defined functions (ADFs) or number of gp.PrimitiveSetTyped object primitives created.
            regression: whether emade is solving a regression problem
            datatype: the type of the input data
        """

        if datatype not in ['textdata','imagedata']:
            raise ValueError("NAS only supports text data and image data. Passed dtype: {}".format(datatype))

        gpFrameworkHelper.set_regression(regression)

        if datatype=='imagedata':
            ll = nnm.LayerList4dim
        elif datatype=='textdata':
            ll = nnm.LayerList3dim

        # Initialize MAIN overarching primitive
        self.pset = gp.PrimitiveSetTyped('MAIN', [EmadeDataPairNN, nnm.ModList], EmadeDataPairNNF)
        self.pset.mod_reshapes = list() # add attribute to track how modules change dimensions of the data; is used for nnlearner generation

        # Add primitives and terminals specific to NNLearner
        gpFrameworkHelper.addNNLearnerPrimitives(self.pset, datatype, ll)
        gpFrameworkHelper.addTerminals(self.pset, datatype, self.data_dims)

        # Initialize ADF primitive set and give it all desired primitives and terminals with inputs and outputs
        # based on data type
        fitness_names = (self.datasetDict[dataset]['name'] for dataset in self.datasetDict)
        fitness_attr = dict(zip(fitness_names, itertools.repeat(creator.FitnessMin)))
        objective_names = (self.objectivesDict[objective]['name'] for objective in self.objectivesDict)
        objectives_attr = dict(zip(objective_names, itertools.repeat(np.ndarray(0))))

        # Add objective attributes to fitness attributes
        fitness_attr.update(objectives_attr)

        self.mod_pset = gp.PrimitiveSetTyped('mod', [ll], nnm.LayerListM)
        creator.create("ADF", list, fitness=creator.FitnessMin, reshape=[],
            pset=self.mod_pset, weights=None, age=0, num_occur=0, retry_time=0, novelties = None,
            mod_num=None, **fitness_attr)
        ephemeral_methods = gpFrameworkHelper.addADFPrimitives(self.mod_pset, datatype)
        gpFrameworkHelper.addTerminals(self.mod_pset, datatype, self.data_dims, exclude_inputlayers=True)

        glob_mods = {}
        data_type = self.datasetDict[0]['type']
        if (os.path.exists("Global_MODS")):
            # Load existing ADFs if available and add them to the pset of MAIN
            # Will probably want to load from SQL in future and incorporate reuse
            with open("Global_MODS", "rb") as mod_file:
                glob_mods = dill.load(mod_file)
            for mod in glob_mods:
                if "mod_" not in mod:
                    continue
                func = gp.compile(glob_mods[mod][1][0], self.mod_pset)
                self.pset.addPrimitive(func, [nnm.LayerListM], nnm.LayerListM, name=glob_mods[mod][0]) # this line lets us chain modules together
                self.pset.context.update({glob_mods[mod][0]: func}) 

                reshape = glob_mods[mod][2]
                if len(reshape)==3:
                    self.pset.mod_reshapes.append((mod, reshape[0], reshape[2])) # only need to track one of the spatial dimensions
                else:
                    self.pset.mod_reshapes.append((mod,)+tuple(reshape))
        else:
            try: 
                with open("Global_MODS", "wb") as mod_file:
                    for i in np.arange(mods):
                        # Create a name for the mod, or automatically defined functions
                        mod_name = 'mod_' + str(i + 1)

                        # Reshape value is the difference to original data_dim if mod is applied. Used in genNNLearner method
                        # All layer primitives have terminals as their argument, so the number of layers in a module = depth - 1
                        # So with min_=2 and max_=4, modules contain 1-3 layers. 
                        # 3 layers was chosen as the maximum because that's the most that's used in a block of layers for Resnet and VGG16
                        expr, reshape = gp.genADF(self.mod_pset, min_=2, max_=4, data_dim=self.data_dims, data_type=data_type, type_=self.mod_pset.ret)
                        tree = gp.PrimitiveTree(expr)
                        tree = [tree]
                        mod_obj = creator.ADF(tree)
                        mod_obj.mod_num = i + 1
                        mod_obj.age = 0
                        mod_obj.num_occur=0
                        mod_obj.name = mod_name
                        mod_obj.reshape = reshape
                        func = gp.compile(tree[0], self.mod_pset)
                        print(mod_name, tree[0].__str__())

                        glob_mods[mod_name] = [mod_name, mod_obj, list(reshape)]
                        self.pset.addPrimitive(func, [nnm.LayerListM], nnm.LayerListM, name=mod_name)
                        self.pset.context.update({mod_name: func}) # pset.addPrimitive wraps func inside a primitive class which we don't want -- this should undo that

                        if len(reshape)==3:
                            self.pset.mod_reshapes.append((mod_name, reshape[0], reshape[2])) # only need to track one of the spatial dimensions
                        else:
                            self.pset.mod_reshapes.append((mod_name,)+reshape)
                    dill.dump(glob_mods, mod_file)
            except Exception as e:
                print(e) 
                raise Exception("Corrupted ADFS file, please delete and restart run.")

    
        # No ADF psets so list is length 1
        self.psets = [self.pset]

        # Using a new gen method which enforces valid generation of individuals
        # At time of writing, requires a minimum tree depth of 5, 
        # which would contain an NNLearner, a Dense Layer or PassLayerList, a flatten/global pooling layer, an InputLayer, and the inputlayer's terminal(s).
        # Obviously this kind of individual would be trivial since no mods are used, so we add 1 to the min depth --> 6. 
        # Max number of mods in an individual will be max_ - 5 when the initial pop is generated
        self.toolbox.register('expr_main', gp.genNNLearner,
                 pset=self.pset, type_=self.pset.ret, min_=6, max_=22, data_dim=self.data_dims, data_type=data_type)

        self.exprs = ['expr_main']

        # We also need a toolbox function for the creation of the main tree
        self.toolbox.register('MAIN', tools.initIterate, gp.PrimitiveTree, self.toolbox.expr_main)

        self.toolbox.register("compile", gp.compileADF, psets=self.psets)

        # This cycle will be used to create the individual
        self.create_cycle = [self.toolbox.MAIN]

        return self.pset, self.mod_pset, ephemeral_methods, self.data_dims


    def logHyperVolume(self, pareto_front, description=""):
        """Calculates the current hypervolume and writes it to disk and stdout.
        Note that the hypervolume function assumes minimization on all objectives.

        Args:
            pareto_front: paretoFront that defines current hypervolumes
            objectivesDict that defines the dimensionality of objective space
                and the limit to each dimension (as defined by "achievable"
                in the input XML file)
            description: a description of the current state of the evolution
        """

        #first obtain the bounds for our hypervolume
        #(these define the hypervolume before any individuals are present)
        objLimits = []
        objGoals = []
        for i, objectiveNumber in enumerate(self.objectivesDict.keys()):
            #(low,high)
            objLimits.append(self.objectivesDict[objectiveNumber]["upper"])
            objGoals.append(self.objectivesDict[objectiveNumber]["lower"])



        individual_list = cp.deepcopy(pareto_front)


        for ind in individual_list:
            newTuple = list(ind.fitness.values)
            for i, obj in enumerate(objLimits):
                if np.isinf(ind.fitness.values[i]):
                    #replace np infinity values with the upper bound for each dimension
                    #to prepare for the deap hypervolume function
                    newTuple[i] = obj
                #subtract out lower bounds on each objective:
                newTuple[i] -= objGoals[i]
            ind.fitness.values = tuple(newTuple)
            print(ind.fitness.values)


        for ind in individual_list:
            for dim, obj in enumerate(ind.fitness.values):
                if obj > objLimits[dim]:
                    #this is a bad thing because an individual with infinity could outperform
                    #these individuals
                    with open('hypervolume' + str(os.getpid()) + '.txt', 'a') as f:
                        f.write("An individual is performing worse than its achievable bound. This may negatively affect hypervolume calculation. " +\
                            "Consider increasing the achievable bound on objective number " + str(dim))
                    print("An individual is performing worse than its achievable bound. This may negatively affect hypervolume calculation.")
                    print("Consider increasing the achievable bound on objective number ", str(dim))
                    sys.stdout.flush()
                elif obj < objGoals[dim]:
                    with open('hypervolume' + str(os.getpid()) + '.txt', 'a') as f:
                        f.write("An individual is performing better than its goal. This may negatively affect hypervolume calculation. " +\
                            "Consider decreasing the goal on objective number " + str(dim))
                    print("An individual is performing better than its goal. This may negatively affect hypervolume calculation.")
                    print("Consider decreasing the goal on objective number ", str(dim))
                    sys.stdout.flush()


        #this is a non-trivial algorithm which improves over a naive
        #high-dimensional riemann sum. Should also be implemented in C
        # unless EMADE throws a warning about the incompatability
        # of your C compiler on install
        hv = 1
        for obj in range(len(objGoals)):
            hv *= (objLimits[obj] - objGoals[obj])
        if individual_list:
            #if there are ind. in front, take out their hv
            hv -= hypervolume(individual_list, tuple(objLimits))


        with open('hypervolume' + str(os.getpid()) + '.txt', 'a') as f:
            f.write(str(description) + ',' + str(hv) + '\n')

        print('Hypervolume: ', hv)
        sys.stdout.flush()

    def setDicts(self, objectivesDict, datasetDict, cacheDict, stats_dict):
        self.cacheDict = cacheDict
        self.statistics_dict = stats_dict
        self.objectivesDict = objectivesDict
        self.datasetDict = datasetDict

    def setObjectives(self):
        """Builds fitness objects based on objectives dictionary

        Args:
            objectivesDict: Objectives dictionary containing information about weights and thresholds
        """
        # Compute arrays of weights and thresholds from objective information
        weights = tuple([self.objectivesDict[objective]['weight'] for objective in self.objectivesDict])
        goals = tuple([self.objectivesDict[objective]['goal'] for objective in self.objectivesDict])
        achievable = tuple([self.objectivesDict[objective]['achievable'] for objective in self.objectivesDict])

        # Define LROI to be hypercube defined by origin and the following point,
        # minimum acceptable objective values
        self.LROI = np.array(goals)

        # Define SROI to be hypercube defined by origin and the following point,
        # region describing where solutions are known to be possible
        self.SROI = np.array(achievable)

        # Uses creator to create a FitnessMin class using the weights for base.Fitness
        creator.create("FitnessMin", base.Fitness, weights=weights)

    def setDatasets(self):
        """Called after setObjectives finishes up initialization by setting up for datasets


        Args:
            datasetDict: Dataset dictionary from xml file containing information about data folds
        """

        fitness_names = (self.datasetDict[dataset]['name'] for dataset in self.datasetDict)
        fitness_attr = dict(zip(fitness_names, itertools.repeat(creator.FitnessMin)))
        objective_names = (self.objectivesDict[objective]['name'] for objective in self.objectivesDict)
        objectives_attr = dict(zip(objective_names, itertools.repeat(np.ndarray(0))))

        # Add objective attributes to fitness attributes
        fitness_attr.update(objectives_attr)
        creator.create("Individual", list, fitness=creator.FitnessMin,
                       pset=self.pset, age=0, modList=[],
                       elapsed_time=0, retry_time=0,
                       novelties = None, parents=[],
                       hash_val=None, **fitness_attr)

        self.toolbox.register("individual", tools.initCycle,
            creator.Individual, self.create_cycle)
        self.toolbox.register("individual_from_seed", creator.Individual)
        self.toolbox.register("population", tools.initRepeat,
            list, self.toolbox.individual)

        self.toolbox.register("selectNSGA2", tools.selNSGA2)
        self.toolbox.register("binaryTournament", selection_methods.selTournamentDCD)

        self.toolbox.register('spea2', tools.selSPEA2)

        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("mateEphemeral", emade_operators.cx_ephemerals)
        self.toolbox.register("mateNNMiddle", emade_operators.one_point_crossover_middle_prim)
        self.toolbox.register("mateNNFull", emade_operators.one_point_crossover_full_prim)

        self.toolbox.register("evaluate", evaluate)
        self.toolbox.register("mutate", gp.mutInsert)
        self.toolbox.register("mutateLearner", emade_operators.insert_modifyLearner)
        self.toolbox.register("mutateEphemeral", gp.mutEphemeral, mode='one')
        self.toolbox.register("mutateNodeReplace", gp.mutNodeReplacement)
        self.toolbox.register("mutateUniform", gp.mutUniform)
        self.toolbox.register("mutateShrink", gp.mutShrink)
        self.toolbox.register("mutateAddLayer", emade_operators.add_layer)
        self.toolbox.register("mutateRemoveLayer", emade_operators.remove_layer)
        self.toolbox.register("mutateModifyLayer", emade_operators.modify_layer)
        # self.toolbox.register("healLearners", emade_operators.check_learner_nlp)
        self.toolbox.register("mutateSwapLayer", emade_operators.swap_layer)
        self.toolbox.register("mutateActivationParam", emade_operators.mut_terminal_type_based, sel_type=nnm.Activation)
        self.toolbox.register("mutateWeightInitializerParam", emade_operators.mut_terminal_type_based, sel_type=nnm.WeightInitializer)
        self.toolbox.register("mutateOptimizerParam", emade_operators.mut_terminal_type_based, sel_type=nnm.Optimizer)
        self.toolbox.register("fitnessScale", HCDScaleMin)
        self.toolbox.register('adjustForDataset', adjustForDataset)
        self.toolbox.register('genePoolPartition', genePoolPartition)

        self.history = tools.History()

        # Depth Limit from Koza
        self.toolbox.decorate("mate",
            gp.staticLimit(operator.attrgetter('height'), 50),
            self.history.decorator)
        self.toolbox.decorate("mateEphemeral",
            gp.staticLimit(operator.attrgetter('height'), 50),
            self.history.decorator)

        #self.toolbox.decorate("mutate", gp.staticLimit(operator.attrgetter('height'),50), self.history.decorator)
        #self.toolbox.decorate("mutateEphemeral", gp.staticLimit(operator.attrgetter('height'),50), self.history.decorator)
        #self.toolbox.decorate("mutateNodeReplace", gp.staticLimit(operator.attrgetter('height'),50), self.history.decorator)
        #self.toolbox.decorate("mutateUniform", gp.staticLimit(operator.attrgetter('height'),50), self.history.decorator)
        #self.toolbox.decorate("mutateShrink", gp.staticLimit(operator.attrgetter('height'),50), self.history.decorator)

        return creator

    def buildClassifier(self):
        """
        Sets up the GP framework to build a classifier
        """
        dataTypeDic = {'featuredata':load_feature_data_from_file, 'streamdata':load_many_to_one_from_file,'filterdata':load_many_to_many_from_file,
                        'imagedata':load_images_from_file,'detectiondata':load_image_csv_from_file,"textdata":load_text_data_from_file,
                        'summarydata':load_text_data_for_summary}

        for dataset in self.datasetDict:
            load_function = dataTypeDic.get(self.datasetDict[dataset]['type'])
            if self.datasetDict[dataset]['pickle'] and (self.datasetDict[dataset]['type']=='recdata' or self.datasetDict[dataset]['type']=='textdata' or self.datasetDict[dataset]['type']=='featuredata'):
                load_function = load_feature_pickle_from_file
            elif self.datasetDict[dataset]['pickle']:
                load_function = load_pickle_from_file
            if load_function == None:
                raise NotImplementedError(self.datasetDict[dataset]['type'] +
                        ' must be featuredata, imagedata, streamdata, detectiondata, filterdata, textdata, or summarydata')

            def reduce_instances(emadeDataTuple):
                proportion = self.datasetDict[dataset]['reduceInstances']
                emadeData, cache = emadeDataTuple
                subset = emadeData.get_instances()[:round(len(emadeData.get_instances())*proportion)]
                emadeData.set_instances(subset)
                return emadeData, cache
            use_cache = self.cacheDict['useCache']
            compress = self.cacheDict['compression']
            train_data_array = [reduce_instances(load_function(folder, use_cache=use_cache, compress=compress))
                                for folder in self.datasetDict[dataset]['trainFilenames']]

            test_data_array = [reduce_instances(load_function(folder, use_cache=use_cache, compress=compress, hash_data=True))
                               for folder in self.datasetDict[dataset]['testFilenames']]


            # truth_data_array = [test_data[0].get_target() for test_data in test_data_array]
            #[test_data[0].set_target(np.full(test_data[0].get_target().shape,np.nan)) for test_data in test_data_array]

            # Stores DataPair object
            self.datasetDict[dataset]['dataPairArray'] = [EmadeDataPair(
                                                        train_data, test_data
                                                        ) for train_data, test_data in
                                                        zip(train_data_array, test_data_array)]
                                                    
            # TODO: Add data_dims information for other datatypes. This data_dims is used in the 
            # generate method to prevent invalid combinations of layers. 
            if (self.datasetDict[0]['type'] == 'imagedata' or self.datasetDict[0]['type'] == 'detectiondata'):
                s = train_data_array[0][0].get_instances()[0].get_stream().get_data().shape
                if (len(s) == 2):
                    s = (s[0], s[0], 1)
                self.data_dims = s
            elif (self.datasetDict[0]['type'] == 'textdata'):
                self.data_dims = train_data_array[0][0].get_numpy().shape

            fitness_names = [self.objectivesDict[objective]['name'] for objective in self.objectivesDict]
            dataset_names = [self.datasetDict[dataset]['name'] for dataset in self.datasetDict]
            for pair in self.datasetDict[dataset]['dataPairArray']:
                pair.set_db_map({'connection_str': self.cacheDict['database'], 'reuse': 1,
                                 'fitness_names': fitness_names, 'dataset_names': dataset_names,
                                 'statistics_dict': self.statistics_dict, 'cache_dict': self.cacheDict, 'is_worker':True})
                pair.set_central(self.cacheDict['central'])
                pair.set_threshold(self.cacheDict['timeThreshold'])
                pair.set_cache_limit(self.cacheDict['cacheLimit'])
                pair.set_caching_mode(use_cache)
                pair.set_compression(compress)
                pair.set_db_use()
                pair.set_truth_data()
                pair.set_multilabel(self.datasetDict[dataset]['multilabel'])
                pair.get_test_data().set_target(np.full(pair.get_test_data().get_target().shape, np.nan)) # clear out test data after setting truth data in data_pair! 
                pair.set_datatype(self.datasetDict[dataset]['type'])
                pair.set_regression(self.datasetDict[dataset]['regression'])
            
            for i in range(len(self.datasetDict[dataset]['dataPairArray'])):
                # preprocess datapair
                self.datasetDict[dataset]['dataPairArray'][i] = preprocessDataPair(self.datasetDict[dataset]['dataPairArray'][i])
            
            # after one hot encoding target data, we can set truth data in datasetDict
            truth_data_array = []
            for pair in self.datasetDict[dataset]['dataPairArray']:
                truth_data_array.append(pair.get_truth_data())
            self.datasetDict[dataset]['truthDataArray'] = truth_data_array

            for i in range(len(self.datasetDict[dataset]['dataPairArray'])):
                # preprocess datapair
                self.datasetDict[dataset]['dataPairArray'][i] = preprocessDataPair(self.datasetDict[dataset]['dataPairArray'][i])
            
            # after one hot encoding target data, we can set truth data in datasetDict
            truth_data_array = []
            for pair in self.datasetDict[dataset]['dataPairArray']:
                truth_data_array.append(pair.get_truth_data())
            self.datasetDict[dataset]['truthDataArray'] = truth_data_array

    def setMemoryLimit(self, memoryLimit):
        """
        Sets the memory limit for evaluation.

        Args:
            memoryLimit: Amount of memory to use for evaluation
        """
        self.mem_limit = memoryLimit

    def buildPredictor(self, trainObsMat, trainValVec, testObsMat, testValVec):
        """Sets up the GP framework to build a predictor

        Args:
            trainObsMat:    M X N matrix containing M observations of N features
            trainValVec:    Length M vector containing output data for M
                            observations
            testObsMat:     A X B matrix containing A observations of B
                            features
            testValVec:     Length A vector containing output data for A
                            observations
        """
        self.trainMat = trainObsMat
        self.trainVec = trainValVec
        self.testObsMat = testObsMat
        self.testValVec = testValVec

def my_str(individual, name=None):
    """Return the string representation of an individual

    Args:
        individual: The individual in question

    Returns:
        the string representation of an individual
    """
    # First we get the string of the main tree
    my_string = str(individual[0])

    if (name):
        my_string = name + ": " + my_string

    if (not name):
        # Next check to see if that tree uses any ADF's
        mod_matches = re.findall(r"mod_(\d+)", my_string)

        # Get the int representing the index from the back of the array e.g. [main, 2, 1, 0]
        mod_matches = np.unique([int(mod_match) for mod_match in mod_matches])

        loaded=False
        while not loaded:
            try:
                with open("Global_MODS", "rb") as mod_file:
                    glob_mods = dill.load(mod_file)
                loaded=True
            except EOFError:
                time.sleep(15)

        while len(mod_matches) > 0:
            for mod_match_num in mod_matches:
                my_string += '\nmod_' + str(mod_match_num) + ': ' + str(glob_mods["mod_" + str(mod_match_num)][1][0].__str__())
            mod_matches = re.findall(r"mod_(\d+)", str(glob_mods["mod_" + str(mod_match_num)][1][0]))
            mod_matches = np.unique([int(mod_match) for mod_match in mod_matches if 'mod_' + mod_match + ': ' not in my_string])

    return my_string


def my_hash(individual):
    """Return the SHA256 hash of an individual

    Returns:
        the SHA256 hash of individual
    """
    return hashlib.sha256(my_str(individual).encode()).hexdigest()

def my_similar(ind1, ind2):
    """Defines whether two individuals are similar or not. Updates
    identical individuals.

    Args:
        ind1: First individual
        ind2: Second individual

    Returns:
        boolean whether individuals are similar
    """
    if my_str(ind1) == my_str(ind2):
        if ind1.age < ind2.age:
            ind1.age = ind2.age
            ind1.fitness = ind2.fitness

            for dataset in _inst.datasetDict:
                setattr(ind1, _inst.datasetDict[dataset]['name'],
                    getattr(ind2, _inst.datasetDict[dataset]['name']))

        elif ind2.age < ind1.age:
            ind2.age = ind1.age
            ind2.fitness = ind1.fitness

            for dataset in _inst.datasetDict:
                setattr(ind2, _inst.datasetDict[dataset]['name'],
                    getattr(ind1, _inst.datasetDict[dataset]['name']))
        return True

    else:
        return False

def almost_equal(val1, val2, accuracy=0.01):
    """Compares val1 to val2 and returns true if within accuracy
    % (default 0.01%)

    Args:
        val1: First value
        val2: Second value

    Returns:
        boolean whether vals are within accuracy specified
    """
    return abs(val1 - val2)/val1 < accuracy/100.0

def my_similar_phenotype(ind1, ind2):
    """Defines whether two individuals are similar or not
    based upon objective scores only if strings do not match.

    Args:
        ind1: First individual
        ind2: Second individual

    Returns:
        boolean whether individuals are similar
    """
    if my_str(ind1) == my_str(ind2):
        return True
    elif ind1.fitness.values == ind2.fitness.values:
        return True
    # Check to see if all values accurate to 0.01%
    elif all(almost_equal(*values) for values in zip(ind1.fitness.values, ind2.fitness.values)):
        return True
    else:
        return False

def clean_individual(individual):
    """
    This method restores all the tracked progress on an individual
    that it may have inheritted from its parent.
    This method works inplace and does not return the individual, but this can be changed
    in the future if we ever need to decorate or something.
    """
    del individual.fitness.values
    for dataset in _inst.datasetDict:
        del getattr(individual, _inst.datasetDict[dataset]['name']).values
    for objective in _inst.objectivesDict:
        setattr(individual, _inst.objectivesDict[objective]['name'], np.ndarray(0))

    individual.retry_time = 0
    individual.elapsed_time = 0
    individual.age = 0
    individual.novelties = []

# Population must be divisible by 4

def master_algorithm(NPOP=12, CXPB=0.90, CXPB_ephemeral=0.01, CXPB_headless_chicken=0,
              CXPB_headless_chicken_ephemeral=0, MUTPB=0.01, MUTLPB=0.01,
              MUTEPHPB=0.01, MUTNRPB=0.01, MUTUPB=0.01, MUTSPB=0.01, OUTP=0.2,
              selections=None, minQueueSize=100, launchSize=100,
              elitePoolSize=1024,
              # seed_file_name=None,
              parents_output=None, pareto_output=None,
              gene_pool_fitness=None, pareto_fitness=None, database_str='sqlite', reuse=None,
              debug=False):
    """Code to run GP.

    This section should probably not need to be modified

    Args:
        NPOP: population number
        CXPB: crossover probability
        CXPB_ephemeral: ephemeral crossover probability
        CXPB_headless_chicken: headless chicken crossover probability
        CXPB_headless_chicken_ephemeral: ephemeral headless chicken crossover
        probability
        MUTPB: insert mutation probability
        MUTLPB: modify learner mutation probability
        MUTEPHPB: ephemeral mutation probability
        MUTNRPB: mutation NR probability
        MUTSPB: mutation S probability
        selection: selection method
        minQueueSize: minimum queue size
        launchSize: starting size of queue
        elitePoolSize: size of elite pool
        seed_file_name: name of seed file
        parents_output: output of parents
        pareto_output: output of pareto front
        gene_pool_fitness: fitness of gene pool
        pareto_fitness: fitness of pareto front
        pool: multiprocessing pool
        debug: whether debug mode is activated
    """
    random.seed(101)
    np.random.seed(101)
    # Set the class debug mode
    _inst.debug = debug

    fitness_names = [_inst.objectivesDict[objective]['name'] for objective in _inst.objectivesDict]
    dataset_names = [_inst.datasetDict[dataset]['name'] for dataset in _inst.datasetDict]

    # Run connection setup
    ConnectionSetup()
    # Initialize database for storing information about each individual
    database = SQLConnectionMaster(connection_str=database_str, reuse=reuse, fitness_names=fitness_names,
                                                dataset_names=dataset_names, statistics_dict=_inst.statistics_dict,
                                                cache_dict=_inst.cacheDict)

    database.add_host('central')

    # Insert modules into SQL table
    loaded=False
    while not loaded:
        try:
            with open("Global_MODS", "rb") as mod_file:
                glob_mods = dill.load(mod_file)
            loaded=True
        except EOFError:
            time.sleep(15)
    for i in glob_mods:
        if "mod_" not in glob_mods[i][0]:
            continue
        mod = glob_mods[i][1]
        data = database.select(mod.mod_num, isModule=True)
        if not data:
            database.insertMod(mod_num=mod.mod_num, age=mod.age, num_occur=mod.num_occur, tree=my_str(mod, mod.name), 
                pickle=cp.deepcopy(mod))
        else:
            database.updateMod(row=data, mod_num=mod.mod_num, age=mod.age, num_occur=mod.num_occur, tree=my_str(mod, mod.name),
            weights=data.weights, pickle=cp.deepcopy(mod))

    try:
        # instead of seeding with pareto, resume run by carrying over parents and elitepool
        assert reuse==1

        # read data about previous run
        with open("metadata.txt","r") as f:
            pid,gens_elapsed = [int(item) for item in f.readlines()]

        # read parents from file
        with open(parents_output + str(pid) + ".txt", "rb") as f:
            parents = get_last_pickled(f)
            first_gen = parents

        # read pareto front from file
        with open(pareto_output+str(pid)+".txt","rb") as f:
            pareto_front = get_last_pickled(f)
        
        elitePool = list(pareto_front)[:elitePoolSize] # for a "true resume", might have to pickle this as well
        
        print("Resumed EMADE instance")
    except:
        print("No EMADE instance to resume. Starting normally")

        # Initialize pareto_front object with function to determine if individuals
        # are similar
        pareto_front = tools.ParetoFront(my_similar_phenotype)

        seeds = database.get_seeded_pareto()

        # Check for seed file, if none initialize population
        if not seeds:
            print('No pareto front to seed')
            print("Building first generation. This can take a few minutes")
            sys.stdout.flush()
            first_gen = _inst.toolbox.population(NPOP)
            print('Got first generation')
            sys.stdout.flush()
        else:
            seed_trees = seeds[:]
            print('Seeded with:')
            print('\n'.join([my_str(i) for i in seed_trees]))
            first_gen = []
            # TODO: keep track of mod names on first gen
            for seed_tree in seed_trees:
                seedling = _inst.toolbox.individual_from_seed(seed_tree[:])
                first_gen.append(seedling)
            if len(first_gen) < NPOP:
                remaining = NPOP - len(first_gen)
                print('Creating', remaining, 'individuals')
                first_gen += _inst.toolbox.population(remaining)
            for ind in first_gen:
                clean_individual(ind)

            sys.stdout.flush()
        _inst.history.update(first_gen)

        invalid_ind_index = [i for i in range(len(first_gen)) if not first_gen[i].fitness.valid]
        print(len(invalid_ind_index), 'to evaluate')
        sys.stdout.flush()

        for i in invalid_ind_index:
            ind = first_gen[i]
            my_hash_id = my_hash(ind)
            ind.hash_val = my_hash_id
            data = database.select(my_hash_id)
            if not data:
                database.insertInd(hash=my_hash_id, tree=my_str(ind), individual=cp.deepcopy(ind),
                                age=ind.age, evaluation_gen=0,
                                evaluation_status=IndividualStatus.NOT_EVALUATED)
            else:
                database.updateInd(row=data, individual=cp.deepcopy(ind), evaluation_gen=0,
                                age=ind.age, evaluation_status=IndividualStatus.NOT_EVALUATED)


        parents = []


        # initialize an empty elite pool
        elitePool = list(map(_inst.toolbox.clone, parents))
        gens_elapsed = 0

        for individual in first_gen:
            layer_frequencies = get_individual_layer_frequencies(individual)
            database.update_layer_frequencies(gens_elapsed, layer_frequencies)
    
    # keep a list of mod names for ease of use later, additionally count num_occurences of mods in individuals
    # TODO: implement with seeding
    for i in first_gen:
        i.modList = []
        for node in i[0]:
            if "mod" in node.name:
                i.modList.append(node.name)
                entry = database.select(int(node.name.split("_")[1]), isModule=True)
                database.updateMod(row=entry, mod_num=entry.mod_num, age=entry.age, tree=entry.tree, pickle=entry.pickle, num_occur=entry.num_occur + 1)
        i.modList.reverse()
        row = database.select(i.hash_val)
        if row:
            database.updateInd(row=row,individual=cp.deepcopy(i),age=i.age, evaluation_status=IndividualStatus.NOT_EVALUATED)
        else:
            database.insertInd(hash=i.hash_val,tree=my_str(i),individual=cp.deepcopy(i),age=i.age, evaluation_gen=0, evaluation_status=IndividualStatus.NOT_EVALUATED)

    # Initialize evaluated pool to be empty
    evaluated_offspring = []

    while gens_elapsed < 10000:
        print('Starting Gen', gens_elapsed)
        sys.stdout.flush()
        if len(parents) > 0: 
            print('Producing offspring')
            sys.stdout.flush()

            # Clone so as not to alter original
            offspring = list(map(_inst.toolbox.clone, parents))

            # clear offspring NNStatistics table tracking
            for child in offspring:
                child.parents = []
                child.intermediate_parent = False

            gen_dict = {"mod_pset": _inst.mod_pset, 
                "pset": _inst.pset, 
                "exprs": _inst.exprs, 
                "toolbox": _inst.toolbox,
                "datasetDict": _inst.datasetDict,
                "objectivesDict": _inst.objectivesDict,
                "data_dim": _inst.data_dims,
                "tries": 0,
                "prev_nodes": []
            }

            offspring = simulateMateMutNN(offspring, gen_dict, CXPB, OUTP, MUTPB, MUTLPB, MUTEPHPB, MUTNRPB, MUTUPB, MUTSPB, isModule=False, isNNLearner=True)
            
            for i in offspring:
                i.modList = []
                for node in i[0]:
                    if "mod" in node.name:
                        i.modList.append(node.name)
                        entry = database.select(int(node.name.split("_")[1]), isModule=True)
                        database.updateMod(row=entry, mod_num=entry.mod_num, age=entry.age, tree=entry.tree, pickle=entry.pickle, num_occur=entry.num_occur + 1)
                i.modList.reverse()
                row = database.select(i.hash_val)
                if row:
                    pass # offspring is a duplicate
                else:
                    database.insertInd(hash=i.hash_val,tree=my_str(i),individual=cp.deepcopy(i),age=i.age, evaluation_gen=0, evaluation_status=IndividualStatus.NOT_EVALUATED)


            # TODO: Pull all mods and do some selection based on their scores
            loaded=False
            while not loaded:
                try:
                    with open("Global_MODS", "rb") as mod_file:
                        glob_mods = dill.load(mod_file)
                    loaded=True
                except EOFError:
                    time.sleep(15)
            mods = [glob_mods[x][1] for x in glob_mods]
            # Don't want to do selection on modules lacking a fitness score
            selectable = [mod for mod in mods if getattr(mod, dataset_names[0]).valid]
            parents = tools.selNSGA2(individuals=selectable, k=math.floor(len(selectable) * 0.8))
            for i in parents:
                setattr(i, 'age', i.age + 1)
            parents = parents + [mod for mod in mods if mod not in selectable]
            # These are the low scoring adfs to get rid of
            badMods = [mod for mod in mods if mod not in parents]

            # TODO: Generate new mods to replace worst mods
            for i in badMods:
                expr, reshape = gp.genADF(_inst.mod_pset, min_=3, max_=5, data_dim=_inst.data_dims, data_type=_inst.datasetDict[0]['type'], type_=_inst.mod_pset.ret)
                tree = gp.PrimitiveTree(expr)
                tree = [tree]
                mod_obj = creator.ADF(tree)
                mod_obj.mod_num = i.mod_num
                mod_obj.age = 0
                mod_obj.num_occur=0
                mod_obj.name = i.name
                glob_mods[i.name] = [i.name, mod_obj, list(reshape)]

            # TODO: Run mate and mutation for mods
            parents = simulateMateMutNN(parents, gen_dict, CXPB, OUTP, MUTPB, MUTLPB, MUTEPHPB, MUTNRPB, MUTUPB, MUTSPB, isModule=True, isNNLearner=False)
            
            # TODO: Update ADFS file with new glob_mods and update SQL table for mods
            # Since I'm using an integer as the key for ADF's in order to keep old adfs relevant in the table I'm generating a new mod num
            # using the total num of mods and gens_elapsed to distinguish old individuals. Not sure how good it will work but experimenting with it
            for old_mod in badMods:
                if gens_elapsed < 10:
                    new_num = int(str(old_mod.mod_num)+"0"+str(gens_elapsed))
                else:
                    new_num = int(str(old_mod.mod_num)+str(gens_elapsed))
                database.insertMod(mod_num=new_num, age=old_mod.age, num_occur=old_mod.num_occur, tree=my_str(old_mod, old_mod.name), pickle=cp.deepcopy(old_mod))
            
            for mod in parents:
                entry = database.select(int(mod.name.split("_")[1]), isModule=True)
                database.updateMod(row=entry, mod_num=entry.mod_num, age=mod.age, tree=my_str(mod, mod.name), pickle=cp.deepcopy(mod), num_occur=mod.num_occur)
                glob_mods[mod.name][1] = mod

            with open("Global_MODS", "wb") as mod_file:
                dill.dump(glob_mods, mod_file)
            
            # TODO: Loop through offspring and update individuals contexts for new mods
            for ind in offspring:
                for mod in glob_mods:
                    func = gp.compile(glob_mods[mod][1][0], _inst.mod_pset)
                    i.pset.context.update({glob_mods[mod][0]: func})

            gene_pool_hashes = [individual.hash_val for individual in offspring]
            print('After cloning from parents', len(np.unique(gene_pool_hashes)), 'out of', len(gene_pool_hashes), 'are unique')
            print('Additionally', len(np.unique([id(individual) for individual in offspring])), 'individuals are unique in memory')
            all_ids = []
            [all_ids.extend([id(tree) for tree in individual]) for individual in offspring]
            print('And', len(np.unique(all_ids)), 'mains and mods are unique')

            inds_to_remove = []
            # Check to see if individual has already been computed before sending it off to be evaluated
            for ind_num in np.arange(len(offspring)):
                ind = offspring[ind_num]
                # If the fitness is not valid it has not already been computed
                if not ind.fitness.valid:
                    # Compute a hash based on the string representation of the individual
                    my_hash_id = my_hash(ind)
                    ind.hash_val = my_hash_id

                    individual = database.select(my_hash_id)
                    # If the individual is in the database and has been processed, let's extract it
                    if individual and (individual.evaluation_status == IndividualStatus.WAITING_FOR_MASTER or individual.evaluation_status == IndividualStatus.EVALUATED):
                        data = cp.deepcopy(individual)
                        offspring[ind_num] = data.pickle
                        print('Already computed', my_str(offspring[ind_num]),
                              'with fitness', str(offspring[ind_num].fitness.values), 'hash', offspring[ind_num].hash_val, 'and age',
                              str(offspring[ind_num].age))
                        if len(offspring[ind_num].fitness.values) == 0:
                            print('Bad fitness for hash', ind.hash_val)
                        sys.stdout.flush()
                    # If we get here, we know that the individual is being evaluated already and we don't want to do it again
                    elif individual:
                        inds_to_remove += [ind_num]
                        print('I am not going to evaluate', my_str(offspring[ind_num]),
                              'with fitness', str(offspring[ind_num].fitness.values), 'hash', offspring[ind_num].hash_val, 'and age',
                              str(offspring[ind_num].age), 'because I am waiting for it to complete')

            # Remove individuals that are already out for evaluation
            print('Offspring has', len(offspring), 'elements before I remove those that are already being processed')
            print('Removing hashes:')
            print([offspring[ind_num].hash_val for ind_num in np.arange(len(offspring)) if ind_num in inds_to_remove])
            offspring = [offspring[ind_num] for ind_num in np.arange(len(offspring)) if ind_num not in inds_to_remove]
            print('Removed', len(inds_to_remove), 'individuals that were already being processed left with', len(offspring))

            # TODO Let's also keep track of individuals being sent off so we don't evaluate the same items over and over while we wait for them to return
            gene_pool_hashes = [individual.hash_val for individual in offspring]
            print('After matings and mutations', len(np.unique(gene_pool_hashes)), 'out of', len(gene_pool_hashes), 'are unique')

            print('Additionally', len(np.unique([id(individual) for individual in offspring])), 'individuals are unique in memory')
            all_ids = []
            [all_ids.extend([id(tree) for tree in individual]) for individual in offspring]
            print('And', len(np.unique(all_ids)), 'mains and mods are unique')

            valid_ind = [ind for ind in offspring if ind.fitness.valid]
            print('Adding', len(valid_ind), 'to evaluated offspring list of length', len(evaluated_offspring))
            evaluated_offspring.extend(valid_ind)
            print('Making length', len(evaluated_offspring))
            invalid_ind = []
            # Build up queue to send individuals off for evaluation, here we want to see if the individual is invalid AND not in the list more than once
            invalid_hashes = []
            for ind in offspring:
                if not ind.fitness.valid and ind.hash_val not in invalid_hashes:
                    invalid_ind.append(ind)
                    invalid_hashes.append(ind.hash_val)
            print('Recomputing fitnesses for', len(invalid_ind), 'offspring') #, adding to', len(fitness_launches), 'launches')

            # Add newly created individuals to database
            for ind in invalid_ind:
                my_hash_id = ind.hash_val
                data = database.select(my_hash_id)
                if not data:
                    database.insertInd(hash=my_hash_id, tree=my_str(ind), individual=cp.deepcopy(ind),
                                    age=ind.age, evaluation_gen=gens_elapsed,
                                    evaluation_status=IndividualStatus.NOT_EVALUATED)

                    # Add this individual's layer frequencies to the Statistics table.
                    layer_frequencies = get_individual_layer_frequencies(ind)
                    database.update_layer_frequencies(gens_elapsed, layer_frequencies)
                else:
                    print('I am trying to update individual', my_str(ind), 'with hash', ind.hash_val, 'but maybe I should not be',
                            'status is', data.evaluation_status)

            print('Querying database for elements remaining in queue')
            sys.stdout.flush()
            start_query = time.time()
            count = database.get_unevaluated()
            print(str(count), 'elements remaining in queue, query complete in', '%.2f' % (time.time()-start_query), 'seconds')
            sys.stdout.flush()

            # Optimize cache
            print("Starting cache optimization")
            sys.stdout.flush()
            database.optimize_cache('central', _inst.cacheDict['cacheLimit'])
            print("Finished cache optimization")
            sys.stdout.flush()

        print('Querying database for elements remaining in queue')
        sys.stdout.flush()
        start_query = time.time()
        count = database.get_unevaluated()
        print(str(count), 'elements remaining in queue, query complete in', '%.2f' % (time.time()-start_query), 'seconds')
        sys.stdout.flush()

        # Produce new offspring if queue is low
        if (count < minQueueSize):
            # Initialize evaluated pool to be empty
            evaluated_offspring = []
            
            print('Updating population')
            sys.stdout.flush()

            # Perform NSGA2 Selection Algorithm
            # Add offspring to existing population
            print('Passed parents update')
            sys.stdout.flush()

            # Add recently evaluated individuals to evaluated_offspring
            evaluated_offspring.extend(database.get_evaluated_individuals())

            for evaluatedIndividual in evaluated_offspring:
                print("TimeStamp | " + str(datetime.now()))
                print('Received: ' + my_str(evaluatedIndividual))
                print('\tWith Hash', evaluatedIndividual.hash_val)
                print('\tWith Fitnesses: ' + str(evaluatedIndividual.fitness.values))
                print('\tWith Age: ' + str(evaluatedIndividual.age))

            # Add the elite pool to the currently evaluated offspring
            gene_pool = elitePool + evaluated_offspring

            # Noticing a high level of repeated genomes in selection process
            # For debugging let's print out some info on redundant genomes at this point
            print('Elite Pool has:', len(elitePool), 'individuals, and', len(np.unique([ind.hash_val for ind in elitePool])), 'are unique')
            print('evaluated offspring has:', len(evaluated_offspring), 'individuals, and', len(np.unique([ind.hash_val for ind in evaluated_offspring])), 'are unique')
            print('Gene Pool has:', len(gene_pool), 'individuals, and', len(np.unique([ind.hash_val for ind in gene_pool])), 'are unique')
            sys.stdout.flush()

            for individual in gene_pool:
                if len(individual.fitness.values) == 0:
                    print('Bad fitness before redundancy check!')

            # At this point let's remove the redundant elements
            hashes = []
            non_redundant_gene_pool = []
            for ind in gene_pool:
                ind_hash = ind.hash_val
                if ind_hash not in hashes:
                    # We need to go back to the hash table here to make sure we have the most current version!
                    # Because I am getting a strange error where I have individuals in my gene pool that are not in my hash table
                    # I am putting a try/except
                    try:
                        non_redundant_gene_pool.append(cp.deepcopy(database.select(ind_hash).pickle))
                        if len(non_redundant_gene_pool[-1].fitness.values) == 0:
                            print('Bad fitness', ind_hash)

                        hashes.append(ind_hash)
                    except Exception as e:
                        print('Hash not in hash table???', ind_hash)
                        print(my_str(ind))
                        print(ind.fitness.values)
                        print(ind.age)
                        sys.stdout.flush()
                        raise e
                else:
                    print('Got Redundant', my_str(ind), ind.fitness.values, ind_hash)
            # Reassign the genepool
            gene_pool = non_redundant_gene_pool
            # For debugging let's print out some info on redundant genomes at this point
            print('Sanity check after removal of redundant individuals')
            print('Elite Pool has:', len(elitePool), 'individuals, and', len(np.unique([ind.hash_val for ind in elitePool])), 'are unique')
            print('evaluated offspring has:', len(evaluated_offspring), 'individuals, and', len(np.unique([ind.hash_val for ind in evaluated_offspring])), 'are unique')
            print('Gene Pool has:', len(gene_pool), 'individuals, and', len(np.unique([ind.hash_val for ind in gene_pool])), 'are unique')
            sys.stdout.flush()

            # Add recently evaluated individuals to History table
            database.add_history(gens_elapsed=gens_elapsed, hashes=np.unique([ind.hash_val for ind in evaluated_offspring]))

            for individual in gene_pool:
                if len(individual.fitness.values) == 0:
                    print('Bad fitness before database update!')

            # Change individuals from WAITING_FOR_MASTER to EVALUATED
            for evaluatedIndividual in evaluated_offspring:
                my_hash_id = evaluatedIndividual.hash_val
                data = database.select(my_hash_id)
                
                database.updateInd(row=data,
                                individual=cp.deepcopy(evaluatedIndividual),
                                age=evaluatedIndividual.age,
                                evaluation_status=IndividualStatus.EVALUATED)
                if (my_str(evaluatedIndividual).__contains__("NNLearner")):
                    cur_nnstatistics = database.selectNN(my_hash_id, evaluatedIndividual.age)
                    if cur_nnstatistics is None:
                        database.insertNN(hash=my_hash_id,
                            age=evaluatedIndividual.age,
                            parents=','.join(evaluatedIndividual.parents) if hasattr(evaluatedIndividual, 'parents') else '',
                            curr_tree=my_str(evaluatedIndividual),
                            individual=cp.deepcopy(evaluatedIndividual),
                            error_string=data.error_string
                        )
                    else:
                        database.updateNN(
                            row=data,
                            individual=cp.deepcopy(evaluatedIndividual),
                            age=evaluatedIndividual.age,
                            parents=','.join(evaluatedIndividual.parents) if hasattr(evaluatedIndividual, 'parents') else '',
                            error_string=data.error_string
                        )

            for dataset in _inst.datasetDict:
                print(len([ind for ind in gene_pool if (ind.age == dataset+1 or ind.age == dataset+1.5)]), _inst.datasetDict[dataset]['name'], 'individuals')

            print('Gene pool assembled')
            sys.stdout.flush()

            # Clear evaluated_offspring List
            evaluated_offspring = []

            # Adjust fitness by data set
            gene_pool = list(map(_inst.toolbox.adjustForDataset, gene_pool))

            gene_pool_hashes = [individual.hash_val for individual in gene_pool]
            print('After adjust for data set', len(np.unique(gene_pool_hashes)), 'out of', len(gene_pool_hashes), 'are unique')

            # Perform NSGA sorting and crowding distance calculations
            try:
                gene_pool = _inst.toolbox.selectNSGA2(gene_pool, len(gene_pool))
            except Exception as e:
                print(e)
                print(len(gene_pool), 'Elements in gene pool')
                for i, individual in enumerate(gene_pool):
                    print(i, my_str(individual), individual.fitness.values)
                raise e
            gene_pool_hashes = [individual.hash_val for individual in gene_pool]
            print('After select nsga2', len(np.unique(gene_pool_hashes)), 'out of', len(gene_pool_hashes), 'are unique')

            print('NSGAII Completed')
            sys.stdout.flush()

            for age in _inst.datasetDict:
                print(len([ind for ind in gene_pool if (ind.age == age+1 or
                    ind.age == age+1.5)]), _inst.datasetDict[age]['name'], 'individuals')

            print(len(gene_pool), 'total individuals')

            print(database.get_num_evaluated(), 'individuals evaluated thus far')
            sys.stdout.flush()

            # Only gets executed when multiple datasets specified
            # Identifies individual that are good at one or both datasets 
            for dataset in np.arange(0, len(_inst.datasetDict.keys())-1):
                individuals_to_age = [
                    ind
                    for ind in gene_pool
                    if ind.age == dataset+1
                    ]
                well_rounded, specialists = _inst.toolbox.genePoolPartition(
                    individuals_to_age,
                    N=launchSize /(2.0*(dataset+1)),
                    dataset=_inst.datasetDict[dataset]['name'])
                individuals_to_add = well_rounded + specialists
                print('Adding', len(individuals_to_add), 'to evaluate to the next data set')
                sys.stdout.flush()

                # Set individuals evaluating higher datasets to NOT_EVALUATED
                for ind in individuals_to_add:
                    ind.age += .5
                    my_hash_id = ind.hash_val
                    data = database.select(my_hash_id)
                    if not data:
                        database.insertInd(hash=my_hash_id, tree=my_str(ind), individual=cp.deepcopy(ind),
                                        age=ind.age, evaluation_gen=gens_elapsed+1, evaluation_start_time=None,
                                        evaluation_status=IndividualStatus.NOT_EVALUATED)
                    else:
                        database.updateInd(row=data, individual=cp.deepcopy(ind),
                                        evaluation_gen=gens_elapsed+1, age=ind.age, evaluation_start_time=None,
                                        evaluation_status=IndividualStatus.NOT_EVALUATED)

                print('fitness launches increasing to', database.get_unevaluated())
                sys.stdout.flush()

                print('Added launches for', _inst.datasetDict[dataset]['name'], 'to', _inst.datasetDict[dataset+1]['name'])
                sys.stdout.flush()

            gene_pool_hashes = [individual.hash_val for individual in gene_pool]
            print('Right before binary tournament', len(np.unique(gene_pool_hashes)), 'out of', len(gene_pool_hashes), 'are unique')

            if len(gene_pool) < launchSize:
                print(f'WARNING: Launch total {launchSize} larger than gene pool {len(gene_pool)}')
                sys.stdout.flush()

            if selections is not None and len(selections) > 0:
                selection_name = list(selections)[0]
                # Create argument dictionary for selection method
                selection_dict = selections[selection_name]
                all_selection_args = dict(selection_dict['args'])
                if selection_dict['dynamic_args']:
                    dynamic_args = {}
                    dynamic_args['gens_elapsed'] = gens_elapsed
                    dynamic_args['acceptable'] = _inst.LROI
                    dynamic_args['goal'] = _inst.SROI
                    dynamic_args['psets'] = _inst.psets
                    dynamic_args['datasetDict'] = _inst.datasetDict
                    all_selection_args['dynamic_args'] = dynamic_args
                fun = selection_dict['fun']
                # Call selection method with required and optional arguments
                # len(parents) == launchSize
                parents = fun(individuals=gene_pool, k=launchSize, **all_selection_args)
            else:
                parents = tools.selNSGA2(individuals=gene_pool, k=launchSize)

            print('Parents Selected', len(parents), 'from', len(gene_pool))
            sys.stdout.flush()

            ####
            # James Rick
            # Updates the Individual Pickle object which is stored on the database.
            # This is required for tracking an individuals novelty over time.
            # Allows for individuals from earlier generations to keep same hash value but change actual indvidiual object
            for ind in gene_pool:
                sql_table_row = database.select(ind.hash_val)
                if not sql_table_row:
                    print('WARNING: This Individual is not in the database.\
                    \nHash:\n%s\nIndividual:\n%s' % (ind.hash_val, my_str(ind)))
                else:
                    print("Hash: %s" % (ind.hash_val))
                    print("sql_table_row.pickle.novelties == ind.novelties: ",
                        sql_table_row.pickle.novelties == ind.novelties, flush=True)
                    database.updateInd(row=sql_table_row, individual=cp.deepcopy(ind), age=ind.age,
                            evaluation_status=sql_table_row.evaluation_status,
                            evaluation_gen=sql_table_row.evaluation_gen,
                            elapsed_time=sql_table_row.elapsed_time,
                            retry_time=sql_table_row.retry_time,
                            evaluation_start_time=sql_table_row.evaluation_start_time,
                            error_string=sql_table_row.error_string)
                    check_table_row = database.select(ind.hash_val)
                    print("check_table_row.pickle.novelties == ind.novelties: ",
                        check_table_row.pickle.novelties == ind.novelties, flush=True)
            ####

            parent_hashes = [parent.hash_val for parent in parents]
            print(len(np.unique(parent_hashes)), 'out of', len(parent_hashes), 'are unique')
            sys.stdout.flush()

            stats_vals = {}
            for stat_name in _inst.statistics_dict:
                stat_dict = _inst.statistics_dict[stat_name]
                val = None
                try:
                    val = stat_dict['fun'](parents)
                except Exception as e:
                    print('Statistic {} could not be evaluated:'.format(stat_name))
                    print(e)
                    sys.stdout.flush()
                stats_vals[stat_name] = val
            database.insert_statistics(gens_elapsed, stats_vals)

            # If there are enough individuals in the gene pool we should
            if len(gene_pool) > elitePoolSize:
                elitePool = _inst.toolbox.spea2(gene_pool, len(gene_pool))
                print([ind.fitness.crowding_dist for ind in elitePool])
                sys.stdout.flush()
                elitePool = elitePool[:elitePoolSize]
            else:
                print('WARNING: Elite pool larger than gene pool')
                print('Gene Pool Size', len(gene_pool))
                sys.stdout.flush()
                elitePool = gene_pool

            print('Updating Pareto Front')
            sys.stdout.flush()

            pareto_front.update(gene_pool)
            print('Pareto Front Updated')
            sys.stdout.flush()

            # Add pareto front individuals to ParetoFront table
            database.add_pareto_front(gens_elapsed=gens_elapsed, hashes=np.unique([ind.hash_val for ind in pareto_front]))


            print('Pareto Front Updated')
            sys.stdout.flush()


            # dump pareto front into file
            with open(pareto_output + str(os.getpid()) + '.txt', 'wb') as f:
                pickle.dump(pareto_front, f, -1)

            # dump parents into file
            with open(parents_output + str(os.getpid()) + '.txt', 'wb') as f:
                pickle.dump(parents, f, -1)

            # genePoolFitnessValues = np.array([i.fitness.values
            #                                   for i in gene_pool])
            # genePoolFitnessValues = np.hstack((gens_elapsed * np.ones(
            #     (len(genePoolFitnessValues), 1)), genePoolFitnessValues))
            if len(pareto_front) > 0:
                paretoFitnessValues = np.array([i.fitness.values
                                                for i in pareto_front])
                paretoFitnessValues = np.hstack((gens_elapsed * np.ones(
                    (len(paretoFitnessValues), 1)), paretoFitnessValues))

                with open(pareto_fitness + str(os.getpid()) + '.txt', 'ab') as f:
                    np.savetxt(f, paretoFitnessValues, delimiter=',')

            try:
                _inst.logHyperVolume(pareto_front, gens_elapsed)
            except Exception as e:
                print("Got error calculating hypervolume. Error:", e)

            if _inst.debug:
                for i in np.arange(len(pareto_front)):
                    print('Pareto Individual ' + str(i) + ' after gen ' +
                          str(gens_elapsed) + ' is ' + my_str(pareto_front[i]) +
                          str(pareto_front[i].fitness.values) + ' Age ' +
                          str(pareto_front[i].age))
                sys.stdout.flush()

            print('Updated Elite Pool')
            sys.stdout.flush()

            # Update statistics at the end of the generation
            database.create_db_statistics('central')

            print('Finished Calculating Statistics')
            sys.stdout.flush()

            gens_elapsed += 1
            with open("metadata.txt","w") as f:
                f.write("\n".join([str(os.getpid()), str(gens_elapsed)]))
        else:
            parents = []
            print('Good night')
            sys.stdout.flush()
            time.sleep(10)
            print('Good morning')
            sys.stdout.flush()

    try:
        database.close()
    except Exception as e:
        print("Error occurred while closing DB Session. TimeStamp:", datetime.now(), "Exception:", e)
        sys.stdout.flush()

# Population must be divisible by 4
def worker_algorithm(pool=None, database_str='sqlite', reuse=None, num_workers=1, debug=False):
    """Code to run GP.

    This section should probably not need to be modified

    Evaluates individuals populated by the master

    Args:
        pool: multiprocessing pool
        database_str: string detailing how to make sql connection
        reuse: True if database is being re-used, False otherwise
        num_workers: size of multiprocessing pool
        debug: whether debug mode is activated
    """
    print('Starting Worker')
    sys.stdout.flush()

    #pingGPU()

    random.seed(101)
    np.random.seed(101)
    # Set the class debug mode
    _inst.debug = debug

    # Get names of objectives and datasets from dictionaries to send to database connection
    fitness_names = [_inst.objectivesDict[objective]['name'] for objective in _inst.objectivesDict]
    dataset_names = [_inst.datasetDict[dataset]['name'] for dataset in _inst.datasetDict]

    # Run connection setup
    ConnectionSetup()
    # Initialize database for storing information about each individual
    database = SQLConnectionMaster(connection_str=database_str, reuse=reuse, fitness_names=fitness_names,
                                                dataset_names=dataset_names, statistics_dict=_inst.statistics_dict,
                                                cache_dict=_inst.cacheDict, is_worker=True)

    # database.add_host('worker' + str(random.randint(0, 10000)))

    # Number of individuals to pull from database dependent on number of workers
    evals_per_worker = 5
    # List of ApplyResult objects generated from pool.apply_async
    fitness_launches = []
    # List of unevaluated Individual objects matching the ApplyResult objects
    uneval_ind = []

    while True:
        # Get more individuals to evaluate if less than 25% of individuals remaining to evaluate
        if len(fitness_launches) / (evals_per_worker * num_workers) < .25:
            # Within get_random_uneval, the selected individuals will be marked as inprogress in the database
            random_uneval = database.get_random_uneval_individuals(evals_per_worker * num_workers)
            print(len(random_uneval), 'New Individuals to evaluate')
            sys.stdout.flush()

            prev_generation = database.num_gens()
            prev_generation_layer_frequencies = database.get_layer_frequencies(prev_generation)
            fitness_launches.extend([pool.apply_async(_inst.toolbox.evaluate, (ind, int(ind.age), prev_generation_layer_frequencies)) for ind in random_uneval])
            # TODO query & pass ADFs and model/weights
            uneval_ind.extend(random_uneval)

        print("Worker Pool Size: " + str(len(fitness_launches)) + " | TimeStamp: " + str(datetime.now()))
        sys.stdout.flush()

        i = 0
        # Check if individuals are evaluated and add them to the database
        while i < len(fitness_launches):
            eval_future = fitness_launches[i]
            print("Checking Status of Individual: " + str(uneval_ind[i].hash_val) + " | TimeStamp: " + str(datetime.now()))
            sys.stdout.flush()
            if eval_future.ready():
                evaluatedIndividual, elapsed_time, error_string, retry = eval_future.get()
                # TODO get weights here, write them to the database if retry is false
                # FAR FUTURE TODO: Make sure ADF scoring is compatible with multiple dataset situations

                loaded=False
                while not loaded:
                    try:
                        with open("Global_MODS", "rb") as mod_file:
                            glob_mods = dill.load(mod_file)
                        loaded=True
                    except EOFError:
                        time.sleep(15)
                # Update ADF scores by averaging with existing score or initializing new score
                for modules in evaluatedIndividual.modList:
                    glob_mods[modules][1].age += 1
                    mod = database.select(int(modules.split("_")[1]), isModule=True)
                    for dataset in dataset_names:
                        if glob_mods[modules][1].fitness.values:
                            modInd = glob_mods[modules][1]
                            score = getattr(modInd, dataset)
                            indScore = getattr(evaluatedIndividual, dataset)
                            sumScore = tuple(map(lambda x, y: x + y, score.values, indScore.values))
                            modInd.fitness.values = tuple(x/2 for x in sumScore)
                            setattr(glob_mods[modules][1], dataset, cp.deepcopy(modInd.fitness))
                        else:
                            setattr(glob_mods[modules][1], dataset, cp.deepcopy(evaluatedIndividual.fitness))
                    database.updateMod(row=mod, mod_num=mod.mod_num, age=glob_mods[modules][1].age, tree=my_str(glob_mods[modules][1][0], glob_mods[modules][0]), pickle=cp.deepcopy(glob_mods[modules][1]), num_occur=mod.num_occur, weights=glob_mods[modules][1].weights)
                    glob_mods[modules][1] = cp.deepcopy(glob_mods[modules][1])
                with open("Global_MODS", "wb") as mod_file:
                    dill.dump(glob_mods, mod_file)

                fitness_launches.pop(i)
                uneval_ind.pop(i)

                print("Removed Individual from Worker Pool | TimeStamp: " + str(datetime.now()))
                sys.stdout.flush()

                my_hash_id = evaluatedIndividual.hash_val
                data = database.select(my_hash_id)

                if not data:
                    if retry:
                        database.insertInd(hash=my_hash_id, tree=my_str(evaluatedIndividual), individual=cp.deepcopy(evaluatedIndividual),
                                        age=evaluatedIndividual.age, elapsed_time=elapsed_time, retry_time=evaluatedIndividual.retry_time,
                                        evaluation_status=IndividualStatus.NOT_EVALUATED,
                                        error_string=error_string)
                    else:
                        database.insertInd(hash=my_hash_id, tree=my_str(evaluatedIndividual), individual=cp.deepcopy(evaluatedIndividual),
                                        age=evaluatedIndividual.age, elapsed_time=elapsed_time, retry_time=evaluatedIndividual.retry_time,
                                        evaluation_status=IndividualStatus.WAITING_FOR_MASTER,
                                        error_string=error_string)
                else:
                    if retry:
                        database.updateInd(row=data, individual=cp.deepcopy(evaluatedIndividual),
                                        age=evaluatedIndividual.age, elapsed_time=elapsed_time,
                                        retry_time=evaluatedIndividual.retry_time, evaluation_start_time=None,
                                        evaluation_status=IndividualStatus.NOT_EVALUATED, error_string=error_string)
                    else:
                        database.updateInd(row=data, individual=cp.deepcopy(evaluatedIndividual),
                                        age=evaluatedIndividual.age, elapsed_time=elapsed_time, retry_time=evaluatedIndividual.retry_time,
                                        evaluation_status=IndividualStatus.WAITING_FOR_MASTER,
                                        error_string=error_string)

                if retry:
                    print("Individual Sent Back for Evaluation | " + "Worker Pool Size: " + str(len(fitness_launches)) + " | TimeStamp: " + str(datetime.now()))
                    print('Received: ' + my_str(evaluatedIndividual))
                    print('\tWith Hash', my_hash_id)
                    print('\tComputed in: ' + str(elapsed_time) + ' seconds')
                    print('\tWith Age: ' + str(evaluatedIndividual.age))
                else:
                    print("Individual Evaluated | " + "Worker Pool Size: " + str(len(fitness_launches)) + " | TimeStamp: " + str(datetime.now()))
                    print('Received: ' + my_str(evaluatedIndividual))
                    print('\tWith Hash', my_hash_id)
                    print('\tComputed in: ' + str(elapsed_time) + ' seconds')
                    print('\tWith Fitnesses: ' + str(evaluatedIndividual.fitness.values))
                    print('\tWith Age: ' + str(evaluatedIndividual.age))
                    if error_string:
                        print('\tWith Error: ', error_string)
                sys.stdout.flush()

                for dataset in _inst.datasetDict:
                    if getattr(evaluatedIndividual, _inst.datasetDict[dataset]['name']).valid:
                        print(_inst.datasetDict[dataset]['name'], str(
                            getattr(evaluatedIndividual, _inst.datasetDict[dataset]['name']).values))
                sys.stdout.flush()
            else:
                i += 1
        time.sleep(10)


def HCDScaleMin(individual):
    """Function to scale the fitness scores of an individual by its distance to
    the hypercube defining the region of interest

    Args:
        individual: the individual in question

    Returns:
        crowding distance
    """
    # Define LROI to be hypercube defined by origin and the following point,
    # minimum acceptable objective values
    LROI = _inst.LROI

    # Define SROI to be hypercube defined by origin and the following point,
    # region describing where solutions are known to be possible
    SROI = _inst.SROI

    myFitness = np.array(individual.fitness.values)
    q = len(myFitness)
    alpha = 0.5
    # Holder coefficient p
    p = 2.0
    HCD = 0.0
    for j in np.arange(q):
        # If fitness is worse than known solution fitness
        if myFitness[j] >= SROI[j]:
            HCD += 1 ** p

        # Fitness is somewhere between known solutions and acceptable
        # objectives
        elif myFitness[j] > LROI[j]:
            HCD += ((myFitness[j] - LROI[j]) / (SROI[j] - LROI[j])) ** p

        # Fitness is better than acceptable objectives
        elif myFitness[j] <= LROI[j]:
            HCD += 0 ** p

        else:
            print('Something is not quite right')
            sys.stdout.flush()
    HCD = HCD ** (1.0 / p)
    HCD = (alpha + HCD) / (1.0 + alpha)
    GHCD = 1.0 / HCD
    # We are trying to maximize crowding distance
    newCrowdingDist = individual.fitness.crowding_dist * GHCD

    return newCrowdingDist


class StdOutQueue(Queue):
    def __init__(self, *args, **kwargs):
        ctx = mp.get_context()
        Queue.__init__(self, *args, **kwargs, ctx=ctx)

    def write(self, msg):
        self.put(msg)

    def flush(self):
        sys.__stdout__.flush()


def get_stack_from_queue(my_queue):
    """Returns a string representation of a queue of strings

    Args:
        my_queue: queue to generate stack from

    Returns:
        stack from queue
    """
    stack_str =  ['Stack Trace:']
    while not my_queue.empty():
        stack_str.append(my_queue.get())
    stack_str = ' '.join(stack_str)
    return stack_str

def handleWorker(func, dataPair, input, return_dict, my_queue, ind_hash):
    """
    Method to wrap a function evaluation.
    Required for memory monitoring and
    management.
    Takes in a function to evaluate (compiled individual), a
    data pair object, and a shared return dictionary where the result
    will be stored.
    """
    def sigterm_handler_sub(signal, frame):
        print("Starting Terminate Signal Handler. PID:", os.getpid(), "TimeStamp:", datetime.now())
        sys.stdout.flush()
        try:
            # Handle caching errors if caching is on
            if dataPair.get_caching_mode():
                # Remove in-progress cache data if it exists
                cache_id = cleanup_trash(ind_hash)
                # Cleanup subprocess db session
                connection = dataPair.get_connection()
                # Set cache row to dirty if the row exists
                if cache_id is not None:
                    connection.set_dirty_rollback(cache_id, os.getpid(), 'central')
                # Close the session/connection with the database
                try:
                    connection.commit()
                except Exception as f:
                    print("Session commit failed in terminate handler. Error:", f)
                try:
                    connection.close()
                except Exception as f:
                    print("Session close failed in terminate handler. Error:", f)

        except Exception as e:
            print("Exception occurred during Terminate Signal handler:", e)

        print("Finished Terminate Signal Handler. PID:", os.getpid(), "TimeStamp:", datetime.now())
        sys.stdout.flush()
        os.kill(os.getpid(), 9)

    signal.signal(signal.SIGTERM, sigterm_handler_sub)

    try:
        ConnectionSetup()

        # temporarily redirect standard out for debugging memory errors
        # sys.stdout = my_queue
        dataPair.set_ind_hash(ind_hash)

        # Running evaluation in separate thread so the main thread can always receive signals
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, dataPair, input)
                result = future.result()
        except Exception as e:
            print(e)
            raise e

        # Commenting out multistream code, we will use this later
        #if _inst.datasetDict[0]['type'] == 'multistreamdata':
        #    test_data_classes = np.array([inst.get_target()
        #                                  for
        #                                  inst in result.get_test_data().get_instances()])
        #    test_data_classes = test_data_classes
        #else:
        #    test_data_classes = np.array([inst.get_target()
        #                                  for
        #                                  inst in result.get_test_data().get_instances()])
        #    test_data_classes = test_data_classes.reshape((-1,))
        #$test_data_classes = [inst.get_target() for inst in result.get_test_data().get_instances()]
        test_data_classes = result.get_test_data().get_target()
        # TODO get model weights from datapair (result) and write to return_dict
        return_dict['result'] = test_data_classes
        return_dict['num_params'] = result.get_num_params()
        os.kill(os.getpid(), 9)
    except (KeyboardInterrupt, Exception) as e:
        if not isinstance(e, KeyboardInterrupt):
            #Store the error as a string
            return_dict['error'] = str(e) + traceback.format_exc()


def evaluate(*inputTuple):
    # TODO handle extra args
    individual = inputTuple[0]
    dataset = inputTuple[1]
    prev_generation_layer_frequencies = inputTuple[2]
    return evaluate_individual(individual, dataset, prev_generation_layer_frequencies)

def evaluate_individual(individual, dataset, prev_generation_layer_frequencies):
    """Shell for evaluation function

    Returns a tuple, containing a tuple of fitness and the individual

    Overall Error, class0Err, class1Err
    class error is the misclassification of a given class,
    e.g. class0Err, is items that should have been 0, but were
    classified otherwise

    Args:
        individual: gp primitive tree
        dataset: type of dataset

    Returns:
        an individual and its fitness
    """
    # Let's experiment with squelching standard out and standard error in order to prevent buffer from filling
    #sys_stdout = sys.stdout
    #sys_stderr = sys.stderr
    #sys.stdout = sys.stderr = open(os.devnull, 'w')
    error_string = ''
    random.seed(101)
    np.random.seed(101)
    #import tensorflow as tf
    #print('evaluate ind gpu check')
    #print(tf.test.is_gpu_available())
    ind_hash = individual.hash_val

    # lower_thresholds = _inst.LROI

    start_time = time.time()

    objectives = np.array([0.0 for objective in _inst.objectivesDict])
    # create a fatal fitness for the minimization scheme
    fatal_fitness = np.array([np.inf for objective in objectives])

    # initialize monte carlo objectives to be 0, this is for a minimization scheme
    monte_carlo_objectives = np.array([0.0 for objective in _inst.objectivesDict])

    # 2 represents possibility of stream_to_features... must have a 2.
    typ = _inst.datasetDict[dataset]['type']
    tree = my_str(individual)
    valid = True

    print("Starting Evaluation of Individual: " + tree)
    print("Hash of the Individual: " + ind_hash)
    print("TimeStamp | " + str(datetime.now()))
    sys.stdout.flush()

    if typ == 'streamdata':
        valid = all(i in tree for i in ['Learner', '2'])
    elif typ in ['featuredata', 'imagedata']:
        valid = all(i in tree for i in ['Learner']) or all (i in tree for i in ['NNLearner'])
    elif typ in ['filterdata', 'detectiondata', 'pickledata']:
        valid = any(i in tree for i in ['CopyStreamToTarget', 'FilterCentroids'])
    elif typ == 'textdata':
        valid = all(i in tree for i in ['Learner', 'Vectorizer']) or all(i in tree for i in ['NNLearner']) or all(i in tree for i in ['Learner', 'Sentiment'])
    elif type == 'summarydata':
        valid == all(i in tree for i in ['Multi'])
    if not valid:
        error_string += 'Tree missing valid primitive for data type\n'
        monte_carlo_objectives = fatal_fitness
    else:
        # TODO: 
        # for each mod string: 
        #   look for match(es) in individual (if it's not a primitive tree object, have to wrap it in primtivetree() then call __str__())
        #   when looking for match, get index of first match, then start looking again at index+len(string) until at end
        #   get number of keras layers in mod
        #   convert string matches to start and end indices of layerlist
        func = _inst.toolbox.compile(expr=individual)

        try:
            # Iterate over data pairs and corresponding truth data for
            # the monte carlo run
            for my_data_pair, my_truth_data in zip(
                    _inst.datasetDict[dataset]['dataPairArray'],
                    _inst.datasetDict[dataset]['truthDataArray']):
                # Copy the objects to avoid damaging the original data

                my_data_pair = cp.deepcopy(my_data_pair)
                my_truth_data = cp.deepcopy(my_truth_data)
                # TODO: write ADF indices and weights to datapair so that NNLearner can use them
                # This is code to handle spinning off a worker for memory management
                socket.setdefaulttimeout(None)
                manager = Manager()
                return_dict = manager.dict()
                return_dict['result'] = []
                return_dict['error'] = None
                return_dict['num_params'] = 0
                my_queue = StdOutQueue()
                my_data_pair.modList = individual.modList
                my_process = Process(target=handleWorker, args=(
                    func, my_data_pair, cp.deepcopy(individual.modList), return_dict, my_queue, ind_hash))
                my_process.daemon = True
                my_process.start()
                # This mem limit needs to be substantially lower when working with multiple worker algorithms on a single machine
                # For now, hard coding this value to 8GB, but should map this to an xml param and stop using percentages TODO
                #mem_limit = _inst.mem_limit/100.0*virtual_memory().total/(1024 * 1024)    #Puts in MB
                mem_limit = _inst.mem_limit # in MB

                #mem_limit = 30*1024    # in MB

                should_kill_mem = False
                should_kill_time = False
                if sys.platform != 'win32':
                    while my_process.is_alive():
                        total_mem = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
                        try: # have been getting a lot of misc errors with this line (returns empty string, argument list too long)
                            total_mem = int(os.popen('ps -p %d -o %s | tail -1' % (
                                my_process.pid, 'rss')).read()) / 1024.0  # MB
                            time.sleep(1)
                        except:
                            time.sleep(15)
                            total_mem = 0 # set to 0 to avoid any errors
                        # print('MEMORY:',total_mem/1024.0)
                        sys.stdout.flush()
                        if total_mem > mem_limit:
                            error_string += 'from resource, my memory usage is: ' + str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) + \
                                   ' and my children: '+ str( resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss ) + '\n'


                            error_string += 'Memory usage' + str( total_mem / (1024.0)) + \
                                   'GB exceeds' + str(mem_limit / (1024.0)) +' GB\n'

                            should_kill_mem = True
                            break
                        # Let's also kill if it's been running for too long to avoid getting stuck
                        elif time.time() - start_time > _inst.time_limit:
                            should_kill_time = True
                            break
                else:
                    while my_process.is_alive():
                        total_mem = psutil.Process(my_process.pid).memory_info().rss / 1024.0
                        mem_limit = mem_limit * 1024

                        time.sleep(1)
                        sys.stdout.flush()
                        if total_mem > mem_limit:
                            error_string += 'from psutil, my memory usage is: ' + str(total_mem)

                            error_string += ' Memory usage ' + str(total_mem / (1024.0)) + \
                                            ' MB exceeds ' + str(mem_limit / (1024.0)) + ' MB\n'

                            should_kill_mem = True
                            break
                        elif time.time() - start_time > _inst.time_limit:
                            should_kill_time = True
                            break
                if should_kill_mem:
                    print("Individual hit MEMORY limit. Hash: " + ind_hash, "PID: {}".format(my_process.pid), "TimeStamp: " + str(datetime.now()))
                    sys.stdout.flush()
                    if sys.platform == 'win32':    
                        my_process.terminate()
                    else:
                        #my_process.terminate()
                        my_process.kill()
                    my_process.join()
                    raise Exception('Process terminated due to MEMORY: ' + get_stack_from_queue(my_queue))
                elif should_kill_time:
                    print("Individual hit TIME limit. Hash: " + ind_hash, "PID: {}".format(my_process.pid), "TimeStamp: " + str(datetime.now()))
                    sys.stdout.flush()
                    if sys.platform == 'win32':
                        my_process.terminate()
                    else:
                        #my_process.terminate()
                        my_process.kill()
                    my_process.join()
                    raise Exception('Process terminated due to TIME: ' + get_stack_from_queue(my_queue))

                my_process.join()

                if return_dict['error'] is not None:
                    # Remove in-progress cache data if it exists
                    cleanup_trash(ind_hash)
                    # Handle lock timeout if it occurred
                    if "lock timeout" in return_dict['error']:
                        # Reassign stdout and stderr
                        # sys.stdout = sys_stdout
                        # sys.stderr = sys_stderr
                        # set retry_time and elapsed time
                        individual.retry_time += time.time() - start_time
                        individual.elapsed_time += time.time() - start_time
                        # bounce the individual back / update it
                        return (individual, individual.elapsed_time, error_string, True)

                    # Error will be a string message, so raise it as an exception
                    raise Exception(return_dict['error'] + ' ' + get_stack_from_queue(my_queue))
                else:
                    result = return_dict['result']

                test_data_classes = result
                individual.num_params = return_dict['num_params']
                print("Individual Fold Finished Evaluating. Hash: " + ind_hash)
                print("TimeStamp | " + str(datetime.now()))
                sys.stdout.flush()

                for objective in _inst.objectivesDict:
                    try:
                        name = _inst.objectivesDict[objective]['name']
                        if 'args' in _inst.objectivesDict[objective]:
                            kwargs = _inst.objectivesDict[objective]['args']
                        else:
                            kwargs = {}
                        kwargs['prev_generation_layer_frequencies'] = prev_generation_layer_frequencies
                        args = [individual, test_data_classes, my_truth_data, name]
                        objectives[objective] = _inst.objectivesDict[objective]['evaluationFunction'](*args, **kwargs)
                        print("Successfully Evaluated Objective: " + _inst.objectivesDict[objective]['name'] + " | " + str(datetime.now()))
                        sys.stdout.flush()
                    except Exception as e:
                        error_string += 'Objective {} failed to be evaluated, returning the exception {}\n'.format(name,str(e)+traceback.format_exc())
                        monte_carlo_objectives = fatal_fitness
                        break
                    if np.isnan(objectives[objective]):
                        error_string += " One or more of the objectives evaluated to NaN."
                        objectives[objective] = np.inf
                # Perfection implies error
                perfection = True
                # Need to check for length, which will never be zero and has no bearing on
                # "perfection" with regards to the data
                for objective_score, objective in zip(objectives, _inst.objectivesDict):
                    if _inst.objectivesDict[objective]['name'] == 'Length':
                        continue
                    if objective_score > 0:
                        perfection = False
                        break
                if perfection:
                    # fatal fitness kills scores anyway
                    error_string += "It was detected that the individual performed perfectly."
                    monte_carlo_objectives = fatal_fitness
                    break
                # accumulate objectives from each of the trials.
                monte_carlo_objectives += objectives
        except Exception as e:
            error_string += '**********' + str(e) + '\n'
            error_string += '----------' + str(traceback.format_exc()) + '\n'
            monte_carlo_objectives = fatal_fitness
            # fatal fitness kills scores anyway
    # Average monte carlo results across number of trials
    monte_carlo_objectives = monte_carlo_objectives/len(_inst.datasetDict[dataset]['truthDataArray'])

    # Check for any infs, make fatal if any
    if np.inf in monte_carlo_objectives:
        #error_string += f"MC-OBJ: {monte_carlo_objectives}\n"
        #error_string += f"num trials expect(1): {len(_inst.datasetDict[dataset]['truthDataArray'])}\n"
        error_string += " At least one objective returned inf. " 
        monte_carlo_objectives = fatal_fitness
    else:
        individual.age = np.floor(individual.age + 1)
    individual.fitness.values = tuple(monte_carlo_objectives)
    setattr(individual, _inst.datasetDict[dataset]['name'],
        cp.deepcopy(individual.fitness))

    # Reassign stdout and stderr
    #sys.stdout = sys_stdout
    #sys.stderr = sys_stderr
    individual.elapsed_time += time.time() - start_time
    return (individual, individual.elapsed_time, error_string, False) # TODO: grab model weights from return dict and return here



def adjustForDataset(individual):
    """This function takes in an individual and adjusts their objective
    based upon the highest tier data set evaluated for it thus far

    Args:
        individual: individual in question

    Returns:
        individual
    """
    offset = GPFramework.constants.TIERED_MULTIPLIER*(len(_inst.datasetDict.keys()) - 1)
    for dataset in _inst.datasetDict:
        if getattr(individual, _inst.datasetDict[dataset]['name']).valid:
            individual.fitness.values = list(map(sum, zip(
                getattr(individual, _inst.datasetDict[dataset]['name']).values),
                (offset, ) * len(getattr(individual,
                    _inst.datasetDict[dataset]['name']).values)))
        offset -= 1*GPFramework.constants.TIERED_MULTIPLIER

    return individual


def genePoolPartition(gene_pool, N=100, dataset='smallFitness'):
    """Partitions specific individuals from gene pool

    Args:
        gene_pool: pool to evaluate
        N: number to partition
        dataset: dataset to use

    Returns:
        well-rounded individuals and specialist individuals
    """
    numSpecialists = int(np.floor(0.25 * N))
    numWellRounded = int(N - numSpecialists)
    wellRounded = []
    specialists = []
    upper_thresholds = _inst.SROI
    lower_thresholds = _inst.LROI
    individuals_to_choose_from = 0
    for ind in gene_pool:
        myFitness = getattr(ind, dataset).values
        #if (1 not in myFitness) and (np.inf not in myFitness):
        if np.inf not in myFitness:
            individuals_to_choose_from += 1
            count = 0
            for i in range(len(upper_thresholds)):
                if myFitness[i] > lower_thresholds[i] and myFitness[i] <= upper_thresholds[i]:
                    count += 1
            if count > 1:
                wellRounded.append(ind)
            elif count == 1:
                specialists.append(ind)
    print('wellRounded, specialists', len(wellRounded), len(specialists), 'out of', numWellRounded, numSpecialists)
    print('out of', len(gene_pool), ',', individuals_to_choose_from, 'were valid')
    sys.stdout.flush()
    if len(wellRounded) > 0:
        wellRounded = _inst.toolbox.spea2(wellRounded, len(wellRounded))
    if len(specialists) > 0:
        specialists = _inst.toolbox.spea2(specialists, len(specialists))
    wellRounded = wellRounded[:numWellRounded]
    specialists = specialists[:numSpecialists]
    return (wellRounded, specialists)

class NoDaemonProcess(mp.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocess.pool.Pool):
    @staticmethod
    def Process(ctx, *args, **kwds):
        return NoDaemonProcess(*args, **kwds)

def get_last_pickled(file_object):
    # goes through (opened) pickled file and returns the last thing written to it
    while True:
        try:
            result = pickle.load(file_object) # afaik the only way is to load one object at a time sequentially
        except EOFError:
            break
    return result
'''
Sets up parameters for GP run
'''
_inst = EMADE()
_inst.debug = True
_inst.mem_limit = 30 # 1GB default
_inst.time_limit = 9000
create_representation = _inst.create_representation
buildClassifier = _inst.buildClassifier
setObjectives = _inst.setObjectives
setDatasets = _inst.setDatasets
setMemoryLimit = _inst.setMemoryLimit
setDicts = _inst.setDicts