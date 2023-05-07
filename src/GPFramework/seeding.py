from deap import gp
import GPFramework.gp_framework_helper as gpFrameworkHelper
from GPFramework.gp_framework_helper import LearnerType
from GPFramework.data import EmadeDataPair, EmadeDataPairNN, EmadeData, load_many_to_one_from_file, load_feature_data_from_file
import GPFramework.eval_methods as eval_methods
from GPFramework.emade_operators import cx_ephemerals
import hashlib
import inspect
import numpy as np
import copy as cp
import matplotlib.pyplot as plt
import pickle
from deap import creator
import time
import random
import argparse as ap
from GPFramework import sql_connection_orm
import xml.etree.ElementTree as ET
from lxml import etree
import os
import re
from deap import base
import itertools
from GPFramework.EMADE import my_str, my_hash

"""
INSTRUCTIONS FOR USE:

1) Update my_learner to utilize get_machine_learner to get your desired learner for your individual
2) Update my_func to be your desired primitive
3) Add ephemeral constants to make sure your desired hyperparameters contain the correct values
4) Make sure you set reuse to 1 or 0 depending on your intention
5) run python src/GPFramework/seeding.py -d [database_info] templates/[.xml file]

"""



# Takes an XML file to get dataset, objective, and evaluation information
parser = ap.ArgumentParser()
parser.add_argument(
    'filename', help='Input to EMADE, see inputSample.xml')
parser.add_argument('-d', '--database', dest='database', default='sqlite', type=str, help='SQLAlchemy connection string in the form of dialect[+driver]://user:password@host/dbname. See http://docs.sqlalchemy.org/en/latest/core/engines.html#sqlalchemy.create_engine')

args = parser.parse_args()
inputFile = args.filename
database_str = args.database


# Valid XML file with inputSchema.xsd using lxml.etree
schema_doc = etree.parse(os.path.join('templates', 'inputSchema.xsd'))
schema = etree.XMLSchema(schema_doc)

doc = etree.parse(inputFile)
# Raise error if invalid XML
try:
    schema.assertValid(doc)
except:
    raise

# Uses xml.etree.ElementTree to parse the XML
tree = ET.parse(inputFile)
root = tree.getroot()




# Initializes the dataset dictionary (Only need this to get the sname of the dataset)
datasetDict = {}
datasetList = root.iter('dataset')
for datasetNum, dataset in enumerate(datasetList):
# Iterate over each dataset and add to dictionary
    monte_carlo = dataset.iter('trial')
    datasetDict[datasetNum] = {'name': dataset.findtext('name'),
                           'type': dataset.findtext('type'),
                           'batchSize': None if dataset.findtext('batchSize') is None else int(dataset.findtext('batchSize')),
                           'trainFilenames':[],
                           'testFilenames':[]}
# The data is already folded for K-fold cross validation. Add these to the trainFilenames
# and testFilenames lists
for trial in monte_carlo:
    datasetDict[datasetNum]['trainFilenames'].append(
        trial.findtext('trainFilename'))
    datasetDict[datasetNum]['testFilenames'].append(
        trial.findtext('testFilename'))

# Initializes the objective dictionary
objectiveDict = {}
objectiveList = root.iter('objective')
evaluationInfo = root.find('evaluation')
evaluationModule = __import__(evaluationInfo.findtext('module'))
for objectiveNum, objective in enumerate(objectiveList):
    # Iterate over each objective and add to dictionary
    objectiveDict[objectiveNum] = {'name': objective.findtext('name'), 'weight': float(objective.findtext('weight')),
    'achievable': float(objective.findtext('achievable')), 'goal': float(objective.findtext('goal')),
    'evaluationFunction': getattr(evaluationModule, objective.findtext('evaluationFunction')),
    'lower': float(objective.findtext('lower')), 'upper': float(objective.findtext('upper'))}

fitness_names = [objectiveDict[objective]['name'] for objective in objectiveDict]
dataset_names = [datasetDict[dataset]['name'] for dataset in datasetDict]


# Set random seeds
random.seed(101)
np.random.seed(101)

# Initialize pset
pset = gp.PrimitiveSetTyped("MAIN", [EmadeDataPairNN], EmadeDataPairNN)
gpFrameworkHelper.addTerminals(pset)
gpFrameworkHelper.addPrimitives(pset)

# Determine whether the problem is a regression (True) or classification (False) problem
gpFrameworkHelper.set_regression(True)

terminals = {}
primitives = {}
ephemerals = {}
for item in pset.mapping:
    if isinstance(pset.mapping[item], gp.Terminal):
        terminals[item] = pset.mapping[item]
    elif isinstance(pset.mapping[item], gp.Primitive):
        primitives[item] = pset.mapping[item]
        # else:
        # ephemerals[item] = pset.mapping[item]
names = []
methods = dir(gp)
for method in methods:
    pointer = getattr(gp, method)
    if inspect.isclass(pointer) and issubclass(pointer, gp.Ephemeral):
        ephemerals[method] = pointer


# Compute arrays of weights and thresholds from objective information
weights = tuple([objectiveDict[objective]['weight'] for objective in objectiveDict])
goals = tuple([objectiveDict[objective]['goal'] for objective in objectiveDict])
achievable = tuple([objectiveDict[objective]['achievable'] for objective in objectiveDict])
LROI = np.array(goals)
creator.create("FitnessMin", base.Fitness, weights=weights)
fitness_names = (datasetDict[dataset]['name'] for dataset in datasetDict)
fitness_attr = dict(zip(fitness_names, itertools.repeat(creator.FitnessMin)))


creator.create("Individual", list, pset=pset, fitness=creator.FitnessMin, age=0, hash_val=None, **fitness_attr)


# Calls learnerGen in gp_framework_helper.py to generate a LearnerType object
def get_machine_learner(learnerName, learnerParams):
    learner = ephemerals['learnerGen']()
    learner.value.learnerName = learnerName
    learner.value.learnerParams = learnerParams
    return learner

pset.context['learnerType'] = LearnerType


"""
Use the get_machine_learner function defined above to generate a LearnerType object.
Pass in the name of the machine learner (you can find a list of all the available learners in the
learnerGen function in gp_framework_helper.py). You also must pass in a learnerParams dictionary
with all the attributes you want your algorithm to have (see learnerGen for that as well).
"""
#Initialize the primitives and terminals that you will use in your function

my_learner = get_machine_learner('DepthEstimate',  {'sampling_rate':1.0})
learner_1 = get_machine_learner('gradient_boosting_regression',  {'learning_rate':0.1, 'n_estimators':100, 'max_depth':3})
learner_2 = get_machine_learner('svm_regression',  {'kernel':0})


#my_learner = get_machine_learner('decision_tree_regression',  {'n_estimators': 100, 'class_weight':0, 'criterion':0})
print(primitives['SingleLearner'], terminals['ARG0'], my_learner)

# Define a PrimitiveTree using the
stream_to_features = ephemerals['myRandTri']()
stream_to_features.value = 2
stream_to_stream = ephemerals['myRandTri']()
stream_to_stream.value = 1

features_to_features = ephemerals['myRandTri']()
features_to_features.value = 0

filter_length = ephemerals['myRandInt']()
filter_length.value = 11
filter_order = ephemerals['myRandInt']()
filter_order.value = 3
filter_deriv = ephemerals['myRandInt']()
filter_deriv.value = 0
delta = ephemerals['myGenFloat']()
delta.value = 10.0
sampling_rate = ephemerals['myGenFloat']()
sampling_rate.value = 2.0*3.2e9 / (3e8/1.33)
neg_one = ephemerals['myGenFloat']()
neg_one.value = -1.0
off_set = ephemerals['myGenFloat']()
off_set.value = 65000.0 # Pulled empirically from plot
search_window_optimal = ephemerals['myRandInt']()
search_window_optimal.value = 15
search_window_standard = ephemerals['myRandInt']()
search_window_standard.value = 11

random_forest_learner = ephemerals['learnerGen']()
random_forest_learner.value.learnerName = 'RandForest'
random_forest_learner.value.learnerParams = None

emade_tree_learner = ephemerals['learnerGen']()
emade_tree_learner.value.learnerName = 'Trees'

p_value_LPC = ephemerals['myRandInt']()
#p_value_LPC.value = -939
p_value_LPC.value = 9

samples_to_cut = ephemerals['myRandInt']()
samples_to_cut.value = 2



#build your function with the primitives here
# my_func = gp.PrimitiveTree([primitives['SingleLearner'],
#                         primitives['myInformedSearch'],
#                         primitives['myRebase'],
#                         primitives['myDeapDataMult'],
#                         primitives['myCutDataLead'], terminals['ARG0'], samples_to_cut, stream_to_stream,
#                         neg_one,
#                         stream_to_stream,
#                         primitives['myPeakFinder'],
#                         #primitives['savGol'],
#                         primitives['myRebase'],
#                         primitives['myDeapDataMult'],
#                         primitives['myCutDataLead'], terminals['ARG0'], samples_to_cut, stream_to_stream,
#                         neg_one,
#                         stream_to_stream,
#                         #filter_length, filter_order, filter_deriv, stream_to_stream,
#                         delta, terminals['trueBool'], stream_to_stream,
#                         search_window_standard, stream_to_features,
#                         primitives['ModifyLearnerFloat'], my_learner, sampling_rate])

my_func = gp.PrimitiveTree([primitives['BaggedLearner'],
                        primitives['SingleLearner'],
                        primitives['myFFT'],
                        primitives['myFFT'],
                        primitives['spLT'],
                        primitives['myCutDataLead'], terminals['ARG0'], samples_to_cut, features_to_features,
                        primitives['myMatchedFiltering'], terminals['ARG0'], features_to_features,
                        primitives['passTriState'],
                        primitives['passTriState'], stream_to_stream,
                        stream_to_features,
                        stream_to_stream, learner_1, learner_2])


my_tree = gp.PrimitiveTree(my_func)
my_individual = creator.Individual([my_tree,my_tree,my_tree,my_tree])

#print the seeded individual to ther terminal
print(my_str(my_individual))

fitness_names = [objectiveDict[objective]['name'] for objective in objectiveDict]
dataset_names = [datasetDict[dataset]['name'] for dataset in datasetDict]


#Set Attributes
my_hash_id = my_hash(my_individual)
my_individual.hash_val = my_hash_id
my_individual.fitness_attr = fitness_attr

print(my_hash_id)


# Initialize database for storing information about each individual
database = sql_connection_orm.SQLConnection(connection_str=database_str, reuse=1, fitness_names=fitness_names,
                                            dataset_names=dataset_names, statistics_dict={})

print("connected to database")


database.insertInd(hash=my_hash_id, individual=cp.deepcopy(my_individual),
                                    age=0, evaluation_gen=0,
                                    evaluation_status=sql_connection_orm.IndividualStatus.NOT_EVALUATED,
                                    tree=my_str(my_individual))
