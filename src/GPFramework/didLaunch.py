import numpy as np
import os
import sys
# Allows you to run Emade locally
sys.path.insert(1, '/home/cameron/Desktop/emade/src')

from GPFramework import EMADE as emade

import multiprocess

import pickle
import argparse as ap

# Get pickled input from launchEMADE as argument
parser = ap.ArgumentParser()
parser.add_argument('filename', help='Pickled input from parsed xml')
parser.add_argument('-n', '--num_workers', dest='n', default=4, type=int, help='Number of workers')
parser.add_argument('-d', '--database', dest='database', default='sqlite', type=str,
                    help='SQLAlchemy connection string in the form of dialect[+driver]://user:password@host/dbname. See http://docs.sqlalchemy.org/en/latest/core/engines.html#sqlalchemy.create_engine')
parser.add_argument('-ma', '--master', dest='master', default=False, action='store_true')
reuse_parser = parser.add_mutually_exclusive_group(required=False)
reuse_parser.add_argument('-r', '--reuse', dest='reuse', action='store_true')
reuse_parser.add_argument('-nr', '--no-reuse', dest='reuse', action='store_false')
reuse_parser.set_defaults(reuse=None)

args = parser.parse_args()
master = args.master
if master:
    if args.database != 'sqlite' and args.reuse is None:
        parser.exit('-d/--database and [-r/--reuse -nr/--no-reuse] must be given together')
inputFile = args.filename
num_workers = args.n
database_str = args.database
reuse = args.reuse
if not master:
    reuse = True

# Load pickled dictionaries
with open(inputFile, 'rb') as pickleFile:
    # Dictionaries must be loaded in the same order they were written
    evolutionParametersDict = pickle.load(pickleFile)
    objectivesDict = pickle.load(pickleFile)
    datasetDict = pickle.load(pickleFile)
    stats_dict = pickle.load(pickleFile)
    misc_dict = pickle.load(pickleFile)
    cacheDict = pickle.load(pickleFile)

# Now that our parameters from the xml have been loaded in, the
# first thing we need to do is tell EMADE how to represent the data.
# We do this by specifying how many mods should be carried through the
# individuals
if 'regression' in evolutionParametersDict:
    regression = int(evolutionParametersDict['regression'])
else:
    regression = False

datatype = datasetDict[0]['type']
mods = 20

emade.setDicts(objectivesDict, datasetDict, cacheDict, stats_dict)
emade.buildClassifier()
emade.setObjectives()
emade.create_representation(mods=mods, regression=regression, datatype = datatype)
emade.setDatasets()
emade.setMemoryLimit(misc_dict['memoryLimit'])
sys.stdout.write('DONE\n')

def main(evolutionParametersDict, objectivesDict, datasetDict, stats_dict, misc_dict, reuse, database_str, num_workers, debug=True):
    """Main Function
    """

    print('Begin GP!\n')
    npop = evolutionParametersDict['initialPopulationSize']
    cxpb = evolutionParametersDict['matingDict']['crossover']
    cxpb_ephemeral = evolutionParametersDict['matingDict']['crossoverEphemeral']
    cxpb_headless_chicken = evolutionParametersDict['matingDict']['headlessChicken']
    cxpb_headless_chicken_ephemeral = evolutionParametersDict['matingDict']['headlessChickenEphemeral']
    MUTPB = evolutionParametersDict['mutationDict']['insert']
    MUTLPB = evolutionParametersDict['mutationDict']['insert modify']
    MUTEPHPB = evolutionParametersDict['mutationDict']['ephemeral']
    MUTNRPB = evolutionParametersDict['mutationDict']['node replace']
    MUTUPB = evolutionParametersDict['mutationDict']['uniform']
    MUTSPB = evolutionParametersDict['mutationDict']['shrink']
    minQueueSize = evolutionParametersDict['minQueueSize']
    launchSize = evolutionParametersDict['launchSize']
    elitePoolSize = evolutionParametersDict['elitePoolSize']
    OUTP = evolutionParametersDict['outlierPenalty']
    seed_file_name = misc_dict['seedFile']

    parents_output = misc_dict['parentsOutput']
    pareto_output = misc_dict['paretoOutput']
    gene_pool_fitness = misc_dict['genePoolFitness']
    pareto_fitness = misc_dict['paretoFitness']

    selection_dict = evolutionParametersDict['selection_dict']

    print(npop, cxpb, cxpb_ephemeral, MUTPB, MUTEPHPB, MUTNRPB, MUTUPB,
          MUTSPB, minQueueSize, launchSize, elitePoolSize)

    if master:
        emade.master_algorithm(NPOP=npop, CXPB=cxpb,
            CXPB_ephemeral=cxpb_ephemeral,
            CXPB_headless_chicken=cxpb_headless_chicken,
            CXPB_headless_chicken_ephemeral=cxpb_headless_chicken_ephemeral,
            MUTPB=MUTPB, MUTLPB=MUTLPB, MUTEPHPB=MUTEPHPB, MUTNRPB=MUTNRPB,
            MUTUPB=MUTUPB, MUTSPB=MUTSPB, OUTP=OUTP, selections=selection_dict,
            minQueueSize=minQueueSize,
            launchSize=launchSize, elitePoolSize=elitePoolSize, # seed_file_name,
            parents_output=parents_output, pareto_output=pareto_output,
            gene_pool_fitness=gene_pool_fitness, pareto_fitness=pareto_fitness,
            database_str=database_str, reuse=reuse, debug=True)
    else:
        print("worker made it")
        sys.stdout.flush()
        pool = emade.MyPool(processes=num_workers)
        emade.worker_algorithm(pool=pool, database_str=database_str,
            reuse=reuse, num_workers=num_workers, debug=True)


if __name__ == '__main__':
    main(evolutionParametersDict, objectivesDict, datasetDict, stats_dict, misc_dict, reuse, database_str, num_workers, debug=True)
