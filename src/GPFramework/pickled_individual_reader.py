import pickle
import argparse as ap
from GPFramework import EMADE as emade
# Replace this by exporting in to pickle file and using a dynamic import
from GPFramework import eval_methods
from GPFramework.gp_framework_helper import LearnerType

import numpy as np

def main():
    print(evalFunctions)
    evaluationModule = __import__('GPFramework.evalFunctions')
    parser = ap.ArgumentParser('Read in pickled GP individuals to use for analysis.')
    parser.add_argument('evaluate_file', type=str)
    parser.add_argument('mode', type=int)
    args = parser.parse_args()

    evaluateLines = []
    with open(args.evaluate_file, 'r') as input:
        for line in input:
            evaluateLines.append(line.rstrip('\n'))

    with open(evaluateLines[0], 'rb') as startup_settings_file:
        evolutionParametersDict = pickle.load(startup_settings_file)
        print(evolutionParametersDict)
        objectivesDict = pickle.load(startup_settings_file)
        datasetDict = pickle.load(startup_settings_file)

        emade.create_representation(mods=3)

        eval = evaluateLines[2].split(',')
        datasetDict[0]['trainFilenames'] = [x for x in eval]
        datasetDict[0]['testFilenames'] = [evaluateLines[3]]

        emade.setObjectives(objectivesDict)
        emade.setDatasets(datasetDict)
        emade.setMemoryLimit(evolutionParametersDict['memoryLimit'])
        emade.buildClassifier()

    with open(evaluateLines[1], 'rb') as pickle_file:
        pareto_front = pickle.load(pickle_file)

    if args.mode == 0:
        predictValuesFitness(pareto_front)
    elif args.mode == 1:
        predictValues(pareto_front)
    elif args.mode == 2:
        evaluateFiteness(pareto_front)
    elif args.mode == 3:
        readFitnessToIndividual(pareto_front)


def predictValuesFitness(pareto_front):
    individuals = [individual for individual in pareto_front]
    best_over = np.inf

    predictions = []
    output = []
    for i, individual in zip(range(len(individuals)), individuals):
        print(i, individual.fitness.values)
        print(emade.my_hash(individual))
        print(emade.my_str(individual))

        emade._inst.pset.context['learnerType'] = LearnerType
        func = emade._inst.toolbox.compile(expr=individual)
        data_pair = emade._inst.datasetDict[0]['dataPairArray'][0]
        truth_data = emade._inst.datasetDict[0]['truthDataArray'][0]
        # print(data_pair.get_train_data().get_instances())
        try:
            test = func(data_pair)
            # test_data_classes = np.array([inst.get_target()[0] for inst in test.get_test_data().get_instances()])
            test_data_classes = test.get_test_data().get_target()
            predictions.append(test_data_classes)
            print('TEST DATA:', [x for x in test_data_classes if not np.isnan(x)])
            print('TRUTH DATA:', truth_data)
        except Exception as e:
            print('Errored on individual', i, str(e))

        emade.evaluate_individual(individual, 0)
        print('Validated To', individual.fitness.values, '\n')
        output.append(individual.fitness.values)

    predictions = np.array(predictions)
    predictions.transpose()
    np.savetxt('evaluated.csv', predictions, delimiter=',')
    output = np.array(output)
    np.savetxt('evaluated.csv', output, delimiter=',')


def predictValues(pareto_front):
    individuals = [individual for individual in pareto_front]
    best_over = np.inf

    predictions = []
    for i, individual in zip(range(len(individuals)), individuals):
        print(i, individual.fitness.values)
        print(emade.my_hash(individual))
        print(emade.my_str(individual))

        emade._inst.pset.context['learnerType'] = LearnerType
        func = emade._inst.toolbox.compile(expr=individual)
        data_pair = emade._inst.datasetDict[0]['dataPairArray'][0]
        truth_data = emade._inst.datasetDict[0]['truthDataArray'][0]
        # print(data_pair.get_train_data().get_instances())
        try:
            test = func(data_pair)
            #test_data_classes = np.array([inst.get_target()[0] for inst in test.get_test_data().get_instances()])
            test_data_classes = test.get_test_data().get_target()
            predictions.append(test_data_classes)
            print('TEST DATA:', [x for x in test_data_classes if not np.isnan(x)])
            print('TRUTH DATA:', truth_data)
        except Exception as e:
            print('Errored on individual', i, str(e))
    predictions = np.array(predictions)
    predictions.transpose()
    np.savetxt('evaluated.csv', predictions, delimiter=',')


def evaluateFiteness(pareto_front):
    individuals = [individual for individual in pareto_front]
    best_over = np.inf

    output = []
    print(len(individuals))
    for i, individual in zip(range(len(individuals)), individuals):
        print(i, individual.fitness.values)
        print(emade.my_hash(individual))
        print(emade.my_str(individual))
        emade.evaluate_individual(individual, 0)
        print('Validated To', individual.fitness.values, '\n')
        output.append(individual.fitness.values)
    output = np.array(output)
    np.savetxt('evaluate_lidar.csv', output, delimiter=',')


def readFitnessToIndividual(pareto_front):
    individuals = [individual for individual in pareto_front]

    print(len(individuals))
    with open('read_individuals', 'w') as writer:
        for i, individual in zip(range(len(individuals)), individuals):
            print(i, individual.fitness.values)
            temp_hash = emade.my_hash(individual)
            temp_str = emade.my_str(individual)
            print(temp_hash)
            print(temp_str)
            writer.write(temp_str + "\n")
            writer.write(temp_hash + "\n")
            writer.write(str(individual.fitness.values) + "\n")
            writer.write("\n")

if __name__ == '__main__':
    main()
