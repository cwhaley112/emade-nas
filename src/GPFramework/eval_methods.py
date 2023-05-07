"""
Programmed by Jason Zutty
Modified by VIP Team
Implements a number of evaluation functions for scoring the population
"""
from collections import Counter
import numpy as np
from numpy.core.fromnumeric import product
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, precision_score, recall_score
from sklearn.metrics import accuracy_score as accuracy
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import re
import sys
import copy

from neural_network_methods import extract_layer_frequencies, get_adf_layer_frequencies, get_individual_layer_frequencies


def distance_from_target(individual, test_data, truth_data, name=None):
    """
    distance between target and truth is the error
    return sum of the total error (distance)

    This method finds the closest true positive (tp) to each target in order.
    Then sums the total combined error of each tp.

    This means if one tp is close to every target, then the algorithm cannot
    rely on that close tp to lower the error on every target.

    However, our objective is consistent because when all the true positives are
    on their respective targets the objective will be minimized to 0.

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    if len(test_data) == 0:
        return np.inf
    elif len([i for i in test_data if len(i) == 0 or (i.shape[0] == 2 and len(i.shape) == 1) or i.shape[1] == 2]) != len(test_data):
        return np.inf

    r = 4.0
    total_error = 0
    for ex, truth in zip(test_data, truth_data):
        if len(ex.shape) == 1 and ex.shape[0] != 0:
            ex = np.expand_dims(ex, axis=0)
        if len(truth.shape) == 1 and truth.shape[0] != 0:
            truth = np.expand_dims(truth, axis=0)
        visited = set()
        for t in truth:
            if np.sum(t) == -2.0:
                total_error += 0
            else:
                # list of distance from target per object within range
                tp = [np.linalg.norm(obj-t).sum() for obj in ex \
                      if np.linalg.norm(obj-t).sum() <= r and \
                      tuple(obj) not in visited]
                if len(tp) > 0:
                    index = np.argmin(tp)
                    visited.add(tuple(ex[index]))
                    total_error += tp[index]

    if total_error == 0:
        return np.inf
    return total_error

def mean_dist_from_target(individual, test_data, truth_data, name=None):
    """
    mean distance between target and truth is the error
    return sum of the total error (distance) / number of true positives

    This method finds the closest true positive (tp) to each target in order.
    Then sums the total combined error of each tp.

    This means if one tp is close to every target, then the algorithm cannot
    rely on that close tp to lower the error on every target.

    However, our objective is consistent because when all the true positives are
    on their respective targets the objective will be minimized to 0.

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    print("Evaluating Objective Mean Dist From Target")

    if len(test_data) == 0:
        print("Case 1 Failed. Individual Hash:", individual.hash_val)
        print("Test Data:", test_data)
        return np.inf
    elif len([i for i in test_data if len(i) == 0 or (i.shape[0] == 2 and len(i.shape) == 1) or i.shape[1] == 2]) != len(test_data):
        print("Case 2 Failed. Individual Hash:", individual.hash_val)
        print("Test Data:", test_data)
        return np.inf

    truth_data = [np.squeeze(arr, axis=0) if len(arr.shape) == 3 else arr for arr in truth_data]

    r = 4.0
    max_penalty = r + 1
    total_errors = []
    for ex, truth in zip(test_data, truth_data):
        if len(ex.shape) == 1 and ex.shape[0] != 0:
            ex = np.expand_dims(ex, axis=0)
        if len(truth.shape) == 1 and truth.shape[0] != 0:
            truth = np.expand_dims(truth, axis=0)

        total_error = 0
        count = 0
        # check if there is no truth
        if np.sum(truth[0]) == -2.0:
            total_errors.append(np.nan)
        else:
            for t in truth:
                tp = [np.linalg.norm(obj-t).sum() for obj in ex \
                      if np.linalg.norm(obj-t).sum() <= r]
                if len(tp) > 0:
                    count += 1
                    index = np.argmin(tp)
                    total_error += tp[index]
     
            if total_error == 0:
                total_errors.append(np.nan)
            else:
                total_errors.append(total_error / count)

    mean_total_error = np.nanmean(total_errors)
    if np.isnan(mean_total_error):
        mean_total_error = max_penalty

    return mean_total_error

def false_positive_centroid(individual, test_data, truth_data, name=None):
    """
    False positive is defined as a detected object not within
    a defined distance from the target

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    print("Evaluating Objective False Positive Centroid")

    if len(test_data) == 0:
        print("Case 1 Failed. Individual Hash:", individual.hash_val)
        print("Test Data:", test_data)
        return np.inf
    elif len([i for i in test_data if len(i) == 0 or (i.shape[0] == 2 and len(i.shape) == 1) or i.shape[1] == 2]) != len(test_data):
        print("Case 2 Failed. Individual Hash:", individual.hash_val)
        print("Test Data:", test_data)
        return np.inf

    truth_data = [np.squeeze(arr, axis=0) if len(arr.shape) == 3 else arr for arr in truth_data]

    r = 4.0
    fp = 0
    # for each true positive lower the false positive count by 1
    # default false positives for each example is # of objects detected
    for ex, truth in zip(test_data, truth_data):
        if len(ex.shape) == 1 and ex.shape[0] != 0:
            ex = np.expand_dims(ex, axis=0)
        if len(truth.shape) == 1 and truth.shape[0] != 0:
            truth = np.expand_dims(truth, axis=0)

        if np.sum(truth) == -2.0:
            fp += len(ex)
        else:
            for obj in ex:
                if len([t for t in truth if np.linalg.norm(obj-t).sum() <= r]) == 0:
                    fp += 1
    return fp

def false_negative_centroid(individual, test_data, truth_data, name=None):
    """
    False negative is defined as as the number of targets not detected by
    the given objects

    Finds the closest prediction to the truth and check if it is within radius r

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    print("Evaluating Objective False Negative Centroid")

    if len(test_data) == 0:
        print("Case 1 Failed. Individual Hash:", individual.hash_val)
        print("Test Data:", test_data)
        return np.inf
    elif len([i for i in test_data if len(i) == 0 or (i.shape[0] == 2 and len(i.shape) == 1) or i.shape[1] == 2]) != len(test_data):
        print("Case 2 Failed. Individual Hash:", individual.hash_val)
        print("Test Data:", test_data)
        return np.inf

    truth_data = [np.squeeze(arr, axis=0) if len(arr.shape) == 3 else arr for arr in truth_data]

    r = 4.0
    fn = 0
    # for each true negative lower the false negative count by 1
    # default false negatives for each example is # of targets
    for ex, truth in zip(test_data, truth_data):
        if len(ex.shape) == 1 and ex.shape[0] != 0:
            ex = np.expand_dims(ex, axis=0)
        if len(truth.shape) == 1 and truth.shape[0] != 0:
            truth = np.expand_dims(truth, axis=0)
        for t in [i for i in truth if np.sum(i) != -2.0]:
            if len(ex) == 0 or np.min(cdist(ex,[t])) > r:
                fn += 1

    return fn

def false_negative_centroid2(individual, test_data, truth_data, name=None):
    """
    False negative is defined as as the number of targets not detected by
    the given objects

    This method prevents predicted objects from validating more than one truth

    Args:
        test_data: test data
        truth_data: labeled data

    Returns:
        objective score
    """
    if len(test_data) == 0:
        return np.inf
    elif len([i for i in test_data if len(i) == 0 or (i.shape[0] == 2 and len(i.shape) == 1) or i.shape[1] == 2]) != len(test_data):
        return np.inf

    r = 4.0
    fn = 0
    # for each true negative lower the false negative count by 1
    # default false negatives for each example is # of targets
    for ex, truth in zip(test_data, truth_data):
        if len(ex.shape) == 1 and ex.shape[0] != 0:
            ex = np.expand_dims(ex, axis=0)
        if len(truth.shape) == 1 and truth.shape[0] != 0:
            truth = np.expand_dims(truth, axis=0)
        # set truth to empty list if there is no truth
        truth = [t for t in truth if np.sum(t) != -2.0]

        while len(truth) > 0 and len(ex) > 0:
            # map predicted objects to empty lists
            my_map = {(obj[0], obj[1]):[] for obj in ex}

            # map truth data to closest predicted objects
            for t in truth:
                ind = np.argmin([np.linalg.norm(t-obj).sum() for obj in ex])
                my_map[(ex[ind][0], ex[ind][1])].append(t)

            truth = []
            ex = []
            for obj in my_map:
                np_obj = np.array([obj[0], obj[1]])
                t_list = my_map[obj]
                if len(t_list) > 0:
                    # find closest truth data
                    ind = np.argmin([np.linalg.norm(t-np_obj).sum() for t in t_list])
                    # check if the truth data is within valid range
                    # if not then everything on the list is a false negative
                    if np.linalg.norm(t_list[ind]-np_obj).sum() > r:
                        fn += len(t_list)
                        ex.append(np_obj)
                    else:
                        # if valid pairing is found remove truth and obj
                        t_list.pop(ind)
                        truth += t_list

        if len(truth) > 0:
            fn += len(truth)

    return fn

def false_positive_multi(individual, test_data, truth_data, name = None):
    test_data = np.array(test_data)
    #truth_data = np.array(truth_data).reshape(-1, 1)
 
    if test_data.shape != truth_data.shape:
        return np.inf
    return np.sum(test_data[truth_data==0] != 0)

def false_negative_multi(individual, test_data, truth_data, name=None):
    test_data = np.array(test_data)
    #truth_data = np.array(truth_data).reshape(-1, 1)
    if truth_data.shape != test_data.shape:
        return np.inf
    return np.sum(test_data[truth_data==1] != 1)

def false_positive(individual, test_data, truth_data, name=None):
    """
    False positive is test_data == 1 when truth_data == 0
    For example adult dataset a 0 represents <= 50,000 while a 1 represents >50,000
    A false positive means predicting greater than 50k when the individual made less

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    truth_data = np.array(truth_data); test_data = np.array(test_data)
    # Put test data in the same form as truth data from buildClassifier
    if truth_data.shape != test_data.shape:
        return np.inf
    return np.sum(test_data[truth_data==0] != 0)

def false_negative(individual, test_data, truth_data, name=None):
    """
    False negative is test_data == 0 when truth_data == 1
    For example adult dataset a 0 represents <= 50,000 while a 1 represents >50,000
    A false negative means predicting less than 50k when the individual made more

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    truth_data = np.array(truth_data); test_data = np.array(test_data)
    # Put test data in the same form as truth data from buildClassifier
    if truth_data.shape != test_data.shape:
        return np.inf
    return np.sum(test_data[truth_data==1] != 1)

def false_positive_rate(individual, test_data, truth_data, name=None):
    """
    False positive is test_data == 1 when truth_data == 0
    For example adult dataset a 0 represents <= 50,000 while a 1 represents >50,000
    A false positive means predicting greater than 50k when the individual made less

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    truth_data = np.array(truth_data); test_data = np.array(test_data)
    if truth_data.shape != test_data.shape:
        return np.inf
    if len(test_data) == 0:
        return 0
    return np.sum(test_data[truth_data==0] != 0) / len(test_data)

def false_negative_rate(individual, test_data, truth_data, name=None):
    """
    False negative is test_data == 0 when truth_data == 1
    For example adult dataset a 0 represents <= 50,000 while a 1 represents >50,000
    A false negative means predicting less than 50k when the individual made more

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    truth_data = np.array(truth_data); test_data = np.array(test_data)
    if truth_data.shape != test_data.shape:
        return np.inf
    if len(test_data) == 0:
        return 0
    return np.sum(test_data[truth_data==1] != 1) / len(test_data)

def roc_auc(individual, test_data, truth_data, name=None):
    """Returns area under receiver operating characteristic

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    if truth_data.shape != test_data.shape:
        return np.inf
    return roc_auc_score(truth_data, test_data)

def precision_auc(individual, test_data, truth_data, name=None):
    """Returns area under precision-recall curve

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    if truth_data.shape != test_data.shape:
        return np.inf
    return average_precision_score(truth_data, test_data)

def f1_score_min(individual, test_data, truth_data, name=None):
    """Returns F1-Score based on precision and recall

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    if truth_data.shape != test_data.shape:
        return np.inf
    return 1 - f1_score(truth_data, test_data)

def confusion_matrix(individual, test_data, truth_data, name=None):
    """
    Generates a confusion matrix 
    """
    truth_data = np.array(truth_data)
    print(len(test_data))
    test_data = np.array(test_data)
    print(truth_data.shape)
    print(test_data.shape)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(truth_data.argmax(axis=1), test_data.argmax(axis=1), labels=np.arange(17))
    import random
    tf = open(str(random.randint(1,10000000000)), 'wb')
    np.save(tf, cm)
    import seaborn as sns
    plot = sns.heatmap(cm, annot=True)
    plot.figure.savefig("output.png")

    with open('truth.npy', 'wb') as f:
        np.save(f, truth_data)
    with open('test.npy', 'wb') as f:
        np.save(f, test_data)

    with open('multitest','a') as multitest:
        multitest.write(str(truth_data.shape))
        multitest.write(str(test_data.shape))

def auroc(individual, test_data, truth_data, name=None):
    """
    uses scikit auroc as eval function
    """
    from sklearn.metrics import roc_auc_score
    truth_data = np.array(truth_data)
    test_data = np.array(test_data)
    if truth_data.shape != test_data.shape:
        return np.inf
    return 1 - roc_auc_score(truth_data, test_data)

def multilabel_accuracy_score(individual, test_data, truth_data, name=None):
        """multilabel version for accuracy score
        eval_methods.accuracy_score calls np.sum() after comparing 2D arrays element wise.
        multilabel requires the entire rows to be equal.
        """
        # truth = copy.deepcopy(truth_data)
        # test = copy.deepcopy(test_data)
        truth_data = np.array(truth_data)
        test_data = np.array(test_data)
       
        # both_shapes = f"tests: {test_data.shape} truth:{truth_data.shape}\n"
        # test_out = f"TEST OLD VS NEW\n{test}\n============\n{test_data}\n"
        # truth_out = f"TRUTH OLD VS NEW\n{truth}\n============\n{truth_data}\n"
        if truth_data.shape != test_data.shape:
            #assert(False), f"SHAPE FAILED\n" + both_shapes + test_out + truth_out
            return np.inf
        if len(truth_data.shape) != 2:
            #assert(False), f"TEST SHAPE FAILED\n" + both_shapes + test_out + truth_out
            return np.inf
        if len(test_data.shape) != 2:
            #assert(False), f"TRAIN SHAPE FAILED\n" + both_shapes + test_out + truth_out 
            return np.inf
        #assert(False), f"IF FAILED, THEN RETURN STATEMENT BAD\n" + both_shapes + test_out + truth_out 
        return 1 - np.sum(np.array(list(map(np.array_equal, test_data, truth_data)))) / len(test_data)

def multilabel_accuracy_score_fail_trivial(individual, test_data, truth_data, name=None):
        """multilabel version for accuracy score
        eval_methods.accuracy_score calls np.sum() after comparing 2D arrays element wise.
        multilabel requires the entire rows to be equal.
        """
        # truth = copy.deepcopy(truth_data)
        # test = copy.deepcopy(test_data)
        truth_data = np.array(truth_data)
        #test_data = np.array(test_data)
       
        # both_shapes = f"tests: {test_data.shape} truth:{truth_data.shape}\n"
        # test_out = f"TEST OLD VS NEW\n{test}\n============\n{test_data}\n"
        # truth_out = f"TRUTH OLD VS NEW\n{truth}\n============\n{truth_data}\n"
        if truth_data.shape != test_data.shape:
            #assert(False), f"SHAPE FAILED\n" + both_shapes + test_out + truth_out
            return np.inf
        if len(truth_data.shape) != 2:
            #assert(False), f"TEST SHAPE FAILED\n" + both_shapes + test_out + truth_out
            return np.inf
        if len(test_data.shape) != 2:
            #assert(False), f"TRAIN SHAPE FAILED\n" + both_shapes + test_out + truth_out 
            return np.inf
        if np.all(test_data == 1) or np.all(test_data == 0):
            # Fail if trivial predictions, i.e. all true or all false
            return np.inf
        #assert(False), f"IF FAILED, THEN RETURN STATEMENT BAD\n" + both_shapes + test_out + truth_out 
        return 1 - np.sum(np.array(list(map(np.array_equal, test_data, truth_data)))) / len(test_data)

def my_str(individual):
    """Return the string representation of an individual

    Args:
        individual: The individual in question

    Returns:
        the string representation of an individual
    """
    # First we get the string of the main tree
    my_string = str(individual[0])

    # Next check to see if that tree uses any ADF's
    adf_matches = re.findall(r"adf_(\d+)", my_string)

    # Get the int representing the index from the back of the array e.g. [main, 2, 1, 0]
    adf_matches = np.unique([int(adf_match) for adf_match in adf_matches])
    while len(adf_matches) > 0:
        for adf_match_num in adf_matches:
            my_string += '\nadf_' + str(adf_match_num) + ': ' + str(individual[-1-adf_match_num])
        adf_matches = re.findall(r"adf_(\d+)", str(individual[-1-adf_match_num]))
        adf_matches = np.unique([int(adf_match) for adf_match in adf_matches if 'adf_' + adf_match + ': ' not in my_string])

    return my_string

def accuracy_score(individual, test_data, truth_data, name=None, **kwargs):
    """return accuracy score with minimization technique

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    # #sklearn accuracy score
    truth_data = np.array(truth_data)
    test_data = np.array(test_data)

    if truth_data.shape != test_data.shape:
        # debugging why test data predictions got corrupted
        print(truth_data, test_data)
        sys.stdout.flush()
        
        import socket
        with open('failedones2.txt','a') as f:
            f.write(my_str(individual)+'truth_data '+' '.join(str(s) for s in truth_data.shape)+'\n')
            f.write(my_str(individual)+'test_data '+' '.join(str(s) for s in test_data.shape)+'\n')
            f.write(str(socket.gethostname())+ '\n\n')
        return np.inf
    return 1 - accuracy(truth_data, test_data)

def precision_min(individual, test_data, truth_data, name=None, **kwargs):
    """return precision score with minimization technique, weighted w/ macro
    NOTE: if using unbalanced multiclass data, change averaging to micro (or just make new eval method)

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    # #sklearn precision score
    truth_data = np.array(truth_data)
    test_data = np.array(test_data)
    if truth_data.shape != test_data.shape:
        # already wrote debugging info in accuracy score -- won't do anything here
        return np.inf
    if len(truth_data.shape)>1:
        return 1 - precision_score(truth_data, test_data, average = 'macro') # micro averaging handles class imbalance better, but just gives accuracy for balanced data
    else:
        return 1 - precision_score(truth_data, test_data) # don't average for binary data -- that will just give us accuracy score

def recall_min(individual, test_data, truth_data, name=None):
    """return recall score with minimization technique, weighted w/ macro
    NOTE: if using unbalanced multiclass data, change averaging to micro (or just make new eval method)

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    # #sklearn recall score
    truth_data = np.array(truth_data)
    test_data = np.array(test_data)
    if truth_data.shape != test_data.shape:
        # already wrote debugging info in accuracy score -- won't do anything here
        return np.inf
    if len(truth_data.shape)>1:
        return 1 - recall_score(truth_data, test_data, average = 'macro') # see comments on precision_min averaging
    else:
        return 1 - recall_score(truth_data, test_data)
    
def num_params(individual, test_data, truth_data, name=None):
    """
    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled ddata
        name: optional name parameter
    Returns:
        the number of parameters in the individual (if NNLearner, None otherwise)

    """
    if individual.num_params == None:
        return np.inf
    else:
        return individual.num_params

def accuracy_score_multi(individual, test_data, truth_data, name=None):
    """return accuracy score for multi-dimensional data, similar to false_positive_multi
    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter
    Returns:
        objective score
    """

    
    if truth_data.shape != test_data.shape:
        return np.inf
    fp = np.sum(test_data[truth_data==0] != 0)
    fn = np.sum(test_data[truth_data==1] != 1)
    size = np.product(truth_data.shape)
    true = size - (fp+fn)
    return 1 - true/size

def objective0EvalFunction(individual, test_data, truth_data, name=None):
    """RMS Error

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    if truth_data.shape != test_data.shape:
        return np.inf
    return np.sqrt(np.mean((test_data-truth_data)**2))

def objective1EvalFunction(individual, test_data, truth_data, name=None):
    """Over Prediction

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    if truth_data.shape != test_data.shape:
        return np.inf
    differences = truth_data - test_data
    overPrediction = differences < 0
    differences = np.array(differences)
    # If there were no under predictions, return 0 error
    if not any(overPrediction):
        return 0.0
    else:
        return np.mean(np.abs(differences[overPrediction]))

def objective2EvalFunction(individual, test_data, truth_data, name=None):
    """Under Prediction

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    if truth_data.shape != test_data.shape:
        return np.inf
    differences = truth_data - test_data
    underPrediction = differences > 0
    differences = np.array(differences)
    # If there were no over predictions, return 0 error
    if not any(underPrediction):
        return 0.0
    else:
        return np.mean(differences[underPrediction])

def objective3EvalFunction(individual, test_data, truth_data, name=None):
    """Scores by height of individual tree

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    return individual[0].height

def objective4EvalFunction(individual, test_data, truth_data, name=None):
    """
    Probability of "Detection"
    Probability of prediciting within 1 decimeter

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    if truth_data.shape != test_data.shape:
        return np.inf
    differences = truth_data - test_data
    under_one_decimeter = np.array(np.abs(differences) <= np.sqrt(0.5**2 + (0.013 * truth_data)**2) - 0.4)

    return 1.0-float(np.sum(under_one_decimeter))/len(under_one_decimeter)

def objective5EvalFunction(indivdual, test_data, truth_data, name=None):
    """Overall mean percent error

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    if truth_data.shape != test_data.shape:
        return np.inf
    differences = abs(truth_data - test_data)
    percents = differences/truth_data
    return abs(np.nanmean(percents))

def objective6EvalFunction(indivdual, test_data, truth_data, name=None):
    """Valid Overall mean percent error

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    if truth_data.shape != test_data.shape:
        return np.inf
    test_data[np.logical_or(np.logical_and(test_data < 0, truth_data != -1), np.logical_and(test_data >= 0, truth_data == -1))] = np.nan
    # test_data[test_data == -1] = np.nan
    differences = abs(truth_data - test_data)
    percents = differences/truth_data
    return abs(np.nanmean(percents))

def objective7EvalFunction(individual, test_data, truth_data, name=None):
    """Valid RMS Error

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    # test_data = np.array(test_data).flatten()
    # truth_data = np.array(truth_data).flatten()
    if truth_data.shape != test_data.shape:
        return np.inf
    test_data = copy.deepcopy(test_data)
    test_data[np.logical_or(np.logical_and(test_data < 0, truth_data != -1), np.logical_and(test_data >= 0, truth_data == -1))] = np.nan
    # testData[testData == -1] = np.nan
    return np.sqrt(np.nanmean((test_data-truth_data)**2))

def objective8EvalFunction(individual, test_data, truth_data, name=None):
    """Valid Over Prediction

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    if truth_data.shape != test_data.shape:
        return np.inf
    differences = truth_data - test_data
    overPrediction = differences < 0
    differences = np.array(differences)
    # If there were no under predictions, return 0 error
    if not any(overPrediction):
        return 0.0
    else:
        differences[np.logical_or(np.logical_and(test_data < 0, truth_data != -1), np.logical_and(test_data >= 0, truth_data == -1))] = np.nan
        # differences[testData == -1] = np.nan
        return np.nanmean(np.abs(differences[overPrediction]))

def objective9EvalFunction(individual, test_data, truth_data, name=None):
    """Valid Under Prediction

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    if truth_data.shape != test_data.shape:
        return np.inf
    differences = truth_data - test_data
    underPrediction = differences > 0
    differences = np.array(differences)
    # If there were no over predictions, return 0 error
    if not any(underPrediction):
        return 0.0
    else:
        differences[np.logical_or(np.logical_and(test_data < 0, truth_data != -1), np.logical_and(test_data >= 0, truth_data == -1))] = np.nan
        # differences[testData < 0] = np.nan
        return np.nanmean(differences[underPrediction])

def objective10EvalFunction(individual, test_data, truth_data, name=None):
    """Number of individuals that could not be evaluated properly

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    if truth_data.shape != test_data.shape:
        return np.inf
    # return np.sum(testData[testData == -1]) * -1
    return np.sum(test_data < 0)

def objective11EvalFunction(individual, test_data, truth_data, name=None):
    """
    False Positive Bottom

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    if truth_data.shape != test_data.shape:
        return np.inf
    # negatives = testData == -1
    # false_array = np.array(testData)[negatives]
    # return np.sum(testData[testData == -1]) * -1
    return np.sum(np.logical_and(test_data >= 0, truth_data == -1))

def objective12EvalFunction(individual, test_data, truth_data, name=None):
    """
    False Negative Bottom

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    # negatives = testData == -1
    # false_array = np.array(testData)[negatives]
    # return np.sum(testData[testData == -1]) * -1
    if truth_data.shape != test_data.shape:
        return np.inf
    return np.sum(np.logical_and(test_data < 0, truth_data != -1))


def class0AccuracyEvalFunction(individual, test_data, truth_data, name=None):
    """
    Error in predicting class 0
    For EEG data, classes are 0 and 4

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    if truth_data.shape != test_data.shape:
        return np.inf
    total = 0
    num_wrong = 0
    for test_point, truth_point in zip(test_data, truth_data):
        if truth_point == 0:
            if test_point < 0 or test_point > 2:
                num_wrong += 1
            total += 1
    if num_wrong == 0:
        # We don't want 'perfect' equate it with 100% error
        return 1.0
    else:
        return float(num_wrong)/float(total)

def class4AccuracyEvalFunction(individual, test_data, truth_data, name=None):
    """
    Error in predicting class 4
    For EEG data, classes are 0 and 4

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    if truth_data.shape != test_data.shape:
        return np.inf
    total = 0
    num_wrong = 0
    for test_point, truth_point in zip(test_data, truth_data):
        if truth_point == 4:
            if test_point <= 2 or test_point > 4:
                num_wrong += 1
            total += 1
    if num_wrong == 0:
        # We don't want 'perfect' equate it with 100% error
        return 1.0
    else:
        return float(num_wrong)/float(total)

def drinking_error_rate(individual, test_data, truth_data, name=None):
    """
    This function assesses the error rate "1 - PD"
    For dog behavioral data in predicting the class number
    associated with drinking

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    if truth_data.shape != test_data.shape:
        return np.inf
    total = 0
    num_wrong = 0
    for test_point, truth_point in zip(test_data, truth_data):
        # Two represents a drinking event, Three is eating, One is Chewing
        if (truth_point == 2):
            if np.isnan(test_point) or test_point <= 1.5 or test_point > 2.5:
                num_wrong += 1
            total += 1
    #if num_wrong == 0:
    #    # Perfection implies overtraining
    #    return 1.0
    #else:
    return float(num_wrong)/float(total)

def drinking_false_alarm_rate(individual, test_data, truth_data, name=None):
    """
    This function assesses the error rate of false alarms
    For dog behavioral data in predicting the class number
    associated with drinking

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    # test_data = np.array(test_data)
    # truth_data = np.array(truth_data)
    if truth_data.shape != test_data.shape:
        return np.inf
    total = 0
    num_wrong = 0
    for test_point, truth_point in zip(test_data, truth_data):
        # Two represents a drinking event
        if truth_point != 2:
            if test_point > 1.5 and test_point <= 2.5:
                num_wrong += 1
            total += 1
    #if num_wrong == 0:
    #    # Perfection implies overtraining
    #    return 1.0
    #else:
    return float(num_wrong)/float(total)

def breadth_eval_function(individual, test_data, truth_data, name=None):
    """
    This function determines a metric for breadth by tracking how many times
    ARG0 appears in the individual.  This allows for competition with fitter solutions.

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    # Compute the string representation of the individual
    string_rep = str(individual)
    # Generate array for appearances of ARG0
    data_appearances = [match.start() for match in re.finditer('ARG0', string_rep)]
    return -1.0*len(data_appearances)


def depth_breadth_eval_function(individual, test_data, truth_data, name=None):
    """
    A shallower tree should always beat a deeper tree.
    Given all else equal, a wider tree should beat a narrower tree.

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    # Compute the string representation of the individual
    string_rep = str(individual[0])
    # Generate array for appearances of ARG0
    data_appearances = [match.start() for match in re.finditer('ARG0', string_rep)]
    # Computed by counting occurrences of EmadeDataPair per primitive
    # Let's make this dynamic in the future by querying all functions
    max_breadth = 3.0
    # Tradeoff formula
    return individual[0].height - len(data_appearances)/(max_breadth**individual[0].height + 1.0)

def num_elements_eval_function(individual, test_data, truth_data, name=None):
    """
    The fewer elements in the tree the better

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    # Let's take the sum of all elements in the tree where the name does not contain adf_
    num_elements = 0
    for tree in individual:
        num_elements += np.sum([1 if 'adf_' not in elem.name else 0 for elem in tree])

    # Let's now put the test cases on the individual to be used for fuzzy selection
    # First get what's there
    test_case_vec = getattr(individual, name)
    # Now stick on what's new
    test_case_vec = np.hstack((test_case_vec, num_elements*np.ones(len(test_data))))
    setattr(individual, name, test_case_vec)

    return num_elements

def num_elements_eval_function_capped(individual, test_data, truth_data, name=None):
    """The fewer elements in the tree the better

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    return max(len(individual), 1707)

def shaking_error_rate(individual, test_data, truth_data, name=None):
    """
    This function assesses the error rate "1 - PD"
    For dog behavioral data in predicting the class number
    associated with shaking

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    if truth_data.shape != test_data.shape:
        return np.inf
    total = 0
    num_wrong = 0
    for test_point, truth_point in zip(test_data, truth_data):
        # Nine represents a shaking event
        if (truth_point == 9):
            if np.isnan(test_point) or test_point <= 8.5 or test_point > 9.5:
                num_wrong += 1
            total += 1
    #if num_wrong == 0:
    #    # Perfection implies overtraining
    #    return 1.0
    #else:
    return float(num_wrong)/float(total)

def shaking_false_alarm_rate(individual, test_data, truth_data, name=None):
    """
    This function assesses the error rate of false alarms
    For dog behavioral data in predicting the class number
    associated with shaking

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    if truth_data.shape != test_data.shape:
        return np.inf
    total = 0
    num_wrong = 0
    for test_point, truth_point in zip(test_data, truth_data):
        # Nine represents a shaking event
        if truth_point != 9:
            if test_point > 8.5 and test_point <= 9.5:
                num_wrong += 1
            total += 1
    #if num_wrong == 0:
    #    # Perfection implies overtraining
    #    return 1.0
    #else:
    return float(num_wrong)/float(total)

def scratching_error_rate(individual, test_data, truth_data, name=None):
    """
    This function assesses the error rate "1 - PD"
    For dog behavioral data in predicting the class number
    associated with scratching

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    if truth_data.shape != test_data.shape:
        return np.inf
    total = 0
    num_wrong = 0
    for test_point, truth_point in zip(test_data, truth_data):
        # Eight represents a scratching event
        #if (truth_point == 8):
        if (truth_point == 1):
            if np.isnan(test_point) or test_point != 1:
                num_wrong += 1
            total += 1
    #if num_wrong == 0:
    #    # Perfection implies overtraining
    #    return 1.0
    #else:
    # First get what's there
    test_case_vec = getattr(individual, name)
    # Now stick on what's new
    # The second == 1 is ignoring things that are not 0 nor 1
    test_case_vec = np.hstack((test_case_vec, np.logical_not(test_data[truth_data==1] == 1)))
    setattr(individual, name, test_case_vec)

    return float(num_wrong)/float(total)


def scratching_false_alarm_rate(individual, test_data, truth_data, name=None):
    """
    This function assesses the error rate of false alarms
    For dog behavioral data in predicting the class number
    associated with scratching

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    if truth_data.shape != test_data.shape:
        return np.inf
    total = 0
    num_wrong = 0
    for test_point, truth_point in zip(test_data, truth_data):
        # Eight represents a scratching event
        #if truth_point != 8:
        if truth_point != 1:
            #if test_point > 7.5 and test_point <= 8.5:
            if test_point == 1:
                num_wrong += 1
            total += 1
    #if num_wrong == 0:
    #    # Perfection implies overtraining
    #    return 1.0
    #else:
    # First get what's there
    test_case_vec = getattr(individual, name)
    # Now stick on what's new
    # The == 1 is ignoring things that are not 0 nor 1
    test_case_vec = np.hstack((test_case_vec, test_data[truth_data == 0] == 1))
    setattr(individual, name, test_case_vec)

    return float(num_wrong)/float(total)


def get_over_predicted_inds(test_data, truth_data, name=None, tolerance=0):
    """Returns over predicted individuals

    Args:
        test_data: test data
        truth_data: labeled data
        name: optional name parameter
        tolerance: tolerance threshold

    Returns:
        objective score
    """
    if truth_data.shape != test_data.shape:
        return np.inf
    return np.nonzero(test_data > truth_data + tolerance)[0]


def get_under_predicted_inds(test_data, truth_data, name=None, tolerance=0):
    """Returns under predicted individuals

    Args:
        test_data: test data
        truth_data: labeled data
        name: optional name parameter
        tolerance: tolerance threshold

    Returns:
        objective score
    """
    if truth_data.shape != test_data.shape:
        return np.inf
    return np.nonzero(test_data < truth_data - tolerance)[0]


def count_over_predictions(individual, test_data, truth_data, name=None, tolerance=0):
    """Counts over predictions

    Args:
        test_data: test data
        truth_data: labeled data
        name: optional name parameter
        tolerance: tolerance threshold

    Returns:
        objective score
    """
    if truth_data.shape != test_data.shape:
        return np.inf
    return len(get_over_predicted_inds(test_data, truth_data, tolerance)) / float(len(test_data))


def count_under_predictions(individual, test_data, truth_data, name=None, tolerance=0):
    """Counts under predictions

    Args:
        test_data: test data
        truth_data: labeled data
        name: optional name parameter
        tolerance: tolerance threshold

    Returns:
        objective score
    """
    if truth_data.shape != test_data.shape:
        return np.inf
    return len(get_under_predicted_inds(test_data, truth_data, tolerance)) / float(len(test_data))


def overall_standard_deviation(individual, test_data, truth_data, name=None):
    """Evaluates overall standard deviation

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    if truth_data.shape != test_data.shape:
        return np.inf
    return np.std(test_data - truth_data)


def standard_deviation_over(individual, test_data, truth_data, name=None, tolerance=0):
    """Evaluates standard deviation of over predicted individuals

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter
        tolerance: tolerance threshold

    Returns:
        objective score
    """
    if truth_data.shape != test_data.shape:
        return np.inf
    over_predicted_inds = get_over_predicted_inds(test_data,
                                                  truth_data,
                                                  tolerance)
    test_subset = test_data[over_predicted_inds]
    truth_subset = truth_data[over_predicted_inds]
    return overall_standard_deviation(individual, test_subset, truth_subset)


def standard_deviation_under(individual, test_data, truth_data, name=None, tolerance=0):
    """Evaluates standard deviation of under predicted individuals

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter
        tolerance: tolerance threshold

    Returns:
        objective score
    """
    if truth_data.shape != test_data.shape:
        return np.inf
    under_predicted_inds = get_under_predicted_inds(test_data,
                                                    truth_data,
                                                    tolerance)
    test_subset = test_data[under_predicted_inds]
    truth_subset = truth_data[under_predicted_inds]
    return overall_standard_deviation(individual, test_subset, truth_subset)


def average_percent_error(individual, test_data, truth_data, name=None):
    """Evaluates average percent error

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    # test_data = np.array(test_data)
    # truth_data = np.array(truth_data)
    if truth_data.shape != test_data.shape:
        return np.inf
    return np.nanmean(np.abs(test_data - truth_data) / truth_data)


def average_precent_error_over(individual, test_data, truth_data, name=None, tolerance=0):
    """Evaluates average percent error for over predicted individuals

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter
        tolerance: tolerance threshold

    Returns:
        objective score
    """
    if truth_data.shape != test_data.shape:
        return np.inf
    over_predicted_inds = get_over_predicted_inds(test_data,
                                                  truth_data,
                                                  tolerance)
    test_subset = test_data[over_predicted_inds]
    truth_subset = truth_data[over_predicted_inds]
    return average_percent_error(individual, test_subset, truth_subset)


def average_precent_error_under(individual, test_data, truth_data, name=None, tolerance=0):
    """Evaluates average percent error for under predicted individuals

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter
        tolerance: tolerance threshold

    Returns:
        objective score
    """
    if truth_data.shape != test_data.shape:
        return np.inf
    under_predicted_inds = get_under_predicted_inds(test_data,
                                                    truth_data,
                                                    tolerance)
    test_subset = test_data[under_predicted_inds]
    truth_subset = truth_data[under_predicted_inds]
    return average_percent_error(individual, test_subset, truth_subset)


def max_over_prediction_error(individual, test_data, truth_data, name=None, tolerance=0):
    """Evaluates max for over predicted individuals

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter
        tolerance: tolerance threshold

    Returns:
        objective score
    """
    if truth_data.shape != test_data.shape:
        return np.inf
    over_predicted_inds = get_over_predicted_inds(test_data,
                                                  truth_data,
                                                  tolerance)
    if len(over_predicted_inds) == 0:
        return np.nan
    test_subset = test_data[over_predicted_inds]
    truth_subset = truth_data[over_predicted_inds]
    return np.max(test_subset - truth_subset)


def max_under_prediction_error(individual, test_data, truth_data, name=None, tolerance=0):
    """Evaluates max for under predicted individuals

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter
        tolerance: tolerance threshold

    Returns:
        objective score
    """
    if truth_data.shape != test_data.shape:
        return np.inf
    under_predicted_inds = get_under_predicted_inds(test_data,
                                                    truth_data,
                                                    tolerance)
    if len(under_predicted_inds) == 0:
        return np.nan
    test_subset = test_data[under_predicted_inds]
    truth_subset = truth_data[under_predicted_inds]
    return np.max(truth_subset - test_subset)


def continuous_mse(individual, test_data, truth_data, name=None):
    """Evaluates continuous mean squared error

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    if truth_data.shape != test_data.shape:
        return np.inf
    error = 0

    for test_item, truth_item in zip(test_data, truth_data):
        error += np.sqrt(np.mean(np.square(truth_item - test_item)))

    return error

def continuous_bias(individual, test_data, truth_data, name=None):
    """Evaluates continuous bias

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    if truth_data.shape != test_data.shape:
        return np.inf
    error = 0
    for test_item, truth_item in zip(test_data, truth_data):
        error += np.mean(np.abs(test_data - truth_data))
    return error

def continuous_var(individual, test_data, truth_data, name=None):
    """Evaluates continuous variance

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    if truth_data.shape != test_data.shape:
        return np.inf
    error = 0
    for test_item, truth_item in zip(test_data, truth_data):
        error += np.var(test_data - truth_data)
    return error

def _partition_distance(p1, p2, n):
    """
    Calculates the partition distance D(p1, p2), defined as the minimum number
    of elements that need to be removed from both partitions to make them
    identical. p1 and p2 should both be a list of lists that form a partition
    of the numbers {0,...,n - 1}. Uses a O(n^3) reduction to a weighted
    assignment problem by Konovalov, Litow and Bajema (2005).
    (https://academic.oup.com/bioinformatics/article/21/10/2463/208566)

    Args:
        p1: list of partitions
        p2: list of partitions
        n: total number of distinct elements

    Returns:
        partition distance between p1 and p2
    """
    m = max(len(p1), len(p2))
    p1_str = np.zeros((m, n), dtype=int)
    for i, x in enumerate(p1):
        for v in x:
            p1_str[i][v] = 1
    p2_str = np.ones((m, n), dtype=int)
    for i, x in enumerate(p2):
        for v in x:
            p2_str[i][v] = 0
    cost = np.array([[np.sum(a * b) for b in p2_str] for a in p1_str])
    row_ind, col_ind = linear_sum_assignment(cost)
    return cost[row_ind, col_ind].sum()

def _fast_partition_distance(p1_list, p2_list, n):
    """
    Calculates the partition distance D(p1, p2), defined as the minimum number
    of elements that need to be removed from both partitions to make them
    identical. p1 and p2 should both be a list of lists that form a partition
    of the numbers {0,...,n - 1}. This method will fail and return 'None' if
    D(p1,p2) >= n / 5. Uses an O(n) algorithm given by Porumbel, Hao and Kuntz (2009).
    (https://www.sciencedirect.com/science/article/pii/S0166218X10003069)


    Args:
        p1: list of partitions
        p2: list of partitions
        n: total number of distinct elements

    Returns:
        partition distance between p1 and p2, or None if the distance is >= n / 5
    """
    p1 = {}
    for i, box in enumerate(p1_list):
        for x in box:
            p1[x] = i
    p2 = {}
    for i, box in enumerate(p2_list):
        for x in box:
            p2[x] = i

    k = max(len(p1_list), len(p2_list))
    similarity = 0
    t = np.empty((k, k), dtype=int)
    m = np.zeros(k, dtype=int)
    sigma = np.zeros(k, dtype=int)
    size_p1 = np.zeros(k, dtype=int)
    size_p2 = np.zeros(k, dtype=int)
    for x in range(n):
        t[p1[x], p2[x]] = 0
    for x in range(n):
        i = p1[x]
        j = p2[x]
        t[i, j] += 1
        size_p1[i] += 1
        size_p2[j] += 1
        if t[i, j] > m[i]:
            m[i] = t[i, j]
            sigma[i] = j
    for i in range(k):
        if m[i] != 0:
            if 3*m[i] <= size_p1[i] +  size_p2[sigma[i]]:
                return None
            similarity = similarity + t[i,sigma[i]]
    return n - similarity

def cluster_partition_distance(individual, test_data, truth_data, name=None):
    """Returns normalized partition distance for labeled cluster data

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    distance_sum = 0
    max_sum = 0
    for test_clusters, truth_clusters in zip(test_data, truth_data):
        # Get last column of target data
        test_clusters = test_clusters[-1].flatten()

        p1_dict = {}
        for i, x in enumerate(test_clusters):
            if x not in p1_dict:
                p1_dict[x] = []
            p1_dict[x].append(i)

        p2_dict = {}
        for i, x in enumerate(truth_clusters):
            if x not in p2_dict:
                p2_dict[x] = []
            p2_dict[x].append(i)

        p1 = list(p1_dict.values())
        p2 = list(p2_dict.values())
        d = _fast_partition_distance(p1, p2, len(test_clusters))
        if d is None:
            d = _partition_distance(p1, p2, len(test_clusters))
        distance_sum += d
        max_sum += len(test_clusters) - 1
    return distance_sum / max_sum

def cluster_error1(individual, test_data, truth_data, name=None):
    """
    Returns normalized type 1 error for labeled cluster data.
    Normalized number of points that should be in the same cluster that are in different clusters.
    Has a minimum value of 0, and maximum value of 1.

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    error_sum = 0
    max_sum = 0
    for test_clusters, truth_clusters in zip(test_data, truth_data):
        # Get last column of target data
        test_clusters = test_clusters[-1].flatten()

        p1_clusters = {}
        for i, x in enumerate(test_clusters):
            if x not in p1_clusters:
                p1_clusters[x] = []
            p1_clusters[x].append(i)

        p2_map = {}
        for i, x in enumerate(truth_clusters):
            p2_map[i] = x

        num_error = 0
        max_error = 0
        for a in list(p1_clusters.values()):
            for i in range(len(a)):
                x = a[i]
                for j in range(len(a)):
                    if i != j:
                        y = a[j]
                        max_error += 1
                        if p2_map[x] != p2_map[y]:
                            num_error += 1
        error_sum += num_error
        max_sum += max_error
    return 0 if max_sum == 0 else error_sum / max_sum

def cluster_error2(individual, test_data, truth_data, name=None):
    """
    Returns normalized type 2 error for labeled cluster data.
    Normalized number of points that should be in different clusters that are in the same cluster.
    Has a minimum value of 0, and maximum value of 1.

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    error_sum = 0
    max_sum = 0
    for test_clusters, truth_clusters in zip(test_data, truth_data):
        # Get last column of test data
        test_clusters = test_clusters[-1].flatten()

        p1_clusters = {}
        for i, x in enumerate(test_clusters):
            if x not in p1_clusters:
                p1_clusters[x] = []
            p1_clusters[x].append(i)

        p2_map = {}
        for i, x in enumerate(truth_clusters):
            p2_map[i] = x

        num_error = 0
        max_error = 0
        p1 = list(p1_clusters.values())
        for i in range(len(p1)):
            a = p1[i]
            for j in range(len(p1)):
                if i != j:
                    b = p1[j]
                    for x in a:
                        for y in b:
                            max_error += 1
                            if p2_map[x] == p2_map[y]:
                                num_error += 1
        error_sum += num_error
        max_sum += max_error
    return 0 if max_sum == 0 else error_sum / max_sum

def layer_novelty(individual, test_data, truth_data, name=None, **kwargs) -> float:
    """Compare this individual's layer frequencies to that of the generation."""
    individual_layer_frequencies = get_individual_layer_frequencies(individual)
    prev_generation_layer_frequencies = kwargs['prev_generation_layer_frequencies']
    return sum(0 if layer in prev_generation_layer_frequencies else 1 for layer in individual_layer_frequencies) / len(individual_layer_frequencies)

# def layer_novelty(individual, test_data, truth_data, name=None) -> float:
#     """Compare this individual's layer frequencies to that of the generation."""
#     individual_layer_frequencies = extract_layer_frequencies(my_str(individual))
#     generation_layer_frequencies = get_adf_layer_frequencies("Global_MODS")
#     return sum(0 if layer in generation_layer_frequencies else 1 for layer in individual_layer_frequencies)

def layer_novelty_prod(individual, test_data, truth_data, name=None) -> float:
    """Compare this individual's layer frequencies to that of the generation."""
    individual_layer_frequencies = extract_layer_frequencies(my_str(individual))
    generation_layer_frequencies = get_adf_layer_frequencies("Global_MODS")
    return product(0 if layer in generation_layer_frequencies else 1 for layer in individual_layer_frequencies)

def layer_novelty_inv(individual, test_data, truth_data, name=None) -> float:
    """Compare this individual's layer frequencies to that of the generation."""
    individual_layer_frequencies = extract_layer_frequencies(my_str(individual))
    generation_layer_frequencies = get_adf_layer_frequencies("Global_MODS")
    return sum(-1 if layer in generation_layer_frequencies else 0 for layer in individual_layer_frequencies)