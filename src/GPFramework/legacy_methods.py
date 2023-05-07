"""
Programmed by Jason Zutty
Modified by VIP Team
Implements a number of machine learning methods for use with deap
This is legacy code no longer used in EMADE
"""
import os
import pickle
import dill
import copy
import numpy as np
import math
import sys
import time
import re
import GPFramework.signal_methods as sm
from GPFramework.data import EmadeDataPair
#from GPFramework.general_functions import mod_select, target_value_check

# Classifiers
from sklearn.neighbors import KNeighborsClassifier, BallTree
from sklearn.linear_model import OrthogonalMatchingPursuit, LogisticRegression, SGDClassifier, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier,ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, NuSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
#from sklearn.gaussian_process import GaussianProcess
from sklearn.gaussian_process import GaussianProcessClassifier
import sklearn.gaussian_process.kernels as GPKernels
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, Birch
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier

# Regressors
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

def NN_keras(data_pair):
    """Runs a neural network classifier using keras and tensorflow

    Args:
        data_pair: given dataset

    Returns:
        the result of the DNN classifier
    """
    data_pair = copy.deepcopy(data_pair)
    training_data = data_pair.get_train_data().get_numpy()
    target_values = np.array([inst.get_target()[0] for
                              inst in data_pair.get_train_data().get_instances()])

    model = KerasClassifier(build_fn=build_dnn_model, epochs=100, batch_size=128, verbose=0)

    model.fit(training_data, target_values)
    testing_data = data_pair.get_test_data().get_numpy()

    predicted = model.predict(testing_data)
    predicted_classes = np.argmax(predicted, axis=1)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_test_data().get_instances(), predicted_classes)]

    # Let's make the predictions a feature through use of the make feature from class,
    # But then restore the training data to the class
    # Set the self-predictions of the training data
    trained_classes = model.predict(training_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), trained_classes)]

    data_pair = sm.makeFeatureFromClass(data_pair, name="dnn")
    # Restore the training data
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), target_values)]

    return data_pair

def knn_scikit(data_pair, k=3, weights='uniform'):
    """Runs a kNN machine learning classifier using scikit-learn

    http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

    Args:
        data_pair: given dataset
        k: number of nearest neighbors
        weights: weight function used in prediction

    Returns:
        the result of a kNN machine learner using scikit-learn
    """
    k = abs(k)
    data_pair = copy.deepcopy(data_pair)
    training_data = data_pair.get_train_data().get_numpy()
    target_values = np.array([inst.get_target()[0] for
                              inst in data_pair.get_train_data().get_instances()])
    # For debugging purposes let's print out name of the function and dimensions of the data
    print('knn_scikit: training', training_data.shape)
    sys.stdout.flush()
    neigh = KNeighborsClassifier(n_neighbors=k, weights=weights)
    neigh.fit(training_data, target_values)
    testing_data = data_pair.get_test_data().get_numpy()
    print('knn_scikit: testing', testing_data.shape)
    sys.stdout.flush()
    predicted_classes = neigh.predict(testing_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_test_data().get_instances(), predicted_classes)]

    # Let's make the predictions a feature through use of the make feature from class,
    # But then restore the training data to the class
    # Set the self-predictions of the training data
    trained_classes = neigh.predict(training_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), trained_classes)]
    data_pair = sm.makeFeatureFromClass(data_pair, name="kNN")
    # Restore the training data
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), target_values)]

    return data_pair


def bayes_scikit(data_pair):
    """Runs a naive bayes machine learning classifier using scikit-learn

    http://scikit-learn.org/stable/modules/naive_bayes.html#naive-bayes

    Args:
        data_pair: given dataset

    Returns:
        the result of a naive bayes machine learner using scikit-learn
    """
    data_pair = copy.deepcopy(data_pair)
    training_data = data_pair.get_train_data().get_numpy()
    target_values = np.array([inst.get_target()[0] for
                              inst in data_pair.get_train_data().get_instances()])

    # For debugging purposes let's print out name of the function and dimensions of the data
    print('bayes_scikit: training', training_data.shape)
    sys.stdout.flush()
    bayes = GaussianNB()
    bayes.fit(training_data, target_values)
    testing_data = data_pair.get_test_data().get_numpy()
    print('bayes_scikit: testing', testing_data.shape)
    sys.stdout.flush()
    predicted_classes = bayes.predict(testing_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_test_data().get_instances(), predicted_classes)]

    # Let's make the predictions a feature through use of the make feature from class,
    # But then restore the training data to the class
    # Set the self-predictions of the training data
    trained_classes = bayes.predict(training_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), trained_classes)]

    data_pair = sm.makeFeatureFromClass(data_pair, name="bayes")
    # Restore the training data
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), target_values)]

    return data_pair

def bayes_scikit_multi(data_pair):
    """Runs a naive bayes machine learning classifier using scikit-learn

    http://scikit-learn.org/stable/modules/naive_bayes.html#naive-bayes

    Args:
        data_pair: given dataset

    Returns:
        the result of a naive bayes machine learner using scikit-learn
    """
    data_pair = copy.deepcopy(data_pair)
    training_data = data_pair.get_train_data().get_numpy()
    target_values = np.array([inst.get_target() for
                              inst in data_pair.get_train_data().get_instances()])
    new_training = []
    new_target = []
    for j, row in enumerate(training_data):
        for i, d in enumerate(row):
            new_training.append([i, d])
            new_target.append(target_values[j][i])
            
    new_training = np.array(new_training)
    new_target = np.array(new_target).reshape(-1, 1)
    training_data = new_training
    target_values = new_target

    # For debugging purposes let's print out name of the function and dimensions of the data
    print('bayes_scikit: training', training_data.shape)
    sys.stdout.flush()
    bayes = GaussianNB()
    bayes.fit(training_data, target_values)
    testing_data = data_pair.get_test_data().get_numpy()

    new_testing = []
    for j, row in enumerate(testing_data):
        for i, d in enumerate(row):
            new_testing.append([i, d])
    testing_data = np.array(new_testing)

    print('bayes_scikit: testing', testing_data.shape)
    sys.stdout.flush()
    predicted_classes = bayes.predict(testing_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_test_data().get_instances(), predicted_classes)]

    # Let's make the predictions a feature through use of the make feature from class,
    # But then restore the training data to the class
    # Set the self-predictions of the training data
    trained_classes = bayes.predict(training_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), trained_classes)]

    #data_pair = sm.makeFeatureFromClass(data_pair, name="bayes")
    # Restore the training data
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), target_values)]

    return data_pair

def omp_scikit(data_pair):
    """Runs an orthogonal matching pursuit machine learning classifier using scikit-learn

    http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuit.html#sklearn.linear_model.OrthogonalMatchingPursuit

    Args:
        data_pair: given dataset

    Returns:
        the result of an orthogonal matching pursuit using scikit-learn
    """
    data_pair = copy.deepcopy(data_pair)
    training_data = data_pair.get_train_data().get_numpy()
    target_values = np.array([inst.get_target()[0] for
                              inst in data_pair.get_train_data().get_instances()])
    # For debugging purposes let's print out name of the function and dimensions of the data
    print('omp_scikit: training', training_data.shape)
    sys.stdout.flush()

    omp = OrthogonalMatchingPursuit()
    omp.fit(training_data, target_values)
    testing_data = data_pair.get_test_data().get_numpy()
    print('omp_scikit: testing', testing_data.shape)
    sys.stdout.flush()
    predicted_classes = omp.predict(testing_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_test_data().get_instances(), predicted_classes)]

    # Let's make the predictions a feature through use of the make feature from class,
    # But then restore the training data to the class
    # Set the self-predictions of the training data
    trained_classes = omp.predict(training_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), trained_classes)]

    data_pair = sm.makeFeatureFromClass(data_pair, name="OMP")
    # Restore the training data
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), target_values)]

    return data_pair


def svc_scikit(data_pair, C=1.0, kernel='rbf'):
    """Runs a SVM machine learning classifier using scikit-learn

    http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

    Args:
        data_pair: given dataset
        kernel: the kernel type used in the algorithm

    Returns:
        a C-Support vector classification using scikit-learn
    """
    data_pair = copy.deepcopy(data_pair)
    training_data = data_pair.get_train_data().get_numpy()
    target_values = np.array([inst.get_target()[0] for
                              inst in data_pair.get_train_data().get_instances()])
    # Check for multiple target values
    target_value_check(target_values)
    # For debugging purposes let's print out name of the function and dimensions of the data
    print('svc_scikit: training', training_data.shape)
    sys.stdout.flush()

    svc = SVC(C=C, kernel=kernel)
    # Putting this in a try due to a number of classes equal to 1 error... this should not be happening
    try:
        svc.fit(training_data, target_values)
    except ValueError as e:
        print(training_data.shape, target_values, np.unique(target_values), target_values.shape)
        sys.stdout.flush
        raise e
    testing_data = data_pair.get_test_data().get_numpy()
    print('svc_scikit: testing', testing_data.shape)
    sys.stdout.flush()
    predicted_classes = svc.predict(testing_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_test_data().get_instances(), predicted_classes)]

    # Let's make the predictions a feature through use of the make feature from class,
    # But then restore the training data to the class
    # Set the self-predictions of the training data
    trained_classes = svc.predict(training_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), trained_classes)]

    data_pair = sm.makeFeatureFromClass(data_pair, name="SVC")
    # Restore the training data
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), target_values)]

    return data_pair

def svc_scikit_multi(data_pair, C=1.0, kernel='rbf'):
    """Runs a SVM machine learning classifier using scikit-learn
    http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    Args:
        data_pair: given dataset
        kernel: the kernel type used in the algorithm
    Returns:
        a C-Support vector classification using scikit-learn
    """
    data_pair = copy.deepcopy(data_pair)
    training_data = data_pair.get_train_data().get_numpy()
    target_values = np.array([inst.get_target() for
                              inst in data_pair.get_train_data().get_instances()])
    new_training = []
    new_target = []
    for j, row in enumerate(training_data):
        for i, d in enumerate(row):
            new_training.append([i, d])
            new_target.append(target_values[j][i])
            
    new_training = np.array(new_training)
    new_target = np.array(new_target).reshape(-1, 1)
    training_data = new_training
    target_values = new_target
    
    # Check for multiple target values
    #target_value_check(target_values)
    # For debugging purposes let's print out name of the function and dimensions of the data
    print('svc_scikit: training', training_data.shape)
    #sys.stdout.flush()

    svc = SVC(C=C, kernel=kernel)
    # Putting this in a try due to a number of classes equal to 1 error... this should not be happening
    try:
        svc.fit(training_data, target_values)
    except ValueError as e:
        print(training_data.shape, target_values, np.unique(target_values), target_values.shape)
        #sys.stdout.flush
        raise e
    testing_data = data_pair.get_test_data().get_numpy()
    new_testing = []
    for j, row in enumerate(testing_data):
        for i, d in enumerate(row):
            new_testing.append([i, d])
    testing_data = np.array(new_testing)

    print('svc_scikit: testing', testing_data.shape)
    #sys.stdout.flush()
    predicted_classes = svc.predict(testing_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_test_data().get_instances(), predicted_classes)]

    # Let's make the predictions a feature through use of the make feature from class,
    # But then restore the training data to the class
    # Set the self-predictions of the training data
    trained_classes = svc.predict(training_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), trained_classes)]
    print(trained_classes.shape)
    print(trained_classes[:2])
    #data_pair = sm.makeFeatureFromClass(data_pair, name="SVC")
    # Restore the training data
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), target_values)]

    return data_pair




def trees_scikit(data_pair, criterion="gini", splitter="best"):
    """Runs a decision tree machine learning classifier using scikit-learn

    http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier

    Args:
        data_pair: given dataset

    Returns:
        the result of a decision tree machine learner using scikit-learn
    """
    data_pair = copy.deepcopy(data_pair)
    training_data = data_pair.get_train_data().get_numpy()
    target_values = np.array([inst.get_target()[0] for
                              inst in data_pair.get_train_data().get_instances()])
    # Check for multiple target values
    target_value_check(target_values)
    # For debugging purposes let's print out name of the function and dimensions of the data
    print('trees_scikit: training', training_data.shape)
    sys.stdout.flush()

    tree = DecisionTreeClassifier(criterion=criterion, splitter=splitter)
    tree.fit(training_data, target_values)
    testing_data = data_pair.get_test_data().get_numpy()
    print('trees_scikit: testing', testing_data.shape)
    sys.stdout.flush()
    predicted_classes = tree.predict(testing_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_test_data().get_instances(), predicted_classes)]

    # Let's make the predictions a feature through use of the make feature from class,
    # But then restore the training data to the class
    # Set the self-predictions of the training data
    trained_classes = tree.predict(training_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), trained_classes)]

    data_pair = sm.makeFeatureFromClass(data_pair, name="Trees")
    # Restore the training data
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), target_values)]

    return data_pair


def random_forest_scikit(data_pair, n_estimators=100, class_weight=None, criterion='gini'):
    """Runs a random forest machine learning classifier using scikit-learn

    http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

    Args:
        data_pair: given dataset
        n_estimators: number of decision trees in the forest
        class_weight: mode of dict, list of dicts for weights associated
        with classes
        criterion: the function to measure the quality of a split

    Returns:
        the result of a random forest machine learner using scikit-learn
    """
    data_pair = copy.deepcopy(data_pair)
    training_data = data_pair.get_train_data().get_numpy()
    target_values = np.array([inst.get_target()[0] for
                              inst in data_pair.get_train_data().get_instances()])
    # Check for multiple target values
    target_value_check(target_values)
    # For debugging purposes let's print out name of the function and dimensions of the data
    print('random_forest_scikit: training', training_data.shape)
    sys.stdout.flush()

    forest = RandomForestClassifier(n_estimators=n_estimators, class_weight=class_weight, criterion=criterion)
    forest.fit(training_data, target_values)
    testing_data = data_pair.get_test_data().get_numpy()
    print('random_forest_scikit: testing', testing_data.shape)
    sys.stdout.flush()
    predicted_classes = forest.predict(testing_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_test_data().get_instances(), predicted_classes)]

    # Let's make the predictions a feature through use of the make feature from class,
    # But then restore the training data to the class
    # Set the self-predictions of the training data
    trained_classes = forest.predict(training_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), trained_classes)]

    data_pair = sm.makeFeatureFromClass(data_pair, name="Forest")
    # Restore the training data
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), target_values)]

    return data_pair


def boosting_scikit(data_pair, learning_rate=0.1,  n_estimators=100, max_depth=3):
    """Runs a gradient boosted machine learning classifier using scikit-learn

    http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html

    Args:
        data_pair: given dataset
        learning_rate: learning rate shrinks the contribution of each tree
        by learning_rate. There is a trade-off between learning_rate and
        n_estimators --sklearn
        n_estimators: number of boosting stages to perform
        max_depth: maximum depth of the individual regression estimators.
        The maximum depth limits the number of nodes in the tree. --sklearn

    Returns:
        the result of a gradient boosted machine learner using scikit-learn
    """
    data_pair = copy.deepcopy(data_pair)
    training_data = data_pair.get_train_data().get_numpy()
    target_values = np.array([inst.get_target()[0] for
                              inst in data_pair.get_train_data().get_instances()])
    # Check for multiple target values
    target_value_check(target_values)
    # For debugging purposes let's print out name of the function and dimensions of the data
    print('boosting_scikit: training', training_data.shape)
    sys.stdout.flush()

    boost = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)
    boost.fit(training_data, target_values)
    testing_data = data_pair.get_test_data().get_numpy()
    print('boosting_scikit: testing', testing_data.shape)
    sys.stdout.flush()
    predicted_classes = boost.predict(testing_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_test_data().get_instances(), predicted_classes)]

    # Let's make the predictions a feature through use of the make feature from class,
    # But then restore the training data to the class
    # Set the self-predictions of the training data
    trained_classes = boost.predict(training_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), trained_classes)]

    data_pair = sm.makeFeatureFromClass(data_pair, name="Boosting")
    # Restore the training data
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), target_values)]

    return data_pair


def MLP_scikit(data_pair, hidden_layer_sizes=(100,100),  alpha=0.0001, learning_rate='constant', learning_rate_init=0.001, momentum=0.9):
    """Runs a multi-layer perceptron machine learning classifier using scikit-learn

    http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

    Args:
        data_pair: given dataset
        hidden_layer_sizes: number of layers and number of nodes in each layer
        alpha: L2 regularization penalty
        learning_rate: learning rate schedule for weight updates --sklearn
        learning_rate_init: the initial learning rate used
        momentum: momentum for gradient descent update

    Returns:
        the result of a multi-layer perceptron machine learner using scikit-learn
    """
    data_pair = copy.deepcopy(data_pair)
    training_data = data_pair.get_train_data().get_numpy()
    target_values = np.array([inst.get_target()[0] for
                              inst in data_pair.get_train_data().get_instances()])
    # Check for multiple target values
    target_value_check(target_values)
    # For debugging purposes let's print out name of the function and dimensions of the data
    print('MLP_scikit: training', training_data.shape)
    sys.stdout.flush()

    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha, learning_rate=learning_rate, learning_rate_init=learning_rate_init, momentum=momentum)
    mlp.fit(training_data, target_values)
    testing_data = data_pair.get_test_data().get_numpy()
    print('MLP_scikit: testing', testing_data.shape)
    sys.stdout.flush()
    predicted_classes = mlp.predict(testing_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_test_data().get_instances(), predicted_classes)]

    # Let's make the predictions a feature through use of the make feature from class,
    # But then restore the training data to the class
    # Set the self-predictions of the training data
    trained_classes = mlp.predict(training_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), trained_classes)]

    data_pair = sm.makeFeatureFromClass(data_pair, name="MLP")
    # Restore the training data
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), target_values)]

    return data_pair


def best_linear_unbiased_estimate(data_pair, theta_0=0.1, theta_l=None,
                                 theta_u=None):
   """
   theta_0 An array with shape (n_features, ) or (1, ).
   The parameters in the autocorrelation model.
   If theta_l and theta_u are also specified,
   theta_0 is considered as the starting point for the maximum likelihood
   estimation of the best set of parameters.
   Default assumes isotropic autocorrelation model with theta0 = 1e-1.
   theta_l
   An array with shape matching theta_0's.
   Lower bound on the autocorrelation parameters for maximum likelihood
   estimation. Default is None, so that it skips maximum likelihood estimation
   and it uses theta_0.
   theta_u
   An array with shape matching theta0's.
   Upper bound on the autocorrelation parameters for maximum likelihood
   estimation. Default is None, so that it skips maximum likelihood estimation
   and it uses theta_0.
   -- From sklearn
   """
   data_pair = copy.deepcopy(data_pair)
   training_data = data_pair.get_train_data().get_numpy()
   target_values = np.array([inst.get_target()[0] for
                             inst in data_pair.get_train_data().get_instances()])
   # Check for multiple target values
   target_value_check(target_values)
   # For debugging purposes let's print out name of the function and dimensions of the data
   print('blup scikit: training', training_data.shape)
   sys.stdout.flush()

   blup = GaussianProcess(theta0=theta_0, thetaL=theta_l, thetaU=theta_u)
   blup.fit(training_data, target_values)
   testing_data = data_pair.get_test_data().get_numpy()
   print('blup scikit: testing', testing_data.shape)
   sys.stdout.flush()
   predicted_classes = blup.predict(testing_data)
   [inst.set_target([target]) for inst, target in
    zip(data_pair.get_test_data().get_instances(), predicted_classes)]

   # Let's make the predictions a feature through use of the make feature from class,
   # But then restore the training data to the class
   # Set the self-predictions of the training data
   trained_classes = blup.predict(training_data)
   [inst.set_target([target]) for inst, target in
    zip(data_pair.get_train_data().get_instances(), trained_classes)]

   data_pair = sm.makeFeatureFromClass(data_pair, name="BLUP")
   # Restore the training data
   [inst.set_target([target]) for inst, target in
    zip(data_pair.get_train_data().get_instances(), target_values)]

   return data_pair


def kmeans_cluster_scikit(data_pair, n_clusters=8):
    """Runs a k-means machine learning clustering using scikit-learn

    http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

    Args:
        data_pair: given dataset
        n_clusters: the number of clusters to form and the number of centroids
        to generate

    Returns:
        the result of a k-means clustering machine learner using scikit-learn
    """
    # Fix to remove deprication warning about booleans
    n_clusters = abs(int(n_clusters))
    data_pair = copy.deepcopy(data_pair)
    training_data = data_pair.get_train_data().get_numpy()
    target_values = np.array([inst.get_target()[0] for
                              inst in data_pair.get_train_data().get_instances()])
    # Check for multiple target values
    target_value_check(target_values)
    # For debugging purposes let's print out name of the function and dimensions of the data
    print('kmeans_cluster_scikit: training', training_data.shape)
    sys.stdout.flush()

    kmeans_cluster = KMeans(n_clusters)
    kmeans_cluster.fit(training_data, target_values)
    testing_data = data_pair.get_test_data().get_numpy()
    print('kmeans_cluster_scikit: testing', testing_data.shape)
    sys.stdout.flush()
    predicted_classes = kmeans_cluster.predict(testing_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_test_data().get_instances(), predicted_classes)]

    # Let's make the predictions a feature through use of the make feature from class,
    # But then restore the training data to the class
    # Set the self-predictions of the training data
    trained_classes = kmeans_cluster.predict(training_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), trained_classes)]

    data_pair = sm.makeFeatureFromClass(data_pair, name="KMEANS")
    # Restore the training data
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), target_values)]

    return data_pair

def agglomerative_cluster_scikit(data_pair, n_clusters=8):
    """Runs an agglomerative clustering machine learning using scikit-learn

    http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html

    Args:
        data_pair: given dataset
        n_clusters: the number of clusters to form and the number of centroids
        to generate

    Returns:
        the result of a agglomerative clustering machine learner using scikit-learn
    """
    n_clusters = abs(int(n_clusters))
    data_pair = copy.deepcopy(data_pair)
    training_data = data_pair.get_train_data().get_numpy()
    target_values = np.array([inst.get_target()[0] for
                              inst in data_pair.get_train_data().get_instances()])
    # Check for multiple target values
    target_value_check(target_values)
    # For debugging purposes let's print out name of the function and dimensions of the data
    print('agglomerative_cluster_scikit: training', training_data.shape)
    sys.stdout.flush()

    agglomerative_cluster = AgglomerativeClustering(n_clusters=n_clusters)
    agglomerative_cluster.fit(training_data, target_values)
    testing_data = data_pair.get_test_data().get_numpy()
    print('agglomerative_cluster_scikit: testing', testing_data.shape)
    sys.stdout.flush()
    predicted_classes = agglomerative_cluster.fit_predict(testing_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_test_data().get_instances(), predicted_classes)]

    # Let's make the predictions a feature through use of the make feature from class,
    # But then restore the training data to the class
    # Set the self-predictions of the training data
    trained_classes = agglomerative_cluster.fit_predict(training_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), trained_classes)]

    data_pair = sm.makeFeatureFromClass(data_pair, name="AGGLOMERATIVE")
    # Restore the training data
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), target_values)]

    return data_pair

def birch_cluster_scikit(data_pair, n_clusters=8, branching_factor=50):
    """Runs an birch clustering machine learning using scikit-learn

    http://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html

    Args:
        data_pair: given dataset
        n_clusters: the number of clusters to form and the number of centroids
        to generate
        branching_factor: Maximum number of CF subclusters in each node

    Returns:
        the result of a birch clustering machine learner using scikit-learn
    """
    n_clusters = abs(int(n_clusters))
    branching_factor = abs(int(branching_factor))
    data_pair = copy.deepcopy(data_pair)
    training_data = data_pair.get_train_data().get_numpy()
    target_values = np.array([inst.get_target()[0] for
                              inst in data_pair.get_train_data().get_instances()])
    # Check for multiple target values
    target_value_check(target_values)
    # For debugging purposes let's print out name of the function and dimensions of the data
    print('birch_cluster_scikit: training', training_data.shape)
    sys.stdout.flush()

    birch_cluster = Birch(n_clusters=n_clusters, branching_factor=branching_factor)
    birch_cluster.fit(training_data, target_values)
    testing_data = data_pair.get_test_data().get_numpy()
    print('birch_cluster_scikit: testing', testing_data.shape)
    sys.stdout.flush()
    predicted_classes = birch_cluster.predict(testing_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_test_data().get_instances(), predicted_classes)]

    # Let's make the predictions a feature through use of the make feature from class,
    # But then restore the training data to the class
    # Set the self-predictions of the training data
    trained_classes = birch_cluster.predict(training_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), trained_classes)]

    data_pair = sm.makeFeatureFromClass(data_pair, name="BIRCH")
    # Restore the training data
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), target_values)]

    return data_pair

def ball_tree_cluster_scikit(data_pair, leaf_size=40):
    """
    Implements scikit ball tree
    http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html
    """
    data_pair = copy.deepcopy(data_pair)
    training_data = data_pair.get_train_data().get_numpy()
    target_values = np.array([inst.get_target()[0] for
                              inst in data_pair.get_train_data().get_instances()])
    # Check for multiple target values
    target_value_check(target_values)
    # For debugging purposes let's print out name of the function and dimensions of the data
    print('ball_tree_cluster_scikit: training', training_data.shape)
    sys.stdout.flush()

    ball_tree_cluster = BallTree(training_data, leaf_size=leaf_size)
    # ball_tree_cluster.fit(training_data, target_values)
    testing_data = data_pair.get_test_data().get_numpy()
    print('ball_tree_cluster_scikit: testing', testing_data.shape)
    sys.stdout.flush()
    predicted_classes = ball_tree_cluster.query(testing_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_test_data().get_instances(), predicted_classes)]

    # Let's make the predictions a feature through use of the make feature from class,
    # But then restore the training data to the class
    # Set the self-predictions of the training data
    trained_classes = ball_tree_cluster.query(testing_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), trained_classes)]

    data_pair = sm.makeFeatureFromClass(data_pair, name="BALLTREE")
    # Restore the training data
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), target_values)]

    return data_pair


def logistic_regression_scikit(data_pair, penalty='l2', dual=False, tol=0.0001,
                               C=1.0, fit_intercept=True, intercept_scaling=1,
                               class_weight=None, solver='liblinear', max_iter=100,
                               multi_class='ovr', warm_start=False):
    """Runs a logistic regression machine learning classifier using scikit-learn

    http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression

    Args:
        data_pair: given dataset
        penalty: number of nearest neighbors
        dual: weight function used in prediction
        tol: Tolerance for stopping criteria --sklearn
        C: Inverse of regularization strength --sklearn
        fit_intercept: Specifies if a constant (a.k.a. bias or intercept)
        should be added to the decision function. --sklearn
        intercept_scaling: changes the intercept
        class_weight: Weights associated with classes in the form
        {class_label: weight} --sklearn
        solver: Algorithm to use in the optimization problem --sklearn
        max_iter: Maximum number of iterations taken for the solvers to
        converge --sklearn
        multi_class: If the option chosen is ‘ovr’, then a binary problem is
        fit for each label. Else the loss minimised is the multinomial loss
        fit across the entire probability distribution. --sklearn
        warm_start: When set to True, reuse the solution of the previous call
        to fit as initialization, otherwise, just erase the previous solution.
        Useless for liblinear solver. --sklearn

    Returns:
        the result of a logistic regression machine learner using scikit-learn
    """
    data_pair = copy.deepcopy(data_pair)
    training_data = data_pair.get_train_data().get_numpy()
    target_values = np.array([inst.get_target()[0] for
                              inst in data_pair.get_train_data().get_instances()])
    # Check for multiple target values
    target_value_check(target_values)
    # For debugging purposes let's print out name of the function and dimensions of the data
    print('logistic_regression_scikit: training', training_data.shape)
    sys.stdout.flush()

    logistic_regression = LogisticRegression(penalty=penalty, dual=dual, tol=tol,
                                             C=C, fit_intercept=fit_intercept,
                                             intercept_scaling=intercept_scaling,
                                             class_weight=class_weight, solver=solver,
                                             max_iter=max_iter,
                                             multi_class=multi_class, warm_start=warm_start)
    logistic_regression.fit(training_data, target_values)
    testing_data = data_pair.get_test_data().get_numpy()
    print('logistic_regression_scikit: testing', testing_data.shape)
    sys.stdout.flush()
    predicted_classes = logistic_regression.predict(testing_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_test_data().get_instances(), predicted_classes)]

    # Let's make the predictions a feature through use of the make feature from class,
    # But then restore the training data to the class
    # Set the self-predictions of the training data
    trained_classes = logistic_regression.predict(training_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), trained_classes)]

    data_pair = sm.makeFeatureFromClass(data_pair, name='LOGR')
    # Restore the training data
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), target_values)]

    return data_pair

def spectral_cluster_scikit(data_pair, n_clusters=8, eigen_solver=None):
    """Runs a spectral machine learning clustering using scikit-learn

    http://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html#sklearn.cluster.SpectralClustering

    .. note::

       This method is currently unused.

    Args:
        data_pair: given dataset
        n_clusters: the dimension of the projection space
        eigen_solver: The eigenvalue decomposition strategy to use

    Returns:
        the result of a spectral clustering machine learner using scikit-learn
    """
    # Convert n_clusters to int in case a boolean is received
    n_clusters = abs(int(n_clusters))
    data_pair = copy.deepcopy(data_pair)
    training_data = data_pair.get_train_data().get_numpy()
    target_values = np.array([inst.get_target()[0] for
                              inst in data_pair.get_train_data().get_instances()])
    # Check for multiple target values
    target_value_check(target_values)
    # For debugging purposes let's print out name of the function and dimensions of the data
    print('spectral_cluster_scikit: training', training_data.shape)
    sys.stdout.flush()

    spectral_cluster = SpectralClustering(n_clusters, eigen_solver=eigen_solver)
    spectral_cluster.fit(training_data, target_values)
    testing_data = data_pair.get_test_data().get_numpy()
    print('spectral_cluster_scikit: testing', testing_data.shape)
    sys.stdout.flush()

    predicted_classes = spectral_cluster.fit_predict(testing_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_test_data().get_instances(), predicted_classes)]

    # Let's make the predictions a feature through use of the make feature from class,
    # But then restore the training data to the class
    # Set the self-predictions of the training data
    trained_classes = spectral_cluster.fit_predict(training_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), trained_classes)]

    data_pair = sm.makeFeatureFromClass(data_pair, name="SPECTRAL")
    # Restore the training data
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), target_values)]

    return data_pair


def gmm_scikit(data_pair, n_components=2, covariance_type='full', tol=0.001,
               reg_covar=1e-06, max_iter=100, n_init=1, init_params='kmeans',
               weights_init=None, means_init=None, precisions_init=None,
               warm_start=False):
    """Runs a gaussian mixture model machine learning clustering using scikit-learn

    http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture

    Args:
        data_pair: given dataset
        n_components: The number of mixture components
        covariance_type: String describing the type of covariance parameters to use
        tol: The convergence threshold. EM iterations will stop when the lower
        bound average gain is below this threshold. --sklearn
        reg_covar: Non-negative regularization added to the diagonal of
        covariance. Allows to assure that the covariance matrices are all
        positive. --sklearn
        max_iter: The number of EM iterations to perform.
        n_init: The number of initializations to perform. The best results
        are kept. --sklearn
        init_params: The method used to initialize the weights, the means and
        the precisions. --sklearn
        weights_init: The user-provided initial weights
        means_init: The user-provided initial means
        precisions_init: The user-provided initial precisions
        warm_start: If ‘warm_start’ is True, the solution of the last fitting is
        used as initialization for the next call of fit(). This can speed up
        convergence when fit is called several time on similar problems.
        --sklearn

    Returns:
        the result of a gaussian mixture model machine learner using scikit-learn
    """
    # Convert n_components to int in case a boolean is received
    n_components = abs(int(n_components))
    data_pair = copy.deepcopy(data_pair)
    training_data = data_pair.get_train_data().get_numpy()
    target_values = np.array([inst.get_target()[0] for
                              inst in data_pair.get_train_data().get_instances()])
    # Check for multiple target values
    target_value_check(target_values)
    # For debugging purposes let's print out name of the function and dimensions of the data
    print('gmm_scikit: training', training_data.shape)
    sys.stdout.flush()

    gmm_mixture = GaussianMixture(n_components=n_components, covariance_type=covariance_type,
                                  tol=tol, reg_covar=reg_covar, max_iter=max_iter, n_init=n_init,
                                  init_params=init_params, weights_init=weights_init, means_init=means_init,
                                  precisions_init=precisions_init, warm_start=warm_start)
    gmm_mixture.fit(training_data, target_values)
    testing_data = data_pair.get_test_data().get_numpy()
    print('gmm_scikit: testing', testing_data.shape)
    sys.stdout.flush()

    predicted_classes = gmm_mixture.predict(testing_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_test_data().get_instances(), predicted_classes)]

    # Let's make the predictions a feature through use of the make feature from class,
    # But then restore the training data to the class
    # Set the self-predictions of the training data
    trained_classes = gmm_mixture.predict(training_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), trained_classes)]

    data_pair = sm.makeFeatureFromClass(data_pair, name="GMM")
    # Restore the training data
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), target_values)]

    return data_pair

# -----------------------------------------  Regression Methods --------------------------------------------------------

def xgboost(data_pair, learning_rate=0.037, max_depth=5, subsample=0.8, xglambda=0.8, alpha=0.4):
    """Runs a xgboost machine learning regressor

    http://xgboost.readthedocs.io/en/latest/get_started/index.html

    Args:
        data_pair: given dataset
        learning_rate: step size shrinkage used in update to prevents overfitting
        max_depth: maximum depth of a tree, increase this value will make the model more complex / likely to be overfitting
        subsample: subsample ratio of the training instance. Setting it to 0.5 means that XGBoost randomly collected half of the data instances to grow trees and this will prevent overfitting
        xglambda: L2 regularization term on weights, increase this value will make model more conservative
        alpha: L1 regularization term on weights, increase this value will make model more conservative

    Returns:
        the result of a xgboost machine learner
    """
    data_pair = copy.deepcopy(data_pair)
    training_data = data_pair.get_train_data().get_numpy()
    target_values = np.array([inst.get_target()[0] for
                              inst in data_pair.get_train_data().get_instances()])
    # Check for multiple target values
    target_value_check(target_values)
    # For debugging purposes let's print out name of the function and dimensions of the data
    print('xgboost: training', training_data.shape)
    sys.stdout.flush()


    dtrain = xgb.DMatrix(training_data, target_values)
    testing_data = data_pair.get_test_data().get_numpy()
    dtest = xgb.DMatrix(testing_data)
    print('xgboost: testing', testing_data.shape)
    sys.stdout.flush()
    y_mean = np.mean(target_values)

    xgb_params = {
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'subsample': subsample,
        'lambda': xglambda,
        'alpha': alpha,

        'base_score': y_mean,
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'silent': 1
    }
    num_boost_round = 242


    model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_round)
    xgb_pred = model.predict(dtest)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_test_data().get_instances(), xgb_pred)]

    # Let's make the predictions a feature through use of the make feature from class,
    # But then restore the training data to the class
    # Set the self-predictions of the training data
    trained_classes = model.predict(dtrain)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), trained_classes)]

    data_pair = sm.makeFeatureFromClass(data_pair, name="xgboost")
    # Restore the training data
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), target_values)]

    return data_pair

def lightgbm(data_pair, max_bin=10, learning_rate=.0021, boosting_type='gbdt'):
    """Runs a LightGBM machine learning regressor

    http://lightgbm.readthedocs.io/en/latest/index.html

    Args:
        data_pair: given dataset
        learning_rate: learning rate shrinks the contribution of each tree
        by learning_rate. There is a trade-off between learning_rate and
        n_estimators --sklearn
        n_estimators: number of boosting stages to perform

    Returns:
        the result of a LightGBM machine learner
    """
    params = {
        'max_bin': 10,
        'learning_rate': 0.0021,  # shrinkage_rate
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'l1',  # or 'mae'
        'sub_feature': 0.5,  # feature_fraction -- OK, back to .5, but maybe later increase this
        'bagging_fraction': 0.85,  # sub_row
        'bagging_freq': 40,
        'num_leaves': 512,  # num_leaf
        'min_data': 500,  # min_data_in_leaf
        'min_hessian': 0.05,  # min_sum_hessian_in_leaf
        'verbose': 0
    }
    data_pair = copy.deepcopy(data_pair)
    training_data = data_pair.get_train_data().get_numpy()
    target_values = np.array([inst.get_target()[0] for
                              inst in data_pair.get_train_data().get_instances()])
    # Check for multiple target values
    target_value_check(target_values)
    # For debugging purposes let's print out name of the function and dimensions of the data
    print('lightgbm: training', training_data.shape)
    sys.stdout.flush()

    d_train = lgb.Dataset(training_data, label=target_values)
    testing_data = data_pair.get_test_data().get_numpy()
    print('lightgbm: testing', testing_data.shape)
    sys.stdout.flush()

    clf = lgb.train(params, d_train, num_boost_round=430)
    lgb_pred = clf.predict(testing_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_test_data().get_instances(), lgb_pred)]

    # Let's make the predictions a feature through use of the make feature from class,
    # But then restore the training data to the class
    # Set the self-predictions of the training data
    trained_classes = clf.predict(training_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), trained_classes)]

    data_pair = sm.makeFeatureFromClass(data_pair, name="lightgbm")
    # Restore the training data
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), target_values)]

    return data_pair


def adaboost_regression_scikit(data_pair, learning_rate=0.1,  n_estimators=100):
    """Runs a adaboost machine learning classifier using scikit-learn

    http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html

    Args:
        data_pair: given dataset
        learning_rate: learning rate shrinks the contribution of each tree
        by learning_rate. There is a trade-off between learning_rate and
        n_estimators --sklearn
        n_estimators: number of boosting stages to perform

    Returns:
        the result of a adaboost machine learner using scikit-learn
    """
    data_pair = copy.deepcopy(data_pair)
    training_data = data_pair.get_train_data().get_numpy()
    target_values = np.array([inst.get_target()[0] for
                              inst in data_pair.get_train_data().get_instances()])
    # Check for multiple target values
    target_value_check(target_values)
    # For debugging purposes let's print out name of the function and dimensions of the data
    print('adaboost_regression_scikit: training', training_data.shape)
    sys.stdout.flush()

    boost = AdaBoostRegressor(learning_rate=learning_rate, n_estimators=n_estimators)
    boost.fit(training_data, target_values)
    testing_data = data_pair.get_test_data().get_numpy()
    print('adaboost_regression_scikit: testing', testing_data.shape)
    sys.stdout.flush()
    predicted_classes = boost.predict(testing_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_test_data().get_instances(), predicted_classes)]

    # Let's make the predictions a feature through use of the make feature from class,
    # But then restore the training data to the class
    # Set the self-predictions of the training data
    trained_classes = boost.predict(training_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), trained_classes)]

    data_pair = sm.makeFeatureFromClass(data_pair, name="adaboost_regression")
    # Restore the training data
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), target_values)]

    return data_pair


def gradient_boosting_regression_scikit(data_pair, learning_rate=0.1,  n_estimators=100, max_depth=3):
    """Runs a gradient boosted machine learning regression using scikit-learn

    http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html

    Args:
        data_pair: given dataset
        learning_rate: learning rate shrinks the contribution of each tree
        by learning_rate. There is a trade-off between learning_rate and
        n_estimators --sklearn
        n_estimators: number of boosting stages to perform
        max_depth: maximum depth of the individual regression estimators.
        The maximum depth limits the number of nodes in the tree. --sklearn

    Returns:
        the result of a gradient boosted machine learner using scikit-learn
    """
    data_pair = copy.deepcopy(data_pair)
    training_data = data_pair.get_train_data().get_numpy()
    target_values = np.array([inst.get_target()[0] for
                              inst in data_pair.get_train_data().get_instances()])
    # Check for multiple target values
    target_value_check(target_values)
    # For debugging purposes let's print out name of the function and dimensions of the data
    print('boosting_regression_scikit: training', training_data.shape)
    sys.stdout.flush()

    boost = GradientBoostingRegressor(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)
    boost.fit(training_data, target_values)
    testing_data = data_pair.get_test_data().get_numpy()
    print('boosting_regression_scikit: testing', testing_data.shape)
    sys.stdout.flush()
    predicted_classes = boost.predict(testing_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_test_data().get_instances(), predicted_classes)]

    # Let's make the predictions a feature through use of the make feature from class,
    # But then restore the training data to the class
    # Set the self-predictions of the training data
    trained_classes = boost.predict(training_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), trained_classes)]

    data_pair = sm.makeFeatureFromClass(data_pair, name="gradient_boosting_regression")
    # Restore the training data
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), target_values)]

    return data_pair


def random_forest_regression_scikit(data_pair, n_estimators=100, criterion='mae'):
    """Runs a random forest machine learning regressor using scikit-learn

    http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

    Args:
        data_pair: given dataset
        n_estimators: number of decision trees in the forest
        criterion: the function to measure the quality of a split

    Returns:
        the result of a random forest machine learner using scikit-learn
    """
    data_pair = copy.deepcopy(data_pair)
    training_data = data_pair.get_train_data().get_numpy()
    target_values = np.array([inst.get_target()[0] for
                              inst in data_pair.get_train_data().get_instances()])
    # Check for multiple target values
    target_value_check(target_values)
    # For debugging purposes let's print out name of the function and dimensions of the data
    print('random_forest_regression_scikit: training', training_data.shape)
    sys.stdout.flush()

    forest = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion)
    forest.fit(training_data, target_values)
    testing_data = data_pair.get_test_data().get_numpy()
    print('random_forest_regression_scikit: testing', testing_data.shape)
    sys.stdout.flush()
    predicted_classes = forest.predict(testing_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_test_data().get_instances(), predicted_classes)]

    # Let's make the predictions a feature through use of the make feature from class,
    # But then restore the training data to the class
    # Set the self-predictions of the training data
    trained_classes = forest.predict(training_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), trained_classes)]

    data_pair = sm.makeFeatureFromClass(data_pair, name="Forest")
    # Restore the training data
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), target_values)]

    return data_pair

def svm_regression_scikit(data_pair, kernel='rbf'):
    """Runs a SVM machine learning regressor using scikit-learn

    http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html

    Args:
        data_pair: given dataset
        kernel: the kernel type used in the algorithm

    Returns:
        a C-Support vector classification using scikit-learn
    """
    data_pair = copy.deepcopy(data_pair)
    training_data = data_pair.get_train_data().get_numpy()
    target_values = np.array([inst.get_target()[0] for
                              inst in data_pair.get_train_data().get_instances()])
    # Check for multiple target values
    target_value_check(target_values)
    # For debugging purposes let's print out name of the function and dimensions of the data
    print('svr_scikit: training', training_data.shape)
    sys.stdout.flush()

    svr = SVR(kernel=kernel)
    # Putting this in a try due to a number of classes equal to 1 error... this should not be happening
    try:
        svr.fit(training_data, target_values)
    except ValueError as e:
        print(training_data.shape, target_values, np.unique(target_values), target_values.shape)
        sys.stdout.flush
        raise e
    testing_data = data_pair.get_test_data().get_numpy()
    print('svr_scikit: testing', testing_data.shape)
    sys.stdout.flush()
    predicted_classes = svr.predict(testing_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_test_data().get_instances(), predicted_classes)]

    # Let's make the predictions a feature through use of the make feature from class,
    # But then restore the training data to the class
    # Set the self-predictions of the training data
    trained_classes = svr.predict(training_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), trained_classes)]

    data_pair = sm.makeFeatureFromClass(data_pair, name="svm_regression")
    # Restore the training data
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), target_values)]

    return data_pair


def descision_tree_regression_scikit(data_pair):
    """Runs a decision tree machine learning regressor using scikit-learn

    http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html

    Args:
        data_pair: given dataset

    Returns:
        the result of a decision tree machine learner using scikit-learn
    """
    data_pair = copy.deepcopy(data_pair)
    training_data = data_pair.get_train_data().get_numpy()
    target_values = np.array([inst.get_target()[0] for
                              inst in data_pair.get_train_data().get_instances()])
    # Check for multiple target values
    target_value_check(target_values)
    # For debugging purposes let's print out name of the function and dimensions of the data
    print('descision_tree_regression_scikit: training', training_data.shape)
    sys.stdout.flush()

    tree = DecisionTreeClassifier()
    tree.fit(training_data, target_values)
    testing_data = data_pair.get_test_data().get_numpy()
    print('descision_tree_regression_scikit: testing', testing_data.shape)
    sys.stdout.flush()
    predicted_classes = tree.predict(testing_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_test_data().get_instances(), predicted_classes)]

    # Let's make the predictions a feature through use of the make feature from class,
    # But then restore the training data to the class
    # Set the self-predictions of the training data
    trained_classes = tree.predict(training_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), trained_classes)]

    data_pair = sm.makeFeatureFromClass(data_pair, name="descision_tree_regression")
    # Restore the training data
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), target_values)]

    return data_pair


def knn_regression_scikit(data_pair, k=3, weights='uniform'):
    """Runs a kNN machine learning regressor using scikit-learn

    http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html

    Args:
        data_pair: given dataset
        k: number of nearest neighbors
        weights: weight function used in prediction

    Returns:
        the result of a kNN machine learner using scikit-learn
    """
    k = abs(k)
    data_pair = copy.deepcopy(data_pair)
    training_data = data_pair.get_train_data().get_numpy()
    target_values = np.array([inst.get_target()[0] for
                              inst in data_pair.get_train_data().get_instances()])
    # For debugging purposes let's print out name of the function and dimensions of the data
    print('knn_regression_scikit: training', training_data.shape)
    sys.stdout.flush()
    neigh = KNeighborsClassifier(n_neighbors=k, weights=weights)
    neigh.fit(training_data, target_values)
    testing_data = data_pair.get_test_data().get_numpy()
    print('knn_regression_scikit: testing', testing_data.shape)
    sys.stdout.flush()
    predicted_classes = neigh.predict(testing_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_test_data().get_instances(), predicted_classes)]

    # Let's make the predictions a feature through use of the make feature from class,
    # But then restore the training data to the class
    # Set the self-predictions of the training data
    trained_classes = neigh.predict(training_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), trained_classes)]
    data_pair = sm.makeFeatureFromClass(data_pair, name="knn_regression")
    # Restore the training data
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), target_values)]

    return data_pair

def batch_SGD_scikit(data_pair, penalty='l2', alpha=0.0001):
    """Runs batch learning on a SGD classifier using scikit-learn

    http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html

    Args:
        data_pair: given dataset
        penalty: regularization term to be used
        alpha: constant that multiplies the regularization term

    Returns:
        the result of a SGD machine learner using scikit-learn
    """
    data_pair = copy.deepcopy(data_pair)
    if isinstance(data_pair, GTMOEPImagePair):
        # Load in data from local files
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        train_labels_list = data_pair.get_train_labels()
    else:
        raise ValueError(
            'This method only trains batched data.'
            )

    model = SGDClassifier(penalty=penalty, alpha=alpha, shuffle=False)

    for data, label_data in zip(train_list, train_labels_list):
        # flatten (size, height, width, 3) image data into (size, n) input data
        # sklearn classifiers only take in 2D input data
        flat_list = []
        for i in range(len(data)):
            flat_data = data[i].flatten()
            flat_list.append(flat_data)
        data = np.vstack(flat_list)
        # train model on one batch of data
        model.partial_fit(data, label_data.ravel(), classes=np.array([0,1]))

    predictions = []
    for data in test_list:
        flat_list = []
        for i in range(len(data)):
            flat_data = data[i].flatten()
            flat_list.append(flat_data)
        data = np.vstack(flat_list)
        # predict on one batch of test data
        predictions.append(model.predict(data))

    # pass list of prediction arrays to be stored properly
    data_pair.set_prediction(predictions)

    return data_pair

def batch_passive_scikit(data_pair, C=1.0):
    """Runs batch learning on a Passive Aggressive classifier using scikit-learn

    http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html

    Args:
        data_pair: given dataset
        C: maximum step size

    Returns:
        the result of a passive aggressive machine learner using scikit-learn
    """
    data_pair = copy.deepcopy(data_pair)
    if isinstance(data_pair, GTMOEPImagePair):
        # Load in data from local files
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        train_labels_list = data_pair.get_train_labels()
    else:
        raise ValueError(
            'This method only trains batched data.'
            )

    model = PassiveAggressiveClassifier(C=C, shuffle=False)

    for data, label_data in zip(train_list, train_labels_list):
        # flatten (size, height, width, 3) image data into (size, n) input data
        # sklearn classifiers only take in 2D input data
        flat_list = []
        for i in range(len(data)):
            flat_data = data[i].flatten()
            flat_list.append(flat_data)
        data = np.vstack(flat_list)
        # train model on one batch of training data
        model.partial_fit(data, label_data.ravel(), classes=np.array([0,1]))

    predictions = []
    for data in test_list:
        flat_list = []
        for i in range(len(data)):
            flat_data = data[i].flatten()
            flat_list.append(flat_data)
        data = np.vstack(flat_list)
        # predict on one batch of test data
        predictions.append(model.predict(data))

    # pass list of prediction arrays to be stored properly
    data_pair.set_prediction(predictions)

    return data_pair

def extra_trees_scikit(data_pair, n_estimators=100, max_depth=6, criterion='entropy'):
    """Runs a random forest machine learning classifier using scikit-learn

    http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

    Args:
        data_pair: given dataset
        n_estimators: number of decision trees in the forest
        class_weight: mode of dict, list of dicts for weights associated
        with classes
        criterion: the function to measure the quality of a split

    Returns:
        the result of a random forest machine learner using scikit-learn
    """
    data_pair = copy.deepcopy(data_pair)
    training_data = data_pair.get_train_data().get_numpy()
    target_values = np.array([inst.get_target()[0] for
                              inst in data_pair.get_train_data().get_instances()])
    # Check for multiple target values
    target_value_check(target_values)
    # For debugging purposes let's print out name of the function and dimensions of the data
    print('extra_trees_scikit: training', training_data.shape)
    sys.stdout.flush()

    forest = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion)
    forest.fit(training_data, target_values)
    testing_data = data_pair.get_test_data().get_numpy()
    print('extra_trees_scikit: testing', testing_data.shape)
    sys.stdout.flush()
    predicted_classes = forest.predict(testing_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_test_data().get_instances(), predicted_classes)]

    # Let's make the predictions a feature through use of the make feature from class,
    # But then restore the training data to the class
    # Set the self-predictions of the training data
    trained_classes = forest.predict(training_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), trained_classes)]

    data_pair = sm.makeFeatureFromClass(data_pair, name="ExtraTrees")
    # Restore the training data
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), target_values)]

    return data_pair

def nusvc_scikit(data_pair, nu=0.9999, kernel='rbf',probability=True):
    """Runs a SVM machine learning classifier using scikit-learn

    http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

    Args:
        data_pair: given dataset
        kernel: the kernel type used in the algorithm

    Returns:
        a C-Support vector classification using scikit-learn
    """
    data_pair = copy.deepcopy(data_pair)
    training_data = data_pair.get_train_data().get_numpy()
    target_values = np.array([inst.get_target()[0] for
                              inst in data_pair.get_train_data().get_instances()])
    # Check for multiple target values
    target_value_check(target_values)
    # For debugging purposes let's print out name of the function and dimensions of the data
    print('nusvc_scikit: training', training_data.shape)
    sys.stdout.flush()

    svc = NuSVC(nu=nu, kernel=kernel,probability=probability)
    # Putting this in a try due to a number of classes equal to 1 error... this should not be happening
    try:
        svc.fit(training_data, target_values)
    except ValueError as e:
        print(training_data.shape, target_values, np.unique(target_values), target_values.shape)
        sys.stdout.flush
        raise e
    testing_data = data_pair.get_test_data().get_numpy()
    print('nusvc_scikit: testing', testing_data.shape)
    sys.stdout.flush()
    predicted_classes = svc.predict(testing_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_test_data().get_instances(), predicted_classes)]

    # Let's make the predictions a feature through use of the make feature from class,
    # But then restore the training data to the class
    # Set the self-predictions of the training data
    trained_classes = svc.predict(training_data)
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), trained_classes)]

    data_pair = sm.makeFeatureFromClass(data_pair, name="NuSVC")
    # Restore the training data
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), target_values)]

    return data_pair

def my_arg_max(data_pair, sampling_rate=1):
    """Runs an arg max

    Assigns the index of the maximum for each row as the class for
    test data

    Args:
        data_pair: given dataset
        sampling_rate: rate of data sample

    Returns:
        the result of an arg max
    """
    data_pair = copy.deepcopy(data_pair)
    # Cache the truth data for the training set
    target_values = np.array([inst.get_target() for
                              inst in data_pair.get_train_data().get_instances()])
    for dataset in [data_pair.get_train_data(), data_pair.get_test_data()]:
        numpy_data = dataset.get_numpy()
        # For debugging purposes let's print out name of the function and dimensions of the data
        print('arg_max', numpy_data.shape)
        sys.stdout.flush()

        maxes = [np.argmax(row) / sampling_rate for row in numpy_data]
        [inst.set_target([target]) for inst, target in
         zip(dataset.get_instances(), maxes)]

    data_pair = sm.makeFeatureFromClass(data_pair, name='ARGMAX')
    # Restore the training data
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), target_values)]

    return data_pair

def my_arg_min(data_pair, sampling_rate=1):
    """Runs an arg min

    Assigns the index of the minimum for each row as the class for
    test data

    Args:
        data_pair: given dataset
        sampling_rate: rate of data sample

    Returns:
        the result of an arg min
    """
    data_pair = copy.deepcopy(data_pair)
    # Cache the truth data for the training set
    target_values = np.array([inst.get_target() for
                              inst in data_pair.get_train_data().get_instances()])
    for dataset in [data_pair.get_train_data(), data_pair.get_test_data()]:
        numpy_data = dataset.get_numpy()
        # For debugging purposes let's print out name of the function and dimensions of the data
        print('arg_min', numpy_data.shape)
        sys.stdout.flush()

        mins = [np.argmax(row) / sampling_rate for row in numpy_data]
        [inst.set_target([target]) for inst, target in
         zip(dataset.get_instances(), mins)]

    data_pair = sm.makeFeatureFromClass(data_pair, name='ARGMIN')
    # Restore the training data
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), target_values)]

    return data_pair


def my_depth_estimate(data_pair, sampling_rate=1, off_nadir_angle=20.0):
    """Runs a depth estimate

    Assigns difference between the first and last elements to
    the class for test data

    Args:
        data_pair: given dataset
        sampling_rate: sample rate of the data
        off_nadir_angle: off_nadir_angle in degrees

    Returns:
        the result of a kNN machine learner using scikit-learn
    """

    off_nadir_angle = math.pi / 180.0 * off_nadir_angle

    data_pair = copy.deepcopy(data_pair)
    # Cache the truth data for the training set
    target_values = np.array([inst.get_target() for
                              inst in data_pair.get_train_data().get_instances()])
    for dataset in [data_pair.get_train_data(), data_pair.get_test_data()]:
        numpy_data = dataset.get_numpy()
        # For debugging purposes let's print out name of the function and dimensions of the data
        # print('depth_estimate', numpy_data.shape)
        # sys.stdout.flush()

        mins = []
        for row, num in zip(numpy_data, range(numpy_data.shape[0])):
            try:
                # print row
                if len(row) == 1:
                    mins.append(-1.0)
                else:
                    opl_estimate = (row[-1] - row[0]) / sampling_rate
                    refracted_angle = math.asin(math.sin(off_nadir_angle) / 1.33)
                    depth_estimate = opl_estimate * math.cos(refracted_angle)
                    mins.append(depth_estimate)
                # mins.append(np.min(np.diff(np.unique(row)))/sampling_rate)
            except IndexError as no_bottom_error:
                mins.append(-1.0)
        [inst.set_target([target]) for inst, target in
         zip(dataset.get_instances(), mins)]

    data_pair = sm.makeFeatureFromClass(data_pair, name='DepthEstimate')
    # Restore the training data
    [inst.set_target([target]) for inst, target in
     zip(data_pair.get_train_data().get_instances(), target_values)]

    return data_pair
