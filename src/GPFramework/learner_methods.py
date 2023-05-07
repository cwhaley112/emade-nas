"""
Programmed by Jason Zutty
Modified by VIP Team
Implements a number of machine learning methods for use with deap
"""
import os
import pickle
import dill
import copy as cp
import numpy as np
import math
import sys
import time
import re
import gc
import traceback
from GPFramework.general_methods import mod_select, target_value_check, map_labels
from GPFramework.cache_methods import check_cache_read, check_cache_write
import scipy.sparse as sci
# Classifiers
try:
    from xgboost import XGBClassifier
except Exception as e:
    print("Unable to import xgboost. Error:", e)
try:
    from lightgbm import LGBMClassifier
except Exception as e:
    print("Unable to import lightgbm. Error:", e)
from sklearn.neighbors import KNeighborsClassifier, BallTree
from sklearn.linear_model import OrthogonalMatchingPursuit, LogisticRegression, SGDClassifier, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
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

# method specific to neural networks
def build_dnn_model():
    """Builds DNN model for keras classifier

    Any keras layers can be slotted into this framework

    Returns:
        compiled dense neural network model
    """
    model = Sequential()
    model.add(Dense(113, input_dim=113, kernel_initializer='normal', activation='relu'))
    # model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

'''
This method is not a primitive.
It is used as a helper method in learner
'''
def makeFeatureFromClass(data_pair, name=''):
    """Makes class labels into a feature

    Adds new feature to existing data by appending results from a machine
    learning algorithm

    Args:
        data_pair: given datapair
        name: name of labels

    Returns:
        Data pair with class feature appended
    """
    data_list = []
    # For debugging purposes let's print out method name
    print('makeFeatureFromClass') ; sys.stdout.flush()
    # Only need to perform this on testData, machine learning methods handle training data
    for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
        new_data_set = cp.deepcopy(data_set)
        instances = new_data_set.get_instances()

        for instance in instances:
            data = instance.get_features().get_data()
            target = cp.deepcopy(instance.get_target())
            if len(data) > 0:
                #detection data check: first check not needed as data.shape should be 2. if instance level data is 2d meaning it has more than one row it is detection data and target data is stacked alongside it
                if  len(data.shape)>1 and  data.shape[0] >1:  
                    target = target.reshape(-1,1)
                else:
                    target = target.reshape(1,-1) # for single class target and multiclass this works
                if sci.issparse(instance.get_features().get_data()):
                    data = sci.hstack([data,target])
                else:
                    data = np.hstack([data, target])

            instance.get_features().set_data(data)

        new_data_set.set_instances(instances)
        data_list.append(new_data_set)

    data_pair.set_train_data(data_list[0])
    data_pair.set_test_data(data_list[1])

    gc.collect(); return data_pair

'''
Model Initialization Methods
'''

def get_scikit_model(learner):
    """Generates a machine learning classifier

    Given a learner object, produce an estimator that can be used
    either by itself, or with an ensemble technique

    Args:
        learner: type of machine learning classifier to return

    Returns:
        a machine learning classifier
    """
    # This can occur if we specify a learner not implemented in scikit
    estimator = None
    param_grid = None
    if learner.name == "KNN":
        k = abs(int(learner.params['K']))
        weights_list = ['uniform', 'distance']
        weights = mod_select(weights_list, learner.params, 'weights')
        estimator = KNeighborsClassifier(n_neighbors=k,
                                         weights=weights)
        param_grid = {'n_neighbors':[2,3,4,5,6],
                      'weights':weights_list}
    elif learner.name == "SVM":
        C = abs(float(learner.params['C']))
        kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        kernel = mod_select(kernels, learner.params, 'kernel')
        estimator = SVC(C=C, kernel=kernel)
        param_grid = {'C':[0.1, 1.0, 10.0],
                      'kernel':kernels}
    elif learner.name == "BAYES":
        estimator = GaussianNB(priors=None)
        param_grid = {'priors':[None]}
    elif learner.name == "DNN":
        epochs = abs(int(learner.params['epochs']))
        batch_size = abs(int(learner.params['batch_size']))
        estimator = KerasClassifier(build_fn=build_dnn_model, epochs=epochs, batch_size=batch_size, verbose=0)
        param_grid = {'epochs':[50, 100],
                      'batch_size':[128, 256]}
    elif learner.name == "DECISION_TREE":
        criterions = ['gini', 'entropy']
        criterion = mod_select(criterions, learner.params, 'criterion')
        splitters = ['best', 'random']
        splitter = mod_select(splitters, learner.params, 'splitter')
        estimator = DecisionTreeClassifier(criterion=criterion, splitter=splitter)
        param_grid = {'criterion':criterions,
                      'splitter':splitters}
    elif learner.name == "RAND_FOREST":
        n_estimators = abs(int(learner.params['n_estimators']))
        criterions = ['gini', 'entropy']
        criterion = mod_select(criterions, learner.params, 'criterion')
        max_depth = abs(int(learner.params['max_depth']))
        class_weights = [None, 'balanced', 'balanced_subsample']
        class_weight = mod_select(class_weights, learner.params, 'class_weight')
        estimator = RandomForestClassifier(n_estimators=n_estimators,
                                           criterion=criterion,
                                           max_depth=max_depth,
                                           class_weight=class_weight)
        param_grid = {'n_estimators':[50, 100, 200],
                      'class_weight':class_weights,
                      'criterion':criterions}
    elif learner.name == "BOOSTING":
        learning_rate = abs(float(learner.params['learning_rate']))
        n_estimators = abs(int(learner.params['n_estimators']))
        max_depth = abs(int(learner.params['max_depth']))
        estimator = GradientBoostingClassifier(learning_rate=learning_rate,
                                            n_estimators=n_estimators,
                                            max_depth=max_depth)
        param_grid = {'learning_rate':[0.1, 1.0, 10.0],
                      'n_estimators':[50, 100, 200]}
    elif learner.name == "KMEANS":
        n_clusters = abs(int(learner.params['n_clusters']))
        estimator = KMeans(n_clusters=n_clusters)
        param_grid = {'n_clusters':[2, 3, 4, 5, 6]}
    elif learner.name == "BLUP":
        estimator = GaussianProcessClassifier(kernel=GPKernels.RBF())
        param_grid = {'kernel':[GPKernels.ConstantKernel(), GPKernels.DotProduct(), GPKernels.ExpSineSquared(),
                                GPKernels.Matern(), GPKernels.RBF(), GPKernels.RationalQuadratic(), GPKernels.WhiteKernel()]}
    elif learner.name == "OMP":
         estimator = OrthogonalMatchingPursuit()
         param_grid = {}
    elif learner.name == "GMM":
         n_components = abs(int(learner.params['n_components']))
         covariance_types = ['full', 'tied', 'diag', 'spherical']
         covariance_type = mod_select(covariance_types, learner.params, 'covariance_type')
         estimator = GaussianMixture(n_components=n_components, covariance_type=covariance_type)
         param_grid = {'n_components':[1, 2, 5, 10], 'covariance_type':covariance_types}
    elif learner.name == "LOGR":
        # Let's look at the integer parameter and mod it to one of two choices
        penalties = ['l2', 'l1']
        penalty = mod_select(penalties, learner.params, 'penalty')
        C = abs(float(learner.params['C']))
        estimator = LogisticRegression(penalty=penalty, C=C)
        param_grid = {'C':[0.1, 1.0, 10.0],
                      'penalty':penalties}
    elif learner.name == "SGD":
        penalties = ['l2', 'l1']
        penalty = mod_select(penalties, learner.params, 'penalty')
        alpha = abs(float(learner.params['alpha']))
        estimator = SGDClassifier(penalty=penalty, alpha=alpha, shuffle=False)
        param_grid = {'alpha':[0.0001, 0.001, 0.01],
                      'penalty':penalties}
    elif learner.name == "PASSIVE":
        C = abs(float(learner.params['C']))
        estimator = PassiveAggressiveClassifier(C=C, shuffle=False)
        param_grid = {'C':[0.1, 1.0, 10.0]}
    elif learner.name == "EXTRATREES":
        n_estimators = abs(int(learner.params['n_estimators']))
        max_depth = abs(int(learner.params['max_depth']))
        criterions = ['gini', 'entropy']
        criterion = mod_select(criterions, learner.params, 'criterion')
        estimator = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion)
        param_grid = {'n_estimators': [10,50,100,200], 'max_depth':[3,6,12], 'criterion':criterions}
    elif learner.name == "XGBOOST":
        # Dropped parameters - num_boost_round (cli arg), silent
        matched_keys = ['learning_rate', 'max_depth', 'n_estimators', 'reg_alpha', 'reg_lambda']
        params = {key: learner.params[key] for key in matched_keys if key in learner.params} # extract keys
        estimator = XGBClassifier(**params)
    elif learner.name == "LIGHTGBM":
        max_depth = abs(int(learner.params['max_depth']))
        learning_rate = abs(float(learner.params['learning_rate']))
        num_leaves = abs(int(learner.params['num_leaves']))
        boosting_types = ['gbdt', 'dart', 'goss', 'rf']
        boosting_type = mod_select(boosting_types, learner.params, 'boosting_type')
        estimator = LGBMClassifier(max_depth=max_depth, learning_rate=learning_rate,
                                   num_leaves=num_leaves, boosting_type=boosting_type)
    elif learner.name == "BOOSTING_REGRESSION":
        learning_rate = abs(float(learner.params['learning_rate']))
        n_estimators = abs(int(learner.params['n_estimators']))
        max_depth = abs(int(learner.params['max_depth']))
        estimator = GradientBoostingRegressor(learning_rate=learning_rate,
                                               n_estimators=n_estimators,
                                               max_depth=max_depth)
    elif learner.name == "ADABOOST_REGRESSION":
        learning_rate = abs(float(learner.params['learning_rate']))
        n_estimators = abs(int(learner.params['n_estimators']))
        estimator = AdaBoostRegressor(learning_rate=learning_rate, n_estimators=n_estimators)
    elif learner.name == "RANDFOREST_REGRESSION":
        n_estimators = abs(int(learner.params['n_estimators']))
        criterions = ['mae', 'mse']
        criterion = mod_select(criterions, learner.params, 'criterion')
        estimator = RandomForestRegressor(n_estimators=n_estimators,
                                           criterion=criterion)
    elif learner.name == "DECISIONTREE_REGRESSION":
        estimator = DecisionTreeRegressor()
    elif learner.name == "KNN_REGRESSION":
        k = abs(int(learner.params['K']))
        weights_list = ['uniform', 'distance']
        weights = mod_select(weights_list, learner.params, 'weights')
        estimator = KNeighborsRegressor(n_neighbors=k,
                                         weights=weights)
    elif learner.name == "SVM_REGRESSION":
        kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        kernel = mod_select(kernels, learner.params, 'kernel')
        estimator = SVR(kernel=kernel)
    elif learner.name == "ARGMAX":
        # Case handled in learner()
        pass
    elif learner.name == "ARGMIN":
        # Case handled in learner()
        pass
    elif learner.name == "DEPTH_ESTIMATE":
        # Case handled in learner()
        pass
    else:
        raise Exception("{} Is not an implemented learner type.".format(learner.name))

    return estimator, param_grid

'''
Learner Primitives
'''

def learner(data_pair, learner, ensemble):
    """
    Core method for all learners
    Creates the model
    Handles caching functions
    Loads and store the data and classification

    Args:
        data_pair: data structure storing train and test data
        learner:   data structure storing information needed to generate model
        ensemble:  data structure storing information needed to ensemble around the model

    Returns:
        updated data pair with new data
    """
    data_pair = cp.deepcopy(data_pair)

    try:
        """
        Cache [Load]
        """
        if data_pair.get_caching_mode():
            # Initialize overhead time start
            oh_time_start = time.time()

            # Initialize method row to None (Important for error handling)
            method_row = None

            # Create a TCP Connection with the db using MySQL
            database = data_pair.get_connection()

            # Calculate hash of current data in data_pair(s)
            previous_hash = data_pair.get_hash()
            
            # Calculate unique method string (unique method name + arguments of the method)
            l_params = re.sub("[,:]", "_", re.sub("[_ '}{]", "", str(learner.params)))
            e_params = re.sub("[,:]", "_", re.sub("[_ '}{]", "", str(ensemble.params)))
            method_string = learner.name + "_" + l_params + "_" + ensemble.name + "_" + e_params + "_"

            # Combine the unique method name + arguments of the method + hash of the previous data
            # To form a unique key of the method call
            method_key = method_string + previous_hash

            overhead_time, method_row, cache_row, hit = check_cache_read(data_pair, database, 
                                                                        method_key, oh_time_start, 
                                                                        target=True)
            if hit: return data_pair

            eval_time_start = time.time()

        """
        Setup Variables
        """
        #Get the underlying base estimator to use
        base_estimator = get_scikit_model(learner)[0]

        # learner setup
        if learner.name == "ARGMAX" or learner.name == "ARGMIN" or learner.name == "DEPTH_ESTIMATE":
            sampling_rate = learner.params["sampling_rate"]
            if sampling_rate <= 0:
                sampling_rate = 1
        # ensemble setup
        elif ensemble.name == "BAGGED":
            base_estimator = BaggingClassifier(base_estimator=base_estimator, random_state=101)
        elif ensemble.name == "ADABOOST":
            # Avoiding 0 estimators error by setting estimators to default
            if ensemble.params["n_estimators"] <= 0:
                ensemble.params["n_estimators"] = 50

            if learner.name == "SVM" or learner.name == "KMEANS" or learner.name == "BLUP" or learner.name == "OMP":
                base_estimator = AdaBoostClassifier(base_estimator=base_estimator,
                                                    n_estimators=ensemble.params["n_estimators"],
                                                    learning_rate=ensemble.params["learning_rate"],
                                                    algorithm='SAMME', random_state=101)
            elif learner.name == "DNN" or learner.name == "KNN":
                raise ValueError('Cannot apply adaboost to this classifier.')
            else:
                base_estimator = AdaBoostClassifier(base_estimator=base_estimator,
                                                    n_estimators=ensemble.params["n_estimators"],
                                                    learning_rate=ensemble.params["learning_rate"],
                                                    random_state=101)
        elif ensemble.name == "GRID":
            parameters = get_scikit_model(learner)[1]
            if parameters is None:
                raise ValueError('Cannot apply grid search to this classifier.')
            base_estimator = GridSearchCV(base_estimator, parameters)

        """
        Main Block
        """
        if learner.name == "ARGMAX":
            # target_values = np.array([inst.get_target() for
            #                         inst in data_pair.get_train_data().get_instances()])
            if data_pair.get_datatype() == 'detectiondata':
                target_values = data_pair.get_train_data().get_flattenned_target()
            else:
                target_values = data_pair.get_train_data().get_target()
                target_value_check(target_values)
            predicted_classes = []
            for i, dataset in enumerate([data_pair.get_train_data(), data_pair.get_test_data()]):
                numpy_data = dataset.get_numpy()

                maxes = [np.argmax(row) / sampling_rate for row in numpy_data]
                predicted_classes += maxes

                dataset.set_target(maxes)
                # [inst.set_target(np.array([target])) for inst, target in
                #  zip(dataset.get_instances(), maxes)]

            data_pair = makeFeatureFromClass(data_pair, name=learner.name)
            # Restore the training data
            # [inst.set_target(np.array([target])) for inst, target in
            #  zip(data_pair.get_train_data().get_instances(), target_values)]
            data_pair.get_train_data().set_target(target_values)
            #data_pair = map_labels(data_pair, target_values, 0)
        elif learner.name == "ARGMIN":
            # target_values = np.array([inst.get_target() for
            #                         inst in data_pair.get_train_data().get_instances()])
            if data_pair.get_datatype() == 'detectiondata':
                target_values = data_pair.get_train_data().get_flattenned_target()
            else:
                target_values = data_pair.get_train_data().get_target()
                target_value_check(target_values)
            predicted_classes = []
            for i, dataset in enumerate([data_pair.get_train_data(), data_pair.get_test_data()]):
                numpy_data = dataset.get_numpy()

                mins = [np.argmin(row) / sampling_rate for row in numpy_data]
                predicted_classes += mins

                dataset.set_target(mins)
                # [inst.set_target(np.array([target])) for inst, target in
                #  zip(dataset.get_instances(), mins)]

            data_pair = makeFeatureFromClass(data_pair, name=learner.name)
            # Restore the training data
            # data_pair.get_train_data().set_target(target_values)
            data_pair = map_labels(data_pair, target_values, 0)
        elif learner.name == "DEPTH_ESTIMATE":
            off_nadir_angle = learner.params["off_nadir_angle"]
            if off_nadir_angle <= 0:
                off_nadir_angle = 20.0
            off_nadir_angle = math.pi / 180.0 * off_nadir_angle
            # target_values = np.array([inst.get_target() for
            #                         inst in data_pair.get_train_data().get_instances()])
            if data_pair.get_datatype() == 'detectiondata':
                target_values = data_pair.get_train_data().get_flattenned_target()
            else:
                target_values = data_pair.get_train_data().get_target()
                target_value_check(target_values)
            # Check for multiple target values
            for i, dataset in enumerate([data_pair.get_train_data(), data_pair.get_test_data()]):
                numpy_data = dataset.get_numpy()

                predicted_classes = []
                for row, num in zip(numpy_data, range(numpy_data.shape[0])):
                    try:
                        if len(row) == 1:
                            predicted_classes.append(-1.0)
                        else:
                            opl_estimate = (row[-1] - row[0]) / sampling_rate
                            refracted_angle = math.asin(math.sin(off_nadir_angle) / 1.33)
                            depth_estimate = opl_estimate * math.cos(refracted_angle)
                            predicted_classes.append(depth_estimate)
                    except IndexError as no_bottom_error:
                        predicted_classes.append(-1.0)
                # [inst.set_target(np.array([target])) for inst, target in
                #  zip(dataset.get_instances(), predicted_classes)]
                dataset.set_target(predicted_classes)
                #data_pair = map_labels(data_pair,predicted_classes, i)


            data_pair = makeFeatureFromClass(data_pair, name=learner.name)
            # Restore the training data
            # [inst.set_target(np.array([target])) for inst, target in
            # zip(data_pair.get_train_data().get_instances(), target_values)]
            data_pair.get_train_data().set_target(target_values)
            #data_pair = map_labels(data_pair, target_values, 0)
        else:
            """
            Load data
            Validate data
            """
            training_object = data_pair.get_train_data()
            testing_object = data_pair.get_test_data()
            training_data = training_object.get_numpy()
            testing_data = testing_object.get_numpy()
            if data_pair.get_datatype() == 'detectiondata':
                target_values = training_object.get_flattenned_target()
            else:
                target_values = training_object.get_target()
                target_value_check(target_values)
            #print('target_values', target_values, 'numpy values', training_data, training_data.shape)
            # Check for multiple target values
            
            # For debugging purposes let's print out name of the function and dimensions of the data
            print(learner.name + ': training', training_data.shape)
            sys.stdout.flush()

            print(learner.name + ': labels', target_values.shape)
            sys.stdout.flush()



            """
            Fit estimator to training data
            Predict labels of testing data
            """
            base_estimator.random_state = 101
            base_estimator.fit(training_data, target_values)
            print(learner.name + ': testing', testing_data.shape)
            sys.stdout.flush()
            predicted_classes = base_estimator.predict(testing_data)
            # data_pair.get_train_data().get_target()
            # data_pair.get_test_data().get_target()
            """
            Map predicted labels back to instances
            """
            data_pair.get_test_data().set_target(predicted_classes)
            """
            Add new feature to data based on predicted labels
            """
            # Let's make the predictions a feature through use of the make feature from class,
            # But then restore the training data to the class
            # Set the self-predictions of the training data
            trained_classes = base_estimator.predict(training_data)
            # data_pair = map_labels(data_pair, trained_classes, 0)
            data_pair.get_train_data().set_target(trained_classes)

            #print(data_pair.get_train_data().get_target(), data_pair.get_test_data().get_target())

            data_pair = makeFeatureFromClass(data_pair, name=learner.name)
            # Restore the training data
            # data_pair = map_labels(data_pair, target_values, 0)
            data_pair.get_train_data().set_target(target_values)

        """
        Cache [Store]
        """
        if data_pair.get_caching_mode():
            # The time from this evaluation isolated
            eval_time = time.time() - eval_time_start

            # Checks if method should be written to cache and updates data_pair
            check_cache_write(data_pair, database, 
                                method_row, cache_row, 
                                method_key, 
                                overhead_time, eval_time,
                                target=True)

    except Exception as e:
        """
        Handle Errors
        """
        if data_pair.get_caching_mode():
            if "pr0c33d" in str(e):
                # Exception was already handled properly
                gc.collect()
                return data_pair

            # Saving a boolean so the string parsing is only done once
            lock_timeout = "lock timeout" in str(e)

            # Method row will already be updated if lock timeout occurs on cache row
            if method_row is not None and not lock_timeout:
                database.update_overhead_time(method_row, overhead_time)
                database.set_error(method_row,
                                   str(e) + traceback.format_exc(),
                                   time.time() - eval_time_start)

            try:
                database.commit()
            except Exception as f:
                print("Database commit failed with Error:", f)
            try:
                database.close()
                del database
            except Exception as f:
                print("Database close failed with Error:", f)
            
        gc.collect()
        raise

    if data_pair.get_caching_mode():
        del database
    gc.collect()
    return data_pair
