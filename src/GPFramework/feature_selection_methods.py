"""
Programmed by Jason Zutty
Modified by VIP Team
Implements a number of feature selection methods for use with deap
"""
import sklearn.feature_selection
import numpy as np
from functools import partial
from GPFramework.data import EmadeDataPair
from GPFramework.constants import TriState
from GPFramework.wrapper_methods import RegistryWrapperFT

SCORING_FUNCTION_LIST = [
    sklearn.feature_selection.f_classif,
    sklearn.feature_selection.chi2,
    sklearn.feature_selection.f_regression
    ]

# Wrapper class to control export namespace
fsw = RegistryWrapperFT([EmadeDataPair, TriState])

def feature_selection_helper(train_data, test_data, target, function):
    new_train_data = function.fit_transform(train_data, target)
    new_test_data = function.transform(test_data)
    return new_train_data, new_test_data

def select_k_best_scikit_setup(scoring_function=0, k=10):
    # force k to be positive
    k = abs(k)
    # Get scoring function
    scoring_function = SCORING_FUNCTION_LIST[scoring_function%len(SCORING_FUNCTION_LIST)]
    # Create a selector for scikit's select k best
    return sklearn.feature_selection.SelectKBest(scoring_function, k)

select_k_best_scikit = fsw.register("mySelKBest", "test_select_k_best", feature_selection_helper, select_k_best_scikit_setup, [int, int])
select_k_best_scikit.__doc__ = """
Returns the result of a select k best method using scikit

http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html

Args:
    scoring_function: function taking two arrays X and y, and returning a
                      pair of arrays (scores, pvalues) or a single array of scores
    k: number of top features to select

Returns:
    feature selection method
"""

def select_percentile_scikit_setup(scoring_function=0, percentile=10):
    # Correct negative percentiles
    percentile = abs(percentile)
    # Get scoring function
    scoring_function = SCORING_FUNCTION_LIST[scoring_function%len(SCORING_FUNCTION_LIST)]
    # Create a selector for scikit's select percentile
    return sklearn.feature_selection.SelectPercentile(scoring_function, percentile)

select_percentile_scikit = fsw.register("mySelPercentile", "test_select_percentile", feature_selection_helper, select_percentile_scikit_setup, [int, int])
select_percentile_scikit.__doc__ = """
Returns the result of a select percentile method using scikit

http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectPercentile.html

Args:
    scoring_function: function taking two arrays X and y, and returning a
                      pair of arrays (scores, pvalues) or a single array of scores
    percentile: percent of features to keep

Returns:
    feature selection method
"""

def select_fpr_scikit_setup(scoring_function=0, alpha=0.05):
    # Get scoring function
    scoring_function = SCORING_FUNCTION_LIST[scoring_function%len(SCORING_FUNCTION_LIST)]
    # Create a selector for scikit's select fpr
    return sklearn.feature_selection.SelectFpr(scoring_function, alpha)

select_fpr_scikit = fsw.register("mySelFpr", "test_select_fpr", feature_selection_helper, select_fpr_scikit_setup, [int, float])
select_fpr_scikit.__doc__ = """
Returns the result of a select fpr method using scikit

http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFpr.html

Args:
    scoring_function: function taking two arrays X and y, and returning a
                      pair of arrays (scores, pvalues) or a single array of scores
    alpha: the highest p-value for features to be kept

Returns:
    feature selection method
"""

def select_fdr_scikit_setup(scoring_function=0, alpha=0.05):
    # Get scoring function
    scoring_function = SCORING_FUNCTION_LIST[scoring_function%len(SCORING_FUNCTION_LIST)]
    # Create a selector for scikit's select fdr
    return sklearn.feature_selection.SelectFdr(scoring_function, alpha)

select_fdr_scikit = fsw.register("mySelFdr", "test_select_fdr", feature_selection_helper, select_fdr_scikit_setup, [int, float])
select_fdr_scikit.__doc__ = """
Returns the result of a select fdr method using scikit

http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFdr.html

Args:
    scoring_function: function taking two arrays X and y, and returning a
                      pair of arrays (scores, pvalues) or a single array of scores
    alpha: the highest uncorrected p-value for features to keep

Returns:
    feature selection method
"""

def select_generic_univariate_scikit_setup(scoring_function=0, mode=0, param=1e-05):
    # Get scoring function
    scoring_function = SCORING_FUNCTION_LIST[scoring_function%len(SCORING_FUNCTION_LIST)]
    # Get the mode
    MODE_LIST = ['percentile', 'k_best', 'fpr', 'fdr', 'fwe']
    mode = MODE_LIST[mode%len(MODE_LIST)]
    # Create a selector for scikit's select generic univariate
    return sklearn.feature_selection.GenericUnivariateSelect(scoring_function, mode, param)

select_generic_univariate_scikit = fsw.register("mySelGenUni", "test_select_generic_univariate", feature_selection_helper, select_generic_univariate_scikit_setup, [int, int, float])
select_generic_univariate_scikit.__doc__ = """
Returns the result of a select generic univariate method using scikit

http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.GenericUnivariateSelect.html

Args:
    scoring_function: function taking two arrays X and y, and returning a
                      pair of arrays (scores, pvalues) or a single array of scores
    mode: feature selection mode
    param: parameter of the corresponding mode

Returns:
    feature selection method
"""

def select_fwe_scikit_setup(scoring_function=0, alpha=0.05):
    # Get scoring function
    scoring_function = SCORING_FUNCTION_LIST[scoring_function%len(SCORING_FUNCTION_LIST)]
    # Create a selector for scikit's select fwe
    return sklearn.feature_selection.SelectFwe(scoring_function, alpha)

select_fwe_scikit = fsw.register("mySelFwe", "test_select_fwe", feature_selection_helper, select_fwe_scikit_setup, [int, float])
select_fwe_scikit.__doc__ = """
Returns the result of a select fwe method using scikit

http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFwe.html

Args:
    scoring_function: function taking two arrays X and y, and returning a
                      pair of arrays (scores, pvalues) or a single array of scores
    alpha: the highest uncorrected p-value for features to keep

Returns:
    feature selection method
"""

def variance_threshold_helper(train_data, test_data, target, function):
    new_train_data = function.fit_transform(train_data)
    new_test_data = function.transform(test_data)
    return new_train_data, new_test_data

def variance_threshold_scikit_setup(threshold=0.0):
    # Make sure the threshold is positive
    threshold = np.abs(threshold)
    # Create a selector for scikit's variance threshold
    return sklearn.feature_selection.VarianceThreshold(threshold)

variance_threshold_scikit = fsw.register("myVarThresh", "test_variance_threshold", variance_threshold_helper, variance_threshold_scikit_setup, [float])
variance_threshold_scikit.__doc__ = """
Returns the result of a variance threshold method using scikit

http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html

Args:
    threshold: features with a training-set variance lower than this
               threshold will be removed. The default is to keep all
               features with non-zero variance. --sklearn

Returns:
    feature selection method
"""
