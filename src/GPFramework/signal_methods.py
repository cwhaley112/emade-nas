"""
Programmed by Jason Zutty
Modified by VIP Team
Implements a number of signal processing methods for use with deap
"""
import numpy as np
import random
import time
from scipy.optimize import curve_fit, nnls
from scipy.stats import multivariate_normal as mvn
from scipy.ndimage import filters
from lmfit import Model, Parameters, minimize
from lmfit.models import GaussianModel
import copy as cp
import scipy.signal
import scipy.fftpack
import sklearn.gaussian_process
import sklearn.cluster
from skimage import restoration
import hmmlearn.hmm
import pywt
import gc
import sys
from scipy import interpolate
import pdb
from PIL import Image
import traceback
import os

from GPFramework.constants import TriState, Axis, TRI_STATE
from GPFramework.wrapper_methods import RegistryWrapperS, RegistryWrapperB
from GPFramework.data import EmadeDataPair, EmadeData
from GPFramework.learner_methods import makeFeatureFromClass
from GPFramework.cache_methods import check_cache_read, check_cache_write, hash_string

smw = RegistryWrapperS([EmadeDataPair, TriState, Axis])
smw_2 = RegistryWrapperS(2*[EmadeDataPair] + 2*[TriState] + 2*[Axis])
smw_3 = RegistryWrapperS(3*[EmadeDataPair] + 3*[TriState] + 3*[Axis])
smwb = RegistryWrapperB()

def copy_stream_to_target(data_pair):
    """Copies the data in the stream objects to the target value
       Note: Always loads stream data

    Args:
        data_pair: given datapair

    Returns:
        Data Pair
    """
    # For debugging purposes let's print out method name
    print("CopyStreamToTarget") ; sys.stdout.flush()

    data_pair = cp.deepcopy(data_pair)

    try:
        """
        Cache [Load] Block
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
            method_string = "CopyStreamToTarget" + "_"

            # Combine the unique method name + arguments of the method + hash of the previous data
            # To form a unique key of the method call
            method_key = method_string + previous_hash

            overhead_time, method_row, cache_row, hit = check_cache_read(data_pair, database, 
                                                                        method_key, oh_time_start, 
                                                                        target=True)
            if hit: return data_pair

            eval_time_start = time.time()

        # Initialize where the data will be temporarily stored
        data_list = []
        # Iterate through train data then test data
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            # Copy the dataSet so as not to destroy original data
            instances = cp.deepcopy(data_set.get_instances())
            # Iterate over all points in the dataSet
            for instance in instances:
                # Capture the data
                data = instance.get_stream().get_data()
                # Set the data over to the taget
                instance.set_target(cp.deepcopy(data))

            data_set.set_instances(instances)
            data_list.append(data_set)

        """
        Update data pair with new data
        """
        data_pair.set_train_data(data_list[0])
        data_pair.set_test_data(data_list[1])

        """
        Cache [Store] Block
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

smwb.register("CopyStreamToTarget", "test_copy_stream_to_target", copy_stream_to_target, None, [EmadeDataPair])

def remove_feature_helper(data, feat_num=0):
    return np.delete(data, feat_num, 1)

remove_feature = smw.register("RemoveFeature", "test_remove_feature", remove_feature_helper, None, [int], TRI_STATE)
remove_feature.__doc__ = """
Removes a feature from the data

Args:
    data: given numpy array
    feat_num: index of the feature to remove

Returns:
    Transformed data
"""

def select_range_helper(data, feat_start=0, feat_stop=1):
    return data[:, feat_start:feat_stop]

select_range = smw.register("SelectRange", "test_select_range", select_range_helper, None, [int, int], TRI_STATE)
select_range.__doc__ = """
Selects a range of data starting from feature feat_start and
stopping before feature feat_stop

Args:
    data: given numpy array
    feat_start: id of feature to start at
    feat_stop: id of feature to stop at

Returns:
    Transformed data
"""

def my_concatenate_helper(data, second_data):
    return np.hstack((data, second_data))

my_concatenate = smw_2.register("myConcatenate", "test_my_concatenate", my_concatenate_helper, None, [], [TriState.FEATURES_TO_FEATURES, TriState.STREAM_TO_STREAM])
my_concatenate.__doc__ = """
Concatenate twos data arrays together

Args:
    data: given numpy array
    second_data: given numpy array

Returns:
    Transformed data
"""

def my_if_then_else(condition_pair, data_pair_1, data_pair_2, mode1, mode2, mode3, column):
    """Consumes three data_pairs. Based upon the truth of the target feature in
    condition_pair, choose correct rows from data_pair_1 and data_pair_2.

    Args:
        condition_pair: given condition pair
        data_pair_1: given Data Pair
        data_pair_2: given Data Pair
        mode1: mode to load and save condition_pair in
        mode2: mode to load and save data_pair_1 in
        mode3: mode to load and save data_pair_2 in
        column: column number

    Returns:
        Data Pair with added if then else feature
    """
    # For debugging purposes let's print out method name
    print("myIfThenElse") ; sys.stdout.flush()

    condition_pair = cp.deepcopy(condition_pair)
    data_pair_1 = cp.deepcopy(data_pair_1)
    data_pair_2 = cp.deepcopy(data_pair_2)

    try:
        # Convert column to an int
        column = int(column)

        """
        Cache [Load]
        """
        if condition_pair.get_caching_mode():
            # Initialize overhead time start
            oh_time_start = time.time()

            # Initialize method row to None (Important for error handling)
            method_row = None

            # Create a TCP Connection with the db using MySQL
            database = condition_pair.get_connection()

            # Calculate hash of current data in data_pair(s)
            previous_hash = hash_string(condition_pair.get_hash() + \
                                        data_pair_1.get_hash() + \
                                        data_pair_2.get_hash())
            
            # Calculate unique method string (unique method name + arguments of the method)
            method_string = "myIfThenElse" + "_" + str(mode1) + "_" + str(mode2) + "_" + str(mode3) + "_" + str(column) + "_"

            # Combine the unique method name + arguments of the method + hash of the previous data
            # To form a unique key of the method call
            method_key = method_string + previous_hash

            overhead_time, method_row, cache_row, hit = check_cache_read(condition_pair, database, 
                                                                            method_key, oh_time_start)
            if hit: return condition_pair

            eval_time_start = time.time()

        data_list = []
        for condition_data_set, data_pair_1_set, data_pair_2_set in \
                zip([condition_pair.get_train_data(), condition_pair.get_test_data()],
                    [data_pair_1.get_train_data(), data_pair_1.get_test_data()],
                    [data_pair_2.get_train_data(), data_pair_2.get_test_data()]):

            cond_instances = cp.deepcopy(condition_data_set.get_instances())
            dp1_instances = cp.deepcopy(data_pair_1_set.get_instances())
            dp2_instances = cp.deepcopy(data_pair_2_set.get_instances())

            for cond_instance, dp1_instance, dp2_instance in zip(cond_instances, dp1_instances, dp2_instances):
                if mode1 is TriState.FEATURES_TO_FEATURES:
                    data = cond_instance.get_features().get_data()
                elif mode1 is TriState.STREAM_TO_STREAM or TriState.STREAM_TO_FEATURES:
                    data = cond_instance.get_stream().get_data()
                if mode2 is TriState.FEATURES_TO_FEATURES:
                    data_1 = dp1_instance.get_features().get_data()
                elif mode2 is TriState.STREAM_TO_STREAM or TriState.STREAM_TO_FEATURES:
                    data_1 = dp1_instance.get_stream().get_data()
                if mode3 is TriState.FEATURES_TO_FEATURES:
                    data_2 = dp2_instance.get_features().get_data()
                elif mode3 is TriState.STREAM_TO_STREAM or TriState.STREAM_TO_FEATURES:
                    data_2 = dp2_instance.get_stream().get_data()

                if data[0,column]:
                    new_data = data_1
                else:
                    new_data = data_2

                if mode1 is TriState.FEATURES_TO_FEATURES:
                    cond_instance.get_features().set_data(new_data)
                elif mode1 is TriState.STREAM_TO_STREAM:
                    cond_instance.get_stream().set_data(new_data)
                elif mode1 is TriState.STREAM_TO_FEATURES:
                    old_features = cond_instance.get_features().get_data()
                    new_features = np.concatenate((old_features, new_data), axis=None) # auto-flattening
                    cond_instance.get_features().set_data(np.reshape(new_features, (1,-1)))

            condition_data_set.set_instances(cond_instances)
            data_list.append(condition_data_set)

        """
        Update data pair with new data
        """
        condition_pair.set_train_data(data_list[0])
        condition_pair.set_test_data(data_list[1])

        """
        Cache [Store] Block
        """
        if condition_pair.get_caching_mode():
            # The time from this evaluation isolated
            eval_time = time.time() - eval_time_start

            check_cache_write(condition_pair, database, 
                              method_row, cache_row, 
                              method_key, 
                              overhead_time, eval_time)

    except Exception as e:
        """
        Handle Errors
        """
        if condition_pair.get_caching_mode():
            if "pr0c33d" in str(e):
                # Exception was already handled properly
                gc.collect()
                return condition_pair

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

    if condition_pair.get_caching_mode():
        del database
    gc.collect()
    return condition_pair

smwb.register("myIfThenElse", "test_my_if_then_else", my_if_then_else, None, [EmadeDataPair, EmadeDataPair, EmadeDataPair, TriState, TriState, TriState, int])

def convolve_func(data, kern=None):
    if data.ndim == 1: # 1D input, 1D output
        return np.convolve(data, kern, mode='same')
    else:
        return np.array([np.convolve(sample, kern, mode='same') for sample in data])

def average_setup(data, window=3):
    if window <= 0:
        window = 3
    kern = np.ones(window)
    kern = kern / float((window))
    return data, { "kern": kern }

my_averager = smw.register("MyAverager", "test_my_averager", convolve_func, average_setup, [int], [TriState.FEATURES_TO_FEATURES, TriState.STREAM_TO_STREAM])
my_averager.__doc__ = """
Runs an averager of length window (Low Pass Filter) on data

Args:
    data: numpy array of example
    window: length of window

Returns:
    Data Pair with averager feature added

"""

def diff_setup(data):
    return data, { "kern": np.array([1., -1.]) }

my_diff = smw.register("MyDiff", "test_my_diff", convolve_func, diff_setup, [], [TriState.FEATURES_TO_FEATURES, TriState.STREAM_TO_STREAM])
my_diff.__doc__ = """
Runs a diff (High Pass Filter) on data

Args:
    data: numpy array of example

Returns:
    Data with diff feature added
"""

def my_corr(data, R=None):
    return np.real(R)

def auto_corr_setup(data):
    F = np.fft.fft(data)
    S = np.conjugate(F)*F
    return data, { "R": np.fft.ifft(S) }

my_auto_corr = smw.register("MyAutoCorr", "test_my_auto_corr", my_corr, auto_corr_setup, [], [TriState.FEATURES_TO_FEATURES, TriState.STREAM_TO_STREAM])
my_auto_corr.__doc__ = """
Computes the autocorrelation using the Wiener-Khinchin Theorem

Args:
    data: numpy array of example

Returns:
    Data Pair with autocorrelation feature added
"""

def my_cross_corr_helper(data, second_data):
    F = np.fft.fft(data)
    G = np.fft.fft(second_data)
    S = np.conjugate(F)*G
    R = np.fft.ifft(S)
    return np.real(R)

my_cross_corr = smw_2.register("CrossCorrelation", "test_my_cross_corr", my_cross_corr_helper, None, [], TRI_STATE)
my_cross_corr.__doc__ = """
Computes the cross correlation of two signals

Args:
    data: given numpy array
    second_data: given numpy array

Returns:
    Transformed data
"""

def dct(data, transform=2, norm=1):
    # There are 3 DCT modes: 1, 2, and 3, so let's mod by 3 and add 1
    transform = transform%3 + 1
    # There are two norms ortho or none, so mod them
    norms = [None, 'ortho']
    norm = norms[norm%len(norms)]

    return scipy.fftpack.dct(data, type=transform, norm=norm)

my_dct = smw.register("MyDCT", "test_dct", dct, None, [int, int], TRI_STATE)
my_dct.__doc__ = """
Performs the discrete cosine transform of a signal x
transform can be 1, 2, or 3. norm can be None or ortho.

http://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.fftpack.dct.html

Args:
    data: numpy array of example
    transform: type of dct
    norm: normalization mode

Returns:
    Data with dct feature added
"""

def my_window(data, w=None):
    return data * w

def w_hann_setup(data, lag=True):
    N = data.shape[1]
    nVec = np.arange(N)
    if lag:
        w = 0.5*(1 - np.cos( (2*np.pi*nVec)/(N - 1) ) )
    else:
        w = 0.5*(1 + np.cos( (2*np.pi*nVec)/(N - 1) ) )
    return data, { "w": w }

window_hann = smw.register("WindowHann", "test_window_hann", my_window, w_hann_setup, [bool], [TriState.FEATURES_TO_FEATURES, TriState.STREAM_TO_STREAM])
window_hann.__doc__ = """
Performs a Hann (Hanning) window on a signal with or without "lag"

Args:
    data: numpy array of example
    lag: whether to include lag

Returns:
    Data with Hann window feature added
"""

def w_hamming_setup(data, lag=True):
    N = data.shape[1]
    nVec = np.arange(N)
    alpha = 0.53836
    beta = 0.46164
    if lag:
        w = alpha - beta*np.cos( (2*np.pi*nVec)/(N-1) )
    else:
        w = alpha + beta*np.cos( (2*np.pi*nVec)/(N-1) )
    return data, { "w": w }

window_hamming = smw.register("WindowHamming", "test_window_hamming", my_window, w_hamming_setup, [bool], [TriState.FEATURES_TO_FEATURES, TriState.STREAM_TO_STREAM])
window_hamming.__doc__ = """
Perfroms a Hamming window on a signal with or without "lag"

Args:
    data: numpy array of example
    lag: whether to include lag

Returns:
    Data with Hamming window feature added
"""

def w_turkey_setup(data, alpha=0.5):
    alpha = float(alpha)
    N = data.shape[1]
    w = np.zeros(N)
    for i in np.arange(N):
        if i < alpha*(N-1)/2:
            w[i] = 0.5*(1 + np.cos( np.pi * ( (2*i)/(alpha*(N-1)) - 1 ) ) )
        elif i <= (N-1)*(1-alpha/2):
            w[i] = 1
        else:
            w[i] = 0.5*(1 + np.cos( np.pi * ( (2*i)/(alpha*(N-1)) - 2/alpha + 1 ) ) )
    return data, { "w": w }

window_turkey = smw.register("WindowTurkey", "test_window_turkey", my_window, w_turkey_setup, [float], [TriState.FEATURES_TO_FEATURES, TriState.STREAM_TO_STREAM])
window_turkey.__doc__ = """
Performs a Tukey window on a signal using a parameter alpha

An alpha of 0 is a rectangular window
An alpha of 1 is a Hann window
Alpha should be between 0 and 1

Args:
    data: numpy array of example
    alpha: type of window

Returns:
    Data with Tukey window feature added
"""

def w_cosine_setup(data):
    N = data.shape[1]
    nVec = np.arange(N)
    w = np.sin( np.pi*nVec / (N-1) )
    return data, { "w": w }

window_cosine = smw.register("WindowCosine", "test_window_cosine", my_window, w_cosine_setup, [], [TriState.FEATURES_TO_FEATURES, TriState.STREAM_TO_STREAM])
window_cosine.__doc__ = """
Performs a Cosine (Sine) window on a signal

Args:
    data: numpy array of example

Returns:
    Data with Cosine window feature added
"""

def w_lanczos_setup(data):
    N = data.shape[1]
    nVec = np.arange(N)
    w = np.sinc( 2.0*nVec/(N-1) - 1 )
    return data, { "w": w }

window_lanczos = smw.register("WindowLanczos", "test_window_lanczos", my_window, w_lanczos_setup, [], [TriState.FEATURES_TO_FEATURES, TriState.STREAM_TO_STREAM])
window_lanczos.__doc__ = """
Performs a Lanczos window on a signal

Args:
    data: numpy array of example

Returns:
    Data with Lanczos window feature added
"""

def w_triangular_setup(data):
    N = data.shape[1]
    nVec = np.arange(N)
    w = 2.0/(N+1)*( (N+1)/2.0 - np.abs( nVec - (N-1)/2.0 ) )
    return data, { "w": w }

window_triangular = smw.register("WindowTriangular", "test_window_triangular", my_window, w_triangular_setup, [], [TriState.FEATURES_TO_FEATURES, TriState.STREAM_TO_STREAM])
window_triangular.__doc__ = """
Performs a triangular window on a signal

Args:
    data: numpy array of example

Returns:
    Data with triangular window feature added
"""

def w_bartlett_setup(data):
    N = data.shape[1]
    nVec = np.arange(N)
    w = 2.0/(N-1)*( (N-1)/2.0 - np.abs(nVec - (N-1)/2.0 ) )
    return data, { "w": w }

window_bartlett = smw.register("WindowBartlett", "test_window_bartlett", my_window, w_bartlett_setup, [], [TriState.FEATURES_TO_FEATURES, TriState.STREAM_TO_STREAM])
window_bartlett.__doc__ = """
Perfoms the Bartlet window on a signal

Args:
    data: numpy array of example

Returns:
    Data with Bartlet window feature added
"""

def w_gaussian_setup(data, sigma=1.0):
    sigma = float(sigma)
    N = data.shape[1]
    nVec = np.arange(N)
    w = np.exp( -0.5*( (nVec - (N-1)/2.0 )/( sigma*(N-1)/2.0 ) )**2 )
    return data, { "w": w }

window_gaussian = smw.register("WindowGaussian", "test_window_gaussian", my_window, w_gaussian_setup, [float], [TriState.FEATURES_TO_FEATURES, TriState.STREAM_TO_STREAM])
window_gaussian.__doc__ = """
Perfoms the Gaussian window on a signal

Args:
    data: numpy array of example
    sigma: sigma value used by gaussian

Returns:
    Data with Gaussian window feature added
"""

def w_bartlett_hann_setup(data):
    a0 = 0.62
    a1 = 0.48
    a2 = 0.38
    N = data.shape[1]
    nVec = np.arange(N)
    w = a0 - a1 * np.abs( nVec/(N-1) - 0.5 ) - a2 * np.cos( (2*np.pi*nVec)/(N-1) )
    return data, { "w": w }

window_bartlett_hann = smw.register("WindowBartlettHann", "test_window_bartlett_hann", my_window, w_bartlett_hann_setup, [], [TriState.FEATURES_TO_FEATURES, TriState.STREAM_TO_STREAM])
window_bartlett_hann.__doc__ = """
Perfoms the Bartlett-Hann window on a signal

Args:
    data: numpy array of example

Returns:
    Data with Bartlett-Hann window feature added
"""

def w_blackman_setup(data, alpha=0.16):
    alpha = float(alpha)
    a0 = (1-alpha)/2.0
    a1 = 0.5
    a2 = alpha/2.0
    N = data.shape[1]
    nVec = np.arange(N)
    w = a0 - a1 * np.cos( (2*np.pi*nVec)/( N-1 ) ) + a2 * np.cos( (4*np.pi*nVec)/( N-1 ) )
    return data, { "w": w }

window_blackman = smw.register("WindowBlackman", "test_window_blackman", my_window, w_blackman_setup, [float], [TriState.FEATURES_TO_FEATURES, TriState.STREAM_TO_STREAM])
window_blackman.__doc__ = """
Perfoms the Blackman window on a signal

Args:
    data: numpy array of example
    alpha: equation parameter

Returns:
    Data with Blackman window feature added
"""

def w_kaiser_setup(data, alpha=3.0, lag=True):
    # Not sure on alpha range on this one
    if alpha < 0:
        alpha = 0
    alpha = float(alpha)

    N = data.shape[1]
    nVec = np.arange(N)
    if lag:
        w = np.i0(np.pi*alpha*np.sqrt(1 - ( 2.0*nVec/(N-1) - 1)**2) ) / np.i0(np.pi*alpha)
    else:
        w = np.i0(np.pi*alpha*np.sqrt(1 - ( 2.0*nVec/(N-1) )**2) ) / np.i0(np.pi*alpha)
    return data, { "w": w }

window_kaiser = smw.register("WindowKaiser", "test_window_kaiser", my_window, w_kaiser_setup, [float, bool], [TriState.FEATURES_TO_FEATURES, TriState.STREAM_TO_STREAM])
window_kaiser.__doc__ = """
Performs a Kaiser window on a signal with or without lag,
given a parameter alpha. Alpha is usually 3

Args:
    data: numpy array of example
    alpha: equation parameter
    lag: whether to include lag

Returns:
    Data with Kaiser window feature added
"""

def w_planck_taper_setup(data, epsilon=0.1):
    # Not sure if this needs to be bounded either
    if epsilon < 0:
        epsilon = 0

    N = data.shape[1]
    nVec = np.arange(N)
    Zplus = 2.0*epsilon*(1.0/(1+2.0*nVec/(N-1.0)) + 1.0/(1.0-2.0*epsilon+2.0*nVec/(N-1.0)))
    Zminus = 2.0*epsilon*(1.0/(1-2.0*nVec/(N-1.0)) + 1.0/(1.0-2.0*epsilon-2.0*nVec/(N-1.0)))
    w = np.zeros(N)
    for i in np.arange(N):
        if i < epsilon*(N-1):
            w[i] = 1.0/(np.exp(Zplus[i]) + 1.0)
        elif i < (1.0-epsilon)*(N-1.0):
            w[i] = 1.0
        elif i <= (N-1.0):
            w[i] = 1.0/(np.exp(Zminus[i]) + 1.0)
        else:
            w[i] = 0.0
    return data, { "w": w }

window_planck_taper = smw.register("WindowPlanckTaper", "test_window_planck_taper", my_window, w_planck_taper_setup, [float], [TriState.FEATURES_TO_FEATURES, TriState.STREAM_TO_STREAM])
window_planck_taper.__doc__ = """
Performs a Planck-taper window on a signal, given a parameter epsilon
Epsilon is 0.1 on wikipedia

Args:
    data: numpy array of example
    epsilon: equation parameter

Returns:
    Data with Planck-taper window feature added
"""

def w_nuttall_setup(data):
    a0 = 0.355768
    a1 = 0.487396
    a2 = 0.144232
    a3 = 0.012604
    N = data.shape[1]
    nVec = np.arange(N)
    w = a0 - a1*np.cos(2.0*np.pi*nVec/(N-1.0)) + a2*np.cos(4.0*np.pi*nVec/(N-1.0)) - a3*np.cos(6.0*np.pi*nVec/(N-1.0))
    return data, { "w": w }

window_nuttall = smw.register("WindowNuttal", "test_window_nutall", my_window, w_nuttall_setup, [], [TriState.FEATURES_TO_FEATURES, TriState.STREAM_TO_STREAM])
window_nuttall.__doc__ = """
Perform a Nuttall window on a dataset

Args:
    data: numpy array of example

Returns:
    Data with Nuttall window feature added
"""

def w_blackman_harris_setup(data):
    a0 = 0.35875
    a1 = 0.48829
    a2 = 0.14128
    a3 = 0.01168
    N = data.shape[1]
    nVec = np.arange(N)
    w = a0 - a1*np.cos(2.0*np.pi*nVec/(N-1.0)) + a2*np.cos(4.0*np.pi*nVec/(N-1.0)) - a3*np.cos(6.0*np.pi*nVec/(N-1.0))
    return data, { "w": w }

window_blackman_harris = smw.register("WindowBlackmanHarris", "test_window_blackman_harris", my_window, w_blackman_harris_setup, [], [TriState.FEATURES_TO_FEATURES, TriState.STREAM_TO_STREAM])
window_blackman_harris.__doc__ = """
Perform a Blackman-Harris window on a dataset

Args:
    data: numpy array of example

Returns:
    Data with Blackman-Harris window feature added
"""

def w_blackman_nuttall_setup(data):
    a0 = 0.3635819
    a1 = 0.4891775
    a2 = 0.1365995
    a3 = 0.0106411
    N = data.shape[1]
    nVec = np.arange(N)
    w = a0 - a1*np.cos(2.0*np.pi*nVec/(N-1.0)) + a2*np.cos(4.0*np.pi*nVec/(N-1.0)) - a3*np.cos(6.0*np.pi*nVec/(N-1.0))
    return data, { "w": w }

window_blackman_nuttall = smw.register("WindowBackmanNuttall", "test_window_blackman_nuttall", my_window, w_blackman_nuttall_setup, [], [TriState.FEATURES_TO_FEATURES, TriState.STREAM_TO_STREAM])
window_blackman_nuttall.__doc__ = """
Perform a Blackman-Nuttall window on a dataset

Args:
    data: numpy array of example

Returns:
    Data with Blackman-Nuttall window feature added
"""

def w_flat_top_setup(data):
    a0 = 1.0
    a1 = 1.93
    a2 = 1.29
    a3 = 0.388
    a4 = 0.028
    N = data.shape[1]
    nVec = np.arange(N)
    w = a0 - a1*np.cos(2.0*np.pi*nVec/(N-1.0)) + a2*np.cos(4.0*np.pi*nVec/(N-1.0)) - a3*np.cos(6.0*np.pi*nVec/(N-1.0)) + a4*np.cos(8.0*np.pi*nVec/(N-1.0))
    return data, { "w": w }

window_flat_top = smw.register("WindowFlatTop", "test_window_flat_top", my_window, w_flat_top_setup, [], [TriState.FEATURES_TO_FEATURES, TriState.STREAM_TO_STREAM])
window_flat_top.__doc__ = """
Perform a flat top window on a dataset

Args:
    data: numpy array of example

Returns:
    Data with flat top window feature added
"""

def sub_sample_data(data_pair, window_size=10, step_size=3):
    """Sub-sample the data

    Args:
        data_pair: given datapair
        window_size: size of the window
        step_size: size of the steps

    Returns:
        Sub sample Data Pair
    """
    # For debugging purposes let's print out method name
    print("mySubSampleData") ; sys.stdout.flush()

    try:
        """
        Cache [Load] Block
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
            method_string = "mySubSampleData" + "_" + str(window_size) + "_" + str(step_size) + "_"

            # Combine the unique method name + arguments of the method + hash of the previous data
            # To form a unique key of the method call
            method_key = method_string + previous_hash

            overhead_time, method_row, cache_row, hit = check_cache_read(data_pair, database, 
                                                                            method_key, oh_time_start)
            if hit: return data_pair

            eval_time_start = time.time()

        # Just a couple of checks on the inputs
        window_size = abs(window_size)
        step_size = abs(step_size)
        if window_size < 1:
            window_size = 1
        if step_size < 1:
            step_size = 1

        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                data = instance.get_stream().get_data()

                N = data.shape[1]
                if window_size > N:
                    window_size = N
                if step_size > N:
                    step_size = N
                i = 0
                while i <= N - window_size:
                    instance.get_stream().set_data(data[:,i:i+window_size])
                    i += step_size

            data_set.set_instances(instances)
            data_list.append(data_set)

        """
        Update data pair with new data
        """
        data_pair.set_train_data(data_list[0])
        data_pair.set_test_data(data_list[1])

        """
        Cache [Store] Block
        """
        if data_pair.get_caching_mode():
            # The time from this evaluation isolated
            eval_time = time.time() - eval_time_start

            # Checks if method should be written to cache and updates data_pair
            check_cache_write(data_pair, database, 
                                method_row, cache_row, 
                                method_key, 
                                overhead_time, eval_time)

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

smwb.register("mySubSampleData", "test_sub_sample_data", sub_sample_data, None, [EmadeDataPair, int, int])

def cut_data_lead_helper(data, samples_to_cut=100):
    # Convert to an int to prevent deprication warnings
    samples_to_cut = int(samples_to_cut)

    data = data[:, samples_to_cut:]
    return data

cut_data_lead = smw.register("CutDataLead", "test_cut_data_lead", cut_data_lead_helper, None, [int], [TriState.FEATURES_TO_FEATURES, TriState.STREAM_TO_STREAM])
cut_data_lead.__doc__ = """
Cut samples_to_cut from the front of the data

Args:
    data: numpy array of example
    samples_to_cut: number of samples to cut

Returns:
    cut data
"""

def dwt_helper(data, axis=-1):
    coeff_list = pywt.wavedec(data, 'db3', axis=axis)
    vec = []
    [vec.extend(level) for level in coeff_list]
    data = np.array(vec)
    return data

my_dwt = smw.register("MyDWT", "test_my_dwt", dwt_helper, None, [int], TRI_STATE)
my_dwt.__doc__ = """
Perform a dwt on data

Args:
    data: numpy array of example
    axis: axis of data to perform on

Returns:
    modified data
"""

def rms_2d(data, axis=0):
    axis_options = [0, 1]
    axis = axis_options[axis%len(axis_options)]
    data = np.array([np.linalg.norm(data, 2, axis=axis)])
    return data

my_rms_2d = smw.register("RMS2D", "test_my_rms_2d", rms_2d, None, [int], [TriState.STREAM_TO_STREAM, TriState.STREAM_TO_FEATURES])
my_rms_2d.__doc__ = """
Computes rms of 2D data

Args:
    data: numpy array of example
    axis: id of which axis to use

Returns:
    Data with rms feature added
"""

def my_norm_helper(data, p=2):
    if p < 1:
        p = 1
    data = np.array([[np.linalg.norm(data, p)]])
    return data

my_norm = smw.register("MyNorm", "test_my_norm", my_norm_helper, None, [int], TRI_STATE)
my_norm.__doc__ = """
Computes the p norm on data

Args:
    data: numpy array of example
    p: value of p

Returns:
    Data with p norm feature added
"""

def my_sum_helper(data):
    data = np.array([[np.sum(data)]])
    return data

my_sum = smw.register("MySum", "test_my_sum", my_sum_helper, None, [], TRI_STATE)
my_sum.__doc__ = """
Sum of an example

Args:
    data: numpy array of example

Returns:
    Data with sum feature added
"""

def my_prod_helper(data):
    data = np.array([[np.prod(data)]])
    return data

my_prod = smw.register("MyProd", "test_my_prod", my_prod_helper, None, [], TRI_STATE)
my_prod.__doc__ = """
Product of an example

Args:
    data: numpy array of example

Returns:
    Data with product feature added
"""

def cum_prod(data, axis=-1):
    if axis != -1:
        axis = axis % len(data.shape)
    data = np.cumprod(data, axis=axis)
    return data

my_cum_prod = smw.register("MyCumProd", "test_my_cum_prod", cum_prod, None, [int], TRI_STATE)
my_cum_prod.__doc__ = """
Cumulative product of an example

Note: This primitive can cause non-finite values when the products
      overflow the data type bounds

Args:
    data: numpy array of example

Returns:
    Data with cumulative product feature added
"""

def cum_sum(data, axis=-1):
    if axis != -1:
        axis = axis % len(data.shape)
    data = np.cumsum(data, axis=axis)
    return data

my_cum_sum = smw.register("MyCumSum", "test_my_cum_sum", cum_sum, None, [int], TRI_STATE)
my_cum_sum.__doc__ = """
Computes the cumulative sum on each point in a dataset.

Args:
    data: numpy array of example

Returns:
    Data with cumulative sum feature added
"""

def my_fft_helper(data):
    data = np.abs(np.fft.fft(data))
    return data

my_fft = smw.register("FFT", "test_fft", my_fft_helper, None, [], TRI_STATE)
my_fft.__doc__ = """
Perform an fft on data

Args:
    data: numpy array of example

Returns:
    Data with fft feature added
"""

def my_abs_helper(data):
    return np.abs(data)

my_abs = smw.register("AbsoluteValue", "test_my_abs", my_abs_helper, None, [], [TriState.FEATURES_TO_FEATURES, TriState.STREAM_TO_STREAM])
my_abs.__doc__ = """
Perform an absolute value on data

Args:
    data: numpy array of example

Returns:
    Data with absolute value feature added
"""

def my_exp_helper(data):
    return np.exp(data)

my_exp = smw.register("Exponetial", "test_my_exp", my_exp_helper, None, [], [TriState.FEATURES_TO_FEATURES, TriState.STREAM_TO_STREAM])
my_exp.__doc__ = """
Perform an exponential on data

Args:
    data: numpy array of example

Returns:
    Data with exponential feature added
"""

def my_tangent_helper(data):
    return np.tan(data)

my_tangent = smw.register("Tangent", "test_my_tangent", my_tangent_helper, None, [], [TriState.FEATURES_TO_FEATURES, TriState.STREAM_TO_STREAM])
my_tangent.__doc__ = """
Perform a tangent on data

Args:
    data: numpy array of example

Returns:
    Data with tangent feature added
"""

def my_cosine_helper(data):
    return np.cos(data)

my_cosine = smw.register("Cosine", "test_my_cosine", my_cosine_helper, None, [], [TriState.FEATURES_TO_FEATURES, TriState.STREAM_TO_STREAM])
my_cosine.__doc__ = """
Perform a cosine on data

Args:
    data: numpy array of example

Returns:
    Data with cosine feature added
"""

def my_sine_helper(data):
    return np.sin(data)

my_sine = smw.register("Sine", "test_my_sine", my_sine_helper, None, [], [TriState.FEATURES_TO_FEATURES, TriState.STREAM_TO_STREAM])
my_sine.__doc__ = """
Perform a sine on data

Args:
    data: numpy array of example

Returns:
    Data with sine feature added
"""

def my_arctangent_helper(data):
    return np.arctan(data)

my_arctangent = smw.register("ArcTangent", "test_my_arctangent", my_arctangent_helper, None, [], [TriState.FEATURES_TO_FEATURES, TriState.STREAM_TO_STREAM])
my_arctangent.__doc__ = """
Perform an arctangent on data

Args:
    data: numpy array of example

Returns:
    Data with arctangent feature added
"""

def my_arccosine_helper(data):
    return np.arccos(data)

my_arccosine = smw.register("ArcCosine", "test_my_arccosine", my_arccosine_helper, None, [], [TriState.FEATURES_TO_FEATURES, TriState.STREAM_TO_STREAM])
my_arccosine.__doc__ = """
Perform an arccosine on data

Args:
    data: numpy array of example

Returns:
    Data with arccosine feature added
"""

def my_arcsine_helper(data):
    return np.arcsin(data)

my_arcsine = smw.register("ArcSine", "test_my_arcsine", my_arcsine_helper, None, [], [TriState.FEATURES_TO_FEATURES, TriState.STREAM_TO_STREAM])
my_arcsine.__doc__ = """
Perform an arcsine on data

Args:
    data: numpy array of example

Returns:
    Data with arcsine feature added
"""

def my_log_helper(data):
    data = np.log(data)
    # mean = np.nanmean(data)
    # mean = mean if mean != np.nan else 0
    return np.nan_to_num(data)

my_log = smw.register("Log", "test_my_log", my_log_helper, None, [], [TriState.FEATURES_TO_FEATURES, TriState.STREAM_TO_STREAM])
my_log.__doc__ = """
Perform a log on data
Then replaces non-finite values with real values
Infinity is replaced with the largest finite values available
NaN is replaced by 0

TODO: Standardize Numpy 1.17 for EMADE and use np.nan_to_num(data, nan=mean) instead

Args:
    data: numpy array of example

Returns:
    Data with log feature added
"""

def my_round_helper(data):
    return np.round(data)

my_round = smw.register("Round", "test_my_round", my_round_helper, None, [], [TriState.FEATURES_TO_FEATURES, TriState.STREAM_TO_STREAM])
my_round.__doc__ = """
Perform a round on data

Args:
    data: numpy array of example

Returns:
    Data with round feature added
"""

def my_kalman_filter_helper(data, Q=1e-5, R=0.1**2):
    Q = np.abs(Q)
    R = np.abs(R)
    N = data.shape[1]
    new_data = []
    for row in data:
        xhat = np.zeros(N)          # A posteri estimate of x
        P = np.zeros(N)             # A posteri error estimate
        xhatminus = np.zeros(N)     # A priori estimate of x
        Pminus = np.zeros(N)        # A priori error estimate
        K = np.zeros(N)             # gain or blending factor
        xhat[0] = row[0]
        P[0] = 1.0                  # What should this be?
        for k in range(1, N):
            xhatminus[k] = xhat[k-1]
            Pminus[k] = P[k-1] + Q

            K[k] = Pminus[k]/(Pminus[k] + R)
            xhat[k] = xhatminus[k] + K[k]*(row[k]-xhatminus[k])
            P[k] = (1 - K[k])*Pminus[k]

        new_data.append(xhat)
    data = np.array(new_data)
    return data

my_kalman_filter = smw.register("KalmanFilter", "test_my_kalman_filter", my_kalman_filter_helper, None, [float, float], [TriState.FEATURES_TO_FEATURES, TriState.STREAM_TO_STREAM])
my_kalman_filter.__doc__ = """
Perform a kalman filter on data

Args:
    data: numpy array of example
    Q: equation paramater
    R: equation parameter

Returns:
    Data with kalman filter feature added
"""

def my_linear_predictive_coding_helper(data, p=5):
    p = np.abs(p)
    rows = []
    for row in data:
        F = np.fft.fft(np.array([row]))
        S = np.conjugate(F)*F
        R = np.fft.ifft(S)
        R = np.real(R)
        A = np.zeros((int(p), int(p)))
        for i in range(0, p):
            for j in range(0, p):
                A[i, j] = R[0, np.abs(i-j)]
        colR = np.transpose(np.reshape(R[0, 1:p+1], (1, -1)))
        coeffs = np.linalg.inv(A)*colR
        coeffs = np.transpose(coeffs)
        row = coeffs[0, :]
        rows.extend(row)
    data = np.array([rows])
    return data

my_linear_predictive_coding = smw.register("LinearPredictiveCoding", "test_my_linear_predictive_coding", my_linear_predictive_coding_helper, None, [int], TRI_STATE)
my_linear_predictive_coding.__doc__ = """
Computes the LPC coefficients

Args:
    data: numpy array of example
    p: equation parameter

Returns:
    Data with LPC coefficients feature added
"""

def my_wiener_filter_helper(data):
    return scipy.signal.wiener(data)

my_wiener_filter = smw.register("WienerFilter", "test_my_wiener_filter", my_wiener_filter_helper, None, [], [TriState.FEATURES_TO_FEATURES, TriState.STREAM_TO_STREAM])
my_wiener_filter.__doc__ = """
Perform a wiener on data

Args:
    data: numpy array of example

Returns:
    Data with wiener feature added
"""

def my_savitzky_golay_filter_helper(data, N=11, M=3, deriv=0):
    N = np.abs(N)
    M = np.abs(M)
    deriv = np.abs(deriv)
    rows = []
    for row in data:
        rows.append(savitzky_golay(row, N, M, deriv))
    data = np.array(rows)
    return data

my_savitzky_golay_filter = smw.register("SavitzkyGolayFilter", "test_my_savitzky_golay_filter", my_savitzky_golay_filter_helper, None, [int, int, int], [TriState.FEATURES_TO_FEATURES, TriState.STREAM_TO_STREAM])
my_savitzky_golay_filter.__doc__ = """
Perform a Savitzky-Golay filter on data

Args:
    data: numpy array of example
    N: inner method paramater
    M: inner method parameter
    deriv: inner method parameter

Returns:
    Data with Savitzky-Golay filter feature added
"""

def gaussian_hmm(data_pair, n_components=2, algorithm='viterbi'):
    # Deactivated until a better version is implemented through VIP
    """Implements a Gaussian hidden markov model

    Args:
        data_pair: given datapair
        n_components: number of states
        algorithm: decoder algorithm

    Returns:
        Data Pair transformed with hidden markov model
    """
    data_pair = cp.deepcopy(data_pair)
    training_data = data_pair.get_train_data().get_numpy()
    # target_values = np.array([inst.get_target()[0] for
    #                           inst in data_pair.get_train_data().get_instances()])
    target_values = data_pair.get_train_data().get_target()
    my_hmm = hmmlearn.hmm.GaussianHMM(n_components = n_components,algorithm=algorithm)
    my_hmm.fit([training_data])

    testing_data = data_pair.get_test_data().get_numpy()
    predicted_classes = my_hmm.predict(testing_data)
    # [inst.set_target([target]) for inst, target in
    #     zip(data_pair.get_test_data().get_instances(), predicted_classes)]
    data_pair.get_test_data().set_target(predicted_classes)

    # Set the self-predictions of the training data
    trained_classes = my_hmm.predict(training_data)
    # [inst.set_target([target]) for inst, target in
    #     zip(data_pair.get_train_data().get_instances(), trained_classes)]
    data_pair.get_train_data().set_target(trained_classes)

    data_pair = makeFeatureFromClass(data_pair, name="HMM")

    gc.collect(); return data_pair

def my_ecdf_helper(data, n_components=2):
    return ecdfRep(data, n_components)

my_ecdf = smw.register("ECDF", "test_ecdf", my_ecdf_helper, None, [int], TRI_STATE)
my_ecdf.__doc__ = """
Perform an ecdf on data

Args:
    data: numpy array of example
    n_components: the dimension of projected space

Returns:
    Data with ecdf feature added
"""

def my_rebase_helper(data):
    return data - np.min(data)

my_rebase = smw.register("Rebase", "test_my_rebase", my_rebase_helper, None, [], [TriState.FEATURES_TO_FEATURES, TriState.STREAM_TO_STREAM])
my_rebase.__doc__ = """
Perform a rebase on data

Args:
    data: numpy array of example

Returns:
    Data with rebase feature added
"""

def my_peak_finder_helper(data, start_delta=5, lookForMax=1):
    rows = []
    for row in data:
        min_val, max_val = np.inf, -np.inf
        min_pos, max_pos = -1, 1
        min_locs = []
        max_locs = []
        delta = start_delta
        for i in np.arange(len(row)):
            val = row[i]
            if val > max_val:
                max_val = val
                max_pos = i
            if val < min_val:
                min_val = val
                min_pos = i

            if lookForMax:
                if val < (max_val - delta):
                    max_locs += [max_pos]
                    min_val = val
                    min_pos = i
                    lookForMax = 0
            else:
                if val > (min_val + delta):
                    min_locs += [min_pos]
                    max_val = val
                    max_pos = i
                    lookForMax = 1
            #delta = delta*0.9999
        #peak_locs = min_locs + max_locs # concat two lists
        peak_locs = max_locs # concat two lists
        not_peaks = [i for i in np.arange(len(row)) if i not in peak_locs]
        row[not_peaks] = 0
        rows.append(row)
    data = np.array(rows)
    return data

my_peak_finder = smw.register("PeakFinder", "test_my_peak_finder", my_peak_finder_helper, None, [float, int], TRI_STATE)
my_peak_finder.__doc__ = """
Perform a peak finder on data
Zero out all but the peaks in the data
delta is the amount to withdraw before locking

Args:
    data: numpy array of example
    start_delta: initial value of delta
    lookForMax: max parameter

Returns:
    Data with peak finder feature added
"""

def my_informed_search_helper(data, peak_data, search_window=5, log_slope_thresh=0.08):
    if peak_data.shape != data.shape:
        raise ValueError('Peaks and data do not match in shape')
    deriv = 2
    order = 3
    window_length = 11
    rows = []
    maxzcs = 0

    # main code
    for row, peak_row in zip(data, peak_data):
        zero_crossings = []
        log_slopes = []
        all_smoothed_data = savitzky_golay(row, window_length,
            order, deriv)
        # search_window
        for i in np.arange(search_window,len(peak_row)):
            if peak_row[i] > 0:
                k = i - search_window
                # Take the Second derivative
                smoothed_data = all_smoothed_data[k:i]
                zero_crossing = None
                # Find the Zero crossing if any
                interest_point_zero_crossings = []
                interest_point_log_slopes = []
                for j in np.arange(search_window-1):
                    if smoothed_data[j]*smoothed_data[j+1] <= 0:
                        log_slope = abs(np.log10(row[k+j+3]+1e-12) - np.log10(row[k+j-3]+1e-12))
                        zero_crossing = k+j

                        # log the point
                        interest_point_zero_crossings.append(zero_crossing)
                        interest_point_log_slopes.append(log_slope)

                interest_point_log_slopes = np.array(interest_point_log_slopes)
                interest_point_zero_crossings = np.array(interest_point_zero_crossings)
                if len(interest_point_log_slopes) > 0:
                    log_slope = np.max(interest_point_log_slopes)
                    if log_slope >= log_slope_thresh:
                        log_slope_loc = np.argmax(interest_point_log_slopes)
                        zero_crossings.append(interest_point_zero_crossings[log_slope_loc])
        if maxzcs == 0:
            # This will protect against empty data
            maxzcs = 1
        # Keep track of most zero crossings for pading
        if len(zero_crossings) > maxzcs:
            maxzcs = len(zero_crossings)
        rows.append(zero_crossings)
    # Pad the rows
    for i in np.arange(len(rows)):
        rows[i].extend(np.ones(maxzcs-len(rows[i]))*-1)
    data = np.array(rows)


    return data

my_informed_search = smw_2.register("MyInformedSearch", "test_my_informed_search", my_informed_search_helper, None, [int, float], TRI_STATE)
my_informed_search.__doc__ = """
Zero out all but the peaks in the data

delta is the amount to withdraw before locking

Args:
    data: numpy array
    peak_data: numpy array
    search_window: size of search window
    log_slope_thresh: equation parameter

Returns:
    Transformed data
"""

def savitzky_golay(y, window_size, order, deriv=0):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.

    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techhniques.

    Args:
        y: the values of the time history of the signal
        window_size: the length of the window. Must be an odd integer number
        order: the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1
        deriv: the order of the derivative to compute
        (default = 0 means only smoothing)

    Returns:
        the smoothed signal (or it's n-th derivative)

    Raises:
        ValueError, TypeError
    """
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv]
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m, y, mode='valid')

def ecdfRep(data, components):
    """Estimate ecdf-representation according to Hammerla, Nils Y., et al.
    "On preserving statistical characteristics of accelerometry data using
    their empirical cumulative distribution." ISWC. ACM, 2013.

    Args:
        data: input data (rows = samples)
        components: number of components to extract per axis

    Returns:
        data representation with M = d*components+d elements
    """
    m = data.mean(1)
    data = np.sort(data, axis=1)
    data = data[:, np.int32(np.around(np.linspace(0,data.shape[1]-1,num=components)))]
    data = data.flatten(0)
    return np.hstack((data, m)).reshape((1, -1))

def myArgMin(dataPair):
    """Calculates Argmin of Data Pair

    Args:
        dataPair: given Data Pair

    Returns:
        modified Data Pair
    """
    dataPair = cp.deepcopy(dataPair)
    mins = [];
    for row in dataPair.testData.numpy['features']:
        mins += [np.argmin(row)]

    dataPair.testData.numpy['classes'] = mins
    dataPair.testData.numpy['classes'] = np.array([[str(int(val))]for val in dataPair.testData.numpy['classes']])
    dataPair.testData.rebuildFromNumpy()
    # TODO: We want to add on new class value as a feature as well
    return dataPair

def richardson_lucy_setup(data, iterations=50):
    psf = np.ones((data.shape)) / data.shape[1]
    return data, { "psf":psf, "iterations":iterations }

def my_richardson_lucy_helper(data, psf=None, iterations=50):
    return restoration.richardson_lucy(data, psf, iterations=iterations)

my_richardson_lucy = smw.register("RichardsonLucy", "test_my_richardson_lucy", my_richardson_lucy_helper, richardson_lucy_setup, [int], [TriState.FEATURES_TO_FEATURES, TriState.STREAM_TO_STREAM])
my_richardson_lucy.__doc__ = """
Perform a richardson-lucy deconvolution on data

Args:
    data: numpy array of example
    iterations: number of iterations for deconvolution

Returns:
    Data with richardson-lucy deconvolution feature added
"""

def supersampling_setup(data, size_factor=0.1, resample=1):
    resample = resample % 4
    if resample == 0:
        resample = Image.BILINEAR
    elif resample == 1:
        resample = Image.NEAREST
    elif resample == 2:
        resample = Image.BICUBIC
    else:
        resample = Image.LANCZOS
    size = (data.shape[0] , int(np.round(data.shape[1]*size_factor)))
    return data, { "size":size, "resample":resample }

def my_supersampling_helper(data, size=None, resample=None):
    im = Image.fromarray(data)
    im.resize(size=size, resample=resample)
    return np.array(im)

my_supersampling = smw.register("Supersampling", "test_my_supersampling", my_supersampling_helper, supersampling_setup, [float, int], [TriState.FEATURES_TO_FEATURES, TriState.STREAM_TO_STREAM])
my_supersampling.__doc__ = """
Perform a supersampling on data

Args:
    data: numpy array of example
    size_factor: Target size for downsampling
    resample: type of resample to use

Returns:
    Data with supersampling feature added
"""

def my_bspline_helper(data, spline_degree=3):
    X = [x+1 for x in range(data.shape[1])]
    # if w_mode:
    #     w_param = np.random.rand(len(X))
    # else:
    #     w_param = np.ones(len(X))
    for dimension in range(data.shape[0]):
        t,c,k = interpolate.splrep(x=X, y=data[dimension], s=0, k=spline_degree)
        spline = interpolate.BSpline(t,c,k)
        data[dimension] = spline(X)
    return data

my_bspline = smw.register("b-spline", "test_my_bspline", my_bspline_helper, None, [], [TriState.FEATURES_TO_FEATURES, TriState.STREAM_TO_STREAM])
my_bspline.__doc__ = """
Perform a b-spline curve fitting on data

Args:
    data: numpy array of example

Returns:
    Data from the curve that was fit using b spline
"""

def matched_filtering_setup(data, method=2):
    method = method % 3
    if method == 0:
        method_name = 'direct'
    elif method == 1:
        method_name = 'fft'
    else:
        method_name = 'auto'
    return data, { "method_name":method_name }

def my_matched_filtering_helper(data, method_name=None):
    for dimension in range(data.shape[0]):
        data[dimension] = scipy.signal.correlate(data[dimension], np.ones(data.shape[1]), mode='valid', method=method_name) / data.shape[1]
    return data

my_matched_filtering = smw.register("MatchedFiltering", "test_my_matched_filtering", my_matched_filtering_helper, matched_filtering_setup, [int], [TriState.FEATURES_TO_FEATURES, TriState.STREAM_TO_STREAM])
my_matched_filtering.__doc__ = """
Perform a matched filtering on data
https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.signal.correlate.html

Args:
    data: numpy array of example
    method: type of method to use

Returns:
    Data with matched filtering feature added
"""

def my_gaussian_peak_lm(data_pair, amp_1, cen_1, wid_1, amp_2, cen_2, wid_2, mode=TriState.FEATURES_TO_FEATURES):
    """Fit the data to the sum of two gaussians using the levenburg-markwardt method

    Args:
        data_pair: given datapair
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Peak locations of two surface responses
    """
    if data_pair.get_caching_mode():
        raise ValueError("Method not currently cached")

    def gaussian(x, amp_1, cen_1, wid_1, amp_2, cen_2, wid_2):
        y = np.zeros_like(x)
        y = (amp_1 * np.exp(-((x - cen_1) / wid_1)**2)) + (amp_2 * np.exp(-((x - cen_2) / wid_2)**2))
        return y

    # def gaussian(x, *params):
    #     y = np.zeros_like(x)
    #     y = (params[0] * np.exp(-((x - params[1]) / params[2])**2)) + (params[3] * np.exp(-((x - params[4]) / params[5])**2))
    #     return y

    data_list = []
    gmodel = Model(gaussian)
    for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
        instances = cp.deepcopy(data_set.get_instances())
        for instance in instances:
            if mode is TriState.FEATURES_TO_FEATURES:
                raise ValueError(
                    'No Feature to Features available for my_gaussian_peak'
                )
            elif mode is TriState.STREAM_TO_STREAM:
                data = instance.get_stream().get_data()
            elif mode is TriState.STREAM_TO_FEATURES:
                data = instance.get_stream().get_data()
            X = [x for x in range(len(data[0]))]
            new_data = []
            for row in data:
                try:
                    result = gmodel.fit(row, x=X, amp_1=amp_1, cen_1=cen_1, wid_1=wid_1,
                                        amp_2=amp_2, cen_2=cen_2, wid_2=wid_2)
                    if mode is TriState.STREAM_TO_STREAM:
                        fit = result.best_fit
                        row = fit
                        row[row < 0] = 0
                    elif mode is TriState.STREAM_TO_FEATURES:
                        row = [result.params['cen_1'].value, result.params['cen_2'].value]
                    new_data.append(row)
                except RuntimeError:
                    print("Error - curve_fit failed")
            new_data = np.array(new_data)
            if mode is TriState.FEATURES_TO_FEATURES:
                raise ValueError(
                    'No Feature to Features available for my_gaussian_peak'
                )
            elif mode is TriState.STREAM_TO_STREAM:
                instance.get_stream().set_data(new_data)
            elif mode is TriState.STREAM_TO_FEATURES:
                instance.get_features().set_data(new_data, labels=np.array(['cen_1', 'cen_2']))
        new_data_set = EmadeData(instances)
        data_list.append(new_data_set)
    data_pair.set_train_data(data_list[0])
    data_pair.set_test_data(data_list[1])

    gc.collect()
    return data_pair

# Taken out of framework due to multiple datapairs
def gaussian_peak_lm(data_pair, peak_data_pair, mode=TriState.FEATURES_TO_FEATURES):
    """Fit the data to the sum of two gaussians using the levenburg-markwardt method

    Args:
        data_pair: given datapair
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Peak locations of two surface responses
    """
    if data_pair.get_caching_mode():
        raise ValueError("Method not currently cached")

    def gaussian(x, amp_1, cen_1, wid_1, amp_2, cen_2, wid_2):
        y = np.zeros_like(x)
        y = (amp_1 * np.exp(-((x - cen_1) / wid_1)**2)) + (amp_2 * np.exp(-((x - cen_2) / wid_2)**2))
        return y

    # def gaussian(x, *params):
    #     y = np.zeros_like(x)
    #     y = (params[0] * np.exp(-((x - params[1]) / params[2])**2)) + (params[3] * np.exp(-((x - params[4]) / params[5])**2))
    #     return y


    data_list = []
    gmodel = Model(gaussian)
    for data_set, peak_set in zip([data_pair.get_train_data(), data_pair.get_test_data()],
                                  [peak_data_pair.get_train_data(), peak_data_pair.get_test_data()]):
        instances = cp.deepcopy(data_set.get_instances())
        peak_instances = cp.deepcopy(peak_set.get_instances())
        for instance, peak_instance in zip(instances, peak_instances):
            if mode is TriState.FEATURES_TO_FEATURES:
                raise ValueError(
                    'No Feature to Features available for my_gaussian_peak'
                )
            elif mode is TriState.STREAM_TO_STREAM:
                data = instance.get_stream().get_data()
                peak_data = peak_instance.get_stream().get_data()
            elif mode is TriState.STREAM_TO_FEATURES:
                data = instance.get_stream().get_data()
                peak_data = peak_instance.get_stream().get_data()
            if peak_data.shape != data.shape:
                raise ValueError('Peaks and data do not match in shape')
            X = [x for x in range(len(data[0]))]
            new_data = []
            for row, peak_row in zip(data, peak_data):
                try:
                    peaks = peak_row[peak_row > 0]
                    if len(peaks) < 2:
                        peaks = np.array(peaks.tolist().append(peaks[0] + np.random.randint(0, 100)))
                    result = gmodel.fit(row, x=X, amp_1=row[peaks[0]], cen_1=peaks[0], wid_1=50,
                                        amp_2=row[peaks[-1]], cen_2=peaks[-1], wid_2=50)
                    row = [result.params['cen_1'].value, result.params['cen_2'].value]
                    new_data.append(row)
                except RuntimeError:
                    print("Error - curve_fit failed")
            new_data = np.array(new_data)
            if mode is TriState.FEATURES_TO_FEATURES:
                raise ValueError(
                    'No Feature to Features available for my_gaussian_peak'
                )
            elif mode is TriState.STREAM_TO_STREAM:
                instance.get_stream().set_data(new_data)
            elif mode is TriState.STREAM_TO_FEATURES:
                instance.get_features().set_data(new_data, labels=np.array(['cen_1', 'cen_2']))
        new_data_set = EmadeData(instances)
        data_list.append(new_data_set)
    data_pair.set_train_data(data_list[0])
    data_pair.set_test_data(data_list[1])

    gc.collect()
    return data_pair

def gaussian_peak_em_helper(data):
    num_components = 2
    X = data.reshape(data.shape[1], data.shape[0])
    n, d = X.shape

    mu = X[np.random.choice(n, num_components, False), :]
    Sigma = [np.eye(d)] * num_components
    w = [1. / num_components] * num_components
    R = np.zeros((n, num_components))
    log_likelihoods = []

    P = lambda mu, s: np.linalg.det(s) ** -.5 * (2 * np.pi) ** (-X.shape[1] / 2.) \
                      * np.exp(-.5 * np.einsum('ij, ij -> i', \
                                               X - mu, np.dot(np.linalg.inv(s), (X - mu).T).T))

    while len(log_likelihoods) < 1000:
        for k in range(num_components):
            R[:, k] = w[k] * P(mu[k], Sigma[k])

        log_likelihood = np.sum(np.log(np.sum(R, axis=1)))
        log_likelihoods.append(log_likelihood)
        R = (R.T / np.sum(R, axis=1)).T
        N_ks = np.sum(R, axis=0)
        for k in range(num_components):
            mu[k] = 1. / N_ks[k] * np.sum(R[:, k] * X.T, axis=1).T
            x_mu = np.matrix(X - mu[k])

            Sigma[k] = np.array(1 / N_ks[k] * np.dot(np.multiply(x_mu.T, R[:, k]), x_mu))

            w[k] = 1. / n * N_ks[k]

        if len(log_likelihoods) < 2: continue
        if np.abs(log_likelihood - log_likelihoods[-2]) < 0.0001: break

    data = []

    p = [0]
    for k in range(num_components):
        p.append(p[k] + w[k])
    for i in range(d):
        for k in range(num_components):
            if random.random() < p[k + 1]:
                data.append(np.random.multivariate_normal(mu[k + 1], Sigma[k + 1]))

    return np.array(data)

gaussian_peak_em = smw.register("GaussianPeakEM", "test_gaussian_peak_em", gaussian_peak_em_helper, None, [], [TriState.FEATURES_TO_FEATURES, TriState.STREAM_TO_STREAM])
gaussian_peak_em.__doc__ = """
Fit the data to the sum of three gaussians using the expectation-maximization method

Args:
    data: numpy array of example

Returns:
    Peak locations of the two surface responses
"""

def lognormal_lm_helper(data):
    def lognormal(x, mean, std):
        return (1/(x*std*np.sqrt(2*np.pi))) * np.exp(-(np.log(x)-mean)**2 /(2*(std**2)))

    X = [x+1 for x in range(data.shape[1])]
    for dimension in range(data.shape[0]):

        popt = [np.mean(data[dimension]), np.std(data[dimension])]

        popt, pcov = curve_fit(lognormal, X, data[dimension], p0 = popt, method='lm',maxfev = 10000)

        data[dimension] = np.array([lognormal(x, popt[0], popt[1]) for x in X])

    return data

lognormal_lm = smw.register("LognormalLM", "test_my_lognormal_lm", lognormal_lm_helper, None, [], [TriState.FEATURES_TO_FEATURES, TriState.STREAM_TO_STREAM])
lognormal_lm.__doc__ = """
Fits a lognormal curve to each dimension and returns a stream from the lognormal distribution using the levenburg-markwardt method

Args:
    data: numpy array of example

Returns:
    Peak locations of surface response
"""

def my_nnls_deconvolution(data_pair, mode=TriState.STREAM_TO_STREAM):
    if data_pair.get_caching_mode():
        raise ValueError("Method not currently cached")

    data_list = []
    for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
        instances = cp.deepcopy(data_set.get_instances())

        for instance in instances:
            if mode is TriState.FEATURES_TO_FEATURES:
                data = instance.get_features().get_data()
            elif mode is TriState.STREAM_TO_STREAM:
                data = instance.get_stream().get_data()
            elif mode is TriState.STREAM_TO_FEATURES:
                raise ValueError(
                    'No Stream to Features available for supersampling Filter'
                )

            if mode is TriState.FEATURES_TO_FEATURES:
                instance.get_features().set_data(data)
            elif mode is TriState.STREAM_TO_STREAM:
                instance.get_stream().set_data(data)

        new_data_set = EmadeData(instances)
        data_list.append(new_data_set)
    data_pair.set_train_data(data_list[0])
    data_pair.set_test_data(data_list[1])

    gc.collect()
    return data_pair

# This method is not designed to work with wrapper. Needs a redesign to be added to framework.
def my_waveform_fitting_2(data_pair, amp_1, cen_1, wid_1, amp_2, cen_2, wid_2, x_1, x_2, x_3, x_4, y_3, y_4, mode=TriState.FEATURES_TO_FEATURES):
    """Fit the data to the sum of two gaussians using the levenburg-markwardt method

    Args:
        data_pair: given datapair
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Peak locations of two surface responses
    """

    # def fit_method(x, amp_1, cen_1, wid_1, amp_2, cen_2, wid_2, x_1, x_2, x_3, x_4, y_3, y_4):
    #     y = np.zeros_like(x)
    #     temp_x = cp.deepcopy(x).astype(float)
    #     temp_x[temp_x < x_1] = 0
    #     temp_x[np.logical_and(temp_x >= x_1, temp_x <= x_2)] = y_3 * ((temp_x[np.logical_and(temp_x >= x_1, temp_x <= x_2)] - x_1) / (x_2 - x_1))
    #     temp_x[np.logical_and(temp_x >= x_2, temp_x <= x_3)] = ((y_3 * x_3 - x_2 * y_4) + temp_x[np.logical_and(temp_x >= x_2, temp_x <= x_3)] * (y_4 - y_3)) / (x_3 - x_2)
    #     temp_x[np.logical_and(temp_x >= x_3, temp_x <= x_4)] = y_4 * ((x_4 - temp_x[np.logical_and(temp_x >= x_3, temp_x <= x_4)]) / (x_4 - x_3))
    #     temp_x[temp_x >= x_4] = 0
    #     import matplotlib.pyplot as plt
    #
    #     x = np.array([i for i in range(len(x))])
    #     y = (amp_1 * np.exp(-1 * ((x - cen_1) / wid_1)**2))
    #     y += (amp_2 * np.exp(-1 * ((x - cen_2) / wid_2)**2))
    #     plt.plot(y)
    #     y += temp_x
    #     plt.plot(y)
    #     plt.show()

    def fit_method(x, amp_1, cen_1, wid_1, amp_2, cen_2, wid_2, a, b, c, d, e, g):
        y = np.zeros_like(x)
        temp_x = cp.deepcopy(x).astype(float)
        temp_x[temp_x <= a] = 0
        temp_x[np.logical_and(a < temp_x, temp_x <= b)] = e * ((temp_x[np.logical_and(a < temp_x, temp_x <= b)] - a) / (b - a))
        temp_x[np.logical_and(b < temp_x, temp_x <= c)] = ((e * c - b * g) + temp_x[np.logical_and(b < temp_x, temp_x <= c)] * (g - e)) / (c - b)
        temp_x[np.logical_and(c < temp_x, temp_x <= d)] = g * ((d - temp_x[np.logical_and(c < temp_x, temp_x <= d)]) / (d - c))
        temp_x[temp_x >= x_4] = 0
        x = np.array([i for i in range(len(x))])
        y = (amp_1 * np.exp(-1 * ((x - cen_1) / wid_1)**2))
        y += (amp_2 * np.exp(-1 * ((x - cen_2) / wid_2)**2))
        y += temp_x

    # def cost_function(params, x, stream_data):
    #     vals = params.valuesdict()
    #     amp_1 = vals['amp_1']
    #     cen_1 = vals['cen_1']
    #     wid_1 = vals['wid_1']
    #     amp_2 = vals['amp_2']
    #     cen_2 = vals['cen_2']
    #     wid_2 = vals['wid_2']
    #
    #     x_1 = vals['x_1']
    #     x_2 = vals['x_2']
    #     x_3 = vals['x_3']
    #     x_4 = vals['x_4']
    #     y_3 = vals['y_3']
    #     y_4 = vals['y_4']
    #
    #     y = np.zeros_like(x)
    #     temp_x = cp.deepcopy(x).astype(float)
    #     temp_x[temp_x < x_1] = 0
    #     temp_x[np.logical_and(temp_x >= x_1, temp_x <= x_2)] = y_3 * ((temp_x[np.logical_and(temp_x >= x_1, temp_x <= x_2)] - x_1) / (x_2 - x_1))
    #     temp_x[np.logical_and(temp_x >= x_2, temp_x <= x_3)] = ((y_3 * x_3 - x_2 * y_4) + temp_x[np.logical_and(temp_x >= x_2, temp_x <= x_3)] * (y_4 - y_3)) / (x_3 - x_2)
    #     temp_x[np.logical_and(temp_x >= x_3, temp_x <= x_4)] = y_4 * ((x_4 - temp_x[np.logical_and(temp_x >= x_3, temp_x <= x_4)]) / (x_4 - x_3))
    #     temp_x[temp_x >= x_4] = 0
    #
    #     x = np.array([i for i in range(len(x))])
    #     y = (amp_1 * np.exp(-1 * ((x - cen_1) / wid_1)**2))
    #     y += (amp_2 * np.exp(-1 * ((x - cen_2) / wid_2)**2))
    #     y += temp_x
    #     return np.sum(abs(stream_data - y))

    data_list = []
    # fit_params = Parameters()
    # fit_params.add('amp_1', value=amp_1)
    # fit_params.add('cen_1', value=cen_1)
    # fit_params.add('wid_1', value=wid_1)
    # fit_params.add('amp_2', value=amp_2)
    # fit_params.add('cen_2', value=cen_2)
    # fit_params.add('wid_2', value=wid_2)
    # fit_params.add('x_1', value=x_1)
    # fit_params.add('x_2', value=x_2)
    # fit_params.add('x_3', value=x_3)
    # fit_params.add('x_4', value=x_4)
    # fit_params.add('y_3', value=y_3)
    # fit_params.add('y_4', value=y_4)
    gmodel = Model(fit_method)
    for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
        instances = cp.deepcopy(data_set.get_instances())
        for instance in instances:
            if mode is TriState.FEATURES_TO_FEATURES:
                raise ValueError(
                    'No Feature to Features available for my_gaussian_peak'
                )
            elif mode is TriState.STREAM_TO_STREAM:
                data = instance.get_stream().get_data()
                indicies = instance.get_stream().get_labels()
            elif mode is TriState.STREAM_TO_FEATURES:
                data = instance.get_stream().get_data()
                indicies = instance.get_stream().get_labels()
            # X = [x for x in range(len(data[0]))]
            new_data = []
            for row in data:
                try:
                    result = gmodel.fit(row, x=indicies, amp_1=amp_1, cen_1=cen_1, wid_1=wid_1,
                                        amp_2=amp_2, cen_2=cen_2, wid_2=wid_2, a=a, b=b, c=c, d=d, e=e, g=g)
                    if mode is TriState.STREAM_TO_STREAM:
                        fit = result.best_fit
                        row = fit
                        row[row < 0] = 0
                    elif mode is TriState.STREAM_TO_FEATURES:
                        row = [result.params['cen_1'].value, result.params['cen_2'].value]
                    new_data.append(row)
                except RuntimeError:
                    print("Error - curve_fit failed")
            new_data = np.array(new_data)
            if mode is TriState.FEATURES_TO_FEATURES:
                raise ValueError(
                    'No Feature to Features available for my_gaussian_peak'
                )
            elif mode is TriState.STREAM_TO_STREAM:
                instance.get_stream().set_data(new_data)
            elif mode is TriState.STREAM_TO_FEATURES:
                instance.get_features().set_data(new_data, labels=np.array(['cen_1', 'cen_2']))
        new_data_set = EmadeData(instances)
        data_list.append(new_data_set)
    data_pair.set_train_data(data_list[0])
    data_pair.set_test_data(data_list[1])

    gc.collect()
    return data_pair

# This method is not designed to work with wrapper. Needs a redesign to be added to framework.
def my_progressive_waveform_decomposition(data_pair, mode=TriState.FEATURES_TO_FEATURES):
    """Fit the data to the sum of two gaussians using the levenburg-markwardt method

    Args:
        data_pair: given datapair
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Peak locations of two surface responses
    """

    def find_local_min(deriv, start=0, direction=1):
        if direction == 1:  # positive direction
            # find negative
            for i, val in enumerate(deriv[start:]):
                if val <= 0:
                    # print(i,val)
                    break
                else:
                    pass
            for j, val in enumerate(deriv[start + i:]):
                if val > 0:
                    # print(j,val)
                    # check to see if it's just noise
                    if deriv[start + i + j + 1] <= 0:
                        continue
                    else:
                        # print(j,val)
                        break
                else:
                    pass
            return start + i + j
        else:
            if start == 0:
                start = len(deriv)
            else:
                pass
            for i, val in enumerate(reversed(deriv[:start])):
                if val >= 0:
                    # print(i,val)
                    break
                else:
                    pass
            for j, val in enumerate(reversed(deriv[:start - i])):
                if val < 0:
                    # print(j,val)
                    # check to see if it's just noise
                    if deriv[start - 1 - i - j - 1] >= 0:
                        continue
                    else:
                        # print(j,val)
                        break
                else:
                    pass
            return start - i - j

    def find_local_max(deriv, start=0, direction=1):
        if direction == 1:  # positive direction
            # find negative
            for i, val in enumerate(deriv[start:]):
                if val >= 0:
                    # print(i,val)
                    break
                else:
                    pass
            for j, val in enumerate(deriv[start + i:]):
                if val < 0:
                    # print(j,val)
                    # check to see if it's just noise
                    if deriv[start + i + j + 1] >= 0:
                        continue
                    else:
                        # print(j,val)
                        break
                else:
                    pass
            return start + i + j - 1
        else:
            if start == 0:
                start = len(deriv)
                # note that we are using len(deriv) and data[len(deriv)] is out of bounds so indexing will be weird here!
            else:
                pass
            for i, val in enumerate(reversed(deriv[:start])):
                if val <= 0:
                    # print(i,val)
                    break
                else:
                    pass
            for j, val in enumerate(reversed(deriv[:start - i])):
                if val > 0:
                    # print(j,val)
                    # check to see if it's just noise
                    if deriv[start - 1 - i - j - 1] <= 0:
                        continue
                    else:
                        # print(j,val)
                        break
                else:
                    pass
            return start - i - j - 1

    def aveX(data, end, direction=1):
        # end is the index value of the last value in the end of the sequence...inclusive! don't accidentally not include it
        if direction == 1:
            # sum(data[:end+1])/(end+1)
            return np.mean(data[:end + 1])
        else:
            # sum(data[end:])/(len(data)-end)
            return np.mean(data[end:])

    def standard_dev(data, minInd, maxInd, direction=1):
        stdev = []
        for val in [maxInd, minInd]:
            aveVal = aveX(data, val, direction)
            if direction == 1:
                sd = np.sqrt(np.mean(np.square(data[:val + 1] - aveVal)))
            else:
                sd = np.sqrt(np.mean(np.square(data[val:] - aveVal)))
            stdev.append(sd)
        return stdev

    def search(data, multiplier=3, noisy=False):
        deriv = np.gradient(data)
        time_blocks = []
        for direction in [1, -1]:
            start = 0
            searching = True
            while searching:
                local_min = find_local_min(deriv, start, direction)
                if noisy & (direction == 1):
                    local_max = local_min + 5
                elif noisy & (direction == -1):
                    local_max = local_min - 5
                else:
                    local_max = find_local_max(deriv, local_min, direction)
                stdev = standard_dev(data, local_min, local_max, direction)
                if (stdev[0] > multiplier * stdev[1]) & (stdev[0] != 0) & (stdev[1] != 0):
                    # good
                    searching = False
                    break
                    # exit()
                else:
                    start = local_min
                    if (direction == 1) & (start >= len(data) - 1):
                        print("reached the end:", start)
                        exit()
                    elif (direction == -1) & (start <= 1):
                        print("reached the end:", start)
                        exit()
                    else:
                        pass
                        # print(local_min, "...didn't work")
                        # _ = input("%s, %s, %s ...didn't work" % (direction, local_min, local_max))
            print("direction:", direction, local_min, local_max)
            time_blocks.append(local_min)
        return time_blocks

    def fit_gauss(x, mu, sig, A):
        # x numpy array
        y = A * np.exp(-np.square(x - mu) / (2 * np.square(sig)))
        return y

    def fit_trap(wave, a, b, c, d, e, g):
        # assuming the index is also the position
        x = np.arange(len(wave))
        y = np.zeros(shape=x.shape, dtype=float)

        y[a + 1:b + 1] = e * ((x[a + 1:b + 1] - a) / (b - a))
        y[b + 1:c + 1] = np.exp(((np.log(g) - np.log(e)) * x[b + 1:c + 1] + (c * np.log(e) - b * np.log(g))) / (c - b))
        y[c + 1:d + 1] = g * ((d - x[c + 1:d + 1]) / (d - c))

        return y

    def residual(params, *args):
        # print(args, type(args))
        wave = np.array(args, dtype=float)
        # print(type(wave),wave.shape)
        x = np.arange(wave.shape[0])
        y = np.zeros(shape=x.shape, dtype=float)
        parvals = params.valuesdict()

        amplitude_1 = parvals['amp_1']
        center_1 = parvals['cen_1']
        half_width_1 = parvals['wid_1']

        amplitude_2 = parvals['amp_2']
        center_2 = parvals['cen_2']
        half_width_2 = parvals['wid_2']

        a = int(parvals['a'])
        b = int(parvals['b'])
        c = int(parvals['c'])
        d = int(parvals['d'])
        e = parvals['e']
        g = parvals['g']

        # trap
        y[a + 1:b + 1] = e * ((x[a + 1:b + 1] - a) / (b - a))
        y[b + 1:c + 1] = np.exp(((np.log(g) - np.log(e)) * x[b + 1:c + 1] + (c * np.log(e) - b * np.log(g))) / (c - b))
        y[c + 1:d + 1] = g * ((d - x[c + 1:d + 1]) / (d - c))

        # gaus 1
        y += amplitude_1 * np.exp(-np.square(x - center_1) / (2 * np.square(half_width_1)))

        # gaus 2
        y += amplitude_2 * np.exp(-np.square(x - center_2) / (2 * np.square(half_width_2)))

        return wave - y

    def fit(params, wave):
        x = np.arange(len(wave))
        y = np.zeros(shape=x.shape, dtype=float)
        ytrap = np.zeros(shape=x.shape, dtype=float)
        y1 = np.zeros(shape=x.shape, dtype=float)
        y2 = np.zeros(shape=x.shape, dtype=float)
        parvals = params.valuesdict()
        '''
        amp_1=result.params['amp_1'].value
        cen_1=result.params['cen_1'].value
        wid_1=result.params['wid_1'].value
        amp_2=result.params['amp_2'].value
        cen_2=result.params['cen_2'].value
        wid_2=result.params['wid_2'].value,
        a=result.params['a'].value
        b=result.params['b'].value
        c=result.params['c'].value
        d=result.params['d'].value
        e=result.params['e'].value
        g=result.params['g'].value
        '''

        amplitude_1 = parvals['amp_1']
        center_1 = parvals['cen_1']
        half_width_1 = parvals['wid_1']

        amplitude_2 = parvals['amp_2']
        center_2 = parvals['cen_2']
        half_width_2 = parvals['wid_2']

        a = int(parvals['a'])
        b = int(parvals['b'])
        c = int(parvals['c'])
        d = int(parvals['d'])
        e = parvals['e']
        g = parvals['g']

        # trap
        ytrap[a + 1:b + 1] = e * ((x[a + 1:b + 1] - a) / (b - a))
        ytrap[b + 1:c + 1] = np.exp(
            ((np.log(g) - np.log(e)) * x[b + 1:c + 1] + (c * np.log(e) - b * np.log(g))) / (c - b))
        ytrap[c + 1:d + 1] = g * ((d - x[c + 1:d + 1]) / (d - c))

        # gaus 1
        y1 = amplitude_1 * np.exp(-np.square(x - center_1) / (2 * np.square(half_width_1)))

        # gaus 2
        y2 = amplitude_2 * np.exp(-np.square(x - center_2) / (2 * np.square(half_width_2)))

        y = ytrap + y1 + y2
        return y, ytrap, y1, y2


    reflection_buffer = 2000
    get_echo = True
    noisy = False
    withQuad = True

    data_list = []
    for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
        instances = cp.deepcopy(data_set.get_instances())
        for instance in instances:
            if mode is TriState.FEATURES_TO_FEATURES:
                raise ValueError(
                    'No Feature to Features available for my_gaussian_peak'
                )
            elif mode is TriState.STREAM_TO_STREAM:
                data = instance.get_stream().get_data()
            elif mode is TriState.STREAM_TO_FEATURES:
                data = instance.get_stream().get_data()
            X = [x for x in range(len(data[0]))]
            new_data = []
            for row in data:
                try:
                    row = np.array(row, dtype=float)
                    transmitted = row[:reflection_buffer]
                    reflected = data[reflection_buffer:]

                    # Data Pretreatment

                    ## Selection of the Effective Part
                    # def find_effective_region(data):
                    transmit_region = search(transmitted, multiplier=10)
                    reflect_region = search(reflected, multiplier=5)

                    # denoise
                    # Make a parameter
                    noise_sampling_count = 100
                    noise_samples = np.concatenate(
                        [reflected[reflect_region[0] - noise_sampling_count:reflect_region[0]],
                         reflected[reflect_region[1]:reflect_region[1] + noise_sampling_count]])
                    m_noise = np.mean(noise_samples)
                    # Make a parameter
                    lambda_noise = 1.1  # 1.1-1.5
                    threshold_noise = m_noise * lambda_noise
                    reflected[reflected < m_noise] = threshold_noise

                    # smooth
                    # assume halfwidth is the width at half height
                    transmitted_sigma = 0.5
                    halfwidth = transmitted_sigma
                    gauss_sigma = halfwidth / 2.355  # http://hyperphysics.phy-astr.gsu.edu/hbase/Math/gaufcn2.html
                    kernel = [1, 1, 1]  # http://dev.theomader.com/gaussian-kernel-calculator/
                    # gaussian_filter(reflected[reflect_region[0]:reflect_region[1]], sigma=, )
                    smooth_echo = scipy.signal.convolve(input=reflected[reflect_region[0]:reflect_region[1]],
                                           weights=kernel)
                    super_ori_echo = cp.deepcopy(smooth_echo)

                    # if withQuad:
                    init_a = smooth_echo.argmax()  # round( len(smooth_echo)*.10 )
                    init_b = init_a + round(len(smooth_echo) * .05)
                    init_c = round(len(smooth_echo) * .80)
                    init_d = round(len(smooth_echo) * .90)
                    init_e = smooth_echo[init_b]
                    init_g = smooth_echo[init_c]
                    trap = fit_trap(smooth_echo, init_a, init_b, init_c, init_d, init_e, init_g)

                    smooth_echo = smooth_echo - trap

                    # else:
                    # peak detection
                    k = 50.0  # 1.5-3.0 ...empirical value direct proprotional to the strength of peak detection
                    C_threshold = k * m_noise  # only local maxima with amplitude larger than threshold will get selected
                    # smooth_echo = smooth_echo[ smooth_echo>C_threshold ]
                    smooth_echo[smooth_echo < C_threshold] = C_threshold
                    ori_echo = cp.deepcopy(smooth_echo)
                    gaus = []
                    amp = []
                    cen = []
                    wid = []
                    # while smooth_echo.max() > C_threshold:
                    for _ in range(2):
                        x_max1 = smooth_echo.argmax()
                        d = np.gradient(smooth_echo)
                        dd = np.gradient(d)
                        infl_1a = find_local_max(dd, start=x_max1, direction=-1)
                        infl_1b = find_local_min(dd, start=x_max1, direction=1)
                        halfwidth_1 = np.abs(infl_1b - infl_1a) / 2.
                        amplitude_1 = smooth_echo[x_max1]
                        # first gaussian fit
                        amp.append(amplitude_1)
                        cen.append(x_max1)
                        wid.append(halfwidth_1)
                        gaus_1 = fit_gauss(x=np.arange(len(smooth_echo)), mu=x_max1, sig=halfwidth_1, A=amplitude_1)
                        smooth_echo = smooth_echo - gaus_1
                        gaus.append(gaus_1)

                    # minimize
                    params = Parameters()
                    params.add('amp_1', value=amp[0], min=0)
                    params.add('cen_1', value=cen[0], min=0)
                    params.add('wid_1', value=wid[0], min=0)

                    params.add('amp_2', value=amp[1], min=0)
                    params.add('cen_2', value=cen[1], min=0)
                    params.add('wid_2', value=wid[1], min=0)

                    params.add('a', value=init_a, min=0)
                    params.add('b', value=init_b, min=0)
                    params.add('c', value=init_c, min=0)
                    params.add('d', value=init_d, min=0)
                    params.add('e', value=init_e, min=0)
                    params.add('g', value=init_g, min=0)

                    result = minimize(residual, params, args=(super_ori_echo))

                    fit_wave, ytrap, y1, y2 = fit(result.params, super_ori_echo)

                    if mode is TriState.STREAM_TO_STREAM:
                        new_data.append(fit_wave)
                    elif mode is TriState.FEATURES_TO_FEATURES:
                        new_data.append([result.params['cen_1'], result.params['cen_2']])


                except RuntimeError:
                    print("Error - curve_fit failed")
            new_data = np.array(new_data)
            if mode is TriState.FEATURES_TO_FEATURES:
                raise ValueError(
                    'No Feature to Features available for my_progressive_waveform_decomposition'
                )
            elif mode is TriState.STREAM_TO_STREAM:
                instance.get_stream().set_data(new_data)
            elif mode is TriState.STREAM_TO_FEATURES:
                instance.get_features().set_data(new_data, labels=np.array(['cen_1', 'cen_2']))
        new_data_set = EmadeData(instances)
        data_list.append(new_data_set)
    data_pair.set_train_data(data_list[0])
    data_pair.set_test_data(data_list[1])

    gc.collect()
    return data_pair
