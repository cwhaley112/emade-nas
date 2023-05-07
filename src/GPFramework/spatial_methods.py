"""
Programmed by Jason Zutty
Modified by VIP Team
Implements a number of spatial methods for use with deap
"""
import copy as cp
import cv2
import numpy as np
from scipy import ndimage
from scipy import signal
from skimage.morphology import reconstruction, disk
from skimage.util import view_as_blocks
from skimage.filters import rank
from skimage import exposure
import heapq
import math
import time
import os
from functools import wraps
from functools import partial
import gc
import traceback
import sep
import sys

from GPFramework.constants import TriState, Axis, TRI_STATE
from GPFramework.wrapper_methods import RegistryWrapperS, RegistryWrapperB
from GPFramework.data import EmadeDataPair, EmadeData
from GPFramework.cache_methods import check_cache_read, check_cache_write, hash_string

smw = RegistryWrapperS([EmadeDataPair, TriState, Axis])
smw_2 = RegistryWrapperS(2*[EmadeDataPair] + 2*[TriState] + 2*[Axis])
smw_4 = RegistryWrapperS(4*[EmadeDataPair] + 4*[TriState] + 4*[Axis])
smwb = RegistryWrapperB()

'''
Helper method for primitives using kernels
'''
def check_kernel_size(k=5):
    """Checks kernel size

    Args:
        k: size

    Returns:
        positive odd integer suitable for use as an image kernel size
    """
    if k < 0:
        k *= -1
    if k == 0:
        k = 1
    if k % 2 == 0:
        k += 1
    return k

def imshow(title, data_pair):
    """Shows an image

    the following code must appear when you want to display the images
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    Args:
        title: title of the window
        data_pair: given datapair
    """
    dataset_number = 0
    for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
        dataset_number += 1
        instances = cp.deepcopy(data_set.get_instances())
        instance_number = 0
        for instance in instances:
            instance_number += 1
            data = instance.get_stream().get_data()

            title += ' d={0:d} i={1:d}'.format(dataset_number, instance_number)
            cv2.imshow(title, data)

def imshow1(title, data_pair):
    """Shows an image

    the following code must appear when you want to display the images
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    Args:
        title: title of the window
        data_pair: given datapair
    """
    dataset_number = 0
    for data_set in [data_pair.get_train_data()]:
        dataset_number += 1
        instances = cp.deepcopy(data_set.get_instances())
        instance_number = 0
        for instance in instances:
            instance_number += 1
            data = instance.get_stream().get_data()

            cv2.imshow(title, data)

def imsave(filename, data_pair):
    """Saves an image

    Args:
        filename: name of the file
        data_pair: given datapair
    """
    dataset_number = 0
    for data_set in [data_pair.get_train_data()]:
        dataset_number += 1
        instances = cp.deepcopy(data_set.get_instances())
        instance_number = 0
        for instance in instances:
            instance_number += 1
            data = instance.get_stream().get_data()

            cv2.imwrite(filename,data)

def print_info(title, data_pair):
    """Prints information about an image

    Args:
        title: title of the image
        data_pair: given datapair
    """
    dataset_number = 0
    for data_set in [data_pair.get_train_data()]:
        dataset_number += 1
        instances = cp.deepcopy(data_set.get_instances())
        instance_number = 0
        for instance in instances:
            instance_number += 1
            data = instance.get_stream().get_data()

            print('\n\n' + title)
            print('dtype: {0:s}'.format(repr(data.dtype)))
            print('max: {0:f}'.format(data.max()))
            print('min: {0:f}'.format(data.min()))
            print('shape: {0:s}'.format(repr(data.shape)))

def select_nd_helper(data, positions, indices):
    ndims = len(data.shape)
    my_slice = [slice(None)] * ndims
    for pos, ind in zip(positions, indices):
        dim = pos % ndims
        my_slice[dim] = ind % data.shape[dim]
    return data[tuple(my_slice)]

def select_1d_helper(data, pos=0, ind=0):
    return select_nd_helper(data, [pos], [ind])

select_1d = smw.register("Select1D", "test_select_1d", select_1d_helper, None, [int, int], TRI_STATE)
select_1d.__doc__ = """
Slices data on one dimension

Args:
    data: numpy array of example
    pos:  dimension to index on
    ind:  index

Returns:
    Transformed data
"""

def select_2d_helper(data, pos1=0, pos2=0, ind1=0, ind2=0):
    return select_nd_helper(data, [pos1, pos2], [ind1, ind2])

select_2d = smw.register("Select2D", "test_select_2d", select_2d_helper, None, [int, int, int, int], TRI_STATE)
select_2d.__doc__ = """
Slices data on two dimensions

Args:
    data:       numpy array of example
    pos1, pos2: dimension to index on
    ind1, ind2: index

Returns:
    Transformed data
"""

def select_3d_helper(data, pos1=0, pos2=0, pos3=0, ind1=0, ind2=0, ind3=0):
    return select_nd_helper(data, [pos1, pos2, pos3], [ind1, ind2, ind3])

select_3d = smw.register("Select3D", "test_select_3d", select_3d_helper, None, [int, int, int, int, int, int], TRI_STATE)
select_3d.__doc__ = """
Slices data on three dimensions

Args:
    data:             numpy array of example
    pos1, pos2, pos3: dimension to index on
    ind1, ind2, ind3: index

Returns:
    Transformed data
"""

def convert_bw_helper(data):
    return cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

convert_bw = smw.register("ConvertBW", "test_convert_bw", convert_bw_helper, None, [], TRI_STATE)
convert_bw.__doc__ = """
Converts a RGB color image to Black&White

Args:
    data: numpy array of example

Returns:
    Transformed data
"""

def background_subtraction_helper(data):
    bkg = sep.Background(data.astype(np.float64))
    return data - bkg

background_subtraction = smw.register("BackgroundSubtraction", "test_background_subtraction", background_subtraction_helper, None, [], TRI_STATE)
background_subtraction.__doc__ = """
Removes background noise on an image
https://sep.readthedocs.io/en/v1.0.x/tutorial.html

Args:
    data: numpy array of example

Returns:
    Transformed data
"""

def gradient_magnitude_helper(data, ksize=1):
    flag = False
    try:
        # check if data is iterable
        iter(data)
        flag = True
    except:
        pass

    # Check if the iterable is empty
    if not (flag and len(data) > 0 and len([i for i in np.array(data).shape if i == 0]) == 0):
        raise ValueError("Data array cannot be empty.")

    img = np.float32(data) / 255.0
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=ksize)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    return mag

gradient_magnitude = smw.register("GradientMagnitude", "test_gradient_magnitude", gradient_magnitude_helper, None, [int], TRI_STATE)
gradient_magnitude.__doc__ = """
Calculates the gradient magnitude of an image
https://www.learnopencv.com/histogram-of-oriented-gradients/

Args:
    data (np.ndarray): numpy array of example
    ksize (int):       kernel_size

Returns:
    Transformed data
"""

def gradient_weighted_helper(data, ksize=1, x_weight=0.5, y_weight=0.5):
    gx = cv2.Sobel(np.float32(data), cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(np.float32(data), cv2.CV_32F, 0, 1, ksize=ksize)
    g = cv2.addWeighted(np.abs(gx), x_weight, np.abs(gy), y_weight, 0)
    return g

gradient_weighted = smw.register("GradientWeighted", "test_gradient_weighted", gradient_weighted_helper, None, [int, float, float], TRI_STATE)
gradient_weighted.__doc__ = """
Calculates the image gradient using a Sobel operator

Args:
    data (np.ndarray): numpy array of example
    ksize (int):       kernel_size
    x_weight (float):  weight of x axis gradient
    y_weight (float):  weight of y axis gradient

Returns:
    Transformed data
"""

def image_peak_finder_helper(data):
    seed = np.copy(data)
    seed[1:-1, 1:-1] = data.min()
    return reconstruction(seed, data, method='dilation')

image_peak_finder = smw.register("ImagePeakFinder", "test_image_peak_finder", image_peak_finder_helper, None, [], TRI_STATE)
image_peak_finder.__doc__ = """
Perform a morphological reconstruction of an image.
https://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.reconstruction

Args:
    data: numpy array of example

Returns:
    Transformed data
"""

def regional_maxima_helper(data, h=0.4):
    seed = np.copy(data).astype(np.float64)
    seed -= h
    return reconstruction(seed, data, method='dilation')

regional_maxima = smw.register("RegionalMaxima", "test_regional_maxima", regional_maxima_helper, None, [float], TRI_STATE)
regional_maxima.__doc__ = """
Finds the regional maxima of an image. Tends to isolate regional maxmia of height h
https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_regional_maxima.html#sphx-glr-auto-examples-color-exposure-plot-regional-maxima-py

Args:
    data: numpy array of example
    h: fixed value used to create the seed

Returns:
    Transformed data
"""

def otsu_binary_threshold_helper(data, max_value=255):
    ret, thresh = cv2.threshold(data, 0, max_value, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return thresh

otsu_binary_threshold = smw.register("OtsuBinaryThreshold", "test_otsu_binary_threshold", otsu_binary_threshold_helper, None, [int], TRI_STATE)
otsu_binary_threshold.__doc__ = """
Otsu's binarization (binary threshold)
Thresholds all values to either zero or max_value

Args:
    data:      numpy array of example
    max_value: maximum value after threshold is performed

Returns:
    Transformed data
"""

def local_pooling_helper(data, bsize=4, method=None):
    # image as a matrix of blocks (of shape (bsize, bsize))
    view = view_as_blocks(data, (bsize, bsize))
    # collapse the last two dimensions in one
    flatten_view = view.reshape(view.shape[0], view.shape[1], -1)

    # resampling the image by taking either the `mean`,
    # the `max` or the `median` value of each blocks.
    return method(flatten_view, axis=2)

def local_pool_mean_setup(data, bsize=4):
    return data, { "bsize":bsize, "method":np.mean }

local_pool_mean = smw.register("LocalPoolingMean", "test_local_pool_mean", local_pooling_helper, local_pool_mean_setup, [int], TRI_STATE)
local_pool_mean.__doc__ = """
Pools mean of square blocks in an image

Args:
    data:  numpy array of example
    bsize: block size

Returns:
    Transformed data
"""

def local_pool_max_setup(data, bsize=4):
    return data, { "bsize":bsize, "method":np.max }

local_pool_max = smw.register("LocalPoolingMax", "test_local_pool_max", local_pooling_helper, local_pool_max_setup, [int], TRI_STATE)
local_pool_max.__doc__ = """
Pools max of square blocks in an image

Args:
    data:  numpy array of example
    bsize: block size

Returns:
    Transformed data
"""

def local_pool_median_setup(data, bsize=4):
    return data, { "bsize":bsize, "method":np.median }

local_pool_median = smw.register("LocalPoolingMedian", "test_local_pool_median", local_pooling_helper, local_pool_median_setup, [int], TRI_STATE)
local_pool_median.__doc__ = """
Pools median of square blocks in an image

Args:
    data:  numpy array of example
    bsize: block size

Returns:
    Transformed data
"""

def adjust_contrast_gamma_helper(data, gamma=2.0, gain=1.0):
    return exposure.adjust_gamma(data, gamma, gain)

adjust_contrast_gamma = smw.register("AdjustConstrastGamma", "test_adjust_contrast_gamma", adjust_contrast_gamma_helper, None, [float, float], TRI_STATE)
adjust_contrast_gamma.__doc__ = """
Adjusts image contrast by performing a Gamma correction

Args:
    data:  numpy array of example
    gamma: non-negative real number
    gain:  constant multiplier

Returns:
    Transformed data
"""

def adjust_contrast_log_helper(data, gain=1.0):
    return exposure.adjust_log(data, gain)

adjust_contrast_log = smw.register("AdjustConstrastLog", "test_adjust_contrast_log", adjust_contrast_log_helper, None, [float], TRI_STATE)
adjust_contrast_log.__doc__ = """
Adjusts image contrast by performing a Logarithmic correction

Args:
    data:  numpy array of example
    gain:  constant multiplier

Returns:
    Transformed data
"""

def equalize_hist_helper(data):
    return exposure.equalize_hist(data)

equalize_hist = smw.register("EqualizeHist", "test_equalize_hist", equalize_hist_helper, None, [], TRI_STATE)
equalize_hist.__doc__ = """
Local contrast enhancement using global histogram equalization

Args:
    data:  numpy array of example
    gain:  constant multiplier

Returns:
    Transformed data
"""

def equalize_adapthist_helper(data, clip_limit=0.01):
    return exposure.equalize_adapthist(zero_one_norm_helper(data))

equalize_adapthist = smw.register("EqualizeAdaptHist", "test_equalize_adapthist", equalize_adapthist_helper, None, [float], TRI_STATE)
equalize_adapthist.__doc__ = """
Local contrast enhancement using adaptive histogram equalization

Args:
    data:       numpy array of example
    clip_limit: clipping limit, normalized between 0 and 1 (higher values give more contrast)

Returns:
    Transformed data
"""

def equalize_disk_helper(data, dsize=30):
    selem = disk(dsize)
    return rank.equalize(zero_one_norm_helper(data), selem=selem)

equalize_disk = smw.register("EqualizeDisk", "test_equalize_disk", equalize_disk_helper, None, [int], TRI_STATE)
equalize_disk.__doc__ = """
Local contrast enhancement using local histogram equalization

Args:
    data:  numpy array of example
    dsize: size of the neighborhood used around one pixel

Returns:
    Transformed data
"""

def mean_filter_helper(data, dsize=3):
    selem = disk(dsize)
    return rank.mean(data, selem=selem)

mean_filter = smw.register("MeanFilter", "test_mean_filter", mean_filter_helper, None, [int], TRI_STATE)
mean_filter.__doc__ = """
Smooths an image by finding the local mean

Args:
    data:  numpy array of example
    dsize: size of the neighborhood used around one pixel

Returns:
    Transformed data
"""

def zero_one_norm_helper(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

zero_one_norm = smw.register("ZeroOneNorm", "test_zero_one_norm", zero_one_norm_helper, None, [], TRI_STATE)
zero_one_norm.__doc__ = """
Normalizes data to between [0,1]

Args:
    data: numpy array of example

Returns:
    Transformed data
"""

def standard_norm_helper(data):
    return (data - np.mean(data)) / np.std(data)

standard_norm = smw.register("StandardNorm", "test_standard_norm", standard_norm_helper, None, [], TRI_STATE)
standard_norm.__doc__ = """
Normalizes data to have mean 0 and std 1

Args:
    data: numpy array of example

Returns:
    Transformed data
"""

def convert_to_counts_helper(data):
    # First get number of unique values
    uniques = np.unique(data)
    # Now we can get the diffs to find the smallest delta
    delta = np.diff(uniques).min()
    # Baseline at 0
    new_data = data - np.min(data)
    new_data = np.floor(new_data/delta)
    return new_data

convert_to_counts = smw.register("ConvertToCounts", "test_convert_to_counts", convert_to_counts_helper, None, [], TRI_STATE)
convert_to_counts.__doc__ = """
This method converts frames to counts based on a smallest delta in values
We use this because a number of opencv methods require integer inputs.

Args:
    data (list of np.ndarray):  list of numpy arrays

Returns:
    Updated video of image frames
"""

def accumulate_weighted_helper(data, alpha=0.01):
    avg = cp.deepcopy(data[0])

    # augment the moving average
    new_data = []
    for i in range(1, len(data)):
        avg = cv2.accumulateWeighted(data[i].astype(np.float32), avg.astype(np.float32), alpha=alpha)
        new_data.append(avg)

    return np.array(new_data)

accumulate_weighted = smw.register("AccumulateWeighted", "test_accumulate_weighted", accumulate_weighted_helper, None, [float], TRI_STATE)
accumulate_weighted.__doc__ = """
Updates a running average over every frame of a video

Args:
    data (list of np.ndarray):  list of numpy arrays
    alpha (float):              weight of current frame vs history of frames (0.01 would weight new frame as 1% and history as 99%)

Returns:
    Updated video of image frames
"""

def image_alignment_ecc_setup(data, second_data, number_of_iterations=50, termination_eps=1e-10, method=0, gfiltsize=3):
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    methods = [cv2.MOTION_HOMOGRAPHY, cv2.MOTION_AFFINE,
               cv2.MOTION_EUCLIDEAN, cv2.MOTION_TRANSLATION]
    i = np.abs(method) % len(methods)
    warp_mode = methods[i]

    if i > 0:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(3, 3, dtype=np.float32)

    if isinstance(data, list):
        sz = data[0].shape
    elif isinstance(data, np.ndarray) and len(data.shape) == 3:
        sz = data.shape[:2]
    else:
        sz = data.shape

    if gfiltsize < 0:
        gfiltsize = np.abs(gfiltsize)

    return data, second_data, { "criteria":criteria, "warp_mode":warp_mode, "warp_matrix":warp_matrix, "sz":sz, "gfiltsize":gfiltsize }

def image_alignment_ecc_helper(data, second_data, criteria=None, warp_mode=None, warp_matrix=None, sz=None, gfiltsize=3):
    bits = np.ceil(np.log2(second_data.max()))
    new_data = []
    for frame, counts in zip(data, second_data):
        try:
            (cc, warp_matrix) = cv2.findTransformECC((second_data[0]/2**bits*256).astype(np.uint8),(counts/2**bits*256).astype(np.uint8), warp_matrix, warp_mode, criteria, None, gfiltsize)
            new_data.append(cv2.warpPerspective(frame, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP))
        except cv2.error as e:
            # did not converge so return original image frame
            new_data.append(frame)

    return np.array(new_data)

image_alignment_ecc = smw_2.register("ImageAlignmentECC", "test_image_alignment_ecc", image_alignment_ecc_helper, image_alignment_ecc_setup, [int, float, int, int], TRI_STATE)
image_alignment_ecc.__doc__ = """
Aligns the current frame to frame 0

Args:
    data (list of np.ndarray):         list of numpy arrays
    second_data (list of np.ndarray):  list of numpy arrays (This method expects this data to be converted to counts)
    number_of_iterations (int):        number of iterations before termination
    termination_eps (float):           termination threshold
    method (int):                      index for which warp mode to use
    gfiltsize (int):                   n x n kernel size for Gaussian filter

Returns:
    Updated video of image frames
"""

def spectral_filter(spatial_primary_pair, spatial_reference_pair, pr_primary_pair, pr_reference_pair):
    dividend = signal.correlate2d(pr_primary_pair, pr_reference_pair, 'same')
    divisor = signal.correlate2d(pr_reference_pair, pr_reference_pair, 'same')
    alpha = cv2.divide(dividend, divisor)

    subtrahend = cv2.multiply(alpha, spatial_reference_pair)
    spectral = cv2.subtract(spatial_primary_pair, subtrahend)
    return spectral


spectral_filter = smw_4.register("SpectralFilter", "test_spectral_filter", spectral_filter, None, [], TRI_STATE)
spectral_filter.__doc__ = """
Spectral filter images for a dual color algorithm

S(x) = spatially filtered image
P(x) = Pre Rejection filtered image
P = Primary image
R = Reference image

a = CORR(P(P), P(R)) / CORR(P(R),P(R))

out = S(P) - aS(R)

Args:
    spatial_primary_pair: Spatially filtered primary image data pair
    spectral_reference_pair: Spatially filtered reference image data pair
    pr_primary_pair: Pre Rejection filtered Primary image data pair
    pr_reference_pair: Pre Rejection filtered reference image data pair

Returns:
    Transformed data pair
"""


def minimum_to_zero_helper(data):
    m = int(data.min())
    if m > 0:
        return cv2.subtract(data, m)
    else:
        return cv2.add(data, -1 * m)

minimum_to_zero = smw.register("MinimumToZero", "test_minimum_to_zero", minimum_to_zero_helper, None, [], TRI_STATE)
minimum_to_zero.__doc__ = """
Shift the entire image such that the minimum value is zero

Args:
    data: numpy array of example

Returns:
    Transformed data
"""

def to_uint8_helper(data):
    norm = (data - data.min()) / (data.max() - data.min())
    return (norm * 255).astype(np.uint8)

to_uint8 = smw.register("ToUint8", "test_to_uint8", to_uint8_helper, None, [], TRI_STATE)
to_uint8.__doc__ = """
Convert to uint8

Args:
    data: numpy array of example

Returns:
    Transformed data
"""

def run_center_of_mass(data):
    estimated_center_of_mass = ndimage.measurements.center_of_mass(data)
    return np.array([estimated_center_of_mass])

center_of_mass = smw.register("CenterOfMass", "test_center_of_mass", run_center_of_mass, None, [], TRI_STATE)
center_of_mass.__doc__ = """
Use scipy.ndimage to consume an image and return the x,y location of its
center of mass

Args:
    data: numpy array of example

Returns:
    Centroided data
"""

def snapshot_helper(coord_data, image, precision=5):
    p = precision
    # expecting array in this format: [[x, y]]
    x, y = coord_data[0].astype(np.uint8)
    # create blank image of 0s
    blank = np.zeros(image.shape)
    # set selected region on blank image
    blank[x-p:x+p,y-p:y+p] = image[x-p:x+p,y-p:y+p]
    return blank

my_snapshot = smw_2.register("MySnapshot", "test_snapshot", snapshot_helper, None, [int], TRI_STATE)
my_snapshot.__doc__ = """
Saves a snapshot of an image onto a blank image

Args:
    coord_data: numpy array containing (x,y) of target location
    image: numpy array of image to pull snapshot from
    precision: how large the snapshot is in NxN.
               correlates to precision of center_of_mass

Returns:
    image snapshot
"""

def to_uint8_scale_setup(data):
    minval = data.min()
    maxval = data.max()
    alpha = 255.0/(maxval-minval)
    beta = -1*minval*255/(maxval-minval)
    return data, { "alpha":alpha, "beta":beta }

def to_uint8_scale_helper(data, alpha=None, beta=None):
    return cv2.convertScaleAbs(data, alpha=alpha, beta=beta)

to_uint8_scale = smw.register("ToUint8Scale", "test_to_uint8_scale", to_uint8_scale_helper, to_uint8_scale_setup, [], TRI_STATE)
to_uint8_scale.__doc__ = """
Scale the image values from 0 to 256, image returned as a uint8

Args:
    data: numpy array of example

Returns:
    Transformed data
"""

def to_float_helper(data):
    return data.astype(np.float)

to_float = smw.register("ToFloat", "test_to_float", to_float_helper, None, [], TRI_STATE)
to_float.__doc__ = """
Convert to float

Args:
    data: numpy array of example

Returns:
    Transformed data
"""

def to_float_normalize_helper(data):
    data = data.astype(np.float)
    return data / data.max()

to_float_normalize = smw.register("ToFloatNorm", "test_to_float_normalize", to_float_normalize_helper, None, [], TRI_STATE)
to_float_normalize.__doc__ = """
Convert to float and normalize

Args:
    data: numpy array of example

Returns:
    Transformed data
"""

# Edge Detection

def canny_setup(data, t1=50, t2=150, apertureSize=3):
    if t1 < 0:
        t1 *= -1
    if t2 < 0:
        t2 *= -1
    x = [t1, t2]
    t1 = min(x)
    t2 = max(x)
    return data, { "t1":t1, "t2":t2, "apertureSize":apertureSize }

def edge_detection_canny_helper(data, t1=50, t2=150, apertureSize=3):
    return cv2.Canny(data, t1, t2, apertureSize=apertureSize, L2gradient=True)

edge_detection_canny = smw.register("EdgeDetectionCanny", "test_edge_detection_canny", edge_detection_canny_helper, canny_setup, [int, int, int], TRI_STATE)
edge_detection_canny.__doc__ = """
Perform the Canny edge detection algorithm on the image using the default sobel window size of three and the
higher accuracy gradient estimation technique

http://docs.opencv.org/trunk/da/d22/tutorial_py_canny.html#gsc.tab=0

Edges > t2 are in the output image
Edges > t1 that touch something > t2 are in the image
Edges < t1 are not in the image

Ratio of t2:t1 should be between 2:1 and 3:1

Args:
    data: numpy array of example
    t1: Lower edge threshold
    t2: Upper edge threshold
    apertureSize: Aperture size of the Sobel gradient function

Returns:
    Transformed data
"""

def detection_setup(data, blockSize=10, kernel_size=5, k=0.04):
    kernel_size = check_kernel_size(kernel_size)
    if blockSize < 0:
        blockSize *= -1
    return data, { "blockSize":blockSize, "ksize":kernel_size, "k":k }

def corner_detection_harris_helper(data, blockSize=10, ksize=5, k=0.04):
    return cv2.cornerHarris(data, blockSize=blockSize, ksize=ksize, k=k)

corner_detection_harris = smw.register("CornerDetectionHarris", "test_corner_detection_harris", corner_detection_harris_helper, detection_setup, [int, int, float], TRI_STATE)
corner_detection_harris.__doc__ = """
Perform Harris corner detection

Args:
    data: numpy array of example
    blocksize: Area on which to construct local gradient
    kernel_size: kernel size for Sobel
    k: Harris detector free parameter

Returns:
    Transformed data, non-normalized float, some values may be negative
"""

def corner_detection_min_eigen_val_helper(data, blockSize=10, ksize=5, k=None):
    return cv2.cornerMinEigenVal(data, blockSize, ksize=ksize)

corner_detection_min_eigen_val = smw.register("CornerDetectionEigen", "test_corner_detection_min_eigen_val", corner_detection_min_eigen_val_helper, detection_setup, [int, int, float], TRI_STATE)
corner_detection_min_eigen_val.__doc__ = """
Perform minimum eigenvalue corner detection

Args:
    data: numpy array of example
    blocksize: Area on which to construct local gradient
    kernel_size: kernel size for Sobel

Returns:
    Transformed data, non-normalized float
"""

# High Pass Filters

def highpass_fourier_ellipsoid_helper(data, size=3):
    if size < 0:
        size *= -1
    org = np.copy(data)
    data = np.fft.fft2(data)
    data = ndimage.fourier_ellipsoid(data, size)
    data = np.fft.ifft2(data)
    data = np.absolute(data)
    data = org - data
    return data

highpass_fourier_ellipsoid = smw.register("HighpassFourierEllipsoid", "test_highpass_fourier_ellipsoid", highpass_fourier_ellipsoid_helper, None, [int], TRI_STATE)
highpass_fourier_ellipsoid.__doc__ = """
Perform an ellipsoid low pass filter in the frequency domain and subtract that from the original image

Args:
    data: numpy array of example
    size: size of the box used for filtering

Returns:
    Transformed data, non-normalized float, some values may be negative
"""

def highpass_irst_setup(data):
    kernel = np.zeros((3,3), np.int)
    kernel[0,1] = -1
    kernel[1,0] = -1
    kernel[1,1] = 5
    kernel[1,2] = -1
    kernel[2,1] = -1
    return data, { "kernel":kernel }

def highpass_irst_helper(data, kernel=None):
    flag = False
    try:
        # check if data is iterable
        iter(data)
        flag = True
    except:
        pass

    # Check if the iterable is empty
    if not (flag and len(data) > 0 and len([i for i in np.array(data).shape if i == 0]) == 0):
        raise ValueError("Data array cannot be empty.")

    return cv2.filter2D(data, -1, kernel)

highpass_irst = smw.register("HighpassIrst", "test_highpass_irst", highpass_irst_helper, highpass_irst_setup, [], TRI_STATE)
highpass_irst.__doc__ = """
Implements the HPF used in the IRST algorithm

Args:
    data: numpy array of example

Returns:
    Transformed data, non-normalized float, some values may be negative
"""

def median_filter_helper(data, kernel_size=3):
    kernel_size = check_kernel_size(kernel_size)
    return signal.medfilt2d(data, kernel_size)

median_filter = smw.register("MedianFilter", "test_median_filter", median_filter_helper, None, [int], TRI_STATE)
median_filter.__doc__ = """
Median filter

Args:
    data: numpy array of example
    kernel_size: size of kernel for std deviation

Returns:
    Transformed data
"""

def lowpass_fourier_shift_helper(data, shift=3):
    if shift < 0:
        shift *= -1
    data = np.fft.fft2(data)
    data = ndimage.fourier_shift(data, shift)
    data = np.fft.ifft2(data)
    data = np.absolute(data)
    return data

lowpass_fourier_shift = smw.register("LowpassFourierShift", "test_lowpass_fourier_shift", lowpass_fourier_shift_helper, None, [int], TRI_STATE)
lowpass_fourier_shift.__doc__ = """
Perform a shift low pass filter in the frequency domain

Args:
    data: numpy array of example
    shift: the size of the box used for filtering

Returns:
    Transformed data
"""

def highpass_fourier_shift_helper(data, shift=3):
    if shift < 0:
        shift *= -1
    org = np.copy(data)
    data = np.fft.fft2(data)
    data = ndimage.fourier_shift(data, shift)
    data = np.fft.ifft2(data)
    data = np.absolute(data)
    data = org - data
    return data

highpass_fourier_shift = smw.register("HighpassFourierShift", "test_highpass_fourier_shift", highpass_fourier_shift_helper, None, [int], TRI_STATE)
highpass_fourier_shift.__doc__ = """
Perform a shift low pass filter in the frequency domain subtracted
from the original image to produce a high pass filtered image

Args:
    data: numpy array of example
    shift: the size of the box used for filtering

Returns:
    Transformed data
"""

def highpass_fourier_gaussian_helper(data, sigma=3):
    if sigma < 0:
        sigma *= -1
    org = np.copy(data)
    data = np.fft.fft2(data)
    data = ndimage.fourier_gaussian(data, sigma)
    data = np.fft.ifft2(data)
    data = np.absolute(data)
    data = org - data
    return data

highpass_fourier_gaussian = smw.register("HighpassFourierGaussian", "test_highpass_fourier_gaussian", highpass_fourier_gaussian_helper, None, [int], TRI_STATE)
highpass_fourier_gaussian.__doc__ = """
Perform a gaussian low pass filter in the frequency domain and subtract
from the original image

Args:
    data: numpy array of example
    sigma: the sigma of the Gaussian kernel

Returns:
    Transformed data
"""

def highpass_fourier_uniform_helper(data, size=3):
    if size < 0:
        size *= -1
    org = np.copy(data)
    data = np.fft.fft2(data)
    data = ndimage.fourier_uniform(data, size)
    data = np.fft.ifft2(data)
    data = np.absolute(data)
    data = org - data
    return data

highpass_fourier_uniform = smw.register("HighpassFourierUniform", "test_highpass_fourier_uniform", highpass_fourier_uniform_helper, None, [int], TRI_STATE)
highpass_fourier_uniform.__doc__ = """
Perform an uniform low pass filter in the frequency domain and
subtract this from the original image

Args:
    data: numpy array of example
    size: the size of the box used for filtering

Returns:
    Transformed data
"""

def highpass_unsharp_mask_setup(data, kernel_size=9, sigma=10, weight=10):
    if sigma == 0:
        sigma = 1
    if sigma < 0:
        sigma *= -1
    if weight < 0:
        weight *= -1
    if weight < 1:
        weight = 1
    alpha = weight
    beta = -1 * (alpha - 1)
    kernel_size = check_kernel_size(kernel_size)
    return data, { "kernel_size":kernel_size, "sigma":sigma, "alpha":alpha, "beta":beta }

def highpass_unsharp_mask_helper(data, kernel_size=9, sigma=10, alpha=None, beta=None):
    flag = False
    try:
        # check if data is iterable
        iter(data)
        flag = True
    except:
        pass

    # Check if the iterable is empty
    if not (flag and len(data) > 0 and len([i for i in np.array(data).shape if i == 0]) == 0):
        raise ValueError("Data array cannot be empty.")
    
    gaussian = cv2.GaussianBlur(data, (kernel_size, kernel_size), sigma)
    data = cv2.addWeighted(data, alpha, gaussian, beta, gamma=0)
    return data

highpass_unsharp_mask = smw.register("HighpassUnsharpMask", "test_highpass_unsharp_mask", highpass_unsharp_mask_helper, highpass_unsharp_mask_setup, [int, int, int], TRI_STATE)
highpass_unsharp_mask.__doc__ = """
Unsharp mask, high pass filter

Args:
    data: numpy array of example
    kernel_size: Positive odd integer > 1.  Size of the kernel used in the
    gaussian filtering step
    sigma: the standard deviation to use in the gaussian filtering step
    weight: weight used to emphasize changes in image

Returns:
    Transformed data
"""

def highpass_laplacian_setup(data, kernel_size=3, scale=1, delta=0):
    if scale < 0:
        scale *= -1
    if scale == 0:
        scale = 1
    kernel_size = check_kernel_size(kernel_size)
    return data, { "kernel_size":kernel_size, "scale":scale, "delta":delta }

def highpass_laplacian_helper(data, kernel_size=3, scale=1, delta=0):
    return cv2.Laplacian(data, ddepth=-1, dst=data, ksize=kernel_size, scale=scale, delta=delta)

highpass_laplacian = smw.register("HighpassLaplacian", "test_highpass_laplacian", highpass_laplacian_helper, highpass_laplacian_setup, [int, int, int], TRI_STATE)
highpass_laplacian.__doc__ = """
Laplacian of an image

Args:
    data: numpy array of example
    kernel_size: Aperture size used to compute the second-derivative filters
    scale: Optional scale factor for the computed Laplacian values.
    By default, no scaling is applied
    delta: Optional delta value that is added to the results

Returns:
    Transformed data
"""

def highpass_sobel_derivative_setup(data, dx=1, dy=1, ksize=5, scale=1, delta=0):
    if scale < 0:
        scale *= -1
    if scale == 0:
        scale = 1
    derivative_order = [1, 2, 3]
    dx = derivative_order[dx % len(derivative_order)]
    dy = derivative_order[dy % len(derivative_order)]
    ksize = check_kernel_size(ksize)
    sizes = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    ksize = sizes[ksize % len(sizes)]
    if dx > dy:
        order = dx
    else:
        order = dy
    if order > ksize:
        if ksize == 1 and order == 3:
            ksize = 5
        else:
            ksize += 2
    return data, { "dx":dx, "dy":dy, "ksize":ksize, "scale":scale, "delta":delta }

def highpass_sobel_derivative_helper(data, dx=1, dy=1, ksize=5, scale=1, delta=0):
    flag = False
    try:
        # check if data is iterable
        iter(data)
        flag = True
    except:
        pass

    # Check if the iterable is empty
    if not (flag and len(data) > 0 and len([i for i in np.array(data).shape if i == 0]) == 0):
        raise ValueError("Data array cannot be empty.")

    return cv2.Sobel(data, ddepth=-1, dx=dx, dy=dy, dst=data, ksize=ksize, scale=scale, delta=delta)

highpass_sobel_derivative = smw.register("HighpassSobelDerivative", "test_highpass_sobel_derivative", highpass_sobel_derivative_helper, highpass_sobel_derivative_setup, [int, int, int, int, int], TRI_STATE)
highpass_sobel_derivative.__doc__ = """
Sobel Derivative
Kernel size has to be greater than the derivative order

Args:
    data: numpy array of example
    dx: first, second, or third derivative [1,2,3]
    dy: first, second, or third derivative [1,2,3]
    ksize: kernel size
    scale: scale factor to be applied to final image
    delta: additive factor to final image

Returns:
    Transformed data
"""

# Low Pass Filters

def lowpass_filter_median_helper(data, filter_size=5):
    return cv2.medianBlur(data, filter_size)

lowpass_filter_median = smw.register("LowpassFilterMedian", "test_lowpass_filter_median", lowpass_filter_median_helper, None, [int], TRI_STATE)
lowpass_filter_median.__doc__ = """
Performs a spatial median lowpass filter on Stream Data

Args:
    data: numpy array of example
    filter_size: must be a positive odd integer > 1

Returns:
    Transformed data
"""

def lowpass_filter_average_helper(data, kernel_size=5):
    flag = False
    try:
        # check if data is iterable
        iter(data)
        flag = True
    except:
        pass

    # Check if the iterable is empty
    if not (flag and len(data) > 0 and len([i for i in np.array(data).shape if i == 0]) == 0):
        raise ValueError("Data array cannot be empty.")
    
    kernel_size = check_kernel_size(kernel_size)
    return cv2.blur(data, (kernel_size, kernel_size), data)

lowpass_filter_average = smw.register("LowpassFilterAverage", "test_lowpass_filter_average", lowpass_filter_average_helper, None, [int], TRI_STATE)
lowpass_filter_average.__doc__ = """
Normalized average low pass filter

Args:
    data: numpy array of example
    kernel_size: kernel size

Returns:
    Transformed data
"""

def lowpass_filter_gaussian_setup(data, kernel_size_x=3, kernel_size_y=3, sigma_x=0.5, sigma_y=0.5):
    if sigma_x < 0:
        sigma_x *= -1
    if sigma_y < 0:
        sigma_y *= -1
    kernel_size_x = check_kernel_size(kernel_size_x)
    kernel_size_y = check_kernel_size(kernel_size_y)
    return data, { "kernel":(kernel_size_x, kernel_size_y), "sigma_x":sigma_x, "sigma_y":sigma_y }

def lowpass_filter_gaussian_helper(data, kernel=None, sigma_x=0.5, sigma_y=0.5):
    flag = False
    try:
        # check if data is iterable
        iter(data)
        flag = True
    except:
        pass

    # Check if the iterable is empty
    if not (flag and len(data) > 0 and len([i for i in np.array(data).shape if i == 0]) == 0):
        raise ValueError("Data array cannot be empty.")
    
    return cv2.GaussianBlur(data, kernel, sigma_x, sigma_y)

lowpass_filter_gaussian = smw.register("LowpassFilterGaussian", "test_lowpass_filter_gaussian", lowpass_filter_gaussian_helper, lowpass_filter_gaussian_setup, [int, int, float, float], TRI_STATE)
lowpass_filter_gaussian.__doc__ = """
Performs a Gaussian low pass filter

Args:
    data: numpy array of example
    kernel_size_x: odd integer > 1
    kernel_size_y: odd integer > 1
    sigma_x: positive integer
    sigma_y: positive integer

Returns:
    Transformed data
"""

def lowpass_filter_bilateral_helper(data, filter_diameter=9, sigma_color=75, sigma_space=75):
    if sigma_color < 0:
        sigma_color *= -1
    if sigma_color == 0:
        sigma_color = 1
    if sigma_space < 0:
        sigma_space *= -1
    if sigma_space == 0:
        sigma_space = 1
    filter_diameter = check_kernel_size(filter_diameter)
    return cv2.bilateralFilter(data, filter_diameter, sigma_color, sigma_space)

lowpass_filter_bilateral = smw.register("LowpassFilterBilateral", "test_lowpass_filter_bilateral", lowpass_filter_bilateral_helper, None, [int, int, int], TRI_STATE)
lowpass_filter_bilateral.__doc__ = """
Performs a bilateral low pass filter

Args:
    data: numpy array of example

    filter_diameter: Diameter of each pixel neighborhood that is used
    during filtering. If it is non-positive, it is computed from sigma_space

    sigma_color: Filter sigma in the color space. A larger value of the
    parameter means that farther colors within the pixel
    neighborhood (see sigmaSpace ) will be mixed together, resulting in
    larger areas of semi-equal color.

    sigma_space: Filter sigma in the coordinate space. A larger value of
    the parameter means that farther pixels will influence each other as
    long as their colors are close enough (see sigmaColor ). When d>0, it
    specifies the neighborhood size regardless of sigmaSpace . Otherwise,
    d is proportional to sigmaSpace

Returns:
    Transformed data
"""

def lowpass_fourier_ellipsoid_helper(data, size=3):
    if size < 0:
        size *= -1
    data = np.fft.fft2(data)
    data = ndimage.fourier_ellipsoid(data, size)
    data = np.fft.ifft2(data)
    data = np.absolute(data)
    return data

lowpass_fourier_ellipsoid = smw.register("LowpassFilterEllipsoid", "test_lowpass_fourier_ellipsoid", lowpass_fourier_ellipsoid_helper, None, [int], TRI_STATE)
lowpass_fourier_ellipsoid.__doc__ = """
Performs an ellipsoid low pass filter in the frequency domain

Args:
    data: numpy array of example
    size: the size of the box used for filtering

Returns:
    Transformed data
"""

def lowpass_fourier_gaussian_helper(data, sigma=3):
    if sigma < 0:
        sigma *= -1
    data = np.fft.fft2(data)
    data = ndimage.fourier_gaussian(data, sigma)
    data = np.fft.ifft2(data)
    data = np.absolute(data)
    return data

lowpass_fourier_gaussian = smw.register("LowpassFourierGaussian", "test_lowpass_fourier_gaussian", lowpass_fourier_gaussian_helper, None, [int], TRI_STATE)
lowpass_fourier_gaussian.__doc__ = """
Perform a Gaussian low pass filter in the frequency domain
Args:
    data: numpy array of example
    sigma: the sigma of the Gaussian kernel

Returns:
    Transformed data
"""

def lowpass_fourier_uniform_helper(data, size=3):
    if size < 0:
        size *= -1
    data = np.fft.fft2(data)
    data = ndimage.fourier_uniform(data, size)
    data = np.fft.ifft2(data)
    data = np.absolute(data)
    return data

lowpass_fourier_uniform = smw.register("LowpassFilterUniform", "test_lowpass_fourier_uniform", lowpass_fourier_uniform_helper, None, [int], TRI_STATE)
lowpass_fourier_uniform.__doc__ = """
Performs an uniform low pass filter in the frequency domain

Args:
    data: numpy array of example
    size: the size of the box used for filtering

Returns:
    Transformed data
"""

# Threshold functions

def threshold_binary_helper(data, threshold=128, maxvalue=255):
    if threshold < 0:
        threshold *= -1
    if maxvalue < 0:
        maxvalue *= -1
    cv2.threshold(data, threshold, maxvalue, cv2.THRESH_BINARY, data)
    return data

threshold_binary = smw.register("ThresholdBinary", "test_threshold_binary", threshold_binary_helper, None, [float, int], TRI_STATE)
threshold_binary.__doc__ = """
Thresholds an image
Sets values > threshold to maxvalue

Args:
    data: numpy array of example
    threshold: threshold value
    maxvalue: value pixel set to if threshold is met

Returns:
    Transformed data
"""

def threshold_binary_max_helper(data, maxvalue=255, ratio=1.0):
    if maxvalue < 0:
        maxvalue *= -1
    threshold = np.floor(data.max() * ratio)
    cv2.threshold(data, threshold, maxvalue, cv2.THRESH_BINARY, data)
    return data

threshold_binary_max = smw.register("ThresholdBinaryMax", "test_threshold_binary_max", threshold_binary_max_helper, None, [int, float], TRI_STATE)
threshold_binary_max.__doc__ = """
Thresholds an image
Sets values > floor(max of array) to maxvalue

Args:
    data: numpy array of example
    maxvalue: value pixel set to if threshold is met

Returns:
    Transformed data
"""

def threshold_binary_float_helper(data, threshold=128, maxvalue=255):
    (data_max_x, data_max_y) = data.shape
    for x in range(0,data_max_x):
        for y in range(0, data_max_y):
            if data[x][y] < threshold:
                data[x][y] = 0
            else:
                data[x][y] = maxvalue
    return data

threshold_binary_float = smw.register("ThresholdBinaryFloat", "test_threshold_binary_float", threshold_binary_float_helper, None, [float, float], TRI_STATE)
threshold_binary_float.__doc__ = """
Thresholds an image
Sets values > threshold to maxvalue

Args:
    data: numpy array of example
    threshold: threshold value
    maxvalue: value pixel set to if threshold is met

Returns:
    Transformed data
"""

def threshold_to_zero_helper(data, threshold=128):
    if threshold < 0:
        threshold *= -1
    if type(threshold) == np.uint8:
        maxvalue = 255
    else:
        maxvalue = 1.0
    cv2.threshold(data, threshold, maxvalue, cv2.THRESH_TOZERO, data)
    return data

threshold_to_zero = smw.register("ThresholdToZero", "test_threshold_to_zero", threshold_to_zero_helper, None, [float], TRI_STATE)
threshold_to_zero.__doc__ = """
Thresholds an image
Sets values > threshold = value, value < threshold = 0

Args:
    data: numpy array of example
    threshold: threshold value

Returns:
    Transformed data
"""

def threshold_to_zero_by_pixel_float_helper(data, second_data):
    (data_max_x, data_max_y) = data.shape
    for x in range(0,data_max_x):
        for y in range(0, data_max_y):
            if data[x][y] < second_data[x][y]:
                data[x][y] = 0
    return data

threshold_to_zero_by_pixel_float = smw_2.register("ThresholdToZeroPixelFloat", "test_threshold_to_zero_by_pixel_float", threshold_to_zero_by_pixel_float_helper, None, [], TRI_STATE)
threshold_to_zero_by_pixel_float.__doc__ = """
Thresholds an image pixel by pixel

If data[x][y] > threshold[x][y] then data[x][y] else 0

Args:
    data: two dimensional image
    second_data: two dimensional image the same size as data_pair to act as
    the threshold value

Returns:
    Transformed data
"""

def threshold_to_zero_float_helper(data, threshold=0.5):
    (data_max_x, data_max_y) = data.shape
    for x in range(0,data_max_x):
        for y in range(0, data_max_y):
            if data[x][y] < threshold:
                data[x][y] = 0
    return data

threshold_to_zero_float = smw.register("ThresholdToZeroFloat", "test_threshold_to_zero_float", threshold_to_zero_float_helper, None, [float], TRI_STATE)
threshold_to_zero_float.__doc__ = """
Thresholds an image
Sets values > threshold = value, value < threshold = 0

Args:
    data: numpy array of example
    threshold: threshold value

Returns:
    Transformed data
"""

def threshold_binary_inverse_helper(data, threshold=128):
    if threshold < 0:
        threshold *= -1
    maxvalue = 255
    cv2.threshold(data, threshold, maxval=maxvalue, type=cv2.THRESH_BINARY_INV)
    return data

threshold_binary_inverse = smw.register("ThresholdBinaryInverse", "test_threshold_binary_inverse", threshold_binary_inverse_helper, None, [float], TRI_STATE)
threshold_binary_inverse.__doc__ = """
Thresholds an image
Sets values > threshold to 0

Args:
    data: numpy array of example
    threshold: threshold value

Returns:
    Transformed data
"""

def threshold_binary_inverse_mask_helper(data, threshold=0.9):
    if threshold < 0:
        threshold *= -1
    retval, mask = cv2.threshold(data, threshold, 1, cv2.THRESH_BINARY_INV)
    return mask

threshold_binary_inverse_mask = smw.register("ThresholdBinaryInverseMask", "test_threshold_binary_inverse_mask", threshold_binary_inverse_mask_helper, None, [float], TRI_STATE)
threshold_binary_inverse_mask.__doc__ = """
Sets all pixels > threshold to 0


Args:
    data: numpy array of example
    block_size: decides the size of neighbourhood area
    c: a constant which is subtracted from the mean calculated

Returns:
    Transformed data
"""

def adaptive_mean_thresholding_helper(data, block_size = 11, c = 2):
    mask = cv2.adaptiveThreshold(data, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c)
    return mask

adaptive_mean_thresholding = smw.register("AdaptiveThresholdMeanMask", "test_adaptive_threshold_mean_mask", adaptive_mean_thresholding_helper, None, [float], TRI_STATE)
adaptive_mean_thresholding.__doc__ = """
Adaptive threshold with mean of neighborhood area


Args:
    data: numpy array of example
    block_size: decides the size of neighbourhood area
    c: a constant which is subtracted from the weighted mean calculated

Returns:
    Transformed data
"""

def adaptive_gaussian_thresholding_helper(data, block_size = 11, c = 2):
    mask = cv2.adaptiveThreshold(data, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c)
    return mask

adaptive_gaussian_thresholding = smw.register("AdaptiveThresholdGaussianMask", "test_adaptive_threshold_gaussian_mask", adaptive_gaussian_thresholding_helper, None, [float], TRI_STATE)
adaptive_gaussian_thresholding.__doc__ = """
Adaptive threshold with weighted gaussian window



Args:
    data: numpy array of example
    threshold: threshold value

Returns:
    Transformed data
"""

def set_to_zero_if_greater_than_data_and_factor_helper(data, second_data, alpha=1):
    (data_max_x, data_max_y) = data.shape

    # if data
    for x in range(0, data_max_x):
        for y in range(0, data_max_y):
            if data[x][y] > alpha * second_data[x][y]:
                data[x][y] = 0
    return data

set_to_zero_if_greater_than_data_and_factor = smw_2.register("SetToZeroIf>Data&Factor", "test_set_to_zero_if_greater_than_data_and_factor", set_to_zero_if_greater_than_data_and_factor_helper, None, [float], TRI_STATE)
set_to_zero_if_greater_than_data_and_factor.__doc__ = """
Transforms an image

Returns data where data_out[x][y] = 0
if data[x][y] > alpha * second_data[x][y] else data_out[x][y] = data[x][y]

Args:
    data: given numpy array
    second_data: given numpy array
    alpha: scaling factor

Returns:
    Transformed data
"""

def set_to_zero_if_less_than_data_and_factor_helper(data, second_data, alpha=1):
    (data_max_x, data_max_y) = data.shape

    # if data
    for x in range(0, data_max_x):
        for y in range(0, data_max_y):
            if data[x][y] < alpha * second_data[x][y]:
                data[x][y] = 0
    return data

set_to_zero_if_less_than_data_and_factor = smw_2.register("SetToZeroIf<Data&Factor", "test_set_to_zero_if_less_than_data_and_factor", set_to_zero_if_less_than_data_and_factor_helper, None, [int], TRI_STATE)
set_to_zero_if_less_than_data_and_factor.__doc__ = """
Transforms an image

Returns data where data_out[x][y] = 0
if data[x][y] < alpha * second_data[x][y] else data_out[x][y] = data[x][y]

Args:
    data: given numpy array
    second_data: given numpy array
    alpha: scaling factor

Returns:
    Transformed data
"""

# Morphological operators

def morph_setup(data, kernel_x=5, kernel_y=5, iterations=1):
    kernel_x = check_kernel_size(kernel_x)
    kernel_y = check_kernel_size(kernel_y)
    return data, { "kernel_x":kernel_x, "kernel_y":kernel_y, "iterations":iterations }

def morph_setup_reduced(data, kernel_x=5, kernel_y=5):
    kernel_x = check_kernel_size(kernel_x)
    kernel_y = check_kernel_size(kernel_y)
    return data, { "kernel_x":kernel_x, "kernel_y":kernel_y }

def morph_erosion_rect_helper(data, kernel_x=5, kernel_y=5, iterations=1):
    kernel_type = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(kernel_type, (kernel_x, kernel_y))
    data = cv2.erode(data, kernel, iterations=iterations)
    return data

morph_erosion_rect = smw.register("MorphErosionRect", "test_morph_erosion_rect", morph_erosion_rect_helper, morph_setup, [int, int, int], TRI_STATE)
morph_erosion_rect.__doc__ = """
Perform morphological erosion on image

Args:
    data: numpy array of example
    kernel_x: positive odd integer length of the kernel's x axis
    kernel_y: positive odd integer length of the kernel's y axis
    iterations: positive integer the number of times to apply the
    morphological operation

Returns:
    Transformed data
"""

def morph_erosion_ellipse_helper(data, kernel_x=5, kernel_y=5, iterations=1):
    kernel_type = cv2.MORPH_ELLIPSE
    kernel = cv2.getStructuringElement(kernel_type, (kernel_x, kernel_y))
    data = cv2.erode(data, kernel, iterations=iterations)
    return data

morph_erosion_ellipse = smw.register("MorphErosionEllipse", "test_morph_erosion_ellipse", morph_erosion_ellipse_helper, morph_setup, [int, int, int], TRI_STATE)
morph_erosion_ellipse.__doc__ = """
Perform morphological erosion on image

Args:
    data: numpy array of example
    kernel_x: positive odd integer length of the kernel's x axis
    kernel_y: positive odd integer length of the kernel's y axis
    iterations: positive integer the number of times to apply the
    morphological operation

Returns:
    Transformed data
"""

def morph_erosion_cross_helper(data, kernel_x=5, kernel_y=5, iterations=1):
    kernel_type = cv2.MORPH_CROSS
    kernel = cv2.getStructuringElement(kernel_type, (kernel_x, kernel_y))
    data = cv2.erode(data, kernel, iterations=iterations)
    return data

morph_erosion_cross = smw.register("MorphErosionCross", "test_morph_erosion_cross", morph_erosion_cross_helper, morph_setup, [int, int, int], TRI_STATE)
morph_erosion_cross.__doc__ = """
Perform morphological erosion on image

Args:
    data: numpy array of example
    kernel_x: positive odd integer length of the kernel's x axis
    kernel_y: positive odd integer length of the kernel's y axis
    iterations: positive integer the number of times to apply the
    morphological operation

Returns:
    Transformed data
"""

def morph_dilate_rect_helper(data, kernel_x=5, kernel_y=5, iterations=1):
    kernel_type = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(kernel_type, (kernel_x, kernel_y))
    data = cv2.dilate(data, kernel, iterations=iterations)
    return data

morph_dilate_rect = smw.register("MorphDilateRect", "test_morph_dilate_rect", morph_dilate_rect_helper, morph_setup, [int, int, int], TRI_STATE)
morph_dilate_rect.__doc__ = """
Perform morphological dilation on image

Args:
    data: numpy array of example
    kernel_x: positive odd integer length of the kernel's x axis
    kernel_y: positive odd integer length of the kernel's y axis
    iterations: positive integer the number of times to apply the
    morphological operation

Returns:
    Transformed data
"""

def morph_dilate_ellipse_helper(data, kernel_x=5, kernel_y=5, iterations=1):
    kernel_type = cv2.MORPH_ELLIPSE
    kernel = cv2.getStructuringElement(kernel_type, (kernel_x, kernel_y))
    data = cv2.dilate(data, kernel, iterations=iterations)
    return data

morph_dilate_ellipse = smw.register("MorphDilateEllipse", "test_morph_dilate_ellipse", morph_dilate_ellipse_helper, morph_setup, [int, int, int], TRI_STATE)
morph_dilate_ellipse.__doc__ = """
Perform morphological dilation on image

Args:
    data: numpy array of example
    kernel_x: positive odd integer length of the kernel's x axis
    kernel_y: positive odd integer length of the kernel's y axis
    iterations: positive integer the number of times to apply the
    morphological operation

Returns:
    Transformed data
"""

def morph_dilate_cross_helper(data, kernel_x=5, kernel_y=5, iterations=1):
    kernel_type = cv2.MORPH_CROSS
    kernel = cv2.getStructuringElement(kernel_type, (kernel_x, kernel_y))
    data = cv2.dilate(data, kernel, iterations=iterations)
    return data

morph_dilate_cross = smw.register("MorphDilateCross", "test_morph_dilate_cross", morph_dilate_cross_helper, morph_setup, [int, int, int], TRI_STATE)
morph_dilate_cross.__doc__ = """
Perform morphological dilation on image

Args:
    data: numpy array of example
    kernel_x: positive odd integer length of the kernel's x axis
    kernel_y: positive odd integer length of the kernel's y axis
    iterations: positive integer the number of times to apply the
    morphological operation

Returns:
    Transformed data
"""

def morph_open_rect_helper(data, kernel_x=5, kernel_y=5):
    kernel_type = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(kernel_type, (kernel_x, kernel_y))
    data = cv2.morphologyEx(data, cv2.MORPH_OPEN, kernel)
    return data

morph_open_rect = smw.register("MorphOpenRect", "test_morph_open_rect", morph_open_rect_helper, morph_setup_reduced, [int, int], TRI_STATE)
morph_open_rect.__doc__ = """
Perform morphological opening on image. opening is erosion
followed by dilation
Useful for removing noise

Args:
    data: numpy array of example
    kernel_x: positive odd integer length of the kernel's x axis
    kernel_y: positive odd integer length of the kernel's y axis

Returns:
    Transformed data
"""

def morph_open_ellipse_helper(data, kernel_x=5, kernel_y=5):
    kernel_type = cv2.MORPH_ELLIPSE
    kernel = cv2.getStructuringElement(kernel_type, (kernel_x, kernel_y))
    data = cv2.morphologyEx(data, cv2.MORPH_OPEN, kernel)
    return data

morph_open_ellipse = smw.register("MorphOpenEllipse", "test_morph_open_ellipse", morph_open_ellipse_helper, morph_setup_reduced, [int, int], TRI_STATE)
morph_open_ellipse.__doc__ = """
Perform morphological opening on image. opening is erosion
followed by dilation
Useful for removing noise

Args:
    data: numpy array of example
    kernel_x: positive odd integer length of the kernel's x axis
    kernel_y: positive odd integer length of the kernel's y axis

Returns:
    Transformed data
"""

def morph_open_cross_helper(data, kernel_x=5, kernel_y=5):
    kernel_type = cv2.MORPH_CROSS
    kernel = cv2.getStructuringElement(kernel_type, (kernel_x, kernel_y))
    data = cv2.morphologyEx(data, cv2.MORPH_OPEN, kernel)
    return data

morph_open_cross = smw.register("MorphOpenCross", "test_morph_open_cross", morph_open_cross_helper, morph_setup_reduced, [int, int], TRI_STATE)
morph_open_cross.__doc__ = """
Perform morphological opening on image. opening is erosion
followed by dilation
Useful for removing noise

Args:
    data: numpy array of example
    kernel_x: positive odd integer length of the kernel's x axis
    kernel_y: positive odd integer length of the kernel's y axis

Returns:
    Transformed data
"""

def morph_close_rect_helper(data, kernel_x=5, kernel_y=5):
    kernel_type = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(kernel_type, (kernel_x, kernel_y))
    data = cv2.morphologyEx(data, cv2.MORPH_CLOSE, kernel)
    return data

morph_close_rect = smw.register("MorphCloseRect", "test_morph_close_rect", morph_close_rect_helper, morph_setup_reduced, [int, int], TRI_STATE)
morph_close_rect.__doc__ = """
Perform morphological close on image. closing is dilation
followed by erosion
Useful for filling in holes in the foreground image

Args:
    data: numpy array of example
    kernel_x: positive odd integer length of the kernel's x axis
    kernel_y: positive odd integer length of the kernel's y axis

Returns:
    Transformed data
"""

def morph_close_ellipse_helper(data, kernel_x=5, kernel_y=5):
    kernel_type = cv2.MORPH_ELLIPSE
    kernel = cv2.getStructuringElement(kernel_type, (kernel_x, kernel_y))
    data = cv2.morphologyEx(data, cv2.MORPH_CLOSE, kernel)
    return data

morph_close_ellipse = smw.register("MorphCloseEllipse", "test_morph_close_ellipse", morph_close_ellipse_helper, morph_setup_reduced, [int, int], TRI_STATE)
morph_close_ellipse.__doc__ = """
Perform morphological close on image. closing is dilation
followed by erosion
Useful for filling in holes in the foreground image

Args:
    data: numpy array of example
    kernel_x: positive odd integer length of the kernel's x axis
    kernel_y: positive odd integer length of the kernel's y axis

Returns:
    Transformed data
"""

def morph_close_cross_helper(data, kernel_x=5, kernel_y=5):
    kernel_type = cv2.MORPH_CROSS
    kernel = cv2.getStructuringElement(kernel_type, (kernel_x, kernel_y))
    data = cv2.morphologyEx(data, cv2.MORPH_CLOSE, kernel)
    return data

morph_close_cross = smw.register("MorphCloseCross", "test_morph_close_cross", morph_close_cross_helper, morph_setup_reduced, [int, int], TRI_STATE)
morph_close_cross.__doc__ = """
Perform morphological close on image. closing is dilation
followed by erosion
Useful for filling in holes in the foreground image

Args:
    data: numpy array of example
    kernel_x: positive odd integer length of the kernel's x axis
    kernel_y: positive odd integer length of the kernel's y axis

Returns:
    Transformed data
"""

def morph_gradient_rect_helper(data, kernel_x=5, kernel_y=5):
    kernel_type = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(kernel_type, (kernel_x, kernel_y))
    data = cv2.morphologyEx(data, cv2.MORPH_GRADIENT, kernel)
    return data

morph_gradient_rect = smw.register("MorphGradientRect", "test_morph_gradient_rect", morph_gradient_rect_helper, morph_setup_reduced, [int, int], TRI_STATE)
morph_gradient_rect.__doc__ = """
Perform morphological gradient on image. Difference between
dilation and erosion.
Results look like the outline of an object

Args:
    data: numpy array of example
    kernel_x: positive odd integer length of the kernel's x axis
    kernel_y: positive odd integer length of the kernel's y axis

Returns:
    Transformed data
"""

def morph_gradient_ellipse_helper(data, kernel_x=5, kernel_y=5):
    kernel_type = cv2.MORPH_ELLIPSE
    kernel = cv2.getStructuringElement(kernel_type, (kernel_x, kernel_y))
    data = cv2.morphologyEx(data, cv2.MORPH_GRADIENT, kernel)
    return data

morph_gradient_ellipse = smw.register("MorphGradientEllipse", "test_morph_gradient_ellipse", morph_gradient_ellipse_helper, morph_setup_reduced, [int, int], TRI_STATE)
morph_gradient_ellipse.__doc__ = """
Perform morphological gradient on image. Difference between
dilation and erosion.
Results look like the outline of an object

Args:
    data: numpy array of example
    kernel_x: positive odd integer length of the kernel's x axis
    kernel_y: positive odd integer length of the kernel's y axis

Returns:
    Transformed data
"""

def morph_gradient_cross_helper(data, kernel_x=5, kernel_y=5):
    kernel_type = cv2.MORPH_CROSS
    kernel = cv2.getStructuringElement(kernel_type, (kernel_x, kernel_y))
    data = cv2.morphologyEx(data, cv2.MORPH_GRADIENT, kernel)
    return data

morph_gradient_cross = smw.register("MorphGradientCross", "test_morph_gradient_cross", morph_gradient_cross_helper, morph_setup_reduced, [int, int], TRI_STATE)
morph_gradient_cross.__doc__ = """
Perform morphological gradient on image. Difference between
dilation and erosion.
Results look like the outline of an object

Args:
    data: numpy array of example
    kernel_x: positive odd integer length of the kernel's x axis
    kernel_y: positive odd integer length of the kernel's y axis

Returns:
    Transformed data
"""

def morph_tophat_rect_helper(data, kernel_x=5, kernel_y=5):
    kernel_type = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(kernel_type, (kernel_x, kernel_y))
    data = cv2.morphologyEx(data, cv2.MORPH_TOPHAT, kernel)
    return data

morph_tophat_rect = smw.register("MorphTophatRect", "test_morph_tophat_rect", morph_tophat_rect_helper, morph_setup_reduced, [int, int], TRI_STATE)
morph_tophat_rect.__doc__ = """
Perform morphological top hat on image. Difference between
close(image) and image

Args:
    data: numpy array of example
    kernel_x: positive odd integer length of the kernel's x axis
    kernel_y: positive odd integer length of the kernel's y axis

Returns:
    Transformed data
"""

def morph_tophat_ellipse_helper(data, kernel_x=5, kernel_y=5):
    kernel_type = cv2.MORPH_ELLIPSE
    kernel = cv2.getStructuringElement(kernel_type, (kernel_x, kernel_y))
    data = cv2.morphologyEx(data, cv2.MORPH_TOPHAT, kernel)
    return data

morph_tophat_ellipse = smw.register("MorphTophatEllipse", "test_morph_tophat_ellipse", morph_tophat_ellipse_helper, morph_setup_reduced, [int, int], TRI_STATE)
morph_tophat_ellipse.__doc__ = """
Perform morphological top hat on image. Difference between
close(image) and image

Args:
    data: numpy array of example
    kernel_x: positive odd integer length of the kernel's x axis
    kernel_y: positive odd integer length of the kernel's y axis

Returns:
    Transformed data
"""

def morph_tophat_cross_helper(data, kernel_x=5, kernel_y=5):
    kernel_type = cv2.MORPH_CROSS
    kernel = cv2.getStructuringElement(kernel_type, (kernel_x, kernel_y))
    data = cv2.morphologyEx(data, cv2.MORPH_TOPHAT, kernel)
    return data

morph_tophat_cross = smw.register("MorphTophatCross", "test_morph_tophat_cross", morph_tophat_cross_helper, morph_setup_reduced, [int, int], TRI_STATE)
morph_tophat_cross.__doc__ = """
Perform morphological top hat on image. Difference between
close(image) and image

Args:
    data: numpy array of example
    kernel_x: positive odd integer length of the kernel's x axis
    kernel_y: positive odd integer length of the kernel's y axis

Returns:
    Transformed data
"""

def morph_blackhat_rect_helper(data, kernel_x=5, kernel_y=5):
    kernel_type = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(kernel_type, (kernel_x, kernel_y))
    data = cv2.morphologyEx(data, cv2.MORPH_BLACKHAT, kernel)
    return data

morph_blackhat_rect = smw.register("MorphBlackhatRect", "test_morph_blackhat_rect", morph_blackhat_rect_helper, morph_setup_reduced, [int, int], TRI_STATE)
morph_blackhat_rect.__doc__ = """
Perform morphological blackhat on image using rectangular kernel.
Difference between close(image) and image

Args:
    data: numpy array of example
    kernel_x: positive odd integer length of the kernel's x axis
    kernel_y: positive odd integer length of the kernel's y axis

Returns:
    Transformed data
"""

def morph_blackhat_ellipse_helper(data, kernel_x=5, kernel_y=5):
    kernel_type = cv2.MORPH_ELLIPSE
    kernel = cv2.getStructuringElement(kernel_type, (kernel_x, kernel_y))
    data = cv2.morphologyEx(data, cv2.MORPH_BLACKHAT, kernel)
    return data

morph_blackhat_ellipse = smw.register("MorphBlackhatEllipse", "test_morph_blackhat_ellipse", morph_blackhat_ellipse_helper, morph_setup_reduced, [int, int], TRI_STATE)
morph_blackhat_ellipse.__doc__ = """
Perform morphological blackhat on image using ellipse kernel.
Difference between close(image) and image

Args:
    data: numpy array of example
    kernel_x: positive odd integer length of the kernel's x axis
    kernel_y: positive odd integer length of the kernel's y axis

Returns:
    Transformed data
"""

def morph_blackhat_cross_helper(data, kernel_x=5, kernel_y=5):
    kernel_type = cv2.MORPH_CROSS
    kernel = cv2.getStructuringElement(kernel_type, (kernel_x, kernel_y))
    data = cv2.morphologyEx(data, cv2.MORPH_BLACKHAT, kernel)
    return data

morph_blackhat_cross = smw.register("MorphBlackhatCross", "test_morph_blackhat_cross", morph_blackhat_cross_helper, morph_setup_reduced, [int, int], TRI_STATE)
morph_blackhat_cross.__doc__ = """
Perform morphological blackhat on image.
Difference between close(image) and image

Args:
    data: numpy array of example
    kernel_x: positive odd integer length of the kernel's x axis
    kernel_y: positive odd integer length of the kernel's y axis

Returns:
    Transformed data
"""

# Scalar Mathematical Operations

def scalar_op_helper(data, func=None, scalar=2.0):
    data = func(data, scalar)
    if data is None:
        raise Exception("OpenCV method returned None (Invalid Data)")
    return data

def scalar_add_setup(data, scalar=1.0):
    return data, {"func":cv2.add, "scalar":scalar}

scalar_add = smw.register("ScalarAdd", "test_scalar_add", scalar_op_helper, scalar_add_setup, [float], TRI_STATE)
scalar_add.__doc__ = """
Add a scalar value to each element of an array

Args:
    data: numpy array of example
    scalar: value to add

Returns:
    Transformed data
"""

def scalar_subtract_setup(data, scalar=1.0):
    return data, {"func":cv2.subtract, "scalar":scalar}

scalar_subtract = smw.register("ScalarSubtract", "test_scalar_subtract", scalar_op_helper, scalar_subtract_setup, [float], TRI_STATE)
scalar_subtract.__doc__ = """
Subtract a scalar value to each element of an array

Args:
    data: numpy array of example
    scalar: value to subtract

Returns:
    Transformed data
"""

def scalar_multiply_setup(data, scalar=1.0):
    return data, {"func":cv2.multiply, "scalar":scalar}

scalar_multiply = smw.register("ScalarMultiply", "test_scalar_multiply", scalar_op_helper, scalar_multiply_setup, [float], TRI_STATE)
scalar_multiply.__doc__ = """
Multiply a scalar value to each element of an array

Args:
    data: numpy array of example
    scalar: value to multiply

Returns:
    Transformed data
"""

def scalar_divide_setup(data, scalar=1.0):
    return data, {"func":cv2.divide, "scalar":scalar}

scalar_divide = smw.register("ScalarDivide", "test_scalar_divide", scalar_op_helper, scalar_divide_setup, [float], TRI_STATE)
scalar_divide.__doc__ = """
Divide a scalar value to each element of an array

Args:
    data: numpy array of example
    scalar: value to divide

Returns:
    Transformed data
"""

# Logic Operators

def bitwise_and_helper(data, second_data):
    return cv2.bitwise_and(data, second_data)

bitwise_and = smw_2.register("BitwiseAnd", "test_bitwise_and", bitwise_and_helper, None, [], TRI_STATE)
bitwise_and.__doc__ = """
Calculates the per-element bit-wise conjunction of two arrays
or an array and a scalar

Args:
    data: given numpy array
    second_data: given numpy array

Returns:
    Transformed data
"""

def bitwise_not_helper(data):
    return cv2.bitwise_not(data)

bitwise_not = smw.register("BitwiseNOT", "test_bitwise_not", bitwise_not_helper, None, [], TRI_STATE)
bitwise_not.__doc__ = """
Inverts every bit of an array

Args:
    data: numpy array of example

Returns:
    Transformed data
"""

def bitwise_or_helper(data, second_data):
    return cv2.bitwise_or(data, second_data)

bitwise_or = smw_2.register("BitwiseOr", "test_bitwise_or", bitwise_or_helper, None, [], TRI_STATE)
bitwise_or.__doc__ = """
Calculates the per-element bit-wise disjunction of two arrays
or an array and a scalar

Args:
    data: given numpy array
    second_data: given numpy array

Returns:
    Transformed data
"""

def bitwise_xor_helper(data, second_data):
    return cv2.bitwise_xor(data, second_data)

bitwise_xor = smw_2.register("BitwiseXOr", "test_bitwise_xor", bitwise_xor_helper, None, [], TRI_STATE)
bitwise_xor.__doc__ = """
Calculates the bitwize exclusive or

Args:
    data: given numpy array
    second_data: given numpy array

Returns:
    Transformed data
"""

# Image Mathematical Operations

def cv2_absdiff_helper(data, second_data):
    return cv2.absdiff(data, second_data)

cv2_absdiff = smw_2.register("Cv2AbsDiff", "test_cv2_absdiff", cv2_absdiff_helper, None, [], TRI_STATE)
cv2_absdiff.__doc__ = """
Calculates the per-element absolute difference between two arrays or
between an array and a scalar
Uses OpenCV method

Args:
    data: given numpy array
    second_data: given numpy array

Returns:
    Transformed data
"""

def absdiff_helper(data, second_data):
    return np.abs(data-second_data)

absdiff = smw_2.register("AbsDiff", "test_absdiff", absdiff_helper, None, [], TRI_STATE)
absdiff.__doc__ = """
Calculates the per-element absolute difference between two arrays or
between an array and a scalar

Args:
    data: given numpy array
    second_data: given numpy array

Returns:
    Transformed data
"""

def cv2_add_helper(data, second_data):
    return cv2.add(data, second_data)

cv2_add = smw_2.register("Cv2Add", "test_cv2_add", cv2_add_helper, None, [], TRI_STATE)
cv2_add.__doc__ = """
Add second_pair to data_pair. Data contains the sum

Args:
    data: given numpy array
    second_data: given numpy array

Returns:
    Transformed data
"""

def cv2_add_weighted_helper(data, second_data, alpha=1, beta=0):
    return cv2.addWeighted(data, alpha, second_data, beta, gamma=0)

cv2_add_weighted = smw_2.register("Cv2AddWeighted", "test_cv2_add_weighted", cv2_add_weighted_helper, None, [int, int], TRI_STATE)
cv2_add_weighted.__doc__ = """
Calculates the weighted sum of two arrays

Args:
    data: given numpy array
    second_data: given numpy array
    alpha: weight assigned to data
    beta: weight assigned to second_data

Returns:
    Transformed data
"""

def cv2_subtract_helper(data, second_data):
    return cv2.subtract(data, second_data)

cv2_subtract = smw_2.register("Cv2Subtract", "test_cv2_subtract", cv2_subtract_helper, None, [], TRI_STATE)
cv2_subtract.__doc__ = """
Subtract second_pair to data_pair. Data contains the result

Args:
    data: given numpy array
    second_data: given numpy array

Returns:
    Transformed data
"""

def subtract_saturate_helper(data, second_data):
    (xmax, ymax) = data.shape
    for x in range(0,xmax):
        for y in range(0,ymax):
            diff = data[x,y] - second_data[x,y]
            if diff < 0:
                data[x,y] = 0
            else:
                data[x,y] = diff
    return data

subtract_saturate = smw_2.register("SubtractSaturate", "test_subtract_saturate", subtract_saturate_helper, None, [], TRI_STATE)
subtract_saturate.__doc__ = """
Subtract second_pair to data_pair. Data contains the result

Args:
    data: given numpy array
    second_data: given numpy array

Returns:
    Transformed data
"""

def cv2_multiply_helper(data, second_data):
    return cv2.multiply(data, second_data)

cv2_multiply = smw_2.register("Cv2Multiply", "test_cv2_multiply", cv2_multiply_helper, None, [], TRI_STATE)
cv2_multiply.__doc__ = """
Multiply second_pair to data_pair. Data contains the result

Args:
    data: given numpy array
    second_data: given numpy array

Returns:
    Transformed data
"""

def multiply_transposed_helper(data, aTa=True):
    return cv2.mulTransposed(data, aTa=aTa)

multiply_transposed = smw.register("MultiplyTransposed", "test_multiply_transposed", multiply_transposed_helper, None, [bool], TRI_STATE)
multiply_transposed.__doc__ = """
Calculates the product of a matrix and its transposition
(data_pair)transpose(data_pair)

Args:
    data: numpy array of example

Returns:
    Transformed data
"""

def random_uniform_helper(data, low=0, high=255):
    return cv2.randu(data, low, high)

random_uniform = smw.register("RandomUniform", "test_random_uniform", random_uniform_helper, None, [int, int], TRI_STATE)
random_uniform.__doc__ = """
Generates a single uniformly-distributed random number or an array of
random numbers

Args:
    data: numpy array of example
    low: Inclusive lower boundary of the generated random numbers
    high: Exclusive upper boundary of the generated random numbers

Returns:
    Transformed data
"""

def random_normal_helper(data, normal_mean=128, std_dev=1):
    return cv2.randn(data, normal_mean, std_dev)

random_normal = smw.register("RandomNormal", "test_random_normal", random_normal_helper, None, [int, int], TRI_STATE)
random_normal.__doc__ = """
Generates a single normally distributed random number or an array of
random numbers

Args:
    data: numpy array of example
    normal_mean: the average
    std_dev: the standard deviation

Returns:
    Transformed data
"""

def random_shuffle_helper(data):
    return cv2.randShuffle(data)

random_shuffle = smw.register("RandomShuffle", "test_random_shuffle", random_shuffle_helper, None, [], TRI_STATE)
random_shuffle.__doc__ = """
Shuffles the array elements randomly

Args:
    data: numpy array of example

Returns:
    Transformed data
"""

def cv2_sqrt_helper(data):
    return cv2.sqrt(data)

cv2_sqrt = smw.register("Cv2Sqrt", "test_cv2_sqrt", cv2_sqrt_helper, None, [], TRI_STATE)
cv2_sqrt.__doc__ = """
Calculates the square root

Args:
    data: numpy array of example

Returns:
    Transformed data
"""

def cv2_divide_helper(data, second_data):
    return cv2.divide(data, second_data)

cv2_divide = smw_2.register("Cv2Divide", "test_cv2_divide", cv2_divide_helper, None, [], TRI_STATE)
cv2_divide.__doc__ = """
Divide second_pair to data_pair. Data contains the result

Args:
    data: given numpy array
    second_data: given numpy array

Returns:
    Transformed data
"""

def np_divide_helper(data, second_data):
    (xmax, ymax) = data.shape
    for x in range(0,xmax):
        for y in range(0,ymax):
            if second_data[x,y] == 0.0:
                data[x,y] = 0
            else:
                data[x,y] = data[x,y] / second_data[x,y]
    return data

np_divide = smw_2.register("NumpyDivide", "test_np_divide", np_divide_helper, None, [], TRI_STATE)
np_divide.__doc__ = """
Divide second_pair to data_pair. Data contains the result

Args:
    data: given numpy array
    second_data: given numpy array

Returns:
    Transformed data
"""

def cv2_pow_helper(data, power=2):
    return cv2.pow(data, power)

cv2_pow = smw.register("Cv2Pow", "test_cv2_pow", cv2_pow_helper, None, [int], TRI_STATE)
cv2_pow.__doc__ = """
Raises every array element to a power

Args:
    data: numpy array of example
    power: power to raise elements to

Returns:
    Transformed data
"""

def cv2_rms_helper(data, kernel_size=5):
    # avg(Img^2)
    img_sq = cv2.multiply(data, data)
    avg_img_sq = cv2.blur(img_sq, ksize=(kernel_size, kernel_size))

    # avg(I)avg(I)
    avg = cv2.blur(data, ksize=(kernel_size, kernel_size))
    avg_sq = cv2.multiply(avg, avg)

    # sqrt(avg(I^2) - avg(I)avg(I))
    diff = cv2.subtract(avg_img_sq,avg_sq)
    data = cv2.sqrt(diff)
    return data

cv2_rms = smw.register("Cv2RMS", "test_cv2_rms", cv2_rms_helper, None, [int], TRI_STATE)
cv2_rms.__doc__ = """
Calculate the RMS value for the image in data_pair

x = width of image
y = height of image
I = image

rms = sqrt(blur(I*I) - blur(I)*blur(I))

Args:
    data: numpy array of example
    kernel_size: size of the kernel :)

Returns:
    Transformed data
"""

def cv2_dct_helper(data):
    flag = False
    try:
        # check if data is iterable
        iter(data)
        flag = True
    except:
        pass
    
    # Check if the iterable is empty
    if not (flag and len(data) > 0 and len([i for i in np.array(data).shape if i == 0]) == 0):
        raise ValueError("Data array cannot be empty.")

    return cv2.dct(data.astype(np.float))

cv2_dct = smw.register("Cv2DCT", "test_cv2_dct", cv2_dct_helper, None, [], TRI_STATE)
cv2_dct.__doc__ = """
Discrete Cosine Transform

Args:
    data: numpy array of example

Returns:
    Transformed data
"""

def cv2_idct_helper(data):
    flag = False
    try:
        # check if data is iterable
        iter(data)
        flag = True
    except:
        pass
    
    # Check if the iterable is empty
    if not (flag and len(data) > 0 and len([i for i in np.array(data).shape if i == 0]) == 0):
        raise ValueError("Data array cannot be empty.")
    return cv2.idct(data)

cv2_idct = smw.register("Cv2iDCT", "test_cv2_idct", cv2_idct_helper, None, [], TRI_STATE)
cv2_idct.__doc__ = """
Inverse Discrete Cosine Transform

Args:
    data: numpy array of example

Returns:
    Transformed data
"""

def cv2_dft_real_helper(data):
    flag = False
    try:
        # check if data is iterable
        iter(data)
        flag = True
    except:
        pass
    
    # Check if the iterable is empty
    if not (flag and len(data) > 0 and len([i for i in np.array(data).shape if i == 0]) == 0):
        raise ValueError("Data array cannot be empty.")
    return cv2.dft(data.astype(np.float), flags=cv2.DFT_REAL_OUTPUT)

cv2_dft_real = smw.register("Cv2DFTReal", "test_cv2_dft_real", cv2_dft_real_helper, None, [], TRI_STATE)
cv2_dft_real.__doc__ = """
Discrete Fourier transform of a 1D or 2D floating-point array.
Returns the real component only.

Args:
    data: numpy array of example

Returns:
    Transformed data
"""

def cv2_idft_helper(data):
    flag = False
    try:
        # check if data is iterable
        iter(data)
        flag = True
    except:
        pass
    
    # Check if the iterable is empty
    if not (flag and len(data) > 0 and len([i for i in np.array(data).shape if i == 0]) == 0):
        raise ValueError("Data array cannot be empty.")
    return cv2.idft(data)

cv2_idft = smw.register("Cv2iDFT", "test_cv2_idft", cv2_idft_helper, None, [], TRI_STATE)
cv2_idft.__doc__ = """
Inverse Discrete Fourier transform of a 1D or 2D floating-point array.
Returns complex

Args:
    data: numpy array of example

Returns:
    Transformed data
"""

def cv2_transpose_helper(data):
    return cv2.transpose(data)

cv2_transpose = smw.register("Cv2Transpose", "test_cv2_transpose", cv2_transpose_helper, None, [], TRI_STATE)
cv2_transpose.__doc__ = """
Transposes a matrix :D

Args:
    data: numpy array of example

Returns:
    Transformed data
"""

def cv2_log_helper(data):
    return cv2.log(np.float32(data))

cv2_log = smw.register("Cv2Log", "test_cv2_log", cv2_log_helper, None, [], TRI_STATE)
cv2_log.__doc__ = """
Calculates the natural logarithm of every array element

Args:
    data: numpy array of example

Returns:
    Transformed data
"""

def cv2_max_helper(data, second_data):
    return cv2.max(data, second_data)

cv2_max = smw_2.register("Cv2Max", "test_cv2_max", cv2_max_helper, None, [], TRI_STATE)
cv2_max.__doc__ = """
Calculates the per-element maximum of two arrays

Args:
    data: given numpy array
    second_data: given numpy array

Returns:
    Transformed data
"""

def cv2_min_helper(data, second_data):
    return cv2.min(data, second_data)

cv2_min = smw_2.register("Cv2Min", "test_cv2_min", cv2_min_helper, None, [], TRI_STATE)
cv2_min.__doc__ = """
Calculates the per-element minimum of two arrays

Args:
    data: given numpy array
    second_data: given numpy array

Returns:
    Transformed data
"""

def scalar_max_helper(data, scalar=1):
    return cv2.max(data, scalar)

scalar_max = smw.register("ScalarMax", "test_scalar_max", scalar_max_helper, None, [int], TRI_STATE)
scalar_max.__doc__ = """
Calculates the per-element maximum of an array and a scalar

Args:
    data: numpy array of example
    scalar: scalar value

Returns:
    Transformed data
"""

def scalar_min_helper(data, scalar=1):
    return cv2.min(data, scalar)

scalar_min = smw.register("ScalarMin", "test_scalar_min", scalar_min_helper, None, [int], TRI_STATE)
scalar_min.__doc__ = """
Calculates the per-element minimum of an array and a scalar

Args:
    data: numpy array of example
    scalar: scalar value

Returns:
    Transformed data
"""

def multiply_spectrum_helper(data, second_data):
    flag = False
    try:
        # check if data is iterable
        iter(data)
        flag = True
    except:
        pass

    # Check if the iterable is empty
    if not (flag and len(data) > 0 and len([i for i in np.array(data).shape if i == 0]) == 0):
        raise ValueError("Data array cannot be empty.")

    return cv2.mulSpectrums(data, second_data, flags=0)

multiply_spectrum = smw_2.register("MultiplySpectrum", "test_multiply_spectrum", multiply_spectrum_helper, None, [], TRI_STATE)
multiply_spectrum.__doc__ = """
Calculates the per-element multiplication of two Fourier spectrums.

Args:
    data: given numpy array
    second_data: given numpy array

Returns:
    Transformed data
"""

# Comparison Operators
def cv2_compare_helper(data, second_data, compare_mode):
    result = cv2.compare(data, second_data, compare_mode)
    # If either data or second_data is an empty array, this will return none instead of an empty array
    if result is None:
        result = np.array([])
    return result


def cv2_equal_setup(data, second_data):
    return data, second_data, {'compare_mode':cv2.CMP_EQ}


cv2_equal = smw_2.register("Cv2Equal", "test_cv2_equal", cv2_compare_helper, cv2_equal_setup, [], TRI_STATE)
cv2_equal.__doc__ = """
Performs the per-element comparison of two arrays or an array and
scalar value

Args:
    data: given numpy array
    second_data: given numpy array

Returns:
    Transformed data
"""


def cv2_greater_than_setup(data, second_data):
    return data, second_data, {'compare_mode':cv2.CMP_GT}


cv2_greater_than = smw_2.register("Cv2GreaterThan", "test_cv2_greater_than", cv2_compare_helper, cv2_greater_than_setup, [], TRI_STATE)
cv2_greater_than.__doc__ = """
Performs the per-element comparison of two arrays or an array and
scalar value

Args:
    data: given numpy array
    second_data: given numpy array

Returns:
    Transformed data
"""

def cv2_greater_than_or_equal_setup(data, second_data):
    return data, second_data, {'compare_mode':cv2.CMP_GE}

cv2_greater_than_or_equal = smw_2.register("Cv2GreaterThanOrEqual", "test_cv2_greater_than_equal", cv2_compare_helper, cv2_greater_than_or_equal_setup, [], TRI_STATE)
cv2_greater_than_or_equal.__doc__ = """
Performs the per-element comparison of two arrays or an array and
scalar value

Args:
    data: given numpy array
    second_data: given numpy array

Returns:
    Transformed data
"""

def cv2_less_than_setup(data, second_data):
    return data, second_data, {'compare_mode':cv2.CMP_LT}

cv2_less_than = smw_2.register("Cv2LessThan", "test_cv2_less_than", cv2_compare_helper, cv2_less_than_setup, [], TRI_STATE)
cv2_less_than.__doc__ = """
Performs the per-element comparison of two arrays or an array and
scalar value

Args:
    data: given numpy array
    second_data: given numpy array

Returns:
    Transformed data
"""

def cv2_less_than_or_equal_setup(data, second_data):
    return data, second_data, {'compare_mode':cv2.CMP_LE}

cv2_less_than_or_equal = smw_2.register("Cv2LessThanOrEqual", "test_cv2_less_than_equal", cv2_compare_helper, cv2_less_than_or_equal_setup, [], TRI_STATE)
cv2_less_than_or_equal.__doc__ = """
Performs the per-element comparison of two arrays or an array and
scalar value

Args:
    data: given numpy array
    second_data: given numpy array

Returns:
    Transformed data
"""

def cv2_not_equal_setup(data, second_data):
    return data, second_data, {'compare_mode':cv2.CMP_NE}

cv2_not_equal = smw_2.register("Cv2NotEqual", "test_cv2_not_equal", cv2_compare_helper, cv2_not_equal_setup, [], TRI_STATE)
cv2_not_equal.__doc__ = """
Performs the per-element comparison of two arrays or an array and
scalar value

Args:
    data: given numpy array
    second_data: given numpy array

Returns:
    Transformed data
"""

# TODO does this even make sense? Check the inRange method
# A: no it does not
# What could possibly go wrong ^_^
def in_range_helper(data, lower_bound=0, upper_bound=128):
    return cv2.inRange(data, lower_bound, upper_bound)

in_range = smw.register("InRange", "test_in_range", in_range_helper, None, [int, int], TRI_STATE)
in_range.__doc__ = """
Checks if array elements lie between the lower bound and the upper bound

Args:
    data: numpy array of example
    lower_bound: lower bound
    upper_bound: upper bound

Returns:
    Transformed data
"""

# Other Operators

def std_deviation_helper(data, kernel_size=5):
    flag = False
    try:
        # check if data is iterable
        iter(data)
        flag = True
    except:
        pass

    # Check if the iterable is empty
    if not (flag and len(data) > 0 and len([i for i in np.array(data).shape if i == 0]) == 0):
        raise ValueError("Data array cannot be empty.")

    kernel_size = check_kernel_size(kernel_size)
    my_mean = cv2.blur(data, (kernel_size, kernel_size))
    data = cv2.subtract(data, my_mean)
    data = cv2.multiply(data, data)
    population = kernel_size * kernel_size
    data = cv2.divide(data, population)
    data = cv2.sqrt(abs(data))
    return data

std_deviation = smw.register("StdDeviation", "test_std_deviation", std_deviation_helper, None, [int], TRI_STATE)
std_deviation.__doc__ = """
Standard deviation of an image
implements sqrt(blur(img^2) - blur(img)^2)

Args:
    data: numpy array of example
    kernel_size: size of kernel

Returns:
    Transformed data
"""

def threshold_nlargest_helper(data, n=5):
    # make it flat and convert to set
    vector = np.reshape(data, -1)
    vector = set(vector)
    # find n largest, get the min, perform a threshold on the min
    largest = heapq.nlargest(n, vector)
    threshold = min(largest)
    if type(threshold) == int:
        maxvalue = 255
    else:
        maxvalue = 1.0
    retval, data = cv2.threshold(data, threshold, maxvalue, cv2.THRESH_TOZERO)
    return data

threshold_nlargest = smw.register("ThresholdNLargest", "test_threshold_nlargest", threshold_nlargest_helper, None, [int], TRI_STATE)
threshold_nlargest.__doc__ = """
Determines the n largest values in image.
Returns an image where pixels >= the min(nlargest)

Args:
    data: numpy array of example
    n: n largest values from the source image

Returns:
    Transformed data
"""

def threshold_nlargest_binary_helper(data, n=5):
    vector = np.reshape(data, -1)
    vector = set(vector)
    largest = heapq.nlargest(n, vector)
    threshold = min(largest)
    if type(threshold) == np.uint8:
        maxvalue = 255
    else:
        maxvalue = 1.0
    retval, data = cv2.threshold(data, threshold, maxvalue, cv2.THRESH_BINARY)
    return data

threshold_nlargest_binary = smw.register("ThresholdNLargestBinary", "test_threshold_nlargest_binary", threshold_nlargest_binary_helper, None, [int], TRI_STATE)
threshold_nlargest_binary.__doc__ = """
Determines the n largest binary values in image.
Returns an image where pixels >= the min(nlargest)

Args:
    data: numpy array of example
    n: n largest values from the source image

Returns:
    Transformed data
"""

# not in primitive set due to returning a list and not arrays
def nlargest(data_pair, n=5):
    """Return a list of the n highest amplitudes in the data_pair

    TODO - returns list

    Args:
        data_pair: given datapair
        n: the number of values to return

    Returns:
        [ [n largest training data], [nlargest test data] ]
    """
    data_list = []
    for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
        instances = cp.deepcopy(data_set.get_instances())

        for instance in instances:
            data = instance.get_stream().get_data()

            # make it flat
            data = np.reshape(data, -1)

            # convert to set
            img = set(data)

            largest = heapq.nlargest(n, img)
            data_list.append(largest)

    return data_list

def scale_abs_helper(data, alpha=1, beta=0):
    return cv2.convertScaleAbs(data, data, alpha, beta)

scale_abs = smw.register("Cv2ScaleAbs", "test_scale_abs", scale_abs_helper, None, [int, int], TRI_STATE)
scale_abs.__doc__ = """
Scales, calculates absolute values, and converts the result to 8-bit

Args:
    data: numpy array of example
    alpha: scale factor
    beta: delta added to the scaled values

Returns:
    Transformed data
"""

#do not include in gp_framework_helper
def get_data_items(data_pair):
    """Returns the data items in a list from the data_pair

    Args:
        data_pair: given datapair

    Returns:
        Transformed data pair
    """
    data_list = []
    for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
        instances = cp.deepcopy(data_set.get_instances())

        for instance in instances:
            data = instance.get_stream().get_data()
            data_list.append(data)

    return data_list

# Contour Helper functions

#do not include in gp_framework_helper
def get_circular_extent(img, contour):
    """Calculates the ratio of occupied pixels to circle area

    Args:
        img: this needs to be the original image
        contour: contour

    Returns:
        extent float
    """
    drawn_contours = np.zeros(img.shape, img.dtype)
    cv2.drawContours(drawn_contours, contour, 0, 255, 2, -1)

    # draw a filled circle around the contour
    mask = np.zeros(img.shape, img.dtype)
    (x, y), radius = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))
    radius = int(radius)
    cv2.circle(mask, center, radius, 255, -1)
    total_area = cv2.countNonZero(mask)

    # mask the original image with the contour image
    img = cv2.bitwise_and(img, drawn_contours)
    object_area = cv2.countNonZero(img)

    return float(object_area) / total_area

#do not include in gp_framework_helper
def aspect_ratio(contour):
    """Calculates the aspect ratio width/height of the contour

    Args:
        contour: contour

    Returns:
        aspect ratio float
    """
    epsilon = 0.01*cv2.arcLength(contour,closed=True)
    approx = cv2.approxPolyDP(contour,epsilon,closed=True)
    x, y, w, h = cv2.boundingRect(contour)
    aspect = float(w) / h
    return aspect

#do not include in gp_framework_helper
def extent(contour):
    """Extent is the ratio of contour area to bounding rectangle area

    Args:
        contour: contour

    Returns:
        float ratio of object area to bounding rectangle area
    """
    area = cv2.contourArea(contour)
    epsilon = 0.1*cv2.arcLength(contour,closed=True)
    approx = cv2.approxPolyDP(contour,epsilon,closed=True)
    w, y, w, h = cv2.boundingRect(approx)
    rect_area = w * h
    e = float(area) / rect_area
    return e

#do not include in gp_framework_helper
def solidity(contour):
    """Solidity is the ratio of contour area to its convex hull area

    Args:
        contour: contour

    Returns:
        solidity
    """
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        hull_area = 1
    s = float(area) / hull_area
    return s

#do not include in gp_framework_helper
def equ_diameter(contour):
    """Calculates the equ_diameter

    Args:
        contour: contour

    Returns:
        equ_diameter
    """
    area = cv2.contourArea(contour)
    equi_diameter = math.sqrt(abs(4 * area / np.pi))
    return equi_diameter

# Contours

# do not include in gp_framework_helper
def get_contours(data_pair):
    """Returns ndarray of contours

    Args:
        data_pair: given datapair

    Returns:
        Transformed data pair
    """
    data_list = []
    for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
        instances = cp.deepcopy(data_set.get_instances())

        for instance in instances:
            data = instance.get_stream().get_data()

            contours, hierarchy = cv2.findContours(data, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            data_list.append(contours)

    return data_list

def contours_all_helper(data):
    contours, hierarchy = cv2.findContours(data, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return data

contours_all = smw.register("ContoursAll", "test_contours_all", contours_all_helper, None, [], TRI_STATE)
contours_all.__doc__ = """
Finds all contours in the image

Args:
    data: numpy array of example

Returns:
    Transformed data
"""

def contours_min_area_helper(data, area=10):
    data = cv2.Canny(data, 50, 150)
    contours, hierarchy = cv2.findContours(data, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if cv2.contourArea(c) >= area]
    # create mask to return just the contours that meet the area requirement
    mask = np.zeros(data.shape, data.dtype)
    cv2.drawContours(mask, contours, -1, 255, 2, -1)
    # contours are drawn with max value 255 therefor all bits set
    data = cv2.bitwise_and(data, mask)
    return data

contours_min_area = smw.register("ContoursMinArea", "test_contours_min_area", contours_min_area_helper, None, [int], TRI_STATE)
contours_min_area.__doc__ = """
Finds all contours that have min area or more

Args:
    data: numpy array of example
    area: uint8,float minimum area

Returns:
    Transformed data
"""

def contours_max_area_helper(data, area=10):
    data = cv2.Canny(data, 50, 150)
    contours, hierarchy = cv2.findContours(data, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if cv2.contourArea(c) <= area]
    # create mask to return just the contours that meet the area requirement
    mask = np.zeros(data.shape, data.dtype)
    cv2.drawContours(mask, contours, -1, 255, 2, -1)
    # contours are drawn with max value 255 therefor all bits set
    data = cv2.bitwise_and(data, mask)
    return data

contours_max_area = smw.register("ContoursMaxArea", "test_contours_max_area", contours_max_area_helper, None, [int], TRI_STATE)
contours_max_area.__doc__ = """
Returns image with all contours that have an area <= area

Args:
    data: numpy array of example
    area: uint8,float minimum area

Returns:
    Transformed data
"""

def contours_convex_concave_helper(data, convex=True):
    data = cv2.Canny(data, 50, 150)
    contours, hierarchy = cv2.findContours(data, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if convex:
        contours = [c for c in contours if cv2.isContourConvex(c)]
    else:
        contours = [c for c in contours if not cv2.isContourConvex(c)]
    mask = np.zeros(data.shape, data.dtype)
    cv2.drawContours(mask, contours, -1, 255, 2, -1)
    data = cv2.bitwise_and(data, mask)
    return data

contours_convex_concave = smw.register("ContoursConvexConcave", "test_contours_convex_concave", contours_convex_concave_helper, None, [bool], TRI_STATE)
contours_convex_concave.__doc__ = """
Returns an image of contours if the contour is convex when convex=True.
When convex=False it will return the concave contours

Args:
    data: numpy array of example
    convex: bool True returns convex contours, false returns
    concave contours

Returns:
    Transformed data
"""

def contours_min_length_helper(data, length=10):
    data = cv2.Canny(data, 50, 150)
    contours, hierarchy = cv2.findContours(data, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if cv2.arcLength(c, closed=True) >= length]
    mask = np.zeros(data.shape, data.dtype)
    cv2.drawContours(mask, contours, -1, 255, 2, -1)
    data = cv2.bitwise_and(data, mask)
    return data

contours_min_length = smw.register("ContoursMinLength", "test_contours_min_length", contours_min_length_helper, None, [int], TRI_STATE)
contours_min_length.__doc__ = """
Returns an image of contours if the contours arc length >= length

Args:
    data: numpy array of example
    length: int / float minimum length of contours to include in the
    output image

Returns:
    Transformed data
"""

def contours_max_length_helper(data, length=10):
    contours, hierarchy = cv2.findContours(data, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if cv2.arcLength(c, closed=True) <= length]
    mask = np.zeros(data.shape, data.dtype)
    cv2.drawContours(mask, contours, -1, 255, 2, -1)
    data = cv2.bitwise_and(data, mask)
    return data

contours_max_length = smw.register("ContoursMaxLength", "test_contours_max_length", contours_max_length_helper, None, [int], TRI_STATE)
contours_max_length.__doc__ = """
Returns an image of contours if the contours arc length <= length

Args:
    data: numpy array of example
    length: int / float minimum length of contours to include in the
    output image

Returns:
    Transformed data
"""

# Contour Masks

def contour_mask_helper(data):
    data = cv2.Canny(data, 50, 150)
    cont, hierarchy = cv2.findContours(data, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(data, cont, -1, 255, 2, -1)
    return data

contour_mask = smw.register("ContourMask", "test_contour_mask", contour_mask_helper, None, [], TRI_STATE)
contour_mask.__doc__ = """
RFinds the contours in an image, fills in the contours, returns an
image suitable for a mask

Args:
    data: numpy array of example

Returns:
    Transformed data
"""

def contour_mask_min_area_helper(data, area=10.0):
    data = cv2.Canny(data, 50, 150)
    contours, hierarchy = cv2.findContours(data, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if cv2.contourArea(c) >= area]
    data = np.zeros(data.shape, data.dtype)
    cv2.drawContours(data, contours, -1, 255, 2, -1)
    return data

contour_mask_min_area = smw.register("ContourMaskMinArea", "test_contour_mask_min_area", contour_mask_min_area_helper, None, [float], TRI_STATE)
contour_mask_min_area.__doc__ = """
Returns a mask of all contours that have an area >= area

Args:
    data: numpy array of example
    area: float of area

Returns:
    Transformed data
"""

def contour_mask_max_area_helper(data, area=10.0):
    data = cv2.Canny(data, 50, 150)
    contours, hierarchy = cv2.findContours(data, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if cv2.contourArea(c) <= area]
    data = np.zeros(data.shape, data.dtype)
    cv2.drawContours(data, contours, -1, 255, 2, -1)
    return data

contour_mask_max_area = smw.register("ContourMaskMaxArea", "test_contour_mask_max_area", contour_mask_max_area_helper, None, [float], TRI_STATE)
contour_mask_max_area.__doc__ = """
Returns a mask of all contours that have an area <= area

Args:
    data: numpy array of example
    area: float of area

Returns:
    Transformed data
"""

def contour_mask_convex_helper(data, convex=True):
    data = cv2.Canny(data, 50, 150)
    contours, hierarchy = cv2.findContours(data, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if convex:
        contours = [c for c in contours if cv2.isContourConvex(c)]
    else:
        contours = [c for c in contours if not cv2.isContourConvex(c)]
    data = np.zeros(data.shape, data.dtype)
    cv2.drawContours(data, contours, -1, 255, 2, -1)
    return data

contour_mask_convex = smw.register("ContourMaskConvex", "test_contour_mask_convex", contour_mask_convex_helper, None, [bool], TRI_STATE)
contour_mask_convex.__doc__ = """
Returns an image of contours if the contour is convex when convex=True.
When convex=False it will return the concave contours

Args:
    data: numpy array of example
    convex: bool True returns convex contours, false returns
    concave contours

Returns:
    Transformed data
"""

def contour_mask_min_length_helper(data, length=10.0):
    data = cv2.Canny(data, 50, 150)
    contours, hierarchy = cv2.findContours(data, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if cv2.contourArea(c) >= length]
    data = np.zeros(data.shape, data.dtype)
    cv2.drawContours(data, contours, -1, 255, 2, -1)
    return data

contour_mask_min_length = smw.register("ContourMaskMinLength", "test_contour_mask_min_length", contour_mask_min_length_helper, None, [float], TRI_STATE)
contour_mask_min_length.__doc__ = """
Returns a mask of all contours that have an arc length >= length

Args:
    data: numpy array of example
    length: arc length requirement

Returns:
    Transformed data
"""

def contour_mask_max_length_helper(data, length=10.0):
    data = cv2.Canny(data, 50, 150)
    contours, hierarchy = cv2.findContours(data, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if cv2.contourArea(c) <= length]
    data = np.zeros(data.shape, data.dtype)
    cv2.drawContours(data, contours, -1, 255, 2, -1)
    return data

contour_mask_max_length = smw.register("ContourMaskMaxLength", "test_contour_mask_max_length", contour_mask_max_length_helper, None, [float], TRI_STATE)
contour_mask_max_length.__doc__ = """
Returns a mask of all contours that have an arc length <= length

Args:
    data: numpy array of example
    length: arc length requirement

Returns:
    Transformed data
"""

def contour_mask_range_length_helper(data, lower_bound=0, upper_bound=128):
    data = cv2.Canny(data, 50, 150)
    contours, hierarchy = cv2.findContours(data, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if lower_bound <= cv2.contourArea(c) <= upper_bound]
    data = np.zeros(data.shape, data.dtype)
    cv2.drawContours(data, contours, -1, 255, 2, -1)
    return data

contour_mask_range_length = smw.register("ContourMaskRangeLength", "test_contour_mask_range_length", contour_mask_range_length_helper, None, [int, int], TRI_STATE)
contour_mask_range_length.__doc__ = """
Returns a mask of all contours that have an arc length >= lower_bound
and <= upper_bound

Args:
    data: numpy array of example
    lower_bound: lower bound
    upper_bound: upper bound

Returns:
    Transformed data
"""

def contour_mask_min_enclosing_circle_helper(data, area=10.0):
    data = cv2.Canny(data, 50, 150)
    contours, hierarchy = cv2.findContours(data, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if cv2.contourArea(c) >= area]
    data = np.zeros(data.shape, data.dtype)
    cv2.drawContours(data, contours, -1, 255, 2, -1)
    return data

contour_mask_min_enclosing_circle = smw.register("ContourMaskMinEnclosingCircle", "test_contour_mask_min_enclosing_circle", contour_mask_min_enclosing_circle_helper, None, [float], TRI_STATE)
contour_mask_min_enclosing_circle.__doc__ = """
Returns a mask of all contours that have a minimum enclosing
circle area >= area

Args:
    data: numpy array of example
    area: minimum area of contours

Returns:
    Transformed data
"""

def contour_mask_max_enclosing_circle_helper(data, area=10.0):
    data = cv2.Canny(data, 50, 150)
    contours, hierarchy = cv2.findContours(data, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if cv2.contourArea(c) <= area]
    data = np.zeros(data.shape, data.dtype)
    cv2.drawContours(data, contours, -1, 255, 2, -1)
    return data

contour_mask_max_enclosing_circle = smw.register("ContourMaskMaxEnclosingCircle", "test_contour_mask_max_enclosing_circle", contour_mask_max_enclosing_circle_helper, None, [float], TRI_STATE)
contour_mask_max_enclosing_circle.__doc__ = """
Returns a mask of all contours that have a minimum enclosing
circle area >= area

Args:
    data: numpy array of example
    area: minimum area of contours

Returns:
    Transformed data
"""

def contour_mask_range_enclosing_circle_helper(data, lower_bound=0, upper_bound=128):
    data = cv2.Canny(data, 50, 150)
    contours, hierarchy = cv2.findContours(data, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if lower_bound <= cv2.contourArea(c) <= upper_bound]
    data = np.zeros(data.shape, data.dtype)
    cv2.drawContours(data, contours, -1, 255, 2, -1)
    return data

contour_mask_range_enclosing_circle = smw.register("ContourMaskRangeEnclosingCircle", "test_contour_mask_range_enclosing_circle", contour_mask_range_enclosing_circle_helper, None, [int, int], TRI_STATE)
contour_mask_range_enclosing_circle.__doc__ = """
Returns mask of all contours that have an area >= lower_bound and area <= upper_bound
circle area >= area

Args:
    data: numpy array of example
    lower_bound: lower bound
    upper_bound: upper bound

Returns:
    Transformed data
"""

def contour_mask_min_extent_enclosing_circle_helper(data, ratio=0.5):
    data = cv2.Canny(data, 50, 150)
    org = np.copy(data)
    contours, hierarchy = cv2.findContours(data, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if get_circular_extent(org, c) >= ratio]
    data = np.zeros(data.shape, data.dtype)
    cv2.drawContours(data, contours, -1, 255, 2, -1)
    return data

contour_mask_min_extent_enclosing_circle = smw.register("ContourMaskMinExtentEnclosingCircle", "test_contour_mask_min_extent_enclosing_circle", contour_mask_min_extent_enclosing_circle_helper, None, [float], TRI_STATE)
contour_mask_min_extent_enclosing_circle.__doc__ = """
Calculates the ratio of occupied pixels in the original image to the
number of pixels in minim enclosing circle of each contour.
The mask returned has objects that are >= ratio in extent

Args:
    data: numpy array of example
    ratio: ratio for comparison

Returns:
    Transformed data
"""

def contour_mask_max_extent_enclosing_circle_helper(data, ratio=0.5):
    data = cv2.Canny(data, 50, 150)
    org = np.copy(data)
    contours, hierarchy = cv2.findContours(data, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if get_circular_extent(org, c) <= ratio]
    data = np.zeros(data.shape, data.dtype)
    cv2.drawContours(data, contours, -1, 255, 2, -1)
    return data

contour_mask_max_extent_enclosing_circle = smw.register("ContourMaskMaxExtentEnclosingCircle", "test_contour_mask_max_extent_enclosing_circle", contour_mask_max_extent_enclosing_circle_helper, None, [float], TRI_STATE)
contour_mask_max_extent_enclosing_circle.__doc__ = """
Calculates the ratio of occupied pixels in the original image to the
number of pixels in minim enclosing circle of each contour.
The mask returned has objects that are <= ratio in extent

Args:
    data: numpy array of example
    ratio: ratio for comparison

Returns:
    Transformed data
"""

def contour_mask_range_extent_enclosing_circle_helper(data, lower_bound=0, upper_bound=128):
    data = cv2.Canny(data, 50, 150)
    org = np.copy(data)
    contours, hierarchy = cv2.findContours(data, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if get_circular_extent(org, c) <= lower_bound and
                get_circular_extent(org, c) <= upper_bound]
    data = np.zeros(data.shape, data.dtype)
    cv2.drawContours(data, contours, -1, 255, 2, -1)
    return data

contour_mask_range_extent_enclosing_circle = smw.register("ContourMaskRangeExtentEnclosingCircle", "test_contour_mask_range_extent_enclosing_circle", contour_mask_range_extent_enclosing_circle_helper, None, [int, int], TRI_STATE)
contour_mask_range_extent_enclosing_circle.__doc__ = """
Calculates the ratio of occupied pixels in the original image to the
number of pixels in minim enclosing circle of each contour.
The mask returned has objects are >= lower_bound and <= upper_bound

Args:
    data: numpy array of example
    lower_bound: lower bound
    upper_bound: upper bound

Returns:
    Transformed data
"""

def contour_mask_min_aspect_ratio_helper(data, ratio=0.5):
    data = cv2.Canny(data, 50, 150)
    contours, hierarchy = cv2.findContours(data, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if aspect_ratio(c) >= ratio]
    data = np.zeros(data.shape, data.dtype)
    cv2.drawContours(data, contours, -1, 255, 2, -1)
    return data

contour_mask_min_aspect_ratio = smw.register("ContourMaskMinAspectRatio", "test_contour_mask_min_aspect_ratio", contour_mask_min_aspect_ratio_helper, None, [float], TRI_STATE)
contour_mask_min_aspect_ratio.__doc__ = """
Returns a mask of all contours that have an aspect ratio >= ratio

Args:
    data: numpy array of example
    ratio: ratio for comparison

Returns:
    Transformed data
"""

def contour_mask_max_aspect_ratio_helper(data, ratio=0.5):
    data = cv2.Canny(data, 50, 150)
    contours, hierarchy = cv2.findContours(data, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if aspect_ratio(c) <= ratio]
    data = np.zeros(data.shape, data.dtype)
    cv2.drawContours(data, contours, -1, 255, 2, -1)
    return data

contour_mask_max_aspect_ratio = smw.register("ContourMaskMinAspectRatio", "test_contour_mask_max_aspect_ratio", contour_mask_max_aspect_ratio_helper, None, [float], TRI_STATE)
contour_mask_max_aspect_ratio.__doc__ = """
Returns a mask of all contours that have an aspect ratio <= ratio

Args:
    data: numpy array of example
    ratio: ratio for comparison

Returns:
    Transformed data
"""

def contour_mask_range_aspect_ratio_helper(data, lower_bound=0, upper_bound=128):
    data = cv2.Canny(data, 50, 150)
    contours, hierarchy = cv2.findContours(data, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if lower_bound <= aspect_ratio(c) <= upper_bound]
    data = np.zeros(data.shape, data.dtype)
    cv2.drawContours(data, contours, -1, 255, 2, -1)
    return data

contour_mask_range_aspect_ratio = smw.register("ContourMaskRangeAspectRatio", "test_contour_mask_range_aspect_ratio", contour_mask_range_aspect_ratio_helper, None, [int, int], TRI_STATE)
contour_mask_range_aspect_ratio.__doc__ = """
Returns a mask of all contours that have an aspect ratio >= lower_bound and <= upper_bound

Args:
    data: numpy array of example
    lower_bound: lower bound
    upper_bound: upper bound

Returns:
    Transformed data
"""

def contour_mask_min_extent_helper(data, boundary=128):
    data = cv2.Canny(data, 50, 150)
    contours, hierarchy = cv2.findContours(data, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if extent(c) >= boundary]
    data = np.zeros(data.shape, data.dtype)
    cv2.drawContours(data, contours, -1, 255, 2, -1)
    return data

contour_mask_min_extent = smw.register("ContourMaskMinExtent", "test_contour_mask_min_extent", contour_mask_min_extent_helper, None, [int], TRI_STATE)
contour_mask_min_extent.__doc__ = """
Ratio of the contour area to the bounding rectangle area

Args:
    data: numpy array of example
    boundary: mask will return contours >= boundary in extent

Returns:
    Transformed data
"""

def contour_mask_max_extent_helper(data, boundary=128):
    data = cv2.Canny(data, 50, 150)
    contours, hierarchy = cv2.findContours(data, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if extent(c) <= boundary]
    data = np.zeros(data.shape, data.dtype)
    cv2.drawContours(data, contours, -1, 255, 2, -1)
    return data

contour_mask_max_extent = smw.register("ContourMaskMaxExtent", "test_contour_mask_max_extent", contour_mask_max_extent_helper, None, [int], TRI_STATE)
contour_mask_max_extent.__doc__ = """
Returns a mask of all contours that have an extent >= ratio.
Extent is the ratio of object area to bounding
rectangle area.

Args:
    data: numpy array of example
    boundary: mask will return contours <= boundary in extent

Returns:
    Transformed data
"""

def contour_mask_range_extent_helper(data, lower_bound=0, upper_bound=128):
    data = cv2.Canny(data, 50, 150)
    contours, hierarchy = cv2.findContours(data, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if lower_bound <= extent(c) <= upper_bound]
    data = np.zeros(data.shape, data.dtype)
    cv2.drawContours(data, contours, -1, 255, 2, -1)
    return data

contour_mask_range_extent = smw.register("ContourMaskRangeExtent", "test_contour_mask_range_extent", contour_mask_range_extent_helper, None, [int, int], TRI_STATE)
contour_mask_range_extent.__doc__ = """
Returns a mask of all contours that have an aspect ratio
>= lower_bound and <= upper_bound

Args:
    data: numpy array of example
    lower_bound: lower bound
    upper_bound: upper bound

Returns:
    Transformed data
"""

def contour_mask_min_solidity_helper(data, boundary=128):
    data = cv2.Canny(data, 50, 150)
    contours, hierarchy = cv2.findContours(data, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if solidity(c) >= boundary]
    data = np.zeros(data.shape, data.dtype)
    cv2.drawContours(data, contours, -1, 255, 2, -1)
    return data

contour_mask_min_solidity = smw.register("ContourMaskMinSolidity", "test_contour_mask_min_solidity", contour_mask_min_solidity_helper, None, [int], TRI_STATE)
contour_mask_min_solidity.__doc__ = """
Solidity is the ratio of contour area to its convex hull area

Args:
    data: numpy array of example
    boundary: mask will return contours >= boundary in extent

Returns:
    Transformed data
"""

def contour_mask_max_solidity_helper(data, boundary=0.5):
    data = cv2.Canny(data, 50, 150)
    contours, hierarchy = cv2.findContours(data, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if solidity(c) <= boundary]
    data = np.zeros(data.shape, data.dtype)
    cv2.drawContours(data, contours, -1, 255, 2, -1)
    return data

contour_mask_max_solidity = smw.register("ContourMaskMaxSolidity", "test_contour_mask_max_solidity", contour_mask_max_solidity_helper, None, [float], TRI_STATE)
contour_mask_max_solidity.__doc__ = """
Solidity is the ratio of contour area to its convex hull area

Args:
    data: numpy array of example
    boundary: mask will return contours <= boundary in extent

Returns:
    Transformed data
"""

def contour_mask_range_solidity_helper(data, lower_bound=0.5, upper_bound=1.0):
    data = cv2.Canny(data, 50, 150)
    contours, hierarchy = cv2.findContours(data, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if lower_bound <= solidity(c) <= upper_bound]
    data = np.zeros(data.shape, data.dtype)
    cv2.drawContours(data, contours, -1, 255, 2, -1)
    return data

contour_mask_range_solidity = smw.register("ContourMaskRangeSolidity", "test_contour_mask_range_solidity", contour_mask_range_solidity_helper, None, [float, float], TRI_STATE)
contour_mask_range_solidity.__doc__ = """
Solidity is the ratio of contour area to its convex hull area

Args:
    data: numpy array of example
    lower_bound: lower bound
    upper_bound: upper bound

Returns:
    Transformed data
"""

def contour_mask_min_equ_diameter_helper(data, boundary=0.5):
    data = cv2.Canny(data, 50, 150)
    contours, hierarchy = cv2.findContours(data, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if equ_diameter(c) >= boundary]
    data = np.zeros(data.shape, data.dtype)
    cv2.drawContours(data, contours, -1, 255, 2, -1)
    return data

contour_mask_min_equ_diameter = smw.register("ContourMaskMinEquDiameter", "test_contour_mask_min_equ_diameter", contour_mask_min_equ_diameter_helper, None, [int], TRI_STATE)
contour_mask_min_equ_diameter.__doc__ = """
Equivalent Diameter is the diameter of the circle whose area is same as the contour area

Args:
    data: numpy array of example
    boundary: mask will return contours >= boundary in extent

Returns:
    Transformed data
"""

def contour_mask_max_equ_diameter_helper(data, boundary=10):
    data = cv2.Canny(data, 50, 150)
    contours, hierarchy = cv2.findContours(data, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if equ_diameter(c) <= boundary]
    data = np.zeros(data.shape, data.dtype)
    cv2.drawContours(data, contours, -1, 255, 2, -1)
    return data

contour_mask_max_equ_diameter = smw.register("ContourMaskMaxEquDiameter", "test_contour_mask_max_equ_diameter", contour_mask_max_equ_diameter_helper, None, [int], TRI_STATE)
contour_mask_max_equ_diameter.__doc__ = """
Equivalent Diameter is the diameter of the circle whose area is same as the contour area

Args:
    data: numpy array of example
    boundary: mask will return contours <= boundary in extent

Returns:
    Transformed data
"""

def contour_mask_range_equ_diameter_helper(data, lower_bound=0, upper_bound=10):
    data = cv2.Canny(data, 50, 150)
    contours, hierarchy = cv2.findContours(data, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if lower_bound <= equ_diameter(c) <= upper_bound]
    data = np.zeros(data.shape, data.dtype)
    cv2.drawContours(data, contours, -1, 255, 2, -1)
    return data

contour_mask_range_equ_diameter = smw.register("ContourMaskRangeEquDiameter", "test_contour_mask_range_equ_diameter", contour_mask_range_equ_diameter_helper, None, [int, int], TRI_STATE)
contour_mask_range_equ_diameter.__doc__ = """
Equivalent Diameter is the diameter of the circle whose area is same as the contour area

Args:
    data: numpy array of example
    lower_bound: lower bound
    upper_bound: upper bound

Returns:
    Transformed data
"""

# functions that return lists - TODO need to find some way to make their return values work in the data_pair scheme
def mean(data_pair):
    # TODO change the way values are returned
    """Calculates the mean

    Args:
        data_pair: given datapair

    Returns:
        [mean, mean]
    """
    m = 0

    data_list = []
    for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
        instances = cp.deepcopy(data_set.get_instances())

        for instance in instances:
            data = instance.get_stream().get_data()

            m = cv2.mean(data)

            instance.get_stream().set_data(data)

        data_list.append(m)

    return data_list

# not in primitive set because it returns a list
def my_sum(data_pair):
    # TODO change the way values are returned
    """Calculates the sum of array elements

    Args:
        data_pair: given datapair

    Returns:
        [sum, sum]
    """
    summation = 0

    data_list = []
    for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
        instances = cp.deepcopy(data_set.get_instances())

        for instance in instances:
            data = instance.get_stream().get_data()

            summation = cv2.sumElems(data)

            instance.get_stream().set_data(data)

        data_list.append(summation)

    return data_list

def correlation_helper(data, second_data):
    return signal.correlate2d(data, second_data, 'same')

correlation = smw_2.register("Correlation", "test_correlation", correlation_helper, None, [], TRI_STATE)
correlation.__doc__ = """
Returns data where data_out[x][y] = 0 if data[x][y] > alpha
* second_data[x][y] else data_out[x][y] = data[x][y]

Args:
    data: given numpy array
    second_data: given numpy array

Returns:
    Transformed data
"""

# do not include in gp_framwork_helper

def min_max_loc(data_pair):
    """Finds the minimum, maximum and their location

    Args:
        data_pair: given datapair

    Returns:
        [min, min] [max, max] [min location, min location] [max location, max location]
    """
    my_minimum = 0
    my_maximum = 0
    min_list = []
    max_list = []
    min_location = None
    max_location = None
    min_location_list = []
    max_location_list = []

    for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
        instances = cp.deepcopy(data_set.get_instances())

        for instance in instances:
            data = instance.get_stream().get_data()

            my_minimum, my_maximum, min_location, max_location = cv2.minMaxLoc(data)

            instance.get_stream().set_data(data)

        min_list.append(my_minimum)
        max_list.append(my_maximum)
        min_location_list.append(min_location)
        max_location_list.append(max_location)

    return min_list, max_list, min_location_list, max_location_list


# do not include in gp_framework_helper

def scalar_pixel_sum(data_pair):
    """TODO

    Sums each pixel in the image

    Args:
        data_pair: given datapair
    Returns:
        list of int
    """
    sum_list = []
    for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
        instances = cp.deepcopy(data_set.get_instances())

        for instance in instances:
            data = instance.get_stream().get_data()
            """"""""""""""""""""""""""""""""
            summation = data.sum
            sum_list.append(summation)
            """"""""""""""""""""""""""""""""
    return sum_list

# functions for the dual color algorithm


# do not include in gp_framework_helper

def get_image_sample(data, x=0, y=0, kernel_size=7, hole_size=3):
    """Returns a kernel_size x kernel_size np.array of data centered about x,y
    The data is set to zero for the items within hole_size centered about x,y

    Args:
        data: 2 dimensional numpy array
        x: x index
        y: y index
        kernel_size: size of kernel
        hole_size: size of the hole

    Returns:
        Transformed data pair
    """
    # create a destination for the median data
    sampled_data = np.zeros((kernel_size, kernel_size), type(data))

    # get half of the kernel size - kernel is an odd integer in length, int division will return int
    kernel_size = kernel_size // 2
    hole_size = hole_size // 2

    # get data dimensions - the value returned is the length not the max index
    # max index = data_max - 1
    (data_max_x, data_max_y) = data.shape
    data_max_x -= 1
    data_max_y -= 1
    samplex=0
    for i in range(-1*kernel_size, kernel_size + 1):
        sampley = 0
        for j in range(-1*kernel_size, kernel_size + 1):
            # dont sample in the hole
            if abs(i) > hole_size or abs(j) > hole_size:
                # use reflection if the kernel window extends beyond the boundry of the image

                if x+i < 0:
                    index_x = -1*(x+i)
                elif x+i > data_max_x:
                    index_x = x - kernel_size*2
                else:
                    index_x = x + i

                if y+j < 0:
                    index_y = -1*(y+j)
                elif y+j > data_max_y:
                    index_y = y - kernel_size*2
                else:
                    index_y = y + j

                sampled_data[samplex][sampley] = data[index_x][index_y]
            sampley += 1
        samplex += 1
    return sampled_data


# do not include this function in the GPFramework function list

def get_image_values(data, x=0, y=0, kernel_size=7, hole_size=3):
    """Returns a numpy array of the pixel values not within the hole

    Args:
        data: 2 dimensional numpy array
        x: x offset into the array
        y: y offset into the array
        kernel_size: size of the convolutional kernel
        hole_size: size of the hole to not sample from

    Returns:
        list
    """
    kernel_size = check_kernel_size(kernel_size)
    hole_size = check_kernel_size(hole_size)

    # create a destination for the data
    list = []

    # get half of the kernel size - kernel is an odd integer in length, int division will return int
    kernel_size /= 2
    hole_size /= 2

    # get data dimensions - the value returned is the length not the max index
    # max index = data_max - 1
    (data_max_x, data_max_y) = data.shape
    data_max_x -= 1
    data_max_y -= 1



    for i in range(-1*kernel_size, kernel_size + 1):
        for j in range(-1*kernel_size, kernel_size + 1):
            # dont sample in the hole
            if abs(i) > hole_size or abs(j) > hole_size:
                # use reflection if the kernel window extends beyond the boundry of the image

                if x+i < 0:
                    index_x = -1*(x+i)
                elif x+i > data_max_x:
                    index_x = x - kernel_size*2
                else:
                    index_x = x + i

                if y+j < 0:
                    index_y = -1*(y+j)
                elif y+j > data_max_y:
                    index_y = y - kernel_size*2
                else:
                    index_y = y + j

                list.append(data[index_x][index_y])
    list = np.array(list)
    return list

# do not include in gp_framework_helper

def not_in_the_hole(x=0,y=0,kernel_size=7, hole_size=3):
    """Returns true if x, y are not in the hole, else returns false

    Args:
        data: 2 dimensional numpy array
        x: x index
        y: y index
        kernel_size: size of kernel
        hole_size: size of the hole

    Returns:
        Boolean
    """
    # get half of the kernel size - kernel is an odd integer in length, int division will return int
    k= kernel_size / 2
    h= hole_size / 2

    hole_min = k-h
    hole_max = k + h

    if x >= hole_min and x <= hole_max and y >= hole_min and y <= hole_max:
        return False
    else:
        return True

def median_filter_hole_helper(data, kernel_size=7, hole_size=3):
    kernel_size = check_kernel_size(kernel_size)
    hole_size = check_kernel_size(hole_size)
    # create a destination for the median data
    median_data = np.zeros(data.shape, data.dtype)

    # get data dimensions - the value returned is the length not the max index
    # max index = data_max - 1
    (data_max_x, data_max_y) = data.shape

    for x in range(0, data_max_x):
        for y in range(0, data_max_y):
            # get a kernel_size x kernel_size sample of the data centered around x,y
            # with any values within the hole_size set = 0
            sample = get_image_sample(data, x, y, kernel_size, hole_size)

            # get only samples that are not in the hole
            sample_list = []
            for i in range(0,kernel_size):
                for j in range(0,kernel_size):
                    if not_in_the_hole(i,j,kernel_size,hole_size):
                        sample_list.append(sample[i,j])
            if len(sample_list) == 0:
                sample_list = [0]

            median = np.median(sample_list)
            # assign the median to the center pixel in the new image
            median_data[x,y] = median
    return data

median_filter_hole = smw.register("MedianFilterHole", "test_median_filter_hole", median_filter_hole_helper, None, [int, int], TRI_STATE)
median_filter_hole.__doc__ = """
Median filter with a hole in the middle that is not included in the
median calculation

Args:
    data: numpy array
    kernel_size: size of kernel window
    hole_size: size of the window to exclude from the calculation

Returns:
    Transformed data
"""

def std_deviation_hole_helper(data, kernel_size=7, hole_size=3):
    kernel_size = check_kernel_size(kernel_size)
    hole_size = check_kernel_size(hole_size)
    # create a destination for the std_deviation data
    new_data = np.zeros(data.shape, data.dtype)

    # get data dimensions - the value returned is the length not the max index
    # max index = data_max - 1
    (data_max_x, data_max_y) = data.shape
    data_max_index_x = data_max_x - 1
    data_max_index_y = data_max_y - 1

    for x in range(0, data_max_x):
        for y in range(0, data_max_y):
            # get a kernel_size x kernel_size sample of the data centered around x,y
            # with any values within the hole_size set = 0
            # sample = get_image_values(data, x, y, kernel_size, hole_size)
            sample = get_image_sample(data, x, y, kernel_size, hole_size)
            # get only samples that are not in the hole
            sample_list = []
            for i in range(0,kernel_size):
                for j in range(0,kernel_size):
                    if not_in_the_hole(i,j,kernel_size, hole_size):
                        sample_list.append(sample[i,j])
            if len(sample_list) == 0:
                sample_list = [0]

            std_deviation = np.std(sample_list)
            # assign the value to the current pixel in the new image
            new_data[x][y] = std_deviation
    return new_data

std_deviation_hole = smw.register("StdDeviationHole", "test_std_deviation_hole", std_deviation_hole_helper, None, [int, int], TRI_STATE)
std_deviation_hole.__doc__ = """
Compute the standard deviation of the image excluding the pixel values
in the hole

Args:
    data: numpy array of example
    kernel_size: size of kernel window
    hole_size: size of the window to exclude from the calculation

Returns:
    Transformed data
"""

def std_deviation_hole_custom_helper(data, second_data, kernel_size=7, hole_size=3):
    kernel_size = check_kernel_size(kernel_size)
    hole_size = check_kernel_size(hole_size)
    data = cv2.subtract(data, second_data)
    data = cv2.multiply(data, data)
    population = kernel_size * kernel_size
    data = cv2.divide(data, population)
    data = cv2.sqrt(abs(data))
    return data

std_deviation_hole_custom = smw_2.register("StdDeviationHoleCustom", "test_std_deviation_hole_custom", std_deviation_hole_custom_helper, None, [int, int], TRI_STATE)
std_deviation_hole_custom.__doc__ = """
Compute the standard deviation of the image excluding the pixel values
in the hole

Args:
    data: given numpy array
    second_data: given numpy array
    kernel_size: size of kernel window
    hole_size: size of the window to exclude from the calculation

Returns:
    Transformed data
"""

def mean_with_hole_helper(data, kernel_size=7, hole_size=3):
    kernel_size = check_kernel_size(kernel_size)
    hole_size = check_kernel_size(hole_size)
    kernel = np.ones((kernel_size, kernel_size), data.dtype)
    zeros = np.zeros((hole_size, hole_size), data.dtype)

    # find center
    x = kernel_size // 2
    h = hole_size // 2

    kernel[x-h : x+h+1, x-h : x+h+1 ] = zeros[0:hole_size, 0:hole_size]

    new_data = cv2.filter2D(data,-1,kernel,borderType=cv2.BORDER_REFLECT_101)
    cv2.divide(new_data, kernel_size * kernel_size, new_data)
    return data

mean_with_hole = smw.register("MeanWithHole", "test_mean_with_hole", mean_with_hole_helper, None, [int, int], TRI_STATE)
mean_with_hole.__doc__ = """
Compute the mean of the image excluding the pixel values in the hole

Args:
    data: given numpy array
    second_data: given numpy array
    kernel_size: size of kernel window
    hole_size: size of the window to exclude from the calculation

Returns:
    Transformed data
"""

def prerejection_helper(data, kernel_size=7, hole_size=3, alpha=1):
    kernel_size = check_kernel_size(kernel_size)
    hole_size = check_kernel_size(hole_size)
    std_dev = std_deviation_hole_helper(data, kernel_size, hole_size)
    new_data = set_to_zero_if_greater_than_data_and_factor_helper(data, std_dev, alpha)

    return new_data

prerejection = smw.register("Prerejection", "test_prerejection", prerejection_helper, None, [int, int, float], TRI_STATE)
prerejection.__doc__ = """
Prerejection filter for dual color algorithm

if pixel std_deviation(data[x,y]) > threshold ---> data[x,y] = 0
Replace pixel with 0 if intensity > local stdard deviation by a given factor

Args:
    data: numpy array
    kernel_size: size of kernel for std deviation
    hole_size: size of the hole for std deviation
    alpha: scaling factor for std dev comparison

Returns:
    Transformed data pair
"""

def rms_hole_helper(data, kernel_size=7, hole_size=3):
    kernel_size = check_kernel_size(kernel_size)
    hole_size = check_kernel_size(hole_size)
    # data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    # create a destination for the std_deviation data
    new_data = np.zeros(data.shape, data.dtype)

    # get data dimensions - the value returned is the length not the max index
    # max index = data_max - 1
    (data_max_x, data_max_y) = data.shape

    for x in range(0, data_max_x):
        for y in range(0, data_max_y):
            # get a kernel_size x kernel_size sample of the data centered around x,y
            # with any values within the hole_size set = 0
            sample = get_image_sample(data,x,y, kernel_size, hole_size)
            # get only samples that are not in the hole
            sample_list = []
            for i in range(0,kernel_size):
                for j in range(0,kernel_size):
                    if not_in_the_hole(i,j,kernel_size, hole_size):
                        sample_list.append(sample[i,j])
            if len(sample_list) == 0:
                sample_list = [0]

            sample_squared = np.multiply(sample_list, sample_list)
            mean_sample_squared = np.average(sample_squared)

            mean = np.average(sample_list)
            mean_squared = mean * mean

            difference = mean_sample_squared - mean_squared
            rms = math.sqrt(abs(difference))

            # assign the value to the current pixel in the new image
            new_data[x][y] = rms
    return data

rms_hole = smw.register("rmsHole", "test_rms_hole", rms_hole_helper, None, [int, int], TRI_STATE)
rms_hole.__doc__ = """
Compute the RMS of the image excluding the pixel values in the hole
rms = sqrt(blur(I*I) - blur(I)*blur(I))
rms = sqrt(avg(I*I) - avg(I)avg(I)

Args:
    data: numpy array of example
    kernel_size: size of kernel window
    hole_size: size of the hole to exclude from the calculation

Returns:
    Transformed data
"""

def mark_targets(data_pair, kernel_size=7):
    # NOT included in framework currently
    """Consolidate points that fall within kernel_size window to a single point
    based on its brightness

    Args:
        data_pair: given datapair
        kernel_size: size of kernel window

    Returns:
        Transformed data pair
    """
    if data_pair.get_caching_mode():
        raise ValueError("Method not currently cached")
    debug = False

    data_list = []
    for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
        instances = cp.deepcopy(data_set.get_instances())

        for instance in instances:
            if debug: print('Mark Targets Get Instance')
            data = instance.get_stream().get_data()

            kernel_size = check_kernel_size(kernel_size)

            # get data dimensions - the value returned is the length not the max index
            (data_max_index_x, data_max_index_y) = data.shape

            # image could have negative numbers
            image_min = np.min(data)
            if image_min > 0:
                image_min = 0
            image_max = np.max(data)

            if debug: print('Mark Targets Before Merge Pixels')
            # loop over entire image
            for x in range(0, data_max_index_x):
                for y in range(0, data_max_index_y):

                    box_max_x = min(x + kernel_size + 1, data_max_index_x)
                    box_max_y = min(y + kernel_size + 1, data_max_index_y)
                    # get a kernel_size x kernel_size sample of the image
                    sample = data[x: box_max_x, y: box_max_y]

                    # determine if there are any occupied pixels
                    maximum = np.max(sample)
                    if maximum > image_min:
                        coordinates = []
                        brightness = []
                        for i in range(x, box_max_x):
                            for j in range(y, box_max_y):
                                # store the x,y coordinates of any pixel whose value = maximum
                                if data[i,j] == maximum:
                                    coordinates.append((i,j))
                                    brightness.append(data[i,j])
                                data[i,j] = image_min

                        # calculate centroid for target
                        x_sum = 0
                        y_sum = 0
                        value = 0
                        for (detected_x, detected_y), bright in zip(coordinates,brightness):
                            x_sum += detected_x
                            y_sum += detected_y
                            value += bright
                        centroid_x = x_sum // len(coordinates)
                        centroid_y = y_sum // len(coordinates)
                        # mark centroid
                        data[centroid_x, centroid_y] = value / len(coordinates)

            if debug: print('Mark Targets After Merge Pixels')
            """
            Something is going on with cv2.rectangle - It will not plot anything past about 1/4 of the width of the image
            it does do its full height
            THe manual plot point below does not appear in the image
            """
            if debug: print('Mark Targets Before Output Image')
            num_targets = 0
            output_image = np.zeros(data.shape, dtype=data.dtype)
            for x in range(0, data_max_index_x):
                for y in range(0, data_max_index_y):
                    if data[x,y] > image_min:
                        output_image[x,y] = data[x,y]
                        #num_targets +=1
                        #point0 = (max(x - kernel_size/2, 0), max(y - kernel_size/2, 0))
                        #point1 = (min(x + kernel_size/2, data_max_index_x), min(y + kernel_size/2,data_max_index_y))
                        #print 'box', point0, point1
                        #cv2.rectangle(output_image, point0, point1, image_max)
            if debug: print('Mark Targets After Output Image')

            instance.get_stream().set_data(output_image)

        new_data_set = EmadeData(instances)
        data_list.append(new_data_set)
    data_pair.set_train_data(data_list[0])
    data_pair.set_test_data(data_list[1])
    if debug: print('Mark Targets Return New Data Pair')
    if debug: print('Data Pair', repr(data_pair))

    return data_pair



def get_target_points(data_pair, mode=TriState.FEATURES_TO_FEATURES):
    # NOT included in framework currently
    """Perform fft on a dataset

    Args:
        data_pair: given datapair
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Transformed data pair
    """
    if data_pair.get_caching_mode():
        raise ValueError("Method not currently cached")
    data_list = []
    for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
        instances = cp.deepcopy(data_set.get_instances())

        for instance in instances:
            if mode is TriState.FEATURES_TO_FEATURES:
                data = instance.get_features().get_data()
            elif mode is TriState.STREAM_TO_STREAM or TriState.STREAM_TO_FEATURES:
                data = instance.get_stream().get_data()

            target_points = []
            (data_max_index_x, data_max_index_y) = data.shape

            # image could have negative numbers
            image_min = np.min(data)
            if image_min > 0:
                image_min = 0

            # loop over entire image
            for x in range(0, data_max_index_x):
                for y in range(0, data_max_index_y):
                    if data[x,y] > image_min:
                        target_points.append([x,y])
            target_points = np.array(target_points)

            if mode is TriState.FEATURES_TO_FEATURES:
                instance.get_features().set_data(target_points)
            elif mode is TriState.STREAM_TO_STREAM:
                instance.get_stream().set_data(data)
            elif mode is TriState.STREAM_TO_FEATURES:
                old_features = instance.get_features().get_data()
                new_features = np.concatenate((old_features.flatten(), target_points.flatten()))

                instance.get_features().set_data(
                                                 np.reshape(new_features, (1,-1))
                                                )

        new_data_set = EmadeData(instances)
        data_list.append(new_data_set)
    data_pair.set_train_data(data_list[0])
    data_pair.set_test_data(data_list[1])

    return data_pair
