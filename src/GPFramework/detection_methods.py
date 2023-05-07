"""
Programmed by Austin Dunn
Implements a number of image data methods designed for object detection
"""
import copy as cp
import cv2
import numpy as np
from scipy import signal
from scipy import ndimage
from scipy.stats import chi2
from spectral import rx
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.morphology import disk
import skimage.filters
import time
import os
from functools import partial
import gc
import traceback
import sep
import sys

from GPFramework.constants import TriState, Axis, TRI_STATE
from GPFramework.wrapper_methods import RegistryWrapperS, RegistryWrapperB, RegistryWrapper
from GPFramework.spatial_methods import check_kernel_size
from GPFramework.data import EmadeDataPair
from GPFramework.cache_methods import check_cache_read, check_cache_write

# helper method do NOT register into framework
def broadcast_inner(X, Y, axis=-1):
    """Broadcasts the inner product over an axis.

    Takes two arrays X and Y of the same shape, and an axis r, and returns an array Z where
    Z[i0,...,ir-1,ir+1,...,in-1] = sum(X[i0,...,ir-1,:,ir+1,...,in-1] * Y[i0,...,ir-1,:,ir+1,...,in-1])
    """
    s1 = X.shape
    s2 = Y.shape
    if s1 != s2:
        raise ValueError('Arrays must be of the same shape')
    axis = axis % len(s1)
    order = list(range(len(s1)))
    del order[axis]
    order.append(axis)
    X = np.transpose(X, order)
    Y = np.transpose(Y, order)
    p = np.product(X.shape[:-1])
    res_shape = X.shape[:-1]
    X = np.reshape(X, (p, X.shape[-1]))
    Y = np.reshape(Y, (p, Y.shape[-1]))
    res = np.sum(X * Y, axis=1)
    return np.reshape(res, res_shape)

# helper method do NOT register into framework
def multinormal_pdf(X, mu, det, covI):
    """Broadcasts the multivariate normal distribution pdf over an array"""
    Y = X - mu
    D = broadcast_inner(Y @ covI, Y, -1)
    E = np.exp(-D / 2.0)
    k = X.shape[-1]
    denom = np.sqrt(np.power(2 * np.pi, k) * det)
    return E / denom

# helper method do NOT register into framework
def truth_points(images, truths, r=0):
    """Returns a list of all pixel values in the 2r+1 by 2r+1 square about every truth point.

    images : (n x c x w x h) array
    truths : Each entry of truths should be a list of points in the range [0, w) x [0, h)
    r : Non-negative integer
    """
    w = 2*r + 1
    h = 2*r + 1
    pts = []
    _, channels, imw, imh = images.shape
    for image, truth in zip(images, truths):
        averages = [np.mean(channel) for channel in image]

        for x, y in truth:
            x = x + r
            y = y + r
            x_grid = int(np.round(x))
            y_grid = int(np.round(y))
            dx = x_grid - x
            dy = y_grid - y
            M = np.float32([[1,0,dx],[0,1,dy]])
            shifted = (cv2.warpAffine(channel, M, (imw + 2*r, imh + 2*r), borderMode=cv2.BORDER_CONSTANT, borderValue=np.float64(avg)) for channel, avg in zip(image, averages))
            windows = [shift[x_grid-2*r:x_grid+1,y_grid-2*r:y_grid+1] for shift in shifted]
            for i in range(w):
                for j in range(h):
                    pts.append([window[i][j] for window in windows])

    return np.array(pts)

# helper method do NOT register into framework
def normal_transform_helper(train_images, train_truths, test_images, r=0):
    """Object detection transformation based on the multivariate normal distribution.
    Finds the multivariate distribution of pixel values in the 2r+1 by 2r+1
    window about each truth point in the training set, and returns the pdf of the distribution
    applied at each pixel in the training and testing sets.

    train_images : (m x c x w x h) array
    train_truths : Each entry of truths should be a list of points in the range [0, w) x [0, h)
    test_images : (n x c x w x h) array
    r : Non-negative integer
    """
    train_images = np.array([np.transpose(image, (2, 0, 1)) for image in train_images])
    test_images = np.array([np.transpose(image, (2, 0, 1)) for image in test_images])
    pts = truth_points(train_images, train_truths, r)
    avg = np.mean(pts, axis=0)
    cov = np.cov(pts.T)
    det = np.linalg.det(cov)
    covI = np.linalg.inv(cov)
    train_images = np.transpose(train_images, (0, 2, 3, 1))
    test_images = np.transpose(test_images, (0, 2, 3, 1))
    train_results = multinormal_pdf(train_images, avg, det, covI)
    test_results = multinormal_pdf(test_images, avg, det, covI)
    return train_results, test_results

# Do not include in framework. Helper method for other detection methods
def gaussian_kernel(size=21, std=3):
    """Returns a 2D Gaussian kernel array

    Args:
        size: defines the (size x size) kernel matrix
        std: standard deviation of gaussian (sigma)
    """
    # generate kernel data
    gkern_1d = signal.gaussian(size, std=std).reshape(size, 1)
    gkern_2d = np.outer(gkern_1d, gkern_1d)
    # normalize and return kernel matrix
    return (gkern_2d - np.mean(gkern_2d)) / np.std(gkern_2d)

# Do not include in framework. Helper method for other detection methods
def kernel_estimation(instances, kernel_size):
    """Estimates a kernel using a point-wise statistical average

    Args:
        instances:   list of (k, m, n) mutli-channel images
        kernel_size: defines the (kernel_size x kernel_size) kernel matrix

    Returns:
        A kernel matrix (kernel_size x kernel_size)
    """
    kernel_size = check_kernel_size(kernel_size)
    if kernel_size == 1:
        kernel_size = 3

    kernel_list = []
    z = kernel_size // 2
    w = z + 1
    for instance in instances:
        # Load the data
        truth = instance.get_target()
        stream = instance.get_stream().get_data()

        # average the channels together
        # handles arbitrary number of channels
        # expecting (k, m, n) format
        if len(stream.shape) == 3:
            temp = np.zeros((stream.shape[1], stream.shape[2]))
            for i in range(stream.shape[0]):
                temp += stream[i,:,:] / stream.shape[0]
            stream = temp
        elif len(stream.shape) == 1:
            raise ValueError("Data must have the shape of a image")

        # Reformat truth data into correct shape if necessary
        if len(truth.shape) > 2:
            truth = np.squeeze(truth, axis=0)
        if len(truth.shape) == 1:
            truth = np.expand_dims(truth, axis=0)

        # make padded image with mean
        stream = np.pad(stream, (w,), 'constant', constant_values=(np.mean(stream),))

        # iterate over all truth objects
        for t in truth:
            if np.sum(t) != -2.0:
                x = int(np.round(t[0]))
                y = int(np.round(t[1]))

                # area calculations
                width = x - (t[0] - .5)
                height = (t[1] + .5) - y
                a1 = width * height
                width = (t[0] + .5) - x
                height = (t[1] + .5) - y
                a2 = width * height
                width = x - (t[0] - .5)
                height = y - (t[1] - .5)
                a3 = width * height
                width = (t[0] + .5) - x
                height = y - (t[1] - .5)
                a4 = width * height

                area_kern = np.array([[a1, a2],
                                        [a3, a4]])

                # adjust for padding
                x += w
                y += w

                # isolate region of interest around object truth
                slice_ = stream[x-z:x+w,y-z:y+w]
                # create rough kernel approximation
                temp_kernel = signal.convolve2d(slice_,
                                                area_kern,
                                                mode='same',
                                                fillvalue=np.mean(slice_))
                # normalize
                if np.std(temp_kernel) != 0:
                    temp_kernel = (temp_kernel - np.mean(temp_kernel)) / np.std(temp_kernel)
                kernel_list.append(temp_kernel)

    if len(kernel_list) > 0:
        kernel = np.mean(kernel_list, axis=0)
    else:
        raise ValueError("Every example has 0 truth positions")

    return kernel

# helper method do NOT register into framework
def round_options(x, m):
    if x == 0:
        return [0]
    if x == m:
        return [m - 1]
    return [i for i in [int(np.floor(x)), int(np.ceil(x))] if 0 <= i < m]

# helper method do NOT register into framework
def closest_grid(x, y, w, h):
    best = None
    best_dist = None
    for x2 in round_options(x, w):
        for y2 in round_options(y, h):
            dist = (x - x2)**2 + (y - y2)**2
            if best is None:
                best = (x2, y2)
                best_dist = dist
            if dist < best_dist:
                best = (x2, y2)
                best_dist = dist
    return best

# helper method do NOT register into framework
def subpixel_window(image, x, y, border, r):
    imw, imh = image.shape
    x_grid, y_grid = closest_grid(x - 0.5, y - 0.5, imw, imh)
    dx = x_grid - x + 0.5 - r
    dy = y_grid - y + 0.5 - r
    M = np.float32([[1,0,-dx],[0,1,-dy]])
    shifted = cv2.warpAffine(image, M, (imw + 2*r, imh + 2*r), borderMode=cv2.BORDER_CONSTANT, borderValue=border)
    return shifted[x_grid:x_grid+2*r+1,y_grid:y_grid+2*r+1]

# helper method do NOT register into framework
def average_kernel(images, truths, r):
    w = 2*r + 1
    h = 2*r + 1
    ker = np.zeros((w, h))
    n = 0

    for image, truth in zip(images, truths):
        im_avg = np.float64(np.mean(image))

        for x, y in truth:
            ker += subpixel_window(image, x, y, im_avg, r)
            n += 1

    ker /= n
    return ker

def detection_wrapper(p_name, helper_function, data_pair, ksize, *args):
    """Approximates a kernel based on truth data from the training set
       Then performs some form of template matching using the kernel as a template
       Note: implements STREAM_TO_STREAM

    Args:
        p_name:           name of primitive
        helper_function:  returns transformed data
        data_pair:        given datapair
        ksize:            defines the (size x size) kernel matrix
        args:             list of arguments

    Returns:
        Data Pair
    """
    # For debugging purposes let's print out method name
    print(p_name) ; sys.stdout.flush()

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
            method_string = p_name + "_" + str(ksize) + "_" + str(args).strip('[]').strip('()').replace(",", "_").replace(" ", "") + "_"

            # Combine the unique method name + arguments of the method + hash of the previous data
            # To form a unique key of the method call
            method_key = method_string + previous_hash

            overhead_time, method_row, cache_row, hit = check_cache_read(data_pair, database, method_key, oh_time_start)
            if hit: return data_pair

            eval_time_start = time.time()

        # generate the kernel
        train_instances = cp.deepcopy(data_pair.get_train_data().get_instances())
        kernel = kernel_estimation(train_instances, ksize)

        # Initialize where the data will be temporarily stored
        data_list = []
        # Iterate through train data then test data
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            # Copy the dataSet so as not to destroy original data
            instances = cp.deepcopy(data_set.get_instances())
            # Iterate over all points in the dataSet
            for instance in instances:
                # Load stream data
                data = instance.get_stream().get_data()

                # Transform data
                full_data = helper_function(data, kernel, *args)

                # Save new data
                instance.get_stream().set_data(full_data)

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
    gc.collect(); return data_pair

class RegistryWrapperD(RegistryWrapper):
    """
    This wrapper is specific to detection methods which estimate a kernel
    Before evaluating on the data (similar to a learner).

    Stores a mapping of primitives used in generating the PrimitiveSet

    The first object of input_types must be a EmadeDataPair

    Args:
        input_types: common inputs for every primitive stored in the wrapper
                     used to create a mapping between arg and index
                     example mapping created: {'EmadeDataPair0': 0, 'TriState1': 1}

    """
    def __init__(self, input_types=[]):
        super().__init__(input_types)

    def register(self, name, test_name, p_fn, input_types, output_type=EmadeDataPair):
        # create wrapped method
        wrapped_fn = partial(detection_wrapper, name, p_fn)

        # wrapped unit test
        if isinstance(test_name, str):
            unit_test = test_name
        else:
            raise ValueError("test_name must be a string (str)")

        # create a mapping for adding primitive to pset
        self.registry[name] = {"function": wrapped_fn,
                               "input_types": input_types,
                               "output_type": output_type,
                               "unit_test": unit_test}

        # return wrapped method
        return wrapped_fn

dew = RegistryWrapperS([EmadeDataPair, TriState, Axis])
dew_2 = RegistryWrapperS(2*[EmadeDataPair] + 2*[TriState] + 2*[Axis])
dewd = RegistryWrapperD([EmadeDataPair, int])
dewb = RegistryWrapperB()

def conv_channel_merge_helper(data, kernel):
    # handles arbitrary number of channels
    # expecting (m, n, k) format
    final = np.zeros((data.shape[0], data.shape[1]))
    for i in range(data.shape[2]):
        final += signal.convolve2d(data[:, :, i], kernel,
                                   mode='same',
                                   fillvalue=np.mean(data[:, :, i]))
    return final

conv_channel_merge = dewd.register("ConvolveChannelMerge", "test_conv_channel_merge", conv_channel_merge_helper, [])
conv_channel_merge.__doc__ = """
Performs convolution over multiple channels to merge them
Designed to be integrated with an object detection algorithm

Args:
    data:      numpy array
    kernel:    (n x n) kernel matrix

Returns:
    single channel image
"""

def cv2_template_matching_helper(data, kernel, ind=0):
    methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
               cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]
    # select the method
    method = methods[np.abs(ind) % len(methods)]

    return cv2.matchTemplate(data.astype(np.float32), kernel.astype(np.float32), method)

cv2_template_matching = dewd.register("Cv2TemplateMatching", "test_cv2_template_matching", cv2_template_matching_helper, [int])
cv2_template_matching.__doc__ = """
Performs template matching over an image with a given kernel
Designed to be integrated with an object detection algorithm (classifier)

Args:
    data_pair: given datapair
    kernel:    (n x n) kernel matrix
    ind:       comparison method use to compare template to the image

Returns:
    matched data with same shape as input
"""

def maximum_filter_helper(data, n_size=7, threshold=27.0):
    data_max = ndimage.filters.maximum_filter(data, n_size)
    maxima = (data == data_max)
    data_min = ndimage.filters.minimum_filter(data, n_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    return maxima

maximum_filter = dew.register("MaximumFilter", "test_maximum_filter", maximum_filter_helper, None, [int, float], TRI_STATE)
maximum_filter.__doc__ = """
Filters out all non local maxima pixels in an image

Args:
    data:      numpy array of example
    n_size:    size of (n x n) neighborhood around local maximum pixels
    threshold: selects which local maxima are valid

Returns:
    Transformed data
"""

def minimum_filter_helper(data, n_size=7, threshold=0.3):
    data_min = ndimage.filters.minimum_filter(data, n_size)
    minima = (data == data_min)
    diff = data_min < threshold
    minima[diff == 0] = 0

    return minima

minimum_filter = dew.register("MinimumFilter", "test_minimum_filter", minimum_filter_helper, None, [int, float], TRI_STATE)
minimum_filter.__doc__ = """
Filters out all non local minima pixels in an image

Args:
    data:      numpy array of example
    n_size:    size of (n x n) neighborhood around local maximum pixels
    threshold: selects which local maxima are valid

Returns:
    Transformed data
"""

def label_objects_helper(data):
    labeled, num_objects = ndimage.label(data)
    slices = ndimage.find_objects(labeled)
    x, y = [], []
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1)/2
        y.append(y_center)

    target = []
    for i,j in zip(x,y):
        target.append(np.array([j,i]))
    return np.array(target)

label_objects = dew.register("LabelObjects", "test_label_objects", label_objects_helper, None, [], TRI_STATE)
label_objects.__doc__ = """
Labels objects located in separate spaces of the image

Args:
    data: numpy array of example

Returns:
    Transformed data
"""

def label_by_com_helper(data, second_data):
    labeled, num_objects = ndimage.label(data)

    output = ndimage.measurements.center_of_mass(second_data, labeled, range(1, num_objects+1))

    return np.array([np.array([x[1], x[0]]) for x in output])

label_by_com = dew_2.register("LabelByCenterOfMass", "test_label_by_com", label_by_com_helper, None, [], TRI_STATE)
label_by_com.__doc__ = """
Labels objects located in separate spaces of the image
Centroids are calculated using center of mass

Args:
    data:        numpy array of example (thresholded image)
    second_data: numpy array of example (original image)

Returns:
    Transformed data
"""

def matched_filtering_2d_setup(data, ksize=11, std=7.5):
    ksize = check_kernel_size(ksize)
    if ksize == 1:
        ksize = 3
    gkern = gaussian_kernel(size=ksize, std=std)
    return data, { "gkern": gkern }

def matched_filtering_2d_helper(data, gkern=None):
    # Expects a 2D image so if we have a kxmxn let's average across k channels
    if len(data.shape) == 3:
        data = np.mean(data, axis=0)

    # grab the spatial dimensions of the image, along with
    # the spatial dimensions of the kernel
    (i_height, i_width) = data.shape[:2]
    (k_height, k_width) = gkern.shape[:2]

    # allocate memory for the output image, taking care to
    # "pad" the borders of the input image so the spatial
    # size (i.e., width and height) are not reduced
    pad = (k_width - 1) // 2
    data = cv2.copyMakeBorder(data, pad, pad, pad, pad,
                              cv2.BORDER_REPLICATE)
    output = np.zeros((i_height, i_width), dtype="float32")

    # loop over the input image, "sliding" the kernel across
    # each (x, y)-coordinate from left-to-right and top-to-bottom
    for y in np.arange(pad, i_height + pad):
        for x in np.arange(pad, i_width + pad):
            # extract the ROI (region of interest) of the image by extracting
            # the *center* region of the current (x, y)-coordinates
            roi = data[y-pad:y+pad+1, x-pad:x+pad+1]

            # normalize roi
            if np.std(roi) > 0:
                roi = (roi - np.mean(roi)) / np.std(roi)

            # perform the actual convolution by taking the
            # element-wise multiplicate between the ROI and
            # the kernel, then summing the matrix
            k = (roi * gkern).sum()

            # store the convolved value in the output (x,y)-
            # coordinate of the output image
            output[y - pad, x - pad] = k

    return output

matched_filtering_2d = dew.register("MatchedFiltering2D", "test_matched_filter_2D", matched_filtering_2d_helper, matched_filtering_2d_setup, [int, float], TRI_STATE)
matched_filtering_2d.__doc__ = """
Finds a bright object in the image which closely matches the kernel

Args:
    data:  numpy array of example
    ksize: size of the kernel (ksize, ksize)
    std:   standard deviation of kernel (sigma)

Returns:
    Transformed data
"""

def object_detection_helper(data, kernel, threshold=5.0, obj_thresh=90.0):
    if obj_thresh < 0:
        obj_thresh *= -1
    if threshold < 0:
        threshold *= -1
    # run SEP's matched_filtering
    objects = sep.extract(data, threshold, filter_kernel=kernel)
    # initialize a dict for mapping objects to scores
    prediction = []
    for obj in objects:
        # get radius of kernel
        t = kernel.shape[0] // 2
        s = t + 1

        # pad input data with kernel radius
        # without the padding objects near the edge of the image will error out
        padded_data = cv2.copyMakeBorder(data, s, s, s, s, cv2.BORDER_REPLICATE)

        # grab center of object and pad it
        # floors float values of coordinates
        x = int(obj['y']) + s
        y = int(obj['x']) + s

        # slice our region of interest out of our original data
        # this will have the same nxn shape as the kernel
        roi = padded_data[x-t:x+s,y-t:y+s]

        # normalize roi
        # this is to prevent larger objects from having a bias
        if np.std(roi) > 0:
            roi = (roi - np.mean(roi)) / np.std(roi)

        # add object to prediction if it meets threshold
        if (roi * kernel).sum() >= obj_thresh:
            prediction.append([obj['y'],obj['x']])

    # return coordinates of selected objects
    return np.array(prediction)

object_detection = dewd.register("ObjectDetection", "test_object_detection", object_detection_helper, [float, float])
object_detection.__doc__ = """
Uses a kernel to find objects in an image.
The kernel is then convoluted on the center of the objects found
(defined by documentation as "on-the-fly filtering")
The coordinates of the object with the highest convolution is returned

Our normalized convolution is used to score objects

https://sep.readthedocs.io/en/v1.0.x/api/sep.extract.html#sep.extract

Args:
    data:       numpy array of example
    kernel:     kernel matrix (size x size)
    threshold:  minimum pixel value needed object detection
    obj_thresh: threshold for selecting valid objects

Returns:
    coordinates of predicted object
"""

def sep_object_detection_helper(data, kernel, threshold=5.0, obj_thresh=70000.0):
    if obj_thresh < 0:
        obj_thresh *= -1
    if threshold < 0:
        threshold *= -1
    # run SEP's matched_filtering
    objects = sep.extract(data, threshold, filter_kernel=kernel)
    # find all objects greater than obj_thresh
    prediction = [[obj['y'],obj['x']] for obj in objects if obj['cflux'] >= obj_thresh]

    # return coordinates of selected objects
    return np.array(prediction)

sep_object_detection = dewd.register("SEPObjectDetection", "test_sep_object_detection", sep_object_detection_helper, [float, float])
sep_object_detection.__doc__ = """
Uses a kernel to find objects in an image.
The kernel is then convoluted on the center of the objects found
(defined by documentation as "on-the-fly filtering")
The coordinates of the object with the highest convolution is returned

SEP's convolution is used to score objects

https://sep.readthedocs.io/en/v1.0.x/api/sep.extract.html#sep.extract

Args:
    data:       numpy array of example
    kernel:     kernel matrix (size x size)
    threshold:  minimum pixel value needed object detection
    obj_thresh: threshold for selecting valid objects

Returns:
    coordinates of predicted object
"""

def log_detection_helper(data, min_sigma=1, max_sigma=30.0, num_sigma=10, threshold=0.1,
                            overlap=0.5):
    if max_sigma < 0:
        max_sigma *= -1
    if num_sigma < 0:
        num_sigma *= -1
    if threshold < 0:
        threshold *= -1

    min_sigma = abs(min_sigma)
    if min_sigma > max_sigma:
        min_sigma, max_sigma = max_sigma, min_sigma

    overlap = abs(overlap)
    if overlap > 1:
        overlap = 1

    # Instance is in the form kxmxn but log detection expects nxmxk
    # It's true that it could already by nxmxk, but more likely not so we'll fix
    if len(data.shape) == 3:
        new_data = np.empty(data.shape[1:]+data.shape[0:1])
        for i, layer in enumerate(data):
            new_data[:, :, i] = layer
        data = new_data

    return blob_log(data, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma,
                    threshold=threshold, overlap=overlap)

log_detection = dew.register("LogDetection", "test_log_detection", log_detection_helper, None, [float, float, int, float, float], TriState)
log_detection.__doc__ = """
Performs object detection using Laplacian of Gaussian

Args:
    data:       numpy array of example
    kernel:     kernel matrix (size x size)
    max_sigma:  the maximum standard deviation for Gaussian kernel
    num_sigma:  The number of intermediate values of standard deviations to
                consider between min_sigma and max_sigma
    threshold:  The absolute lower bound for scale space maxima.
                Local maxima smaller than thresh are ignored.
                Reduce this to detect blobs with less intensities.

Returns:
    coordinates of predicted object
"""
def dog_detection_helper(data, min_sigma=1, max_sigma=30, sigma_ratio=1.6,
                         threshold=0.1, overlap=0.5):
    # @Austin, can't we do this in the setup function??
    # @Austin, also, this operates on 3D images, where the first column would be the row, do we handle this later?
    if max_sigma < 0:
        max_sigma *= -1
    if threshold < 0:
        threshold *= -1

    min_sigma = abs(min_sigma)
    if min_sigma > max_sigma:
        min_sigma, max_sigma = max_sigma, min_sigma

    sigma_ratio = abs(sigma_ratio)
    overlap = abs(overlap)
    if overlap > 1:
        overlap = 1

    # Instance is in the form kxmxn but dog detection expects nxmxk
    # It's true that it could already by nxmxk, but more likely not so we'll fix
    # will run if smallest dimension is the first
    if len(data.shape) == 3 and np.argmin(data.shape) == 0:
        new_data = np.empty(data.shape[1:]+data.shape[0:1])
        for i, layer in enumerate(data):
            new_data[:, :, i] = layer
        data = new_data

    return blob_dog(data, min_sigma=min_sigma, max_sigma=max_sigma,
                    sigma_ratio=sigma_ratio, threshold=threshold, overlap=0.5)

dog_detection = dew.register("DogDetection", "test_dog_detection", dog_detection_helper, None, [float, float, float, float, float], TriState)
dog_detection.__doc__ = """
Performs object detection using Difference of Gaussian

Args:
    data:       numpy array of example
    kernel:     kernel matrix (size x size)
    max_sigma:  the maximum standard deviation for Gaussian kernel
    threshold:  The absolute lower bound for scale space maxima.
                Local maxima smaller than thresh are ignored.
                Reduce this to detect blobs with less intensities.

Returns:
    coordinates of predicted object
"""

def doh_detection_helper(data, min_sigma=1, max_sigma=30, num_sigma= 10, threshold=0.01, overlap=0.5):
    if max_sigma < 0:
        max_sigma *= -1

    min_sigma = abs(min_sigma)

    if threshold < 0:
        threshold *= -1

    if min_sigma > max_sigma:
        min_sigma, max_sigma = max_sigma, min_sigma
    overlap = abs(overlap)
    if overlap > 1:
        overlap = 1

    # doh works on 2D Arrays,
    # if dealing with k channels, in the form of kxnxm,
    # let's average average over k channels
    if len(data.shape)== 3:
        data = np.mean(data, axis=0)

    return blob_doh(data, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold, overlap=overlap)

doh_detection = dew.register("DohDetection", "test_doh_detection", doh_detection_helper, None , [float, float, int, float, float], TriState)
doh_detection.__doc__ = """
Performs object detection using Difference of Hessian

Args:
    data:       numpy array of example
    kernel:     kernel matrix (size x size)
    max_sigma:  the maximum standard deviation for Gaussian kernel
    threshold:  The absolute lower bound for scale space maxima.
                Local maxima smaller than thresh are ignored.
                Reduce this to detect blobs with less intensities.

Returns:
    coordinates of predicted object
"""

def rx_anomaly_detector_helper(data, threshold=0.99999):
    threshold = abs(threshold)

    # Instance is in the form kxmxn but anomaly detection expects nxmxk
    # It's true that it could already by nxmxk, but more likely not so we'll fix
    # will run if smallest dimension is the first
    if len(data.shape) == 3 and np.argmin(data.shape) == 0:
        new_data = np.empty(data.shape[1:]+data.shape[0:1])
        for i, layer in enumerate(data):
            new_data[:, :, i] = layer
        data = new_data

    # calculate rx values
    rxvals = rx(data)

    # initialize percent point function
    nbands = data.shape[-1]
    P = chi2.ppf(threshold, nbands)

    # threshold on probability
    data = 1 * (rxvals > P)
    return data.astype(np.float64)

rx_anomaly_detector = dew.register("RXAnomalyDetector", "test_anomaly_detection", rx_anomaly_detector_helper, None, [float], TRI_STATE)
rx_anomaly_detector.__doc__ = """
"The RX anomaly detector uses the squared Mahalanobis distance as a measure
 of how anomalous a pixel is with respect to an assumed background."

https://www.spectralpython.net/algorithms.html#rx-anomaly-detector

Args:
    data:      numpy array of example
    threshold: probability threshold of chi-squared continuous random variable
               percent point function (inverse of cdf — percentiles)

Returns:
    Transformed data
"""

def gaussian_filter_helper(data, sigma=1.0):
    sigma = abs(sigma)
    # Instance is in the form kxmxn but gaussian filter expects nxmxk
    # It's true that it could already by nxmxk, but more likely not so we'll fix
    # Unfortunately only expects MxNx3 or single channel
    if len(data.shape) == 3 and data.shape[0] == 3:
        new_data = np.empty(data.shape[1:]+data.shape[0:1])
        for i, layer in enumerate(data):
            new_data[:, :, i] = layer
        data = new_data
    elif len(data.shape) == 3:
        data = np.mean(data, axis=0)

    return skimage.filters.gaussian(data, sigma=sigma)

gaussian_filter = dew.register("GaussianFilter", "test_gaussian_filter", gaussian_filter_helper, None, [float], TRI_STATE)
gaussian_filter.__doc__ = """
"Multi-dimensional Gaussian filter."
https://scikit-image.org/docs/stable/api/skimage.filters

Args:
    data:      numpy array
    threshold: float

Returns:
    Transformed data
"""

def sobel_filter_helper(data):
    # Sobel filter wants a 2D array
    if len(data.shape) == 3:
        data = np.mean(data, axis=0)
    return skimage.filters.sobel(data)

sobel_filter = dew.register("SobelFilter", "test_sobel_filter", sobel_filter_helper, None, [], TRI_STATE)
sobel_filter.__doc__ = """
"Find the edge magnitude using the Sobel transform."
https://scikit-image.org/docs/stable/api/skimage.filters

Args:
    data:      numpy array
    threshold: float

Returns:
    Transformed data
"""

def ski_median_filter_helper(data, dsize=3):
    dsize = abs(dsize)
    my_disk = disk(dsize)
    # Putting it in normal image format, will run if 3D and not already fixed
    if len(data.shape) == 3 and data.shape[0] != 3:
        new_data = np.empty(data.shape[1:]+data.shape[0:1])
        new_disk = np.empty(my_disk.shape+data.shape[0:1])
        for i, layer in enumerate(data):
            new_data[:, :, i] = layer
            new_disk[:, :, i] = my_disk
        data = new_data
    else:
        new_disk = my_disk

    # Normalize data into 0 - 1
    data = (data - data.min()) / (data.max() - data.min())

    return skimage.filters.median(data, selem=new_disk)

ski_median_filter = dew.register("SkiMedianFilter", "test_ski_median_filter", ski_median_filter_helper, None, [int], TRI_STATE)
ski_median_filter.__doc__ = """
"Return local median of an image."
https://scikit-image.org/docs/stable/api/skimage.filters

Args:
    data:      numpy array
    threshold: float

Returns:
    Transformed data
"""

def my_binary_threshold_helper(data, threshold=1.2e-9):
    return 1.0 * (data >= threshold)

my_binary_threshold = dew.register("MyBinaryThreshold", "test_binary_threshold", my_binary_threshold_helper, None, [float], TRI_STATE)
my_binary_threshold.__doc__ = """
Sets the value of all pixels with values less than the threshold to 0,
and pixels with values at least the threshold to 1.

Args:
    data:      (c x w x h) array
    threshold: float

Returns:
    Transformed data
"""

def normal_likelihood(data_pair, r=0):
    """Object detection transformation based on the multivariate normal distribution.
       Note: implements STREAM_TO_STREAM

    Args:
        data_pair: given datapair
        r:         Non-negative integer
        threshold: float

    Returns:
        Data Pair
    """
    # For debugging purposes let's print out method name
    print("NormalLikelihood") ; sys.stdout.flush()

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
            method_string = "NormalLikelihood" + "_" + str(r) + "_"

            # Combine the unique method name + arguments of the method + hash of the previous data
            # To form a unique key of the method call
            method_key = method_string + previous_hash

            overhead_time, method_row, cache_row, hit = check_cache_read(data_pair, database,
                                                                            method_key, oh_time_start)
            if hit: return data_pair

            eval_time_start = time.time()

        # make r non-negative
        r = np.abs(r)

        # setup data instances
        train_instances = cp.deepcopy(data_pair.get_train_data().get_instances())
        test_instances = cp.deepcopy(data_pair.get_test_data().get_instances())

        # create lists of data
        train_images = np.array([i.get_stream().get_data() for i in train_instances])
        train_truths = np.array([i.get_target() if np.sum(i.get_target()) != -2.0 else [] for i in train_instances])
        test_images = np.array([i.get_stream().get_data() for i in test_instances])

        """
        Run the algorithm
        """
        train_images2, test_images2 = normal_transform_helper(train_images,
                                                              train_truths,
                                                              test_images,
                                                              r=r)

        # update instance objects with new data
        for test_instance, test_image in zip(test_instances, test_images2):
            test_instance.get_stream().set_data(test_image)

        for train_instance, train_image in zip(train_instances, train_images2):
            train_instance.get_stream().set_data(train_image)

        """
        Update data pair with new data
        """
        data_pair.get_train_data().set_instances(train_instances)
        data_pair.get_test_data().set_instances(test_instances)

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
    gc.collect(); return data_pair
dewb.register("NormalLikelihood", "test_normal_likelihood", normal_likelihood, None, [EmadeDataPair, int])

def sep_detection_window(data_pair, r=0, ksize=5, threshold=5.0):
    """1. Estimate kernel from input data
       2. Runs sep's matched filtering to find centroids
       3. Create windows centered around each centroid
       4. Stores labels of windows
       5. Stores window centroids in datapair attribute

       Note:    implements STREAM_TO_FEATURES
       Warning: overrides previous feature data

    Args:
        data_pair: given datapair
        r:         Non-negative integer
        ksize:     defines the (size x size) kernel matrix
        threshold: float

    Returns:
        Data Pair - image windows with shape (2r+1, 2r+1)
    """
    # For debugging purposes let's print out method name
    print("SepDetectionWindow") ; sys.stdout.flush()

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
            method_string = "SepDetectionWindow" + "_" + str(r) + "_" + str(ksize) + "_" + str(threshold) + "_"

            # Combine the unique method name + arguments of the method + hash of the previous data
            # To form a unique key of the method call
            method_key = method_string + previous_hash

            overhead_time, method_row, cache_row, hit = check_cache_read(data_pair, database,
                                                                            method_key, oh_time_start,
                                                                            target=True)
            if hit: return data_pair

            eval_time_start = time.time()

        # generate the kernel
        train_instances = cp.deepcopy(data_pair.get_train_data().get_instances())
        kernel = kernel_estimation(train_instances, ksize)

        # parameter check
        if threshold < 0:
            threshold *= -1
        r = np.abs(r)
        u, v = (r, r+1)
        # assuming data is a (n, m) matrix
        if r > train_instances[0].get_stream().get_data().shape[0] or r > train_instances[0].get_stream().get_data().shape[1]:
            raise Exception("r is out of image data bounds.")

        # Initialize where the data will be temporarily stored
        data_list = []
        # Iterate through train data then test data
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            # Copy the dataSet so as not to destroy original data
            instances = cp.deepcopy(data_set.get_instances())
            # Iterate over all points in the dataSet
            for instance in instances:
                # Load stream data
                data = instance.get_stream().get_data()
                truth = instance.get_target()
                if len(truth.shape) > 1 and truth.shape[0] == 1:
                    truth = truth[0]

                # run SEP's matched_filtering
                objects = sep.extract(data, threshold, filter_kernel=kernel)

                w_list = []
                l_list = []
                c_list = []
                for obj in objects:
                    x, y = obj['x'], obj['y']
                    x2 = int(np.round(x))
                    y2 = int(np.round(y))
                    window = data[x2-u:x2+v,y2-u:y2+v]
                    centroid = np.array([y,x])

                    # check if centroid is a true positive or false positive
                    dist = 4.0
                    if len([t for t in truth if np.linalg.norm(centroid-t).sum() <= dist]) > 0:
                        l_list.append(1)
                    else:
                        l_list.append(0)

                    w_list.append(window)
                    c_list.append(centroid)

                # Save new data
                # Note: this overwrites existing feature data to make sure the data format is correct
                if len(w_list) > 0:
                    instance.set_target(np.array([l_list]))
                    instance.get_features().set_data(np.stack(w_list, axis=0))
                    instance.labels = np.stack(c_list, axis=0)
                else:
                    instance.set_target(np.array([]))
                    instance.get_features().set_data(np.array([]))
                    instance.labels = np.array([])

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
    gc.collect(); return data_pair
dewb.register("SepDetectionWindow", "test_sep_detection_window", sep_detection_window, None, [EmadeDataPair, int, int, float])

def maximum_window(data_pair, r=0, neighborhood_size=7, filter_threshold=1.0):
    """1. Run maxmium filter to find local maxima
       2. Find centroids of local maxima
       3. Create windows centered around each centroid
       4. Stores labels of windows
       5. Stores window centroids in datapair attribute

       Note:    implements STREAM_TO_FEATURES
       Warning: overrides previous feature data

    Args:
        data_pair:         given datapair
        r:                 Non-negative integer
        neighborhood_size: size of neighboring pixel range used to calculate local maxima
        filter_threshold:  float

    Returns:
        Data Pair - image windows with shape (2r+1, 2r+1)
    """
    # For debugging purposes let's print out method name
    print("MaximumWindow") ; sys.stdout.flush()

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
            method_string = "MaximumWindow" + "_" + str(r) + "_" + str(neighborhood_size) + "_" + str(filter_threshold) + "_"

            # Combine the unique method name + arguments of the method + hash of the previous data
            # To form a unique key of the method call
            method_key = method_string + previous_hash

            overhead_time, method_row, cache_row, hit = check_cache_read(data_pair, database,
                                                                            method_key, oh_time_start,
                                                                            target=True)
            if hit: return data_pair

            eval_time_start = time.time()

        # parameter check
        if filter_threshold < 0:
            filter_threshold *= -1
        r = np.abs(r)
        u, v = (r, r+1)
        # assuming data is a (n, m) matrix
        train_instances = cp.deepcopy(data_pair.get_train_data().get_instances())
        if r > train_instances[0].get_stream().get_data().shape[0] or r > train_instances[0].get_stream().get_data().shape[1]:
            raise Exception("r is out of image data bounds.")

        # Initialize where the data will be temporarily stored
        data_list = []
        # Iterate through train data then test data
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            # Copy the dataSet so as not to destroy original data
            instances = cp.deepcopy(data_set.get_instances())
            # Iterate over all points in the dataSet
            for instance in instances:
                # Load stream data
                data = instance.get_stream().get_data()
                truth = instance.get_target()
                if len(truth.shape) > 1 and truth.shape[0] == 1:
                    truth = truth[0]

                data_max = ndimage.filters.maximum_filter(data, neighborhood_size)
                maxima = (data == data_max)
                diff = data_max >= filter_threshold
                maxima[diff == 0] = 0

                # Find Centroids
                labeled, num_objects = ndimage.label(maxima)
                slices = ndimage.find_objects(labeled)

                # Pad image to make sure all the windows are the same size
                new_data = cv2.copyMakeBorder(data, r, r, r, r, cv2.BORDER_REPLICATE)

                w_list = []
                l_list = []
                c_list = []
                for dy,dx in slices:
                    # Convert subpixel centroid to int pixel location
                    # Then shift it into the same location in the padded image
                    x = (dx.start + dx.stop - 1)/2
                    y = (dy.start + dy.stop - 1)/2
                    x2 = int(np.round(x)) + r
                    y2 = int(np.round(y)) + r

                    # Slice Window from Image
                    window = new_data[x2-u:x2+v,y2-u:y2+v]
                    centroid = np.array([y,x])

                    # Check if centroid is a true positive or false positive
                    # TODO: Make this distance a parameter
                    dist = 4.0
                    if len([t for t in truth if np.linalg.norm(centroid-t).sum() <= dist]) > 0:
                        l_list.append(1)
                    else:
                        l_list.append(0)

                    w_list.append(window)
                    c_list.append(centroid)

                # Save new data
                # Note: this overwrites existing feature data to make sure the data format is correct
                if len(w_list) > 0:
                    instance.set_target(np.array(l_list))
                    instance.get_features().set_data(np.stack(w_list, axis=0))
                    instance.labels = np.stack(c_list, axis=0)
                else:
                    instance.set_target(np.array([]))
                    instance.get_features().set_data(np.array([]))
                    instance.labels = np.array([])

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
    gc.collect(); return data_pair
dewb.register("MaximumWindow", "test_maximum_window", maximum_window, None, [EmadeDataPair, int, int, float])

def filter_centroids(data_pair):
	"""Assumes a learner/classifier was ran before this method and stored labels
	   This method filters out window centroids labeled 0 by the learner/classifier
	   Then stores the centroids in the target of each instance (for evaluation)

	   Note: this method does not alter the stream or feature data of each instance

	Args:
		data_pair: given datapair

	Returns:
		Data Pair
	"""
	# For debugging purposes let's print out method name
	print("FilterCentroids") ; sys.stdout.flush()

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
			method_string = "FilterCentroids" + "_"

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
				# Load the target
				target = instance.get_target()
				if len(target) > 0:
					# assume instance has a .labels attribute
					prediction = [centroid for label,centroid in zip(target, instance.labels) if label]

					# Set new prediction
					instance.set_target(np.array(prediction))
				else:
					instance.set_target([np.array([-1.0, -1.0])]) # can change to instance.set_target(np.array([[-1.0, -1.0]])) or instance.set_target([[-1.0,-1.0]])

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
	gc.collect(); return data_pair
dewb.register("FilterCentroids", "test_filter_centroids", filter_centroids, None, [EmadeDataPair])

def ccorr_object_filter(data_pair, second_pair, r=6, threshold=6.6e-3):
    """Calculates an average kernel from image windows centered on predicted centroids
       Performs cross-correlation between the average kernel and each centroid window
       Centroids with a cross-correlation < threshold are filtered out

       Note: only loads stream data
       Note: this method does not alter the stream or feature data of each instance

    Args:
        data_pair:   datapair with detected objects
        second_pair: datapair with original image and truth
        r:           Non-negative integer
        threshold:   float

    Returns:
        Data Pair
    """
    # For debugging purposes let's print out method name
    print("CCorrObjectFilter") ; sys.stdout.flush()

    data_pair = cp.deepcopy(data_pair)
    second_pair = cp.deepcopy(second_pair)

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
            method_string = "CCorrObjectFilter" + "_" + str(r) + "_" + str(threshold) + "_"

            # Combine the unique method name + arguments of the method + hash of the previous data
            # To form a unique key of the method call
            method_key = method_string + previous_hash

            overhead_time, method_row, cache_row, hit = check_cache_read(data_pair, database,
                                                                            method_key, oh_time_start,
                                                                            target=True)
            if hit: return data_pair

            eval_time_start = time.time()

        # parameter check
        if threshold < 0:
            threshold *= -1
        r = np.abs(r)
        # assuming data is a (n, m) matrix
        train_instances = cp.deepcopy(second_pair.get_train_data().get_instances())
        test_instances = cp.deepcopy(second_pair.get_test_data().get_instances())
        train_instances_t = cp.deepcopy(data_pair.get_train_data().get_instances())
        test_instances_t = cp.deepcopy(data_pair.get_test_data().get_instances())
        if r > train_instances[0].get_stream().get_data().shape[0] or r > train_instances[0].get_stream().get_data().shape[1]:
            raise Exception("r is out of image data bounds.")

        train_images = [instance.get_stream().get_data() for instance in train_instances]
        train_truths = [instance.get_target() if np.sum(instance.get_target()) != -2.0 else [] for instance in train_instances]
        train_objects = [instance.get_target() for instance in train_instances_t]
        test_images = [instance.get_stream().get_data() for instance in test_instances]
        test_objects = [instance.get_target() for instance in test_instances_t]

        ker = average_kernel(train_images, train_truths, r)
        ker /= np.sum(ker)

        train_filtered = []
        for im, objects, truth in zip(train_images, train_objects, train_truths):
            im_avg = np.float64(np.mean(im))
            filtered = []
            for x, y in objects:
                window = subpixel_window(im, x, y, im_avg, r)
                window /= np.sum(window)
                if np.sum(window * ker) >= threshold:
                    filtered.append(np.array([x, y]))
            train_filtered.append(filtered)

        test_filtered = []
        for im, objects in zip(test_images, test_objects):
            im_avg = np.float64(np.mean(im))
            filtered = []
            for x, y in objects:
                window = subpixel_window(im, x, y, im_avg, r)
                window /= np.sum(window)
                if np.sum(window * ker) >= threshold:
                    filtered.append(np.array([x, y]))
            test_filtered.append(filtered)

        """
        Update data pair with new data
        """
        for instance, target in zip(train_instances_t, train_filtered):
            instance.set_target(target)
        data_pair.get_train_data().set_instances(train_instances_t)

        for instance, target in zip(test_instances_t, test_filtered):
            instance.set_target(target)
        data_pair.get_test_data().set_instances(test_instances_t)

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
    gc.collect(); return data_pair
dewb.register("CCorrObjectFilter", "test_ccorr_object_filter", ccorr_object_filter, None, [EmadeDataPair, EmadeDataPair, int, float])

def max_loc_helper(data):
    minval, maxval, minloc, maxloc = cv2.minMaxLoc(data)
    return [maxloc]

max_loc = dew.register("MaxLoc", "test_max_loc", max_loc_helper, None, [], TRI_STATE)
max_loc.__doc__ = """
Returns the maximum pixel location

Args:
    data: numpy array

Returns:
    Transformed data
"""

def min_loc_helper(data):
    minval, maxval, minloc, maxloc = cv2.minMaxLoc(data)
    return [minloc]

min_loc = dew.register("MinLoc", "test_min_loc", min_loc_helper, None, [], TRI_STATE)
min_loc.__doc__ = """
Returns the minimum pixel location

Args:
    data: numpy array

Returns:
    Transformed data
"""

def create_bbox_helper(data, second_data, r=3):
    r = 2*np.abs(r) if r != 0 else 2

    new_data = []
    for frame, loc in zip(second_data, data):
        frame = cv2.copyMakeBorder(frame, r+1, r+1, r+1, r+1, cv2.BORDER_REPLICATE)
        new_data.append([frame[(loc[0]+r+1)-r:(loc[0]+r+1)+r+1,(loc[1]+r+1)-r:(loc[1]+r+1)+r+1] for i in loc])

    return new_data

create_bbox = dew_2.register("CreateBBox", "test_create_bbox", create_bbox_helper, None, [int], TRI_STATE)
create_bbox.__doc__ = """
Creates (2*r+1 x 2*r+1) bounding boxes from pixel locations

Args:
    data:        list of numpy arrays
    second_data: list of numpy arrays
    r (int):     dimension used to create bounding box

Returns:
    Transformed data
"""


def get_centroids_helper(data):
    # This method runs assuming the input data is coming from some method and
    # Just grabs the first two columns which should be the x,y locations for
    # The centroids

    data = data[:,:2]

    return data

get_centroids = dew.register('GetCentroids', "test_get_centroids", get_centroids_helper, None, [], TriState)
