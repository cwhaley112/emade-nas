"""
Programmed by William Samuelson
Implements a number of image feature extraction methods from skimage
"""
from GPFramework.data import EmadeDataPair
from GPFramework.constants import TriState, Axis, TRI_STATE
from GPFramework.wrapper_methods import RegistryWrapperS
from skimage.feature import hog, daisy, greycomatrix, greycoprops, canny
from skimage import exposure
from skimage.color import rgb2gray
import numpy as np
import math

few = RegistryWrapperS([EmadeDataPair, TriState, Axis])

def hog_helper(image, multi=False, orientations=8, pixels_per_cell=None, cells_per_block=None):
    fd = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, multichannel=multi)
    return fd

def hog_setup(image, multi=False, orientations=8, pixels_per_cell=None, cells_per_block=None):
    orientations = orientations if orientations > 0 else 1
    pixels_per_cell = (pixels_per_cell,)*2 if pixels_per_cell > 0 else (1,)*2
    cells_per_block = (cells_per_block,)*2 if cells_per_block > 0 else (1,)*2
    return image, { "multi":multi, "orientations":orientations, "pixels_per_cell":pixels_per_cell, "cells_per_block":cells_per_block }

hog_feature = few.register("Hog", "test_hog_feature", hog_helper, hog_setup, [bool, int, int, int], TRI_STATE)
hog_feature.__doc__ = """
    Implements HOG feature extraction

    Args:
        image: numpy array of example
        convert_bw: convert image to black&white
        orientations: number of orientation bins
        pixels_per_cell: size (in pixels) of a cell
        cells_per_block: number of cells in each block

    Returns:
        HOG feature descriptor
    """

def daisy_helper(image, step=25, radius=10, rings=3, histograms=8, orientations=8):
    image = daisy(image, step=step, radius=radius, rings=rings, histograms=histograms, orientations=orientations)
    # image = np.reshape(image, (1, image.shape[0]))
    return image

def daisy_setup(image, step=25, radius=10, rings=3, histograms=8, orientations=8):
    step = step if step > 0 else 1
    radius = radius if radius > 0 else 1
    rings = rings if rings > 0 else 1
    histograms = histograms if histograms > 0 else 1
    orientations = orientations if orientations > 0 else 1
    return image, { 'step':step, 'radius':radius, 'rings':rings, 'histograms':histograms, 'orientations':orientations }

daisy_feature = few.register("Daisy", "test_daisy_feature", daisy_helper, daisy_setup, [int, int, int, int, int], [TriState.FEATURES_TO_FEATURES, TriState.STREAM_TO_STREAM])
daisy_feature.__doc__ = """
    Implements DAISY feature extraction

    Args:
        image: numpy array of example
        step: distance between descriptor sampling points
        radius: radius (in pixels) of the outermost ring
        ring: number of rings
        histograms: number of histograms sampled per ring
        orientations: number of orientations (bins) per histogram

    Returns:
        DAISY feature descriptor
    """

# TODO: Get William to modify these methods to work in new framework
def _glcm_function(image, distances, horizontal):
    glcm = greycomatrix(image.astype(np.uint8), [distances], [0 if horizontal else np.pi / 2],
                        levels=256, symmetric=True, normed=True)
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy',
                  'correlation', 'ASM']
    return [greycoprops(glcm, prop)[0, 0] for prop in properties]

def glcm_feature(data_pair, distances=5, horizontal=True):
    """Implements GLCM feature"""
    distances = distances if distances > 0 else 1
    return _generic_feature(data_pair, _glcm_function, convert_bw=True,
                            kwargs={'distances':distances,
                                    'horizontal': horizontal})

def _glcm_patches_function(image, pixels_per_patch, distances, horizontal):
    image = image.astype(np.uint8)
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy',
                  'correlation', 'ASM']
    features = []
    for r in range(math.ceil(image.shape[0] / pixels_per_patch)):
        for c in range(math.ceil(image.shape[1] / pixels_per_patch)):
            row_start = r*pixels_per_patch
            row_end = min(row_start + pixels_per_patch, image.shape[0])
            col_start = c*pixels_per_patch
            col_end = min(col_start + pixels_per_patch, image.shape[1])
            patch = image[row_start:row_end, col_start:col_end]
            glcm = greycomatrix(patch, [distances],
                                [0 if horizontal else np.pi / 2],
                                levels=256, symmetric=True, normed=True)
            features += [greycoprops(glcm, prop)[0, 0] for prop in properties]
    return features

def glcm_patches_feature(data_pair, pixels_per_patch=16, distances=5, horizontal=True):
    """Implements GLCM patches feature"""
    distances = distances if distances > 0 else 1
    pixels_per_patch = pixels_per_patch if pixels_per_patch > 0 else 1
    return _generic_feature(data_pair, _glcm_patches_function, convert_bw=True,
                            kwargs={'pixels_per_patch': pixels_per_patch,
                                    'distances':distances,
                                    'horizontal': horizontal})
