"""
Programmed by Jason Zutty
Modified by VIP Team
Implements decomposition methods
"""
from GPFramework.constants import TriState
from GPFramework.wrapper_methods import RegistryWrapperFT
from GPFramework.data import EmadeDataPair

from sklearn.manifold import SpectralEmbedding
from sklearn.decomposition import PCA, SparsePCA, FastICA

# Wrapper class to control export namespace
dmw = RegistryWrapperFT([EmadeDataPair, TriState])

def decomposition_helper(train_data, test_data, target, function):
    new_train_data = function.fit_transform(train_data)
    new_test_data = function.transform(test_data)
    return new_train_data, new_test_data

def my_pca_setup(data, n_components=3, whiten=False):
    return PCA(n_components=n_components, whiten=whiten)

my_pca = dmw.register("myPCA", "test_my_pca", decomposition_helper, my_pca_setup, [int, bool])
my_pca.__doc__ = """
Implements scikit's pca method

http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

Args:
    data: numpy array of example
    n_components: number of components to keep
    whiten: When True (False by default) the components_ vectors are
            multiplied by the square root of n_samples and then divided by the
            singular values to ensure uncorrelated outputs with unit
            component-wise variances.

    Whitening will remove some information from the transformed signal
    (the relative variance scales of the components) but can sometime
    improve the predictive accuracy of the downstream estimators by making
    their data respect some hard-wired assumptions. --sklearn

Returns:
    transformed data
"""

def my_sparse_pca_setup(data, n_components=3, alpha=1.0):
    return SparsePCA(n_components=n_components, alpha=alpha)

my_sparse_pca = dmw.register("mySparsePCA", "test_my_sparse_pca", decomposition_helper, my_sparse_pca_setup, [int, float])
my_sparse_pca.__doc__ = """
Implements scikit's sparse pca method

http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparsePCA.html#sklearn.decomposition.SparsePCA

Args:
    data: numpy array of example
    n_components: number of components to keep
    alpha: Sparsity controlling parameter. Higher values lead to sparser components.

Returns:
    transformed data
"""

def my_ica_setup(data, n_components=3, whiten=True):
    return FastICA(n_components=n_components, whiten=whiten)

my_ica = dmw.register("myICA", "test_my_ica", decomposition_helper, my_ica_setup, [int, bool])
my_ica.__doc__ = """
Implements scikit's ica method

http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html

Args:
    data: numpy array of example
    n_components: number of components to keep
    whiten: When True (False by default) the components_ vectors are
            multiplied by the square root of n_samples and then divided by the
            singular values to ensure uncorrelated outputs with unit
            component-wise variances.

    Whitening will remove some information from the transformed signal
    (the relative variance scales of the components) but can sometime
    improve the predictive accuracy of the downstream estimators by making
    their data respect some hard-wired assumptions. --sklearn

Returns:
    transformed data
"""

def my_spectral_embedding_helper(train_data, test_data, target, function):
    new_train_data = function.fit_transform(train_data)
    new_test_data = function.fit_transform(test_data)
    return new_train_data, new_test_data

def my_spectral_embedding_setup(data, n_components=3):
    return SpectralEmbedding(n_components=n_components)

my_spectral_embedding = dmw.register("mySpectralEmbedding", "test_my_spectral_embedding", my_spectral_embedding_helper, my_spectral_embedding_setup, [int])
my_spectral_embedding.__doc__ = """
Implements scikit's spectral embedding method

http://scikit-learn.org/stable/modules/generated/sklearn.manifold.SpectralEmbedding.html

Args:
    data: numpy array of example
    n_components: number of components to keep

Returns:
    transformed data
"""
