import enum
from glob import glob
import dill
import numpy as np
import copy as cp
from tensorflow.keras.models import Model#, model_from_json
import sys

from tensorflow.python.keras.engine.functional import Functional
sys.path.insert(1, '/home/cameron/Desktop/emade/src')
from GPFramework.data import EmadeDataPair, EmadeDataPairNN, EmadeDataPairNNF, EmadeData
import tensorflow as tf
import keras

from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, GRU, Conv2D, SeparableConv2D, LeakyReLU, Conv1D, MaxPool1D, MaxPool2D, MaxPool3D, AveragePooling1D, AveragePooling2D, GlobalMaxPool1D, GlobalMaxPool2D, Flatten, GlobalAveragePooling1D,GlobalAveragePooling2D, Concatenate, Add, Bidirectional, Attention, ZeroPadding2D, BatchNormalization, LayerNormalization, Permute, Reshape
from tensorflow.keras.layers import Activation as Activation_Wrapper
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.activations import linear, sigmoid, relu
from tensorflow.keras.regularizers import l2 
from tensorflow.keras.preprocessing.text import Tokenizer, one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras_pickle_wrapper import KerasPickleWrapper
from tensorflow.keras.initializers import RandomUniform, Constant, he_normal, GlorotNormal, GlorotUniform

# tensorflow.compat.v1.disable_eager_execution()
# from transformers import TFDistilBertModel,DistilBertTokenizer #, DistilBertConfig # requires TF >=2.2

from sklearn import preprocessing

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
import pickle
import sys
import time
import gc  
from enum import Enum
import json
from sklearn.preprocessing import LabelEncoder
from einops.layers.keras import Rearrange, Reduce

import random
from deap import gp
import re
from collections import Counter

MAXLEN=500 # scale to dataset? max sequence in amazon is 257
NUMWORDS=20000


tf.random.set_seed(42)
np.random.seed(101)

## NNLEARNER CLASSES, ENUMS, AND HELPER FUNCTIONS

class LayerList:
    def __init__(self, modList=[]):
        self.mylist = InitLayerList()
        self.modList = modList
        self.curr = 0
    def __getitem__(self, key):
        return self.mylist[key]
    def __len__(self):
        return len(self.mylist)

# Class recognizer to ensure appending of module to the ends of module individuals
class LayerListM(LayerList):
    def __init__(self):
        super().__init__()

class LayerList4dim(LayerList):
    def __init__(self):
        super().__init__()

class LayerList3dim(LayerList):
    def __init__(self):
        super().__init__()

class LayerList2dim(LayerList):
    def __init__(self):
        super().__init__()

class LayerListFinal(LayerList):
    def __init__(self):
        super().__init__()

class DataDims(list):
    def __init__(self):
        super().__init__()

class ModList(list):
    def __init__(self):
        pass

class DataType(str):
    def __init__(self):
        super().__init__()

class LayerUnits(int):
    pass

class ConvKernelSize(int):
    kernSize1 = 1
    kernSize3 = 3
    kernSize5 = 5

class PaddingType(str):
    VALID="valid"
    SAME="same"

class StrideSize(int):
    stride1 = 1
    stride2 = 2

class PoolingLayer():
    pass
class PoolSize(int):
    PoolSize2 = 2
    PoolSize3 = 3
def NoPooling():
    return None

class SkipConnection3dim():
    pass
class SkipConnection4dim():
    pass
def NoSkip():
    return (None, None)
def AddLayer(activation):
    activation = activation_handler(activation, "relu")
    return (None, activation)


class Activation(Enum):
    """
        Enum to describe various activation functions
    """
    RELU ='relu'
    ELU =  'elu'
    SELU = 'selu'
    LEAKY_RELU = 'leaky_relu'
    SIGMOID = 'sigmoid'
    SOFTMAX = 'softmax'
    TANH='tanh'
    GELU = 'gelu'
    LINEAR = 'linear'
    DEFAULT = None


def activation_handler(activation_enum, default_activation):
    if activation_enum.value == None:
        return default_activation
    else:
        return activation_enum.value

class Optimizer(Enum):
    '''
        enum for different optimizers supported by keras
    '''
    ADAM = 'adam'
    SGD = 'SGD'
    RMSPROP = 'RMSprop'
    ADADELTA = 'Adadelta'
    ADAGRAD = 'Adagrad'
    ADAMAX = 'Adamax'
    NADAM = 'Nadam'
    FTRL = 'Ftrl'


class PretrainedModel(Enum):
    '''
        enum for different pretrained models supported by keras applications
    '''
    MOBILENET = 'mobilenet'
    INCEPTION = 'inception'
    VGG = 'vgg'

class WeightInitializer(Enum):
    """
    Enum for different types of weight initializers. 
    """
    RANDOMUNIFORM = RandomUniform
    GLOROTUNIFORM = GlorotUniform
    GLOROTNORMAL = GlorotNormal
    HE = he_normal


class PretrainedEmbedding(Enum):
    GLOVE = 1
    FASTTEXT = 2
    GLOVETWITTER = 3
    GLOVEFASTTEXT = 4

class TimedStopping(Callback):
    '''
    Stop training when enough time has passed.
    # Arguments
        seconds: maximum time before stopping.
        verbose: verbosity mode.
    '''
    def __init__(self, seconds=None, verbose=0):
        super(Callback, self).__init__()

        self.start_time = 0
        self.seconds = seconds
        self.verbose = verbose

    def on_train_begin(self, logs={}):
        self.start_time = time.time()
    
    def on_epoch_end(self, epoch, log={}):
        if time.time() - self.start_time > self.seconds:
            self.model.stop_training = True
            if self.verbose:
                print("Stopping after %s seconds" % self.seconds)

class Normalization():
    """
    Defines enums for normalization of layer outputs
    """
    NONE = None
    BATCHNORM = BatchNormalization
    LAYERNORM = LayerNormalization

class DropoutLayer(float):
    """
    Defines enums for dropout in modules
    """
    NONE = None
    Dropout20 = 0.2

def PassLayerList(layerlist):
    """
    Passes unmodified layerlist.
    Useful for when we want to optionally add a layer which does not change the layerlist shape
    BUT we want to limit the number of these layers that are added, so we say it returns a different
    type of layerlist in gpframework
    """
    return layerlist

class PatchSize(int):
    """
    Placeholder class for patch sizes
    """
    pass

class ImageAugmentation():
    pass

class ImageFlip():
    NONE = ""
    HFLIP = "horizontal"
    BOTH = "horizontal_and_vertical"

class ImageRotation(float):
    # numbers are a fraction of 2 pi
    NONE = 0.0
    ROTATE10 = 0.1
    ROTATE20 = 0.2

class ImageZoom(float):
    NONE = 0.0
    ZOOM10 = 0.1
    ZOOM20 = 0.2

class ImageTranslation(float):
    NONE = 0.0
    TRANSLATE10 = 0.1
    TRANSLATE20 = 0.2


# SKIP CONNECTION PRIMITIVES

def DenseConnection(activation, normalization):   
    """
    Wrapper for Dense layer located in skip connection
    Calls DenseLayer with placeholder out_dim which will get written when
    Module() runs
    """
    return DenseLayer(1, activation, normalization)

def Conv1DConnection(activation, kernel_dim, strides, pad_type, normalization):
    """
    Wrapper for Conv1D layer located in skip connection
    Calls ConvLayer with placeholder out_dim which will get written when
    Module() runs
    """
    return ConvLayer(1, activation, kernel_dim, strides, pad_type, normalization, None, Conv1D)

def Conv2DConnection(activation, kernel_dim, strides, pad_type, normalization):
    """
    Wrapper for Conv2D layer located in skip connection
    Calls ConvLayer with placeholder out_dim which will get written when
    Module() runs
    """
    return ConvLayer(1, activation, kernel_dim, strides, pad_type, normalization, None, Conv2D)

# INPUT LAYERS

def InputLayer(modList=None):
    """
    Creates an empty layerlist and adds an input placeholder to the beginning

    Returns:
        LayerList with "input" within it and modules stored as a class attribute (for weight sharing purposes)
    """

    if modList is None:
        modList = []

    empty_layerlist = LayerList(modList=modList)
    empty_layerlist.mylist.append("input")
    return empty_layerlist
    
def MobileNetInputLayer():
    empty_layerlist = LayerList()
    empty_layerlist.mylist.append({'type': PretrainedModel.MOBILENET})
    return empty_layerlist 

def VGGInputLayer():
    empty_layerlist = LayerList()
    empty_layerlist.mylist.append({'type': PretrainedModel.VGG})
    return empty_layerlist 

def InceptionInputLayer():
    empty_layerlist = LayerList()
    empty_layerlist.mylist.append({'type': PretrainedModel.INCEPTION})
    return empty_layerlist 

def PretrainedEmbeddingLayer(data_pair, initializer, layerlist): 
    """Creates Embedding layer  
    Args:   
        empty_model: empty_model as terminal    
        data_pair: given dataset    
        layerlist: layerlist to append to 
    Returns:    
        Keras Embedding Layer   
    """ 
    maxlen = MAXLEN 
    numwords=NUMWORDS   
    vocab_size = data_pair.vocabsize
    tok  = data_pair.tokenizer
    out = {PretrainedEmbedding.GLOVE:100, PretrainedEmbedding.GLOVEFASTTEXT:501, PretrainedEmbedding.FASTTEXT:300, PretrainedEmbedding.GLOVETWITTER:200 }
    out_dim = out[initializer]   
    initializer = Constant(get_embedding_matrix(initializer, vocab_size, tok))
    layerlist.mylist.append(Embedding(vocab_size,out_dim , input_length=maxlen, embeddings_initializer=initializer))
    return layerlist


def EmbeddingLayer(out_dim, data_pair, initializer, layerlist):
    maxlen = MAXLEN 
    numwords=NUMWORDS   
    out_dim = abs(out_dim)  
    if data_pair.get_datatype()=='textdata':
        size = data_pair.vocabsize
        tok = data_pair.tokenizer
    else:
        size = len(data_pair.get_train_data().get_numpy())
        maxlen = 1
    initializer = initializer.value()  
    layerlist.mylist.append(Embedding(size, out_dim, input_length=maxlen, embeddings_initializer=initializer))    
    return layerlist

def PatchEmbedding1DInputLayer(embedding_dim, patch_size, activation, normalization, data_dims, modList=None):
    return PatchEmbedding(embedding_dim, patch_size, activation, normalization, data_dims, modList, True)

def PatchEmbedding2DInputLayer(embedding_dim, patch_size, activation, normalization, data_dims, modList=None):
    return PatchEmbedding(embedding_dim, patch_size, activation, normalization, data_dims, modList, False)

def PatchEmbedding(embedding_dim, patch_size, activation_enum, normalization, data_dims, modList, flatten):
    """
    Splits images into patches of size (patch_size, patch_size) and projects each one
    to a vector of length embedding_dim

    Assumes image height and width % patch_size == 0 

    Does not add positional embedding (like ViT) because we don't use any positionally invariant layers (yet)

    Args:
        embedding_dim: dimension of patch embeddings
        patch_size: size of patch (in pixels). can be an int (square patch) or tuple of 2 ints
        activation_enum: optional nonlinearity applied to embeddings. Default activation is none
        normalization: optional normalization applied to embeddings
        data_dims: dimensions of input images
        modList: list of modules (contains weight info)
        flatten: bool, flattens spatial dimensions if true (like MLP Mixer and ViT), 
                 else they are maintained (like ConvMixer)
    """
    if modList is None:
        modList = []

    activation = activation_handler(activation_enum, 'linear')
    # per-patch embedding is the same as non-overlapping convolutional kernel
    patch_layer = Conv2D(embedding_dim, kernel_size=patch_size, strides=patch_size, activation=activation)

    temp_layerlist = []
    temp_layerlist.append(patch_layer)

    if flatten:
        reshape_layer = Rearrange('b h w c -> b (h w) c')
        temp_layerlist.append(reshape_layer)

    if normalization is not None:
        temp_layerlist.append(normalization())
    
    temp_layerlist = ['input'] + temp_layerlist
    outputs, input_layers = compile_layerlist(temp_layerlist, [], data_dims, 'imagedata')
    # create a unique name based on the args
    name = "_".join([str(elem) for elem in [embedding_dim, patch_size, activation, str(normalization).split('.')[-1], flatten]])
    name = re.sub("[^0-9a-zA-Z_]+","",name)
    model = Model(inputs=input_layers, outputs=outputs, name=name)
    
    loaded=False
    while not loaded:
        try:
            with open("Global_MODS", "rb") as mod_file:
                glob_mods = dill.load(mod_file)
            loaded=True
        except EOFError:
            time.sleep(15)
    if name in glob_mods.keys() and glob_mods[name][1] is not None:
        # set weights
        model.set_weights(glob_mods[name][1])

    empty_layerlist = LayerList(modList=modList)
    empty_layerlist.mylist.append('input')
    empty_layerlist.mylist.append(model)
    return empty_layerlist

# this isn't exactly an input layer but not sure where else to put it
def ImageAugmentationMod(input_shape, hflip, rotation, zoom, translation):
    # builds keras functional model with image augmentation layers
    # the following layers aren't implemented:
    #   - RandomHeight, RandomWidth, and RandomCrop because they change the spatial dimensions
    #     of the train images and I'm not exactly sure how to fine tune the model to take 
    #     full sized test images if we use these TODO
    #   - RandomContrast because it can change pixel intensity to be outside of the range 0-1
    #     which we'd never see in the test data
    inp = Input(shape=input_shape)
    x=inp

    if not any([hflip, rotation, zoom, translation]):
        return None # empty model is lame so returning none

    if hflip:
        x = tf.keras.layers.RandomFlip(hflip)(x)
    if rotation:
        x = tf.keras.layers.RandomRotation(rotation)(x)
    if zoom:
        x = tf.keras.layers.RandomZoom(zoom)(x)
    if translation:
        x = tf.keras.layers.RandomTranslation(translation, translation)(x) # uses default fill_mode and same value for height/width factors
    
    return Model(inputs=inp, outputs=x, name="ImageAugmentation")

# STANDARD NEURAL NETWORK LAYERS

def DenseLayer(out_dim, activation, normalization, layerlist=None, dropout_rate=None):   
    """Creates dense layer  
    Args:   
        out_dim: ouput dimension    
        activation_func: object of class Activation, specifies activation function to use   
    Returns:    
        Keras Dense Layer   
    """ 

    activation = activation_handler(activation, 'relu')
    layer = Dense(out_dim, activation=activation)
    if layerlist is not None:
        layerlist.mylist.append(layer)
        if normalization is not None:
            layerlist.mylist.append(normalization())
        if dropout_rate is not None:
            layerlist.mylist.append(Dropout(dropout_rate))
        return layerlist
    else:
        layer.activation = linear
        if normalization is not None:
            return layer, activation, normalization()
        else:
            return layer, activation

def Conv1DLayer(output_dim, activation, kernel_dim, strides, pad_type, normalization, layerlist=None):
    """
    Wrapper for Conv1D layer
    """
    return ConvLayer(output_dim, activation, kernel_dim, strides, pad_type, normalization, layerlist, Conv1D)

def Conv2DLayer(output_dim, activation, kernel_dim, strides, pad_type, normalization, layerlist=None):
    """
    Wrapper for Conv2D layer
    """
    return ConvLayer(output_dim, activation, kernel_dim, strides, pad_type, normalization, layerlist, Conv2D)

def SeparableConv2DLayer(output_dim, activation, kernel_dim, strides, pad_type, normalization, layerlist):
    """
    Wrapper for Depthwise Separable Conv2D layer
    """
    return ConvLayer(output_dim, activation, kernel_dim, strides, pad_type, normalization, layerlist, SeparableConv2D)

def ConvLayer(output_dim, activation, kernel_dim, strides, pad_type, normalization, layerlist, function):
    """
    Creates convolution layer based on what `function` is 

    Args:
        output_dim: output dimension
        activation: activation function
        kernel_dim: kernel size (assumes int; uses same value for all spatial dimensions)
        strides: stride for conv operations (same note as above)
        pad_type: enum for 'valid' or 'same' padding
        layerlist: list of previous layers and models (may be None)
        function: one of Conv1D or Conv2D

    Returns:
        Keras Conv Layer
    """
    activation = activation_handler(activation, 'relu')

    layer = function(output_dim, kernel_size=kernel_dim, activation=activation, strides=strides, padding=pad_type)

    if layerlist is not None:
        layerlist.mylist.append(layer)
        if normalization is not None:
            layerlist.mylist.append(normalization())
        return layerlist
    else:
        layer.activation = linear
        if normalization is not None:
            return layer, activation, normalization()
        else:
            return layer, activation

def LSTMLayer(out_dim, activation, bidir, ret_seq, layerlist):
    """Creates lstm layer

    Args:
        out_dim: ouput dimension

    Returns:
        Keras LSTM Layer
    """
    activation = activation_handler(activation, 'tanh')
    if bidir:
        #layerlist.mylist.append(Bidirectional(LSTM(out_dim,  kernel_initializer = initializer.value[0](), kernel_regularizer = l2(10**(-1*regularizer)))))
        layerlist.mylist.append(Bidirectional(LSTM(out_dim, activation=activation, return_sequences=ret_seq)))
    else:
        layerlist.mylist.append(LSTM(out_dim, activation=activation))
    return layerlist    

def GRULayer(out_dim, activation, regularizer, bidir, ret_seq, layerlist):
    """Creates gru layer

    Args:
        out_dim: ouput dimension

    Returns:
        Keras GRU Layer
    """
    activation = activation_handler(activation, 'tanh')

    if regularizer ==0:
        reg = None
    else:
        reg = l2(10**(-1*regularizer)) 
    if bidir:
        layerlist.mylist.append(Bidirectional(GRU(out_dim, return_sequences=ret_seq, kernel_regularizer =reg )))
    else:
        layerlist.mylist.append(GRU(out_dim, kernel_regularizer = l2(10**(-1*regularizer))))
    return layerlist

def OutputLayer(layerlist):
    layerlist.mylist.append("output")
    return layerlist  

def MaxPoolingLayer1D(pool_size, layerlist=None):
    """Creates MaxPooling layer

    Args:
        pool_size: pooling size
        strides: stride for pooling operations
        
    Returns:
        Keras MaxPooling Layer
    """

    layer = MaxPool1D(pool_size)
    if layerlist is not None:
        layerlist.mylist.append(layer) 
        return layerlist
    else:
        return layer

def MaxPoolingLayer2D(pool_size, layerlist=None):
    """Creates MaxPooling layer

    Args:
        pool_size: pooling size
        strides: stride for pooling operations
        
    Returns:
        Keras MaxPooling Layer
    """
    pool_size = (pool_size, pool_size)
    layer = MaxPool2D(pool_size)
    if layerlist is not None:
        layerlist.mylist.append(layer) 
        return layerlist
    else:
        return layer

def AveragePoolingLayer1D(pool_size, layerlist=None):
    """Creates AveragePooling layer

    Args:
        pool_size: pooling size
        strides: stride for pooling operations
        
    Returns:
        Keras AveragePooling Layer
    """

    layer = AveragePooling1D(pool_size)
    if layerlist is not None:
        layerlist.mylist.append(layer) 
        return layerlist
    else:
        return layer

def AveragePoolingLayer2D(pool_size, layerlist=None):
    """Creates AveragePooling layer

    Args:
        pool_size: pooling size
        strides: stride for pooling operations
        
    Returns:
        Keras AveragePooling Layer
    """
    pool_size = (pool_size, pool_size)
    layer = AveragePooling2D(pool_size)
    if layerlist is not None:
        layerlist.mylist.append(layer) 
        return layerlist
    else:
        return layer

def FlattenLayer(layerlist):
    """Creates Flatten layer
    Returns:
        Keras Flatten Layer
    """
    # Some wierd behavior causes this to add a bunch of layers randomly
    layerlist.mylist.append(Flatten())
    return layerlist

def ZeroPadding2DLayer(layerlist):
    """
    Adds Zero Padding Layer

    Returns:
        Keras Zero Padding Layer
    """
    layerlist.mylist.append(ZeroPadding2D())
    return layerlist

def GlobalMaxPoolingLayer1D(layerlist):
    """Creates GlobalMaxPooling layer

        
    Returns:
        Keras GlobalMaxPooling Layer
    """
    layerlist.mylist.append(GlobalMaxPool1D()) 
    return layerlist

def GlobalAveragePoolingLayer1D(layerlist):
    """Creates GlobalAveragePooling layer

        
    Returns:
        Keras MaxPooling Layer
    """
    layerlist.mylist.append(GlobalAveragePooling1D()) 
    return layerlist


def GlobalMaxPoolingLayer2D(layerlist):
    """Creates GlobalMaxPooling layer

        
    Returns:
        Keras GlobalMaxPooling Layer
    """
    layerlist.mylist.append(GlobalMaxPool2D()) 
    return layerlist


def GlobalAveragePoolingLayer2D(layerlist):
    """Creates GlobalMaxPooling layer

        
    Returns:
        Keras GlobalMaxPooling Layer
    """
    layerlist.mylist.append(GlobalAveragePooling2D()) 
    return layerlist

def AttentionLayer(layerlist, layerlist2):
    empty_layerlist = LayerList()
    attentionlist = [Attention(),layerlist.mylist,layerlist2.mylist]
    empty_layerlist.mylist.append(attentionlist)
    return empty_layerlist

def InitLayerList():
    return []

def ConcatenateLayer(*argv):
    # getting an error where the layerlist has a regular list inside of it, raising an error when layerlist.mylist gets called
    empty_layerlist = LayerList()
    concatlist = [Concatenate(axis=-1)]
    concatlist = concatlist + [arg.mylist for arg in argv]
    empty_layerlist.mylist.append(concatlist)
    return empty_layerlist

def ResBlock(filters, layerlist):
    # empty_layerlist = LayerList()
    # addlist = [Add()]
    # addlist = addlist + [arg.mylist for arg in argv]
    # empty_layerlist.mylist.append(addlist)
    # return empty_layerlist

    layerlist.mylist.append(('resblock',filters))
    return layerlist

def BERTInputLayer(data_pair, out_dim, activationDense, regularizer):
    '''
    Creates an empty layerlist with a DistilBERT pretrained model with a dense layer attached to the end
        - regular BERT layer is currently not modular because of input issues (it doesn't recognize hidden states from other layers at the moment)
    Also tokenizes data with the library's native tokenizer

    returns: Dense layer in tensor form
    '''
    if data_pair.get_datatype() == 'textdata':
        empty_layerlist = LayerList()
        activationDense = activation_handler(activationDense,'tanh')

        # config = DistilBertConfig(dim=MAXLEN)
        transformer = TFDistilBertModel.from_pretrained('distilbert-base-uncased')#,config=config)
        # print(transformer, x_train[0])
        input_ids = Input(shape=(MAXLEN,),dtype=tf.int32)
        # print(input_ids.shape, input_ids[0])
        sequence_output = transformer(input_ids)[0]
        cls_token = sequence_output[:,0,:]

        if regularizer==0:
            reg=None
        else:
            reg=l2(10**(-1*regularizer))

        out = Dense(out_dim,activation=activationDense,kernel_regularizer=reg)(cls_token)

        empty_layerlist.mylist.append(input_ids)
        empty_layerlist.mylist.append(out)
    elif data_pair.get_datatype() == 'imagedata': # does nothing for image data
        s = data_pair.get_train_data().get_instances()[0].get_stream().get_data().shape # input shape
        empty_layerlist.mylist.append(Input(shape=s))
    return empty_layerlist

# PREPROCESSING FUNCTIONS

def tokenizer(data_pair, maxlen, numwords):
    train_data = data_pair.get_train_data().get_numpy().flatten()
    test_data = data_pair.get_test_data().get_numpy().flatten()
    tokenizer = Tokenizer(num_words = numwords)
    tokenizer.fit_on_texts(train_data)
    x_train = tokenizer.texts_to_sequences(train_data)
    x_test = tokenizer.texts_to_sequences(test_data)
    x_train = pad_sequences(x_train, maxlen = maxlen)
    x_test = pad_sequences(x_test, maxlen = maxlen)

    data_list = []
    for transformed, dataset in zip([x_train, x_test], [data_pair.get_train_data(), data_pair.get_test_data()]):
        instances = cp.deepcopy(dataset.get_instances())
        for i, instance in enumerate(instances):
            instance.get_features().set_data([transformed[i]])
        new_dataset = EmadeData(instances)
        data_list.append(new_dataset)
    # data_pair = EmadeDataPair(train_data=(data_list[0], None), test_data=(data_list[1], None))
    data_pair.set_train_data(data_list[0])
    data_pair.set_test_data(data_list[1])
    vocabsize = np.amax(x_train) +1
    return data_pair, vocabsize, tokenizer

def processImages(data_pair):
    # for now, this scales images to be between 0-1 instead of 0-255
    training_data = np.array([instance.get_stream().get_data() for instance in data_pair.get_train_data().get_instances()])
    testing_data = np.array([instance.get_stream().get_data() for instance in data_pair.get_test_data().get_instances()])
    # these are numpy arrays of shape (n_examples, image_width, image_height)
    
    # get the max intensity of the train data (in case it's not out of 255)
    ceiling = np.max(training_data.flatten())
    x_train = training_data/ceiling
    x_test = testing_data/ceiling

    # overwrite train/test data in data_pair
    data_list = []
    for transformed, dataset in zip([x_train, x_test], [data_pair.get_train_data(), data_pair.get_test_data()]):
        instances = cp.deepcopy(dataset.get_instances())
        for i, instance in enumerate(instances):
            instance.get_features().set_data([transformed[i]])
        new_dataset = EmadeData(instances)
        data_list.append(new_dataset)

    data_pair.set_train_data(data_list[0])
    data_pair.set_test_data(data_list[1])

    return data_pair

def preprocessDataPair(data_pair):
        # Does nontrivial preprocessing on the original datapair instead of
        # preprocessing for every individual

        # for now, this means tokenizing text data, one hot encoding
        # target values for multiclass data, and normalizing image data

        if data_pair.get_datatype()=='textdata':
            data_pair, vocabsize, tok = tokenizer(data_pair, MAXLEN, NUMWORDS)
            data_pair.vocabsize = vocabsize
            data_pair.tokenizer = tok
        elif data_pair.get_datatype()=='imagedata' or data_pair.get_datatype()=='detectiondata':
            data_pair = processImages(data_pair)

        elif data_pair.get_datatype()=='recdata':
            pass
        else:
            pass

        if len(np.unique(data_pair.get_train_data().get_target())) > 2:
            # data is multiclass, one hot encode target data so that 
            # categorical crossentropy works
            le = preprocessing.OneHotEncoder(sparse=False)
            target_values = data_pair.get_train_data().get_target().reshape(-1,1)
            le.fit(target_values)
            target_values = le.transform(target_values)
            truth_data = data_pair.get_truth_data().reshape(-1,1)
            truth_data = le.transform(truth_data)
            
            data_pair.get_train_data().set_target(target_values)
            data_pair._truth_data = truth_data # test data was already cleared so I don't think I can set it using a class method
        
        return data_pair

def all_caps(word):
    return len(word) > 1 and word.isupper()

def embed_word(embedding_matrix,i,word,embeddings_index_ft,embeddings_index_tw):
    embedding_vector_ft = embeddings_index_ft.get(word)
    if embedding_vector_ft is not None:
        if all_caps(word):
            last_value = np.array([1])
        else:
            last_value = np.array([0])
        embedding_matrix[i,:300] = embedding_vector_ft
        embedding_matrix[i,500] = last_value
        embedding_vector_tw = embeddings_index_tw.get(word)
        if embedding_vector_tw is not None:
            embedding_matrix[i,300:500] = embedding_vector_tw



def get_glove_fasttext_embedding_matrix(vocab_size, tokenizer):
    def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
    EMBEDDING_FILE_TWITTER = 'embeddings/glove.twitter.27B.200d.txt'
    EMBEDDING_FILE_FASTTEXT = 'embeddings/crawl-300d-2M.vec'
    embeddings_index_ft = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE_FASTTEXT,encoding='utf-8'))
    embeddings_index_tw = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE_TWITTER,encoding='utf-8'))
    embedding_matrix = np.zeros((vocab_size,501))   
    something_tw = embeddings_index_tw.get("something")
    something_ft = embeddings_index_ft.get("something")
    something = np.zeros((501,))
    something[:300,] = something_ft
    something[300:500,] = something_tw
    something[500,] = 0
    for word, i in tokenizer.word_index.items():
        if i > vocab_size - 1: break
 
        if embeddings_index_ft.get(word) is not None:
            embed_word(embedding_matrix,i,word, embeddings_index_ft,embeddings_index_tw)
        else:
            # change to > 20 for better score.
            if len(word) > 20: 
                embedding_matrix[i] = something
            else:
                #word2 = correction(word)
                word2 = cp.deepcopy(word)
                if embeddings_index_ft.get(word2) is not None:
                    embed_word(embedding_matrix,i,word2, embeddings_index_ft,embeddings_index_tw)
                else:
                    word2 = cp.deepcopy(word)
                   # word2 = correction(singlify(word))
                    if embeddings_index_ft.get(word2) is not None:
                        embed_word(embedding_matrix,i,word2, embeddings_index_ft,embeddings_index_tw)
                    else:
                        embedding_matrix[i] = something
    return embedding_matrix

def get_fasttext_embedding_matrix(vocab_size, tokenizer):
    """Creates glove embedding matrix

    Returns:
        glove embedding matrix
    """
    def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
    EMBEDDING_FILE_FASTTEXT = 'embeddings/crawl-300d-2M.vec'
    embeddings_index_ft = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE_FASTTEXT,encoding='utf-8'))

    embeddings_dictionary = embeddings_index_ft
    embedding_matrix = np.zeros((vocab_size, 300))  
    for word, index in tokenizer.word_index.items():
        if index > vocab_size - 1:
            break
        else:
            embedding_vector = embeddings_dictionary.get(word)  
            if embedding_vector is not None:    
                embedding_matrix[index] = embedding_vector  
    return embedding_matrix 


def get_embedding_matrix(initializer, vocab_size, tokenizer): 
    """Creates glove embedding matrix

    Returns:
        glove embedding matrix
    """
    def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
    
    if initializer is WeightInitializer.GLOVEFASTTEXT:
        return get_glove_fasttext_embedding_matrix(vocab_size, tokenizer)
    elif initializer is WeightInitializer.FASTTEXT:
        EMBEDDING_FILE_FASTTEXT = 'embeddings/crawl-300d-2M.vec'
        embeddings_dictionary = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE_FASTTEXT,encoding='utf-8'))
        embedding_matrix = np.zeros((vocab_size, 300))
    elif initializer is WeightInitializer.GLOVE:
        EMBEDDING_FILE_GLOVE = 'embeddings/glove.6B.100d.txt'
        embeddings_dictionary = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE_GLOVE,encoding='utf-8'))
        embedding_matrix = np.zeros((vocab_size, 100))
    elif initializer is WeightInitializer.GLOVETWITTER:
        EMBEDDING_FILE_TWITTER = 'embeddings/glove.twitter.27B.200d.txt'
        embeddings_dictionary = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE_TWITTER,encoding='utf-8'))
        embedding_matrix = np.zeros((vocab_size, 200))
    for word, index in tokenizer.word_index.items():
        if index > vocab_size - 1:
            break
        else:
            embedding_vector = embeddings_dictionary.get(word)  
            if embedding_vector is not None:    
                embedding_matrix[index] = embedding_vector  
    return embedding_matrix 

def tokenize_BERT(data_pair,maxlen):
    train_data = data_pair.get_train_data().get_numpy().flatten()
    test_data = data_pair.get_test_data().get_numpy().flatten()

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    x_train=[]
    x_test=[]

    def convert(texts,tok_output,maxlen):
        for text in texts:
            text = tokenizer.tokenize(text)
            text = text[:maxlen-2]
            input_sequence = ["[CLS]"] + text + ["[SEP]"]
            pad_len = maxlen - len(input_sequence)

            tokens = tokenizer.convert_tokens_to_ids(input_sequence)
            tokens +=[0]*pad_len

            tok_output.append(np.asarray(tokens))
        return tok_output
    x_train = np.asarray(convert(train_data,x_train,maxlen))
    x_test = np.asarray(convert(test_data,x_test,maxlen))
    return x_train,x_test



# NNLEARNER AND MODULE CODE

def find_layer(comp_list):
    # takes an uncompiled layerlist (comp_list) and goes through it in reverse order until it finds a layer that has an activation

    for i, layer in enumerate(comp_list[::-1]):
        if hasattr(layer, "activation"):
            layer_index = -1 - i
            break
    return layer_index
def Module(data_type, data_dims, layerlist, skip_connection, pooling_layer, dropout_rate):
    """
    Core method for modules
        - Compiles everything in its layerlist except for other models/modules into a keras model
        - also adds skip connection, a pooling layer, and a dropout layer to the model if applicable
        - Overwrites out dimensions of layers in the module such that they increase sequentially. E.g., Dense32(Conv64(Dense(32))) --> Dense64(Conv64(Dense(32)))
    """
    # Compile everything except other models/modules
    ret_list = [mod for mod in layerlist.mylist if type(mod) == keras.engine.functional.Functional or mod == 'input'] # the part of the layerlist contributed by inputlayers and modules
    comp_list = [mod for mod in layerlist.mylist if type(mod) !=  keras.engine.functional.Functional] # the layerlist that's a part of this module

    if ret_list != ["input"]:
        data_dims = ret_list[-1].output_shape[1:] # store the output shape of the previous module
    out_dim = None

    # Go through layers sequentially. If a layer's out dimension is lower than the one before it, inherit the previous layer's out dim
    last_layer = None
    for i,layer in enumerate(comp_list):
        if hasattr(layer, "filters") or hasattr(layer, "units"):
            if hasattr(layer, "filters"):
                attr = "filters"
            else:
                attr = "units"

            if out_dim is None or out_dim < getattr(layer, attr):
                out_dim = getattr(layer, attr)
            elif out_dim > getattr(layer, attr):
                setattr(layer, attr, out_dim)
            
            last_layer = (i, attr) # lets us track where the last "real" layer in comp_list is


    # if a skip connection is used, we need to override some things to make sure the input and output tensor shapes are the same
    # - change out_dim of the skip connection's layer (if applicable) to match the last layer in the module
    # - remove the activation of the last layer 
    if skip_connection != (None, None):
        layer_index = find_layer(comp_list)

        if skip_connection[0] is None:
            # apply a "bottleneck" (change the filters so we can have a vanilla skip connection) if needed
            if data_dims[-1] != out_dim:
                setattr(comp_list[last_layer[0]], last_layer[1], data_dims[-1])
        else:
            # skip connection's layer had placeholder value, replace with out dimension of module's layerlist
            if hasattr(skip_connection[0], "units"):
                setattr(skip_connection[0], "units", out_dim)
            else:
                setattr(skip_connection[0], "filters", out_dim)
            
            comp_list[layer_index].activation = linear


    outputs, input_layers = compile_layerlist(comp_list, [], data_dims, data_type)

    if skip_connection != (None, None):
        if skip_connection[0] is None: # simple add layer
            residual = input_layers[0]
        else: # another layer in the skip connection
            if len(skip_connection) == 3: 
                # last element of skip_connection tuple is a normalization layer
                residual = skip_connection[2](skip_connection[0](input_layers[0]))
            else:
                residual = skip_connection[0](input_layers[0])
        
        # before we add layers, need to check that their spatial dimensions match
        # if they don't, we add zero padding to the smaller one (currently only works for image data)
        if data_type == "imagedata" and outputs.shape[1:3] != residual.shape[1:3]:
            dims = [outputs.shape[1], residual.shape[1]]
            lower = np.argmin(dims) # we add padding to the spatially smaller one
            if lower==0:
                shape_diff = residual.shape[1] - outputs.shape[1]
            else:
                shape_diff = outputs.shape[1] - residual.shape[1]

            if shape_diff % 2 == 0:
                # can add symmetric padding
                padding = shape_diff//2
            else:
                smaller = int(np.floor(shape_diff/2))
                padding = ((smaller, smaller+1), (smaller, smaller+1))

            if lower==0:
                outputs = ZeroPadding2D(padding=padding)(outputs)
            else:
                residual = ZeroPadding2D(padding=padding)(residual)

        outputs = Add()([outputs, residual])
        outputs = Activation_Wrapper(skip_connection[1])(outputs) # add nonlinearity here
    
    if pooling_layer is not None:
        outputs = pooling_layer(outputs)
    
    if dropout_rate is not None:
        outputs = Dropout(dropout_rate)(outputs)

    # modList can be empty when I'm doing mating/mutation or just spawning modules outside of an nnlearner
    if (len(layerlist.modList) > layerlist.curr):
        module_name = layerlist.modList[layerlist.curr]
        try:
            loaded=False
            while not loaded:
                try:
                    with open("Global_MODS", "rb") as mod_file:
                        glob_mods = dill.load(mod_file)
                    loaded=True
                except EOFError:
                    time.sleep(15)
            model = Model(inputs=input_layers, outputs=outputs, name=module_name + "_" + str(layerlist.curr))
            if (glob_mods[module_name][1].weights):
                model.set_weights(glob_mods[module_name][1].weights)
        except Exception as e:
            print(e)
            model = Model(inputs=input_layers, outputs=outputs, name=module_name)
    else:
        model = Model(inputs=input_layers, outputs=outputs)
    layerlist.curr += 1
    ret_list.append(model)
    layerlist.mylist = ret_list
    return layerlist

def NNLearner(data_pair, layerlist, optimizer_enum, image_augmentations=None):
    """
    Core method for all neural nets
    Creates the model
    Handles caching functions
    Loads and store the data and classification

    Args:
        data_pair: data structure storing train and test data
        layerlist: data structure storing model layers
        optimizer_enum: object storing which optimizer to use

    Returns:
        updated data pair with new data
    """
    data_pair = cp.deepcopy(data_pair)
    optimizer = optimizer_enum.value

    training_object = data_pair.get_train_data()
    testing_object = data_pair.get_test_data()

    if data_pair.get_datatype()=='textdata': 
        training_data = training_object.get_numpy()
        testing_data = testing_object.get_numpy()
    elif data_pair.get_datatype()=='imagedata' or data_pair.get_datatype()=='detectiondata':
        training_data = np.array([instance.get_stream().get_data() for instance in training_object.get_instances()])
        testing_data = np.array([instance.get_stream().get_data() for instance in testing_object.get_instances()])
        # target_values = data_pair.get_train_data().get_target()#.reshape(-1,1)
        # truth_data = data_pair.get_truth_data()#.reshape(-1,1)
    elif data_pair.get_datatype()=='recdata':
        # might be broken
        training_data = training_object.get_numpy().astype(int)
        testing_data = testing_object.get_numpy().astype(int)
        # target_values = data_pair.get_train_data().get_target()#.reshape(-1,1)
        # truth_data = data_pair.get_truth_data()
        traintestuserdata = np.hstack([training_data[:,0], testing_data[:,0]])
        traintestitemdata = np.hstack([training_data[:,1], testing_data[:,1]])
        #print(traintestuserdata)
        user_enc = LabelEncoder()
        user_enc.fit(traintestuserdata)
        user_train_data = user_enc.transform(training_data[:,0])
        user_test_data = user_enc.transform(testing_data[:,0])
        item_enc = LabelEncoder()
        item_enc.fit(traintestitemdata)
        item_train_data = item_enc.transform(training_data[:,1])
        item_test_data = item_enc.transform(testing_data[:,1])
        #print(target_values, truth_data)
            
    else:
        raise NotImplementedError('only text data, image data, and detection data currently work with NNLearner. Passed datatype: {}'.format(data_pair.get_datatype()))

    target_values = training_object.get_target()
    truth_data = data_pair.get_truth_data()

    batch_size = 100
    epochs = 8 # 8 is what CoDeepNEAT did

    if image_augmentations is not None:
        layerlist.mylist.insert(1,image_augmentations)

    start_time = time.time()
    print("layerlist: ", layerlist.mylist)
    for mod in layerlist.mylist:
        if type(mod) == keras.engine.functional.Functional:
            mod.summary()
    
    try:
        '''
        Main Block
        '''
        if layerlist.mylist[-1]=='output': # no need to require an output layer if we manually add it after compile_layerlist runs. Keeping the primitive for legacy support (and thus this code)
            layerlist.mylist.pop()

        # before we compile the layerlist, let's make sure all the functional models have unique names (can cause errors)
        names = set()
        i=0
        while i < len(layerlist.mylist):
            node = layerlist.mylist[i]
            if type(node) == keras.engine.functional.Functional:
                if node._name not in names:
                    names.add(node._name)
                    i+=1
                else:
                    node._name = node._name + "_0"
            else:
                i+=1
        curr_layer, input_layers = compile_layerlist(layerlist.mylist, [], data_pair.get_train_data().get_instances()[0].get_stream().get_data().shape, data_pair.get_datatype(), isNNLearner=True)
        
        if len(data_pair.get_train_data().get_target().shape)==1:
            out_dim = 1
            activation = 'sigmoid'
        else:
            out_dim = data_pair.get_train_data().get_target().shape[1]
            activation = 'softmax'
        
        newlayer = Dense(out_dim, activation=activation)

        # We allow NNLearners to flatten by themselves but if they don't we lend a helping hand for sake of efficiency
        if (len(curr_layer.shape) > 2):
            curr_layer = Flatten()(curr_layer)
        outputs = newlayer(curr_layer)

        model = Model(inputs=input_layers, outputs=outputs)
        

        if data_pair.get_datatype() == 'recdata':
            raise NotImplementedError
            n_users = len(user_enc.classes_)
            n_items = len(item_enc.classes_)
            numforembedding = [n_users, n_items]
            i = 0
            config = model.to_json()
            print("config: ",type(config))
            data = json.loads(config)
            for layer in data['config']['layers']:
                if layer['class_name']=='Embedding':
                    if data_pair.get_datatype() == 'recdata':
                        layer['config']['input_dim'] = numforembedding[i]
                        i+=1
                    # elif data_pair.get_datatype() == 'textdata':
                    #     layer['config']['input_dim'] = vocab_size

            json_object = json.dumps(data)
            print(json_object)
            model = model_from_json(json_object)

        elif data_pair.get_regression():
            model.compile(loss= 'mean_squared_error', optimizer = optimizer, metrics= ['accuracy'])

        elif out_dim != 1:
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        else:
            model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        model.summary()

        """
        Fit estimator to training data
        Predict labels of testing data
        """


        train_data_list = []
        test_data_list = []
        if data_pair.get_datatype()=='recdata':
            train_data_list.append(user_train_data)
            test_data_list.append(user_test_data)
            train_data_list.append(item_train_data)
            test_data_list.append(item_test_data)
        else:
            for _ in range(len(input_layers)):
                train_data_list.append(training_data)
                test_data_list.append(testing_data)
  
        es = EarlyStopping(monitor='val_loss', min_delta=0.001, verbose=1, patience=1, restore_best_weights=True)
        if not data_pair.get_regression():
            # we use a second early stopping callback to monitor training with respect to a "trivial" baseline. If train acc is worse than baseline, we stop
            # we define trivial accuracy as less than 1% better than a random guess for a classification task
            # In the future, might be faster to define a custom EarlyStopping class for es2 that looks at accuracy after a few batches instead of after an epoch
            if out_dim == 1:
                trivial_acc = 0.51
            else:
                trivial_acc = 1/out_dim + 0.01
            es2 = EarlyStopping(monitor='accuracy', min_delta=0.0, verbose=1, patience=1, restore_best_weights=True, baseline=trivial_acc)
        else:
            es2 = None
        ts = TimedStopping(seconds=1800, verbose=1)
        cb_list = [es,ts]
        if es2 is not None:
            cb_list.append(es2)

        # we specify the number of steps to take through the training/validation data because
        # if the dataset size is not divisible by the batch size, an issue can happen where
        # training never goes past the first epoch
        steps_per_epoch = training_data.shape[0]//batch_size
        # validation_steps = testing_data.shape[0]//batch_size # see above; commented out cause the error doesn't seem to happen for validation data

        history = model.fit(train_data_list, target_values, batch_size=batch_size, epochs = epochs, validation_data=(test_data_list, truth_data), callbacks=cb_list, verbose=2, steps_per_epoch = steps_per_epoch)
        # history = model.fit(train_data_list, target_values, batch_size=batch_size, epochs = epochs, validation_data=(test_data_list, truth_data), callbacks=None, verbose=2, steps_per_epoch = steps_per_epoch) # removed callbacks for testing TODO

        sys.stdout.flush()
        targ_pred = model.predict(test_data_list)

        #Converting predictions to label
        if not data_pair.get_regression():
            if out_dim != 1:
                pred = tf.one_hot(np.argmax(targ_pred,axis=1),depth=out_dim)
            else:
                pred = np.round(targ_pred)

        data_pair.set_num_params(model.count_params())   

        data_pair.get_test_data().set_target(pred)#.ravel())
        
        loaded=False
        while not loaded:
            try:
                with open("Global_MODS", "rb") as mod_file:
                    glob_mods = dill.load(mod_file)
                loaded=True
            except EOFError:
                time.sleep(15)
        temp = [node for node in layerlist if type(node) == keras.engine.functional.Functional and node.name != "ImageAugmentation"]
        if (len(temp) > 0) and len(data_pair.modList) > 0:
            i=0
            j=0
            embed_first = len(temp)>len([elem for elem in data_pair.modList if elem.startswith("mod_")]) # true if we have to deal with an embedding layer
            while i<len(temp):
                if data_pair.modList[i+j] == 'input':
                    j+=1
                    continue

                # Alternative to this: pass weights down instead of making them global (?)
                if not (embed_first and i==0):
                    glob_mods[data_pair.modList[i+j]][1].weights = temp[i].get_weights() 
                else:
                    # we're dealing with an embedding layer. This isn't stored in modlist
                    # so we access it in glob_mods using its name
                    try:
                        glob_mods[temp[i].name] = [temp[i].name, temp[i].get_weights(), None]
                    except Exception as e:
                        glob_mods[temp[i].name] = [temp[i].name, None, None]
                    j-=1
                i+=1
            with open("Global_MODS", "wb") as mod_file:
                dill.dump(glob_mods, mod_file)
        else:
            if len(temp)>0:
                print("Did not share weights because modlist is empty (why is modlist empty???)")

        layerlist.curr = 0
        sys.stdout.flush()




    except (KeyboardInterrupt, Exception) as e:
        """
        Handle Errors
        """
        if data_pair.get_caching_mode():
            if isinstance(e, KeyboardInterrupt):
                database.set_method_time(data_row, time.time() - start_time)
                del database
            else:
                if "kill_process" in str(e):
                    database.set_special_error("time/memory errored occurred \
                                               but data was saved")
                    del database
                    # after the data is saved we need to kill the process
                    # before it goes further past the time/memory limit
                    os.kill(os.getpid(), 9)
                database.set_error(data_row, str(e))
                del database
        raise e

    #if data_pair.get_caching_mode():
        #del database

    gc.collect();

    return data_pair
   
def compile_layerlist(layerlist, input_layers, data_dim, datatype, isNNLearner=False):
    curr_layer = None
    for layer_index, layer in enumerate(layerlist):
        if isinstance(layer, list):
            out_layers = []
            for i in range(1,len(layer),1):
                #print('layer: ', layer[i])
                out_layer, input_layers = compile_layerlist(layer[i], input_layers, data_dim, datatype)
                out_layers.append(out_layer)
            curr_layer = layer[0](out_layers)
        else:
            if isinstance(layer, tf.Tensor):
                #print('layer2: ', layer)
                input_layers.append(layer)
                curr_layer = layer
            
            # Handle input/output layer
            if isinstance(layer, str):
                if layer == 'input' and curr_layer == None:
                    if datatype == 'textdata':
                        # before we add the input layer, need to check if the next layer is 
                        # an embedding layer -- if not, we add an out dimension of 1 to the shape
                        if layer_index != len(layerlist)-1 and isinstance(layerlist[layer_index+1], tf.keras.layers.Embedding): 
                            new_layer = Input(shape=(MAXLEN,))
                        else:
                            new_layer = Input(shape=(MAXLEN,1))
                    elif datatype == 'imagedata':
                        s = data_dim # input shape
                        if (len(s) == 2):
                            s = (s[0], s[0], 1)
                        new_layer = Input(shape=s)
                    elif datatype == 'recdata':
                        new_layer = Input((1,))
                    input_layers.append(new_layer)
                    curr_layer = new_layer
                elif layer == 'output':
                    layerlist.mylist.pop()

            # # Handle Keras applications
            # elif isinstance(layer, dict):
            #     if layer['type'] == PretrainedModel.MOBILENET:
            #         s = data_pair.get_train_data().get_instances()[0].get_stream().get_data().shape
            #         new_layer = MobileNet(input_shape=s, include_top = False, weights = "imagenet")
            #         input_layers.append(new_layer.inputs)
            #         new_layer = new_layer.output
            #         curr_layer = new_layer
            #     if layer['type'] == PretrainedModel.INCEPTION:
            #         s = data_pair.get_train_data().get_instances()[0].get_stream().get_data().shape
            #         new_layer = InceptionV3(input_shape=s, include_top = False, weights = "imagenet")
            #         input_layers.append(new_layer.inputs)
            #         new_layer = new_layer.output
            #         curr_layer = new_layer
            #     if layer['type'] == PretrainedModel.VGG:
            #         s = data_pair.get_train_data().get_instances()[0].get_stream().get_data().shape
            #         new_layer = VGG16(input_shape=s, include_top = False, weights = "imagenet")
            #         input_layers.append(new_layer.inputs)
            #         new_layer = new_layer.output
            #         curr_layer = new_layer
            elif isinstance(layer, tuple):
                if layer[0]=='resblock':
                    bn1 = BatchNormalization()(curr_layer)
                    act1 = relu(bn1)
                    conv1 = Conv2D(filters=layer[1], kernel_size=(3, 3), strides=(2, 2))(act1)
                    # print('conv1.shape', conv1.shape)
                    bn2 = BatchNormalization()(conv1)
                    act2 = relu(bn2)
                    conv2 = Conv2D(filters=layer[1], kernel_size=(3, 3), strides=(1, 1))(act2)
                    # print('conv2.shape', conv2.shape)
                    residual = Conv2D(1, (1, 1), strides=(1, 1))(conv2)
                    
                    x = Conv2D(filters=layer[1], kernel_size=(3, 3), strides=(2, 2))(curr_layer)
                    # print('x.shape', x.shape)
                    curr_layer = Add()([x, residual])

            else:
                if curr_layer == None:
                    curr_layer = layer
                    input_layers.append(layer)
                else:
                    if isNNLearner and type(layer) == keras.engine.functional.Functional:
                        layer = change_model(layer, curr_layer.shape)
                    curr_layer = layer(curr_layer)
    return curr_layer, input_layers

def change_model(model, new_input_shape):
    # replace input shape of first layer
    model.layers[0]._batch_input_shape = new_input_shape

    # rebuild model architecture by exporting and importing via json
    new_model = keras.models.model_from_json(model.to_json())
    #new_model.summary()

    # copy weights from old model to new one
    for layer in new_model.layers:
        try:
            layer.set_weights(model.get_layer(name=layer.name).get_weights())
        except:
            print("Could not transfer weights for layer {}".format(layer.name))

    # # test new model on a random input image
    # X = np.random.rand(10, 40, 40, 3)
    # y_pred = new_model.predict(X)
    # print(y_pred)

    return new_model


def compiled(num_classes):
    """provides the list of arguments for the for the model compiler
    Args:
        num_classes: the number of classes within the target data
    Returns:
        return arguments the configuration of the model including the optimizer, the loss function and the list of metric to be evaluated
    """
    if num_classes <= 2:
        return ['binary_crossentropy', 'adam', ['accuracy']]
    else:
        return ['categorical_crossentropy', 'adam', ['accuracy']]

def get_adf_layer_frequencies(filename: str) -> dict:
    """Get the layer frequencies of all global ADFs."""
    with open(filename, "rb") as f:
        adfs = dill.load(f)
    return {f'{mod}': extract_layer_frequencies(str(mod_val[1][0])) for mod, mod_val in adfs.items()}


def get_individual_layer_frequencies(individual) -> Counter:
    """Given an individual containing ADFs, extract its layer frequencies."""
    adf_layer_frequencies = get_adf_layer_frequencies("Global_MODS")
    individual_layer_frequencies = extract_layer_frequencies(str(individual[0]))
    layer_frequencies = Counter()
    for layer, frequency in individual_layer_frequencies.items():
        if layer.startswith('mod_'):
            layer_frequencies += adf_layer_frequencies[layer]
        elif layer != 'NNLearner':
            layer_frequencies[layer] += frequency
    return layer_frequencies


def extract_layer_frequencies(layers: str) -> Counter:
    """Given a string of layers, extract frequency counts."""
    counts = Counter(re.findall('\w+(?=\()', layers))
    del counts['Module']
    return counts

###############################################################################################################
################# OVERRIDING DEAP METHODS #####################################################################
###############################################################################################################

def genADF(pset, min_, max_, data_dim, data_type, type_=None):
        """Generate an expression where each leaf has the same depth
        between *min* and *max*.

        :param pset: Primitive set from which primitives are selected.
        :param min_: Minimum height of the produced trees.
        :param max_: Maximum Height of the produced trees.
        :param type_: The type that should return the tree when called, when
                    :obj:`None` (default) the type of :pset: (pset.ret)
                    is assumed.
        :returns: A full tree with all leaves at the same depth.
        """

        def condition(height, depth):
            """Expression generation stops when the depth is equal to height."""
            return depth == height

        return generateADF(pset, min_, max_, condition, data_dim, data_type, type_)


def genNNLearner(pset, min_, max_, data_dim, data_type, type_=None):
        """Generate an expression where each leaf has the same depth
        between *min* and *max*.

        :param pset: Primitive set from which primitives are selected.
        :param min_: Minimum height of the produced trees.
        :param max_: Maximum Height of the produced trees.
        :param type_: The type that should return the tree when called, when
                    :obj:`None` (default) the type of :pset: (pset.ret)
                    is assumed.
        :returns: A full tree with all leaves at the same depth.
        """

        def condition(height, depth):
            """Expression generation stops when the depth is equal to height."""
            return depth == height
        def condition2(height, depth):
            """Expression generation will stop after this primitive; send a signal to add an input layer"""
            return depth+1 == height
        loaded=False
        while not loaded:
            try:
                with open("Global_MODS", "rb") as mod_file:
                    glob_mods = dill.load(mod_file)
                loaded=True
            except EOFError:
                time.sleep(15)
        return generateNNLearner(pset, min_, max_, condition, data_dim, data_type, glob_mods, type_, condition2)

# genADF now provides valid mods alongside a reshape value which describes 
# the change in dimension of an input given this mod.
def generateADF(pset, min_, max_, condition, data_dim, data_type, type_=None):
    """Generate a Tree as a list of list. The tree is build
    from the root to the leaves, and it stop growing when the
    condition is fulfilled.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param condition: The condition is a function that takes two arguments,
                    the height of the tree to build and the current
                    depth in the tree.
    :param type_: The type that should return the tree when called, when
                :obj:`None` (default) the type of :pset: (pset.ret)
                is assumed.
    :returns: A grown tree with leaves at possibly different depths
            dependending on the condition function.
    """
    valid=False
    orig_pset = cp.deepcopy(pset)
    while not valid:
        reshape = 0
        pset = cp.deepcopy(orig_pset)
        type_ = pset.ret
        if type_ is None:
            type_ = pset.ret
        expr = []
        height = random.randint(min_, max_)
        stack = [(0, type_)]
        while len(stack) != 0:
            depth, type_ = stack.pop()
            if condition(height, depth) or len(pset.primitives[type_])==0:
                try:
                    term = random.choice(pset.terminals[type_])
                except IndexError:
                    _, _, traceback = sys.exc_info()
                    raise IndexError("The gp.generate function tried to add "\
                                    "a terminal of type '%s', but there is "\
                                    "none available." % (type_,)).with_traceback(traceback)
                if gp.isclass(term):
                    term = term()
                expr.append(term)
            else:
                try:
                    prim = random.choice(pset.primitives[type_])
                except IndexError:
                    _, _, traceback = sys.exc_info()
                    raise IndexError("The gp.generate function tried to add "\
                                    "a primitive of type '%s', but there is "\
                                    "none available." % (type_,)).with_traceback(traceback)
                expr.append(prim)
                for arg in reversed(prim.args):
                    stack.append((depth + 1, arg))

        try: 
            # We're compiling automatically defined functions using keras to ensure 
            # our layer combinations are valid before using them. 
            tree = gp.PrimitiveTree(expr)
            func = gp.compile(tree, pset)
            func = func(InputLayer())
            layers, _ = compile_layerlist(func, [], data_dim, data_type)
            # This subtracts our input data size from the output dimensions of our compiled layerlist.
            # Upon second glance, I thought we should prevent the case where output dimensions might be 0.
            reshape = np.subtract(data_dim, layers.shape[1:])
            cond = sum(1 for number in layers.shape[1:] if number <= 0)
            if (cond > 0):
                raise Exception("Possible zero output is not valid for nnlearners.")
            valid=True
        except Exception as e:
            # return generateADF(orig_pset, min_, max_, condition, data_dim, data_type, orig_pset.ret)
            print(str(tree))
            print(e,'\n')
            # raise e
    return expr, reshape

# We need to prevent deap from being able to do invalid layer combinations that lead to error strings.
# We're gonna do this by manually overriding the generate mehod for neural nets and making considerations for the data dimensions.
def generateNNLearner(pset, min_, max_, condition, data_dim, data_type, glob_mods, type_=None, condition2=None):
    """Generate a Tree as a list of list. The tree is build
    from the root to the leaves, and it stop growing when the
    condition is fulfilled.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param condition: The condition is a function that takes two arguments,
                    the height of the tree to build and the current
                    depth in the tree.
    :param type_: The type that should return the tree when called, when
                :obj:`None` (default) the type of :pset: (pset.ret)
                is assumed.
    :param condition2: Similar to condition, but is True if this is 
                the last primitive that can be added before condition is met
    :returns: A grown tree with leaves at possibly different depths
            dependending on the condition function.
    """
    valid=False
    height = random.randint(min_, max_)
    while not valid:
        if type_ is None:
            type_ = pset.ret
        expr = []
        stack = [(0, type_)]
        while len(stack) != 0:
            depth, type_ = stack.pop()
            tempPSET = [x for x in pset.primitives[type_]]
            if condition(height, depth) or len(pset.primitives[type_])==0:
                try:
                    term = random.choice(pset.terminals[type_])
                except IndexError:
                    _, _, traceback = sys.exc_info()
                    raise IndexError("The gp.generate function tried to add "\
                                    "a terminal of type '%s', but there is "\
                                    "none available." % (type_,)).with_traceback(traceback)

                if gp.isclass(term):
                    term = term()
                expr.append(term)
                if term.ret == ModList:
                    # we've finished building the layerlist, let's compile to check validity
                    layerlist = expr[2:] # this line assumes layerlist is the 2nd arg in nnlearner 
                    tree = gp.PrimitiveTree(layerlist)
                    temp_context = cp.deepcopy(pset.context)
                    temp_context["ARG1"] = [] # pass empty modlist as placeholder
                    try:
                        func = eval(str(tree), temp_context, {})
                        # uncomment below to look at generated models
                        # curr_layer, input_layers = compile_layerlist(func, [], data_dim, data_type, isNNLearner=True)
                        # model = Model(inputs=input_layers, outputs=curr_layer)
                        # model.summary()
                        valid=True
                    except Exception as e:
                        # failed to compile (probably because of a negative dimension size)
                        # print(e)
                        if "Negative dimension size" not in str(e):
                            print(str(tree))
                            print(e,'\n')
                            type_ = None
                            # raise e
                            break
                        else:
                            # instead of regenerating from scratch, try to fix dimension size errors by randomly
                            # selecting a module and replacing it with a new one.
                            # This new module is randomly selected from the set of all mods minus those that 
                            # reduce spatial size the most, with higher probability assigned to those
                            # with less spatial reduction.
                            reshape_list = pset.mod_reshapes.copy()

                            spatial_changes = [reshape[1] for reshape in reshape_list]
                            max_reshape = max(spatial_changes)
                            reshape_list = [reshape for reshape in reshape_list if reshape[1] != max_reshape]

                            mod_list = [reshape[0] for reshape in reshape_list] # list of mod names
                            spatial_changes = [reshape[1] for reshape in reshape_list]
                            
                            # create pdf for module sampling (doesn't need to sum to 1, random.choices will create valid cdf for us)
                            pdf = [1/change if change!=0 else 1 for change in spatial_changes]

                            while not valid:
                                replace_index = random.choice([i for i in range(len(expr)) if expr[i].name.startswith('mod_')])
                                mod_name = random.choices(mod_list, weights=pdf)[0]
                                expr[replace_index] = pset.mapping[mod_name]

                                try:
                                    func = eval(str(gp.PrimitiveTree(expr[2:])), temp_context, {})
                                    valid=True
                                except Exception as e:
                                    pass

            else:
                try:
                    # check for condition2
                    if condition2 is not None and condition2(height, depth):
                        # need to get a primitive other than a module
                        # modules input and output a layerlist; we need something that 
                        # inputs only terminals but outputs a layerlist --> input layers
                        # easiest way to restrict the search is to look something with different inputs than outputs
                        terminal_primitives = [x for x in pset.primitives[type_] if type_ not in x.args]
                        prim = random.choice(terminal_primitives)
                    elif condition2 is not None:
                        # we don't want to choose an InputLayer here because then we're not fully utilizing the depth
                        non_terminal_primitives = [x for x in pset.primitives[type_] if ModList not in x.args]
                        prim = random.choice(non_terminal_primitives)
                    else:
                        prim = random.choice(tempPSET)
                except IndexError:
                    _, _, traceback = sys.exc_info()
                    raise IndexError("The gp.generate function tried to add "\
                                    "a primitive of type '%s', but there is "\
                                    "none available." % (type_,)).with_traceback(traceback)
                expr.append(prim)
                for arg in reversed(prim.args):
                    stack.append((depth + 1, arg))
    return expr