"""
Coded by Jason Zutty
Modified by VIP Team
Module to organize primitive and terminal additions
"""
import GPFramework.neural_network_methods as nnm
import GPFramework.learner_methods as lm
import GPFramework.signal_methods as sm
import GPFramework.spatial_methods as sp
import GPFramework.feature_extraction_methods as fem
import GPFramework.feature_selection_methods as fsm
import GPFramework.clustering_methods as cm
import GPFramework.decomposition_methods as dm
import GPFramework.operator_methods as opm
import GPFramework.text_processing_methods as tp
import GPFramework.legacy_methods as legacy
import GPFramework.detection_methods as dem
from GPFramework.data import EmadeDataPair
from GPFramework.data import EmadeDataPairNN, EmadeDataPairNNF # data pair that's only modified by Neural Network primitives
from GPFramework import general_methods
from GPFramework.constants import TriState, Axis, TRI_STATE, AXIS
from importlib import import_module
import glob
import json

def set_regression(boolean):
    global regression
    regression = boolean

class LearnerType:
    def __init__(self, name, params):
        self.name = name
        self.params = params

    def __repr__(self):
        return 'LearnerType(\'' + str(self.name) + '\', ' + str(self.params) + ')'

class EnsembleType:
    def __init__(self, name, params):
        self.name = name
        self.params = params

    def __repr__(self):
        return 'EnsembleType(\'' + str(self.name) + '\', ' + str(self.params) + ')'

def registry_stuff(pset):
    # absolutely no idea what this does but I'm not gonna risk breaking things

    primitive_registries = [
        sm.smw, sm.smw_2, sm.smw_3, sm.smwb,
        sp.smw, sp.smw_2, sp.smw_4, sp.smwb,
        fem.few, fsm.fsw, cm.cmw,
        dm.dmw, opm.opw, opm.opw_2, opm.opw_3,
        dem.dew, dem.dew_2, dem.dewd, dem.dewb
    ]
    terminal_registry = []
    ephemeral_methods = {}

    # add submodules into dictionaries
    dirs = glob.glob('src/GPFramework/projects/**/*.json')
    for directory in dirs:
        with open(directory) as f:
            d = json.load(f)
        m = import_module("GPFramework.projects." + d["name"] + ".methods")
        t = import_module("GPFramework.projects." + d["name"] + ".terminals")
        e = import_module("GPFramework.projects." + d["name"] + ".ephemerals")
        for wrap in d["primitive_wrappers"]:
            primitive_registries.append(getattr(m, wrap))
        for wrap in d["terminal_wrappers"]:
            terminal_registry += getattr(t, wrap).get_registry()

        # add all registered ephemerals into the primitive set
        for c in d["ephemerals"]:
            em = getattr(e, c)()
            pset.addEphemeralConstant(*em.get_info())
            pset.context.update(em.get_context())
            for key in em.get_context():
                ephemeral_methods[key] = str(em)

    # add all registered primitives into the primitive set
    for registry in primitive_registries:
        registered_primitives = registry.get_registry()
        for name in registered_primitives:
            info = registered_primitives[name]
            pset.addPrimitive(info["function"],
                              registry.prependInputs + info["input_types"],
                              info["output_type"], name=name)
    return ephemeral_methods

def addADFPrimitives(pset, datatype, allow_reshape= False, regression = False):
    """Takes in a pset and adds primitives

    Args:
        pset: primitive set
    """
    
    ephemeral_methods = registry_stuff(pset)

    # Neural Network primitives

    if datatype == 'textdata' or allow_reshape:
        # Regular NN Layers
        pset.addPrimitive(nnm.DenseLayer, [nnm.LayerUnits, nnm.Activation, nnm.Normalization, nnm.LayerList3dim], nnm.LayerList3dim, name='DenseLayer3dim')
        pset.addPrimitive(nnm.LSTMLayer, [nnm.LayerUnits, nnm.Activation, bool,bool, nnm.Normalization, nnm.LayerList3dim], nnm.LayerList3dim, name='LSTMLayer')
        pset.addPrimitive(nnm.Conv1DLayer, [nnm.LayerUnits, nnm.Activation, nnm.ConvKernelSize, nnm.StrideSize, nnm.PaddingType, nnm.Normalization, nnm.LayerList3dim], nnm.LayerList3dim, name='Conv1DLayer')
        pset.addPrimitive(nnm.GRULayer, [int, nnm.Activation, int, bool, bool, nnm.Normalization, nnm.LayerList3dim], nnm.LayerList3dim, name='GRULayer')
        
        # Skip Connection Layers
        pset.addPrimitive(nnm.Conv1DConnection, [nnm.Activation, nnm.ConvKernelSize, nnm.StrideSize, nnm.PaddingType, nnm.Normalization], nnm.SkipConnection3dim, name="Conv1DConnection")
        pset.addPrimitive(nnm.DenseConnection, [nnm.Activation, nnm.Normalization], nnm.SkipConnection3dim, name="DenseConnection3dim")
        pset.addPrimitive(nnm.NoSkip, [], nnm.SkipConnection3dim, name="NoSkipConnection3dim")
        pset.addPrimitive(nnm.AddLayer, [nnm.Activation], nnm.SkipConnection3dim, name="SkipConnection3dim")

        # Pooling Layers
        pset.addPrimitive(nnm.MaxPoolingLayer1D, [nnm.PoolSize], nnm.PoolingLayer, name="MaxPool1D")
        pset.addPrimitive(nnm.AveragePoolingLayer1D, [nnm.PoolSize], nnm.PoolingLayer, name="AvgPool1D")

        # Module Primitive
        pset.addPrimitive(nnm.Module, [nnm.DataType, nnm.DataDims, nnm.LayerList3dim, nnm.SkipConnection3dim, nnm.PoolingLayer, nnm.DropoutLayer], nnm.LayerListM, name='Module')

    if datatype == 'imagedata': # working with image data
        # Regular NN Layers
        pset.addPrimitive(nnm.DenseLayer, [nnm.LayerUnits, nnm.Activation, nnm.Normalization, nnm.LayerList4dim], nnm.LayerList4dim, name='DenseLayer4dim')
        pset.addPrimitive(nnm.Conv2DLayer, [nnm.LayerUnits, nnm.Activation, nnm.ConvKernelSize, nnm.StrideSize, nnm.PaddingType, nnm.Normalization, nnm.LayerList4dim], nnm.LayerList4dim, name='Conv2DLayer')
        pset.addPrimitive(nnm.SeparableConv2DLayer, [nnm.LayerUnits, nnm.Activation, nnm.ConvKernelSize, nnm.StrideSize, nnm.PaddingType, nnm.Normalization, nnm.LayerList4dim], nnm.LayerList4dim, name='SeparableConv2DLayer')

        # Skip Connection Layers
        pset.addPrimitive(nnm.Conv2DConnection, [nnm.Activation, nnm.ConvKernelSize, nnm.StrideSize, nnm.PaddingType, nnm.Normalization], nnm.SkipConnection4dim, name="Conv2DConnection")
        pset.addPrimitive(nnm.DenseConnection, [nnm.Activation, nnm.Normalization], nnm.SkipConnection4dim, name="DenseConnection4dim")
        pset.addPrimitive(nnm.NoSkip, [], nnm.SkipConnection4dim, name="NoSkipConnection4dim")
        pset.addPrimitive(nnm.AddLayer, [nnm.Activation], nnm.SkipConnection4dim, name="SkipConnection4dim")

        # Pooling Layers
        pset.addPrimitive(nnm.MaxPoolingLayer2D, [nnm.PoolSize], nnm.PoolingLayer, name="MaxPool2D")
        pset.addPrimitive(nnm.AveragePoolingLayer2D, [nnm.PoolSize], nnm.PoolingLayer, name="AvgPool2D")

        # Module Primitive
        pset.addPrimitive(nnm.Module, [nnm.DataType, nnm.DataDims, nnm.LayerList4dim, nnm.SkipConnection4dim, nnm.PoolingLayer, nnm.DropoutLayer], nnm.LayerListM, name='Module')

    # TODO: get these working or delete them
    # pset.addPrimitive(nnm.ResBlock, [int, nnm.LayerList], nnm.LayerList, name='ResBlock')
    #pset.addPrimitive(nnm.ConcatenateLayer, [nnm.LayerList, nnm.LayerList], nnm.LayerList, name='ConcatenateLayer2')
    #pset.addPrimitive(nnm.ConcatenateLayer, [nnm.LayerList, nnm.LayerList, nnm.LayerList], nnm.LayerList, name='ConcatenateLayer3')
    #pset.addPrimitive(nnm.ConcatenateLayer, [nnm.LayerList, nnm.LayerList, nnm.LayerList, nnm.LayerList], nnm.LayerList, name='ConcatenateLayer4')
    #pset.addPrimitive(nnm.AttentionLayer, [nnm.LayerList, nnm.LayerList], nnm.LayerList, name= 'AttentionLayer')

    pset.addPrimitive(nnm.NoPooling, [], nnm.PoolingLayer, name="NoPooling")
    return ephemeral_methods

def addNNLearnerPrimitives(pset, datatype, ll, allow_reshape=False):

    # Add layers that flatten the output
    if datatype == 'textdata' or allow_reshape:
        pset.addPrimitive(nnm.GlobalMaxPoolingLayer1D, [nnm.LayerListM], nnm.LayerList2dim, name = 'GlobalMaxPoolingLayer1D')
        pset.addPrimitive(nnm.GlobalAveragePoolingLayer1D, [nnm.LayerListM], nnm.LayerList2dim, name = 'GlobalAveragePoolingLayer1D')

        # Add NNLearner Primitive
        pset.addPrimitive(nnm.NNLearner, [EmadeDataPairNN, nnm.LayerListFinal, nnm.Optimizer], EmadeDataPairNNF, name="NNLearner")

    elif datatype == 'imagedata': # working with image data
        pset.addPrimitive(nnm.GlobalMaxPoolingLayer2D, [nnm.LayerListM], nnm.LayerList2dim, name = 'GlobalMaxPoolingLayer2D')
        pset.addPrimitive(nnm.GlobalAveragePoolingLayer2D, [nnm.LayerListM], nnm.LayerList2dim, name = 'GlobalAveragePoolingLayer2D')

        # data augmentation primitive 
        pset.addPrimitive(nnm.ImageAugmentationMod, [nnm.DataDims, nnm.ImageFlip, nnm.ImageRotation, nnm.ImageZoom, nnm.ImageTranslation], nnm.ImageAugmentation, name="ImageAugmentation")

        # Add NNLearner Primitive
        pset.addPrimitive(nnm.NNLearner, [EmadeDataPairNN, nnm.LayerListFinal, nnm.Optimizer, nnm.ImageAugmentation], EmadeDataPairNNF, name="NNLearner")
    
    pset.addPrimitive(nnm.FlattenLayer, [nnm.LayerListM], nnm.LayerList2dim, name = 'FlattenLayer')

    # Optionally can add a fully connected layer after the flatten layer
    # We add a maximum of one of these layers by returning a different layerlist type
    pset.addPrimitive(nnm.DenseLayer, [nnm.LayerUnits, nnm.Activation, nnm.Normalization, nnm.LayerList2dim, nnm.DropoutLayer], nnm.LayerListFinal, name='DenseLayer')
    pset.addPrimitive(nnm.PassLayerList, [nnm.LayerList2dim], nnm.LayerListFinal, name='PassLayerList')

    # Input Layers are added here
    # Note: 
    #       1) Input layers MUST take nnm.ModList as its FINAL input and return a nnm.LayerListM.
    #       2) All their arguments also must be satisfiable by terminals (ARG1 is a ModList terminal).
    #       If 1 & 2 don't hold, valid NNs probably won't be generated correctly.
    pset.addPrimitive(nnm.InputLayer, [nnm.ModList], nnm.LayerListM, name="Input")
    if datatype == 'textdata':
        #pset.addPrimitive(nnm.EmbeddingLayer, [nnm.LayerUnits, nnm.WeightInitializer], nnm.LayerList3dim, name='EmbeddingLayer') # TODO: should functionally replace inputlayerterminal since it makes a layerlist -- change nnm code accordingly
        #pset.addPrimitive(nnm.PretrainedEmbeddingLayer, [nnm.PretrainedEmbedding], nnm.LayerList3dim, name='PretrainedEmbeddingLayer') # TODO: should functionally replace inputlayerterminal since it makes a layerlist -- change nnm code accordingly
        
        # Transfer Learning
        # pset.addPrimitive(nnm.BERTInputLayer, nnm.LayerList3dim, name='BERTInputLayer')
        pass
    elif datatype == 'imagedata' and allow_reshape:
        pset.addPrimitive(nnm.PatchEmbedding1DInputLayer, [nnm.LayerUnits, nnm.PatchSize, nnm.Activation, nnm.Normalization, nnm.DataDims, nnm.ModList], nnm.LayerListM, name="PatchEmbedding1DInputLayer")
    elif datatype == "imagedata":
        pset.addPrimitive(nnm.PatchEmbedding2DInputLayer, [nnm.LayerUnits, nnm.PatchSize, nnm.Activation, nnm.Normalization, nnm.DataDims, nnm.ModList], nnm.LayerListM, name="PatchEmbedding2DInputLayer")

        # Transfer Learning:
        # pset.addPrimitive(nnm.MobileNetInputLayer, nnm.LayerList4dim, name='MobileNetInputLayer')
        # pset.addPrimitive(nnm.VGGInputLayer, nnm.LayerList4dim, name='VGGInputLayer')
        # pset.addPrimitive(nnm.InceptionInputLayer, nnm.LayerList4dim, name='InceptionInputLayer')
    

def addTerminals(pset, datatype, data_dims, exclude_inputlayers=False):
    """Takes in a pset and adds terminals

    Args:
        pset: primitive set
    """

    # Terminals (Constants)
    pset.addTerminal(True, bool, name='trueBool')
    pset.addTerminal(False, bool, name='falseBool')

    # if exclude_inputlayers:
    pset.addTerminal(datatype, nnm.DataType, name='datatype')
    pset.addTerminal(data_dims, nnm.DataDims, name='data_dims')
        

    # Pretrained Embeddings
    pset.addTerminal(nnm.PretrainedEmbedding.GLOVE, nnm.PretrainedEmbedding, name='gloveWeights')
    pset.addTerminal(nnm.PretrainedEmbedding.FASTTEXT, nnm.PretrainedEmbedding, name='fasttextWeights')
    pset.addTerminal(nnm.PretrainedEmbedding.GLOVEFASTTEXT, nnm.PretrainedEmbedding, name='gloveFasttextWeights')
    pset.addTerminal(nnm.PretrainedEmbedding.GLOVETWITTER, nnm.PretrainedEmbedding, name='gloveTwitterWeights')

    # Layer Weight Initializers
    pset.addTerminal(nnm.WeightInitializer.GLOROTUNIFORM, nnm.WeightInitializer, name='glorotUniformWeights')
    pset.addTerminal(nnm.WeightInitializer.GLOROTNORMAL, nnm.WeightInitializer, name='glorotNormalWeights')
    pset.addTerminal(nnm.WeightInitializer.HE, nnm.WeightInitializer, name='heWeights')
    pset.addTerminal(nnm.WeightInitializer.RANDOMUNIFORM, nnm.WeightInitializer, name='randomUniformWeights')
    
    ### Only using Relu and Default to constrain search space
    pset.addTerminal(nnm.Activation.RELU, nnm.Activation, name='reluActivation')
    # pset.addTerminal(nnm.Activation.ELU, nnm.Activation, name='eluActivation')
    # pset.addTerminal(nnm.Activation.SELU, nnm.Activation, name='seluActivation')
    pset.addTerminal(nnm.Activation.GELU, nnm.Activation, name='geluActivation') # apparently has some regularization properties
    pset.addTerminal(nnm.Activation.LINEAR, nnm.Activation, name='linearActivation')
    # pset.addTerminal(nnm.Activation.SIGMOID, nnm.Activation, name='sigmoidActivation')
    # pset.addTerminal(nnm.Activation.SOFTMAX, nnm.Activation, name='softmaxActivation')
    # pset.addTerminal(nnm.Activation.TANH, nnm.Activation, name='tanhActivation')
    pset.addTerminal(nnm.Activation.DEFAULT, nnm.Activation, name='defaultActivation')

    ### for architecture search, sticking with adam seems to work best
    pset.addTerminal(nnm.Optimizer.ADAM, nnm.Optimizer, name='AdamOptimizer')
    # pset.addTerminal(nnm.Optimizer.SGD, nnm.Optimizer, name='SGDOptimizer')
    # pset.addTerminal(nnm.Optimizer.RMSPROP, nnm.Optimizer, name='RMSpropOptimizer')
    # pset.addTerminal(nnm.Optimizer.ADADELTA, nnm.Optimizer, name='AdadeltaOptimizer')
    # pset.addTerminal(nnm.Optimizer.ADAGRAD, nnm.Optimizer, name='AdagradOptimizer')
    # pset.addTerminal(nnm.Optimizer.ADAMAX, nnm.Optimizer, name='AdamaxOptimizer')
    # pset.addTerminal(nnm.Optimizer.NADAM, nnm.Optimizer, name='NadamOptimizer')
    # pset.addTerminal(nnm.Optimizer.FTRL, nnm.Optimizer, name='FtrlOptimizer')

    # set layer units for network layers and embedding layers
    min_layer_units=32
    max_layer_units = 256
    step_size = 32
    terminal_range(list(range(min_layer_units, max_layer_units+1, step_size)), pset, nnm.LayerUnits, name = "LayerUnit")
    # adding bigger out dims:
    terminal_range(list(range(512, 2048+1, 256)), pset, nnm.LayerUnits, name = "LayerUnit")

    pset.addTerminal(nnm.ConvKernelSize.kernSize1, nnm.ConvKernelSize, name='KernelSize1')
    pset.addTerminal(nnm.ConvKernelSize.kernSize3, nnm.ConvKernelSize, name='KernelSize3')
    pset.addTerminal(nnm.ConvKernelSize.kernSize5, nnm.ConvKernelSize, name='KernelSize5')

    pset.addTerminal(nnm.PoolSize.PoolSize2, nnm.PoolSize, "PoolSize2")
    # pset.addTerminal(nnm.PoolSize.PoolSize3, nnm.PoolSize, "PoolSize3")

    pset.addTerminal(nnm.PaddingType.VALID, nnm.PaddingType, name='ValidPadding')
    pset.addTerminal(nnm.PaddingType.SAME, nnm.PaddingType, name='SamePadding')

    pset.addTerminal(nnm.StrideSize.stride1, nnm.StrideSize, name="Stride1")
    pset.addTerminal(nnm.StrideSize.stride2, nnm.StrideSize, name="Stride2")

    pset.addTerminal(nnm.Normalization.NONE, nnm.Normalization, name="No_Normalization")
    pset.addTerminal(nnm.Normalization.BATCHNORM, nnm.Normalization, name="Batch_Normalization")
    pset.addTerminal(nnm.Normalization.LAYERNORM, nnm.Normalization, name="Layer_Normalization")

    pset.addTerminal(nnm.DropoutLayer.NONE, nnm.DropoutLayer, name="NoDropout")
    pset.addTerminal(nnm.DropoutLayer.Dropout20, nnm.DropoutLayer, name="Dropout20")

    pset.addTerminal(nnm.ImageFlip.NONE, nnm.ImageFlip, name="NoFlip")
    pset.addTerminal(nnm.ImageFlip.HFLIP, nnm.ImageFlip, name="FlipHorizontal")
    pset.addTerminal(nnm.ImageFlip.BOTH, nnm.ImageFlip, name="FlipHorizontalVertical")

    pset.addTerminal(nnm.ImageRotation.NONE, nnm.ImageRotation, name="NoRotation")
    pset.addTerminal(nnm.ImageRotation.ROTATE10, nnm.ImageRotation, name="Rotate10")
    pset.addTerminal(nnm.ImageRotation.ROTATE20, nnm.ImageRotation, name="Rotate20")

    pset.addTerminal(nnm.ImageZoom.NONE, nnm.ImageZoom, name="NoZoom")
    pset.addTerminal(nnm.ImageZoom.ZOOM10, nnm.ImageZoom, name="Zoom10")
    pset.addTerminal(nnm.ImageZoom.ZOOM20, nnm.ImageZoom, name="Zoom20")

    pset.addTerminal(nnm.ImageTranslation.NONE, nnm.ImageTranslation, name="NoTranslation")
    pset.addTerminal(nnm.ImageTranslation.TRANSLATE10, nnm.ImageTranslation, name="Translation10")
    pset.addTerminal(nnm.ImageTranslation.TRANSLATE20, nnm.ImageTranslation, name="Translation20")

    terminal_range(get_patch_sizes(data_dims[0]), pset, nnm.PatchSize, "PatchSize")

def terminal_range(sequence, pset, ret_type, name):
    # adds numeric terminals to pset that return ret_type and are named name+str(number)
    for num in sequence:
        pset.addTerminal(num, ret_type, name=name+str(num))

# create patch size terminals based on how big the images are
# The ViT paper had success with dividing images into 14x14 patches up to 32x32, 
# so we'll use patch sizes that give us anything in that range
def get_patch_sizes(imsize):
    # assumes images are square
    # imsize: pixel height of image
    patch_sizes = []
    for i in range(1,imsize//14+1):
        if imsize%i==0 and imsize//i>=14 and imsize//i <=32:
            patch_sizes.append(i)
    return patch_sizes