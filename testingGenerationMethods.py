from deap import gp
import sys
sys.path.insert(1, '/home/cameron/Desktop/emade/src')
import GPFramework.gp_framework_helper as gpFrameworkHelper
from GPFramework import neural_network_methods as nnm
from GPFramework.neural_network_methods import compile_layerlist, genNNLearner, genADF
from GPFramework.data import EmadeDataPairNN, EmadeDataPairNNF
from deap import creator
import dill

gp.genNNLearner = genNNLearner
gp.genADF = genADF

class NNLearnerGen():
    def __init__(self) -> None:
        super().__init__()
        self.modGen = ModuleGen()
        self.data_dims  = (32, 32, 1)
        self.mods = 5
        self.glob_mods = {}
        # Change if needed
        self.datatype = 'imagedata'
        if self.datatype=='imagedata':
            ll = nnm.LayerList4dim
        elif self.datatype=='textdata':
            ll = nnm.LayerList3dim

        # Initialize MAIN overarching primitive
        self.pset = gp.PrimitiveSetTyped('MAIN', [EmadeDataPairNN], EmadeDataPairNNF)

        # Add primitives and terminals specific to NNLearner
        gpFrameworkHelper.addNNLearnerPrimitives(self.pset, self.datatype, ll)
        gpFrameworkHelper.addTerminals(self.pset, self.datatype, self.data_dims)

        with open("Global_MODS", "wb") as mod_file:
            for i in range(self.mods):
                # Create a name for the mod, or automatically defined functions
                mod_name = 'mod_' + str(i + 1)

                # Reshape value is the difference to original data_dim if mod is applied. Used in genNNLearner method
                expr, reshape = gp.genADF(self.modGen.mod_pset, min_=3, max_=5, data_dim=self.data_dims, data_type=self.datatype, type_=self.modGen.mod_pset.ret)
                tree = gp.PrimitiveTree(expr)
                tree = [tree]
                mod_obj = creator.ADF(tree)
                mod_obj.mod_num = i + 1
                mod_obj.age = 0
                mod_obj.num_occur=0
                mod_obj.name = mod_name
                mod_obj.reshape = reshape
                func = gp.compile(tree[0], self.modGen.mod_pset)
                print(mod_name, tree.__str__())
                
                self.glob_mods[mod_name] = [mod_name, mod_obj, list(reshape)]
                self.pset.addPrimitive(func, [nnm.LayerListM], nnm.LayerListM, name=mod_name)
                self.pset.context.update({mod_name: func})
            dill.dump(self.glob_mods, mod_file)

    def testPruningNNLearner(self):
        expr = gp.genNNLearner(self.pset, min_= 3, max_= 5, data_dim = self.data_dims, data_type = self.datatype, type_ = self.pset.ret)
        tree = gp.PrimitiveTree(expr)
        tree = [tree]
        func = gp.compile(tree[0], self.pset)
        #had = func(nnm.InputLayer())

class ModuleGen():
    def __init__(self) -> None:
        super().__init__()
        self.data_dims  = (32, 32, 1)
        self.mod_pset = gp.PrimitiveSetTyped('mod', [nnm.LayerList4dim], nnm.LayerListM)
        creator.create("ADF", list, reshape=[],
            pset=self.mod_pset, age=0, num_occur=0, retry_time=0, novelties = None,
            mod_num=None)
        gpFrameworkHelper.addADFPrimitives(self.mod_pset, 'imagedata')
        gpFrameworkHelper.addTerminals(self.mod_pset, 'imagedata', self.data_dims , exclude_inputlayers=True)

    # Test whether Modules can be generated and nested within each other without error
    def testNestedModule(self):
        expr, reshape = gp.genADF(self.mod_pset, min_=3, max_=5, data_dim=self.data_dims , data_type='imagedata', type_=self.mod_pset.ret)
        expr_2, reshape_2 = gp.genADF(self.mod_pset, min_=3, max_=5, data_dim=self.data_dims , data_type='imagedata', type_=self.mod_pset.ret)
        
        tree = gp.PrimitiveTree(expr)
        tree = [tree]
        func = gp.compile(tree[0], self.mod_pset)

        tree_2 = gp.PrimitiveTree(expr_2)
        tree_2 = [tree]
        func_2 = gp.compile(tree[0], self.mod_pset)

        had = func(nnm.InputLayer())
        had = func_2(had)
        assert len(had.mylist) == 3

        layer, _ = compile_layerlist(had, [], (32, 32, 1), 'imagedata', isNNLearner=True)
        print("Success")

# NNLearnerGen().testPruningNNLearner()
ModuleGen().testNestedModule()