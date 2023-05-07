import GPFramework.learner_methods as lm
import GPFramework.signal_methods as sm
import GPFramework.spatial_methods as sp
import GPFramework.feature_extraction_methods as fem
import GPFramework.feature_selection_methods as fsm
import GPFramework.clustering_methods as cm
import GPFramework.decomposition_methods as dm
import GPFramework.operator_methods as opm
import GPFramework.detection_methods as dem

from GPFramework.UnitTests.learner_methods_unit_test import MethodsUnitTest
from GPFramework.UnitTests.signal_methods_unit_test import SignalUnitTest
from GPFramework.UnitTests.spatial_methods_unit_test import SpatialUnitTest
from GPFramework.UnitTests.feature_extraction_unit_test import FeatureExtractionUnitTest
from GPFramework.UnitTests.feature_selection_unit_test import FeatureSelectUnitTest
from GPFramework.UnitTests.clustering_methods_unit_test import ClusterUnitTest
from GPFramework.UnitTests.decomposition_methods_unit_test import DecompositionUnitTest
from GPFramework.UnitTests.operator_methods_unit_test import OperatorUnitTest
from GPFramework.UnitTests.detection_methods_unit_test import DetectionUnitTest

if __name__ == '__main__':
    primitive_registries = [
        (sm.smw, SignalUnitTest, "Signal_Methods_One"), (sm.smw_2, SignalUnitTest, "Signal_Methods_Two"), (sm.smwb, SignalUnitTest, "Signal_Methods_Base"),
        (sp.smw, SpatialUnitTest, "Spatial_Methods_One"), (sp.smw_2, SpatialUnitTest, "Spatial_Methods_Two"), (sp.smwb, SpatialUnitTest, "Spatial_Methods_Base"),
        (fem.few, FeatureExtractionUnitTest, "Feature_Extraction_Methods"), (fsm.fsw, FeatureSelectUnitTest, "Feature_Select_Methods"), 
        (cm.cmw, ClusterUnitTest, "Clustering_Methods"), (dm.dmw, DecompositionUnitTest, "Decomposition_Methods"), 
        (opm.opw, OperatorUnitTest, "Operator_Methods_One"), (opm.opw_2, OperatorUnitTest, "Operator_Methods_Two"),
        (dem.dew, DetectionUnitTest, "Detection_Methods_One"), (dem.dew_2, DetectionUnitTest, "Detection_Methods_Two"), 
        (dem.dewd, DetectionUnitTest, "Detection_Methods_Wrapper"), (dem.dewb, DetectionUnitTest, "Detection_Methods_Base")
    ]
    
    for registry, tester, name in primitive_registries:
        print("Validating:", name)
        if registry.validate_tests(tester):
            print("Registry Validated: {}".format(name))
        print("\n")