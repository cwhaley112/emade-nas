'''
Programmed by Austin Dunn and Joel Ye
Implements wrapper methods for other modules
'''
from GPFramework.cache_methods import check_cache_read, check_cache_write, hash_string
from GPFramework.constants import TriState, Axis, TRI_STATE, AXIS
from GPFramework.general_methods import flatten_obj_array
from GPFramework.data import EmadeDataPair
from abc import ABC, abstractmethod
from functools import partial
import numpy as np
import copy as cp
import traceback
import time
import sys
import gc
import os

class TerminalWrapper:
    """
    Stores a mapping of terminals used in generating the PrimitiveSet

    Args:
        type: common output type for everything stored in the TerminalWrapper

    """
    def __init__(self, output_type=None):
        self.output_type = output_type
        self.registry = []

    def register(self, name, value, output_type=None):
        if not isinstance(name, str):
            raise TypeError("terminal name must be a string")

        # assign output_type if a common output_type is not defined
        out = self.output_type
        if out is None:
            if output_type is not None:
                out = output_type
            else:
                raise Exception("Output type of terminal cannot be None.")

        # add tuple containing relevant terminal information
        self.registry.append((value, out, name))

    def get_registry(self):
        return self.registry

    # adds another registry list to the existing one
    # type-safe
    def add(self, other):
        counter = 0
        for i in other:
            if len(i) == 3:
                counter += 1
        if len(counter) == len(other):
            self.registry += other
        else:
            raise Exception("input must be a list with format [(value, output_type, name)...]")

class EphemeralWrapper:
    """
    Stores a mapping of ephemeral constants used in generating the PrimitiveSet

    Args:
        type: common output type for everything stored in the EphemeralWrapper

    """
    def __init__(self, output_type=None):
        self.output_type = output_type
        self.registry = []

    def register(self, name, method, method_list, output_type=None):
        if not isinstance(name, str):
            raise TypeError("ephemeral name must be a string")

        # assign output_type if a common output_type is not defined
        out = self.output_type
        if out is None:
            if output_type is not None:
                out = output_type
            else:
                raise Exception("Output type of terminal cannot be None.")

        # add tuple containing relevant terminal information
        self.registry.append((name, method, out, method_list))

    def get_registry(self):
        return self.registry

    # adds another registry list to the existing one
    # type-safe
    def add(self, other):
        counter = 0
        for i in other:
            if len(i) == 4:
                counter += 1
        if len(counter) == len(other):
            self.registry += other
        else:
            raise Exception("input must be a list with format [(name, method, output_type)...]")

class EphemeralConstant(ABC):
    """
    Stores a mapping of ephemeral constants used in generating the PrimitiveSet

    Args:
        generator: generator method used to generate EphemeralConstant
        context:   dictionary mapping strings to method references
        type:      common type for everything stored in the EphemeralWrapper

    """
    def __init__(self, generator, context, my_type):
        self.generator = generator
        self.context = context
        self.my_type = my_type

    def get_generator(self):
        return self.generator

    def get_context(self):
        return self.context

    def get_type(self):
        return self.my_type

    def get_info(self):
        return self.__str__(), self.generator, self.my_type

    @abstractmethod
    def __str__(self):
        pass

class RegistryWrapper(ABC):
    """
    Abstract Base Class cannot be Instantiated

    Stores a mapping of primitives used in generating the PrimitiveSet
    Can be subclassed for different types of primitives

    The first object of input_types must be a EmadeDataPair

    Args:
        input_types (list<types>): common inputs for every primitive stored in the wrapper
                                   used to create a mapping between arg and index
                                   example mapping created: {'EmadeDataPair0': 0, 'TriState1': 1}

    """
    def __init__(self, input_types=[]):
        self.prependInputs = input_types
        self.kwpos = {}
        for i in range(len(input_types)):
            self.kwpos[input_types[i].__name__ + str(i)] = i
        self.registry = {}

    @abstractmethod
    def register(self, name, test_name, p_fn, s_fn, input_types, output_type=EmadeDataPair):
        """
        Registers a Primitive
        
        Args:
            name (String):             unique name of the primitive
            test_name (String):        name of the unit test of the primitive
            p_fn (function pointer):   main helper function of the primitive (augments data)
            s_fn (function pointer):   setup function of the primitive, which only runs once
            input_types (list<types>): input types of the primitive
            output_type (type):        return type of the primitive
        """
        pass

    def get_registry(self):
        return self.registry

    def add(self, other):
        """
        Adds another registry dictionary to the existing one
        type-safe
        
        Args:
            other (RegistryWrapper): other registry to combine with self.registry
        """
        try:
            for key in other:
                other[key]["function"]
        except KeyError:
            raise Exception("input must be a registry format dictionary")

        self.registry.update(other)

    def run_test(self, primitive, test_class):
        """
        Runs a specific unit test
        
        Args:
            primitive (String): name of primitive (key for self.registry)
            test_class (class): unit test class for a module (i.e. spatial_methods)
        """
        # Setup unit test
        test = self.registry[primitive]["unit_test"]
        tester = test_class()
        tester.setUp()

        # Run unit test
        getattr(tester, test)()
        print("Unit Test {} Succeeded.".format(test))
  
    def has_test(self, test, test_class):
        """
        Returns True if the unit test exists otherwise False
        
        Args:
            test (String):      name of unit test
            test_class (class): unit test class for a module (i.e. spatial_methods)
        """
        return hasattr(test_class, test)

    def validate_tests(self, test_class):
        """
        Returns True if all unit tests exist otherwise False
        
        Args:
            test (String):      name of unit test
            test_class (class): unit test class for a module (i.e. spatial_methods)
        """
        return_value = True
        for primitive in self.registry:
            if not self.has_test(self.registry[primitive]["unit_test"], test_class):
                print("Unit Test for {} Does Not Exist".format(primitive))
                return_value = False
            else:
                print("Validation Successful for {}".format(primitive))
        return return_value


class RegistryWrapperS(RegistryWrapper):
    """
    This wrapper is a standard registry wrapper for primitives
    Used by signal_methods, spatial_methods, operator_methods, and feature_extraction_methods

    Stores a mapping of primitives used in generating the PrimitiveSet

    The first object of input_types must be a EmadeDataPair

    Args:
        input_types: common inputs for every primitive stored in the wrapper
                     used to create a mapping between arg and index
                     example mapping created: {'EmadeDataPair0': 0, 'TriState1': 1}

    """
    def __init__(self, input_types=[]):
        super().__init__(input_types)

    def register(self, name, test_name, p_fn, s_fn, input_types, modes=TRI_STATE,
                 output_type=EmadeDataPair, prep_fn=None, e_fn=None):
        """
        Registers a Primitive
        
        Args:
            name (String):              unique name of the primitive
            test_name (String):         name of the unit test of the primitive
            p_fn (function pointer):    main helper function of the primitive (augments data)
            s_fn (function pointer):    setup function of the primitive, which only runs once
            input_types (list<types>):  input types of the primitive
            modes (list<Enum>):         list of valid modes
            output_type (type):         return type of the primitive
            prep_fn (function pointer): preparation function to transform data before every primitive
            e_fn (function pointer):    exit function to transform data after every primitive
        """
        # create wrapped method
        wrapped_fn = partial(primitive_wrapper, name, p_fn, s_fn,
                             self.kwpos, modes, prep_fn, e_fn)
  
        # wrapped unit test
        if isinstance(test_name, str):
            unit_test = test_name
        else:
            raise ValueError("test_name must be a string (str)")

        # create a mapping for adding primitive to pset
        self.registry[name] = {"function": wrapped_fn,
                               "input_types": input_types,
                               "supported_modes": modes,
                               "output_type": output_type,
                               "unit_test": unit_test}

        # return wrapped method
        return wrapped_fn

class RegistryWrapperFT(RegistryWrapper):
    """
    This wrapper is specific to methods which fit and transform data
    but do not modify the target

    Stores a mapping of primitives used in generating the PrimitiveSet

    The first object of input_types must be a EmadeDataPair

    Args:
        input_types: common inputs for every primitive stored in the wrapper
                     used to create a mapping between arg and index
                     example mapping created: {'EmadeDataPair0': 0, 'TriState1': 1}

    """
    def __init__(self, input_types=[]):
        super().__init__(input_types)

    def register(self, name, test_name, p_fn, s_fn, input_types, output_type=EmadeDataPair):
        # create wrapped method
        wrapped_fn = partial(fit_transform_wrapper, name, p_fn, s_fn)

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

class RegistryWrapperCM(RegistryWrapper):
    """
    This wrapper is specific to clustering_methods

    Stores a mapping of primitives used in generating the PrimitiveSet

    The first object of input_types must be a EmadeDataPair

    Args:
        input_types: common inputs for every primitive stored in the wrapper
                     used to create a mapping between arg and index
                     example mapping created: {'EmadeDataPair0': 0, 'TriState1': 1}

    """
    def __init__(self, input_types=[]):
        super().__init__(input_types)

    def register(self, name, test_name, p_fn, s_fn, input_types, output_type=EmadeDataPair):
        # create wrapped method
        wrapped_fn = partial(clustering_wrapper, name, p_fn)

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

class RegistryWrapperB(RegistryWrapper):
    """
    This is a base registry wrapper for use by any abnormal primitives
    Typically these methods do not use a primitive wrapper

    Stores a mapping of primitives used in generating the PrimitiveSet

    The first object of input_types must be a EmadeDataPair

    Args:
        input_types: common inputs for every primitive stored in the wrapper
                     used to create a mapping between arg and index
                     example mapping created: {'EmadeDataPair0': 0, 'TriState1': 1}

    """
    def __init__(self, input_types=[]):
        super().__init__(input_types)

    def register(self, name, test_name, p_fn, s_fn, input_types, output_type=EmadeDataPair):
        # wrapped unit test
        if isinstance(test_name, str):
            unit_test = test_name
        else:
            raise ValueError("test_name must be a string (str)")

        # create a mapping for adding primitive to pset
        self.registry[name] = {"function": p_fn,
                               "input_types": input_types,
                               "output_type": output_type,
                               "unit_test": unit_test}

        # return wrapped method
        return p_fn

class RegistryWrapperSP(RegistryWrapper):
    """
    NO LONGER USED. THIS IS AN EXAMPLE FOR PREP + EXIT METHODS

    This wrapper is specific to spatial_methods

    Stores a mapping of primitives used in generating the PrimitiveSet

    The first object of input_types must be a EmadeDataPair

    Args:
        input_types: common inputs for every primitive stored in the wrapper
                     used to create a mapping between arg and index
                     example mapping created: {'EmadeDataPair0': 0, 'TriState1': 1}

    """
    def __init__(self, input_types=[]):
        super().__init__(input_types)

    def register(self, name, test_name, p_fn, s_fn, input_types, modes=TRI_STATE,
                 output_type=EmadeDataPair, prep_fn=None, e_fn=None):

        # setup preparation and exit methods
        prep_fn = partial(spatial_prep_wrapper, prep_fn)
        e_fn = partial(spatial_exit_wrapper, e_fn)

        # create wrapped method
        wrapped_fn = partial(primitive_wrapper, name, p_fn, s_fn,
                             self.kwpos, modes, prep_fn, e_fn)

        # wrapped unit test
        if isinstance(test_name, str):
            unit_test = test_name
        else:
            raise ValueError("test_name must be a string (str)")

        # create a mapping for adding primitive to pset
        self.registry[name] = {"function": wrapped_fn,
                               "input_types": input_types,
                               "supported_modes": modes,
                               "output_type": output_type,
                               "unit_test": unit_test}

        # return wrapped method
        return wrapped_fn

def spatial_prep_wrapper(p_fn, data):
    '''
    Formats data into channel format expected by primitives

    Args:
        p_fn:    prep method given by primitive
        data:    instance of data set
        s_data:  instance of a second data set
    '''
    result = []
    for d in data:
        if len(d.shape) == 3:
            # reshape from (k, m, n) to (m, n, k)
            result.append(np.transpose(d, (1, 2, 0)))
        else:
            result.append(d)

    if p_fn is not None:
        return p_fn(*result)

    return result

def spatial_exit_wrapper(e_fn, data):
    '''
    Reverses original reshape of data

    Args:
        e_fn:    exit method given by primitive
        data:    instance of data set
        s_data:  instance of a second data set
    '''
    result = []
    for d in data:
        if len(d.shape) == 3:
            # reshape from (m, n, k) to (k, m, n)
            result.append(np.transpose(d, (2, 0, 1)))
        else:
            result.append(d)

    if e_fn is not None:
        return e_fn(*result)

    return result

"""
Modified by Gabriel Wang
"""
def primitive_wrapper(p_name, primitive_f, setup_f, kwpos, supported_modes=TRI_STATE, prep_f=None, exit_f=None, *args, **kwargs):
    """Wrapper method for handling data in primitives

        kwpos provides *args indices of data_pairs, mode (their indices in the input type of the primitive). Note: Methods in sm that don't use mode generally assume feature to feature transform. Any utility objects e.g. convolution kernels should be initialized in f_setup, or passed in as kwargs
        Supports primitives that operate on multiple data pairs. Abstracts data traversal, uses function on inner data.
        Checklist for users:
        - Write f_setup - will receive first instance of data, and primitive args.
            - If defined, return named arguments that f will use
        - Write f - f is given data of an instance, and named arguments in f_setup
            - If f_setup not defined, f will receive data and primitive args
        - Docstring must be defined using __doc__
        - Primitives that use a setup method must be called without kwargs in unit tests

    Args:
        p_name:          string name of the primitive
        primitive_f:     method to call on data
        setup_f:         optional setup method
        kwpos:           maps location of datapairs and mode in args
        supported_modes: list of valid modes for this primitive
        prep_f:          formats data before primitive method call
        exit_f:          formats data after primitive method call
        args:            primitive method arguments (list type)
        kwargs:          primitive method keyword arguments (dict type)

    Returns:
        updated data pair with new data
    """
    # For debugging purposes let's print out method name
    print(p_name) ; sys.stdout.flush()

    """
    Validation Block
    """
    # parse args
    data_pairs = []
    modes = []
    axes = []
    for key in kwpos:
        if 'EmadeDataPair' in key:
            data_pairs.append(cp.deepcopy(args[kwpos[key]]))
        elif 'TriState' in key:
            modes.append(cp.deepcopy(args[kwpos[key]]))
        elif 'Axis' in key:
            axes.append(cp.deepcopy(args[kwpos[key]]))

    if len(data_pairs) == 0:
        raise NotImplementedError("primitive_wrapper does not support primitives that don't use EmadeDataPair")

    # we should not have to check both supported_modes and TRI_STATE if they are equal.
    for mode in modes:
        if mode not in supported_modes or mode not in TRI_STATE:
            raise ValueError('Unsupported TriState {} provided to {}'.format(mode, p_name))

    for axis in axes:
        if axis not in AXIS:
            raise ValueError('Unsupported Axis {} provided to {}'.format(axis, p_name))

    crop_args = len(kwpos.keys())
    # Fragile checking - note, if items are passed in as kwargs (i.e. in unit tests), we fail. Sol: no kwargs in unit test
    if max(kwpos.values()) >= crop_args:
        raise ValueError("Inappropriate indices registered in kwpos")
    if len(args) < crop_args:
        raise ValueError("{} expected at least {} args but got {} args".format(p_name, crop_args, len(args)))
    for data_pair in data_pairs:
        if not isinstance(data_pair, EmadeDataPair):
            raise ValueError("{} expected EmadeDataPair for data_pair argument but got {}".format(p_name, data_pair))

    data_pair = data_pairs[0]
    args = args[crop_args:]
    # ^ the decision to not let inner functions see data pair, mode, explicitly

    # Note - if setup_f is None, we forward all args passed to primitive to inner func
    # The choice is made to withhold primitive args if setup_f is provided, setup_f must label them as kwargs
    # This makes the data processing signature as concise as possible
    is_setup = False
    helper_kwargs = {} # Set up reference

    def setup_wrap(data):
        nonlocal helper_kwargs, args, is_setup
        if setup_f is not None:
            *data, new_kwargs = setup_f(*data, *args)
            helper_kwargs = {**kwargs, **new_kwargs}
            args = ()
            is_setup = True
        return data

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
            previous_hash = ""
            for i in range(len(data_pairs)):
                previous_hash += data_pairs[i].get_hash() + "_"
            previous_hash = hash_string(previous_hash[:-1])

            # Calculate unique method string (unique method name + arguments of the method)
            mode_axis_string = ""
            for mode in modes:
                mode_axis_string += str(mode) + "_"
            for axis in axes:
                mode_axis_string += str(axis) + "_"

            method_string = p_name + "_" + str(args).strip('[]').strip('()').replace(",", "_").replace(" ", "") + "_" + mode_axis_string

            # Combine the unique method name + arguments of the method + hash of the previous data
            # To form a unique key of the method call
            method_key = method_string + previous_hash

            overhead_time, method_row, cache_row, hit = check_cache_read(data_pair, database, method_key, oh_time_start)
            if hit: return data_pair

            eval_time_start = time.time()

        data_list = []
        # data_set_list is a list of lists containing train and test data
        data_set_list = []
        for dataset in data_pairs:
            data_set_list.append([dataset.get_train_data(), dataset.get_test_data()])
        # train_test_section is a list of all train or test datasets
        for train_test_section in zip(*data_set_list):
            # i_list is a list of dataset instances
            i_list = []
            for dataset in train_test_section:
                i_list.append(cp.deepcopy(dataset.get_instances()))

            for instances in zip(*i_list):
                """
                Load Block
                """
                data = []
                for instance, mode in zip(instances, modes):
                    if mode is TriState.FEATURES_TO_FEATURES:
                        data.append(instance.get_features().get_data())
                    elif mode is TriState.STREAM_TO_STREAM or TriState.STREAM_TO_FEATURES:
                        data.append(instance.get_stream().get_data())

                # Check to make sure no datasets are empty
                not_empty = True
                for i in data:
                    if len(i) == 0:
                        not_empty = False
                        break
                """
                Call Block
                """

                if not_empty:
                    default = True
                    if not is_setup:
                        data = setup_wrap(data)
                    output_args = []
                    for data_given, mode, axis in zip(data, modes, axes):
                        if mode is TriState.STREAM_TO_STREAM or TriState.STREAM_TO_FEATURES:
                            if axis is not Axis.FULL:
                                default = False
                                indexing_arr = [slice(0,dim) for dim in data_given.shape]
                                j = lambda x,y,z:x[:y]+[z]+x[y+1:]
                                output_args_section = []
                                for i in range(data_given.shape[axis.value]):
                                    output_args_section.append(data_given[tuple((j)(indexing_arr, axis.value, i))])
                                output_args.append(output_args_section)

                                if len(data) == len(output_args):
                                    output = [primitive_f(*i, *args, **helper_kwargs) for i in zip(*output_args)]
                                    try:
                                        data = np.stack(output, axis=axes[0].value)
                                    except ValueError:
                                        data = np.array(output)

                    if default:
                        if prep_f is not None:
                            data = prep_f(data)

                        data = np.array(primitive_f(*data, *args, **helper_kwargs))

                        if exit_f is not None:
                            data = exit_f(data)

                    """
                    Store Block
                    """
                    for instance, mode in zip(instances, modes):
                        if mode is TriState.FEATURES_TO_FEATURES:
                            instance.get_features().set_data(data)
                        elif mode is TriState.STREAM_TO_STREAM:
                            instance.get_stream().set_data(data)
                        elif mode is TriState.STREAM_TO_FEATURES:
                            if data.dtype == object:
                                data = flatten_obj_array(len(data.shape), data)
                            old_features = instance.get_features().get_data()
                            new_features = np.concatenate((old_features, data), axis=None) # auto-flattening
                            instance.get_features().set_data(np.reshape(new_features, (1,-1)))
            train_test_section[0].set_instances(i_list[0])
            data_list.append(train_test_section[0])
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


"""
Written by Austin Dunn
"""
def fit_transform_wrapper(p_name, helper_function, setup_function, data_pair, mode, *args):
	"""Template for primitives which fit and transform data
	   but do not modify the target

	Args:
		p_name:           name of primitive
		helper_function:  returns transformed data
		setup_function:   returns method to fit to data
		data_pair:        given data pair
		mode:             mode to load and save data in
		args:             list of arguments

	Returns:
		modified data pair
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
			method_string = p_name + "_" + str(args).strip('[]').strip('()').replace(",", "_").replace(" ", "") + "_" + str(mode) + "_"

			# Combine the unique method name + arguments of the method + hash of the previous data
			# To form a unique key of the method call
			method_key = method_string + previous_hash

			overhead_time, method_row, cache_row, hit = check_cache_read(data_pair, database, method_key, oh_time_start)
			if hit: return data_pair

			eval_time_start = time.time()

		train_instances = cp.deepcopy(data_pair.get_train_data().get_instances())
		test_instances = cp.deepcopy(data_pair.get_test_data().get_instances())

		'''
		Load
		'''
		if mode is TriState.FEATURES_TO_FEATURES:
			train_data = np.vstack([instance.get_features().get_data() for instance in train_instances])
			# target_values = np.array([instance.get_target() for instance in train_instances])
			target_values = data_pair.get_train_data().get_target()
			test_data = np.vstack([instance.get_features().get_data() for instance in test_instances])
		elif mode is TriState.STREAM_TO_STREAM or TriState.STREAM_TO_FEATURES:
			train_data = np.vstack([instance.get_stream().get_data().flatten() for instance in train_instances])
			# target_values = np.array([instance.get_target() for instance in train_instances])
			target_values = data_pair.get_train_data().get_target()
			test_data = np.vstack([instance.get_stream().get_data().flatten() for instance in test_instances])

		'''
		Transform
		'''
		method = setup_function(*args)
		new_train_data, new_test_data = helper_function(train_data,
														test_data,
														target_values,
														method)

		'''
		Save
		'''
		instance_list = []
		for old_dataset, new_dataset in zip([train_instances, test_instances], [new_train_data, new_test_data]):
			for old_instance, new_instance in zip(old_dataset, new_dataset):
				if mode is TriState.FEATURES_TO_FEATURES or TriState.STREAM_TO_FEATURES:
					old_instance.get_features().set_data(new_instance)
				elif mode is TriState.STREAM_TO_STREAM:
					old_instance.get_stream().set_data(new_instance)
			instance_list.append(old_dataset)

		# create data_list to match rest of template
		data_pair.get_train_data().set_instances(instance_list[0])
		data_pair.get_test_data().set_instances(instance_list[1])
		data_list = [data_pair.get_train_data(),
					 data_pair.get_test_data()]

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

def clustering_wrapper(p_name, function, data_pair, *args):
    """Applies the given sklearn clusterer to the given EMADE data pair

    Args:
        p_name:      name of the method
        function:    clustering method
        data_pair:   EMADE data pair to cluster
        args:        list of arguments

    Returns:
        EMADE data object with cluster classes appended to stream data
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
            method_string = p_name + "_" + str(args).strip('[]').strip('()').replace(",", "_").replace(" ", "") + "_"

            # Combine the unique method name + arguments of the method + hash of the previous data
            # To form a unique key of the method call
            method_key = method_string + previous_hash

            overhead_time, method_row, cache_row, hit = check_cache_read(data_pair, database, method_key, oh_time_start)
            if hit: return data_pair

            eval_time_start = time.time()

        clusterer = function(*args)
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                data = instance.get_stream().get_data()

                clusterMatrix = np.transpose(data)
                resultVector = clusterer.fit_predict(clusterMatrix)
                instance.get_stream().set_data(np.append(data, [resultVector], axis=0))

            # Build a new data set from the new instances
            data_set.set_instances(instances)
            # Append to the list so we can build the pair
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
