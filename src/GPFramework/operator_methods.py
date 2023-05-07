"""
Programmed by Jason Zutty and Austin Dunn
Implements a number of operator methods for use with deap
"""
from GPFramework.constants import TriState, Axis, TRI_STATE
from GPFramework.wrapper_methods import RegistryWrapperS
from GPFramework.data import EmadeDataPair

import numpy as np

opw = RegistryWrapperS([EmadeDataPair, TriState, Axis])
opw_2 = RegistryWrapperS(2*[EmadeDataPair] + 2*[TriState] + 2*[Axis])
opw_3 = RegistryWrapperS(3*[EmadeDataPair] + 3*[TriState] + 3*[Axis])

def my_add_helper(data, value):
    return data + value

my_add_pair = opw_2.register("EmadeDataAddPair", "test_my_add_pair", my_add_helper, None, [], TRI_STATE)
my_add_pair.__doc__ = """
Adds two data pairs together

Args:
    data:  numpy array of example
    value: numpy array of second data pair
"""

def my_add_helper_triple(data, value, value2):
    return data + value + value2

my_add_pair_triple = opw_3.register("EmadeDataAddPairTriple", "test_my_add_pair_triple", my_add_helper_triple, None, [], TRI_STATE)
my_add_pair_triple.__doc__ = """
Adds three data pairs together

Args:
    data:  numpy array of example
    value: numpy array of second data pair
    value2: numpy array of third data pair
"""

def fraction_setup(data, value):
    if value != 0:
        value = 1 / value
    else:
        value = np.inf
    return data, {'value':value }

def fraction_helper(data, value=2):
    return data * value

fraction = opw.register("Fraction", "test_fraction", fraction_helper, fraction_setup, [int], TRI_STATE)
fraction.__doc__ = """
Multiplies every element of a numpy array by a fraction

Args:
    data: numpy array
    value: fraction denominator
"""

def triple_fraction_setup(data1, data2, data3, value):
    if value != 0:
        value = 1 / value
    else:
        value = np.inf
    return data1, data2, data3, {'value':value }

def my_fraction_triple(data1, data2, data3, value=2):
    return data1 * value + data2 * value + data3 * value

my_frac_triple = opw_3.register("FractionTriple", "test_my_frac_triple", my_fraction_triple, triple_fraction_setup, [int], TRI_STATE)
my_frac_triple.__doc__ = """
Adds three data pairs together after dividing them all by a value (linear combination)

Args:
    data1: numpy array of first data pair
    data2: numpy array of second data pair
    data3: numpy array of third data pair
    value: fraction denominator
"""

def double_fraction_setup(data1, data2, value):
    if value != 0:
        value = 1 / value
    else:
        value = np.inf
    return data1, data2, {'value':value }

def my_fraction_double(data1, data2, value=2):
    return data1 * value + data2 * value

my_frac_double = opw_2.register("FractionDouble", "test_my_frac_double", my_fraction_double, double_fraction_setup, [int], TRI_STATE)
my_frac_double.__doc__ = """
Adds three data pairs together after dividing them all by a value (linear combination)

Args:
    data1: numpy array of first data pair
    data2: numpy array of second data pair
    value: fraction denominator
"""

my_add = opw.register("EmadeDataAddInt", "test_my_add", my_add_helper, None, [int], TRI_STATE)
my_add.__doc__ = """
Adds an integer to each instance of a data pair

Args:
    data:  numpy array of example
    value: integer to add
"""

my_add_float = opw.register("EmadeDataAddFloat", "test_my_add_float", my_add_helper, None, [float], TRI_STATE)
my_add_float.__doc__ = """
Adds a float to each instance of a data pair

Args:
    data:  numpy array of example
    value: float to add
"""

def my_subtract_helper(data, value):
    return data - value

my_subtract_pair = opw_2.register("EmadeDataSubtractPair", "test_my_subtract_pair", my_subtract_helper, None, [], TRI_STATE)
my_subtract_pair.__doc__ = """
Subtracts two data pairs together

Args:
    data:  numpy array of example
    value: numpy array of second data pair
"""

my_subtract = opw.register("EmadeDataSubtractInt", "test_my_subtract", my_subtract_helper, None, [int], TRI_STATE)
my_subtract.__doc__ = """
Subtracts an integer to each instance of a data pair

Args:
    data:  numpy array of example
    value: integer to subtract
"""

my_subtract_float = opw.register("EmadeDataSubtractFloat", "test_my_subtract_float", my_subtract_helper, None, [float], TRI_STATE)
my_subtract_float.__doc__ = """
Subtracts a float to each instance of a data pair

Args:
    data:  numpy array of example
    value: float to subtract
"""

def my_divide_helper(data, value):
    data = data / value
    if not np.isfinite(data).all():
        data = np.nan_to_num(data)
    return data

def my_divide_int_helper(data, value):
    data = data // value
    if not np.isfinite(data).all():
        data = np.nan_to_num(data)
    return data

my_divide_pair = opw_2.register("EmadeDataDividePair", "test_my_divide_pair", my_divide_helper, None, [], TRI_STATE)
my_divide_pair.__doc__ = """
Divides two data pairs together

Args:
    data:  numpy array of example
    value: numpy array of second data pair
"""

my_divide = opw.register("EmadeDataDivideInt", "test_my_divide", my_divide_helper, None, [int], TRI_STATE)
my_divide.__doc__ = """
Divides an integer to each instance of a data pair

Args:
    data:  numpy array of example
    value: integer to divide
"""

my_divide_float = opw.register("EmadeDataDivideFloat", "test_my_divide_float", my_divide_helper, None, [float], TRI_STATE)
my_divide_float.__doc__ = """
Divides a float to each instance of a data pair

Args:
    data:  numpy array of example
    value: float to divide
"""

my_divide_int = opw.register("EmadeDataIntegerDivide", "test_my_divide_int", my_divide_int_helper, None, [float], TRI_STATE)
my_divide_int.__doc__ = """
Divides a float to each instance of a data pair using Integer Division

Args:
    data:  numpy array of example
    value: float to divide
"""

my_divide_int_pair = opw_2.register("EmadeDataIntegerDividePair", "test_my_divide_int_pair", my_divide_int_helper, None, [], TRI_STATE)
my_divide_int_pair.__doc__ = """
Divides a float to each instance of a data pair using Integer Division

Args:
    data:  numpy array of example
    value: numpy array of second data pair
"""

def my_multiply_helper(data, value):
    return data * value

my_multiply_pair = opw_2.register("EmadeDataMultiplyPair", "test_my_multiply_pair", my_multiply_helper, None, [], TRI_STATE)
my_multiply_pair.__doc__ = """
Multiplies two data pairs together

Args:
    data:  numpy array of example
    value: numpy array of second data pair
"""

my_multiply = opw.register("EmadeDataMultiplyInt", "test_my_multiply", my_multiply_helper, None, [int], TRI_STATE)
my_multiply.__doc__ = """
Multiplies an integer to each instance of a data pair

Args:
    data:  numpy array of example
    value: integer to multiply
"""

my_multiply_float = opw.register("EmadeDataMultiplyFloat", "test_my_multiply_float", my_multiply_helper, None, [float], TRI_STATE)
my_multiply_float.__doc__ = """
Multiplies a float to each instance of a data pair

Args:
    data:  numpy array of example
    value: float to multiply
"""

def my_np_multiply_helper(data, value):
    return np.multiply(data, value)

my_np_multiply_pair = opw_2.register("EmadeDataNumpyMultiplyPair", "test_my_np_multiply_pair", my_np_multiply_helper, None, [], TRI_STATE)
my_np_multiply_pair.__doc__ = """
Multiplies two data pairs together using numpy's operator

Args:
    data:  numpy array of example
    value: numpy array of second data pair
"""

my_np_multiply = opw.register("EmadeDataNumpyMultiplyInt", "test_my_np_multiply", my_np_multiply_helper, None, [int], TRI_STATE)
my_np_multiply.__doc__ = """
Multiplies an integer to each instance of a data pair using numpy's operator

Args:
    data:  numpy array of example
    value: integer to multiply
"""

my_np_multiply_float = opw.register("EmadeDataNumpyMultiplyFloat", "test_my_np_multiply_float", my_np_multiply_helper, None, [float], TRI_STATE)
my_np_multiply_float.__doc__ = """
Multiplies a float to each instance of a data pair using numpy's operator

Args:
    data:  numpy array of example
    value: float to multiply
"""
