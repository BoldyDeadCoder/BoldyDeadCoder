import math
import re
import sympy as sp
from fractions import Fraction
import pandas as pd  # If you plan to return pandas Series
import itertools
import json
import networkx as nx
import matplotlib.pyplot as plt
import logging
from multiprocessing import Pool, Manager
import os
import time
from tqdm import tqdm
from math import isclose
from itertools import permutations
from decimal import Decimal, getcontext, InvalidOperation
from cmath import sqrt
from typing import Union, Tuple
import numpy as np
from sympy import solve, limit, series, oo, SympifyError, exp, Function, sympify, Poly, latex, galois_group, simplify, \
    expand
from sympy import symbols, diff, integrate, div
from sympy.combinatorics import Permutation, PermutationGroup

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

e, x, y, j, b, a, c, n, d, Z, G, H, F, R, S, P, V, W, M, i = symbols('e, x, y, j, b, a, c, n, d, Z, G, H, F, R, S, P, '
                                                                     'V, W, M, i')


def safe_sympify(expr):
    """
    Safely converts a string to a Sympy expression, catching errors.
    """
    try:
        return safe_sympify(expr)
    except SympifyError:
        return None


def validate_limits(input_str):
    """
    Validates and parses limit input, returning numerical limits or None if invalid.
    """
    try:
        if input_str.strip().lower() == 'oo':
            return oo
        else:
            return float(input_str)
    except ValueError:
        return None


def group_operation(x, y, operation_type):
    operations = {
        "addition": lambda x, y: x + y,
        "multiplication": lambda x, y: x * y,
        "matrix_multiplication": lambda x, y: np.dot(x, y),
        "string_concatenation": lambda x, y: x + y,
    }

    operation = operations.get(operation_type)
    if operation:
        return operation(x, y)
    else:
        raise ValueError(f"Unsupported operation type: {operation_type}")


def list_all_generators(group, operation):
    """
    Lists all generators of the group.
    """
    # Input validation
    if not group:
        raise ValueError("Group cannot be empty.")
    if not callable(operation):
        raise ValueError("Operation must be a callable function.")

    # Memoization cache for the operation
    operation_cache = {}

    # Function to apply the operation with memoization
    def apply_operation(x, y):
        key = (x, y)
        if key not in operation_cache:
            operation_cache[key] = operation(x, y)
        return operation_cache[key]

    # Function to check if an element is a generator
    def is_generator(element):
        try:
            generated_elements = set()
            current_element = element
            for _ in range(len(group)):
                if current_element in generated_elements:
                    return False  # Early termination
                generated_elements.add(current_element)
                current_element = apply_operation(current_element, element)
            return len(generated_elements) == len(group)
        except Exception as e:
            logging.error(f"Error in is_generator for element {element}: {e}")
            return False

    # Parallel processing to compute generators
    try:
        with Pool() as pool:
            generators = [element for element, is_gen in zip(group, pool.map(is_generator, group)) if is_gen]
    except Exception as e:
        logging.error(f"Error in parallel processing: {e}")
        generators = []

    return generators


def parse_group_equation(equation, delimiter=',', valid_element_pattern=None):
    try:
        # Check for required characters
        if '=' not in equation or '{' not in equation or '}' not in equation:
            raise ValueError("Equation must contain '=', '{', and '}'.")

        # Extract and clean the group string
        group_str = equation.split('=')[1].strip().replace('{', '').replace('}', '')
        group = [element.strip() for element in group_str.split(delimiter)]

        # Validate each group element
        if valid_element_pattern is not None:
            pattern = re.compile(valid_element_pattern)
            for element in group:
                if not pattern.match(element):
                    raise ValueError(f"Invalid element format: {element}")

        return group
    except ValueError as e:
        logging.error(f"Error parsing group equation: {e}")
        raise


# Group Theory
def analyze_and_visualize_subgroups(group, operation, export_filename="/mnt/data/subgroups.json"):
    """
    Identifies all subgroups within the group, exports the subgroups to a JSON file,
    and visualizes the subgroups using networkx and matplotlib.
    """

    def is_subgroup(subset, group, operation):
        identity = None
        for g in group:
            if operation(g, g) == g:  # Simplistic check for identity
                identity = g
                break
        if not identity or identity not in subset:
            return False
        for a in subset:
            for b in subset:
                if operation(a, b) not in subset:
                    return False
        for a in subset:
            inverse_found = False
            for b in subset:
                if operation(a, b) == identity and operation(b, a) == identity:
                    inverse_found = True
                    break
            if not inverse_found:
                return False
        return True

    subgroups = []
    for i in range(1, len(group) + 1):
        for subset in itertools.combinations(group, i):
            if is_subgroup(subset, group, operation):
                subgroups.append(set(subset))

    # Export subgroups to JSON
    with open(export_filename, "w") as f:
        json.dump(list(map(list, subgroups)), f)

    # Visualize subgroups
    G = nx.Graph()
    for element in group:
        G.add_node(str(element))
    for subgroup in subgroups:
        H = G.subgraph(map(str, subgroup))
        pos = nx.spring_layout(H)
        nx.draw(H, pos, with_labels=True, node_color='skyblue')
    plt.show()

    return subgroups  # Optionally return the subgroups for further processing


class InvalidInputError(Exception):
    """Custom exception for invalid input."""
    pass


class RelationError(InvalidInputError):
    """Custom exception for errors related to relations."""
    pass


class TimeoutError(InvalidInputError):
    """Custom exception for exceeding the execution time limit."""
    pass


def calculate_group_order(generators, relations, profiling=False, timeout=None, cycle_length=None, symmetric=False,
                          transitive=False, disjoint_sets=False):
    # Input validation
    if not all(isinstance(generator, int) for generator in generators):
        raise InvalidInputError("Generators must be a list of integers.")
    if len(generators) != len(set(generators)):
        raise InvalidInputError("Generators must be unique.")
    if not all(
            isinstance(relation, tuple) and len(relation) > 1 and all(isinstance(i, int) for i in relation) for relation
            in relations):
        raise RelationError("Relations must be a list of tuples of integers, each with at least two elements.")
    if not isinstance(profiling, bool):
        raise InvalidInputError("Profiling must be a boolean value.")
    if timeout is not None and (not isinstance(timeout, (int, float)) or timeout <= 0):
        raise InvalidInputError("Timeout must be a positive number.")

    start_time = time.time()

    # Check relation consistency
    max_relation_index = max(max(relation) for relation in relations)
    if max_relation_index >= len(generators):
        raise RelationError("Relations contain indices that are out of bounds for the generators list.")
    for relation in relations:
        if len(relation) != len(set(relation)):
            raise RelationError("Each relation must contain unique elements.")
        if cycle_length is not None and len(relation) != cycle_length:
            raise RelationError(f"All relations must form cycles of length {cycle_length}.")

    # Check for symmetry
    if symmetric:
        for relation in relations:
            if tuple(reversed(relation)) not in relations:
                raise RelationError("Relations must be symmetric.")

    # Check for transitivity
    if transitive:
        for relation in relations:
            for other_relation in relations:
                if relation[-1] == other_relation[0] and (relation[0], other_relation[-1]) not in relations:
                    raise RelationError("Relations must be transitive.")

    # Check for disjoint sets
    if disjoint_sets:
        all_elements = set()
        for relation in relations:
            if any(element in all_elements for element in relation):
                raise RelationError("Relations must form disjoint sets.")
            all_elements.update(relation)

    # Function to check if a permutation is valid
    def is_valid_permutation(perm, valid_perms_cache):
        perm_key = tuple(perm)
        if perm_key in valid_perms_cache:
            return valid_perms_cache[perm_key]

        perm_dict = {i: perm[i] for i in range(len(generators))}
        for relation in relations:
            for i in range(len(relation)):
                if perm_dict[relation[i]] != relation[(i + 1) % len(relation)]:
                    valid_perms_cache[perm_key] = False
                    return False

        valid_perms_cache[perm_key] = True
        return True

    # Generate all permutations
    all_perms = list(permutations(range(len(generators))))

    # Parallel processing to compute valid permutations
    try:
        with Manager() as manager:
            valid_perms_cache = manager.dict()
            with Pool(os.cpu_count()) as pool:
                valid_perms = list(
                    tqdm(pool.starmap(is_valid_permutation, [(perm, valid_perms_cache) for perm in all_perms]),
                         total=len(all_perms), desc="Calculating valid permutations"))
    except Exception as e:
        raise InvalidInputError(f"Error during parallel processing: {e}")

    # Count the number of valid permutations
    valid_count = sum(valid_perms)

    # Logging and profiling
    end_time = time.time()
    logging.info(f"Number of valid permutations: {valid_count}")
    if profiling:
        logging.info(f"Execution time: {end_time - start_time:.2f} seconds")

    # Check for timeout
    if timeout is not None and (end_time - start_time) > timeout:
        raise TimeoutError("Function execution exceeded the specified timeout.")

    return valid_count


def calculate_group_elements(group):
    return list(group.elements)


def calculate_group_isomorphisms(group1, group2):
    return group1.is_isomorphic(group2)


# Ring Theory
def calculate_ring_order(ring):
    return len(ring)


def calculate_ring_elements(ring):
    return list(ring.elements)


def calculate_ideals(ring):
    return list(ring.ideals())


def calculate_ring_isomorphisms(ring1, ring2):
    return ring1.is_isomorphic(ring2)


# Field Theory
def calculate_field_order(field):
    return len(field)


def calculate_field_elements(field):
    return list(field.elements)


def calculate_subfields(field):
    return list(field.subfields())


def calculate_field_isomorphisms(field1, field2):
    return field1.is_isomorphic(field2)


def create_permutation_group(*permutations):
    perms = [Permutation(p) for p in permutations]
    return PermutationGroup(*perms)


def calculate_galois_group(polynomial):
    x = symbols('x')
    return galois_group(polynomial, x)


def calculate_angle(line1, line2):
    # Calculate the direction of the lines
    dir1 = [line1[1][0] - line1[0][0], line1[1][1] - line1[0][1]]
    dir2 = [line2[1][0] - line2[0][0], line2[1][1] - line2[0][1]]

    # Calculate the angle between the lines
    dot_product = dir1[0] * dir2[0] + dir1[1] * dir2[1]
    mag1 = math.sqrt(dir1[0] ** 2 + dir1[1] ** 2)
    mag2 = math.sqrt(dir2[0] ** 2 + dir2[1] ** 2)

    # Check if the lines are parallel
    if mag1 == 0 or mag2 == 0:
        return "The lines are parallel."

    cos_angle = dot_product / (mag1 * mag2)
    angle = math.degrees(math.acos(cos_angle))

    return angle


def calculate_derivative(expr):
    """
    Calculates the derivative of an expression with respect to x.
    """
    try:
        derivative = diff(expr, x)
        return derivative
    except SympifyError as e:
        print(f"Error calculating derivative: {e}")
        return None


def calculate_integral(expr, a=None, b=None):
    """
    Calculates the integral of an expression. If a and b are provided, calculates a definite integral.
    """
    try:
        if a is not None and b is not None:
            integral = integrate(expr, (x, a, b))
        else:
            integral = integrate(expr, x)
        return integral
    except SympifyError as e:
        print(f"Error calculating integral: {e}")
        return None


def calculate_dot_product(v1, v2):
    return np.dot(v1, v2)


def calculate_cross_product(v1, v2):
    return np.cross(v1, v2)


def calculate_determinant(matrix):
    return np.linalg.det(matrix)


def parse_complex_input(equation):
    """
    Parses a complex number from a given equation string.
    Supports input in formats like 'Z = a+bj' or directly 'a+bj'.
    """
    try:
        if '=' in equation:
            _, equation = equation.split('=')
        # Replace 'j' with 'j*' for Python complex number compatibility if necessary
        equation = equation.replace('j', 'j*' if 'j*' not in equation else 'j')
        z = complex(eval(equation.strip()))
        return z
    except Exception as e:
        print(f"Error parsing complex number: {e}")
        return None


def calculate_complex_division(z1, z2):
    """
    Handles division by zero more gracefully for complex numbers.
    """
    try:
        return z1 / z2
    except ZeroDivisionError:
        return "Error: Division by zero is not allowed."


def parse_vector_input(input_str):
    """
    Parses a string input into a vector, supporting various formats.
    This includes handling inputs with parentheses or brackets,
    inputs with mixed separators (spaces, commas), and inputs with negative numbers or scientific notation.
    """
    # Remove common vector notations (parentheses, brackets) and strip whitespace
    cleaned_input = re.sub(r'[()\[\]]', '', input_str).strip()

    # Use regex to find all numbers, including negatives and scientific notation
    numbers = re.findall(r'-?\d+\.?\d*(?:e-?\d+)?', cleaned_input)

    try:
        # Convert found strings to float
        vector = [float(num) for num in numbers]
        if not vector:
            raise ValueError("No valid numbers found. Please enter a valid vector format.")
        return vector
    except ValueError as e:
        # Rethrow the error with a more descriptive message
        raise ValueError("Invalid vector format. Ensure you're using numbers separated by commas or spaces.") from e


def calculate_vector_dot_product(v1, v2, pad_value=0, elementwise_op=None, final_callback=None):
    """
    Enhanced dot product calculation with support for complex numbers, custom elementwise operations,
    and a final callback function.
    """
    max_length = max(len(v1), len(v2))
    v1_padded = v1 + [pad_value] * (max_length - len(v1))
    v2_padded = v2 + [pad_value] * (max_length - len(v2))

    if elementwise_op:
        product_sum = sum(elementwise_op(a, b) for a, b in zip(v1_padded, v2_padded))
    else:
        product_sum = sum(a * b for a, b in zip(v1_padded, v2_padded))

    if final_callback:
        return final_callback(product_sum)
    else:
        return product_sum


def is_numeric(value):
    """Check if the value is numeric (int, float, complex)."""
    return isinstance(value, (int, float, complex))


def flatten_vector_input(vector):
    """
    Recursively flattens a vector input from nested lists/tuples into a plain list and verifies numeric components.
    """
    flattened_vector = []
    for item in vector:
        if isinstance(item, (list, tuple)):
            flattened_vector.extend(flatten_vector_input(item))
        elif is_numeric(item):
            flattened_vector.append(item)
        else:
            raise ValueError("Vector components must be numeric.")
    return flattened_vector


def calculate_vector_cross_product(*vectors):
    """
    Sophisticated calculation of the cross product for two or more 3-dimensional vectors, handling nested and mixed structures.
    """
    if len(vectors) < 2:
        raise ValueError("At least two vectors are required for the cross product.")
    vectors = [flatten_vector_input(v) for v in vectors]
    for v in vectors:
        if len(v) != 3:
            raise ValueError("All vectors must be 3-dimensional for the cross product.")
    result_vector = vectors[0]
    for v in vectors[1:]:
        result_vector = [
            result_vector[1] * v[2] - result_vector[2] * v[1],
            result_vector[2] * v[0] - result_vector[0] * v[2],
            result_vector[0] * v[1] - result_vector[1] * v[0]
        ]
    return result_vector


def calculate_vector_addition(*vectors, in_place=False, scale_factor=None, normalize=False, return_magnitude=False,
                              norm_type='euclidean', skip_non_numeric=False, default_non_numeric=0,
                              custom_function=None):
    """
    Adds multiple vectors of potentially different dimensions together, leveraging NumPy for efficiency.
    Offers options for in-place modification, scaling, normalization, returning the magnitude of the result vector,
    handling non-numeric elements, and applying a custom function to elements before addition.

    Parameters:
    - vectors: Variable number of iterable vectors (lists, tuples, etc.).
    - in_place (bool): If True, modifies the first vector in-place. Default is False.
    - scale_factor (Number, optional): Scales each vector by this factor before addition.
    - normalize (bool): If True, normalizes the result vector to have a magnitude of 1.
    - return_magnitude (bool): If True, returns the magnitude of the result vector instead.
    - norm_type (str or int): Specifies the type of norm for normalization and magnitude calculation. Can be 'euclidean' (default), 'l1', 'l2', or any other p-norm indicated by an integer.
    - skip_non_numeric (bool): If True, non-numeric elements are skipped, using default_non_numeric instead.
    - default_non_numeric (Number): Default value to replace non-numeric elements if skip_non_numeric is True.
    - custom_function (callable): Optional function to apply to each element before addition.

    Returns:
    - A NumPy array of the result vector, or its magnitude.

    Note: This function requires NumPy.
    """
    logging.info("Starting vector addition with {} vectors".format(len(vectors)))

    # Determine the maximum length of the input vectors
    max_length = max(len(v) for v in vectors)
    logging.info("Maximum vector length determined: {}".format(max_length))

    # Initialize the result vector
    result = np.zeros(max_length)
    logging.info("Initialized result vector with zeros.")

    # Process each vector
    for index, vector in enumerate(vectors):
        # Convert to NumPy array for efficient computation
        np_vector = np.array(vector, dtype=float)
        logging.debug("Converted vector {} to NumPy array.".format(index + 1))

        # Apply custom function if provided
        if custom_function is not None:
            np_vector = custom_function(np_vector)
            logging.debug("Applied custom function to vector {}.".format(index + 1))

        # Handle non-numeric values
        if skip_non_numeric:
            non_finite_mask = ~np.isfinite(np_vector)
            if np.any(non_finite_mask):
                np_vector[non_finite_mask] = default_non_numeric
                logging.warning(
                    "Non-numeric values found in vector {}; replaced with default value {}.".format(index + 1,
                                                                                                    default_non_numeric))

        # Scale the vector if a scale factor is provided
        if scale_factor is not None:
            np_vector *= scale_factor
            logging.debug("Scaled vector {} by factor {}.".format(index + 1, scale_factor))

        # Pad the vector with zeros if it's shorter than the max length
        padded_vector = np.pad(np_vector, (0, max_length - len(np_vector)), 'constant', constant_values=0)

        # Add the processed vector to the result
        result += padded_vector

    logging.info("Completed adding vectors.")

    # Normalize the result vector if requested
    if normalize:
        norm = np.linalg.norm(result, ord={'euclidean': 2, 'l1': 1, 'l2': 2}.get(norm_type, norm_type))
        if norm > 0:
            result /= norm
        logging.info("Normalized the result vector.")

    # Return the magnitude of the result vector if requested
    if return_magnitude:
        magnitude = np.linalg.norm(result, ord={'euclidean': 2, 'l1': 1, 'l2': 2}.get(norm_type, norm_type))
        logging.info("Returning the magnitude of the result vector: {}".format(magnitude))
        return magnitude

    logging.info("Returning the result vector.")
    return result


def calculate_vector_subtraction(*vectors, in_place=False, padding_value=0, return_type='auto'):
    """
    Enhanced vector subtraction with support for broadcasting, improved error handling,
    and flexible return types.

    Parameters:
    - vectors (sequence of sequences or scalars): Input vectors for subtraction.
    - in_place (bool): If True, modifies the first vector in-place (only for mutable sequences).
    - padding_value (numeric): Value to use for padding shorter vectors.
    - return_type (str): Specifies the return type. Options are 'list', 'tuple', 'numpy', 'pandas', or 'auto'.
      The 'auto' option returns the same type as the first vector or numpy.ndarray if mixed types are provided.

    Returns:
    - The result of subtracting the given vectors, in the specified return type.

    Raises:
    - ValueError: For invalid input values or types.
    """
    if not vectors:
        raise ValueError("No vectors provided for subtraction.")

    # Convert all inputs to numpy arrays for consistency
    np_vectors = []
    for v in vectors:
        if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
            np_vectors.append(np.asarray(v, dtype=np.complex_))
        elif isinstance(v, (int, float, complex)):
            np_vectors.append(np.array([v]))
        else:
            raise ValueError(f"Unsupported vector type: {type(v)}")

    max_length = max(v.size for v in np_vectors)
    result_vector = np_vectors[0].copy()

    # Resize or pad the first vector if not in-place
    if not in_place:
        result_vector = np.pad(result_vector, (0, max_length - result_vector.size), 'constant',
                               constant_values=padding_value)

    # Subtract subsequent vectors
    for vector in np_vectors[1:]:
        vector = np.pad(vector, (0, max_length - vector.size), 'constant', constant_values=padding_value)
        result_vector -= vector

    logging.debug(f"Subtraction completed with result of size {result_vector.size}.")

    # Handle return type
    if return_type == 'auto':
        if isinstance(vectors[0], np.ndarray):
            return result_vector
        elif isinstance(vectors[0], pd.Series):
            return pd.Series(result_vector)
        elif isinstance(vectors[0], tuple):
            return tuple(result_vector)
        else:
            return result_vector.tolist()
    elif return_type == 'list':
        return result_vector.tolist()
    elif return_type == 'tuple':
        return tuple(result_vector)
    elif return_type == 'numpy':
        return result_vector
    elif return_type == 'pandas':
        return pd.Series(result_vector)
    else:
        raise ValueError("Invalid return type specified.")


def calculate_scalar_multiplication(scalar, *vectors, right_scalar=False, pad_value=0, callback=None):
    """
    Performs scalar or element-wise vector multiplication on one or more vectors, with optional padding and a callback function.
    """
    max_length = max(len(vector) for vector in vectors) if vectors else 0
    result_vectors = []

    for vector in vectors:
        extended_vector = vector + [pad_value] * (max_length - len(vector))
        if isinstance(scalar, list):
            extended_scalar = scalar + [pad_value] * (max_length - len(scalar))
            result_vector = [a * b for a, b in zip(extended_vector, extended_scalar)]
        else:
            result_vector = [scalar * element for element in extended_vector]

        if callback:
            result_vector = [callback(element) for element in result_vector]

        result_vectors.append(result_vector)

    return result_vectors[0] if len(vectors) == 1 else result_vectors


def calculate_polynomial_addition(p1, p2, symbolic: bool, variables: str, pretty_print: bool):
    vars = symbols(variables)
    p1_poly, p2_poly = Poly(sympify(p1), vars), Poly(sympify(p2), vars)
    result = p1_poly + p2_poly
    if symbolic:
        return latex(result.as_expr()) if pretty_print else str(result.as_expr())
    else:
        return result.all_coeffs()


def calculate_polynomial_subtraction(p1: Union[str, Poly],
                                     p2: Union[str, Poly],
                                     symbolic: bool,
                                     variables: str,
                                     pretty_print: bool,
                                     simplify_result: bool = True) -> Union[str, Poly]:
    """
    Subtracts two polynomials with enhanced functionalities including automatic simplification,
    support for multiple variables, pretty printing, and error handling.

    Parameters:
    - p1, p2 (Union[str, Poly]): Polynomials to subtract, in various formats.
    - symbolic (bool): If True, returns a symbolic string representation.
    - variables (str): Comma-separated variable names for multivariable polynomials.
    - pretty_print (bool): If True, returns the result in LaTeX format for readability.
    - simplify_result (bool): If True, simplifies the result before returning.

    Returns:
    - Union[str, Poly]: The difference of the two polynomials, formatted as specified.
    """
    # Convert comma-separated variable names into SymPy symbols
    vars = symbols(variables)

    # Convert input to SymPy Poly objects if they are not already
    try:
        if isinstance(p1, str):
            p1 = Poly(sympify(p1), vars)
        if isinstance(p2, str):
            p2 = Poly(sympify(p2), vars)
    except SympifyError as e:
        raise ValueError(f"Error converting input to polynomial: {e}")

    # Perform polynomial subtraction
    result_poly = p1 - p2

    # Simplify the result if requested
    if simplify_result:
        result_poly = simplify(result_poly)

    # Format output based on the 'symbolic' and 'pretty_print' flags
    if symbolic:
        if pretty_print:
            return latex(result_poly.as_expr())
        else:
            return str(result_poly.as_expr())
    else:
        # Return as a Poly object or list of coefficients based on preference
        return result_poly if not symbolic else result_poly.all_coeffs()


def calculate_polynomial_multiplication(p1: Union[str, Poly],
                                        p2: Union[str, Poly],
                                        symbolic: bool = False,
                                        simplify_result: bool = True,
                                        variables: str = 'x',
                                        pretty_print: bool = False) -> Union[str, Poly]:
    """
    Multiplies two polynomials, supporting symbolic expressions, SymPy Poly objects,
    and multiple variables. Offers options for simplification and pretty printing.

    Parameters:
    - p1, p2 (Union[str, Poly]): Polynomials to multiply, as strings or Poly objects.
    - symbolic (bool): If True, returns a symbolic string representation.
    - simplify_result (bool): If True, simplifies the result before returning.
    - variables (str): Comma-separated variable names for multivariable polynomials.
    - pretty_print (bool): If True, returns the result in LaTeX format for readability.

    Returns:
    - Union[str, Poly]: The product of the two polynomials, formatted as specified.
    """
    # Parse variables
    vars = symbols(variables)

    # Ensure input is in Poly form
    if isinstance(p1, str):
        p1 = Poly(sympify(p1), vars)
    if isinstance(p2, str):
        p2 = Poly(sympify(p2), vars)

    # Perform multiplication
    result_poly = p1 * p2
    if simplify_result:
        result_poly = expand(result_poly)

    # Format output
    if symbolic:
        result_str = str(result_poly) if not pretty_print else latex(result_poly)
        return result_str
    else:
        # Returning as a Poly object; users can extract coeffs or further manipulate if desired
        return result_poly


def calculate_polynomial_division(p1: Union[str, Poly],
                                  p2: Union[str, Poly],
                                  symbolic: bool = False,
                                  variables: str = 'x',
                                  pretty_print: bool = False) -> Union[Tuple[str, Poly], str]:
    """
    Divides two polynomials, supporting inputs as symbolic expressions or SymPy Poly objects,
    with options for simplification, multiple variables, and pretty printing.

    Parameters:
    - p1, p2 (Union[str, Poly]): The dividend and divisor polynomials, respectively.
    - symbolic (bool): If True, returns a symbolic string representation of the division result.
    - variables (str): Comma-separated string of variable names for handling multivariable polynomials.
    - pretty_print (bool): If True and symbolic is True, returns the result in LaTeX format.

    Returns:
    - Union[Tuple[str, Poly], str]: The quotient and remainder as a tuple (if symbolic=False), or
      the division result as a string (if symbolic=True). If pretty_print is True, the result
      is returned in LaTeX format.
    """
    # Convert variables string to SymPy symbols
    vars = symbols(variables)

    # Ensure inputs are SymPy Poly objects
    if isinstance(p1, str):
        p1 = Poly(sympify(p1), vars)
    if isinstance(p2, str):
        p2 = Poly(sympify(p2), vars)

    # Perform polynomial division
    quotient, remainder = div(p1, p2, domain='QQ')

    if symbolic:
        # Format result as a symbolic expression or LaTeX
        result = f"Quotient: {latex(quotient.as_expr()) if pretty_print else quotient.as_expr()}, " \
                 f"Remainder: {latex(remainder.as_expr()) if pretty_print else remainder.as_expr()}"
        return result
    else:
        # Return quotient and remainder as Poly objects or in the specified format
        return quotient, remainder


def solve_polynomial_equation(polynomial: Union[str, Poly],
                              variables: str,
                              pretty_print: bool = False,
                              symbolic_output: bool = False) -> Union[str, list]:
    """
    Solves a polynomial equation for its roots, with options for output formatting.

    Parameters:
    - polynomial (Union[str, Poly]): The polynomial equation to solve.
    - variables (str): Comma-separated string of variables in the polynomial equation.
    - pretty_print (bool): If True, returns the result in LaTeX format (for Jupyter environments).
    - symbolic_output (bool): If True, returns the solutions as a symbolic string.

    Returns:
    - Union[str, list]: Solutions of the polynomial equation, formatted based on the input flags.
    """
    # Convert input string to a SymPy expression, respecting the specified variables
    vars = symbols(variables)
    if isinstance(polynomial, str):
        polynomial = sympify(polynomial, locals={str(v): v for v in vars})

    # Solve the polynomial equation
    primary_var = vars[0] if isinstance(vars, tuple) else vars  # Use the first variable if multiple are provided
    solutions = solve(polynomial, primary_var)

    # Format output based on flags
    if symbolic_output:
        solutions_str = ', '.join([latex(sol, mode='inline') if pretty_print else str(sol) for sol in solutions])
        if pretty_print:
            from IPython.display import display, Math
            display(Math(solutions_str))
            return ""  # Return an empty string to avoid duplicate output outside Jupyter
        else:
            return solutions_str
    else:
        # Return as a list for direct manipulation or further processing
        return solutions


def detect_content_type(matrix):
    """
    Enhanced function to detect if the matrix content is numeric, symbolic, or mixed.
    """
    has_numeric = has_symbolic = False
    for row in matrix:
        for val in row:
            if isinstance(val, sp.Expr):
                # Checking specifically for symbolic expressions
                has_symbolic = True
                if val.is_Number:
                    # sp.Expr that is a Number is still considered numeric
                    has_numeric = True
            elif isinstance(val, (int, float)):
                has_numeric = True
            else:
                # Any non-SymPy and non-numeric value will be considered symbolic for safety
                has_symbolic = True

    # Determining the content type based on the flags
    if has_numeric and has_symbolic:
        return "mixed"
    elif has_symbolic:
        return "symbolic"
    return "numeric"


def pretty_print_matrix(matrix, title="Matrix", unicode_borders=False, colorize=True):
    """
    Unified pretty_print_matrix function that combines improved formatting,
    optional Unicode borders, and color-coding.
    """
    print(f"\n{title}:")
    content_type = detect_content_type(matrix)
    if isinstance(matrix, sp.Matrix):
        matrix = matrix.tolist()

    # Setting up border characters based on the unicode_borders flag
    border_char = '│' if unicode_borders else '|'
    dash_char = '─' if unicode_borders else '-'
    corner_char = '┼' if unicode_borders else '+'

    # Determine column widths
    col_widths = [max(len(format_element(row[i], 0, content_type, colorize=False)) for row in matrix) + 2 for i in
                  range(len(matrix[0]))]
    border_line = corner_char + corner_char.join(dash_char * width for width in col_widths) + corner_char

    # Print top border
    print(border_line)
    for row in matrix:
        formatted_row = [format_element(row[i], col_widths[i], content_type, colorize) for i in range(len(row))]
        row_str = ' '.join(formatted_row)
        print(f"{border_char} {row_str} {border_char}")
    # Print bottom border
    print(border_line)


def safe_eval(expr):
    """
    Safely evaluates a mathematical expression, including handling fractions, parentheses, and symbolic expressions.
    """
    try:
        # Handle numeric and fractional inputs
        return float(Fraction(expr))
    except ValueError:
        # Handle symbolic expressions
        return sp.sympify(expr)


def input_matrix(prompt):
    """
    Enhances user input for matrix to handle mathematical expressions, symbolic expressions, and dynamic sizes.
    Provides immediate visual feedback on entered matrix.
    """
    print(prompt + "\n(Examples: '1/2', '2*(3+4)', 'x + y'. Press 'r' to restart, 'd' when done.)")
    matrix = []
    cols = None

    while True:
        row_input = input("Enter next row values separated by space or 'r' to restart, 'd' to finish: ").strip().lower()
        if row_input == 'r':
            print("Restarting matrix input...")
            matrix.clear()  # Clear the matrix and restart input
            continue
        elif row_input == 'd' and matrix:
            break  # Finish input if 'd' is entered and matrix is not empty
        try:
            row = [safe_eval(val) for val in row_input.split()]
            if not matrix:  # If first row, set the number of columns
                cols = len(row)
            elif len(row) != cols:
                raise ValueError(f"All rows must have the same number of columns. Expected {cols}, got {len(row)}.")
            matrix.append(row)
        except (ValueError, SyntaxError, sp.SympifyError) as e:
            print(f"Invalid input: {e}. Please try again.")

    # Provide visual feedback of entered matrix
    pretty_print_matrix(matrix, "Entered Matrix")
    return matrix


def convert_to_sympy_matrix(matrix):
    """Converts a matrix to a SymPy Matrix for symbolic calculations."""
    if isinstance(matrix, np.ndarray):
        return sp.Matrix(matrix.tolist())
    elif isinstance(matrix, list):
        return sp.Matrix(matrix)
    return matrix  # Assume it's already a SymPy Matrix if not list or ndarray


def matrix_addition(m1, m2):
    """
    Robustly calculates the addition of two matrices, supporting symbolic expressions,
    and automatically adjusting for different sized matrices.
    """
    # Convert inputs to SymPy matrices for uniform handling
    matrix1 = convert_to_sympy_matrix(m1)
    matrix2 = convert_to_sympy_matrix(m2)

    # Ensure both matrices have the same dimensions, pad with zeros if necessary
    rows1, cols1 = matrix1.shape
    rows2, cols2 = matrix2.shape
    if (rows1, cols1) != (rows2, cols2):
        max_rows, max_cols = max(rows1, rows2), max(cols1, cols2)
        matrix1 = matrix1.row_join(sp.zeros(rows1, max_cols - cols1)).col_join(sp.zeros(max_rows - rows1, max_cols))
        matrix2 = matrix2.row_join(sp.zeros(rows2, max_cols - cols2)).col_join(sp.zeros(max_rows - rows2, max_cols))

    # Perform addition
    result_matrix = matrix1 + matrix2

    # Convert back to list if both inputs were lists for consistency
    if isinstance(m1, list) and isinstance(m2, list):
        return result_matrix.tolist()
    # For numpy array inputs, return a numpy array
    elif isinstance(m1, np.ndarray) and isinstance(m2, np.ndarray):
        return np.array(result_matrix.tolist())
    # Otherwise, return as a SymPy Matrix
    return result_matrix


def save_matrix_to_file(matrix, filename="matrix_result.txt", format="txt"):
    if format == "csv":
        with open(filename, "w") as file:
            for row in matrix:
                file.write(",".join(str(val) for val in row) + "\n")
    elif format == "json":
        with open(filename, "w") as file:
            json.dump(matrix, file)
    else:  # Default to plain text
        with open(filename, "w") as file:
            for row in matrix:
                file.write(" ".join(str(val) for val in row) + "\n")
    print(f"Matrix saved to {filename} in {format} format.")


def load_matrix_from_file(filename):
    with open(filename, "r") as file:
        matrix = [line.strip().split() for line in file.readlines()]
    return [[sp.sympify(value) for value in row] for row in matrix]


def input_or_load_matrix(prompt):
    choice = input(prompt + " Enter 'file' to load from a file, or anything else to input manually: ").strip().lower()
    if choice == 'file':
        filename = input("Enter the filename: ").strip()
        return load_matrix_from_file(filename)
    else:
        return input_matrix("Enter the matrix")


def substitute_symbols(matrix):
    """Substitutes symbols in a symbolic matrix with user-provided values."""
    substituted_matrix = []
    for row in matrix:
        new_row = []
        for element in row:
            if isinstance(element, sp.Expr) and element.free_symbols:
                for symbol in element.free_symbols:
                    value = sp.sympify(input(f"Enter value for {symbol}: "))
                    element = element.subs(symbol, value)
            new_row.append(element)
        substituted_matrix.append(new_row)
    return substituted_matrix


def plot_matrix(matrix, title="Matrix Visualization"):
    """
    Enhanced plotting function that can handle both numeric and symbolic matrices.
    For numeric matrices, it creates a detailed visualization with cell annotations.
    """
    # Convert symbolic matrix to numeric if possible
    if any(isinstance(item, sp.Expr) for row in matrix for item in row):
        try:
            # Attempt to evaluate the matrix if it's purely symbolic (no free symbols)
            matrix_eval = [[float(item.evalf()) if item.is_Number else None for item in row] for row in matrix]
            if all(item is not None for row in matrix_eval for item in row):
                matrix = np.array(matrix_eval)
            else:
                print("Plotting is only available for numeric matrices or symbolic matrices with no free symbols.")
                return
        except Exception as e:
            print(f"Unable to evaluate symbolic matrix for plotting: {e}")
            return
    else:
        matrix = np.array(matrix)  # Ensure matrix is a NumPy array for plotting

    # Proceed with plotting for numeric matrices
    fig, ax = plt.subplots()
    cax = ax.matshow(matrix, cmap='viridis')
    fig.colorbar(cax)

    # Annotate the cells with the numeric values
    for (i, j), val in np.ndenumerate(matrix):
        ax.text(j, i, f"{val:.2f}", ha='center', va='center', color='white' if val < matrix.mean() else 'black')

    plt.title(title)
    plt.show()


def calculate_matrix_subtraction_operational(m1, m2):
    """
    Core operational function for matrix subtraction, handling both symbolic and numeric matrices.
    """
    matrix1, matrix2 = ensure_matrix_format(m1), ensure_matrix_format(m2)

    if isinstance(matrix1, sp.Matrix) or isinstance(matrix2, sp.Matrix):
        result = matrix1 - matrix2
    else:
        result = np.subtract(matrix1, matrix2)

    return result.tolist() if hasattr(result, 'tolist') else result


def calculate_matrix_subtraction(m1, m2):
    """
    Enhanced matrix subtraction function integrating user input, symbolic substitution, and additional features.
    """
    print("Input for the first matrix:")
    m1 = input_or_load_matrix("Enter the first matrix")
    print("Input for the second matrix:")
    m2 = input_or_load_matrix("Enter the second matrix")

    # Handle symbolic substitutions if applicable
    has_symbols = any(isinstance(item, sp.Expr) for row in m1 + m2 for item in row)
    if has_symbols:
        print("Symbolic expressions detected. You'll have the option to substitute symbols with values.")
        if input("Substitute symbols in the first matrix? (yes/no): ").strip().lower() == 'yes':
            m1 = substitute_symbols(m1)
        if input("Substitute symbols in the second matrix? (yes/no): ").strip().lower() == 'yes':
            m2 = substitute_symbols(m2)

    # Perform matrix subtraction using the core operational function
    result_list = calculate_matrix_subtraction_operational(m1, m2)

    # Pretty print the result
    pretty_print_matrix(result_list, "Result Matrix")

    # Save and plot functionalities
    if input("Do you want to save the result matrix? (yes/no): ").strip().lower() == 'yes':
        filename = input("Enter the filename (default 'matrix_result.txt'): ").strip() or "matrix_result.txt"
        format_choice = input("Choose the format - txt, csv, or json (default 'txt'): ").strip() or "txt"
        save_matrix_to_file(result_list, filename, format_choice)

    if all(isinstance(val, (int, float)) for row in result_list for val in row) and \
            input("Do you want to plot the result matrix? (yes/no): ").strip().lower() == 'yes':
        plot_matrix(np.array(result_list), "Subtraction Result")


def ensure_matrix_format(matrix):
    if any(isinstance(item, sp.Expr) for row in matrix for item in row):
        return sp.Matrix(matrix)
    else:
        return np.array(matrix)


def calculate_matrix_multiplication():
    print("Matrix Multiplication")
    m1 = input_matrix("Enter the first matrix (type 'done' when finished with the matrix):")
    m2 = input_matrix("Enter the second matrix (type 'done' when finished with the matrix):")

    matrix1 = ensure_matrix_format(m1)
    matrix2 = ensure_matrix_format(m2)

    # Dimension compatibility check
    try:
        if isinstance(matrix1, np.ndarray) and isinstance(matrix2, np.ndarray):
            assert matrix1.shape[1] == matrix2.shape[0], "Dimension mismatch"
            result = np.dot(matrix1, matrix2)
        elif isinstance(matrix1, sp.Matrix) and isinstance(matrix2, sp.Matrix):
            result = matrix1 * matrix2
        else:
            raise ValueError("Unsupported matrix types.")
    except (AssertionError, sp.ShapeError, ValueError) as e:
        print(f"Error: {e}")
        return

    result_list = result.tolist() if hasattr(result, 'tolist') else result

    print("Result of Matrix Multiplication:")
    for row in result_list:
        print(" ".join(str(val) for val in row))

    # Symbolic evaluation option
    if any(isinstance(item, sp.Expr) for row in result_list for item in row):
        if input("Evaluate symbolic expressions in the result? (yes/no): ").strip().lower() == 'yes':
            result_list = [[val.evalf() for val in row] for row in result_list]

    # Visualization option for numeric results
    if all(isinstance(val, (int, float, sp.Number)) for row in result_list for val in row):
        if input("Visualize the result matrix? (yes/no): ").strip().lower() == 'yes':
            plot_matrix(np.array(result_list), "Result of Matrix Multiplication")

    # Save result option
    if input("Save the result matrix to a file? (yes/no): ").strip().lower() == 'yes':
        filename = input("Enter filename (e.g., 'result_matrix.txt'): ").strip()
        format_choice = input("File format (txt, csv, json): ").strip().lower()
        save_matrix_to_file(result_list, filename, format_choice)  # Assuming this function is implemented as discussed


def format_element(element, width, content_type, colorize=True):
    """
    Unified function to format an element for pretty printing, combining dynamic coloring,
    precision control, and optional colorization.
    """
    # Default formatting and colors
    reset_color = "\033[0m"
    color = "\033[94m" if content_type == "numeric" else "\033[92m" if content_type == "symbolic" else "\033[0m"
    negative_color = "\033[91m"  # Red for negative numbers

    # Formatting based on content type
    if content_type == 'numeric':
        formatted = f"{float(element):.2f}".rjust(width)  # Ensure conversion to float for consistent formatting
    else:
        formatted = str(element).rjust(width)

    # Apply colorization based on the value and user choice
    if colorize:
        if content_type == 'numeric' and float(element) < 0:
            return f"{negative_color}{formatted}{reset_color}"
        else:
            return f"{color}{formatted}{reset_color}" if content_type != "mixed" else formatted
    else:
        return formatted


def calculate_matrix_division():
    print("Matrix Division (A * B^-1)")
    m1 = input_matrix("Enter the first matrix (A):")
    m2 = input_matrix("Enter the second matrix (B):")

    matrix1 = ensure_matrix_format(m1)
    matrix2 = ensure_matrix_format(m2)

    try:
        # Check for invertibility of the second matrix
        if isinstance(matrix2, sp.Matrix):
            if matrix2.det() == 0:
                raise ValueError("The second matrix is not invertible.")
            matrix2_inv = matrix2.inv()
        elif isinstance(matrix2, np.ndarray):
            det = np.linalg.det(matrix2)
            if det == 0 or np.isclose(det, 0):
                raise ValueError("The second matrix is not invertible.")
            matrix2_inv = np.linalg.inv(matrix2)
        else:
            raise TypeError("Unknown matrix type.")

        # Perform matrix division
        result = matrix1 * matrix2_inv
        result_list = result.tolist() if hasattr(result, 'tolist') else result

        # Display the result
        pretty_print_matrix(result_list, "Result of Matrix Division (A * B^-1)")
    except ValueError as e:
        print(f"Error: {e}")
    except TypeError as e:
        print(f"Error: {e}")


def calculate_matrix_inverse(m1):
    return np.linalg.inv(m1).tolist()


def calculate_matrix_transpose(m1):
    return np.transpose(m1).tolist()


def validate_complex_input(complex_number):
    try:
        # Convert the input to a complex number
        complex_number = complex(complex_number)
        return complex_number
    except ValueError:
        print("Error: Input is not a valid complex number.")
        return None


def validate_abelian_group_elements(group):
    # This is a placeholder function. You'll need to replace this with your actual validation logic.
    if isinstance(group, list) and all(isinstance(element, int) for element in group):
        return group
    else:
        print("Invalid group elements. Please enter a list of integers.")
        return None


def calculate_power(base: Decimal, exponent: Decimal) -> Decimal:
    getcontext().prec = 50000  # Set the precision for high precision arithmetic
    try:
        # Validate inputs
        if base == 0 and exponent < 0:
            raise ValueError("Cannot raise zero to a negative power.")
        elif base < 0 and not exponent.is_integer():
            raise ValueError("Cannot raise a negative number to a fractional power.")

        result = base ** exponent
        return result
    except (ValueError, OverflowError, InvalidOperation) as e:
        print(f"Error: {e}")
        return None


def calculate_factorial(n: int) -> int:
    try:
        # Check if the input is a non-negative integer
        if n < 0 or not isinstance(n, int):
            raise ValueError("Factorial is only defined for non-negative integers")
        else:
            # Calculate the factorial
            factorial = 1
            for i in range(1, n + 1):
                factorial *= i
            return factorial
    except ValueError as e:
        print(f"Error: {e}")


def calculate_absolute_value(n: float) -> float:
    try:
        # Check if the input is a number
        if not isinstance(n, (int, float)):
            raise ValueError("Absolute value is only defined for real numbers")
        else:
            # Calculate the absolute value
            if n < 0:
                return -n
            else:
                return n
    except ValueError as e:
        print(f"Error: {e}")


def calculate_logarithm(n: float, base: float) -> float:
    return math.log(n, base)


def calculate_sine(angle_in_degrees: float) -> float:
    angle_in_radians = math.radians(angle_in_degrees)
    return math.sin(angle_in_radians)


def calculate_cosine(angle_in_degrees: float) -> float:
    angle_in_radians = math.radians(angle_in_degrees)
    return math.cos(angle_in_radians)


def hyperbolic_tangent_in_degrees(degree):
    radian = math.radians(degree)
    return math.tanh(radian)


def hyperbolic_operations(operation, x):
    if operation == 'sinh':
        return math.sinh(x)
    elif operation == 'cosh':
        return math.cosh(x)
    elif operation == 'tanh':
        return math.tanh(x)
    elif operation == 'asinh':
        return math.asinh(x)
    elif operation == 'acosh':
        return math.acosh(x)
    elif operation == 'atanh':
        return math.atanh(x)
    else:
        return "Invalid operation"


def str_to_matrix(matrix_str):
    """Convert a string to a matrix. The string should have rows separated by semicolons and columns separated by
    commas."""
    try:
        matrix = [[float(num) for num in row.split(',')] for row in matrix_str.split(';')]
        np_matrix = np.array(matrix, dtype=float)
        if np_matrix.ndim != 2:
            print("Error: Input is not a valid matrix.")
            return None
        return matrix
    except ValueError:
        print("Error: Matrix contains non-numeric values.")
        return None


def matrix_add(m1_str, m2_str):
    """Add two matrices. The matrices should be strings with rows separated by semicolons and columns separated by
    commas."""
    m1 = str_to_matrix(m1_str)
    m2 = str_to_matrix(m2_str)
    if len(m1) != len(m2) or len(m1[0]) != len(m2[0]):
        return "Error: The matrices must have the same dimensions to be added."
    return [[m1[i][j] + m2[i][j] for j in range(len(m1[0]))] for i in range(len(m1))]


def matrix_sub(m1_str, m2_str):
    """Subtract two matrices. The matrices should be strings with rows separated by semicolons and columns separated
    by commas."""
    m1 = str_to_matrix(m1_str)
    m2 = str_to_matrix(m2_str)
    if len(m1) != len(m2) or len(m1[0]) != len(m2[0]):
        return "Error: The matrices must have the same dimensions to be subtracted."
    return [[m1[i][j] - m2[i][j] for j in range(len(m1[0]))] for i in range(len(m1))]


def matrix_mul(m1_str, m2_str):
    """Multiply two matrices. The matrices should be strings with rows separated by semicolons and columns separated
    by commas."""
    m1 = str_to_matrix(m1_str)
    m2 = str_to_matrix(m2_str)
    if len(m1[0]) != len(m2):
        return "Error: The number of columns in the first matrix must be equal to the number of rows in the second " \
               "matrix."
    return [[sum(a * b for a, b in zip(m1_row, m2_col)) for m2_col in zip(*m2)] for m1_row in m1]


def matrix_div(m1_str, m2_str):
    """Divide two matrices. The matrices should be strings with rows separated by semicolons and columns separated by
    commas."""
    m1 = str_to_matrix(m1_str)
    m2 = str_to_matrix(m2_str)
    m2_inv = np.linalg.inv(m2)
    if m2_inv is None:
        return "Error: The second matrix is not invertible."
    return matrix_mul(m1, m2_inv)


def preprocess_equation(equation: str) -> str:
    """
    Preprocess the equation to handle common input formats and issues,
    including explicit multiplication signs, exponentiation notation, and special functions.
    """
    # Remove extra spaces
    equation = equation.replace(' ', '')

    # Handle unary minus
    equation = re.sub(r'(?<=^)-', '0-', equation)
    equation = re.sub(r'(?<=\()-', '0-', equation)

    # Add explicit multiplication
    equation = re.sub(r'(?<=[0-9\)])\(', '*(', equation)
    equation = re.sub(r'(?<=[0-9a-zA-Z\)])\b', '*', equation)
    equation = re.sub(r'\b(?=[a-zA-Z\(])', '*', equation)

    # Replace caret notation with ** for exponentiation
    equation = equation.replace('^', '**')

    # Handle decimal numbers
    equation = re.sub(r'(?<=\d)\.(?=\d)', '*0.', equation)
    equation = re.sub(r'(?<=\D)\.(?=\d)', '0.', equation)
    equation = re.sub(r'(?<=\d)\.(?=\D)', '0.*', equation)

    # Handle special constants and functions
    equation = re.sub(r'\bpi\b', 'pi', equation)
    equation = re.sub(r'\be\b', 'E', equation)
    equation = re.sub(r'\bE\^', 'exp(', equation) + (')' if 'E^' in equation else '')

    # Handle trigonometric functions
    for func in ['sin', 'cos', 'tan', 'csc', 'sec', 'cot']:
        equation = re.sub(fr'\b{func}\b', fr'{func}(', equation) + (')' if func in equation else '')

    # Handle square root notation
    equation = equation.replace('sqrt', 'sqrt(') + (')' if 'sqrt' in equation else '')

    # Handle logarithm notation
    equation = equation.replace('log', 'log(') + (')' if 'log' in equation else '')

    # Handle fractions
    equation = re.sub(r'(\d+)/(\d+)', r'(\1/\2)', equation)

    return equation


def parse_input(equation: str, variables: str, differentiate: str = None, integrate_var: str = None, simplify_expr: bool = False, substitutions: dict = None) -> "Expr":
    """
    Parses a polynomial equation from a string into a SymPy expression, considering the specified variables.
    Optionally performs differentiation, integration, simplification, and substitution.
    """
    try:
        equation = preprocess_equation(equation)
        # Convert variable names to SymPy symbols
        vars = symbols(variables)
        # Replace '^' with '**' for exponentiation
        equation = equation.replace('^', '**')
        # Parse the equation string to a SymPy expression
        expr = sympify(equation, locals={str(v): v for v in vars})

        # Perform differentiation if specified
        if differentiate:
            expr = diff(expr, symbols(differentiate))

        # Perform integration if specified
        if integrate_var:
            expr = integrate(expr, symbols(integrate_var))

        # Simplify the expression if specified
        if simplify_expr:
            expr = simplify(expr)

        # Perform substitutions if specified
        if substitutions:
            expr = expr.subs(substitutions)

        return expr
    except SympifyError as e:
        raise ValueError(f"Error parsing the equation: {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error: {e}")


def perform_operation(operation, expr):
    """
    Extends the operation handling to include more mathematical functions.
    """
    if operation in ["sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh"]:
        # Handle trigonometric and hyperbolic operations
        result = eval(f"{operation}(expr)")
        print(f"{operation}({expr}) = {result.evalf()}")
    elif operation == "exp":
        # Exponential function
        result = exp(expr)
        print(f"exp({expr}) = {result.evalf()}")
    elif operation == "solve":
        # Solve equations
        solutions = solve(expr, x)
        print(f"Solutions: {solutions}")
    else:
        print("Unsupported operation or not yet implemented.")
    try:
        if operation == "custom":
            # Use the custom function on the expression
            result = CustomFunction(expr)
            print(f"CustomFunction applied to {expr} = {result}")
        elif operation == "solve":
            # Solve the expression for x
            solutions = solve(expr, x)
            print(f"Solutions: {solutions}")
        # Add more elif clauses here for other operations
        else:
            print("Unsupported operation.")
    except Exception as e:
        print(f"An error occurred: {e}")


def perform_arthmetic(operation, Var1=None, Var2=None):
    try:
        if operation == "+":
            return Var1 + Var2
        elif operation == "-":
            return Var1 - Var2
        elif operation == "*":
            return Var1 * Var2
        elif operation == "/":
            if Var2 == 0:
                return "Error: Division by zero is not allowed."
            return Var1 / Var2
        else:
            print("Unsupported operation or incorrect input. Please try again.")
    except Exception as e:
        return f"An error occurred during the operation: {e}"


def safe_input_number(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Invalid input, please enter a valid number.")


def calculate_mod(z):
    """Calculates the modulus of a complex number."""
    return abs(z)


def calculate_conj(z):
    """Calculates the conjugate of a complex number."""
    return z.conjugate()


def calculate_sqrt(z):
    """Calculates the square root of a complex number."""
    return sqrt(z)


def calculate_cbrt(z):
    """Calculates the cube root of a complex number. Python's cmath does not include a cube root function,
    so we define our own."""
    return z ** (1 / 3)


def calculate_log_derivative(z):
    """Calculates the derivative of the logarithm of a complex number, which is 1/z."""
    if isclose(abs(z), 0.0, abs_tol=1e-9):
        return "Error: Cannot calculate the logarithmic derivative at 0."
    return 1 / z


class CommutativeRing:
    def __init__(self, elements):
        self.elements = elements

    def add(self, a, b):
        if a in self.elements and b in self.elements:
            return (a + b) % len(self.elements)
        else:
            print("Error: Elements not in ring.")
            return None

    def multiply(self, a, b):
        if a in self.elements and b in self.elements:
            return (a * b) % len(self.elements)
        else:
            print("Error: Elements not in ring.")
            return None


class CustomFunction(Function):
    # Example custom function definition
    @classmethod
    def eval(cls, expr):
        # This is where you define the custom operation.
        # For demonstration, let's square the expression and add 1.
        return expr ** 2 + 1


structure = None  # Declare the global variable outside the function


def main():
    global structure, Var1, result, field2, group2, m2, p2, v2, ring2, v1, equation
    while True:
        try:
            operation = input("Enter an operator (+, -, *, /, exit, log, sin, cos, power, sinh, cosh, tanh, asinh, "
                              "acosh, atanh, fact, abs, sqrt, cbrt, log_der, polar, iota, mod, conj, cube_roots_of_unity, "
                              "complex_roots, euler, gaussian, sum_of_cubes, derivative, integral, dot_product, cross_product, "
                              "determinant, angle, complex_add, complex_sub, complex_mul, complex_div, vector_add, vector_sub, "
                              "scalar_mul, poly_add, poly_sub, poly_mul, poly_div, solve_poly, matrix_add, matrix_sub, matrix_mul, "
                              "matrix_div, matrix_trans, matrix_inv, group_order, group_elements, subgroups, group_isomorphisms, "
                              "ring_order, ring_elements, ideals, ring_isomorphisms, field_order, field_elements, subfields, "
                              "field_isomorphisms, trig, hyperbolic, exp, solve, list_generators").strip()

            if operation in ["+", "-", "*", "/"]:
                if operation == "exit":
                    print("Exiting program.")
                    break
                Var1 = safe_input_number("Enter a number: ")
                Var2 = safe_input_number("Enter another number: ")
                result = perform_arthmetic(operation, Var1, Var2)
                print(f"Result: {result}")
            elif operation in ["sqrt", "cbrt", "log_der"]:
                if operation == "exit":
                    print("Exiting the program")
                    break
                expr_input = input("Enter your expression: ").strip()
                expr = safe_sympify(expr_input)

                if expr is None:
                    print("Invalid expression. Please enter a valid mathematical expression.")
                    continue

                if operation == "solve":
                    solutions = solve(expr, x)
                    print(f"Solutions: {solutions}")
                elif operation == "eval":
                    result = expr.evalf()
                    print(f"Result: {result}")
                elif operation == "diff":
                    derivative = diff(expr, x)
                    print(f"Derivative: {derivative}")
                elif operation == "integrate":
                    limits_input = input("Enter limits of integration 'a, b' or press enter for indefinite: ").strip()
                    if limits_input:
                        a_str, b_str = limits_input.split(',')
                        a, b = validate_limits(a_str), validate_limits(b_str)
                        if a is None or b is None:
                            print("Invalid limits. Please enter valid numerical limits.")
                            continue
                        integral = integrate(expr, (x, a, b))
                    else:
                        integral = integrate(expr, x)
                    print(f"Integral: {integral}")
                elif operation == "limit":
                    point_input = input("Enter the point to approach (use 'oo' for infinity): ").strip()
                    point = validate_limits(point_input)
                    if point is None:
                        print("Invalid point. Please enter a valid numerical point or 'oo' for infinity.")
                        continue
                    lim = limit(expr, x, point)
                    print(f"Limit: {lim}")
                elif operation == "series":
                    order_input = input("Enter the order of the series expansion: ").strip()
                    try:
                        order = int(order_input)
                        ser = series(expr, x, 0, order)
                        print(f"Series Expansion: {ser}")
                    except ValueError:
                        print("Invalid order. Please enter a valid integer.")
                else:
                    print("Unsupported operation.")
            elif operation == ["polar", "iota", "mod", "conj", "sqrt", "cbrt", "log_der", "trig", "hyperbolic", "exp",
                               "solve"]:
                if operation == "exit":
                    print("Exiting the program")
                    break
                equation = input("Enter your expression or equation: ").strip()
                try:
                    expr = sympify(equation)
                    perform_operation(operation, expr)
                except SympifyError as e:
                    print(f"Error: {e}. Please enter a valid mathematical expression.")
            elif operation == "mod, conj, sqrt, cbrt":
                if operation == "exit":
                    print("Exiting the program")
                    return
                equation = input("Enter a full equation (e.g., 'x = a + bi' or 'x = a'): ").strip()
                z = parse_complex_input(equation)
                if z is None:
                    return
                operations = {
                    "mod": calculate_mod,
                    "conj": calculate_conj,
                    "sqrt": calculate_sqrt,
                    "cbrt": calculate_cbrt,
                    "log_der": calculate_log_derivative,
                }
                if operation in operations:
                    result = operations[operation](z)
                    print(f"Result: {result}")
                else:
                    print("Unsupported operation or invalid input.")
            # Add more elif statements for new operations here...
            elif operation in ["complex_add", "complex_sub", "complex_mul", "complex_div"]:
                if operation == "exit":
                    print("Exiting the program")
                    break
                equation1 = input("Enter the first complex number equation: ").strip()
                z1 = parse_complex_input(equation1)
                if z1 is None:
                    return

                equation2 = input("Enter the second complex number equation: ").strip()
                z2 = parse_complex_input(equation2)
                if z2 is None:
                    return

                operations = {
                    "complex_add": lambda x, y: x + y,
                    "complex_sub": lambda x, y: x - y,
                    "complex_mul": lambda x, y: x * y,
                    "complex_div": calculate_complex_division,
                }

                if operation in operations:
                    result = operations[operation](z1, z2)
                    print(f"Result: {result}")
            elif operation in ["vector_add", "vector_sub", "scalar_mul", "cross_product", "dot_product"]:
                try:
                    vectors = []
                    num_vectors = 0  # Initialize num_vectors to avoid referenced before assignment error
                    if operation == "scalar_mul":
                        scalar_input = input("Enter the scalar value or a vector for element-wise multiplication: ")
                        scalar = parse_vector_input(
                            scalar_input) if ',' in scalar_input or ' ' in scalar_input else float(scalar_input)
                        num_vectors = int(input("Enter the number of vectors to multiply: "))
                        for i in range(num_vectors):
                            vector_input = input(f"Enter vector {i + 1}: ")
                            vectors.append(parse_vector_input(vector_input))
                        result = calculate_scalar_multiplication(scalar, *vectors, pad_value=0, callback=None)
                    elif operation == "cross_product":
                        num_vectors = int(input("Enter the number of vectors for cross product: "))
                        for i in range(num_vectors):
                            vector_input = input(f"Enter vector {i + 1}: ")
                            vectors.append(parse_vector_input(vector_input))
                        result = calculate_vector_cross_product(*vectors)
                    elif operation == "dot_product":
                        num_vectors = int(input("Enter the number of vectors for dot product: "))
                        if num_vectors < 2:
                            print("Dot product requires exactly two vectors.")
                        else:
                            v1_input = input("Enter the first vector: ")
                            v2_input = input("Enter the second vector: ")
                            v1 = parse_vector_input(v1_input)
                            v2 = parse_vector_input(v2_input)
                            result = calculate_vector_dot_product(v1, v2)
                    elif operation == "vector_add":
                        num_vectors = int(input("Enter the number of vectors: "))
                        for i in range(num_vectors):
                            vector_input = input(f"Enter vector {i + 1}: ")
                            vectors.append(parse_vector_input(vector_input))
                        result = calculate_vector_addition(*vectors)
                    elif operation == "vector_sub":
                        num_vectors = int(input("Enter the number of vectors: "))
                        for i in range(num_vectors):
                            vector_input = input(f"Enter vector {i + 1}: ")
                            vectors.append(parse_vector_input(vector_input))
                        result = calculate_vector_subtraction(*vectors)  # Enhanced subtraction function is called here
                    print(f"Result: {result}")
                except ValueError as e:
                    print(f"Error: {e}")
            elif operation in ["poly_add", "poly_sub", "poly_mul", "poly_div", "solve_poly"]:
                try:
                    operation = input("Enter the operation (add, sub, mul, div): ").strip()
                    variables = input(
                        "Enter the variables used in your polynomials, separated by commas (e.g., 'x,y'): ").strip()

                    # Capture the first polynomial equation
                    equation1 = input("Enter the first polynomial equation (e.g., 2*x^2 + 3*x - 5): ").strip()
                    p1 = parse_input(equation1, variables)

                    # For operations other than solving, capture the second polynomial equation
                    if operation != "solve_poly":
                        equation2 = input("Enter the second polynomial equation: ").strip()
                        p2 = parse_input(equation2, variables)
                    else:
                        p2 = None

                    variables = input(
                        "Enter the variables used in your polynomials, separated by commas (e.g., 'x,y'): ").strip()
                    symbolic_output = input("Do you want symbolic output (yes/no)? ").strip().lower() == 'yes'
                    pretty_print = input("Do you want pretty print output (yes/no)? ").strip().lower() == 'yes'

                    # Dictionary mapping operations to their respective functions
                    operation_funcs = {
                        'add': calculate_polynomial_addition,
                        'sub': calculate_polynomial_subtraction,
                        'mul': calculate_polynomial_multiplication,
                        'div': calculate_polynomial_division
                    }
                    if operation == "solve_poly":
                        equation = input("Enter the polynomial equation to solve (e.g., 2*x^2 + 3*x - 5 = 0): ").strip()
                        # Assuming the RHS is 0 or handled by the function; adjust as necessary
                        polynomial = equation.split('=')[0].strip()

                        # Calling the solve_polynomial_equation function
                        solutions = solve_polynomial_equation(polynomial, variables, pretty_print, symbolic_output)

                        # Format and display the output based on user preferences
                        if pretty_print and symbolic_output:
                            # The pretty print functionality is handled within the function
                            pass  # Nothing more to do since display is called within the function
                        else:
                            print(f"Solutions: {solutions}")
                    else:
                        # Select and call the appropriate function
                        if operation in operation_funcs:
                            func = operation_funcs[operation]
                            result = func(p1, p2, symbolic=symbolic_output, variables=variables,
                                          pretty_print=pretty_print) if p2 else None
                        else:
                            print("Invalid operation specified.")
                            return
                        if pretty_print and symbolic_output:
                            try:
                                from IPython.display import display, Math
                                display(Math(result))
                            except ImportError:
                                print(f"Result (LaTeX): {result}")
                            else:
                                print(f"Result: {result}")
                except ValueError as e:
                    print(f"Error: {e}")
            elif operation in ["matrix_add", "matrix_sub", "matrix_mul", "matrix_div", "matrix_trans",
                               "matrix_inv"]:
                try:
                    # Skip pre-input for matrix_div as it's handled within the function
                    if operation == "matrix_div":
                        calculate_matrix_division()  # Invokes user input within the function
                        continue  # Proceed to the next iteration of the loop

                    # For matrix multiplication, call the enhanced function directly
                    elif operation == "matrix_mul":
                        calculate_matrix_multiplication()  # Directly call the enhanced function
                        continue  # Skip the rest of the loop

                    # For other operations, collect inputs before operation selection
                    m1 = m2 = None
                    if operation in ["matrix_add", "matrix_sub"]:
                        m1 = input_matrix("Enter the first matrix:")
                        m2 = input_matrix("Enter the second matrix:")
                    elif operation in ["matrix_trans", "matrix_inv"]:
                        m1 = input_matrix("Enter the matrix:")

                    # Map operation to function
                    operation_function_map = {
                        "matrix_add": matrix_addition,
                        "matrix_sub": calculate_matrix_subtraction,
                        # "matrix_div" is not listed here as it's already been handled
                        "matrix_trans": calculate_matrix_transpose,
                        "matrix_inv": calculate_matrix_inverse
                    }

                    # Execute operation
                    if operation in operation_function_map:
                        result = operation_function_map[operation](m1) if operation in ["matrix_trans",
                                                                                        "matrix_inv"] else \
                            operation_function_map[operation](m1, m2)
                        pretty_print_matrix(result, "Result Matrix")
                    else:
                        print("Unsupported operation.")

                except Exception as e:
                    print(f"Error: {e}")
            elif operation in ["group_order", "group_elements", "subgroups", "group_isomorphisms", "list_generators"]:
                try:
                    logging.basicConfig(level=logging.INFO)
                    equation1 = input("Enter the first group equation: ")
                    group = parse_group_equation(equation1, delimiter=',', valid_element_pattern=r'^[a-zA-Z0-9]+$')
                    print(f"Parsed group: {group}")
                except ValueError as e:
                    print(e)
                    exit()  # Use exit() instead of return if this code is not inside a function

                if operation == "group_isomorphisms":
                    try:
                        equation2 = input("Enter the second group equation: ")
                        group2 = parse_group_equation(equation2, delimiter=',', valid_element_pattern=r'^[a-zA-Z0-9]+$')
                        print(f"Parsed second group: {group2}")
                    except ValueError as e:
                        print(e)
                        exit()  # Use exit() instead of return if this code is not inside a function

                elif operation == "list_generators":
                    # Define all operation types
                    operation_types = ["addition", "multiplication", "matrix_multiplication", "string_concatenation"]

                    # Iterate over each operation type and perform the corresponding group operation
                    for operation_type in operation_types:
                        try:
                            print(f"Using operation type: {operation_type}")
                            generators = list_all_generators(group, lambda x, y: group_operation(x, y, operation_type))
                            print(f"Generators of the group: {generators}")
                        except ValueError as e:
                            print(e)
                            exit()  # Use exit() instead of return if this code is not inside a function
                        except Exception as e:
                            print(f"Error in list_generators with {operation_type}: {e}")

                try:
                    if operation == "subgroups":
                        analyze_and_visualize_subgroups(group, lambda x, y: group_operation(x, y, operations))
                    elif operation == "group_order":
                        print(
                            f"Result: {calculate_group_order(group)}")  # Assuming this function is integrated as shown earlier
                    elif operation == "group_elements":
                        print(
                            f"Result: {calculate_group_elements(group)}")  # Assuming this function is defined elsewhere
                    elif operation == "group_isomorphisms" and 'group2' in locals():
                        print(
                            f"Result: {calculate_group_isomorphisms(group, group2)}")  # Assuming this function is defined elsewhere
                except Exception as e:
                    print(f"Error: {e}")
            elif operation in ["ring_order", "ring_elements", "ideals", "ring_isomorphisms"]:
                try:
                    equation1 = parse_input(input("Enter the first ring equation: "))
                    # Check if equation is in the format 'R = {a, b, c, ...}'
                    if '=' in equation1 and '{' in equation1 and '}' in equation1:
                        ring = equation1.split('=')[1].strip().replace('{', '').replace('}', '').split(',')
                    else:
                        raise ValueError
                    if operation == "ring_isomorphisms":
                        equation2 = parse_input(input("Enter the second ring equation: "))
                        # Check if equation is in the format 'S = {d, e, f, ...}'
                        if '=' in equation2 and '{' in equation2 and '}' in equation2:
                            ring2 = equation2.split('=')[1].strip().replace('{', '').replace('}', '').split(',')
                        else:
                            raise ValueError
                except ValueError:
                    print(
                        "Invalid equation format. Please enter a valid ring equation in the format 'R = {a, b, c, "
                        "...}'.")
                    return

                if operation == "ring_order":
                    print(f"Result: {calculate_ring_order(ring)}")
                elif operation == "ring_elements":
                    print(f"Result: {calculate_ring_elements(ring)}")
                elif operation == "ideals":
                    print(f"Result: {calculate_ideals(ring)}")
                elif operation == "ring_isomorphisms":
                    print(f"Result: {calculate_ring_isomorphisms(ring, ring2)}")
                elif operation in ["field_order", "field_elements", "subfields", "field_isomorphisms" "exit"]:
                    if operation == "exit":
                        break
                    try:
                        equation1 = parse_input(input("Enter the first field equation: "))
                        # Check if equation is in the format 'F = {a, b, c, ...}'
                        if '=' in equation1 and '{' in equation1 and '}' in equation1:
                            field = equation1.split('=')[1].strip().replace('{', '').replace('}', '').split(',')
                        else:
                            raise ValueError
                        if operation == "field_isomorphisms":
                            equation2 = parse_input(input("Enter the second field equation: "))
                            # Check if equation is in the format 'G = {d, e, f, ...}'
                            if '=' in equation2 and '{' in equation2 and '}' in equation2:
                                field2 = equation2.split('=')[1].strip().replace('{', '').replace('}', '').split(',')
                            else:
                                raise ValueError
                    except ValueError:
                        print(
                            "Invalid equation format. Please enter a valid field equation in the format 'F = {a, b, "
                            "c, ...}'.")
                        return

                    if operation == "field_order":
                        print(f"Result: {calculate_field_order(field)}")
                    elif operation == "field_elements":
                        print(f"Result: {calculate_field_elements(field)}")
                    elif operation == "subfields":
                        print(f"Result: {calculate_subfields(field)}")
                    elif operation == "field_isomorphisms":
                        print(f"Result: {calculate_field_isomorphisms(field, field2)}")
                elif operation == "create_permutation_group":
                    if operation == "exit":
                        break
                    try:
                        equation = parse_input(input("Enter the permutation equation: "))
                        # Check if equation is in the format 'P = (a b c), (d e f), ...'
                        if '=' in equation and '(' in equation and ')' in equation:
                            permutations = equation.split('=')[1].strip().split(',')
                            # Remove parentheses and split by spaces to get individual elements
                            permutations = [p.replace('(', '').replace(')', '').split() for p in permutations]
                        else:
                            raise ValueError
                    except ValueError:
                        print(
                            "Invalid equation format. Please enter a valid permutation equation in the format 'P = (a "
                            "b c), (d e f), ...'.")
                        return

                    print(f"Permutation Group: {create_permutation_group(*permutations)}")
