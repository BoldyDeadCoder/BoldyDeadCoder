import math
import re
import logging
import pandas as pd  # If you plan to return pandas Series
from math import isclose
from decimal import Decimal, getcontext, InvalidOperation
from cmath import sqrt
from typing import List, Union, Tuple
from IPython.display import display, Math

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


def is_element_generator(element, group):
    """
    Checks if the given element is a generator of the group.
    This is a placeholder function; actual implementation depends on the operation and group representation.
    """
    # Implement logic based on the group's operation to check if `element` can generate all other elements
    generated = {element}
    # This is a simplistic and incorrect implementation placeholder.
    # You would need to apply the operation repeatedly to `element` to try to generate all other elements.
    return generated == set(group)


# Group Theory
def calculate_group_order(group):
    """
    Calculates the order of the group and checks if the group is cyclic.

    :param group: List or set representing the group elements
    :return: Tuple containing the order of the group and whether it is cyclic
    """
    group_order = len(group)
    is_cyclic = any(is_element_generator(element, group, None) for element in
                    group)  # Assuming `None` as a placeholder for the actual operation

    return group_order, is_cyclic


def calculate_group_elements(group):
    return list(group.elements)


def calculate_subgroups(group):
    return list(group.subgroups())


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


def calculate_matrix_addition(m1, m2):
    matrix1 = np.array(m1)
    matrix2 = np.array(m2)
    return np.add(matrix1, matrix2).tolist()


def calculate_matrix_subtraction(m1, m2):
    matrix1 = np.array(m1)
    matrix2 = np.array(m2)
    return np.subtract(matrix1, matrix2).tolist()


def calculate_matrix_multiplication(m1, m2):
    matrix1 = np.array(m1)
    matrix2 = np.array(m2)
    return np.dot(matrix1, matrix2).tolist()


def calculate_matrix_division(m1, m2):
    matrix1 = np.array(m1)
    matrix2 = np.array(m2)
    if np.linalg.det(matrix2) != 0:  # Check if the second matrix is invertible
        return np.dot(matrix1, np.linalg.inv(matrix2)).tolist()
    else:
        return "Error: The second matrix is not invertible."


def calculate_matrix_inverse(matrix):
    if np.linalg.det(matrix) != 0:  # Check if the matrix is invertible
        return np.linalg.inv(matrix).tolist()
    else:
        return "Error: The matrix is not invertible."


def calculate_matrix_transpose(matrix):
    return np.transpose(matrix).tolist()


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


def preprocess_equation(equation):
    """
    Preprocess the equation to handle common input formats and issues,
    such as adding explicit multiplication signs and fixing exponentiation notation.
    """
    # Add explicit multiplication between numbers/variables and parentheses or between variable and number
    equation = re.sub(r'(?<=[0-9)])(?=\()', '*(', equation)
    equation = re.sub(r'(?<=[0-9)])(?=[a-zA-Z])', '*', equation)
    equation = re.sub(r'(?<=[a-zA-Z])(?=[0-9(])', '*', equation)

    # Replace caret notation with ** for exponentiation
    equation = equation.replace('^', '**')

    return equation


def parse_input(equation: str, variables: str) -> "Expr":
    """
    Parses a polynomial equation from a string into a SymPy expression, considering the specified variables.
    """
    equation = preprocess_equation(equation)
    # Convert variable names to SymPy symbols
    vars = symbols(variables)
    # Replace '^' with '**' for exponentiation
    equation = equation.replace('^', '**')
    # Parse the equation string to a SymPy expression
    expr = sympify(equation, locals={str(v): v for v in vars})
    return expr


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
                              "field_isomorphisms, trig, hyperbolic, exp, solve").strip()

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
