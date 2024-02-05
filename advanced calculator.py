import math
from sympy import symbols, diff, integrate, FiniteField, galois_group, simplify, expand, solve
from sympy.combinatorics import Permutation, PermutationGroup
import numpy as np


# Group Theory
def calculate_group_order(group):
    return len(group)


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


def create_finite_field(p, n=1):
    return FiniteField(p ** n)


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


def calculate_square_root(num):
    return math.sqrt(num)


def calculate_cube_root(num):
    return num ** (1 / 3)


def calculate_logarithmic_derivative(num):
    return 1 / num


def convert_to_polar_form(complex_num):
    return abs(complex_num), np.angle(complex_num)


def calculate_power_of_iota(n):
    return ['1', 'i', '-1', '-i'][n % 4]


def calculate_modulus(complex_num):
    return abs(complex_num)


def calculate_conjugate(complex_num):
    return complex_num.conjugate()


def sum_of_cubes(n):
    return (n * (n + 1) / 2) ** 2


def calculate_derivative(f, x):
    return diff(f, x)


def calculate_integral(f, x, a, b):
    return integrate(f, (x, a, b))


def calculate_dot_product(v1, v2):
    return np.dot(v1, v2)


def calculate_cross_product(v1, v2):
    return np.cross(v1, v2)


def calculate_determinant(matrix):
    return np.linalg.det(matrix)


def calculate_complex_addition(z1, z2):
    return z1 + z2


def calculate_complex_subtraction(z1, z2):
    return z1 - z2


def calculate_complex_multiplication(z1, z2):
    return z1 * z2


def calculate_complex_division(z1, z2):
    if z2 != 0:
        return z1 / z2
    else:
        return "Error: Division by zero is not allowed."


def calculate_vector_addition(v1, v2):
    return [i + j for i, j in zip(v1, v2)]


def calculate_vector_subtraction(v1, v2):
    return [i - j for i, j in zip(v1, v2)]


def calculate_scalar_multiplication(scalar, vector):
    return [scalar * i for i in vector]


def calculate_polynomial_addition(p1, p2):
    x = symbols('x')  # Define the variable of the polynomial
    polynomial1 = simplify(p1)
    polynomial2 = simplify(p2)
    return simplify(polynomial1 + polynomial2)


def calculate_polynomial_subtraction(p1, p2):
    x = symbols('x')  # Define the variable of the polynomial
    polynomial1 = simplify(p1)
    polynomial2 = simplify(p2)
    return simplify(polynomial1 - polynomial2)


def calculate_polynomial_multiplication(p1, p2):
    x = symbols('x')  # Define the variable of the polynomial
    polynomial1 = expand(p1)
    polynomial2 = expand(p2)
    return expand(polynomial1 * polynomial2)


def calculate_polynomial_division(p1, p2):
    x = symbols('x')  # Define the variable of the polynomial
    polynomial1 = simplify(p1)
    polynomial2 = simplify(p2)
    return simplify(polynomial1 / polynomial2)


def solve_polynomial_equation(polynomial, variable):
    var = symbols(variable)  # Define the variable of the polynomial
    return solve(polynomial, var)


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


def main():
    while True:
        try:
            operation = input("Enter an operator (+, -, *, /, sqrt, cbrt, log_der, polar, iota, mod, conj, "
                              "cube_roots_of_unity, complex_roots, euler, gaussian, sum_of_cubes, derivative, "
                              "integral, dot_product, cross_product, determinant, angle, complex_add, complex_sub, "
                              "complex_mul, complex_div, vector_add, vector_sub, scalar_mul, poly_add, poly_sub, "
                              "poly_mul, poly_div, solve_poly, matrix_add, matrix_sub, matrix_mul, matrix_div, "
                              "matrix_trans, matrix_inv, group_order, group_elements, subgroups, group_isomorphisms, "
                              "ring_order, ring_elements, ideals, ring_isomorphisms, field_order, field_elements, "
                              "subfields, field_isomorphisms): ")

            if operation in ["+", "-", "*", "/"]:
                Var1 = float(input("Enter a number: "))
                Var2 = float(input("Enter another number: "))
                if operation == "+":
                    print(f"Result: {Var1 + Var2}")
                elif operation == "-":
                    print(f"Result: {Var1 - Var2}")
                elif operation == "*":
                    print(f"Result: {Var1 * Var2}")
                elif operation == "/":
                    if Var2 != 0:
                        print(f"Result: {Var1 / Var2}")
                    else:
                        print("Error: Division by zero is not allowed.")
            elif operation in ["sqrt", "cbrt", "log_der"]:
                Var1 = float(input("Enter a number: "))
                if operation == "sqrt":
                    print(f"Square root of {Var1}: {calculate_square_root(Var1):.4f}")
                elif operation == "cbrt":
                    print(f"Cube root of {Var1}: {calculate_cube_root(Var1):.4f}")
                elif operation == "log_der":
                    print(f"Derivative of ln({Var1}) at {Var1}: {calculate_logarithmic_derivative(Var1):.4f}")
            elif operation in ["polar", "iota", "mod", "conj"]:
                Var1 = float(input("Enter the real part of the complex number: "))
                Var2 = float(input("Enter the imaginary part of the complex number: "))
                complex_number = complex(Var1, Var2)
                if operation == "polar":
                    magnitude, argument = convert_to_polar_form(complex_number)
                    print(
                        f"Polar form of {complex_number}: Magnitude = {magnitude:.4f}, Argument = {argument:.4f} radians")
                elif operation == "iota":
                    n = int(input("Enter an integer for the power of iota: "))
                    print(f"i^{n} = {calculate_power_of_iota(n)}")
                elif operation == "mod":
                    print(f"Modulus of {complex_number}: {calculate_modulus(complex_number):.4f}")
                elif operation == "conj":
                    print(f"Conjugate of {complex_number}: {calculate_conjugate(complex_number)}")
            # Add more elif statements for the new operations here...
            elif operation in ["complex_add", "complex_sub", "complex_mul", "complex_div"]:
                z1 = complex(input("Enter the first complex number: "))
                z2 = complex(input("Enter the second complex number: "))
                if operation == "complex_add":
                    print(f"Result: {calculate_complex_addition(z1, z2)}")
                elif operation == "complex_sub":
                    print(f"Result: {calculate_complex_subtraction(z1, z2)}")
                elif operation == "complex_mul":
                    print(f"Result: {calculate_complex_multiplication(z1, z2)}")
                elif operation == "complex_div":
                    print(f"Result: {calculate_complex_division(z1, z2)}")
            elif operation in ["vector_add", "vector_sub", "scalar_mul"]:
                v1 = list(map(float, input("Enter the first vector (comma-separated): ").split(',')))
                v2 = list(map(float, input("Enter the second vector (comma-separated): ").split(',')))
                if operation == "vector_add":
                    print(f"Result: {calculate_vector_addition(v1, v2)}")
                elif operation == "vector_sub":
                    print(f"Result: {calculate_vector_subtraction(v1, v2)}")
                elif operation == "scalar_mul":
                    scalar = float(input("Enter the scalar: "))
                    print(f"Result: {calculate_scalar_multiplication(scalar, v1)}")
            elif operation in ["poly_add", "poly_sub", "poly_mul", "poly_div", "solve_poly"]:
                p1 = input("Enter the first polynomial: ")
                p2 = input("Enter the second polynomial: ")
                if operation == "poly_add":
                    print(f"Result: {calculate_polynomial_addition(p1, p2)}")
                elif operation == "poly_sub":
                    print(f"Result: {calculate_polynomial_subtraction(p1, p2)}")
                elif operation == "poly_mul":
                    print(f"Result: {calculate_polynomial_multiplication(p1, p2)}")
                elif operation == "poly_div":
                    print(f"Result: {calculate_polynomial_division(p1, p2)}")
                elif operation == "solve_poly":
                    variable = symbols(input("Enter the variable to solve for: "))
                    print(f"Result: {solve_polynomial_equation(p1, variable)}")
            elif operation in ["matrix_add", "matrix_sub", "matrix_mul", "matrix_div", "matrix_trans", "matrix_inv"]:
                m1 = [[float(j) for j in i.split(',')] for i in input(
                    "Enter the first matrix (rows separated by semicolons, elements within a row separated by commas): ").split(
                    ';')]
                m2 = [[float(j) for j in i.split(',')] for i in input(
                    "Enter the second matrix (rows separated by semicolons, elements within a row separated by "
                    "commas): ").split(
                    ';')]
                if operation == "matrix_add":
                    print(f"Result: {calculate_matrix_addition(m1, m2)}")
                elif operation == "matrix_sub":
                    print(f"Result: {calculate_matrix_subtraction(m1, m2)}")
                elif operation == "matrix_mul":
                    print(f"Result: {calculate_matrix_multiplication(m1, m2)}")
                elif operation == "matrix_div":
                    print(f"Result: {calculate_matrix_division(m1, m2)}")
                elif operation == "matrix_trans":
                    print(f"Result: {calculate_matrix_transpose(m1)}")
                elif operation == "matrix_inv":
                    print(f"Result: {calculate_matrix_inverse(m1)}")
            elif operation in ["group_order", "group_elements", "subgroups", "group_isomorphisms"]:
                group = input("Enter the group: ")
                if operation == "group_order":
                    print(f"Result: {calculate_group_order(group)}")
                elif operation == "group_elements":
                    print(f"Result: {calculate_group_elements(group)}")
                elif operation == "subgroups":
                    print(f"Result: {calculate_subgroups(group)}")
                elif operation == "group_isomorphisms":
                    group2 = input("Enter the second group: ")
                    print(f"Result: {calculate_group_isomorphisms(group, group2)}")
            elif operation in ["ring_order", "ring_elements", "ideals", "ring_isomorphisms"]:
                ring = input("Enter the ring: ")
                if operation == "ring_order":
                    print(f"Result: {calculate_ring_order(ring)}")
                elif operation == "ring_elements":
                    print(f"Result: {calculate_ring_elements(ring)}")
                elif operation == "ideals":
                    print(f"Result: {calculate_ideals(ring)}")
                elif operation == "ring_isomorphisms":
                    ring2 = input("Enter the second ring: ")
                    print(f"Result: {calculate_ring_isomorphisms(ring, ring2)}")
            elif operation in ["field_order", "field_elements", "subfields", "field_isomorphisms"]:
                field = input("Enter the field: ")
                if operation == "field_order":
                    print(f"Result: {calculate_field_order(field)}")
                elif operation == "field_elements":
                    print(f"Result: {calculate_field_elements(field)}")
                elif operation == "subfields":
                    print(f"Result: {calculate_subfields(field)}")
                elif operation == "field_isomorphisms":
                    field2 = input("Enter the second field: ")
                    print(f"Result: {calculate_field_isomorphisms(field, field2)}")
            else:
                print("Error: Invalid operator. Please enter valid options.")
        except ValueError:
            print("Error: Invalid input. Please enter valid numbers.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
