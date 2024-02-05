import cmath
import math
import sympy as sp
from sympy import symbols, Matrix


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


def calculate_dot_product(v1, v2):
    """
    Computes the dot product of two vectors.
    """
    return sum(i * j for i, j in zip(v1, v2))


def calculate_determinant(matrix):
    """
    Computes the determinant of a matrix.
    """
    return Matrix(matrix).det()


def calculate_cross_product(v1, v2):
    """
    Computes the cross product of two vectors.
    """
    return [v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]]


def calculate_integral(f, x, a, b):
    """
    Computes the definite integral of a function from a to b.
    """
    x = sp.symbols('x')
    f = sp.sympify(f)
    integral = sp.integrate(f, (x, a, b))
    return integral


def solve_complex_argument(z):
    """
    Calculates the argument (phase) of a complex number.
    """
    return cmath.phase(z)


def calculate_square_root(x):
    """
    Computes the square root of a non-negative number.
    """
    return math.sqrt(x)


def calculate_cube_root(x):
    """
    Computes the cube root of a number.
    """
    return x ** (1 / 3)


def calculate_logarithmic_derivative(x):
    """
    Computes the derivative of the natural logarithm function at a given point.
    """
    return 1 / x


def convert_to_polar_form(z):
    """
    Converts a complex number to polar form (magnitude and argument).
    """
    magnitude = abs(z)
    argument = cmath.phase(z)
    return magnitude, argument


def calculate_power_of_iota(n):
    """
    Computes the value of i raised to the power of n.
    """
    return cmath.exp(1j * n * math.pi / 2)


def calculate_modulus(z):
    """
    Computes the modulus (magnitude) of a complex number.
    """
    return abs(z)


def calculate_conjugate(z):
    """
    Computes the conjugate of a complex number.
    """
    return z.conjugate()


def sum_of_cubes(n):
    """
    Computes the sum of the cubes of the first n natural numbers.
    """
    return (n * (n + 1) // 2) ** 2


def calculate_derivative(f, x):
    """
    Computes the derivative of a function at a given point.
    """
    x = sp.symbols('x')
    f = sp.sympify(f)
    derivative = sp.diff(f, x)
    return derivative


def main():
    while True:
        try:
            Var1 = float(input("Enter a number: "))
            Var2 = float(input("Enter another number: "))
            operation = input("Enter an operator (+, -, *, /, sqrt, cbrt, log_der, polar, iota, mod, conj, "
                              "cube_roots_of_unity, complex_roots, euler, gaussian, sum_of_cubes, derivative, "
                              "integral, dot_product, cross_product, determinant, angle): ")

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
            elif operation == "sqrt":
                print(f"Square root of {Var1}: {calculate_square_root(Var1):.4f}")
            elif operation == "cbrt":
                print(f"Cube root of {Var1}: {calculate_cube_root(Var1):.4f}")
            elif operation == "log_der":
                print(f"Derivative of ln({Var1}) at {Var1}: {calculate_logarithmic_derivative(Var1):.4f}")
            elif operation == "polar":
                complex_number = complex(Var1, Var2)
                magnitude, argument = convert_to_polar_form(complex_number)
                print(f"Polar form of {complex_number}: Magnitude = {magnitude:.4f}, Argument = {argument:.4f} radians")
            elif operation == "iota":
                n = int(input("Enter an integer for the power of iota: "))
                print(f"i^{n} = {calculate_power_of_iota(n)}")
            elif operation == "mod":
                complex_number = complex(Var1, Var2)
                print(f"Modulus of {complex_number}: {calculate_modulus(complex_number):.4f}")
            elif operation == "conj":
                complex_number = complex(Var1, Var2)
                print(f"Conjugate of {complex_number}: {calculate_conjugate(complex_number)}")
            elif operation == "sum_of_cubes":
                print(f"Sum of the cubes of the first {int(Var1)} natural numbers: {sum_of_cubes(int(Var1))}")
            elif operation == "derivative":
                f = input("Enter a function: ")
                x = symbols('x')
                print(f"Derivative of {f} at {Var1}: {calculate_derivative(f, x).subs(x, Var1)}")
            elif operation == "integral":
                f = input("Enter a function: ")
                a = float(input("Enter the lower limit of integration: "))
                b = float(input("Enter the upper limit of integration: "))
                x = symbols('x')
                print(f"Definite integral of {f} from {a} to {b}: {calculate_integral(f, x, a, b)}")
            elif operation == "dot_product":
                v1 = [float(i) for i in input("Enter the first vector (comma-separated): ").split(',')]
                v2 = [float(i) for i in input("Enter the second vector (comma-separated): ").split(',')]
                print(f"Dot product of {v1} and {v2}: {calculate_dot_product(v1, v2)}")
            elif operation == "cross_product":
                v1 = [float(i) for i in input("Enter the first vector (comma-separated): ").split(',')]
                v2 = [float(i) for i in input("Enter the second vector (comma-separated): ").split(',')]
                print(f"Cross product of {v1} and {v2}: {calculate_cross_product(v1, v2)}")
            elif operation == "determinant":
                matrix = [[float(j) for j in i.split(',')] for i in input(
                    "Enter the matrix (rows separated by semicolons, elements within a row separated by commas): ").split(
                    ';')]
                print(f"Determinant of the matrix {matrix}: {calculate_determinant(matrix)}")
            elif operation == "angle":
                while True:
                    try:
                        line1 = [tuple(map(float,
                                           input("Enter the first point of the first line (comma-separated): ").split(
                                               ','))),
                                 tuple(map(float,
                                           input("Enter the second point of the first line (comma-separated): ").split(
                                               ',')))]
                        line2 = [tuple(map(float,
                                           input("Enter the first point of the second line (comma-separated): ").split(
                                               ','))),
                                 tuple(map(float,
                                           input("Enter the second point of the second line (comma-separated): ").split(
                                               ',')))]
                        print(f"Angle between the lines: {calculate_angle(line1, line2)}")
                        break
                    except ValueError:
                        print("Error: Invalid input. Please enter valid numbers in the format 'x,y'.")
            else:
                print("Error: Invalid operator. Please enter valid options.")
        except ValueError:
            print("Error: Invalid input. Please enter valid numbers.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
