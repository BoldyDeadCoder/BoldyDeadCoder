def union(a, b):
    a = set(map(int, a.split(',')))
    b = set(map(int, b.split(',')))
    return a.union(b)

def intersection(a, b):
    a = set(map(int, a.split(',')))
    b = set(map(int, b.split(',')))
    return a.intersection(b)

def difference(a, b):
    a = set(map(int, a.split(',')))
    b = set(map(int, b.split(',')))
    return a.difference(b)

def symmetric_difference(a, b):
    a = set(map(int, a.split(',')))
    b = set(map(int, b.split(',')))
    return a.symmetric_difference(b)

def calculate(operation, a, b):
    if operation in ["union", "intersection", "difference", "symmetric_difference"]:
        return operation(a, b)
    elif operation in ["calculate_dot_product", "calculate_cross_product"]:
        v1 = list(map(float, a.split(',')))
        v2 = list(map(float, b.split(',')))
        if operation == "calculate_dot_product":
            return calculate_dot_product(v1, v2)
        elif operation == "calculate_cross_product":
            return calculate_cross_product(v1, v2)
    elif operation == "calculate_determinant":
        matrix = [list(map(float, row.split(','))) for row in a.split(';')]
        return calculate_determinant(matrix)
    elif operation in ["complex_add", "complex_sub", "complex_mul", "complex_div"]:
        z1 = validate_complex_input(a)
        z2 = validate_complex_input(b)
        if z1 is None or z2 is None:
            return "Error: Invalid complex number input."
        # Add your complex operations here
    elif operation in ["matrix_add", "matrix_sub", "matrix_mul", "matrix_div"]:
        m1 = [row.split(',') for row in a.split(';')]
        m2 = [row.split(',') for row in b.split(';')]
        m1 = validate_matrix_input(m1)
        m2 = validate_matrix_input(m2)
        if m1 is None or m2 is None:
            return "Error: Invalid matrix input."
        # Add your matrix operations here
    else:
        return "Error: Invalid operation."

def main():
    while True:
        try:
            operation = input("Enter the operation: ")
            a = input("Enter the first input: ")
            b = input("Enter the second input: ")
            result = calculate(operation, a, b)
            print(f"Result: {result}")
        except ValueError:
            print("Error: Invalid input. Please enter valid numbers.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
