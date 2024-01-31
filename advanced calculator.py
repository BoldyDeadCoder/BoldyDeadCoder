def calculate():
    try:
        Var1 = int(input('Type here: '))
        Var2 = int(input('Type here: '))
        operation = input("Type here '+, -, *, /': ")
        if operation == '+':
            print(Var1 + Var2)
        elif operation == '-':
            print(Var1 - Var2)
        elif operation == '*':
            print(Var1 * Var2)
        elif operation == '/':
            if Var2 != 0:
                print(Var1 / Var2)
            else:
                print("Error: Division by zero is not allowed.")
        else:
            print("Error: Invalid operator. Please enter '+, -, *, /'.")
    except ValueError:
        print("Error: Invalid input. Please enter a number.")

# Call the function twice to replicate your original code
calculate()
calculate()
