import time
import sys
Var1 = int(input('Type here: '))
Var2 = int(input('Type here: '))
# int() means a number or integer, input() means to get the users input.
print("1.add 'add 'in the next line if you want to add")
print("2.add 'minus' in the next line if you want to subtract")
print("3.add 'multiply' in the next line if you want to multiply")
print("4.add 'divide' if in the next line you want to divide")
time.sleep(2)

if Var1 == input():
    print("Here you will put your equations in the calculator.")
elif input("Type here 'add' if you want to automatically add the problem: ") == "add":
    print(int(Var1 + Var2))
elif input("Type here 'minus' if you want to automatically subtract the problem: ") == "minus":
    print(int(Var1 - Var2))
elif input("Type here 'multiply' if you want automatically to multiply the problem: ") == "multiply":
    print(int(Var1 * Var2))
elif input("Type here 'divide' if you want to automatically divide the problem: ") == "divide":
    print(int(Var1 / Var2))
print("if you want to add, multiply, divide, or subtract decimals then only type the decimal problem.")
time.sleep(2)

Var3 = float(input('3.Type here: '))
Var4 = float(input('4.Type here: '))

if Var3 == input():
    print("Here you will put your equations in the calculator.")
elif input("Type here 'add' if you want to automatically add the problem: ") == "add":
    print(float(Var3 + Var4))
elif input("Type here 'subtract' if you want to automatically subtract the problem: ") == "subtract":
    print(float(Var3 - Var4))
elif input("Type here 'multiply' if you want to automatically multiply the problem: ") == "multiply":
    print(float(Var3 * Var4))
elif input("Type here 'divide' if you want to automatically divide the problem: ") == "divide":
    print(float(Var3 / Var4))
print("end of program" + "Goodbye")
sys.exit()
