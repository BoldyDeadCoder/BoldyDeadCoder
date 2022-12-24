import time
import sys
Var1 = int(input('Type here: '))
Var2 = int(input('Type here: '))
# int() means a number or integer, input() means to get the users input.
print("1.'add' in the next line if you want to add")
adding = input("Type here 'add': ")
if adding == 'add':
    print(Var1 + Var2)
    time.sleep(2)
    print("2.'minus' in the next line if you want to subtract")
    adding = input("Type here 'minus': ")
if adding == 'minus':
    print(Var1 - Var2)
    time.sleep(2)
    print("3.'multiply' in the next line if you want to multiply")
    multiplying = input("Type here 'multiply': ")
if adding == 'multiply':
    print(Var1 * Var2)
    time.sleep(2)
    print("4.'divide' if in the next line you want to divide")
    dividing = input("Type here 'divide'")
if adding == 'divide':
    print(Var1 / Var2)
sys.exit()
