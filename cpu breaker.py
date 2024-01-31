import os
import ctypes, sys

def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

if is_admin():
    d
    def create_and_run_vbs(file_name):
    with open(file_name + ".txt", 'w') as f:
        f.writelines("Do\n")
        f.writelines("Loop\n")

    base = os.path.splitext(file_name + ".txt")[0]
    os.rename(file_name + ".txt", base + ".vbs")
    os.system("Start " + file_name + ".vbs")

create_and_run_vbs("im_watching_file")
create_and_run_vbs("This_man_file")
create_and_run_vbs("HI_file")
create_and_run_vbs("Girl!")
create_and_run_vbs("Life_file")
    def create_read_only_file(filename, content):
        with open(filename, 'w') as file:
            file.write(content)
            os.chmod(filename, 0o444)  # Makes the file read-only
            # Usage
create_read_only_file('test.txt', 'This is a test')


    print("I have admin privileges!")
else:
    # Re-run the program with admin rights
    ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)
