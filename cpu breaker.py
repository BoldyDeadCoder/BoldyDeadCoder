import os

def create_and_run_vbs(file_name):
    with open(file_name + ".txt", 'w') as f:
        f.writelines("Do\n")
        f.writelines("Loop\n")

    base = os.path.splitext(file_name + ".txt")[0]
    os.rename(file_name + ".txt", base + ".vbs")
    os.system("Start " + file_name + ".vbs")

create_and_run_vbs("my_file")
create_and_run_vbs("This_file")
create_and_run_vbs("HI")
create_and_run_vbs("keen")
create_and_run_vbs("f_file")
