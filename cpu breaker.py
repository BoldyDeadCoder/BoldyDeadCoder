import os

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
