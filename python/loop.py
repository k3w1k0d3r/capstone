import subprocess
import sys
for i in range(3):
	subprocess.check_output([sys.executable, "datacollect.py", "train"])
	stuff = subprocess.check_output([sys.executable, "train.py", "train"])
	print(stuff.decode())
	subprocess.check_output([sys.executable, "datacollect.py", "eval"])
	stuff = subprocess.check_output([sys.executable, "train.py", "eval"])
	print(stuff.decode())
