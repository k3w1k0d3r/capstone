import subprocess
import sys
for i in range(1):
	subprocess.check_output([sys.executable, "datacollect.py"])
	stuff = subprocess.check_output([sys.executable, "train.py"])
	print(stuff.decode())
