import subprocess
import sys
with open("../models/current/is_initial", "r") as f:
	initial = int(f.read())
for i in range(1):
	subprocess.check_output([sys.executable, "datacollect.py"])
	stuff = subprocess.check_output([sys.executable, "train.py"])
	print(stuff.decode())
'''
	if(initial):
		stuff = subprocess.check_output([sys.executable, "update_model1.py"])
	else:
		stuff = subprocess.check_output([sys.executable, "update_model2.py"])
'''
