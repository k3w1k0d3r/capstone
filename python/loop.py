import subprocess
import sys
import time
from getkey import getkey
from sys import exit
from threading import Event, Thread
SHUTDOWN = Event()
def shutdown_on_key():
	while True:
		key_pressed = getkey(blocking=True)
		if key_pressed == 'q':
			SHUTDOWN.set()
			return
t = Thread(target=shutdown_on_key)
t.start()
while not SHUTDOWN.is_set():
	subprocess.check_output([sys.executable, "datacollect.py"])
	stuff = subprocess.check_output([sys.executable, "train.py"])
	print(stuff.decode())
	time.sleep(600)
	stuff = subprocess.check_output([sys.executable, "update_model.py"])
	print(stuff.decode())
	key_pressed = getkey(blocking=False)
	if key_pressed == "q":
		exit()
	time.sleep(1800)
print("bye")
