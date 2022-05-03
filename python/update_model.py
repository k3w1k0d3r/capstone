
import json
import cpp_wrapper
import os
import time
import multiprocessing
with open("../config.json", "r") as f:
        config = json.load(f)
good_num = 0
for i in range(config["eval_batch_size"]):
	pool = multiprocessing.Pool(processes=1)
	good_num+=pool.map(cpp_wrapper.testgame, range(1))[0]
	pool.close()
	time.sleep(10)
print("Success rate: %s"%(good_num/config["eval_batch_size"]))
if(good_num/config["eval_batch_size"]>=config["threshold"]):
	os.system("cp -r -T ../models/temporary/ ../models/current/")
