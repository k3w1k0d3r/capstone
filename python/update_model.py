import json
import cpp_wrapper
import os
with open("../config.json", "r") as f:
        config = json.load(f)
good_num = 0
for i in range(config["eval_batch_size"]):
	good_num+=cpp_wrapper.testgame()
print("Success rate: %s"%(good_num/config["eval_batch_size"]))
if(good_num/config["eval_batch_size"]>=config["threshold"]):
	os.system("cp -r -T ../models/temporary/ ../models/current/")
