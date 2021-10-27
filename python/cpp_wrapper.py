import sys
import os
from os import path
import json
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import build.MCTS_exten
def playgame(use_queue=False, temperature=1, path="../models/current/model.pb"):
	with open("../config.json", "r") as f:
		config = json.load(f)
	iterations = config["iterations"]
	thread_count = config["thread_count"]
	epsilon = config["epsilon"]
	Alpha = config["alpha"]
	return build.MCTS_exten.playgame(use_queue=use_queue, temperature=temperature, path=path, iterations=iterations, thread_count=thread_count, epsilon=epsilon, Alpha=Alpha)
