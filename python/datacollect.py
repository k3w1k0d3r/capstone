import cpp_wrapper
import numpy as np
import json
import sys
with open("../config.json", "r") as f:
	config = json.load(f)
for i in range(config["batch_size"]):
	result = cpp_wrapper.playgame()
	positions = result[0]
	policies = result[1]
	winner = result[2]
	x_i = np.zeros((positions.shape[0]*2, 19, 19, 2))
	y_i = np.zeros((positions.shape[0]*2, 362))
	mask_i = np.zeros((positions.shape[0]*2, 362))
	for j in range(positions.shape[0]):
		for k in range(19):
			for L in range(19):
				x_i[2*j, k, L, 0] = positions[j, k, L]==-1
				x_i[2*j+1, k, L, 0] = positions[j, k, L]==1
				x_i[2*j, k, L, 1] = positions[j, k, L]==1
				x_i[2*j+1, k, L, 1] = positions[j, k, L]==-1
		y_i[2*j] = np.append(policies[j], 2*(winner==-1)-1)
		y_i[2*j+1] = np.append(policies[j], 2*(winner==1)-1)
		turn = j-1
		if((turn%4==0) or (turn%4==1)):
			mask_i[2*j+1] = np.ones(362)
		else:
			mask_i[2*j] = np.ones(362)
	if(i==0):
		x = np.copy(x_i)
		y = np.copy(y_i)
		mask = np.copy(mask_i)
	else:
		x = np.concatenate((x, x_i))
		y = np.concatenate((y, y_i))
		mask = np.concatenate((mask, mask_i))
np.save("data/x.npy", x)
np.save("data/y.npy", y)
np.save("data/mask.npy", mask)
