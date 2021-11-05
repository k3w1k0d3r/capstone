import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import dtypes
from tensorflow.keras.regularizers import l2
from tensorflow.python.tools import optimize_for_inference_lib
from google.protobuf import text_format
import numpy as np
import time
import json
physical_devices = tf.config.list_physical_devices("GPU")
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

with open("../config.json", "r") as f:
	config = json.load(f)
def tail(inp):
	x = Conv2D(64, (3, 3), padding="same", use_bias=False, kernel_regularizer=l2(config["lambda"]))(inp)
	x = BatchNormalization()(x)
	x = ReLU()(x)
	return x
def residual(x):
	y = Conv2D(64, (3, 3), padding="same", use_bias=False, kernel_regularizer=l2(config["lambda"]))(x)
	y = BatchNormalization()(y)
	y = ReLU()(y)
	y = Conv2D(64, (3, 3), padding="same", use_bias=False, kernel_regularizer=l2(config["lambda"]))(y)
	y = BatchNormalization()(y)
	x+=y
	x = ReLU()(x)
	return x
def policy_head(x):
	x = Conv2D(2, (1, 1), use_bias=False, kernel_regularizer=l2(config["lambda"]))(x)
	x = BatchNormalization()(x)
	x = ReLU()(x)
	x = Flatten()(x)
	out = Dense(361, kernel_regularizer=l2(.0001), bias_regularizer=l2(config["lambda"]))(x)
	return out
def value_head(x):
	x = Conv2D(1, (1, 1), use_bias=False, kernel_regularizer=l2(config["lambda"]))(x)
	x = BatchNormalization()(x)
	x = ReLU()(x)
	x = Flatten()(x)
	x = Dense(64, activation="relu", kernel_regularizer=l2(config["lambda"]), bias_regularizer=l2(config["lambda"]))(x)
	out = Dense(1, activation="tanh", kernel_regularizer=l2(config["lambda"]), bias_regularizer=l2(config["lambda"]))(x)
	return out

inp = Input((19, 19, 2), name="input_node")
x = tail(inp)
for i in range(4):
	x = residual(x)
out = Concatenate(name="output_node")([policy_head(x), value_head(x)])
model = Model(inp, out)

@tf.function(input_signature=[tf.TensorSpec(shape=(None, 19, 19, 2), dtype=tf.float32)])
def to_save(x):
	return model(x)
f = to_save.get_concrete_function()
constantGraph = convert_to_constants.convert_variables_to_constants_v2(f)
#tf.io.write_graph(constantGraph.graph.as_graph_def(), "../models/initial", "model.pbtxt")
output_graph_def = optimize_for_inference_lib.optimize_for_inference(constantGraph.graph.as_graph_def(), ["x"], ["model/output_node/concat"], dtypes.float32.as_datatype_enum, False)
graph_io.write_graph(output_graph_def, "../models/initial/", "model.pb", as_text=False)
model.save("../models/initial/model.h5")
os.system("cp -r -T ../models/initial/ ../models/current/")
with open("../models/current/is_initial", "w") as f:
	f.write("1")
