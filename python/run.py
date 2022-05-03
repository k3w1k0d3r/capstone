import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import json
import sys
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import dtypes
from tensorflow.python.tools import optimize_for_inference_lib
from google.protobuf import text_format
from os import path
import math
from custom_datacollect import data_collect, cpp_wrapper, np, time, multiprocessing, progressbar
from threading import Event, Thread
SHUTDOWN = Event()
def shutdown_on_key():
	while True:
		key_pressed = getkey(blocking=True)
		if key_pressed == "q":
			SHUTDOWN.set()
			return
t = Thread(target=shutdown_on_key)
t.start()
with open("../config.json", "r") as f:
	config = json.load(f)
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
model = keras.models.load_model("../models/current/model.h5")
optimizer = keras.optimizers.RMSprop()
def get_loss(x, y, mask, model, predictions):
	V_mask = np.zeros(y.shape)
	for i in range(len(V_mask)):
		V_mask[i][-1] = 1
	P_mask = np.ones(y.shape)
	for i in range(len(P_mask)):
		P_mask[i][-1]*=0
	V_mask = tf.convert_to_tensor(V_mask)
	P_mask = tf.convert_to_tensor(P_mask)
	P_mask = P_mask*mask
	V_mask = V_mask*mask
	V_loss = K.sum(K.square(K.cast(V_mask, tf.float32)*(K.cast(y, tf.float32)-predictions)))
	P_loss = -K.sum(K.cast(P_mask, tf.float32)*K.cast(y, tf.float32)*K.log(K.softmax(predictions*K.cast(P_mask, tf.float32)))) #change to keras backend
	loss = V_loss+P_loss
	for l in model.layers:
		if hasattr(l, "kernel_regularizer") and l.kernel_regularizer:
			loss+=x.shape[0]*l.kernel_regularizer(l.kernel)
		if hasattr(l, "bias_regularizer") and l.bias_regularizer:
			loss+=x.shape[0]*l.bias_regularizer(l.bias)
	return loss
def get_accuracy(y, predictions):
	V_mask = np.zeros(y.shape)
	for i in range(len(V_mask)):
		V_mask[i][-1] = 1
	V_mask = tf.convert_to_tensor(V_mask)
	return 1-K.sum(K.cast(V_mask, tf.float32)*K.abs(K.round((predictions+1)/2)-(K.cast(y, tf.float32)+1)/2))/y.shape[0]
@tf.function
def step(x, y, mask, config):
	losses = []
	accs = []
	for i in range(x.shape[0]//config["model_batch_size"]):
		with tf.GradientTape() as tape:
			predictions = model(x[i*config["model_batch_size"]:(i+1)*config["model_batch_size"]], training=True)
			loss = get_loss(x[i*config["model_batch_size"]:(i+1)*config["model_batch_size"]], y[i*config["model_batch_size"]:(i+1)*config["model_batch_size"]], mask[i*config["model_batch_size"]:(i+1)*config["model_batch_size"]], model, predictions)
			accuracy = get_accuracy(y[i*config["model_batch_size"]:(i+1)*config["model_batch_size"]], predictions)
			losses.append(loss)
			accs.append(accuracy)
		gradients = tape.gradient(loss, model.trainable_variables)
		optimizer.apply_gradients(zip(gradients, model.trainable_variables))
	return losses, accs
def format_output(loss, accuracy):
    return "loss: %s\naccuracy: %s"%(loss.numpy(), accuracy.numpy())
while(not SHUTDOWN.is_set()):
    print("STARTING NEXT ROUND")
	x, y, mask = data_collect(config["batch_size"])

	for i in range(config["epochs"]):
		losses, accs = step(x, y, mask, config)
    for i in range(len(losses)):
        print(format_output(losses[i], accs[i]))
	@tf.function(input_signature=[tf.TensorSpec(shape=model.layers[0].input_shape[0], dtype=tf.float32)])
	def to_save(x):
		return model(x)
	f = to_save.get_concrete_function()
	constantGraph = convert_to_constants.convert_variables_to_constants_v2(f)
	output_graph_def = optimize_for_inference_lib.optimize_for_inference(constantGraph.graph.as_graph_def(), ["x"], ["model/output_node/concat"], dtypes.float32.as_datatype_enum, False)
	time.sleep(600)

	good_num = 0
    for i in progressbar.progressbar(range(config["eval_batch_size"])):
		pool = multiprocessing.Pool(processes=1)
		good_num+=pool.map(cpp_wrapper.testgame, range(1))[0]
		pool.close()
		time.sleep(10)
    print("Success Rate: %s"%(good_num/config["eval_batch_size"]))
	if(good_num/config["eval_batch_size"]>=config["threshold"]):
		graph_io.write_graph(output_graph_def, "../models/current/", "model.pb", as_text=False)
		model.save("../models/current/model.h5")
	time.sleep(1800)
