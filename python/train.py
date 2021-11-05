import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from os import path
import sys
import numpy as np
import time
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
model = keras.models.load_model("../models/current/model.h5")
with open("../models/current/is_initial", "r") as f:
	initial = int(f.read())
optimizer = keras.optimizers.RMSprop()
def get_loss(x, y, mask, model, predictions):
	predictions = model(x, training=True)
	V_mask = np.zeros(y.shape)
	for i in range(len(V_mask)):
		V_mask[i][-1] = 1
	P_mask = np.ones(y.shape)
	for i in range(len(P_mask)):
		P_mask[i][-1]*=0
	V_mask = tf.convert_to_tensor(V_mask)
	P_mask = tf.convert_to_tensor(P_mask)
	P_mask = P_mask*mask
	V_loss = K.sum(K.square(K.cast(V_mask, tf.float32)*(K.cast(y, tf.float32)-predictions)))
	P_loss = K.sum(K.cast(P_mask, tf.float32)*K.cast(y, tf.float32)*K.log(K.softmax(predictions*K.cast(P_mask, tf.float32)))) #change to keras backend
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
	return 0
@tf.function
def step(x, y, mask):
	with tf.GradientTape() as tape:
		predictions = model(x, training=True)
		loss = get_loss(x, y, mask, model, predictions)
		accuracy = get_accuracy(y, predictions)
	gradients = tape.gradient(loss, model.trainable_variables)
	optimizer.apply_gradients(zip(gradients, model.trainable_variables))
	return loss, accuracy
def format_output(loss, accuracy):
	return "loss: %s\naccuracy: %s"%(loss.numpy(), accuracy.numpy())
x = np.load("data/x.npy")
y = np.load("data/y.npy")
mask = np.load("data/mask.npy")
loss, accuracy = step(x, y, mask)
print(format_output(loss, accuracy))
