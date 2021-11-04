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
	V_loss = tf.reduce_sum(tf.square(tf.cast(V_mask, tf.float32)*(tf.cast(y, tf.float32)-predictions)))
	P_loss = tf.reduce_sum(tf.cast(P_mask, tf.float32)*tf.cast(y, tf.float32)*tf.math.log(tf.nn.softmax(predictions*tf.cast(P_mask, tf.float32)))) #change to keras backend
	loss = V_loss+P_loss
	for l in model.layers:
		if hasattr(l, "kernel_regularizer") and l.kernel_regularizer:
			loss+=l.kernel_regularizer(l.kernel)
		if hasattr(l, "bias_regularizer") and l.bias_regularizer:
			loss+=l.bias_regularizer(l.bias)
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
	if(sys.argv[1]=="train"):
		optimizer.apply_gradients(zip(gradients, model.trainable_variables))
	return loss, accuracy
def format_output(loss, accuracy):
	return "%s loss: %s\n%s accuracy: %s\n"%(sys.argv[1], loss, sys.argv[1], accuracy)
x = np.load("data/x.npy")
y = np.load("data/y.npy")
mask = np.load("data/mask.npy")
loss, accuracy = step(x, y, mask)
print(format_output(loss, accuracy))
