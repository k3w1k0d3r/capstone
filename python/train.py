import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from os import path
import sys
import os
sys.path.append( path.dirname(path.dirname(path.abspath(__file__))))
model = keras.models.load_model("../models/current/model.h5")
os.system("python datacollect.py")
