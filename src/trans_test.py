import numpy as np
import cPickle as pickle
np.random.seed(1337)  # for reproducibility

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, LSTM
from keras.layers.wrappers import Bidirectional
from keras.utils import np_utils
from keras import backend as K
from keras.utils import np_utils
import tensorflow as tf
import numpy
import core.speaker_train_splitter as sts

import core.data_gen as dg
tf.python.control_flow_ops = tf


batch_size = 128
nb_classes = 630
nb_epoch = 12



# Load Training Data
with open('../data/training/TIMIT_extracted/train_data_630.pickle', 'rb') as f:
    (X, y, speaker_names) = pickle.load(f)



X_t, X_v, y_t, y_v = dg.splitter(X, y, 0.125, 8)

print X_t.shape
print y_t.shape
print X_v.shape
print y_v.shape













