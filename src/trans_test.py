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
import tensorflow as tf
import core.segment_batchiterator as seg_bi
import core.speaker_train_splitter as sts
import core.transform_input as ti
tf.python.control_flow_ops = tf


batch_size = 128
nb_classes = 10
nb_epoch = 12



# Load Training Data
with open('../data/training/TIMIT_extracted/train_data_10_not_clustering_vs_reynolds.pickle', 'rb') as f:
    (X, y, speaker_names) = pickle.load(f)


print X[1,0,:,249]
print X[0,0].shape

ti.transform(X, y)