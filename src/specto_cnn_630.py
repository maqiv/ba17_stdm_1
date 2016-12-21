import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import matplotlib.pyplot as plt
np.random.seed(1337)  # for reproducibility

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D, LSTM, TimeDistributedDense
from keras.layers.wrappers import Bidirectional
from keras.utils import np_utils
from keras import backend as K
import core.data_gen as dg
import tensorflow as tf
tf.python.control_flow_ops = tf


batch_size = 128
nb_classes = 630
nb_epoch = 12
n_input = 128 # spect input (128*100)
n_steps = 100 # timesteps


# Load Training Data
# Load Training Data
with open('../data/training/TIMIT_extracted/train_data_630.pickle', 'rb') as f:
    (X, y, speaker_names) = pickle.load(f)

# Load test Data
with open('../data/training/TIMIT_extracted/test_data_630.pickle', 'rb') as g:
    (Xt, yt, speaker_names_t) = pickle.load(g)

# input image dimensions
spect_height, spect_width = 128, 100
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (4, 4)
# convolution kernel size
kernel_size = (3, 3)
strides = (2,2)
#input shape
input_shape = (1, spect_height, spect_width)

print X.shape


# convert class vectors to binary class matrices
#y = np_utils.to_categorical(y, nb_classes)
#yt = np_utils.to_categorical(yt, nb_classes)

#print y
#print yt

model = Sequential()

#layer 1
model.add(Convolution2D(32, 8, 1, activation='relu',
                        border_mode='valid',
                        input_shape=input_shape, dim_ordering="th"))
model.add(Activation('relu'))
#layer 2
model.add(MaxPooling2D(pool_size=pool_size, strides=strides, dim_ordering="th"))
#layer 3
model.add(Convolution2D(64 ,8,1, activation='relu', dim_ordering="th"))
model.add(Activation('relu'))
model.add(Dropout(0.5))
#layer 4
model.add(MaxPooling2D(pool_size=pool_size, strides=strides,dim_ordering="th"))
#layer 5
model.add(Flatten())
model.add(Dense(nb_classes*10))
model.add(Activation('relu'))
model.add(Dense(nb_classes*10/2))
model.add(Activaiton('relu'))




#model.add(Dropout(0.50))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
get_3rd_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[3].output])


print model.layers[5]
print model.layers[5].output
# output in test mode = 0
#layer_output = get_3rd_layer_output([X, 0])[0]
#print layer_output.shape
# output in train mode = 1
#layer_output = get_3rd_layer_output([X, 1])[0]
#print layer_output.shape

adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)


model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
batch_iterator_train = seg_bi.SegmentBatchIterator(batch_size=128)
batch_iterator_test = seg_bi.SegmentBatchIterator(batch_size=128)





Xd, yd = dg.createData(X, y, 5)
Yd = np_utils.to_categorical(yd, nb_classes)
#train model
history = model.fit(Xd, Yd, batch_size=128, nb_epoch=50, verbose=2, validation_split=0.2, shuffle=True)

#plot(model, to_file='model.png')


sav = "keras_cnn_test02.png"
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_acc', 'val_acc'], loc='upper left')
plt.savefig(sav)

#Save Model and Weights
model_json = model.to_json()
with open('../data/nets/cnn_speaker2.json' ,'w') as json_file:
    json_file.write(model_json)
model.save_weights("../data/nets/cnn_speaker2.h5")
print "Net saved"