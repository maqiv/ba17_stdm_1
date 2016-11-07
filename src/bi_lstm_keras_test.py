import numpy as np
import cPickle as pickle
np.random.seed(1337)  # for reproducibility

import keras
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, LSTM
from keras.layers.wrappers import Bidirectional
from keras.utils import np_utils
from keras import backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
import core.segment_batchiterator as seg_bi
import core.speaker_train_splitter as sts
tf.python.control_flow_ops = tf


batch_size = 128
nb_classes = 40
nb_epoch = 12
n_input = 128 # spect input (128*100)
n_steps = 100 # timesteps



# Load Training Data
with open('../data/training/TIMIT_extracted/train_data_40.pickle', 'rb') as f:
    (X, y, speaker_names) = pickle.load(f)

# Load test Data
with open('../data/training/TIMIT_extracted/test_data_40.pickle', 'rb') as g:
    (Xt, yt, speaker_names_t) = pickle.load(g)

# input image dimensions
spect_height, spect_width = 128, 100
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (4, 4)
# convolution kernel size
kernel_size = (3, 3)
#input shape
input_shape = (spect_height, spect_width)

print X.shape


# convert class vectors to binary class matrices
#y = np_utils.to_categorical(y, nb_classes)
#yt = np_utils.to_categorical(yt, nb_classes)

#print y
#print yt

model = Sequential()

# LSTM LAyer
#model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(Bidirectional(LSTM(128),input_shape = (100, 128)))
model.add(Dense(40))
model.add(Activation('softmax'))
#get_3rd_layer_output = K.function([model.layers[0].input, K.learning_phase()],
 #                                 [model.layers[3].output])

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

def  transformy(y):
    yn = np.zeros((batch_size, nb_classes))
    k = 0
    for v in y:
      #print v
        yn[k][v] =1
        k +=1
    return yn

def generator():
  while 1:
    count = 1;
    for batch_x, batch_y in batch_iterator_train(X, y):
      batch_y = transformy(batch_y)
      batch_x = batch_x.reshape((batch_size, n_steps, n_input))
      count += 1
      yield (batch_x, batch_y)

def valGenerator():
  while 1:
    count = 1;
    for batch_xt, batch_yt in batch_iterator_test(Xt, yt):
      batch_yt = transformy(batch_yt)
      batch_xt = batch_xt.reshape((batch_size, n_steps, n_input))
      count += 1
      yield (batch_xt, batch_yt)



#plot(model, to_file='model.png')


history = model.fit_generator(generator(), 128, 1000, 
              verbose=1, callbacks=[], validation_data=valGenerator(), 
              nb_val_samples=128, class_weight=None, max_q_size=10, 
              nb_worker=1, pickle_safe=False)

sav = "keras_biLSTM_1Layer_128_40sp.png"
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_acc', 'val_acc'], loc='upper left')
plt.savefig(sav)
#plt.show()


#history = model.fit(X, y, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(Xt, yt))
#score = model.evaluate(Xt, yt, verbose=1)
#print('Test score:', score[0])
#print('Test accuracy:', score[1])