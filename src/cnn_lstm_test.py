import numpy as np
import cPickle as pickle
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import core.segment_batchiterator as seg_bi
import core.speaker_train_splitter as sts

batch_size = 128
nb_classes = 10
nb_epoch = 12



# Load Training Data
with open('../data/training/TIMIT_extracted/train_data_10_130ms_2.pickle', 'rb') as f:
    (X, y, speaker_names) = pickle.load(f)

# Load test Data
with open('../data/training/TIMIT_extracted/test_data_10_130ms_2.pickle', 'rb') as g:
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
input_shape = (8, spect_height, spect_width)

print X.shape


# convert class vectors to binary class matrices
y = np_utils.to_categorical(y, nb_classes)
yt = np_utils.to_categorical(yt, nb_classes)

#print y
#print yt

model = Sequential()

#layer 1
model.add(Convolution2D(32, 8, 1, activation='relu',
                        border_mode='valid',
                        input_shape=input_shape))
#layer 2
model.add(MaxPooling2D(pool_size=pool_size, strides=(2,2), dim_ordering="th"))
#layer 3
model.add(Convolution2D(64 ,8,1, activation='relu', dim_ordering="th"))
#layer 4
model.add(MaxPooling2D(pool_size=pool_size, strides=(2,2),dim_ordering="th"))
#layer 4
model.add(Dense(100))
model.add(Bidirectional(LSTM(64)))


adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)


model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

#model.fit(X, Y, batch_size=batch_size, nb_epoch=nb_epoch,
#          verbose=1, validation_data=(Xt, yt))
#score = model.evaluate(Xt, yt, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])