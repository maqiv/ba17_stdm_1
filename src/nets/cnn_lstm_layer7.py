import core.plot_saver as ps
import core.data_gen as dg
import analysis.data_analysis as da
import numpy as np
import cPickle as pickle
np.random.seed(1337)  # for reproducibility
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D, LSTM, GRU
from keras.layers.wrappers import Bidirectional
from keras.utils import np_utils
from keras import backend as K
import tensorflow as tf
tf.python.control_flow_ops = tf


'''This Class Trains a Bidirectional LSTM with 2 Layers and a Dropout Layer
    Parameters:
    name: Network/experiment name to save artifacts
    training_data: name of the Training data file, expected to be train_xxxx.pickle
    pool_size: Kernel Size of the Pooling layers
    strides: Strides of the Pooling layer
    n_hidden1: Units of the first LSTM Layer
    n_hidden2: Units of the second LSTM Layer
    n_classes: Amount of output classes (Speakers in Trainingset)
    n_epoch: Number of Epochs to train the Network
    input: Input format for the CNN-LSTM (FrequncyXTime)
'''

class cnn_lstm_layer7(object):
    def __init__(self, name, training_data, pool_size =(4,4), strides=(2,2), n_hidden_lstm1 = 128, n_hidden_lstm2 =128, n_classes = 100, n_epoch = 1000, input =(128, 100) ):
        self.network_name = name
        self.training_data = training_data
        self.test_data = 'test'+training_data[5:]
        self.pool_size = pool_size
        self.strides = strides
        self.n_hidden_lstm1 = n_hidden_lstm1
        self.n_hidden_lstm2 = n_hidden_lstm2
        self.n_classes = n_classes
        self.n_epoch = n_epoch
        self.input = (1, input[0], input[1])
        self.segment_size = input[1]
        self.run_network()


    def create_net(self):
        model = Sequential()
        #layer 1
        model.add(Convolution2D(32, 8, 1, activation='relu', border_mode='valid', input_shape=self.input, dim_ordering="th"))
        #layer 2
        model.add(MaxPooling2D(pool_size=self.pool_size, strides=self.strides, dim_ordering="th"))
        #layer 3
        model.add(Convolution2D(64 ,8,1, activation='relu', dim_ordering="th"))
        #layer 4
        model.add(MaxPooling2D(pool_size=self.pool_size, strides=self.strides,dim_ordering="th"))
        #layer 5
        model.add(Flatten())
        model.add(Dense(self.n_classes*10))
        model.add(Activation('relu'))
        #Layer 6
        model.add(Dropout(50))
        #Layer 7
        model.add(Dense((10*self.n_classes)/2))
        model.add(Activation('relu'))
        #LSTM
        model.add(Reshape(((10*self.n_classes/2), 1)))
        model.add(Bidirectional(LSTM(self.n_hidden_lstm1,return_sequences=True)))
        model.add(Bidirectional(LSTM(self.n_hidden_lstm2)))
        model.add(Dense(self.n_classes))
        model.add(Activation('softmax'))
        adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(loss='categorical_crossentropy',
                      optimizer=adam,
                      metrics=['accuracy'])
        return model


    def create_train_data(self):
        with open('../data/training/TIMIT_extracted/'+self.training_data, 'rb') as f:
          (X, y, speaker_names) = pickle.load(f)
    
        X_t, X_v, y_t, y_v = dg.splitter(X, y, 0.125, 8)
        return X_t, y_t, X_v, y_v

    def create_callbacks(self):
        csv_logger = keras.callbacks.CSVLogger('../data/experiments/logs/'+self.network_name+'.csv')
        net_saver = keras.callbacks.ModelCheckpoint("../data/experiments/nets/"+self.network_name+"_best.h5", monitor='val_loss', verbose=1, save_best_only=True)
        return [csv_logger, net_saver]

    def run_network(self):
        print "creating net"
        model = self.create_net()
        calls = self.create_callbacks()
        
        print "loading training data........"
        X_t, y_t, X_v, y_v = self.create_train_data()
        train_gen = dg.batch_generator(X_t, y_t, 128, segment_size=self.segment_size)
        val_gen = dg.batch_generator(X_v, y_v, 128, segment_size=self.segment_size)
        batches_t = ((X_t.shape[0]+128 -1 )// 128)*128
        batches_v = ((X_v.shape[0]+128 -1 )// 128)*128
        print "data loaded and batchgenerators created"
        print "starting training"
        history = model.fit_generator(train_gen, batches_t, self.n_epoch, 
                    verbose=2, callbacks=calls, validation_data=val_gen, 
                    nb_val_samples=batches_v, class_weight=None, max_q_size=10, 
                    nb_worker=1, pickle_safe=False)
        print "training_finished"
        ps.save_accuracy_plot(history, self.network_name)
        ps.save_loss_plot(history, self.network_name)
        print "saving model"
        model.save("../data/experiments/nets/"+self.network_name+".h5")
        print "evaluating model"
        da.calculate_test_acccuracies(self.network_name, self.test_data, True, True, False, segment_size=100)    