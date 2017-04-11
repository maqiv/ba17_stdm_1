import numpy as np
import core.settings as settings
import core.plot_saver as ps
import cPickle as pickle
np.random.seed(1337)  # for reproducibility
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import LSTM, GRU
from keras.layers.wrappers import Bidirectional
from keras.utils import np_utils
from keras import backend as K
import tensorflow as tf
import core.data_gen as dg
import analysis.data_analysis as da
import core.pairwise_kl_divergence as kld
#tf.python.control_flow_ops = tf


'''This Class Trains a Bidirectional LSTM with 2 Layers and a Dropout Layer
    Parameters:
    name: Network/experiment name to save artifacts
    training_data: name of the Training data file, expected to be train_xxxx.pickle
    n_hidden1: Units of the first LSTM Layer
    n_hidden2: Units of the second LSTM Layer
    n_hidden3: Units of the third LSTM Layer
    n_hidden4: Units of the fourth LSTM Layer
    n_classes: Amount of output classes (Speakers in Trainingset)
    n_epoch: Number of Epochs to train the Network
    segment_size: Segment size that is used as input 100 equals 1 second with current Spectrogram extraction
    frequency: size of the frequency Dimension of the Input Spectrogram
'''
class bilstm_4layer(object):

    def __init__(self, name, training_data, n_hidden1=128, n_hidden2=128, n_hidden3=128, n_hidden4=128, n_classes=630, n_epoch=1000, segment_size=15, frequency=128 ):
        self.network_name = name
        self.training_data = training_data
        self.test_data = 'test'+training_data[5:]
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_hidden3 = n_hidden3
        self.n_hidden4 = n_hidden4
        self.n_classes = n_classes
        self.n_epoch = n_epoch
        self.segment_size = segment_size
        self.input = (segment_size, frequency)
        print self.network_name
        self.run_network()
    
    def create_net(self):
        model = Sequential()
        model.add(Bidirectional(LSTM(self.n_hidden1, return_sequences=True), input_shape=self.input))
        model.add(Dropout(0.50))
        model.add(Bidirectional(LSTM(self.n_hidden2, return_sequences=True)))
        model.add(Bidirectional(LSTM(self.n_hidden3, return_sequences=True)))
        model.add(Dropout(0.50))
        model.add(Bidirectional(LSTM(self.n_hidden4)))
        model.add(Dense(10*self.n_classes))
        model.add(Dense(self.n_classes))
        model.add(Activation('softmax'))
        adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(loss='kullback_leibler_divergence',
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
        model = self.create_net()
        calls = self.create_callbacks()
        
        X_t, y_t, X_v, y_v = self.create_train_data()
        train_gen = dg.batch_generator_lstm(X_t, y_t, 128, segment_size=self.segment_size)
        val_gen = dg.batch_generator_lstm(X_v, y_v, 128, segment_size=self.segment_size)
        batches_t = ((X_t.shape[0]+128 -1 )// 128)
        batches_v = ((X_v.shape[0]+128 -1 )// 128)
        
        history = model.fit_generator(train_gen, steps_per_epoch = batches_t, epochs = self.n_epoch, 
                    verbose=2, callbacks=calls, validation_data=val_gen, 
                    validation_steps=batches_v, class_weight=None, max_q_size=10, 
                    nb_worker=1, pickle_safe=False)
        ps.save_accuracy_plot(history, self.network_name)
        ps.save_loss_plot(history, self.network_name)
        
        print "saving model"
        model.save(settings.NET_PATH+self.network_name+".h5")
        #da.calculate_test_acccuracies(self.network_name, self.test_data, True, True, True, segment_size=self.segment_size)    





