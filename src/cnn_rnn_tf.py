import core.plot_saver as ps
import core.data_gen as dg
import analysis.data_analysis as da

import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

import cPickle as pickle

class cnn_rnn_tf(object):

    def __init__(self, name, training_data, n_hidden1=128, n_hidden2=128, n_classes=630, n_epoch=1000, segment_size=100, frequency=128):
        self.segment_size=segment_size
        self.run_network()

    def create_net(self):
        x_input = tf.placeholder(tf.float32, shape=(None, 128, self.segment_size, 1))
        conv1 = tf.layers.conv2d(inputs=x_input, filters=16, kernel_size=[8, 8], padding="same", activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[4, 4], strides=2)
        conv2 = tf.layers.conv2d(inputs=pool1, filters=32, kernel_size=[6, 6], padding="same", activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[4, 4], strides=2)
        
        # Dense layer with units as total # of speakers * 10
        dense1 = tf.layers.dense(inputs=pool2, units=10*100, activation=tf.nn.relu)

        return dense1


    def create_train_data(self):
        with open('../data/training/TIMIT_extracted/'+self.training_data, 'rb') as f:
          (X, y, speaker_names) = pickle.load(f)
    
        X_t, X_v, y_t, y_v = dg.splitter(X, y, 0.125, 8)
        return X_t, y_t, X_v, y_v

    #def create_callbacks(self):
    #    return [csv_logger, net_saver]

    def run_network(self):
        print "creating net"
        network = self.create_net()
        #calls = self.create_callbacks()
        
        print "loading training data........"
        X_t, y_t, X_v, y_v = self.create_train_data()
        train_gen = dg.batch_generator(X_t, y_t, 128, segment_size=self.segment_size)
        val_gen = dg.batch_generator(X_v, y_v, 128, segment_size=self.segment_size)
        batches_t = ((X_t.shape[0]+128 -1 )// 128)*128
        batches_v = ((X_v.shape[0]+128 -1 )// 128)*128
        
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=y_t))
        
        optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)

        print "data loaded and batchgenerators created"
        print "starting training"
        
        print "training_finished"
        ps.save_accuracy_plot(history, self.network_name)
        ps.save_loss_plot(history, self.network_name)
        print "saving model"
        model.save("../data/experiments/nets/"+self.network_name+".h5")
        print "evaluating model"
        da.calculate_test_acccuracies(self.network_name, self.test_data, True, True, False, segment_size=100)    
