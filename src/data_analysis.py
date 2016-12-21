import cPickle as pickle
import numpy as np
from sklearn.metrics import accuracy_score
import core.settings
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D, LSTM, TimeDistributedDense
from keras.layers.wrappers import Bidirectional
from keras.utils import np_utils
from keras.models import model_from_json
from keras import backend as K
from keras import metrics
import tensorflow as tf
import core.data_gen as dg
import matplotlib.pyplot as plt
import core.settings as settings
from keras.models import load_model
from keras import backend as K
tf.python.control_flow_ops = tf

net_name = "BiLSTM128_l2_150ms_sp630_ep4000"
write_to_file = True


def calculate_test_acccuracies(network_name, one_file, write_to_file):
    with open('../data/training/TIMIT_extracted/test_data_630.pickle', 'rb') as f:
        (X, y, s_list) = pickle.load(f)

    if one_file == True:
        model = load_model('../data/experiments/nets/'+network_name+'.h5')

    else :
        json_file = open('../data/nets/cnn_speaker02.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights("../data/nets/cnn_speaker.02h5")

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy', 'categorical_accuracy', ])

    print "Data extraction..."
    X_test, y_test = dg.generate_test_data(X, y)
    print "Data extraction done!"
    X_t = X_test.reshape(X_test.shape[0],X_test.shape[3], X_test.shape[2])
    print X_t.shape
    print "Test output..."
    output = model.predict(X_t, batch_size=128, verbose= 1)
    y_t = np_utils.to_categorical(y_test, 630)
    eva = model.evaluate(X_t, y_t, batch_size=128, verbose = 2)
    k_nearest2 =  K.eval(metrics.top_k_categorical_accuracy(tf.pack(y_t),tf.pack(output), k=2))
    k_nearest3 =  K.eval(metrics.top_k_categorical_accuracy(tf.pack(y_t),tf.pack(output), k=3))
    k_nearest5 =  K.eval(metrics.top_k_categorical_accuracy(tf.pack(y_t),tf.pack(output), k=5))
    k_nearest10 =  K.eval(metrics.top_k_categorical_accuracy(tf.pack(y_t),tf.pack(output), k=10))
    
    print output.shape
    output_sum = np.zeros((settings.NUM_OF_SPEAKERS, settings.NUM_OF_SPEAKERS))
    output_geom = np.zeros((settings.NUM_OF_SPEAKERS, settings.NUM_OF_SPEAKERS))
    y_pred_max = np.zeros(settings.NUM_OF_SPEAKERS)
    y_pred_median = np.zeros(settings.NUM_OF_SPEAKERS)
    for i in range(settings.NUM_OF_SPEAKERS):
        indices = np.where(y_test == i)[0]
        speaker_output = np.take(output, indices, axis=0)
        max_val = 0
        for o in speaker_output:
            output_sum[i] = np.add(output_sum[i], o)
            output_geom[i] = np.multiply(output_geom[i], o)

            if np.max(o) > max_val:
                max_val = np.max(o)
                y_pred_max[i] = np.argmax(o)
        output_geom[i] = np.power(output_geom[i], 1/len(speaker_output))

    y_pred_mean = np.zeros(settings.NUM_OF_SPEAKERS)
    y_pred_geom = np.zeros(settings.NUM_OF_SPEAKERS)
    for i in range(len(output_sum)):
        y_pred_mean[i] = np.argmax(output_sum[i])
        y_pred_geom[i] = np.argmax(output_sum[i])

    y_correct = np.arange(settings.NUM_OF_SPEAKERS)
    
    print "geometric wrong"
    for j in range(len(y_correct)):
        if y_correct[j] != y_pred_geom[j]:
            print "Speaker: "+str(y_correct[j])+", Pred: "+str(y_pred_geom[j]) 
            ind = np.argpartition(output_sum[j], -5)[-5:]
            print np.argmax(output_sum[j])
            print ind[np.argsort(output_sum[j][ind])]
    
    print "mean wrong"
    for j in range(len(y_correct)):
        if y_correct[j] != y_pred_mean[j]:
            print "Speaker: "+str(y_correct[j])+", Pred: "+str(y_pred_mean[j]) 
            ind = np.argpartition(output_sum[j], -5)[-5:]
            print np.argmax(output_sum[j])
            print ind[np.argsort(output_sum[j][ind])]

    print model.metrics_names        
    print eva
    print "Acc: %.4f" %eva[2]
    print "k2: %.4f" %k_nearest2
    print "k3: %.4f" %k_nearest3
    print "k5: %.4f" %k_nearest5
    print "k10: %.4f" %k_nearest10
    print "Accuracy (Max.): %.4f" % accuracy_score(y_correct, y_pred_max)
    print "Accuracy (Mean): %.4f" % accuracy_score(y_correct, y_pred_mean)
    print "Accuracy (Geom): %.4f" % accuracy_score(y_correct, y_pred_geom)
    if write_to_file == True:   
        with open('../data/experiments/logs/test_scores.txt', 'ab') as f:
            f.write('---------- '+network_name+'---------------\n')
            f.write("Accuracy: %.4f \n" %eva[2])
            f.write("Accuracy (Max.): %.4f \n" % accuracy_score(y_correct, y_pred_max))
            f.write("Accuracy (Mean): %.4f \n" % accuracy_score(y_correct, y_pred_mean))
            f.write("Accuracy (Geom): %.4f \n\n" % accuracy_score(y_correct, y_pred_geom))
        #11 (mit 7) und 54 (mit 49) sind falsch

if __name__ == '__main__':
    calculate_test_acccuracies(net_name, True, True)