import cPickle as pickle
import numpy as np
from sklearn.metrics import accuracy_score
import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
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






def generate_cluster_output(network_name, test_data, output_file, one_file, write_to_file, is_LSTM, segment_size =50):
    with open(settings.DATA_PATH+test_data, 'rb') as f:
        (X, y, s_list) = pickle.load(f)

    if one_file == True:
        model = load_model(settings.NET_PATH+network_name+'.h5')

    else :
        json_file = open('../data/nets/cnn_speaker02.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights("../data/nets/cnn_speaker.02h5")

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy', 'categorical_accuracy', ])

    print "Data extraction..."
    X_test, y_test = dg.generate_test_data(X, y, segment_size)
    n_classes = np.amax(y_test)+1
    print "Data extraction done!"
    if is_LSTM == True :
        X_test = X_test.reshape(X_test.shape[0],X_test.shape[3], X_test.shape[2])
    print "Test output..."
    print model.layers[1].output.get_shape()
    print model.layers[2].output.get_shape()
    im_model = Model(input = model.input, output = model.layers[2].output)
    data_out = im_model.predict(X_test)
    da = np.asarray(data_out)
    print da.shape
    with open(settings.CLUSTER_OUTPUT_PATH+output_file, 'wb') as f:
        pickle.dump((da, y_test, s_list), f, -1)


if __name__ == "__main__":
    generate_cluster_output('cluster_train_dropout_500ms_512_100sp_best', 'test_clustering_10.pickle', 'test_cluster_out_10sp_500ms_512_100sp', True, True, True)
    generate_cluster_output('cluster_train_dropout_500ms_512_100sp_best', 'train_clustering_10.pickle', 'train_cluster_out_10sp_500ms_512_100sp', True, True, True)