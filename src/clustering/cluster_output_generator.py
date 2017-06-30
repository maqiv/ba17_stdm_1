import os
import csv
import sys
import json
import time
import logging
import datetime
import numpy as np
import tensorflow as tf

import core.data_gen as dg
import cPickle as pickle

CKPT_FILE = '../data/experiments/logs/sess_' + sys.argv[2] + '/final_model.save'
SPKR_PCKL = 'train_speakers_5_clustering_vs_reynolds_v3'
SPKR_FILE = '../data/training/TIMIT_extracted/' + SPKR_PCKL + '.pickle'
CLST_OUTP = '../data/experiments/cluster_outputs'

def load_model(tf_session, ckpt_file):
    print ckpt_file
    saver = tf.train.import_meta_graph(ckpt_file + '.meta')
    saver.restore(tf_session, ckpt_file)
    graph = tf.get_default_graph()
    x_input = graph.get_tensor_by_name('Placeholders/Placeholder:0')
    gru_out = graph.get_tensor_by_name('GRU/rnn/gru_cell_22/add:0')
    return x_input, gru_out

if __name__ == '__main__':
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    x_input, model_out = load_model(sess, CKPT_FILE)

    # Evaluate the network
    with open(SPKR_FILE, 'rb') as f:
        (raw_x_data, raw_y_data, speaker_names) = pickle.load(f)
        print raw_x_data.shape
        x_data, y_data = dg.generate_test_data(raw_x_data, raw_y_data, segment_size=100)

    print x_data.shape
    fdict = {x_input: np.reshape(x_data, [x_data.shape[0],
                                          x_data.shape[2],
                                          x_data.shape[3],
                                          x_data.shape[1]])}
    net_output = model_out.eval(feed_dict=fdict, session=sess)
    
    # Write output file for clustering
    with open(os.path.join(CLST_OUTP, (sys.argv[2] + '_' + SPKR_PCKL + '.pickle')), 'wb') as f:
        pickle.dump((net_output, y_data, speaker_names), f, -1)
    
    sess.close()










