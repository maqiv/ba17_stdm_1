import os
import csv
import json
import time
import logging
import datetime
import numpy as np
import tensorflow as tf
import core.pairwise_kl_divergence as kld

import core.data_gen as dg
import cPickle as pickle

'''
Generates the Cluster output files, for the provided Tensorflow model.
The GRU out layer name of the Model needs to be defined  in parameter output_layer

'''
output_layer =  'GRU/rnn/gru_cell_22/add:0'
sess = tf.Session()
saver = tf.train.import_meta_graph('/media/sf_patrickgerber/cluster_outputs/model.ckpt-8999.meta')
saver.restore(sess, '/media/sf_patrickgerber/cluster_outputs/model.ckpt-8999')

graph = tf.get_default_graph()

def load_settings(settings_file):
    with open(settings_file) as json_settings_file:
        json_settings = json.load(json_settings_file)
    return json_settings


def initialize_logger(self):
    today_now = datetime.datetime.now()
    date_time = '_{:04d}-{:02d}-{:02d}_{:02d}-{:02d}-{:02d}'.format(
                        today_now.year, today_now.month, today_now.day,
                        today_now.hour, today_now.minute, today_now.second)
    # Python logger
    cnn_rnn_tf_1.logger = logging.getLogger(__name__)
    cnn_rnn_tf_1.logger.setLevel(logging.getLevelName(cnn_rnn_tf_1None['logging']['level']))
    log_file_name = cnn_rnn_tf_1None['logging']['file_name_prefix'] + date_time + '.log'
    log_file_path = os.path.join(cnn_rnn_tf_1None['logging']['file_path'], log_file_name)
    log_file_handler = logging.FileHandler(log_file_path)
    log_file_handler.setLevel(logging.getLevelName(cnn_rnn_tf_1None['logging']['level']))
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file_handler.setFormatter(log_formatter)
    cnn_rnn_tf_1.logger.addHandler(log_file_handler)

today_now = datetime.datetime.now()
date_time = '_{:04d}-{:02d}-{:02d}_{:02d}-{:02d}-{:02d}'.format(
                        today_now.year, today_now.month, today_now.day,
                        today_now.hour, today_now.minute, today_now.second)


gru_out = graph.get_tensor_by_name(output_layer)
x_input = graph.get_tensor_by_name('Placeholders/Placeholder:0')
out_labels = graph.get_tensor_by_name('Placeholders/Placeholder_1:0')
cnn_rnn_tf_1None = load_settings('nets/crt_settings.json')



# Write output file for clustering

with open(os.path.join(cnn_rnn_tf_1None['gru']['data_path'], cnn_rnn_tf_1None['gru']['train_data_file_40sp']), 'rb') as f:
    (train_raw_x_data, raw_y_data, train_speaker_names) = pickle.load(f)
    train_x_data, train_y_data = dg.generate_test_data(train_raw_x_data, raw_y_data, segment_size=cnn_rnn_tf_1None['segment_size'])

train_net_output = gru_out.eval(feed_dict={x_input: np.reshape(train_x_data, [train_x_data.shape[0], train_x_data.shape[2], train_x_data.shape[3], train_x_data.shape[1]])}, session=sess)


with open(os.path.join(cnn_rnn_tf_1None['gru']['data_path'], cnn_rnn_tf_1None['gru']['test_data_file_40sp']), 'rb') as f:
    (test_raw_x_data, raw_y_data, test_speaker_names) = pickle.load(f)
    test_x_data, test_y_data = dg.generate_test_data(test_raw_x_data, raw_y_data, segment_size=cnn_rnn_tf_1None['segment_size'])

test_net_output = gru_out.eval(feed_dict={x_input: np.reshape(test_x_data, [test_x_data.shape[0], test_x_data.shape[2], test_x_data.shape[3], test_x_data.shape[1]])}, session=sess)

with open(os.path.join(cnn_rnn_tf_1None['cluster_output_path'], (cnn_rnn_tf_1None['cluster_output_train_file_40'] + date_time + '.pickle')), 'wb') as f:
    pickle.dump((train_net_output, train_y_data, train_speaker_names), f, -1)

with open(os.path.join(cnn_rnn_tf_1None['cluster_output_path'], (cnn_rnn_tf_1None['cluster_output_test_file_40'] + date_time +  '.pickle')), 'wb') as f:
    pickle.dump((test_net_output, test_y_data, test_speaker_names), f, -1)

with open(os.path.join(cnn_rnn_tf_1None['gru']['data_path'], cnn_rnn_tf_1None['gru']['test_data_file_60sp']), 'rb') as f:
    (test_raw_x_data, raw_y_data, test_speaker_names) = pickle.load(f)
    test_x_data, test_y_data = dg.generate_test_data(test_raw_x_data, raw_y_data, segment_size=cnn_rnn_tf_1None['segment_size'])

test_net_output_60 = gru_out.eval(feed_dict={x_input: np.reshape(test_x_data, [test_x_data.shape[0], test_x_data.shape[2], test_x_data.shape[3], test_x_data.shape[1]])}, session=sess)

with open(os.path.join(cnn_rnn_tf_1None['gru']['data_path'], cnn_rnn_tf_1None['gru']['train_data_file_60sp']), 'rb') as f:
    (train_raw_x_data, raw_y_data, train_speaker_names) = pickle.load(f)
    train_x_data, train_y_data = dg.generate_test_data(train_raw_x_data, raw_y_data, segment_size=cnn_rnn_tf_1None['segment_size'])

train_net_output_60 = gru_out.eval(feed_dict={x_input: np.reshape(train_x_data, [train_x_data.shape[0], train_x_data.shape[2], train_x_data.shape[3], train_x_data.shape[1]])}, session=sess)


with open(os.path.join(cnn_rnn_tf_1None['cluster_output_path'], (cnn_rnn_tf_1None['cluster_output_train_file_60'] + date_time + '.pickle')), 'wb') as f:
    pickle.dump((train_net_output_60, train_y_data, train_speaker_names), f, -1)

with open(os.path.join(cnn_rnn_tf_1None['cluster_output_path'], (cnn_rnn_tf_1None['cluster_output_test_file_60'] + date_time +  '.pickle')), 'wb') as f:
    pickle.dump((test_net_output_60, test_y_data, test_speaker_names), f, -1)




with open(os.path.join(cnn_rnn_tf_1None['gru']['data_path'], cnn_rnn_tf_1None['gru']['test_data_file_80sp']), 'rb') as f:
    (test_raw_x_data, raw_y_data, test_speaker_names) = pickle.load(f)
    test_x_data, test_y_data = dg.generate_test_data(test_raw_x_data, raw_y_data, segment_size=cnn_rnn_tf_1None['segment_size'])

test_net_output_80 = gru_out.eval(feed_dict={x_input: np.reshape(test_x_data, [test_x_data.shape[0], test_x_data.shape[2], test_x_data.shape[3], test_x_data.shape[1]])}, session=sess)


with open(os.path.join(cnn_rnn_tf_1None['gru']['data_path'], cnn_rnn_tf_1None['gru']['train_data_file_80sp']), 'rb') as f:
    (train_raw_x_data, raw_y_data, train_speaker_names) = pickle.load(f)
    train_x_data, train_y_data = dg.generate_test_data(train_raw_x_data, raw_y_data, segment_size=cnn_rnn_tf_1None['segment_size'])

train_net_output_80 = gru_out.eval(feed_dict={x_input: np.reshape(train_x_data, [train_x_data.shape[0], train_x_data.shape[2], train_x_data.shape[3], train_x_data.shape[1]])}, session=sess)



with open(os.path.join(cnn_rnn_tf_1None['cluster_output_path'], (cnn_rnn_tf_1None['cluster_output_train_file_80'] + date_time + '.pickle')), 'wb') as f:
    pickle.dump((train_net_output_80, train_y_data, train_speaker_names), f, -1)

with open(os.path.join(cnn_rnn_tf_1None['cluster_output_path'], (cnn_rnn_tf_1None['cluster_output_test_file_80'] + date_time + '.pickle')), 'wb') as f:
    pickle.dump((test_net_output_80, test_y_data, test_speaker_names), f, -1)


