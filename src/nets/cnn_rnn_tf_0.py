
# coding: utf-8

# In[ ]:

import os
import json
import time
import numpy as np
import tensorflow as tf

import core.data_gen as dg
import cPickle as pickle

with open('crt_settings.json') as json_settings_file:
    stngs = json.load(json_settings_file)


# In[ ]:

def tf_log_dir():
    current_workdir = os.getcwd()
    tstamp = int(time.time())
    sess_dir_name = 'sess_%s' % tstamp
    dirty_path = os.path.join(current_workdir, stngs['log_dir'], sess_dir_name)
    return os.path.realpath(dirty_path)

# Parse training data to matrices

def create_train_data():
    with open(os.path.join(stngs['cnn']['train_data_path'], stngs['cnn']['train_data_file']), 'rb') as f:
      (X, y, speaker_names) = pickle.load(f)

    X_t, X_v, y_t, y_v = dg.splitter(X, y, 0.125, 8)
    return X_t, y_t, X_v, y_v

# Create data

X_t, y_t, X_v, y_v = create_train_data()
train_gen = dg.batch_generator(X_t, y_t, batch_size=stngs['batch_size'], segment_size=stngs['segment_size'])
val_gen = dg.batch_generator(X_v, y_v, batch_size=stngs['batch_size'], segment_size=stngs['segment_size'])
batches_t = ((X_t.shape[0]+128 -1 )// 128)*128
batches_v = ((X_v.shape[0]+128 -1 )// 128)*128


# In[ ]:

# Create basic net infrastructure

# Placeholders
with tf.name_scope('Placeholders'):
    x_input = tf.placeholder(tf.float32, shape=(None, stngs['frequencies'], stngs['segment_size'], 1))
    out_labels = tf.placeholder(tf.float32, shape=(None, stngs['segment_size']))

with tf.name_scope('Convolution_1'):
    conv1 = tf.layers.conv2d(inputs=x_input, filters=32, kernel_size=[8, 1], padding="same", activation=tf.nn.relu)

with tf.name_scope('MaxPooling_1'):
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[4, 4], strides=[2, 2])

with tf.name_scope('Convolution_2'):
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[8, 1], padding="same", activation=tf.nn.relu)

with tf.name_scope('MaxPooling_2'):
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[4, 4], strides=[2, 2])
    
with tf.name_scope('Reshape'):
    dim1 = int(pool2.shape[3] * pool2.shape[1])
    dim2 = int(pool2.shape[2])
    lstm_input = tf.reshape(pool2, [-1, dim1, dim2])

with tf.name_scope('GRU'):
    x_gru = tf.unstack(lstm_input, lstm_input.get_shape()[1], 1)
    gru_cell = tf.contrib.rnn.GRUCell(stngs['gru']['neurons_number'])
    dense, _ = tf.contrib.rnn.static_rnn(gru_cell, x_gru, dtype='float')
    gru_out = dense[-1]

with tf.name_scope('Softmax'):
    gru_soft_out = tf.layers.dense(inputs=gru_out, units=stngs['total_speakers'], activation=tf.nn.softmax)


# In[ ]:

# Cross entropy and optimizer

with tf.name_scope('Optimizer'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=gru_soft_out, labels=out_labels))
    tf.summary.scalar('loss', cross_entropy)
    optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)


# In[ ]:

# Training

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Tensorboard
tb_merged = tf.summary.merge_all()
tb_saver = tf.train.Saver()
tb_log_dir = tf_log_dir()
tb_train_writer = tf.summary.FileWriter(tb_log_dir, sess.graph)

run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

for step in range(stngs['batch_loops']):
    start_time = time.time()
    
    # Get next batch
    x_b_t, y_b = train_gen.next()
    # Reshape the x_b batch with channel as last dimension
    x_b = np.reshape(x_b_t, [stngs['batch_size'], stngs['frequencies'], stngs['segment_size'], 1])
    
    # Execute training
    feed_dict = { x_input: x_b, out_labels: y_b }
    _, loss_value = sess.run([optimizer, cross_entropy], feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
    
    # define accuracy
    correct_pred = tf.equal(tf.argmax(gru_soft_out, 1), tf.argmax(out_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    sess_acc = sess.run(accuracy, feed_dict={ x_input: x_b, out_labels: y_b }, options=run_options, run_metadata=run_metadata)
    #print("Round %d accuracy: %s" % (step, sess_acc))
    
    if step % 10 == 0:
        tb_summary_str = sess.run(tb_merged, feed_dict={ x_input: x_b, out_labels: y_b })
        tb_train_writer.add_run_metadata(run_metadata, 'step_{:04d}'.format(step))
        tb_train_writer.add_summary(tb_summary_str, step)
        tb_train_writer.flush()
    
    if step % 100 == 0:
        ckpt_file = os.path.join(tb_log_dir, stngs['ckpt_file_pfx'])
        tb_saver.save(sess, ckpt_file, global_step=step)

model_meta_file = os.path.join(tb_log_dir, stngs['model_file_name'])
tb_saver.save(sess, model_meta_file)


# In[ ]:

with open(os.path.join(stngs['gru']['data_path'], stngs['gru']['data_file']), 'rb') as f:
    (raw_x_data, raw_y_data, test_speaker_names) = pickle.load(f)
    test_x_data, test_y_data = dg.generate_test_data(raw_x_data, raw_y_data, segment_size=stngs['segment_size'])

net_output = gru_soft_out.eval(feed_dict={x_input: np.reshape(test_x_data, [196, 128, 100, 1])}, session=sess)
#gru_output = gru_out.eval(feed_dict={x_input: np.reshape(test_x_data, [196, 128, 100, 1]), out_labels: np.reshape(test_y_data[:100,], [-1, 100])}, session=sess)


# In[ ]:

# Write output file for clustering
with open(os.path.join(stngs['cluster_output_path'], stngs['cluster_output_file']), 'wb') as f:
    pickle.dump((test_x_data, test_y_data, test_speaker_names), f, -1)


# In[ ]:



