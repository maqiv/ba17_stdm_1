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

class cnn_rnn_tf_0(object):
    
    stngs = None
    logger = None
    date_time = None
    
    def __init__(self, network_settings_file):
        cnn_rnn_tf_0.stngs = self.load_settings(network_settings_file)
        self.initialize_logger()
        cnn_rnn_tf_0.logger.info("Calling run_network()")
        self.run_network()
    
    def initialize_logger(self):
        today_now = datetime.datetime.now()
        self.date_time = '_{:04d}-{:02d}-{:02d}_{:02d}-{:02d}-{:02d}'.format(
                            today_now.year, today_now.month, today_now.day,
                            today_now.hour, today_now.minute, today_now.second)

        # Python logger
        cnn_rnn_tf_0.logger = logging.getLogger(__name__)
        cnn_rnn_tf_0.logger.setLevel(logging.getLevelName(cnn_rnn_tf_0.stngs['logging']['level']))

        log_file_name = cnn_rnn_tf_0.stngs['logging']['file_name_prefix'] + self.date_time + '.log'

        log_file_path = os.path.join(cnn_rnn_tf_0.stngs['logging']['file_path'], log_file_name)
        log_file_handler = logging.FileHandler(log_file_path)
        log_file_handler.setLevel(logging.getLevelName(cnn_rnn_tf_0.stngs['logging']['level']))
        log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_file_handler.setFormatter(log_formatter)

        cnn_rnn_tf_0.logger.addHandler(log_file_handler)

        
    def load_settings(self, settings_file):
        with open(settings_file) as json_settings_file:
            json_settings = json.load(json_settings_file)
        return json_settings
        
        
    def tf_log_dir(self):
        current_workdir = os.getcwd()
        tstamp = int(time.time())
        sess_dir_name = 'sess_%s' % tstamp
        dirty_path = os.path.join(current_workdir, cnn_rnn_tf_0.stngs['tf_log_dir'], sess_dir_name)
        return os.path.realpath(dirty_path)


    # Parse training data to matrices
    def create_train_data(self):
        with open(os.path.join(cnn_rnn_tf_0.stngs['cnn']['train_data_path'], cnn_rnn_tf_0.stngs['cnn']['train_data_file']), 'rb') as f:
          (X, y, speaker_names) = pickle.load(f)

        X_t, X_v, y_t, y_v = dg.splitter(X, y, 0.125, 8)
        return X_t, y_t, X_v, y_v

    
    # Create basic net infrastructure
    def create_net(self):
        cnn_rnn_tf_0.logger.info("Creating placeholders")
        with tf.name_scope('Placeholders'):
            x_input = tf.placeholder(tf.float32, shape=(None, cnn_rnn_tf_0.stngs['segment_size'], cnn_rnn_tf_0.stngs['frequencies']))
            out_labels = tf.placeholder(tf.float32, shape=(None, 100))


        cnn_rnn_tf_0.logger.info("Creating BiLSTM")
        cnn_rnn_tf_0.logger.info(tf.unstack(x_input, x_input.get_shape()[1], 1))
        cnn_rnn_tf_0.logger.info("x_input:"+str(x_input.get_shape()))
        fw_cell = tf.contrib.rnn.LSTMCell(cnn_rnn_tf_0.stngs['lstm']['neuron_number'], state_is_tuple=True)
        fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=0.5)
        bw_cell = tf.contrib.rnn.LSTMCell(cnn_rnn_tf_0.stngs['lstm']['neuron_number'], state_is_tuple=True)
        bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=0.5)
        fw_cell = tf.contrib.rnn.MultiRNNCell([fw_cell] * cnn_rnn_tf_0.stngs['lstm']['layers'], state_is_tuple=True)
        bw_cell = tf.contrib.rnn.MultiRNNCell([bw_cell] * cnn_rnn_tf_0.stngs['lstm']['layers'], state_is_tuple=True)
        lstm_out_f, _, _ = tf.contrib.rnn.static_bidirectional_rnn(fw_cell, bw_cell,
                                               tf.unstack(x_input, x_input.get_shape()[1], 1),
                                               dtype=tf.float32, )
        
        lstm_out = lstm_out_f[-1]
        cnn_rnn_tf_0.logger.info("lstm_out:"+str(lstm_out.get_shape()))


        #with tf.name_scope('GRU'):
        #    x_gru = tf.unstack(lstm_input, lstm_input.get_shape()[1], 1)
        #    gru_cell = tf.contrib.rnn.GRUCell(cnn_rnn_tf_0.stngs['gru']['neurons_number'])
        #    dense, _ = tf.contrib.rnn.static_rnn(gru_cell, x_gru, dtype='float')
        #    gru_out = dense[-1]

        cnn_rnn_tf_0.logger.info("Creating dense layer 1")
        with tf.name_scope('Softmax'):
            dense1 = tf.layers.dense(inputs=lstm_out, units=1000, activation=tf.nn.relu)

        cnn_rnn_tf_0.logger.info("Creating dense layer 2")
        with tf.name_scope('Softmax'):
            dense2 = tf.layers.dense(inputs=dense1, units=500, activation=tf.nn.relu)

        cnn_rnn_tf_0.logger.info("Creating output layer")
        with tf.name_scope('Softmax'):
            out = tf.layers.dense(inputs=dense2, units=cnn_rnn_tf_0.stngs['total_speakers'], activation=tf.nn.softmax)


        # Cross entropy and optimizer
        cnn_rnn_tf_0.logger.info("Create optimizer and loss function")
        with tf.name_scope('Optimizer'):
            cnn_rnn_tf_0.logger.info("Create cross entropy function")
            kld_loss = kld.pairwise_kl_divergence(out_labels, out)
            tf.summary.scalar('loss', kld_loss)
            cnn_rnn_tf_0.logger.info("Create AdamOptimizer and add cross_entropy as minimize function")
            optimizer = tf.train.AdamOptimizer().minimize(kld_loss)

        with tf.name_scope('Accuracy'):
            correct_pred = tf.equal(tf.argmax(out, 1), tf.argmax(out_labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            tf.summary.scalar('accuracy', accuracy)
        
        return optimizer, lstm_out, dense1, dense2, out, kld_loss, accuracy, x_input, out_labels

    
    def run_network(self):
        # Create training batches
        cnn_rnn_tf_0.logger.info("Creating training batches")
        X_t, y_t, X_v, y_v = self.create_train_data()
        train_gen = dg.batch_generator_lstm(X_t, y_t, batch_size=cnn_rnn_tf_0.stngs['batch_size'], segment_size=cnn_rnn_tf_0.stngs['segment_size'])
        val_gen = dg.batch_generator_lstm(X_v, y_v, batch_size=cnn_rnn_tf_0.stngs['batch_size'], segment_size=cnn_rnn_tf_0.stngs['segment_size'])
        # Create network model and tensors
        cnn_rnn_tf_0.logger.info("Initialize network model")
        optimizer, lstm_out, dense1, dense2, out, kld_loss, accuracy, x_input, out_labels = self.create_net()
        
        # CNN Training
        cnn_rnn_tf_0.logger.info("Initialize tensorflow session")
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # Tensorboard
        cnn_rnn_tf_0.logger.info("Initialize tensorboard dependencies")
        tb_merged = tf.summary.merge_all()
        tb_saver = tf.train.Saver()
        tb_log_dir = self.tf_log_dir()

        if not os.path.exists(tb_log_dir):
            os.makedirs(tb_log_dir)

        tb_train_log_dir = (tb_log_dir + '_train')
        tb_train_writer = tf.summary.FileWriter(tb_train_log_dir, sess.graph)
        tb_val_log_dir = (tb_log_dir + '_val')
        tb_val_writer = tf.summary.FileWriter(tb_val_log_dir, sess.graph)

        run_options = tf.RunOptions()
        run_metadata = tf.RunMetadata()

        csv_file_handler = open(os.path.join(tb_log_dir, (cnn_rnn_tf_0.stngs['csv_file_pfx'] + self.date_time + '.csv')), 'a')
        csv_writer = csv.writer(csv_file_handler)

        cnn_rnn_tf_0.logger.info("Start training")
        for step in range(cnn_rnn_tf_0.stngs['batch_loops']):
            start_time = time.time()

            # Get next batch
            x_b, y_b = train_gen.next()
            x_vb, y_vb = val_gen.next()
            # Reshape the x_b batch with channel as last dimension
            #x_b = np.reshape(x_b_t, [cnn_rnn_tf_0.stngs['batch_size'], cnn_rnn_tf_0.stngs['segment_size'], cnn_rnn_tf_0.stngs['frequencies'], 1])
            #x_vb = np.reshape(x_vb_t, [cnn_rnn_tf_0.stngs['batch_size'], cnn_rnn_tf_0.stngs['segment_size'], cnn_rnn_tf_0.stngs['frequencies'], 1])
            cnn_rnn_tf_0.logger.info("x_b:"+str(x_b.shape))
            # Execute training
            train_feed_dict = { x_input: x_b, out_labels: y_b }
            val_feed_dict = { x_input: x_vb, out_labels: y_vb }

            _, loss_value = sess.run([optimizer, kld_loss], feed_dict=train_feed_dict, options=run_options, run_metadata=run_metadata)
            sess_acc = sess.run(accuracy, feed_dict=train_feed_dict, options=run_options, run_metadata=run_metadata)

            val_acc = sess.run(accuracy, feed_dict=val_feed_dict, options=run_options, run_metadata=run_metadata)
            val_loss = sess.run(kld_loss, feed_dict=val_feed_dict, options=run_options, run_metadata=run_metadata)
            
            duration = time.time() - start_time
            cnn_rnn_tf_0.logger.info('Round %d (%f s): train_accuracy %f, train_loss %f , val_accuracy %f, val_loss %f', step, duration, sess_acc, loss_value, val_acc, val_loss)
            csv_writer.writerow([step, sess_acc, loss_value, val_acc, val_loss])

            if step % 100 == 0:
                tb_train_summary_str = sess.run(tb_merged, feed_dict=train_feed_dict)
                tb_train_writer.add_run_metadata(run_metadata, 'step_{:04d}'.format(step))
                tb_train_writer.add_summary(tb_train_summary_str, step)
                tb_train_writer.flush()

                tb_val_summary_str = sess.run(tb_merged, feed_dict=val_feed_dict)
                tb_val_writer.add_run_metadata(run_metadata, 'step_{:04d}'.format(step))
                tb_val_writer.add_summary(tb_val_summary_str, step)
                tb_val_writer.flush()

            if step % 500 == 0:
                ckpt_file = os.path.join(tb_log_dir, cnn_rnn_tf_0.stngs['ckpt_file_pfx'])
                tb_saver.save(sess, ckpt_file, global_step=step)

        csv_file_handler.close()

        # Save the meta model
        cnn_rnn_tf_0.logger.info("Saving meta model")
        model_meta_file = os.path.join(tb_log_dir, cnn_rnn_tf_0.stngs['model_file_name'])
        tb_saver.save(sess, model_meta_file)

        # Evaluate the network
        cnn_rnn_tf_0.logger.info("Loading train data for GRU evaluation")
        with open(os.path.join(cnn_rnn_tf_0.stngs['clustering']['data_path'], cnn_rnn_tf_0.stngs['clustering']['train_data_file_40']), 'rb') as f:
            (train_raw_x_data, raw_y_data, train_speaker_names) = pickle.load(f)
            train_x_data, train_y_data = dg.generate_test_data(train_raw_x_data, raw_y_data, segment_size=cnn_rnn_tf_0.stngs['segment_size'])

        train_net_output = lstm_out.eval(feed_dict={x_input: train_x_data.reshape(train_x_data.shape[0],train_x_data.shape[3], train_x_data.shape[2])}, session=sess)
        
        cnn_rnn_tf_0.logger.info("Loading test data for GRU evaluation")
        with open(os.path.join(cnn_rnn_tf_0.stngs['clustering']['data_path'], cnn_rnn_tf_0.stngs['clustering']['test_data_file_40']), 'rb') as f:
            (test_raw_x_data, raw_y_data, test_speaker_names) = pickle.load(f)
            test_x_data, test_y_data = dg.generate_test_data(test_raw_x_data, raw_y_data, segment_size=cnn_rnn_tf_0.stngs['segment_size'])

        test_net_output = lstm_out.eval(feed_dict={x_input: test_x_data.reshape(test_x_data.shape[0],test_x_data.shape[3], test_x_data.shape[2])}, session=sess)


        # Write output file for clustering
        cnn_rnn_tf_0.logger.info("Write outcome to pickle files for clustering")
        with open(os.path.join(cnn_rnn_tf_0.stngs['clustering']['cluster_output_path'], (cnn_rnn_tf_0.stngs['clustering']['cluster_output_train_file'] + self.date_time + '.pickle')), 'wb') as f:
            pickle.dump((train_net_output, train_y_data, train_speaker_names), f, -1)
        
        with open(os.path.join(cnn_rnn_tf_0.stngs['clustering']['cluster_output_path'], (cnn_rnn_tf_0.stngs['clustering']['cluster_output_test_file'] + self.date_time + '.pickle')), 'wb') as f:
            pickle.dump((test_net_output, test_y_data, test_speaker_names), f, -1)
