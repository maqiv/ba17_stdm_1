import os
import csv
import json
import time
import logging
import datetime
import numpy as np
import tensorflow as tf

import core.data_gen as dg
import cPickle as pickle

class cnn_rnn_tf_3(object):

    stngs = None
    logger = None
    date_time = None

    def __init__(self, network_settings_file, sess_dir):
        cnn_rnn_tf_1.stngs = self.load_settings(network_settings_file)
        self.sess_dir = sess_dir
        self.initialize_logger()
        cnn_rnn_tf_3.logger.info("Calling run_network()")
        self.run_network()

    def initialize_logger(self):
        today_now = datetime.datetime.now()
        self.date_time = '_{:04d}-{:02d}-{:02d}_{:02d}-{:02d}-{:02d}'.format(
                today_now.year, today_now.month, today_now.day,
                today_now.hour, today_now.minute, today_now.second)

        # Python logger
        cnn_rnn_tf_3.logger = logging.getLogger(__name__)
        cnn_rnn_tf_3.logger.setLevel(logging.getLevelName(cnn_rnn_tf_3.stngs['logging']['level']))

        log_file_name = cnn_rnn_tf_3.stngs['logging']['file_name_prefix'] + self.date_time + '.log'

        log_file_path = os.path.join(cnn_rnn_tf_3.stngs['logging']['file_path'], log_file_name)
        log_file_handler = logging.FileHandler(log_file_path)
        log_file_handler.setLevel(logging.getLevelName(cnn_rnn_tf_3.stngs['logging']['level']))
        log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_file_handler.setFormatter(log_formatter)

        cnn_rnn_tf_3.logger.addHandler(log_file_handler)


    def load_settings(self, settings_file):
        with open(settings_file) as json_settings_file:
            json_settings = json.load(json_settings_file)
        return json_settings


    def tf_log_dir(self):
        current_workdir = os.getcwd()
        tstamp = int(time.time())
        sess_dir_name = 'sess_%s' % tstamp
        dirty_path = os.path.join(current_workdir, cnn_rnn_tf_3.stngs['tf_log_dir'], sess_dir_name)
        return os.path.realpath(dirty_path)


    # Parse training data to matrices
    def create_train_data(self):
        with open(os.path.join(cnn_rnn_tf_3.stngs['cnn']['train_data_path'], cnn_rnn_tf_3.stngs['cnn']['train_data_file']), 'rb') as f:
            (X, y, speaker_names) = pickle.load(f)

        X_t, X_v, y_t, y_v = dg.splitter(X, y, 0.125, 8)
        return X_t, y_t, X_v, y_v


    def create_optimizer(self, gru_soft_out, out_labels):
    # Create basic net infrastructure
        cnn_rnn_tf_1.logger.info("Create loss function")

        with tf.name_scope('ACC'):
            correct_pred = tf.equal(tf.argmax(gru_soft_out, 1), tf.argmax(out_labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            tf.summary.scalar('accuracy', accuracy)
        
        return  accuracy


    def run_network(self):
        # Reset the graph to free up resources
        tf.reset_default_graph()

        # Create training batches
        cnn_rnn_tf_3.logger.info("Creating training batches")
        X_t, y_t, X_v, y_v = self.create_train_data()
        train_gen = dg.batch_generator(X_t, y_t, batch_size=cnn_rnn_tf_3.stngs['batch_size'], segment_size=cnn_rnn_tf_3.stngs['segment_size'])
        val_gen = dg.batch_generator(X_v, y_v, batch_size=cnn_rnn_tf_3.stngs['batch_size'], segment_size=cnn_rnn_tf_3.stngs['segment_size'])
        # Create network model and tensors

        # CNN Training
        cnn_rnn_tf_3.logger.info("Initialize tensorflow session")
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph('../data/experiments/logs/sess_' + self.sess_dir + '/final_model.save.meta')
        saver.restore(sess, '../data/experiments/logs/sess_' + self.sess_dir + '/final_model.save')
        graph = tf.get_default_graph()
        gru_out = graph.get_tensor_by_name('GRU/rnn/gru_cell_22/add:0')
        x_input = graph.get_tensor_by_name('Placeholders/Placeholder:0')
        out_labels = graph.get_tensor_by_name('Placeholders/Placeholder_1:0')
        gru_soft_out = graph.get_tensor_by_name('Softmax/dense/Softmax:0')
        optimizer = graph.get_operation_by_name('Optimizer/Adam')
        cross_entropy = graph.get_tensor_by_name('Optimizer/Mean:0')
        #accuracy = graph.get_tensor_by_name()
        cnn_rnn_tf_1.logger.info("Initialize network model")
        accuracy = self.create_optimizer(gru_soft_out, out_labels)

        # Tensorboard
        cnn_rnn_tf_3.logger.info("Initialize tensorboard dependencies")
        tb_merged = tf.summary.merge_all()
        tb_saver = tf.train.Saver()
        tb_log_dir = self.tf_log_dir()

        if not os.path.exists(tb_log_dir):
            os.makedirs(tb_log_dir)

        tb_train_log_dir = (tb_log_dir + '_train')
        tb_train_writer = tf.summary.FileWriter(tb_train_log_dir, sess.graph)
        tb_val_log_dir = (tb_log_dir + '_val')
        tb_val_writer = tf.summary.FileWriter(tb_val_log_dir, sess.graph)

        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        csv_file_handler = open(os.path.join(tb_log_dir, (cnn_rnn_tf_3.stngs['csv_file_pfx'] + self.date_time + '.csv')), 'a')
        csv_writer = csv.writer(csv_file_handler)

        cnn_rnn_tf_3.logger.info("Start training")
        for step in range(cnn_rnn_tf_3.stngs['batch_loops']):
            start_time = time.time()

            # Get next batch
            x_b_t, y_b = train_gen.next()
            x_vb_t, y_vb = val_gen.next()
            # Reshape the x_b batch with channel as last dimension
            x_b = np.reshape(x_b_t, [cnn_rnn_tf_3.stngs['batch_size'], cnn_rnn_tf_3.stngs['frequencies'], cnn_rnn_tf_3.stngs['segment_size'], 1])
            x_vb = np.reshape(x_vb_t, [cnn_rnn_tf_3.stngs['batch_size'], cnn_rnn_tf_3.stngs['frequencies'], cnn_rnn_tf_3.stngs['segment_size'], 1])

            # Execute training
            train_feed_dict = { x_input: x_b, out_labels: y_b }
            val_feed_dict = { x_input: x_vb, out_labels: y_vb }

            _, loss_value = sess.run([optimizer, cross_entropy], feed_dict=train_feed_dict, options=run_options, run_metadata=run_metadata)
            sess_acc = sess.run(accuracy, feed_dict=train_feed_dict, options=run_options, run_metadata=run_metadata)

            val_acc = sess.run(accuracy, feed_dict=val_feed_dict, options=run_options, run_metadata=run_metadata)
            val_loss = sess.run(cross_entropy, feed_dict=val_feed_dict, options=run_options, run_metadata=run_metadata)

            duration = time.time() - start_time
            cnn_rnn_tf_3.logger.info('Round %d (%f s): train_accuracy %f, train_loss %f , val_accuracy %f, val_loss %f', (step + 1), duration, sess_acc, loss_value, val_acc, val_loss)
            csv_writer.writerow([(step + 1), sess_acc, loss_value, val_acc, val_loss])

            if (step + 1) % cnn_rnn_tf_1.stngs['summary_write_interval'] == 0:
                tb_train_summary_str = sess.run(tb_merged, feed_dict=train_feed_dict)
                tb_train_writer.add_run_metadata(run_metadata, 'step_{:04d}'.format(step))
                tb_train_writer.add_summary(tb_train_summary_str, step)
                tb_train_writer.flush()

                tb_val_summary_str = sess.run(tb_merged, feed_dict=val_feed_dict)
                tb_val_writer.add_run_metadata(run_metadata, 'step_{:04d}'.format(step))
                tb_val_writer.add_summary(tb_val_summary_str, step)
                tb_val_writer.flush()

            if (step + 1) % cnn_rnn_tf_1.stngs['ckpt_write_interval'] == 0:
                ckpt_file = os.path.join(tb_log_dir, cnn_rnn_tf_3.stngs['ckpt_file_pfx'])
                tb_saver.save(sess, ckpt_file, global_step=step)

        csv_file_handler.close()

        # Save the meta model
        cnn_rnn_tf_3.logger.info("Saving meta model")
        model_meta_file = os.path.join(tb_log_dir, cnn_rnn_tf_3.stngs['model_file_name'])
        tb_saver.save(sess, model_meta_file)

        # Evaluate the network for 40 speakers
        cnn_rnn_tf_3.logger.info("Loading train data for GRU evaluation for 40 speakers")
        with open(os.path.join(cnn_rnn_tf_3.stngs['gru']['data_path'], cnn_rnn_tf_3.stngs['gru']['train_data_file_40sp']), 'rb') as f:
            (train_raw_x_data_40, raw_y_data_40, train_speaker_names_40) = pickle.load(f)
            train_x_data_40, train_y_data_40 = dg.generate_test_data(train_raw_x_data_40, raw_y_data_40, segment_size=cnn_rnn_tf_3.stngs['segment_size'])

        train_net_output_40 = gru_out.eval(feed_dict={x_input: np.reshape(train_x_data_40, [train_x_data_40.shape[0], train_x_data_40.shape[2], train_x_data_40.shape[3], train_x_data_40.shape[1]])}, session=sess)

        cnn_rnn_tf_3.logger.info("Loading test data for GRU evaluation for 40 speakers")
        with open(os.path.join(cnn_rnn_tf_3.stngs['gru']['data_path'], cnn_rnn_tf_3.stngs['gru']['test_data_file_40sp']), 'rb') as f:
            (test_raw_x_data_40, raw_y_data_40, test_speaker_names_40) = pickle.load(f)
            test_x_data_40, test_y_data_40 = dg.generate_test_data(test_raw_x_data_40, raw_y_data_40, segment_size=cnn_rnn_tf_3.stngs['segment_size'])

        test_net_output_40 = gru_out.eval(feed_dict={x_input: np.reshape(test_x_data_40, [test_x_data_40.shape[0], test_x_data_40.shape[2], test_x_data_40.shape[3], test_x_data_40.shape[1]])}, session=sess)


        # Write output file for clustering for 40 speakers
        cnn_rnn_tf_3.logger.info("Write outcome to pickle files for clustering for 40 speakers")
        with open(os.path.join(cnn_rnn_tf_3.stngs['cluster_output_path'], (cnn_rnn_tf_3.stngs['cluster_output_train_file_40'] + self.date_time + '.pickle')), 'wb') as f:
            pickle.dump((train_net_output_40, train_y_data_40, train_speaker_names_40), f, -1)

        with open(os.path.join(cnn_rnn_tf_3.stngs['cluster_output_path'], (cnn_rnn_tf_3.stngs['cluster_output_test_file_40'] + self.date_time + '.pickle')), 'wb') as f:
            pickle.dump((test_net_output_40, test_y_data_40, test_speaker_names_40), f, -1)

        
        # Evaluate the network for 60 speakers
        cnn_rnn_tf_3.logger.info("Loading train data for GRU evaluation for 60 speakers")
        with open(os.path.join(cnn_rnn_tf_3.stngs['gru']['data_path'], cnn_rnn_tf_3.stngs['gru']['train_data_file_60sp']), 'rb') as f:
            (train_raw_x_data_60, raw_y_data_60, train_speaker_names_60) = pickle.load(f)
            train_x_data_60, train_y_data_60 = dg.generate_test_data(train_raw_x_data_60, raw_y_data_60, segment_size=cnn_rnn_tf_3.stngs['segment_size'])

        train_net_output_60 = gru_out.eval(feed_dict={x_input: np.reshape(train_x_data_60, [train_x_data_60.shape[0], train_x_data_60.shape[2], train_x_data_60.shape[3], train_x_data_60.shape[1]])}, session=sess)

        cnn_rnn_tf_3.logger.info("Loading test data for GRU evaluation for 60 speakers")
        with open(os.path.join(cnn_rnn_tf_3.stngs['gru']['data_path'], cnn_rnn_tf_3.stngs['gru']['test_data_file_60sp']), 'rb') as f:
            (test_raw_x_data_60, raw_y_data_60, test_speaker_names_60) = pickle.load(f)
            test_x_data_60, test_y_data_60 = dg.generate_test_data(test_raw_x_data_60, raw_y_data_60, segment_size=cnn_rnn_tf_3.stngs['segment_size'])

        test_net_output_60 = gru_out.eval(feed_dict={x_input: np.reshape(test_x_data_60, [test_x_data_60.shape[0], test_x_data_60.shape[2], test_x_data_60.shape[3], test_x_data_60.shape[1]])}, session=sess)


        # Write output file for clustering for 60 speakers
        cnn_rnn_tf_3.logger.info("Write outcome to pickle files for clustering for 60 speakers")
        with open(os.path.join(cnn_rnn_tf_3.stngs['cluster_output_path'], (cnn_rnn_tf_3.stngs['cluster_output_train_file_60'] + self.date_time + '.pickle')), 'wb') as f:
            pickle.dump((train_net_output_60, train_y_data_60, train_speaker_names_60), f, -1)

        with open(os.path.join(cnn_rnn_tf_3.stngs['cluster_output_path'], (cnn_rnn_tf_3.stngs['cluster_output_test_file_60'] + self.date_time + '.pickle')), 'wb') as f:
            pickle.dump((test_net_output_60, test_y_data_60, test_speaker_names_60), f, -1)
        
        
        # Evaluate the network for 80 speakers
        cnn_rnn_tf_3.logger.info("Loading train data for GRU evaluation for 80 speakers")
        with open(os.path.join(cnn_rnn_tf_3.stngs['gru']['data_path'], cnn_rnn_tf_3.stngs['gru']['train_data_file_80sp']), 'rb') as f:
            (train_raw_x_data_80, raw_y_data_80, train_speaker_names_80) = pickle.load(f)
            train_x_data_80, train_y_data_80 = dg.generate_test_data(train_raw_x_data_80, raw_y_data_80, segment_size=cnn_rnn_tf_3.stngs['segment_size'])

        train_net_output_80 = gru_out.eval(feed_dict={x_input: np.reshape(train_x_data_80, [train_x_data_80.shape[0], train_x_data_80.shape[2], train_x_data_80.shape[3], train_x_data_80.shape[1]])}, session=sess)

        cnn_rnn_tf_3.logger.info("Loading test data for GRU evaluation for 80 speakers")
        with open(os.path.join(cnn_rnn_tf_3.stngs['gru']['data_path'], cnn_rnn_tf_3.stngs['gru']['test_data_file_80sp']), 'rb') as f:
            (test_raw_x_data_80, raw_y_data_80, test_speaker_names_80) = pickle.load(f)
            test_x_data_80, test_y_data_80 = dg.generate_test_data(test_raw_x_data_80, raw_y_data_80, segment_size=cnn_rnn_tf_3.stngs['segment_size'])

        test_net_output_80 = gru_out.eval(feed_dict={x_input: np.reshape(test_x_data_80, [test_x_data_80.shape[0], test_x_data_80.shape[2], test_x_data_80.shape[3], test_x_data_80.shape[1]])}, session=sess)


        # Write output file for clustering for 80 speakers
        cnn_rnn_tf_3.logger.info("Write outcome to pickle files for clustering for 80 speakers")
        with open(os.path.join(cnn_rnn_tf_3.stngs['cluster_output_path'], (cnn_rnn_tf_3.stngs['cluster_output_train_file_80'] + self.date_time + '.pickle')), 'wb') as f:
            pickle.dump((train_net_output_80, train_y_data_80, train_speaker_names_80), f, -1)

        with open(os.path.join(cnn_rnn_tf_3.stngs['cluster_output_path'], (cnn_rnn_tf_3.stngs['cluster_output_test_file_80'] + self.date_time + '.pickle')), 'wb') as f:
            pickle.dump((test_net_output_80, test_y_data_80, test_speaker_names_80), f, -1)

        # Close the Tensorflow session
        sess.close()

        # Remove Logging Handlers
        for h in list(cnn_rnn_tf_3.logger.handlers):
            cnn_rnn_tf_3.logger.removeHandler(h)
            h.flush()
            h.close()
