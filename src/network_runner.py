#import nets.bilstm_2layer as bi2
#import nets.bilstm_2layer_dropout as bi2d
#import nets.bilstm_2layer_dropout_plus_2dense as lstm2_dense
#import nets.bilstm_4layer_plus_dense as lstm4_dense
#import nets.bilstm_4layer as bi4
#import nets.bilstm_3layer as bi3
#import nets.bilstm_4layer as bi4
#import nets.cnn_lstm_layer7 as cnn_lstm
import nets.bilstm_2layer_dropout_plus_2dense as lstm2_dense
#import nets.bilstm_4layer_plus_dense as lstm4_dense
#import nets.bilstm_2layer_dropout_kld as bi2dkld
import nets.cnn_rnn_tf_0 as crt0
import nets.bilstm_tf_1	as bt1




#cnn_lstm.cnn_lstm_layer7('CNN_L7_LSTM128x2_PS_4_1_sp100_v2', 'train_data_100.pickle', pool_size=(4,1), strides=(2,1))

#cnn_lstm.cnn_lstm_layer7('CNN_L7_LSTM128x2_PS_6_1_sp100_v2', 'train_data_100.pickle', pool_size=(6,1), strides=(3,1))
#cnn_lstm.cnn_lstm_layer7('CNN_L7_LSTM128x2_PS_8_1_sp100_v2', 'train_data_100.pickle', pool_size=(8,1), strides=(4,1))

#bi2.bilstm_2layer('cluster_train_01', 'train_clustering_590.pickle', n_classes=590, n_epoch=1000, segment_size=15)
#bi2.bilstm_2layer('cluster_train_01', 'train_clustering_590.pickle', n_hidden1=128, n_hidden2=128, n_classes=590, n_epoch=1000, segment_size=15)
#bi2d.bilstm_2layer_dropout('cluster_train_dropout_150ms', 'train_clustering_590.pickle', n_hidden1=256, n_hidden2=128, n_classes=590, n_epoch=4000, segment_size=15)
#bi2d.bilstm_2layer_dropout('cluster_train_dropout_500ms_256_100sp', 'test_data_100.pickle', n_hidden1=256, n_hidden2=256, n_classes=100, n_epoch=4000, segment_size=50)
#bi2d.bilstm_2layer_dropout('cluster_train_droput_500ms_128_400sp', 'train_clustering_400.pickle', n_hidden1=128, n_hidden2=128, n_classes=400, n_epoch=4000, segment_size=50)
#bi2d.bilstm_2layer_dropout('kld_clust_lstm2', 'speakers_100_50w_50m_not_reynolds.pickle', n_hidden1=256, n_hidden2=256, n_classes=100, n_epoch=1, segment_size=50)
lstm2_dense.bilstm_2layer_dropout('kld_run20170421', 'speakers_100_50w_50m_not_reynolds.pickle', n_hidden1=256, n_hidden2=256, n_classes=100, n_epoch=1000, segment_size=50)
#bi4.bilstm_4layer('kld_clust_lstm4', 'speakers_100_50w_50m_not_reynolds.pickle', n_hidden1=256, n_hidden2=256, n_hidden3 = 256, n_hidden4=256, n_classes=100, n_epoch=2000, segment_size=50)
#lstm4_dense.bilstm_4layer('kld_clust_lstm4_dense', 'speakers_100_50w_50m_not_reynolds.pickle', n_hidden1=256, n_hidden2=256, n_hidden3 = 256, n_hidden4=256, n_classes=100, n_epoch=2000, segment_size=50)

#bi4.bilstm_4layer('clustering_lstm4', 'speakers_100_50w_50m_not_reynolds.pickle', n_hidden1=256, n_hidden2=256, n_hidden3 = 256, n_hidden4=256, n_classes=100, n_epoch=4000, segment_size=50)
#lstm4_dense.bilstm_4layer('clustering_lstm4_dense', 'speakers_100_50w_50m_not_reynolds.pickle', n_hidden1=256, n_hidden2=256, n_hidden3 = 256, n_hidden4=256, n_classes=100, n_epoch=4000, segment_size=50)
#network = test.test('test01', 'test_data_10_not_clustering_vs_reynolds.pickle', n_classes=10, n_epoch=4)
#network.run_network()

#bt1.cnn_rnn_tf_0('nets/bt_settings.json')

