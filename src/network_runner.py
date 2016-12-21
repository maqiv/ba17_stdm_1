import nets.bilstm_2layer as bi2
import nets.bilstm_2layer_dropout as bi2d
import nets.bilstm_3layer as bi3
import nets.bilstm_4layer as bi4
import nets.cnn_lstm_layer7 as cnn_lstm





#cnn_lstm.cnn_lstm_layer7('CNN_L7_LSTM128x2_PS_4_1_sp100_v2', 'train_data_100.pickle', pool_size=(4,1), strides=(2,1))

#cnn_lstm.cnn_lstm_layer7('CNN_L7_LSTM128x2_PS_6_1_sp100_v2', 'train_data_100.pickle', pool_size=(6,1), strides=(3,1))
#cnn_lstm.cnn_lstm_layer7('CNN_L7_LSTM128x2_PS_8_1_sp100_v2', 'train_data_100.pickle', pool_size=(8,1), strides=(4,1))

bi2.bilstm_2layer('test_001', 'train_data_100.pickle', n_classes=100, n_epoch=1000, segment_size=100)
bi2.bilstm_2layer('test_002', 'train_data_630.pickle', n_hidden1=128, n_hidden2=128, n_classes=630, n_epoch=1000, segment_size=15)


#network = test.test('test01', 'test_data_10_not_clustering_vs_reynolds.pickle', n_classes=10, n_epoch=4)
#network.run_network()
