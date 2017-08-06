
import nets.bilstm_2layer_dropout_plus_2dense as lstm2_dense

#import nets.cnn_rnn_tf_0 as crt0
#import nets.cnn_rnn_tf_1 as crt1
#import nets.cnn_rnn_tf_2 as crt2
#import nets.cnn_rnn_tf_3a as crt3a
#import nets.cnn_rnn_tf_3b as crt3b
#import nets.cnn_rnn_tf_4 as crt4
#import nets.cnn_rnn_tf_5_cont as crt5






lstm2_dense.bilstm_2layer_dropout('20170426_lstm2_dense_kld_100batch_new_labels', 'speakers_100_50w_50m_not_reynolds.pickle', n_hidden1=256, n_hidden2=256, n_classes=100, n_epoch=1000, segment_size=50)



#crt0.cnn_rnn_tf_0('nets/crt_settings.json',
#                    n_filter1=32,
#                    n_kernel1=[8, 1],
#                    n_pool1=[4, 1],
#                    n_strides1=[2, 1],
#                    n_filter2=64,
#                    n_kernel2=[8, 1],
#                    n_pool2=[4, 1],
#                    n_strides2=[2, 1],
#                    n_gru_neurons=256
#                )
#

#crt1.cnn_rnn_tf_1('nets/crt_settings.json')

#crt2.cnn_rnn_tf_2('nets/crt_settings.json',
#                    n_filter1=16,
#                    n_kernel1=[7, 7],
#                    n_pool1=[3, 3],
#                    n_strides1=[2, 1],
#                    n_filter2=32,
#                    n_kernel2=[5, 5],
#                    n_pool2=[3, 3],
#                    n_strides2=[2, 1],
#                    n_filter3=32,
#                    n_kernel3=[3, 3],
#                    n_pool3=[3, 3],
#                    n_strides3=[2, 1],
#                    n_filter4=32,
#                    n_kernel4=[3, 3],
#                    n_pool4=[3, 3],
#                    n_strides4=[2, 1],
#                    n_gru_neurons=32
#                )

#crt3a.cnn_rnn_tf_3a('nets/crt_settings.json',
#                    n_filter1=32,
#                    n_kernel1=[8, 1],
#                    n_pool1=[4, 4],
#                    n_strides1=[2, 1],
#                    n_filter2=64,
#                    n_kernel2=[6, 1],
#                    n_pool2=[3, 3],
#                    n_strides2=[2, 1],
#                    n_dense1=200
#                )


#crt3b.cnn_rnn_tf_3b('nets/crt_settings.json',
#                    n_filter1=32,
#                    n_kernel1=[8, 1],
#                    n_pool1=[4, 4],
#                    n_strides1=[2, 1],
#                    n_filter2=64,
#                    n_kernel2=[6, 1],
#                    n_pool2=[3, 3],
#                    n_strides2=[2, 1],
#                    n_dense1=200,
#                    n_dense2=250
#                )

#crt4.cnn_rnn_tf_4('nets/crt_settings.json')

#crt5.cnn_rnn_tf_5('nets/crt_settings.json', 'sess_14952524548')
