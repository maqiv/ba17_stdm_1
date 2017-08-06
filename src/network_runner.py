import nets.bilstm_2layer_dropout_plus_2dense as lstm2_dense

lstm2_dense.bilstm_2layer_dropout('20170426_lstm2_dense_kld_100batch_new_labels', 'speakers_100_50w_50m_not_reynolds.pickle', n_hidden1=256, n_hidden2=256, n_classes=100, n_epoch=1000, segment_size=50)
