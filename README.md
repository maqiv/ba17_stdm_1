# Machine Learning for Speaker Clustering
Bachelor's Thesis - by Patrick Gerber and Sebastian Glinski-Haefeli

### Processing

#### Pre-Processing:
- `src/core/spectrogram_generation/main_data_extractor.py`: Extract mel-spectrograms from speaker audio files and save them to `data/training/TIMIT_extracted` in .pickle file format

#### Processing (Training):
- `src/network_runner.py`: Wrapper functions to start training a model with appropriate parameters. In this case it's the network `src/nets/bilstm_2layer_dropout_plus_2dense.py` regarding the results from the bachelor's thesis ba17_stdm_1.

#### Post-Processing:
- `src/keras_cluster_output_generator.py`: Execute clustering procedure on appropriate extracted speaker data with the previously trained network (-model)
- `src/clustering/cluster_tester.py`: Calculate missclassification rate and optionally show a scatter plot for processed clusterings

### Required folder structure
* data/experiments/logs/
* data/experiments/nets/
* data/experiments/plots/
* data/speaker_lists/
* data/training/TIMIT/
* data/training/TIMIT_extracted/
