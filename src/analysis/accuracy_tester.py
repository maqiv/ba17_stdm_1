import numpy as np

from src.network_training import clustering_network
from src.core import network_factory
from src.core import analytics


if __name__ == "__main__":
    NETWORK_FILE = '../../data/experiments/clusteringKL/networks_final/adadelta/03_network_params_adadelta_default_params_2000_valids.pickle'
    TEST_DATA_FILE = '../../data/training/TIMIT_extracted/test_data_100_50w_50m_not_reynolds.pickle'

    outputs, targets, speaker_names_test = clustering_network.generate_output(network_params_file_in=NETWORK_FILE,
                                                                            input_file=TEST_DATA_FILE,
                                                                            output_file_out=None,
                                                                            network_fun=network_factory.create_network_100_speakers,
                                                                            get_conv_output=None,
                                                                            output_layer=11, overlapping=True)

    confusion = analytics.ConfusionMatrix(len(set(targets)))
    predictions = np.argmax(outputs, axis=1)
    confusion.add_predictions(targets, predictions)

    accs = confusion.calculate_accuracies()
    accuracy = accs.sum() / len(accs) * 100

    print accuracy
