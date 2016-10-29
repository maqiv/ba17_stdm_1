from random import randint

import numpy as np

import theano
from theano import tensor as T

import lasagne
import src.core.objectives_clustering as objectives_clustering
import src.core.input_data_factory as idf
import src.core.settings as settings


###################
# BATCH ITERATORS #
###################

class BatchIterator(object):
    def __init__(self, batchsize, batches_per_epoch):
        self.batchsize = batchsize
        self.batches_per_epoch = batches_per_epoch

    def iterate(self, inputs, targets):
        raise NotImplementedError


class SpectTrainBatchIterator(BatchIterator):
    def __init__(self, batchsize, batches_per_epoch, speaker_offset_limit=1, segments_per_sentence=1):
        super(SpectTrainBatchIterator, self).__init__(batchsize, batches_per_epoch)
        self.speaker_offset_limit = speaker_offset_limit
        self.segments_per_sentence = segments_per_sentence

    def iterate(self, inputs, targets):
        assert len(inputs) == len(targets)
        for i in range(0, self.batches_per_epoch):
            Xb = np.zeros((self.batchsize, 1, settings.FREQ_ELEMENTS, settings.ONE_SEC), dtype=np.float32)
            yb = np.zeros(self.batchsize, dtype=np.int32)

            # here one batch is generated
            for j in range(0, self.batchsize, self.speaker_offset_limit * self.segments_per_sentence):
                speaker_idx = randint(0, len(inputs) - 1)
                for speaker_offset in range(self.speaker_offset_limit):
                    speaker_idx_offset = speaker_idx + speaker_offset
                    if speaker_idx_offset >= len(inputs):
                        speaker_idx_offset = speaker_idx - speaker_offset
                    spect = idf.extract_spectrogram(inputs[speaker_idx_offset, 0])
                    self.extract_segments(Xb, yb, targets, j + speaker_offset * 2, speaker_idx_offset, spect,
                                          self.segments_per_sentence)
            yield Xb, yb

    @staticmethod
    def extract_segments(Xb, yb, targets, insert_idx, speaker_idx, spect, num_segments):
        for i in range(num_segments):
            seg_idx = randint(0, spect.shape[1] - settings.ONE_SEC)
            Xb[insert_idx + i, 0] = spect[:, seg_idx:seg_idx + settings.ONE_SEC]
            if targets is not None:
                yb[insert_idx + i] = targets[speaker_idx]


class SpectValidBatchIterator(BatchIterator):
    def iterate(self, inputs, targets):
        assert len(inputs) == len(targets)

        seg_count = 0
        for input in inputs:
            seg_count += idf.extract_spectrogram(input[0]).shape[
                             1] / settings.ONE_SEC + 1  # todo: make the segments overlapping by half a second

        speaker_idx = 0
        segment_pos = 0
        iterations = seg_count - self.batchsize + 1
        if seg_count < self.batchsize:
            iterations = seg_count
        for start_idx in range(0, iterations, self.batchsize):
            Xb = np.zeros((self.batchsize, 1, settings.FREQ_ELEMENTS, settings.ONE_SEC), dtype=np.float32)
            yb = []

            spect = idf.extract_spectrogram(inputs[speaker_idx, 0])
            for batch_pos in range(0, self.batchsize):
                if targets is not None:
                    yb.append(targets[speaker_idx])
                if segment_pos == spect.shape[1] / settings.ONE_SEC:
                    # add the last segment backwards to ensure nothing is left out
                    Xb[batch_pos, 0] = spect[:, spect.shape[1] - settings.ONE_SEC:spect.shape[1]]
                else:
                    seg_idx = segment_pos * settings.ONE_SEC
                    Xb[batch_pos, 0] = spect[:, seg_idx:seg_idx + settings.ONE_SEC]
                segment_pos += 1
                if segment_pos == spect.shape[1] / settings.ONE_SEC + 1 and speaker_idx + 1 < len(inputs):
                    speaker_idx += 1
                    spect = idf.extract_spectrogram(inputs[speaker_idx, 0])
                    segment_pos = 0
                elif segment_pos == spect.shape[1] / settings.ONE_SEC + 1 and speaker_idx + 1 == len(inputs):
                    break
            yield Xb[0:len(yb)], np.asarray(yb, dtype=np.int32)


class SpectWithSeparateConvTrainBatchIterator(SpectTrainBatchIterator):
    def __init__(self, batchsize, batches_per_epoch, input_dim, get_conv_output):
        super(SpectWithSeparateConvTrainBatchIterator, self).__init__(batchsize, batches_per_epoch)
        self.input_dim = input_dim
        self.get_conv_output = get_conv_output

    def iterate(self, inputs, targets):
        for Xb_spect, yb in super(SpectWithSeparateConvTrainBatchIterator, self).iterate(inputs, targets):
            Xb = np.zeros((self.batchsize, 1, 1, self.input_dim), dtype=np.float32)
            conv_outputs = self.get_conv_output(Xb_spect)
            for i in range(len(conv_outputs)):
                Xb[i, 0, 0] = conv_outputs[i]
            yield Xb, yb


class SpectWithSeparateConvValidBatchIterator(SpectTrainBatchIterator):
    def __init__(self, batchsize, batches_per_epoch, input_dim, get_conv_output):
        super(SpectWithSeparateConvValidBatchIterator, self).__init__(batchsize, batches_per_epoch)
        self.input_dim = input_dim
        self.get_conv_output = get_conv_output

    def iterate(self, inputs, targets):
        for Xb_spect, yb in super(SpectWithSeparateConvValidBatchIterator, self).iterate(inputs, targets):
            Xb = np.zeros((self.batchsize, 1, 1, self.input_dim), dtype=np.float32)
            conv_outputs = self.get_conv_output(Xb_spect)
            for i in range(len(conv_outputs)):
                Xb[i, 0, 0] = conv_outputs[i]
            yield Xb, yb


##########
# HELPER #
##########

def create_loss_functions_kl_div(input_var, network, target_var, margin):
    # define loss expression
    prediction = lasagne.layers.get_output(network)
    loss = objectives_clustering.mean_loss_kl_div(prediction, target_var, margin)

    # define update expression for training
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adadelta(loss, params, learning_rate=1.0, rho=0.95, epsilon=1e-6)
    #updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.001)

    # loss expression for validation/testing (disable dropouts etc.)
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = objectives_clustering.mean_loss_kl_div(test_prediction, target_var, margin)

    # create accuracy expression
    test_class_predict = T.argmax(test_prediction, axis=1)

    # compile theano functions for training and validation/accuracy
    train_fn = theano.function([input_var, target_var, margin], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var, margin], [test_loss, test_class_predict])  # , test_acc])
    return train_fn, val_fn
