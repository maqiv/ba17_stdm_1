import cPickle as pickle

import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.layers as layers
from tensorflow.python.ops import rnn, rnn_cell
import core.segment_batchiterator as seg_bi
import core.speaker_train_splitter as sts
#import logger as log
from core import settings, network_helper, parser

map_fn = tf.map_fn


# Load Training Data
with open('../data/training/TIMIT_extracted/train_data_10_130ms_2.pickle', 'rb') as f:
    (X, y, speaker_names) = pickle.load(f)

# Load test Data
with open('../data/training/TIMIT_extracted/test_data_10_130ms_2.pickle', 'rb') as g:
    (Xt, yt, speaker_names_t) = pickle.load(g)


## Graph Parameters
INPUT_SIZE    =  128    # 2 bits per timestep
TINY          = 1e-6    # to avoid NaNs in logs
LEARNING_RATE = 0.01

training_iters = 100000
batch_size = 128
display_step = 10


## graphs
layers = 2
n_input = 128 # spect input (128*100)
n_steps = 100 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # speakers

## Load Graph and Training parameters from ini file
training_iters, layers, LEARNING_RATE, n_hidden, batch_size = parser.getParameters()


# Define placeholder input and output structure
inputs = tf.placeholder("float", [None, n_steps, n_input])
out = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    # Hidden layer weights => 2*n_hidden because of forward + backward cells
    'out': tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def BiRNN(x, weights, biases):

    # Prepare data shape to match `bidirectional_rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    inputs = tf.transpose(x, [1, 0, 2])
    # Reshape to (n_steps*batch_size, n_input)
    inputs= tf.reshape(inputs, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    inputs = tf.split(0, n_steps, inputs)


    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    lstm_fw_multicell = tf.nn.rnn_cell.MultiRNNCell([lstm_fw_cell]*layers)
    lstm_bw_multicell = tf.nn.rnn_cell.MultiRNNCell([lstm_bw_cell]*layers)

    # Get lstm cell output
    try:
        outputs, _, _ = rnn.bidirectional_rnn(lstm_fw_multicell, lstm_bw_multicell, inputs,
                                              dtype=tf.float32)
    except Exception: # Old TensorFlow version only returns outputs not states
        outputs = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, inputs,
                                        dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

with tf.name_scope("LSTM") as scope:
    pred = BiRNN(inputs, weights, biases)

iterat = seg_bi.SegmentBatchIterator(batch_size=batch_size)
iter_t = seg_bi.SegmentBatchIterator(batch_size=512)
print Xt.shape
print yt.shape

for xtb, ytb in iter_t(Xt, yt):
    print xtb.shape
    print ytb.shape

def  transformy(y):
    yn = np.zeros((batch_size, n_classes))
    k = 0
    for v in y:
    	#print v
        yn[k][v] =1
        k +=1
    return yn

def  transformty(y):
    yn = np.zeros((512, n_classes))
    k = 0
    for v in y:
    	#print v
        yn[k][v] =1
        k +=1
    return yn


##cost and optimizer
with tf.name_scope("cost_function") as scope:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, out))
    tf.scalar_summary("cost_function", cost)

with tf.name_scope("train") as scope:  
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)



def logResults(il, al ,ll,tl, at):
    f = open("logfile.txt", 'a');
    s = "#########################################################################\n ## Hidden:"+str(n_hidden)+"    timesteps: "+str(n_steps)+\
        "     Learning_Rate: "+str(LEARNING_RATE)+" Batch Size: "+str(batch_size)+\
        "\n#################################################################################\n"
    f.write(s)
    c = 0
    for it in il:
        s1 = "Iter " + str(it)+",  MB_Loss: "+str(ll[c])+", ACC: "+str(al[c])
        s1 = s1+"\n"
        c += 1
        f.write(s1)
    s2 = "Test ACC: "+str(at)+"\n"
    f.write(s2)
    f.close()
    t = "Train Acc (blue) vs Test Acc (green) \nLayers: "+str(layers)+"Hidden:"+str(n_hidden)+" |timesteps: "+str(n_steps)+ " |Learning_Rate: "+str(LEARNING_RATE)+" |Batch Size: "+str(batch_size)
    sav = "plt_layers_"+str(layers)+"_run3_hidden_"+str(n_hidden)+"_tstp_"+str(n_steps)+ "_lrt_"+str(LEARNING_RATE)+"_bsz_"+str(batch_size)+".png"
    plt.title(t)
    #plt.text(1,1, te)
    plt.plot(al, 'b', tl, 'g')
    plt.xlabel("Minibatches")
    plt.ylabel("Accuracy")
    plt.savefig(sav)
 

## define accuraxy
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(out,1))

with tf.name_scope("accuracy") as scope :
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

merged_summary = tf.merge_all_summaries()

# Initializing the variables
init = tf.initialize_all_variables()
acc_list=[]
los_list=[]
iter_list=[]
testAcc_list =[]

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1

    #logging with tensorboard
    summary_writer = tf.train.SummaryWriter('/home/patman/pa/results/', graph_def=sess.graph_def)
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        #batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        #print "entered while loop"
        for xb, yb in iterat(X,y):
            batch_x = xb
            batch_x = batch_x.reshape((batch_size, n_steps, n_input))
            yb = transformy(yb)
            #write_batch(batch_x, yb, step)
            #print "calling optimizer"
            sess.run(optimizer, feed_dict={inputs: batch_x, out: yb})
            if step % display_step == 0:
                acc = sess.run(accuracy, feed_dict={inputs: batch_x, out: yb})
                loss = sess.run(cost, feed_dict={inputs: batch_x, out: yb})
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
                iter_list.append(step*batch_size)
                acc_list.append(acc)
                los_list.append(loss)
                summary_str = sess.run(merged_summary, feed_dict={inputs: batch_x, out: yb} )
                summary_writer.add_summary(summary_str, step*batch_size)
                for xtb, ytb in iter_t(Xt, yt):
                    batch_xt = xtb
                    batch_xt = batch_xt.reshape((512, n_steps, n_input))
                    # Run optimization op (backprop)
                    ytb = transformty(ytb)
                    acct = sess.run(accuracy, feed_dict={inputs: batch_xt, out: ytb})
                    testAcc_list.append(acct)
                    print "Testing Acc: "+ str(acct)

        step += 1
    print("Optimization Finished!")
    acct = 0.0
    for xtb, ytb in iter_t(Xt, yt):
            batch_xt = xtb
            batch_xt = batch_xt.reshape((512, n_steps, n_input))
            # Run optimization op (backprop)
            ytb = transformty(ytb)
            acct = sess.run(accuracy, feed_dict={inputs: batch_xt, out: ytb})
            print "Testing Acc: "+ str(acct)

    logResults(iter_list,acc_list,los_list, testAcc_list, acct)

