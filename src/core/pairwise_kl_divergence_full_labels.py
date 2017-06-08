import tensorflow as tf
import numpy as np
import cPickle as pickle
import pdb 
from tensorflow.python import debug as tf_debug




tf_l = tf.Variable(0. , name='loss')
x = tf.constant(0.)
margin = tf.constant(2.)
loss = tf.Variable(0.)
sum_loss = tf.Variable(0.)

def return_zero():
    #return tf.add(tf.constant(0.), tf.constant(1e-16))
    return tf.constant(0.)

def return_one():
    #return tf.add(tf.constant(1.), tf.constant(1e-16))
    return tf.constant(1.)

def loss_with_kl_div(P, Q, same , margin):
    epsilon = tf.constant(1e-16)
    P = tf.add(epsilon, P)
    Q = tf.add(epsilon, Q)
    same = tf.to_float(same)
    Ids = tf.abs(tf.subtract(tf.to_float(same), tf.constant(1.)))

    KLPQ = tf.reduce_sum(tf.multiply(P, tf.log(tf.divide(P, Q))))
    KLQP = tf.reduce_sum(tf.multiply(Q, tf.log(tf.divide(Q, P))))
    lossPQ = tf.add(tf.multiply(same, KLPQ), tf.multiply(Ids, tf.maximum(tf.constant(0.), tf.subtract(margin, KLPQ))))
    lossQP = tf.add(tf.multiply(same, KLQP), tf.multiply(Ids, tf.maximum(tf.constant(0.), tf.subtract(margin, KLQP))))
    L = tf.add(lossPQ, lossQP)
    return L

def outerLoop(x, tf_l, predictions, labels, margin):
    
    tf_l = tf.add(tf_l, loss_with_kl_div(predictions[labels[x][0]], predictions[labels[x][1]], labels[x][2], margin))
    return tf.add(x, 1), tf_l, predictions, labels, margin

def outerLoop_condition(x, tf_l, predictions, labels, margin):

    return tf.less(x, tf.shape(labels)[0])

def pairwise_kl_divergence(labels, predictions):
    x = tf.constant(0)
    margin = tf.constant(2.)
    #loss = tf.Variable(0.)
    #tf_l = tf.Variable(0. , name='loss')
    sum_loss = tf.while_loop(outerLoop_condition, outerLoop, [x, tf_l, predictions, tf.to_int32(labels), margin], name='outerloop')
    
    pairs = tf.shape(labels)[0]
    loss = tf.divide(sum_loss[1], tf.to_float(pairs))
    return loss
    
if __name__ == "__main__":

    epsilon = 1e-16
    test_pred = [[1., 2., 3.], [4., 2., 3.], [6., 3., 2.], [4., 1., 5.], [2., 5., 8.]]
    #test_targ = [1, 1, 6]
    test_targ = [[1., 0., 0.], [1., 0., 0.], [0., 0., 1.], [0., 1., 0.], [0., 1. , 0.]]
    test_margin = 2.
    predictions = tf.placeholder('float', [None, None])
    targets = tf.placeholder('int32', [None, None])
    margin = tf.placeholder('float', None)
    print y.shape
    print y.shape
    yl = []
    for l in y:
        for i in range(len(l)):
                if l[i] == 1. :
                    yl.append(i)

    print yl
    pair_list = []
    for i in range(len(yl)):
        j = i+1
        for j in range(i+1,len(yl)):
                if (yl[i] == yl[j]):
                    pair_list.append((i , j, 1))
                else:
                    pair_list.append((i , j, 0))
    y = np.vstack(pair_list)
    print y
    margin = tf.stack(test_margin)
    result = pairwise_kl_divergence(targets, predictions)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    summary = tf.summary.merge_all()
    saver = tf.train.Saver()
    train_writer = tf.summary.FileWriter('../../data/experiments/graph/loss_graph', sess.graph)
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    print 'running stuff'
    res = sess.run(result , feed_dict= {targets:y, predictions:X})
    print(res)
