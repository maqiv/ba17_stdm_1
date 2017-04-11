import tensorflow as tf
import numpy as np
import cPickle as pickle
import pdb 
from tensorflow.python import debug as tf_debug




tf_l = tf.Variable(0. , name='loss')
x = tf.constant(0)
margin = tf.constant(2.)
loss = tf.Variable(0.)
sum_loss = tf.Variable(0.)

def return_zero():
    return tf.add(tf.constant(0.), tf.constant(1e-16))

def return_one():
    return tf.add(tf.constant(1.), tf.constant(1e-16))

def loss_with_kl_div(P, xp, Q, xq, margin):
    with tf.device('/cpu:0'):
        epsilon = tf.constant(1e-16)
        P += epsilon
        Q += epsilon

        Is = tf.cond(tf.reduce_all(tf.equal(xq, xp)), return_one, return_zero)
        Ids = abs(Is-1)
    
        KLPQ = tf.reduce_sum(P * tf.log(P / Q))
        KLQP = tf.reduce_sum(Q * tf.log(Q / P))
        lossPQ = Is * KLPQ + Ids * tf.maximum(0., margin - KLPQ)
        lossQP = Is * KLQP + Ids * tf.maximum(0., margin - KLQP)
        L = lossPQ + lossQP
        return L	

def outerLoop(x, tf_l, predictions, labels, margin):
    with tf.device('/cpu:0'):

        def innerLoop(y,x, tf_l, predictions, labels, margin):
            with tf.device('/cpu:0'):
                tf_l = tf.add(tf_l, loss_with_kl_div(predictions[x], labels[x], predictions[y], labels[y], margin))
                y += 1
                return y, x, tf_l, predictions, labels, margin
    
        def innerLoop_cond(y ,x, tf_l, predictions, labels, margin):
            with tf.device('/cpu:0'):
                return tf.less(y, tf.shape(predictions)[0])
    
        y = tf.constant(0)
        res = tf.while_loop(innerLoop_cond, innerLoop, [y,x,tf_l, predictions, labels, margin], name='innerloop')
        return tf.add(x, 1), res[2], predictions, labels, margin

def outerLoop_condition(x, tf_l, predictions, labels, margin):
    with tf.device('/cpu:0'):
        return tf.less(x, tf.shape(predictions)[0])

def pairwise_kl_divergence(labels, predictions):
    with tf.device('/cpu:0'):
        #x = tf.constant(0)
        #margin = tf.constant(2.)
        #loss = tf.Variable(0.)
        #tf_l = tf.Variable(0. , name='loss')
        sum_loss = tf.while_loop(outerLoop_condition, outerLoop, [x, tf_l, predictions, labels, margin], name='outerloop')
        loss = tf.div(sum_loss[1], (tf.to_float(tf.shape(predictions)[0]) *2.))
        print loss
        return loss
    
if __name__ == "__main__":
    with open('/home/patman/pa/1_Code/data/experiments/cluster_outputs/test_for_kld.pickle', 'rb') as f:
        (X, y, s_list) = pickle.load(f)

    epsilon = 1e-16
    test_pred = [[1., 2., 3.], [4., 2., 3.], [6., 3., 2.]]
    #test_targ = [1, 1, 6]
    test_targ = [[1., 0., 0.], [1., 0., 0.], [0., 0., 1.]]
    test_margin = 2.
    predictions = tf.placeholder('float', [None, None])
    targets = tf.placeholder('float', [None, None])
    margin = tf.placeholder('float', None)
    print y.shape
    print X.shape
    margin = tf.stack(test_margin)
    result = pairwise_kl_divergence(targets, predictions)
    sess = tf.Session()
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    init = tf.global_variables_initializer()
    sess.run(init)
    summary = tf.summary.merge_all()
    saver = tf.train.Saver()
    train_writer = tf.summary.FileWriter('../../data/experiments/graph/loss_graph', sess.graph)
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    print 'running stuff'
    res = sess.run(result , feed_dict= {targets:test_targ, predictions:test_pred})
    print(res)
        
        