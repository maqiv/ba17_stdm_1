import tensorflow as tf
import numpy as np
import pickle

epsilon = 1e-16


def return_zero():
    with tf.device('/cpu:0'):
        return tf.constant(0.)

def return_one():
    with tf.device('/cpu:0'):
        return tf.constant(1.)

def loss_with_kl_div(P, xp, Q, xq, margin):
    with tf.device('/cpu:0'):
        P += epsilon
        Q += epsilon
        Is = tf.cond(tf.reduce_all(tf.equal(xq, xp)), return_one, return_zero)
        Ids = abs(Is-1)

        KLPQ = tf.reduce_sum(P * tf.log(P/ Q))
        KLQP = tf.reduce_sum(Q *tf.log(Q / P))
        lossPQ = Is * KLPQ + Ids * tf.maximum(0., margin - KLPQ)
        lossQP = Is * KLQP + Ids * tf.maximum(0., margin - KLQP)
        L = lossPQ + lossQP
        return L	

#def outerLoop(x, tf_l, predictions, labels, margin):
#    
#    print 'body'
#    print predictions.get_shape()
#    print labels.get_shape()
#
#    def innerLoop(y,x, tf_l, predictions, labels, margin):
#        y = 0
#        for  y in range(128):
#            tf_l = tf.add(tf_l, loss_with_kl_div(predictions[x], labels[x], predictions[y], labels[y], margin))
#            #y = tf.add(y,1)
#        return y, x, tf_l, predictions, labels, margin
#   
#    #def innerLoop_cond(y ,x, tf_l, predictions, labels, margin):
#    #    return tf.less(y, tf.shape(predictions)[0])
#
#    y = 0
#    #res = tf.while_loop(innerLoop_cond, innerLoop, [y,x,tf_l, predictions, labels, margin],back_prop=False, name='innerloop')
#    for x in range(128):
#        res = innerLoop(y, x, tf_l, predictions, labels, margin)
#    return tf.add(x, 1), res[2], predictions, labels, margin

#def outerLoop_condition(x, tf_l, predictions, labels, margin):
#    print "cond"
#    return tf.less(x, tf.shape(predictions)[0])

def pairwise_kl_divergence(labels, predictions):
    with tf.device('/cpu:0'):
        x = tf.constant(0)
        margin = tf.constant(2.)
        tf.shape(predictions)
        tf_l = tf.Variable(0. , name='loss')
    #tf_l = tf.Variable(0. , name='loss')
    #sum_loss = tf.while_loop(outerLoop_condition, outerLoop, [x, tf_l, predictions, labels, margin], back_prop=False,name='outerloop')
        for x in range(128):
            for y in range(128):
                tf_l += tf_l +loss_with_kl_div(predictions[x], labels[x], predictions[y], labels[y], margin)

        #sum_loss = outerLoop(x, tf_l, predictions, labels, margin)
        loss = tf_l / (128.*2.)
        return loss
    
if __name__ == "__main__":
    with open('/home/patman/pa/1_Code/data/experiments/cluster_outputs/test_for_kld.pickle', 'rb') as f:
        (X, y, s_list) = pickle.load(f)

    epsilon = 1e-16
    test_pred = [[1., 2., 3.], [4., 2., 3.], [6., 3., 2.]]
    #test_targ = [1, 1, 6]
    test_targ = [[1., 0., 0.], [1., 0., 0.], [0., 0., 1.]]
    test_margin = 2.
    predictions = tf.placeholder('float', [128, None])
    targets = tf.placeholder('float', [128, None])
    margin = tf.placeholder('float', None)
    tf_p = tf.stack(X)
    tf_t = tf.stack(y)
    print y[1]
    print X[1]
    margin = tf.stack(test_margin)
    print "setting up predictions"
    result = pairwise_kl_divergence(targets, predictions)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    summary = tf.summary.merge_all()
    saver = tf.train.Saver()
    train_writer = tf.summary.FileWriter('../../data/experiments/graph/loss_graph', sess.graph)
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    print "running seesion"
    res = sess.run(result , feed_dict= {targets:y, predictions:X})
    print(res)
        