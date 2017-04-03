import tensorflow as tf
import numpy as np

epsilon = 1e-16
tf_l = tf.Variable(0. , name='loss')

def return_zero():
    return tf.constant(0.)

def return_one():
    return tf.constant(1.)

def loss_with_kl_div(P, xp, Q, xq, margin):
    P += epsilon
    Q += epsilon
    print xq.get_shape()
    print xp.get_shape()
    print tf.equal(xq, xp)
    Is = tf.cond(tf.reduce_all(tf.equal(xq, xp)), return_one, return_zero)
    Ids = abs(Is-1)

    KLPQ = tf.reduce_sum(P * tf.log(P / Q))
    KLQP = tf.reduce_sum(Q * tf.log(Q / P))
    lossPQ = Is * KLPQ + Ids * tf.maximum(0., margin - KLPQ)
    lossQP = Is * KLQP + Ids * tf.maximum(0., margin - KLQP)
    L = lossPQ + lossQP
    return L	

def outerLoop(x, tf_l, predictions, labels, margin):
    print 'body'
    print predictions.get_shape()
    print labels.get_shape()

    def innerLoop(y,x, tf_l, predictions, labels, margin):
        tf_l = tf.add(tf_l, loss_with_kl_div(predictions[x], labels[x], predictions[y], labels[y], margin))
        y += 1
        return y, x, tf_l, predictions, labels, margin
   
    def innerLoop_cond(y ,x, tf_l, predictions, labels, margin):
        return tf.less(y, tf.shape(predictions)[0])

    y = tf.constant(0)
    res = tf.while_loop(innerLoop_cond, innerLoop, [y,x,tf_l, predictions, labels, margin])
    return tf.add(x, 1), res[2], predictions, labels, margin

def outerLoop_condition(x, tf_l, predictions, labels, margin):
    print "cond"
    return tf.less(x, tf.shape(predictions)[0])

def pairwise_kl_divergence(labels, predictions):
    x = tf.constant(0)
    margin = tf.constant(2.)
    tf.shape(predictions)
    #tf_l = tf.Variable(0. , name='loss')
    sum_loss = tf.while_loop(outerLoop_condition, outerLoop, [x, tf_l, predictions, labels, margin])
    print sum_loss[0]
    loss = sum_loss[1] / (tf.to_float(tf.shape(predictions)[0]) *2.)
    return loss
    
if __name__ == "__main__":
    epsilon = 1e-16
    test_pred = [[1., 2., 3.], [4., 2., 3.], [6., 3., 2.]]
    #test_targ = [1, 1, 6]
    test_targ = [[1., 0., 0.], [1., 0., 0.], [0., 0., 1.]]
    test_margin = 2.
    predictions = tf.placeholder('float', [None, None])
    targets = tf.placeholder('float', [None])
    margin = tf.placeholder('float', None)
    tf_p = tf.stack(test_pred)
    tf_t = tf.stack(test_targ)
    margin = tf.stack(test_margin)
    print tf_p.get_shape()[0]
    with tf.Session():
        tf.initialize_all_variables().run()
        result = pairwise_kl_divergence(tf_t, tf_p)
        print(result.eval())
        