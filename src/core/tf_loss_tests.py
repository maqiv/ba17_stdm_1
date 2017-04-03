import tensorflow as tf
import numpy as np

epsilon = 1e-16
test_pred = [[1., 2., 3.], [4., 2., 3.], [6., 3., 2.]]
test_targ = [1, 1, 6]
test_margin = 2
predictions = tf.placeholder('float', [None, None])
targets = tf.placeholder('float', [None])
margin = tf.placeholder('float', None)
tf_p = tf.stack(test_pred)
tf_t = tf.stack(test_targ)
tf_l = tf.Variable(0. , name='loss')

def return_zero():
    return tf.constant(0.)

def return_one():
    return tf.constant(1.)

def loss_with_kl_div(P, xp, Q, xq, margin):
    P += epsilon
    Q += epsilon

    Is = tf.cond(tf.equal(xq, xp), return_one, return_zero)
    Ids = abs(Is-1)

    KLPQ = tf.reduce_sum(P * tf.log(P / Q))
    KLQP = tf.reduce_sum(Q * tf.log(Q / P))
    lossPQ = Is * KLPQ + Ids * tf.maximum(0., margin - KLPQ)
    lossQP = Is * KLQP + Ids * tf.maximum(0., margin - KLQP)
    L = lossPQ + lossQP
    print tf.Print(L, [L])
    return L

def test(x):
    return x
	

def body(x, tf_l):
    print 'body'
    #tf_l = tf.add(tf_l,loss_with_kl_div(tf_p[x],tf_t[x] , tf_p[x+1], tf_t[x+1], tf.constant(2.)))
    #tf_l = loss_with_kl_div(tf_p[x],tf_t[x] , tf_p[x+1], tf_t[x+1], tf.constant(2.))
    #return tf.add(x, 1), tf_l

    def innerLoop(y, tf_l):
        tf_l = tf.add(tf_l, loss_with_kl_div(tf_p[x], tf_t[x], tf_p[y], tf_t[y], tf.constant(2.)))
        y += 1
        return y, tf_l
   
    def innerLoop_cond(y , tf_l):
        return tf.less(y, tf_p.get_shape()[0])

    y = tf.constant(0)
    tut = tf.while_loop(innerLoop, innerLoop_cond, (y, tf_l))
    return tf.add(x, 1), tut[1]

def condition(x, tf_l):
    print "cond"
    print tf_p.get_shape()[0]
    return x < tf_p.get_shape()[0]


    


if __name__ == "__main__":
    test_pred = [[1, 2, 3], [4, 2, 3], [6, 3, 2]]
    test_targ = [1, 1, 6]
    test_margin = 2


    predictions = tf.placeholder('float', [None, None])
    targets = tf.placeholder('float', [None])
    margin = tf.placeholder('float', None)

    res = tf.map_fn(test, predictions)
    x = tf.constant(0)

    with tf.Session():
        tf.initialize_all_variables().run()
        print x
        result = tf.while_loop(condition, body, [x, tf_l])
        tf_r = tf.div(result[1], 3)
        print(result[1].eval())
        print(tf_r.eval())


    #init = tf.global_variables_initializer()
    #sess = tf.Session()
    #sess.run(init)
    #loss_value = sess.run(res, feed_dict={predictions: test_pred})
    #print loss_value

