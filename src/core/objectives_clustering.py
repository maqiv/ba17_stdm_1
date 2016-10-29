import theano
import theano.tensor as T

epsilon = 1e-16  # to avoid log(0) or division by 0


def loss_with_kl_div(P, xp, Q, xq, margin):
    P += epsilon
    Q += epsilon

    Is = theano.ifelse.ifelse(T.eq(xq, xp), 1., 0.)
    Ids = abs(Is-1)

    KLPQ = T.sum(P * T.log(P / Q))
    KLQP = T.sum(Q * T.log(Q / P))
    lossPQ = Is * KLPQ + Ids * T.max([0, margin - KLPQ])
    lossQP = Is * KLQP + Ids * T.max([0, margin - KLQP])
    L = lossPQ + lossQP

    return L


def foreach(predictions, targets, margin):
    result, updates = theano.scan(fn=step,
                                  sequences=[predictions, targets],
                                  non_sequences=[predictions, targets, margin], strict=True)
    return result


def step(prediction, target, predictions, targets, margin):
    result, updates = theano.scan(fn=loss_with_kl_div,
                                  sequences=[predictions, targets],
                                  non_sequences=[prediction, target, margin], strict=True)
    return result


def mean_loss_kl_div(predictions, targets, margin):
    err_mat = foreach(predictions, targets, margin)
    length = err_mat.shape[0]
    mean_err = err_mat.sum() / ((length - 1) * length)
    return mean_err


if __name__ == "__main__":
    # Some test code
    test_pred = [[1, 2, 3], [4, 2, 3], [6, 3, 2]]
    test_targ = [1, 1, 6]
    test_margin = 2

    predictions = T.fmatrix('p')
    targets = T.dvector('t')
    margin = T.scalar('margin')

    loss = mean_loss_kl_div(predictions, targets, margin)
    loss_fun = theano.function([predictions, targets, margin], loss)
    mean_err = loss_fun(test_pred, test_targ, test_margin)

    foreach_prep = foreach(predictions, targets, margin)
    foreach_fun = theano.function([predictions, targets, margin], foreach_prep)
    err_mat = foreach_fun(test_pred, test_targ, test_margin)
    err = err_mat.sum() / ((len(err_mat) - 1) * len(err_mat))


    def loss(predictions, targets, margin, f):
        assert len(predictions) == len(targets)
        L_sum = 0
        for i in range(len(predictions)):
            for j in range(len(predictions)):
                L_sum += f(predictions[i], targets[i], predictions[j], targets[j], margin)
        return L_sum / (2 * len(predictions))


    xp = T.scalar('xp')
    xq = T.scalar('xq')
    p = T.fvector('P')
    q = T.fvector('Q')
    result = loss_with_kl_div(p, xp, q, xq, margin)
    f = theano.function([p, xp, q, xq, margin], result)
    mean_np = loss(test_pred, test_targ, test_margin, f)

    assert (mean_err == err == mean_np)
    print('Run without errors!')
