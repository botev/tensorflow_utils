import tensorflow as tf

__all__ = [
    "huber_loss",
    "soft_huber",
    "student"
]


def huber_loss(x, y, a=1.0, eps=1.0):
    """
    if |x| < eps:
        f(x) = a x^2 / 2.0
    else:
        f(x) = a * eps * (|x| - eps / 2.0)
    :param x:
    :param y:
    :param a:
    :param eps:
    :return:
    """
    abs_error = tf.abs(x - y)
    q = tf.minimum(abs_error, eps)
    f = a * tf.square(q) / 2.0 + a * eps * abs_error - a * eps * q
    return f


def soft_huber(x, y, alpha=1.0):
    """
    A soft variant of the huber loss:
        f(x) = alpha * log(cosh(x/alpha))
             = |x| + alpha * softplus( - 2.0 |x| / alpha) - alpha * log(2.0)
        f'(x) = tanh(x / alpha)

    The results are similar to huber_loss(x, y, a=1/eps,eps=eps)
    """
    abs_error = tf.abs(x - y)
    f = abs_error + alpha * tf.nn.softplus(- 2.0 * abs_error / alpha) - alpha * tf.log(2.0)
    return f


def student(x, y, v=1.0):
    """
    The centered Student-t negative log-likelihood:
        f(x) = (v + 1.0) * log(1 + x^2/v) / 2.0
    """
    return (v + 1.0) * tf.log(1 + tf.square(x - y) / v) / 2.0
