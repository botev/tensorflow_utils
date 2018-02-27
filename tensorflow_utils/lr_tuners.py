import tensorflow as tf


__all__ = [
    "bayesian_tuner"
]


def bayesian_tuner(learning_rate, global_step,
                   loss, buffer_size,
                   decay_rate_down=0.5, decay_rate_up=1.2,
                   mu_0=0.0, precision_0=1.0,
                   a0=1.0, b0=1.0,
                   dtype=tf.float32,
                   name=None):
    if global_step is None:
        raise ValueError("global_step is required for inverse_time_decay.")
    with tf.variable_scope("BayesianTuner", reuse=tf.AUTO_REUSE):
        # Convert all to tensors
        global_step = tf.cast(global_step, tf.int64)
        decay_rate_down = tf.convert_to_tensor(decay_rate_down, dtype=dtype)
        decay_rate_up = tf.convert_to_tensor(decay_rate_up, dtype=dtype)
        mu_0 = tf.convert_to_tensor(mu_0, dtype=dtype)
        precision_0 = tf.convert_to_tensor(precision_0, dtype=dtype)
        a0 = tf.convert_to_tensor(a0, dtype=dtype)
        b0 = tf.convert_to_tensor(b0, dtype=dtype)
        # Make variables
        learning_rate = tf.get_variable("learning_rate", shape=(), dtype=dtype,
                                        initializer=tf.constant_initializer(learning_rate))
        buffer = tf.get_variable("buffer", shape=(buffer_size, ), dtype=loss.dtype)
        # Update buffer
        index = tf.mod(tf.cast(global_step, tf.int64), buffer_size)
        y = tf.scatter_update(buffer, index, loss)
        update_op = tf.assign(buffer, y)
        # Remove the mean of y
        y = y - tf.reduce_mean(y)

        # Set x and x^T x
        x = tf.range(0, buffer_size, dtype=dtype) - (buffer_size - 1) / 2
        x /= (buffer_size - 1) / 2
        x_t_x = tf.reduce_sum(x ** 2)
        # Equations for the parameters of the Normal-Inverse-Gamma
        precision_n = x_t_x + precision_0
        mu_n = tf.reduce_sum(precision_0 * mu_0 + x * y) / precision_n
        a_n = a0 + buffer_size / 2
        b_n = b0 + (tf.reduce_sum(y * y) + mu_0 * precision_0 * mu_0 - mu_n * precision_n * mu_n) / 2
        c_n = tf.sqrt(a_n * precision_n / b_n)
        learning_rate = tf.Print(learning_rate, [mu_n, c_n], "params")
        decay_rate_up = tf.cast(decay_rate_up, dtype)
        decay_rate_down = tf.cast(decay_rate_down, dtype)

        return learning_rate, update_op
