import tensorflow as tf
from math import pi

__all__ = [
    "calculate_variance_factor",
    "gaussian_initializer",
    "gaussian_regularizer"
]


TRUNCATED_NORMALIZER = 0.9544997361036415828294821039889939129352569580078125


def calculate_variance_factor(shape,
                              mode="FAN_AVG"):
    """
    Calculates a scaling factor based on the shape provided.

    Based on the mode the factor is:
        NULL:
            factor = 1.0
        FAN_IN:
            factor = Number of input neurons
        FAN_OUT:
            factor = Number of output neurons
        FAN_AVG:
            factor = Average of number of input and output neurons

    Args:
        shape: Shape. The shape based on which to calculate the factor.
        mode: String. The mode to use for calculation.

    Returns:
        The scaling factor for the shape provided.

    Raises:
        TypeError: if `mode` is not in ['NULL', 'FAN_IN', 'FAN_OUT', 'FAN_AVG'].
    """
    if mode not in ["NULL", "FAN_IN", "FAN_OUT", "FAN_AVG"]:
        raise TypeError("Unknown mode %s [NULL, FAN_IN, FAN_OUT, FAN_AVG]", mode)
    # Estimating fan_in and fan_out is not possible to do perfectly, but we try.
    # This is the right thing for matrix multiply and convolutions.
    if shape:
        if isinstance(shape[0], tf.Dimension):
            fan_in = float(shape[-2].value) if len(shape) > 1 else float(shape[-1].value)
            fan_out = float(shape[-1].value)
        else:
            fan_in = float(shape[-2]) if len(shape) > 1 else float(shape[-1])
            fan_out = float(shape[-1])
    else:
        fan_in = 1.0
        fan_out = 1.0
    for dim in shape[:-2]:
        if isinstance(dim, tf.Dimension):
            fan_in *= float(dim.value)
            fan_out *= float(dim.value)
        else:
            fan_in *= float(dim)
            fan_out *= float(dim)
    if mode == "NULL":
        factor = 1.0
    elif mode == 'FAN_IN':
        # Count only number of input connections.
        factor = fan_in
    elif mode == 'FAN_OUT':
        # Count only number of output connections.
        factor = fan_out
    elif mode == 'FAN_AVG':
        # Average number of inputs and output connections.
        factor = (fan_in + fan_out) / 2.0
    else:
        raise NotImplementedError()
    return factor


def gaussian_initializer(mode="FAN_AVG",
                         scale_factor=1.0,
                         mean=0.0,
                         truncated=False,
                         seed=None,
                         dtype=tf.float32):
    """
    Returns an initializer that generates tensors from a gaussian distribution.

    The initializer uses the following formula for calculating the variance of
    the gaussian distribution:

        variance = scale_factor / mode_factor

    Args:
        mode: String.  See `calculate_variance_factor`
        truncated: Bool.  Whether to use truncated gaussian instead of normal.
        scale_factor: Float.  Additional multiplicative scale of the variance.
        mean: Float.  Mean of the distribution.
        seed: A Python integer. Used to create random seeds. See
              @{tf.set_random_seed} for behavior.
        dtype: The data type. Only floating point types are supported.

    Returns:
        An initializer that generates tensors from a gaussian distribution.

    Raises:
        TypeError: if `dtype` is not a floating point type.
        TypeError: if `mode` is not in ['NULL', 'FAN_IN', 'FAN_OUT', 'FAN_AVG'].
        ValueError: if `scale_factor` is an integer.
    """
    scale_factor = tf.convert_to_tensor(scale_factor, dtype=tf.float32)
    if not dtype.is_floating:
        raise TypeError("Cannot create initializer for non-floating point type.")
    if mode not in ["NULL", "FAN_IN", "FAN_OUT", "FAN_AVG"]:
        raise TypeError("Unknown mode %s [NULL, FAN_IN, FAN_OUT, FAN_AVG]", mode)

    def _initializer(shape, dtype=dtype, partition_info=None):
        if not dtype.is_floating:
            raise TypeError("Cannot create initializer for non-floating point type.")
        stddev = tf.sqrt(scale_factor / calculate_variance_factor(shape, mode))
        if truncated:
            return tf.truncated_normal(shape, mean, stddev, dtype, seed=seed)
        else:
            return tf.random_normal(shape, mean, stddev, dtype, seed=seed)
    return _initializer


def gaussian_regularizer(mode="FAN_AVG",
                         scale_factor=1.0,
                         mean=0.0,
                         truncated=False):
    """
    Returns a regularizer based on the log probability of the equivalent
    gaussian initializer.

    The regularizer uses the following formula for calculating the variance of
    the gaussian distribution:

        variance = scale_factor / mode_factor

    Args:
        mode: String.  See `calculate_variance_factor`
        truncated: Bool.  Whether to use truncated gaussian instead of normal.
        scale_factor: Float.  Additional multiplicative scale of the variance.
        mean: Float.  Mean of the distribution.

    Returns:
        A regularizer that generates tensors from a gaussian distribution.

    Raises:
        TypeError: if `dtype` is not a floating point type.
        ValueError: if `scale_factor` is an integer.
    """
    if mode not in ["NULL", "FAN_IN", "FAN_OUT", "FAN_AVG"]:
        raise TypeError('Unknown mode %s [FAN_IN, FAN_OUT, FAN_AVG]', mode)

    scale_factor = tf.cast(scale_factor, tf.float32)

    # For a single tensor
    def _log_prob_single(tensor):
        stddev = tf.sqrt(scale_factor / calculate_variance_factor(tensor.shape, mode))
        z = (tensor - mean) / stddev
        log_prob_z = - (z ** 2 + tf.log(2 * pi)) / 2
        log_prob = tf.reduce_sum(log_prob_z)
        if truncated:
            from numpy import inf
            log_prob -= tf.log(TRUNCATED_NORMALIZER)
            invalid = tf.logical_or(tf.less_equal(z, -2), tf.greater_equal(z, 2))
            log_prob = tf.where(invalid, -inf, log_prob)
        # Return negative as this is a regularizer
        return - log_prob

    # For one or more tensors
    def _log_prob_fn(tensors):
        if isinstance(tensors, (list, tuple)):
            return sum(_log_prob_single(tensor) for tensor in tensors)
        else:
            return _log_prob_single(tensors)

    return _log_prob_fn


def double_mixture_initializer(mode="FAN_AVG",
                               scale_factor=0.5,
                               mean=0.0,
                               truncated=False,
                               seed=None,
                               dtype=tf.float32):
    """
    Returns an initializer that generates tensors from a mixture of two
    gaussian distributions.

    The initializer uses the following formula for calculating the variance of
    the gaussian distribution:

        variance = scale_factor / mode_factor
        mean = +/- 3 * sqrt(variance)

    Args:
        mode: String.  See `calculate_variance_factor`
        truncated: Bool.  Whether to use truncated gaussian instead of normal.
        scale_factor: Float.  Additional multiplicative scale of the variance.
        mean: Float.  Mean of the distribution.
        seed: A Python integer. Used to create random seeds. See
              @{tf.set_random_seed} for behavior.
        dtype: The data type. Only floating point types are supported.

    Returns:
        An initializer that generates tensors from a gaussian distribution.

    Raises:
        TypeError: if `dtype` is not a floating point type.
        TypeError: if `mode` is not in ['NULL', 'FAN_IN', 'FAN_OUT', 'FAN_AVG'].
        ValueError: if `scale_factor` is an integer.
    """
    scale_factor = tf.convert_to_tensor(scale_factor, dtype=tf.float32)
    if not dtype.is_floating:
        raise TypeError("Cannot create initializer for non-floating point type.")
    if mode not in ["NULL", "FAN_IN", "FAN_OUT", "FAN_AVG"]:
        raise TypeError("Unknown mode %s [NULL, FAN_IN, FAN_OUT, FAN_AVG]", mode)

    def _initializer(shape, dtype=dtype, partition_info=None):
        if not dtype.is_floating:
            raise TypeError("Cannot create initializer for non-floating point type.")
        stddev = tf.sqrt(scale_factor / calculate_variance_factor(shape, mode))
        if truncated:
            s1 = tf.truncated_normal(shape, mean - 3 * stddev, stddev, dtype, seed=seed)
            s2 = tf.truncated_normal(shape, mean + 3 * stddev, stddev, dtype, seed=seed)
        else:
            s1 = tf.random_normal(shape, mean - 3 * stddev, stddev, dtype, seed=seed)
            s2 = tf.random_normal(shape, mean + 3 * stddev, stddev, dtype, seed=seed)
        index = tf.cast(tf.random_uniform(shape, seed=seed) < 0.5, dtype=dtype)
        return index * s1 + (1.0 - index) * s2
    return _initializer


def double_mixture(mode="FAN_AVG",
                   scale_factor=1.0,
                   mean=0.0,
                   truncated=False):
    """
    Returns a regularizer based on the log probability of the equivalent
    mixture of two gaussian distributions initializer.

    The regularizer uses the following formula for calculating the variance of
    the gaussian distribution:

        variance = scale_factor / mode_factor
        mean = +/- 3 * sqrt(variance)

    Args:
        mode: String.  See `calculate_variance_factor`
        truncated: Bool.  Whether to use truncated gaussian instead of normal.
        scale_factor: Float.  Additional multiplicative scale of the variance.
        mean: Float.  Mean of the distribution.

    Returns:
        A regularizer that generates tensors from a gaussian distribution.

    Raises:
        TypeError: if `dtype` is not a floating point type.
        ValueError: if `scale_factor` is an integer.
    """
    if mode not in ["NULL", "FAN_IN", "FAN_OUT", "FAN_AVG"]:
        raise TypeError('Unknown mode %s [FAN_IN, FAN_OUT, FAN_AVG]', mode)
    scale_factor = tf.cast(scale_factor, tf.float32)

    # For a single tensor
    def _log_prob_single(tensor):
        stddev = tf.sqrt(scale_factor / calculate_variance_factor(tensor.shape, mode))
        z1 = (tensor - mean - 3 * stddev) / stddev
        log_prob_z1 = - (z1 ** 2 + tf.log(2 * pi)) / 2
        log_prob1 = tf.reduce_sum(log_prob_z1)
        z2 = (tensor - mean + 3 * stddev) / stddev
        log_prob_z2 = - (z2 ** 2 + tf.log(2 * pi)) / 2
        log_prob2 = tf.reduce_sum(log_prob_z2)
        if truncated:
            from numpy import inf
            log_prob1 -= tf.log(TRUNCATED_NORMALIZER)
            invalid = tf.logical_or(tf.less_equal(z1, -2), tf.greater_equal(z1, 2))
            log_prob1 = tf.where(invalid, -inf, log_prob1)
            log_prob2 -= tf.log(TRUNCATED_NORMALIZER)
            invalid = tf.logical_or(tf.less_equal(z2, -2), tf.greater_equal(z2, 2))
            log_prob2 = tf.where(invalid, -inf, log_prob2)
        # Return negative as this is a regularizer
        m = tf.maximum(log_prob1, log_prob2) - tf.log(2.0)
        log_prob1 -= m
        log_prob2 -= m
        log_prob = m + tf.log(tf.exp(log_prob1) + tf.exp(log_prob2))
        return - log_prob

    # For one or more tensors
    def _log_prob_fn(tensors):
        if isinstance(tensors, (list, tuple)):
            return sum(_log_prob_single(tensor) for tensor in tensors)
        else:
            return _log_prob_single(tensors)

    return _log_prob_fn
