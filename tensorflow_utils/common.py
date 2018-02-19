from warnings import warn
import tensorflow as tf
from .channels import channels_axis

__all__ = [
    "activation_from_string",
    "flatten_tensors",
    "un_flatten_tensors",
    "make_custom_getter",
    "FlattenFunctionWrapper",
]


def celu(x, axis=None):
    """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
    if axis is not None:
        return tf.nn.elu(tf.concat(values=[x, -x], axis=axis))
    if x.shape.ndims == 2:
        return tf.nn.elu(tf.concat(values=[x, -x], axis=1))
    else:
        return tf.nn.elu(tf.concat(values=[x, -x], axis=channels_axis()))


def activation_from_string(name, *args, **kwargs):
    """
    Returns a callable activation function.

    *** If name is not a string will return it instead and print a warning ***

    Args:
        name: String. Name of the activation function.
        args: Any extra arguments to pass to the activation function.
        kwargs: Any extra keyword arguments to pass to the activation function.

    Returns:
        A callable activation function.

    Raises:
        NotImplementedError: if the `name` is not found.
    """
    if not isinstance(name, str):
        warn("activation_from_string was given not a string, but " + str(name))
        return name
    elif name == "tanh":
        return tf.tanh
    elif name == "sigmoid":
        return tf.sigmoid
    elif name == "relu":
        return tf.nn.relu
    elif name == "crelu":
        return tf.nn.crelu
    elif name == "lrelu":
        return lambda x: tf.nn.leaky_relu(x, *args, **kwargs)
    elif name == "lrelu_01":
        return lambda x: tf.nn.leaky_relu(x, 0.1)
    elif name == "elu":
        return tf.nn.elu
    elif name == "celu":
        return  celu
    elif name == "selu":
        return tf.nn.selu
    else:
        raise NotImplementedError()


def flatten_tensors(tensors):
    """
    Flattens and concatenates all tensor in order into a single vector.

    Args:
        tensors: List. All tensors to concatenate.

    Returns:
        A single value representing the tensors concatenation.

    Raises:
        NotImplementedError: if the `name` is not found.
    """
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]
    tensors = [tf.reshape(t, [t.shape.num_elements()]) for t in tensors]
    return tf.concat(tensors, axis=0)


def un_flatten_tensors(tensors, value):
    """
    Reverses the `flatten_tensor` function for the vector provided.

    Args:
        tensors: List.  A list of tensor with original shapes.
        value: Vector.  The value that should be redistributed to tensors.

    Returns:
        A list of tensors derived from value.
    """
    i = 0
    values = list()
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]
    for t in tensors:
        t_value = value[i: i + t.shape.num_elements()]
        values.append(tf.reshape(t_value, t.shape))
        i += t.shape.num_elements()
    return values


def make_custom_getter(dictionary, raise_not_in_dict=False):
    """
    Returns a custom getter function which can be used in @{tf.get_variable}.

    The functions first attempts to retrieve the requested variables via their
    names from the dictionary provided and if they are not in it it falls back
    to the standard getter function.

    Args:
        dictionary: Dict.  A dictionary mapping shared names of variables to
            values.
        raise_not_in_dict: Bool.  Whether to raise an error if any requested
            variable is not in the dictionary.

    Returns:
        A custom getter function.
    """
    if dictionary is None or len(dictionary) == 0:
        return None

    def custom_getter(name, getter, shape, dtype, **kwargs):
        if dictionary.get(name) is not None:
            value = dictionary[name]
            if value.dtype != dtype:
                raise ValueError("The provided variable is not the same dtype.")
            if not value.shape == shape and shape is not None:
                raise ValueError("The provided variable has shape " + str(value.shape) +
                                 ", but the variable has shape " + str(shape))
            return value
        elif raise_not_in_dict:
            raise ValueError("The variable %s was not provided a value and "
                             "raise_not_in_dict is set to True." % (name, ))
        else:
            return getter(name=name, shape=shape, dtype=dtype, **kwargs)

    return custom_getter


class FlattenFunctionWrapper(object):
    """
    Takes a single tensorflow function with parameters provided and transforms
    it to a callable function which to take as inputs an arbitrary single
    vector value representing the flattened values of all parameters.

    Args:
        func: Callable.  The original function.
        parameters: List.  The original parameters of the function.
        use_parameter_names: Bool.  If all parameters are tensorflow variables
            we can use their names as keyword arguments to `func`.
    """
    def __init__(self, func, parameters, use_parameter_names=True):
        self.parameters = parameters
        self.func = func
        self.use_parameter_names = use_parameter_names

    @property
    def x(self):
        return flatten_tensors(self.parameters)

    def __call__(self, x_value):
        values = un_flatten_tensors(self.parameters, x_value)
        if isinstance(self.parameters, dict):
            kwargs = dict((k, value) for (k, _), value in zip(self.parameters.items(), values))
            return self.func(**kwargs)
        elif self.use_parameter_names:
            kwargs = dict((p.name, value) for p, value in zip(self.parameters, values))
            return self.func(**kwargs)
        else:
            return self.func(*values)


