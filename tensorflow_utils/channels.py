import tensorflow as tf
from tensorflow.python.client import device_lib

__all__ = [
    "default_device",
    "data_format",
    "channels_axis",
    "to_channel_format",
    "from_channel_format",
    "index_channel_axis",
    "slice_chanel_axis",
]


def default_device():
    """
    Returns:
        What is the default tensorflow device on your machine.
        Order of precedence is [TPU, GPU, CPU].
    """
    local_device_protos = device_lib.list_local_devices()
    if len([x.name for x in local_device_protos if x.device_type == 'TPU']) > 0:
        return "tpu"
    elif len([x.name for x in local_device_protos if x.device_type == 'GPU']) > 0:
        return "gpu"
    else:
        return "cpu"


def data_format(device=None):
    """
    Args:
        device: String.  Non-default device to use.
    Returns:
         The correct data format for convolutions based on the device.
    """
    device = device or default_device()
    if device == "cpu":
        return "channels_last"
    elif device == "gpu":
        return "channels_first"
    else:
        raise NotImplementedError()


def channels_axis(device=None):
    """
    Args:
        device: String.  Non-default device to use.
    Returns:
         The correct channels axis for convolutions based on the device.
    """
    if data_format(device) == "channels_first":
        return 1
    else:
        return -1


def to_channel_format(x, x_format="nchw", device=None):
    """
    Converts the input to the correct channels format based on the device.

    Args:
        x: Tensor
        x_format: String.  One of ['channel', 'ncwh', 'nwhc']
        device: String.  Non-default device to use.
    Returns:
         The input `x` with correctly ordered axes.
    """
    if x_format.lower() == "channel" or x.shape.ndims == 2:
        return x
    elif x.shape.ndims == 3:
        return tf.expand_dims(x, channels_axis(device))
    elif x_format.lower() == "ncwh":
        if data_format() == "channels_first":
            return x
        else:
            return tf.transpose(x, [0, 2, 3, 1])
    elif x_format.lower() == "nwhc":
        if data_format() == "channels_last":
            return x
        else:
            return tf.transpose(x, [0, 3, 1, 2])


def from_channel_format(x, target_format="ncwh", device=None):
    """
    Converts the input to the target channels format based on the device.
    The input is always assumed to be in the correct channels format.

    Args:
        x: Tensor
        target_format: String.  One of ['channel', 'nwh', 'ncwh', 'nwhc']
        device: String.  Non-default device to use.
    Returns:
         The input `x` with correctly ordered axes.
    """
    if target_format.lower() == "channel" or x.shape.ndims == 2:
        return x
    elif x.shape.ndims == 3:
        raise NotImplementedError()
    elif target_format == "nwh":
        return tf.reshape(x, (x.shape[0], x.shape[2], x.shape[3]))
    elif target_format.lower() == "ncwh":
        if data_format(device) == "channels_first":
            return x
        else:
            return tf.transpose(x, [0, 3, 1, 2])
    elif target_format.lower() == "nwhc":
        if data_format(device) == "channels_last":
            return x
        else:
            return tf.transpose(x, [0, 2, 3, 1])


def index_channel_axis(x, i, device=None):
    """
    Indexes the tensor `x` using `i` on the correct channels axis.

    Args:
        x: Tensor
        i: Tensor
        device: String.  Non-default device to use.
    Returns:
         The input `x` with correctly ordered axes.
    """
    if x.shape.ndims == 1:
        raise NotImplementedError()
    if x.shape.ndims == 2:
        return x[:, i]
    elif x.shape.ndims == 3:
        raise NotImplementedError()
    elif channels_axis(device) == 1:
        return x[:, i]
    else:
        return x[:, :, :, i]


def slice_chanel_axis(x, slc, device=None):
    """
    Slices the tensor `x` using `slc` on the correct channels axis.

    Args:
        x: Tensor
        slc: Slice
        device: String.  Non-default device to use.
    Returns:
         The input `x` with correctly ordered axes.
    """
    if x.shape.ndims == 1:
        raise NotImplementedError()
    if x.shape.ndims == 2:
        return x[:, slc]
    elif x.shape.ndims == 3:
        raise NotImplementedError()
    else:
        full_slice = [slice(None)] * x.shape.ndims
        full_slice[channels_axis(device)] = slc
        return x[full_slice]
