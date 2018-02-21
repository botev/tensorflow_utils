import tensorflow as tf
from .channels import data_format, channels_axis
from .priors import gaussian_initializer

__all__ = [
    "residual_block",
    "unet_resnet"
]


def residual_block(x_in, name,
                   filters=None,
                   activation=tf.nn.selu,
                   kernel_size=(3, 3),
                   strides=(1, 1),
                   transposed=False,
                   bottleneck=None,
                   use_batch_norm=True,
                   training=True,
                   bottleneck_factor=4):
    """ A Residual Block as described in https://arxiv.org/abs/1512.03385 """
    axis = channels_axis()
    filters = x_in.shape[axis] if filters is None else filters
    init_conv = tf.layers.conv2d_transpose if transposed else tf.layers.conv2d
    if isinstance(strides, int):
        strides = (strides, strides)

    with tf.variable_scope(name):
        x = x_in
        if use_batch_norm:
            # Batch Norm
            x = tf.layers.batch_normalization(x, axis=axis, name="bn_1", training=training)
        # Nonlinearity
        x = activation(x)

        if bottleneck:
            # With a bottleneck layer
            bottleneck_filters = filters // bottleneck_factor

            # 1 x 1 convolution to reduce channels
            x = init_conv(x, name="conv_1",
                          filters=bottleneck_filters,
                          kernel_size=(1, 1),
                          strides=strides,
                          use_bias=(not use_batch_norm),
                          kernel_initializer=gaussian_initializer(),
                          data_format=data_format(), padding="SAME")
            if use_batch_norm:
                # Batch Norm
                x = tf.layers.batch_normalization(x, axis=axis, name="bn_2", training=training)
            # Nonlinearity
            x = activation(x)
            # 3 x 3 convolution
            x = tf.layers.conv2d(x, name="conv_2",
                                 filters=bottleneck_filters,
                                 kernel_size=kernel_size,
                                 use_bias=(not use_batch_norm),
                                 kernel_initializer=gaussian_initializer(),
                                 data_format=data_format(), padding="SAME")

            if use_batch_norm:
                # Batch Norm
                x = tf.layers.batch_normalization(x, axis=axis, name="bn_3", training=training)
            # Nonlinearity
            x = activation(x)
            # 1 x 1 convolution to increase channels
            x = tf.layers.conv2d(x, name="conv_3",
                                 filters=filters,
                                 kernel_size=(1, 1),
                                 use_bias=(not use_batch_norm),
                                 kernel_initializer=gaussian_initializer(),
                                 data_format=data_format(), padding="SAME")
        else:
            # 3 x 3 convolution
            x = init_conv(x, name="conv_1",
                          filters=filters,
                          kernel_size=kernel_size,
                          strides=strides,
                          use_bias=(not use_batch_norm),
                          kernel_initializer=gaussian_initializer(),
                          data_format=data_format(), padding="SAME")
            if use_batch_norm:
                # Batch Norm
                x = tf.layers.batch_normalization(x, axis=axis, name="bn_2", training=training)
            # Nonlinearity
            x = activation(x)
            # 3 x 3 convolution
            x = tf.layers.conv2d(x, name="conv_2",
                                 filters=filters,
                                 kernel_size=kernel_size,
                                 use_bias=(not use_batch_norm),
                                 kernel_initializer=gaussian_initializer(),
                                 data_format=data_format(), padding="SAME")
        if tuple(strides) != (1, 1) or filters != x_in.shape[axis]:
            # If reduced spatial size or different number of channels we need special shortcut connections
            x_in = init_conv(x_in, name="shortcut",
                             filters=filters,
                             kernel_size=strides,
                             strides=strides,
                             kernel_initializer=gaussian_initializer(),
                             data_format=data_format(), padding="SAME")
        # Identity mapping
        return x + x_in


def unet_resnet(x, num_main_blocks, num_res_blocks,
                init_filters, output_filters=None,
                res_block_func=residual_block,
                up_sampling_func=None,
                num_res_blocks_bottleneck=None,
                skip_connections=True,
                in_kernel_size=5,
                out_kernel_size=5,
                filters_factor=2,
                spatial_factor=2,
                name=None,
                reuse=tf.AUTO_REUSE,
                **kwargs):
    # Sort out default values
    name = "resnet" if name is None else name
    if not isinstance(num_res_blocks, (list, tuple)):
        num_res_blocks = [num_res_blocks] * num_main_blocks
    if num_res_blocks_bottleneck is None:
        num_res_blocks_bottleneck = num_res_blocks[-1]
    output_filters = init_filters if output_filters is None else output_filters
    if up_sampling_func is None:
        up_sampling_func = tf.layers.conv2d_transpose

    # Values for skip connections
    skips = list()
    with tf.variable_scope(name, reuse=reuse):
        filters = init_filters
        # Initial convolution of the image
        x = tf.layers.conv2d(x, filters, name="conv_in",
                             kernel_size=in_kernel_size, strides=1,
                             kernel_initializer=gaussian_initializer(),
                             data_format=data_format(), padding="SAME")
        # First block does not reduce spatial dimensions or increase filter size
        with tf.variable_scope("block_0"):
            # Repeated residual blocks
            for res_block in range(num_res_blocks[0]):
                x = res_block_func(x, filters=filters, name="res_" + str(res_block), **kwargs)
            # Append for skip connections
            skips.append(x)

        # Normal blocks
        for main_block, res_blocks in enumerate(num_res_blocks[1:]):
            with tf.variable_scope("block_" + str(main_block + 1)):
                # Increase number of filters
                filters *= filters_factor
                # Initial convolution to shrink spatial dimensions
                x = res_block_func(x, filters=filters, name="res_0", strides=spatial_factor, **kwargs)
                # Repeated residual blocks
                for res_block in range(1, res_blocks):
                    x = res_block_func(x, filters=filters, name="res_" + str(res_block), **kwargs)
                # Append for skip connections
                skips.append(x)

        # Bottleneck transformation does not go for skips
        with tf.variable_scope("bottleneck"):
            # Increase number of filters
            filters *= filters_factor
            # Initial convolution to shrink spatial dimensions
            x = res_block_func(x, filters=filters, name="res_0", strides=spatial_factor, **kwargs)
            # Repeated residual blocks
            for res_block in range(1, num_res_blocks_bottleneck):
                x = res_block_func(x, filters=filters, name="res_" + str(res_block), **kwargs)

        iterator = list(zip(range(num_main_blocks), num_res_blocks, skips))
        for main_block, res_blocks, skip in iterator[::-1]:
            with tf.variable_scope("up_block_" + str(main_block + 1)):
                # Reduce number of filters
                filters //= filters_factor
                # Up sample
                x = up_sampling_func(x, filters, kernel_size=3,
                                     strides=spatial_factor,
                                     name="up_sample_" + str(main_block + 1),
                                     data_format=data_format(), padding="SAME",
                                     **kwargs)
                if skip_connections:
                    # Skip connection
                    x = tf.concat([x, skip], axis=channels_axis())
                # Repeated residual blocks
                for res_block in range(res_blocks):
                    x = res_block_func(x, filters=filters, name="res_" + str(res_block), **kwargs)

        # Final convolution of the image
        x = tf.layers.conv2d(x, output_filters, name="conv_out",
                             kernel_size=out_kernel_size,
                             kernel_initializer=gaussian_initializer(),
                             data_format=data_format(), padding="SAME")
        return x
