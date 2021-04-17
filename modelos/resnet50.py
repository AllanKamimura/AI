# resnet-50(2015)
# arxiv paper: https://arxiv.org/abs/1512.03385
# tensorflow implementation: https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/python/keras/applications/resnet.py#L453-L472

import tensorflow as tf

def res_conv(x, filters, activation = "relu", epsilon = 1.001e-5, strides = (1, 1)):
    x_short = tf.keras.layers.Conv2D(filters = 4 * filters, kernel_size = (1, 1), kernel_initializer = "he_uniform",
                                     strides = strides)(x)
    x_short = tf.keras.layers.BatchNormalization(epsilon = epsilon)(x_short)

    x = tf.keras.layers.Conv2D(filters = filters, kernel_size = (1, 1), kernel_initializer = "he_uniform",
                               strides = strides)(x)
    x = tf.keras.layers.BatchNormalization(epsilon = epsilon)(x)
    x = tf.keras.layers.Activation(activation = activation)(x)

    x = tf.keras.layers.Conv2D(filters = filters, kernel_size = (3, 3), padding = "same", kernel_initializer = "he_uniform")(x)
    x = tf.keras.layers.BatchNormalization(epsilon = epsilon)(x)
    x = tf.keras.layers.Activation(activation = activation)(x)

    x = tf.keras.layers.Conv2D(filters = 4 * filters, kernel_size = (1, 1), kernel_initializer = "he_uniform")(x)
    x = tf.keras.layers.BatchNormalization(epsilon = epsilon)(x)

    x = tf.keras.layers.Add()([x, x_short])
    y = tf.keras.layers.Activation(activation = activation)(x)
    return y

def res_id(x, filters, activation = "relu", epsilon = 1.001e-5):
    x_short = x

    x = tf.keras.layers.Conv2D(filters = filters, kernel_size = (1, 1), kernel_initializer = "he_uniform")(x)
    x = tf.keras.layers.BatchNormalization(epsilon = epsilon)(x)
    x = tf.keras.layers.Activation(activation = activation)(x)

    x = tf.keras.layers.Conv2D(filters = filters, kernel_size = (3, 3), padding = "same", kernel_initializer = "he_uniform")(x)
    x = tf.keras.layers.BatchNormalization(epsilon = epsilon)(x)
    x = tf.keras.layers.Activation(activation = activation)(x)

    x = tf.keras.layers.Conv2D(filters = 4 * filters, kernel_size = (1, 1), kernel_initializer = "he_uniform")(x)
    x = tf.keras.layers.BatchNormalization(epsilon = epsilon)(x)

    x = tf.keras.layers.Add()([x, x_short])
    y = tf.keras.layers.Activation(activation = activation)(x)
    return y

def model(shape, pool):
    input = tf.keras.layers.Input(shape = shape, name = "input_layer")

    x = tf.keras.layers.ZeroPadding2D(padding=((3, 3)))(input)

    x = tf.keras.layers.Conv2D(filters = 64, kernel_size = (7, 7), kernel_initializer = "he_uniform",
                               strides = (2, 2))(x)
    x = tf.keras.layers.BatchNormalization(epsilon = 1.001e-5)(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = tf.keras.layers.MaxPooling2D(3, strides = 2)(x)

    x = res_conv(x = x, filters = 64)
    x = res_id(x = x, filters = 64)
    x = res_id(x = x, filters = 64)

    x = res_conv(x = x, filters = 128, strides = (2, 2))
    x = res_id(x = x, filters = 128)
    x = res_id(x = x, filters = 128)
    x = res_id(x = x, filters = 128)

    x = res_conv(x = x, filters = 256, strides = (2, 2))
    x = res_id(x = x, filters = 256)
    x = res_id(x = x, filters = 256)
    x = res_id(x = x, filters = 256)
    x = res_id(x = x, filters = 256)
    x = res_id(x = x, filters = 256)

    x = res_conv(x = x, filters = 512, strides = (2, 2))
    x = res_id(x = x, filters = 512)
    x = res_id(x = x, filters = 512)

    if pool == "average":
        y = tf.keras.layers.GlobalAveragePooling2D()(x)
    if pool == "max":
        y = tf.keras.layers.GlobalMaxPooling2D()(x)

    model = tf.keras.Model(
        inputs = input,
        outputs = y,
        name = "resnet_model"
    )

    return model