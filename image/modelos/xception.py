# xception(2016)
# arxiv paper: https://arxiv.org/abs/1610.02357
# tensorflow implementation: https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/python/keras/applications/xception.py#L50-L315

import tensorflow as tf
def xception_convA(x, filters, activation = "relu"):
    x_short = tf.keras.layers.Conv2D(filters = filters, kernel_size = (1, 1), strides = (2, 2), kernel_initializer = "he_uniform",
                                     padding = "same", use_bias = False)(x)
    x_short = tf.keras.layers.BatchNormalization()(x_short)

    x = tf.keras.layers.SeparableConv2D(filters = filters, kernel_size = (3, 3), kernel_initializer = "he_uniform",
                                        depthwise_initializer = "he_uniform", pointwise_initializer = "he_uniform",
                                        padding = "same", use_bias = False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation = activation)(x)

    x = tf.keras.layers.SeparableConv2D(filters = filters, kernel_size = (3, 3), kernel_initializer = "he_uniform",
                                        depthwise_initializer = "he_uniform", pointwise_initializer = "he_uniform",
                                        padding = "same", use_bias = False)(x)
    x = tf.keras.layers.BatchNormalization()(x) 

    x = tf.keras.layers.MaxPooling2D(pool_size = (3, 3), strides = (2, 2),
                                     padding = "same")(x)
    y = tf.keras.layers.Add()([x, x_short])  

    return y

def xception_convB(x, filters, activation = "relu"):
    x_short = x

    x = tf.keras.layers.Activation(activation = activation)(x)
    x = tf.keras.layers.SeparableConv2D(filters = filters, kernel_size = (3, 3), kernel_initializer = "he_uniform",
                                        depthwise_initializer = "he_uniform", pointwise_initializer = "he_uniform",
                                        padding = "same", use_bias = False)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Activation(activation = activation)(x)
    x = tf.keras.layers.SeparableConv2D(filters = filters, kernel_size = (3, 3), kernel_initializer = "he_uniform",
                                        depthwise_initializer = "he_uniform", pointwise_initializer = "he_uniform",
                                        padding = "same", use_bias = False)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Activation(activation = activation)(x)
    x = tf.keras.layers.SeparableConv2D(filters = filters, kernel_size = (3, 3), kernel_initializer = "he_uniform",
                                        depthwise_initializer = "he_uniform", pointwise_initializer = "he_uniform",
                                        padding = "same", use_bias = False)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    y = tf.keras.layers.Add()([x, x_short])

    return y

def xception_convC(x, filters, activation = "relu"):
    x_short = tf.keras.layers.Conv2D(filters = filters, kernel_size = (1, 1), strides = (2, 2), kernel_initializer = "he_uniform",
                                     padding = "same", use_bias = False)(x)
    x_short = tf.keras.layers.BatchNormalization()(x_short)

    x = tf.keras.layers.Activation(activation = activation)(x)
    x = tf.keras.layers.SeparableConv2D(filters = 728, kernel_size = (3, 3), kernel_initializer = "he_uniform",
                                        depthwise_initializer = "he_uniform", pointwise_initializer = "he_uniform",
                                        padding = "same", use_bias = False)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Activation(activation = activation)(x)
    x = tf.keras.layers.SeparableConv2D(filters = filters, kernel_size = (3, 3), kernel_initializer = "he_uniform",
                                        depthwise_initializer = "he_uniform", pointwise_initializer = "he_uniform",
                                        padding = "same", use_bias = False)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPooling2D(pool_size = (3, 3), strides = (2, 2),
                                     padding = "same")(x)
    y = tf.keras.layers.Add()([x, x_short])  

    return y

def model(shape, pool):
    input = tf.keras.layers.Input(shape = shape)

    x = tf.keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), strides = (2, 2), kernel_initializer = "he_uniform",
                               use_bias = False)(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation = "relu")(x)

    x = tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), kernel_initializer = "he_uniform",
                               use_bias = False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation = "relu")(x)
    
    x = xception_convA(x = x, filters = 128)
    x = xception_convA(x = x, filters = 256)
    x = xception_convA(x = x, filters = 728)

    x = xception_convB(x = x, filters = 728)
    x = xception_convB(x = x, filters = 728)
    x = xception_convB(x = x, filters = 728)
    x = xception_convB(x = x, filters = 728)
    x = xception_convB(x = x, filters = 728)
    x = xception_convB(x = x, filters = 728)
    x = xception_convB(x = x, filters = 728)
    x = xception_convB(x = x, filters = 728)

    x = xception_convC(x = x, filters = 1024)

    x = tf.keras.layers.SeparableConv2D(filters = 1536, kernel_size = (3, 3), kernel_initializer = "he_uniform",
                                        depthwise_initializer = "he_uniform", pointwise_initializer = "he_uniform",
                                        padding = "same", use_bias = False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation = "relu")(x)

    x = tf.keras.layers.SeparableConv2D(filters = 2048, kernel_size = (3, 3), kernel_initializer = "he_uniform",
                                        depthwise_initializer = "he_uniform", pointwise_initializer = "he_uniform",
                                        padding = "same", use_bias = False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation = "relu")(x)

    if pool == "average":
        y = tf.keras.layers.GlobalAveragePooling2D()(x)
    if pool == "max":
        y = tf.keras.layers.GlobalMaxPooling2D()(x)

    model = tf.keras.Model(
        inputs = input,
        outputs = y,
        name = "xception_model"
    )

    return model