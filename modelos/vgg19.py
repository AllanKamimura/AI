# vgg19(2014)
# arxiv paper: https://arxiv.org/abs/1409.1556
# tensorflow implementation: https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/python/keras/applications/vgg19.py#L45-L231

import tensorflow as tf

def block(x, n_convs, filters, block_name, kernel_size = (3, 3), activation = "relu", pool_size = (2, 2), pool_stride = (2, 2)):
  '''
  Defines a block in the VGG network.

  Returns:
    tensor containing the max-pooled output of the convolutions
  '''

  for i in range(n_convs):
      x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding = "same", name="{}_conv{}".format(block_name, i + 1),
                                 kernel_initializer = "he_uniform")(x)
    
  x = tf.keras.layers.MaxPooling2D(pool_size = pool_size, strides = pool_stride, name = "{}_pool{}".format(block_name, i+1))(x)

  return x

def model(shape, pool):
    '''
    Build the blocks into a sequential model

    Returns:
        a sequential model
    '''
    input = tf.keras.layers.Input(shape = shape, name = "input_layer")
    x = input

    x = block(x, n_convs = 2, filters = 64, block_name = "block1")
    x = block(x, n_convs = 2, filters = 128, block_name = "block2")
    x = block(x, n_convs = 4, filters = 256, block_name = "block3")
    x = block(x, n_convs = 4, filters = 512, block_name = "block4")
    x = block(x, n_convs = 4, filters = 512, block_name = "block5")

    if pool == "average":
        y = tf.keras.layers.GlobalAveragePooling2D()(x)
    if pool == "max":
        y = tf.keras.layers.GlobalMaxPooling2D()(x)

    model = tf.keras.Model(
        inputs = input,
        outputs = y,
        name = "vgg19_model"
    )
    return model