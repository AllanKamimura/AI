import tensorflow as tf

def preprocessing(input_shape):
    input_signal = tf.keras.layers.Input(input_shape)    
    i = tf.keras.layers.Reshape((input_shape[:-1]))(input_signal)

    model = tf.keras.models.Model(
        inputs = input_signal,
        outputs = i,
        name = "preprocessing"
    )

    return model