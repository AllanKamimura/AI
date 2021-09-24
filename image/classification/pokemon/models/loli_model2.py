import tensorflow as tf

height, width = 224,224

def base_model(shape):
	input_layer = tf.keras.layers.Input(shape = shape, name = "input_layer")
	
	mobile_model = tf.keras.applications.MobileNetV3Small(include_top = False,
														  input_shape = shape,
														  pooling = "max",
														  weights = "imagenet",
														  dropout_rate = 0.1,
														  )
	y = mobile_model(input_layer)

	model = tf.keras.models.Model(
		inputs = input_layer,
		outputs = y,
		name = "mobile_max")  
	return model

def my_model(n_class):
	model = tf.keras.models.Sequential([
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(units = 1024, activation = "relu", kernel_initializer = 'he_uniform'),
		tf.keras.layers.Dropout(0.3),
		tf.keras.layers.BatchNormalization(),
		tf.keras.layers.Dense(units = 512, activation = "relu", kernel_initializer = 'he_uniform'),
		tf.keras.layers.Dropout(0.2),
		tf.keras.layers.BatchNormalization(),
		tf.keras.layers.Dense(units = 256, activation = "relu", kernel_initializer = 'he_uniform'),
		tf.keras.layers.Dropout(0.1),
		tf.keras.layers.BatchNormalization(),
		tf.keras.layers.Dense(units = n_class, activation = "softmax")
	], name = "my_dense")
	return model

def loli_model():
	print("load_model start")
	conv_model = base_model(shape = (height,width,3))
	fc_model = my_model(n_class = 149)

	poke_model = tf.keras.models.Sequential([conv_model, fc_model], name = "my_model")
	poke_model.load_weights("../weights/loli_model2.hdf5")
	print("load_model done")
	return poke_model

