from keras.layers import Dense, Flatten, Activation, TimeDistributed, Reshape
from keras.layers.core import Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D, UpSampling1D
from keras.layers import AveragePooling1D
from keras.layers import GlobalMaxPooling1D
from keras.layers import GlobalAveragePooling1D
from keras.layers import Input, Concatenate, concatenate
from keras.models import Model
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
import keras.backend as K

def autoencoder(input_shape, output_shape, kernel_size, padding_name='valid', weight_decay=1e-4, dropout_rate=0.2):
	'''
	This function aim to ....
	Arguments:
	input_shape -- 

	Returns:

	'''

	if K.image_dim_ordering() == 'channels_first':
		add_axis = 1
	else:
		add_axis = -1

	inputs = Input(shape=input_shape)

	#Encoder
	'''
	encoder = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(inputs)
	'''
	encoder = Activation('relu')(inputs)
	encoder = Conv1D(16, kernel_size, padding=padding_name,
						kernel_initializer='glorot_uniform',
							kernel_regularizer=l2(weight_decay))(encoder)
	encoder = Dropout(dropout_rate)(encoder)
	encoder = MaxPooling1D(pool_size=2)(encoder)

	encoder = Activation('relu')(encoder)
	encoder = Conv1D(64, kernel_size, padding=padding_name,
						kernel_initializer='glorot_uniform',
							kernel_regularizer=l2(weight_decay))(encoder)
	encoder = Dropout(dropout_rate)(encoder)
	encoder = MaxPooling1D(pool_size=2)(encoder)

	encoder = Activation('relu')(encoder)
	encoder = Conv1D(128, kernel_size, padding=padding_name,
						kernel_initializer='glorot_uniform',
							kernel_regularizer=l2(weight_decay))(encoder)
	encoder = Dropout(dropout_rate)(encoder)
	encoder = MaxPooling1D(pool_size=2)(encoder)
	'''
	encoder = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(encoder)
	encoder = Activation('relu')(encoder)
	encoder = Conv1D(128, 4, padding=padding_name,
						kernel_initializer='glorot_uniform',
							kernel_regularizer=l2(weight_decay))(encoder)
	encoder = Dropout(dropout_rate)(encoder)
	encoder = MaxPooling1D(pool_size=2.5)(encoder)
	'''

	#flat = TimeDistributed(Flatten())(encoder)
	#encoded = Dense(64, activation='relu')(flat)

	#print('shape of encoded {}'.format(K.int_shape(encoded)))

	padding_name = 'same'
	#Decoder

	decoder = Activation('relu')(encoder)
	decoder = Conv1D(128, kernel_size, padding=padding_name,
						kernel_initializer='glorot_uniform',
							kernel_regularizer=l2(weight_decay))(decoder)
	decoder = Dropout(dropout_rate)(decoder)
	decoder = UpSampling1D(2)(decoder)

	decoder = Activation('relu')(decoder)
	decoder = Conv1D(64, kernel_size, padding=padding_name,
						kernel_initializer='glorot_uniform',
							kernel_regularizer=l2(weight_decay))(decoder)
	decoder = Dropout(dropout_rate)(decoder)
	decoder = UpSampling1D(2)(decoder)

	decoder = Activation('relu')(decoder)
	decoder = Conv1D(16, kernel_size, padding=padding_name,
						kernel_initializer='glorot_uniform',
							kernel_regularizer=l2(weight_decay))(decoder)
	decoder = Dropout(dropout_rate)(decoder)
	decoder = UpSampling1D(2)(decoder)
	'''
	decoder = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(decoder) 
	decoder = Activation('relu')(decoder)
	decoder = Conv1D(16, 4, padding='same',
						kernel_initializer='glorot_uniform',
							kernel_regularizer=l2(weight_decay))(decoder)
	decoder = Dropout(dropout_rate)(decoder)
	decoder = UpSampling1D(2)(decoder)
	'''
	#decoder = Activation('sigmoid')(decoder)
	decoded = Conv1D(output_shape, kernel_size, padding=padding_name,
						kernel_initializer='glorot_uniform',
							kernel_regularizer=l2(weight_decay))(decoder)

	#flat = Flatten()(decoder)
	#decoded = Dense(2400, activation='relu')(flat)
	#decoded = Reshape((1200,2))(decoded)

	#print('shape of decoded {}'.format(K.int_shape(decoded)))

	model = Model(inputs, decoded)

	return model

def aemodel(input_shape, input_model, input_layers, weight_decay=1e-4, dropout_rate=0.2):
	'''
	This functio aim to pick up encoder and add more layers

	Arguments:
	input_shape --
	input_model --
	input_layers -- 
	dropout_rate --

	Returns:
	model --
	'''
	inputs = Input(shape=input_shape)
	layer_out = input_model.layers[input_layers].output
	newlayer = UpSampling1D(4)(layer_out)
	newlayer = Conv1D(64, 4, padding='same',
                    kernel_initializer='glorot_uniform',
                    kernel_regularizer=l2(weight_decay))(newlayer)
	newlayer = Dropout(dropout_rate)(newlayer)
	newlayer = MaxPooling1D(pool_size=2)(newlayer)

	newlayer = Conv1D(256, 4, padding='same',
                    kernel_initializer='glorot_uniform',
                    kernel_regularizer=l2(weight_decay))(newlayer)
	newlayer = Dropout(dropout_rate)(newlayer)
	newlayer = MaxPooling1D(pool_size=2)(newlayer)

	newlayer = Activation('relu')(newlayer)
	newlayer = Conv1D(128, 4, padding='same',
	                    kernel_initializer='glorot_uniform',
	                    kernel_regularizer=l2(weight_decay))(newlayer)
	newlayer = Dropout(dropout_rate)(newlayer)
	newlayer = MaxPooling1D(pool_size=2)(newlayer)

	newlayer = Activation('relu')(newlayer)
	newlayer = Conv1D(64, 4, padding='same',
	                    kernel_initializer='glorot_uniform',
	                    kernel_regularizer=l2(weight_decay))(newlayer)
	newlayer = Dropout(dropout_rate)(newlayer)
	newlayer = MaxPooling1D(pool_size=2)(newlayer)

	newlayer = Activation('relu')(newlayer)
	newlayer = Conv1D(16, 4, padding='same',
	                    kernel_initializer='glorot_uniform',
	                    kernel_regularizer=l2(weight_decay))(newlayer)
	newlayer = Dropout(dropout_rate)(newlayer)
	#newlayer = GlobalAveragePooling1D()(newlayer)

	newlayer = Flatten()(newlayer)
	newlayer = Dense(64, activation='relu', 
						kernel_initializer='glorot_uniform', 
						kernel_regularizer=l2(weight_decay))(newlayer)
	newlayer = Dropout(dropout_rate)(newlayer)
	newlayer = Dense(32, activation='relu',
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(newlayer)
	newlayer = Dropout(dropout_rate)(newlayer)
	newlayer = Dense(16, activation='relu',
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(newlayer)
	newlayer = Dropout(dropout_rate)(newlayer)

	out = Dense(1, activation='sigmoid', 
	               kernel_initializer='glorot_uniform',
	               kernel_regularizer=l2(weight_decay))(newlayer)

	finalmodel = Model(input=input_model.input, output=out)

	# Freeze the layers
	for layer in finalmodel.layers[:input_layers]:
	    layer.trainable = False

	return finalmodel



def concatmodel(input_shape_1, input_shape_2, input_model, input_layers, dropout_rate=0.2, weight_decay=1e-4):
	'''
	This function aim to concatenate two part of NN

	Arguments:
	input_shape --
	another_model --
	dropout_rate --

	Returns:
	model --
	'''

	#Accelarte channel
	inputs_1 = Input(shape=input_shape_1)

	ori_layer = Activation('relu')(inputs_1)
	ori_layer = Conv1D(16, 4, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(ori_layer)
	ori_layer = Dropout(dropout_rate)(ori_layer)
	ori_layer = MaxPooling1D(pool_size=2)(ori_layer)

	ori_layer = Activation('relu')(ori_layer)
	ori_layer = Conv1D(32, 4, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(ori_layer)
	ori_layer = Dropout(dropout_rate)(ori_layer)
	ori_layer = MaxPooling1D(pool_size=2)(ori_layer)

	ori_layer = Activation('relu')(ori_layer)
	ori_layer = Conv1D(64, 4, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(ori_layer)
	ori_layer = Dropout(dropout_rate)(ori_layer)
	ori_layer = MaxPooling1D(pool_size=2)(ori_layer)

	ori_layer = Activation('relu')(ori_layer)
	ori_layer = Conv1D(128, 4, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(ori_layer)
	ori_layer = Dropout(dropout_rate)(ori_layer)
	ori_layer = MaxPooling1D(pool_size=2)(ori_layer)

	ori_layer = Activation('relu')(ori_layer)
	ori_layer = Conv1D(256, 4, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(ori_layer)
	ori_layer = Dropout(dropout_rate)(ori_layer)
	ori_layer = MaxPooling1D(pool_size=2)(ori_layer)

	ori_layer = Activation('relu')(ori_layer)
	ori_layer = Conv1D(256, 4, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(ori_layer)
	ori_layer = Dropout(dropout_rate)(ori_layer)
	ori_layer = MaxPooling1D(pool_size=2)(ori_layer)

	ori_layer = Flatten()(ori_layer)
	ori_layer = Dense(256, activation='relu', 
						kernel_initializer='glorot_uniform', 
						kernel_regularizer=l2(weight_decay))(ori_layer)
	ori_layer = Dropout(dropout_rate)(ori_layer)

	ori_layer = Dense(128, activation='relu', 
						kernel_initializer='glorot_uniform', 
						kernel_regularizer=l2(weight_decay))(ori_layer)
	ori_layer = Dropout(dropout_rate)(ori_layer)

	ori_layer = Dense(64, activation='relu', 
						kernel_initializer='glorot_uniform', 
						kernel_regularizer=l2(weight_decay))(ori_layer)
	ori_layer = Dropout(dropout_rate)(ori_layer)

	ori_layer = Dense(32, activation='relu', 
						kernel_initializer='glorot_uniform', 
						kernel_regularizer=l2(weight_decay))(ori_layer)
	ori_layer = Dropout(dropout_rate)(ori_layer)

	model1 = Model(input=inputs_1, output=ori_layer)

	#=====================================================================
	#=====================================================================
	'''
	#PPG channel
	inputs_2 = Input(shape=input_shape_2)
	ano_layer = Activation('relu')(inputs_2)
	ano_layer = Conv1D(16, 4, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(ano_layer)
	ano_layer = Dropout(dropout_rate)(ano_layer)
	ano_layer = MaxPooling1D(pool_size=2)(ano_layer)

	ano_layer = Activation('relu')(ano_layer)
	ano_layer = Conv1D(32, 4, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(ano_layer)
	ano_layer = Dropout(dropout_rate)(ano_layer)
	ano_layer = MaxPooling1D(pool_size=2)(ano_layer)

	ano_layer = Activation('relu')(ano_layer)
	ano_layer = Conv1D(64, 4, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(ano_layer)
	ano_layer = Dropout(dropout_rate)(ano_layer)
	ano_layer = MaxPooling1D(pool_size=2)(ano_layer)

	ano_layer = Activation('relu')(ano_layer)
	ano_layer = Conv1D(128, 4, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(ano_layer)
	ano_layer = Dropout(dropout_rate)(ano_layer)
	ano_layer = MaxPooling1D(pool_size=2)(ano_layer)

	ano_layer = Activation('relu')(ano_layer)
	ano_layer = Conv1D(128, 4, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(ano_layer)
	ano_layer = Dropout(dropout_rate)(ano_layer)
	ano_layer = MaxPooling1D(pool_size=2)(ano_layer)

	ano_layer = Flatten()(ano_layer)
	ano_layer = Dense(256, activation='relu', 
						kernel_initializer='glorot_uniform', 
						kernel_regularizer=l2(weight_decay))(ano_layer)
	ano_layer = Dropout(dropout_rate)(ano_layer)

	ano_layer = Dense(128, activation='relu', 
						kernel_initializer='glorot_uniform', 
						kernel_regularizer=l2(weight_decay))(ano_layer)
	ano_layer = Dropout(dropout_rate)(ano_layer)

	ano_layer = Dense(64, activation='relu', 
						kernel_initializer='glorot_uniform', 
						kernel_regularizer=l2(weight_decay))(ano_layer)
	ano_layer = Dropout(dropout_rate)(ano_layer)

	ano_layer = Dense(32, activation='relu', 
						kernel_initializer='glorot_uniform', 
						kernel_regularizer=l2(weight_decay))(ano_layer)
	ano_layer = Dropout(dropout_rate)(ano_layer)
	'''

	inputs_2 = Input(shape=input_shape_2)
	layer_out = input_model.layers[input_layers].output
	newlayer = UpSampling1D(4)(layer_out)
	newlayer = Conv1D(64, 4, padding='same',
                    kernel_initializer='glorot_uniform',
                    kernel_regularizer=l2(weight_decay))(newlayer)
	newlayer = Dropout(dropout_rate)(newlayer)
	newlayer = MaxPooling1D(pool_size=2)(newlayer)

	newlayer = Conv1D(256, 4, padding='same',
                    kernel_initializer='glorot_uniform',
                    kernel_regularizer=l2(weight_decay))(newlayer)
	newlayer = Dropout(dropout_rate)(newlayer)
	newlayer = MaxPooling1D(pool_size=2)(newlayer)

	newlayer = Activation('relu')(newlayer)
	newlayer = Conv1D(128, 4, padding='same',
	                    kernel_initializer='glorot_uniform',
	                    kernel_regularizer=l2(weight_decay))(newlayer)
	newlayer = Dropout(dropout_rate)(newlayer)
	newlayer = MaxPooling1D(pool_size=2)(newlayer)

	newlayer = Activation('relu')(newlayer)
	newlayer = Conv1D(64, 4, padding='same',
	                    kernel_initializer='glorot_uniform',
	                    kernel_regularizer=l2(weight_decay))(newlayer)
	newlayer = Dropout(dropout_rate)(newlayer)
	newlayer = MaxPooling1D(pool_size=2)(newlayer)

	newlayer = Activation('relu')(newlayer)
	newlayer = Conv1D(16, 4, padding='same',
	                    kernel_initializer='glorot_uniform',
	                    kernel_regularizer=l2(weight_decay))(newlayer)
	newlayer = Dropout(dropout_rate)(newlayer)
	#newlayer = GlobalAveragePooling1D()(newlayer)

	newlayer = Flatten()(newlayer)
	newlayer = Dense(64, activation='relu', 
						kernel_initializer='glorot_uniform', 
						kernel_regularizer=l2(weight_decay))(newlayer)
	newlayer = Dropout(dropout_rate)(newlayer)
	newlayer = Dense(32, activation='relu',
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(newlayer)
	newlayer = Dropout(dropout_rate)(newlayer)
	newlayer = Dense(16, activation='relu',
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(newlayer)
	newlayer = Dropout(dropout_rate)(newlayer)

	out = Dense(1, activation='sigmoid', 
	               kernel_initializer='glorot_uniform',
	               kernel_regularizer=l2(weight_decay))(newlayer)

	finalmodel = Model(input=input_model.input, output=out)

	# Freeze the layers
	for layer in finalmodel.layers[:input_layers]:
	    layer.trainable = False

	return finalmodel

	model2 = Model(input=inputs_2, output=ano_layer)

	#===============================================================
	#===============================================================

	if K.image_dim_ordering() == 'channels_first':
		add_axis = 1
	else:
		add_axis = -1

	#concat_layer = aemodel((1000, 2), input_model, input_layer)

	#connect_list = [ori_layer]
	#connect_list.append(ano_layer)

	#final_layer = concatenate(connect_list[:], axis=add_axis)

	concatenated = concatenate([ori_layer, ano_layer])

	#final_layer = Concatenate(axis=add_axis)(connect_list)
	
	#final_layer = Dropout(dropout_rate)(final_layer)

	final_layer = Dense(32, activation='relu', 
						kernel_initializer='glorot_uniform', 
						kernel_regularizer=l2(weight_decay))(concatenated)
	final_layer = Dropout(dropout_rate)(final_layer)

	final_layer = Dense(16, activation='relu', 
						kernel_initializer='glorot_uniform', 
						kernel_regularizer=l2(weight_decay))(final_layer)
	final_layer = Dropout(dropout_rate)(final_layer)

	final_layer = Dense(1, activation='relu', 
						kernel_initializer='glorot_uniform', 
						kernel_regularizer=l2(weight_decay))(final_layer)

	finalmodel = Model(input=[inputs_1, inputs_2], output=final_layer)

	return finalmodel







