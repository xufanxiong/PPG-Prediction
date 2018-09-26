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

def modify_model(input_shape, real_label, prediction, kernel_size, dropout_rate=0.2, weight_decay=1e-4):
	'''
	This function

	Arguments:

	Returns:

	'''

	if K.image_dim_ordering() == 'channels_first':
		add_axis = 1
	else:
		add_axis = -1

	y_label = real_label - prediction

	inputs = Input(shape=input_shape)

	mod_layers = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(inputs)
	mod_layers= Activation('relu')(mod_layers)
	mod_layers = Conv1D(16, kernel_size, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(mod_layers)
	mod_layers = Dropout(dropout_rate)(mod_layers)
	mod_layers = MaxPooling1D(pool_size=2)(mod_layers)

	mod_layers = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(mod_layers)
	mod_layers= Activation('relu')(mod_layers)
	mod_layers = Conv1D(32, kernel_size, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(mod_layers)
	mod_layers = Dropout(dropout_rate)(mod_layers)
	mod_layers = MaxPooling1D(pool_size=2)(mod_layers)

	mod_layers = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(mod_layers)
	mod_layers= Activation('relu')(mod_layers)
	mod_layers = Conv1D(64, kernel_size, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(mod_layers)
	mod_layers = Dropout(dropout_rate)(mod_layers)
	mod_layers = MaxPooling1D(pool_size=2)(mod_layers)

	mod_layers = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(mod_layers)
	mod_layers= Activation('relu')(mod_layers)
	mod_layers = Conv1D(128, kernel_size, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(mod_layers)
	mod_layers = Dropout(dropout_rate)(mod_layers)
	mod_layers = MaxPooling1D(pool_size=2)(mod_layers)

	mod_layers = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(mod_layers)
	mod_layers= Activation('relu')(mod_layers)
	mod_layers = Conv1D(128, kernel_size, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(mod_layers)
	mod_layers = Dropout(dropout_rate)(mod_layers)
	mod_layers = MaxPooling1D()(mod_layers)

	mod_layers = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(mod_layers)
	mod_layers= Activation('relu')(mod_layers)
	mod_layers = Conv1D(256, kernel_size, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(mod_layers)
	mod_layers = Dropout(dropout_rate)(mod_layers)
	mod_layers = GlobalMaxPooling1D()(mod_layers)

	mod_layers = Dense(256, activation='relu',
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(mod_layers)
	mod_layers = Dropout(dropout_rate)(mod_layers)

	mod_layers = Dense(64, activation='relu',
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(mod_layers)
	mod_layers = Dropout(dropout_rate)(mod_layers)

	mod_layers = Dense(1,
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(mod_layers)

	model = Model(input=inputs, output=mod_layers)

	return model, y_label


