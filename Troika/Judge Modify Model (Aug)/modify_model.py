from keras.layers import Dense, Flatten, Activation, TimeDistributed, Reshape
from keras.layers.core import Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D, UpSampling1D
from keras.layers.recurrent import LSTM
from keras.layers import AveragePooling1D
from keras.layers import GlobalMaxPooling1D
from keras.layers import GlobalAveragePooling1D
from keras.layers import Input, Concatenate, concatenate
from keras.models import Model
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
import keras.backend as K

def modify_model(input_shape, kernel_size, pool_size=4, dropout_rate=0.2, weight_decay=1e-4):
	'''
	This function

	Arguments:

	Returns:

	'''

	if K.image_dim_ordering() == 'channels_first':
		add_axis = 1
	else:
		add_axis = -1

	inputs = Input(shape=input_shape)

	output = Conv1D(64, kernel_size, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(inputs)
	output = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(output)
	output = Activation('relu')(output)
	#output = Dropout(dropout_rate)(output)

	output = Conv1D(64, kernel_size, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(output)
	output = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(output)
	output = Activation('relu')(output)
	output = Dropout(dropout_rate)(output)
	output = Conv1D(64, kernel_size, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(output)
	output = MaxPooling1D(pool_size=pool_size)(output)

	output = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(output)
	output = Activation('relu')(output)	
	output = Dropout(dropout_rate)(output)
	output = Conv1D(64, kernel_size, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(output)
	output = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(output)
	output = Activation('relu')(output)
	output = Dropout(dropout_rate)(output)
	output = Conv1D(64, kernel_size, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(output)
	output = MaxPooling1D(pool_size=pool_size)(output)

	output = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(output)
	output = Activation('relu')(output)	
	output = Dropout(dropout_rate)(output)
	output = Conv1D(128, kernel_size, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(output)
	output = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(output)
	output = Activation('relu')(output)
	output = Dropout(dropout_rate)(output)
	output = Conv1D(128, kernel_size, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(output)
	output = MaxPooling1D(pool_size=pool_size)(output)

	output = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(output)
	output = Activation('relu')(output)	
	output = Dropout(dropout_rate)(output)
	output = Conv1D(128, kernel_size, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(output)
	output = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(output)
	output = Activation('relu')(output)
	output = Dropout(dropout_rate)(output)
	output = Conv1D(128, kernel_size, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(output)
	output = MaxPooling1D(pool_size=pool_size)(output)

	output = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(output)
	output = Activation('relu')(output)	
	output = Dropout(dropout_rate)(output)
	output = Conv1D(192, kernel_size, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(output)
	output = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(output)
	output = Activation('relu')(output)
	output = Dropout(dropout_rate)(output)
	output = Conv1D(192, kernel_size, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(output)
	output = MaxPooling1D(pool_size=pool_size)(output)

	output = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(output)
	output = Activation('relu')(output)	
	output = Dropout(dropout_rate)(output)
	output = Conv1D(192, kernel_size, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(output)
	output = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(output)
	output = Activation('relu')(output)
	output = Dropout(dropout_rate)(output)
	output = Conv1D(192, kernel_size, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(output)
	output = MaxPooling1D(pool_size=pool_size)(output)

	output = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(output)
	output = Activation('relu')(output)	
	output = Dropout(dropout_rate)(output)
	output = Conv1D(256, kernel_size, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(output)
	output = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(output)
	output = Activation('relu')(output)
	output = Dropout(dropout_rate)(output)
	output = Conv1D(256, kernel_size, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(output)
	output = MaxPooling1D(pool_size=pool_size)(output)

	output = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(output)
	output = Activation('relu')(output)	
	output = Dropout(dropout_rate)(output)
	output = Conv1D(256, kernel_size, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(output)
	output = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(output)
	output = Activation('relu')(output)
	output = Dropout(dropout_rate)(output)
	output = Conv1D(256, kernel_size, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(output)
	output = MaxPooling1D(pool_size=pool_size)(output)

	#output = BatchNormalization(axis=add_axis, 
	#								gamma_regularizer=l2(weight_decay), 
	#								beta_regularizer=l2(weight_decay))(output)
	output = Activation('relu')(output)
	
	#output = Flatten()(output)
	#output = Dense(100, activation='sigmoid',
	#					kernel_initializer='glorot_uniform',
	#					kernel_regularizer=l2(weight_decay))(output)
	output = LSTM(units=256, return_sequences=True)(output)
	output = Dropout(dropout_rate)(output)

	output = LSTM(units=64, return_sequences=False)(output)
	output = Dropout(dropout_rate)(output)

	output = Dense(1, kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay),
						bias_initializer='zeros')(output)

	model = Model(input=inputs, output=output)

	return model



