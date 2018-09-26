from keras.layers import Dense, Flatten, Activation, TimeDistributed, Reshape
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D, UpSampling1D
from keras.layers import AveragePooling1D
from keras.layers import GlobalMaxPooling1D
from keras.layers import GlobalAveragePooling1D
from keras.layers import Input
from keras.models import Model
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
import keras.backend as K

def autoencoder(input_shape, weight_decay=1e-4):
	'''
	This function aim to ....
	Arguments:
	input_shape -- 

	Returns:

	'''
	inputs = Input(shape=input_shape)

	#Encoder
	encoder = Activation('relu')(inputs)
	encoder = Conv1D(16, 4, padding='same',
						kernel_initializer='glorot_uniform',
							kernel_regularizer=l2(weight_decay))(encoder)
	encoder = MaxPooling1D(pool_size=2)(encoder)
	encoder = Activation('relu')(encoder)
	encoder = Conv1D(32, 4, padding='same',
						kernel_initializer='glorot_uniform',
							kernel_regularizer=l2(weight_decay))(encoder)
	encoder = MaxPooling1D(pool_size=2)(encoder)
	encoder = Activation('relu')(encoder)
	encoder = Conv1D(64, 4, padding='same',
						kernel_initializer='glorot_uniform',
							kernel_regularizer=l2(weight_decay))(encoder)
	encoder = MaxPooling1D(pool_size=2)(encoder)

	#flat = TimeDistributed(Flatten())(encoder)
	#encoded = Dense(64, activation='relu')(flat)

	#print('shape of encoded {}'.format(K.int_shape(encoded)))

	#Decoder
	decoder = Activation('relu')(encoder)
	decoder = Conv1D(64, 4, padding='same',
						kernel_initializer='glorot_uniform',
							kernel_regularizer=l2(weight_decay))(decoder)
	decoder = UpSampling1D(2)(decoder)
	decoder = Activation('relu')(decoder)
	decoder = Conv1D(32, 4, padding='same',
						kernel_initializer='glorot_uniform',
							kernel_regularizer=l2(weight_decay))(decoder)
	decoder = UpSampling1D(2)(decoder)
	decoder = Activation('relu')(decoder)
	decoder = Conv1D(16, 4, padding='same',
						kernel_initializer='glorot_uniform',
							kernel_regularizer=l2(weight_decay))(decoder)
	decoder = UpSampling1D(2)(decoder)
	#decoder = Activation('relu')(decoder)
	decoded = Conv1D(2, 4, padding='same',
						kernel_initializer='glorot_uniform',
							kernel_regularizer=l2(weight_decay))(decoder)

	#flat = Flatten()(decoder)
	#decoded = Dense(2400, activation='relu')(flat)
	#decoded = Reshape((1200,2))(decoded)

	#print('shape of decoded {}'.format(K.int_shape(decoded)))

	model = Model(inputs, decoded)

	return model

