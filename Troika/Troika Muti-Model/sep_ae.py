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

''''''

def concatmodel(input_shape_1, input_shape_2, kernel_size, input_model, input_layers, dropout_rate=0.2, weight_decay=1e-4):
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

	if K.image_dim_ordering() == 'channels_first':
		add_axis = 1
	else:
		add_axis = -1

	inputs_1 = Input(shape=input_shape_1)

	ori_layer = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(inputs_1)
	ori_layer = Activation('relu')(inputs_1)
	ori_layer = Conv1D(16, kernel_size, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(ori_layer)
	ori_layer = Dropout(dropout_rate)(ori_layer)
	ori_layer = MaxPooling1D(pool_size=2)(ori_layer)

	ori_layer = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(ori_layer)
	ori_layer = Activation('relu')(ori_layer)
	ori_layer = Conv1D(32, kernel_size, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(ori_layer)
	ori_layer = Dropout(dropout_rate)(ori_layer)
	ori_layer = MaxPooling1D(pool_size=2)(ori_layer)

	ori_layer = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(ori_layer)
	ori_layer = Activation('relu')(ori_layer)
	ori_layer = Conv1D(64, kernel_size, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(ori_layer)
	ori_layer = Dropout(dropout_rate)(ori_layer)
	ori_layer = MaxPooling1D(pool_size=2)(ori_layer)

	ori_layer = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(ori_layer)
	ori_layer = Activation('relu')(ori_layer)
	ori_layer = Conv1D(128, kernel_size, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(ori_layer)
	ori_layer = Dropout(dropout_rate)(ori_layer)
	ori_layer = MaxPooling1D(pool_size=2)(ori_layer)

	ori_layer = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(ori_layer)
	ori_layer = Activation('relu')(ori_layer)
	ori_layer = Conv1D(256, kernel_size, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(ori_layer)
	ori_layer = Dropout(dropout_rate)(ori_layer)
	ori_layer = MaxPooling1D(pool_size=2)(ori_layer)

	ori_layer = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(ori_layer)
	ori_layer = Activation('relu')(ori_layer)
	ori_layer = Conv1D(512, kernel_size, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(ori_layer)
	ori_layer = Dropout(dropout_rate)(ori_layer)
	ori_layer = MaxPooling1D(pool_size=2)(ori_layer)

	ori_layer = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(ori_layer)
	ori_layer = Activation('relu')(ori_layer)
	ori_layer = Conv1D(1024, kernel_size, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(ori_layer)
	ori_layer = Dropout(dropout_rate)(ori_layer)
	ori_layer = GlobalAveragePooling1D()(ori_layer)

	ori_layer = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(ori_layer)
	#ori_layer = Flatten()(ori_layer)
	ori_layer = Dense(1024, activation='relu', 
						kernel_initializer='glorot_uniform', 
						kernel_regularizer=l2(weight_decay))(ori_layer)
	ori_layer = Dropout(dropout_rate)(ori_layer)

	ori_layer = Dense(512, activation='relu', 
						kernel_initializer='glorot_uniform', 
						kernel_regularizer=l2(weight_decay))(ori_layer)
	ori_layer = Dropout(dropout_rate)(ori_layer)

	ori_layer = Dense(258, activation='relu', 
						kernel_initializer='glorot_uniform', 
						kernel_regularizer=l2(weight_decay))(ori_layer)
	ori_layer = Dropout(dropout_rate)(ori_layer)

	ori_layer = Dense(64, activation='relu', 
						kernel_initializer='glorot_uniform', 
						kernel_regularizer=l2(weight_decay))(ori_layer)
	ori_layer = Dropout(dropout_rate)(ori_layer)

	model1 = Model(input=inputs_1, output=ori_layer)

	#=====================================================================
	#=====================================================================
	'''
	#PPG channel
	inputs_2 = Input(shape=input_shape_2)

	ano_layer = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(inputs_2)
	ano_layer = Activation('relu')(inputs_2)
	ano_layer = Conv1D(16, 4, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(ano_layer)
	ano_layer = Dropout(dropout_rate)(ano_layer)
	ano_layer = MaxPooling1D(pool_size=2)(ano_layer)

	ano_layer = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(ano_layer)
	ano_layer = Activation('relu')(ano_layer)
	ano_layer = Conv1D(32, 4, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(ano_layer)
	ano_layer = Dropout(dropout_rate)(ano_layer)
	ano_layer = MaxPooling1D(pool_size=2)(ano_layer)

	ano_layer = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(ano_layer)
	ano_layer = Activation('relu')(ano_layer)
	ano_layer = Conv1D(64, 4, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(ano_layer)
	ano_layer = Dropout(dropout_rate)(ano_layer)
	ano_layer = MaxPooling1D(pool_size=2)(ano_layer)

	ano_layer = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(ano_layer)
	ano_layer = Activation('relu')(ano_layer)
	ano_layer = Conv1D(128, 4, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(ano_layer)
	ano_layer = Dropout(dropout_rate)(ano_layer)
	ano_layer = MaxPooling1D(pool_size=2)(ano_layer)

	ano_layer = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(ano_layer)
	ano_layer = Activation('relu')(ano_layer)
	ano_layer = Conv1D(256, 4, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(ano_layer)
	ano_layer = Dropout(dropout_rate)(ano_layer)
	ano_layer = MaxPooling1D(pool_size=2)(ano_layer)

	ano_layer = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(ano_layer)
	ano_layer = Activation('relu')(ano_layer)
	ano_layer = Conv1D(512, 4, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(ano_layer)
	ano_layer = Dropout(dropout_rate)(ano_layer)
	ano_layer = MaxPooling1D(pool_size=2)(ano_layer)

	ano_layer = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(ano_layer)
	ano_layer = Activation('relu')(ano_layer)
	ano_layer = Conv1D(1024, 4, padding='same', 
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(ano_layer)
	ano_layer = Dropout(dropout_rate)(ano_layer)
	ano_layer = GlobalMaxPooling1D()(ano_layer)

	ano_layer = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(ano_layer)
	#ano_layer = Flatten()(ano_layer)
	ano_layer = Dense(1024, activation='relu', 
						kernel_initializer='glorot_uniform', 
						kernel_regularizer=l2(weight_decay))(ano_layer)
	ano_layer = Dropout(dropout_rate)(ano_layer)

	ano_layer = Dense(512, activation='relu', 
						kernel_initializer='glorot_uniform', 
						kernel_regularizer=l2(weight_decay))(ano_layer)
	ano_layer = Dropout(dropout_rate)(ano_layer)

	ano_layer = Dense(256, activation='relu', 
						kernel_initializer='glorot_uniform', 
						kernel_regularizer=l2(weight_decay))(ano_layer)
	ano_layer = Dropout(dropout_rate)(ano_layer)

	ano_layer = Dense(64, activation='relu', 
						kernel_initializer='glorot_uniform', 
						kernel_regularizer=l2(weight_decay))(ano_layer)
	ano_layer = Dropout(dropout_rate)(ano_layer)
	'''

	#inputs_2 = Input(shape=input_shape_2)
	inputs_2 = input_model.input
	layer_out = input_model.layers[input_layers].output

	newlayer = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(layer_out)
	newlayer = Activation('relu')(newlayer)
	newlayer = UpSampling1D(4)(layer_out)
	newlayer = Conv1D(128, kernel_size, padding='same',
                    kernel_initializer='glorot_uniform',
                    kernel_regularizer=l2(weight_decay))(newlayer)
	newlayer = Dropout(dropout_rate)(newlayer)
	newlayer = MaxPooling1D(pool_size=2)(newlayer)

	newlayer = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(newlayer)
	newlayer = Activation('relu')(newlayer)
	newlayer = Conv1D(128, kernel_size, padding='same',
                    kernel_initializer='glorot_uniform',
                    kernel_regularizer=l2(weight_decay))(newlayer)
	newlayer = Dropout(dropout_rate)(newlayer)
	newlayer = MaxPooling1D(pool_size=2)(newlayer)

	newlayer = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(newlayer)
	newlayer = Activation('relu')(newlayer)
	newlayer = Conv1D(256, kernel_size, padding='same',
	                    kernel_initializer='glorot_uniform',
	                    kernel_regularizer=l2(weight_decay))(newlayer)
	newlayer = Dropout(dropout_rate)(newlayer)
	newlayer = MaxPooling1D(pool_size=2)(newlayer)

	newlayer = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(newlayer)
	newlayer = Activation('relu')(newlayer)
	newlayer = Conv1D(512, kernel_size, padding='same',
	                    kernel_initializer='glorot_uniform',
	                    kernel_regularizer=l2(weight_decay))(newlayer)
	newlayer = Dropout(dropout_rate)(newlayer)
	newlayer = MaxPooling1D(pool_size=2)(newlayer)

	newlayer = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(newlayer)
	newlayer = Activation('relu')(newlayer)
	newlayer = Conv1D(1024, kernel_size, padding='same',
	                    kernel_initializer='glorot_uniform',
	                    kernel_regularizer=l2(weight_decay))(newlayer)
	newlayer = Dropout(dropout_rate)(newlayer)
	newlayer = GlobalAveragePooling1D()(newlayer)

	#newlayer = Flatten()(newlayer)
	newlayer = BatchNormalization(axis=add_axis, 
									gamma_regularizer=l2(weight_decay), 
									beta_regularizer=l2(weight_decay))(newlayer)
	newlayer = Dense(512, activation='relu', 
						kernel_initializer='glorot_uniform', 
						kernel_regularizer=l2(weight_decay))(newlayer)
	newlayer = Dropout(dropout_rate)(newlayer)
	newlayer = Dense(256, activation='relu',
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(newlayer)
	newlayer = Dropout(dropout_rate)(newlayer)
	newlayer = Dense(128, activation='relu',
						kernel_initializer='glorot_uniform',
						kernel_regularizer=l2(weight_decay))(newlayer)
	newlayer = Dropout(dropout_rate)(newlayer)

	newlayer = Dense(64, activation='sigmoid', 
	               kernel_initializer='glorot_uniform',
	               kernel_regularizer=l2(weight_decay))(newlayer)
	newlayer = Dropout(dropout_rate)(newlayer)            

	#finalmodel = Model(input=input_model.input, output=out)

	model2 = Model(input=inputs_2, output=newlayer)

	# Freeze the layers
	for layer in model2.layers[:input_layers]:
	    layer.trainable = False
  
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

	concatenated = concatenate([ori_layer, newlayer])

	#final_layer = Concatenate(axis=add_axis)(connect_list)
	
	#final_layer = Dropout(dropout_rate)(final_layer)

	final_layer = Dense(64, activation='relu', 
						kernel_initializer='glorot_uniform', 
						kernel_regularizer=l2(weight_decay))(concatenated)
	final_layer = Dropout(dropout_rate)(final_layer)

	final_layer = Dense(32, activation='relu', 
						kernel_initializer='glorot_uniform', 
						kernel_regularizer=l2(weight_decay))(final_layer)
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