import numpy as np
import scipy.io as sio

def data_in(rd_seed=1):
	'''
	This functio aim to import data
	'''
	np.random.seed(rd_seed)

	for i in range(1, 13):
		name_x = 'Training_data/DATA_%.2d_TYPE02.mat'%i
		in_tem_x = sio.loadmat(name_x)['sig']
		in_tem_x = in_tem_x.T

		name_y = 'Training_data/DATA_%.2d_TYPE02_BPMtrace.mat'%i
		in_tem_y = sio.loadmat(name_y)['BPM0']

		X_train_tem = np.zeros((in_tem_y.shape[0], 125*8, 5))

		k = 0

		for j in range(in_tem_y.shape[0]):
			X_train_tem[j, :, :] =  in_tem_x[k:(k+125*8), 1:]
			k += 2*125

		if i == 1:
			X_train = X_train_tem
			y_train = in_tem_y
		else:
			X_train = np.concatenate((X_train, X_train_tem), axis=0)
			y_train = np.concatenate((y_train, in_tem_y), axis=0)

	for i in range(1, 11):
		name_x = 'TestData_change_name/TEST_%.2d.mat'%i
		in_tem_x = sio.loadmat(name_x)['sig']
		in_tem_x = in_tem_x.T

		name_y = 'TrueBPM_change_name/TEST_y_%.2d.mat'%i
		in_tem_y = sio.loadmat(name_y)['BPM0']

		X_train_tem = np.zeros((in_tem_y.shape[0], 125*8, 5))

		k = 0

		for j in range(in_tem_y.shape[0]):
			X_train_tem[j, :, :] =  in_tem_x[k:(k+125*8), :]
			k += 2*125

		X_train = np.concatenate((X_train, X_train_tem), axis=0)
		y_train = np.concatenate((y_train, in_tem_y), axis=0)

	in_tem_x = sio.loadmat('Extra_TrainingData/DATA_S04_T01.mat')['sig'][1:].T
	in_tem_y = sio.loadmat('Extra_TrainingData/BPM_S04_T01.mat')['BPM0']

	X_train_tem = np.zeros((in_tem_y.shape[0], 125*8, 5))

	k = 0

	for i in range(in_tem_y.shape[0]):
		X_train_tem[i, :, :] = in_tem_x[k:(k+125*8), :]
		k += 2*125

	X_train = np.concatenate((X_train, X_train_tem), axis=0) 
	y_train = np.concatenate((y_train, in_tem_y), axis=0)

	return X_train, y_train


def normali(input):
	'''
	This function aim to normalize X_train
	'''
	for i in range(input.shape[2]):
		tem = input[:, :, i]
		tem = tem/(np.max(tem) - np.min(tem))
		input[:, :, i] = tem
	return input
