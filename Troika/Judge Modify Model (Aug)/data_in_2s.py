import numpy as np
import scipy.io as sio

def data_in(testnum, rd_seed=1):
	'''
	This function aim to import data
	'''
	np.random.seed(rd_seed)

	list_num = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
	if testnum in list_num:
		list_num = np.delete(list_num, testnum-1)

	f = 1
	for i in list_num:

		name_x = 'Training_data/DATA_%.2d_TYPE02.mat'%i
		in_tem_x = sio.loadmat(name_x)['sig']
		in_tem_x = in_tem_x.T

		name_y = 'Training_data/DATA_%.2d_TYPE02_BPMtrace.mat'%i
		in_tem_y = sio.loadmat(name_y)['BPM0']

		X_train_tem = np.zeros((in_tem_y.shape[0]*2, 125*8, 5))

		k = 0

		for j in range(in_tem_y.shape[0]*2-1):
			X_train_tem[j, :, :] =  in_tem_x[k:(k+125*8), 1:]
			k += 125

		tem_y = np.zeros((in_tem_y.shape[0]*2, 1))
		index = [a*2 for a in range(in_tem_y.shape[0])]
		tem_y[index] = in_tem_y

		for b in range(tem_y.shape[0]-1):
			if tem_y[b] == 0:
				tem_y[b] = (tem_y[b-1] + tem_y[b+1])/2.

		tem_y = tem_y[:-1, :]
		X_train_tem = X_train_tem[:-1, :, :]

		if f == 1:
			X_train = X_train_tem
			y_train = tem_y
		else:
			X_train = np.concatenate((X_train, X_train_tem), axis=0)
			y_train = np.concatenate((y_train, tem_y), axis=0)
		f += 1

	list_num_2 = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
	if testnum in list_num_2:
		list_num_2 = np.delete(list_num_2, testnum-13)
	list_num_2[:] = [x - 12 for x in list_num_2]

	for i in list_num_2:
		name_x = 'TestData_change_name/TEST_%.2d.mat'%i
		in_tem_x = sio.loadmat(name_x)['sig']
		in_tem_x = in_tem_x.T

		name_y = 'TrueBPM_change_name/TEST_y_%.2d.mat'%i
		in_tem_y = sio.loadmat(name_y)['BPM0']

		X_train_tem = np.zeros((in_tem_y.shape[0]*2, 125*8, 5))

		k = 0

		for j in range(in_tem_y.shape[0]*2-1):
			X_train_tem[j, :, :] =  in_tem_x[k:(k+125*8), :]
			k += 125

		tem_y = np.zeros((in_tem_y.shape[0]*2, 1))
		index = [a*2 for a in range(in_tem_y.shape[0])]
		tem_y[index] = in_tem_y

		for b in range(tem_y.shape[0]-1):
			if tem_y[b] == 0:
				tem_y[b] = (tem_y[b-1] + tem_y[b+1])/2.

		X_train = np.concatenate((X_train, X_train_tem), axis=0)
		y_train = np.concatenate((y_train, tem_y), axis=0)

	in_tem_x = sio.loadmat('Extra_TrainingData/DATA_S04_T01.mat')['sig'][1:].T
	in_tem_y = sio.loadmat('Extra_TrainingData/BPM_S04_T01.mat')['BPM0']

	X_train_tem = np.zeros((in_tem_y.shape[0]*2, 125*8, 5))

	k = 0

	for i in range(in_tem_y.shape[0]*2-1):
		X_train_tem[i, :, :] = in_tem_x[k:(k+125*8), :]
		k += 125

		tem_y = np.zeros((in_tem_y.shape[0]*2, 1))
		index = [a*2 for a in range(in_tem_y.shape[0])]
		tem_y[index] = in_tem_y

		for b in range(tem_y.shape[0]-1):
			if tem_y[b] == 0:
				tem_y[b] = (tem_y[b-1] + tem_y[b+1])/2.

	X_train = np.concatenate((X_train, X_train_tem), axis=0) 
	y_train = np.concatenate((y_train, tem_y), axis=0)

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
