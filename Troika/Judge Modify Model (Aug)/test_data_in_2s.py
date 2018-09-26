import numpy as np
import scipy.io as sio

def test_data_in(num_test, rd_seed=1):
	'''
	This function aim to import data
	'''
	np.random.seed(rd_seed)

	if num_test <= 12:
		name_x = 'Training_data/DATA_%.2d_TYPE02.mat'%num_test
		in_tem_x = sio.loadmat(name_x)['sig']
		in_tem_x = in_tem_x.T

		name_y = 'Training_data/DATA_%.2d_TYPE02_BPMtrace.mat'%num_test
		in_tem_y = sio.loadmat(name_y)['BPM0']

		X_test_tem = np.zeros((in_tem_y.shape[0]*2, 125*8, 5))

		k = 0

		for j in range(in_tem_y.shape[0]*2-1):
			X_test_tem[j, :, :] =  in_tem_x[k:(k+125*8), 1:]
			k += 125

		tem_y = np.zeros((in_tem_y.shape[0]*2, 1))
		index = [a*2 for a in range(in_tem_y.shape[0])]
		tem_y[index] = in_tem_y

		for b in range(tem_y.shape[0]-1):
			if tem_y[b] == 0:
				tem_y[b] = (tem_y[b-1] + tem_y[b+1])/2.

		y_test = tem_y[:-1, :]
		X_test = X_test_tem[:-1, :, :]

		return X_test, y_test

	elif num_test > 12:
		num_test = num_test - 12
		name_x = 'TestData_change_name/TEST_%.2d.mat'%num_test
		in_tem_x = sio.loadmat(name_x)['sig']
		in_tem_x = in_tem_x.T

		name_y = 'TrueBPM_change_name/TEST_y_%.2d.mat'%num_test
		in_tem_y = sio.loadmat(name_y)['BPM0']

		X_test_tem = np.zeros((in_tem_y.shape[0]*2, 125*8, 5))

		k = 0

		for j in range(in_tem_y.shape[0]*2-1):
			X_test_tem[j, :, :] =  in_tem_x[k:(k+125*8), :]
			k += 125

		tem_y = np.zeros((in_tem_y.shape[0]*2, 1))
		index = [a*2 for a in range(in_tem_y.shape[0])]
		tem_y[index] = in_tem_y

		for b in range(tem_y.shape[0]-1):
			if tem_y[b] == 0:
				tem_y[b] = (tem_y[b-1] + tem_y[b+1])/2.

		y_test = tem_y[:-1, :]
		X_test = X_test_tem[:-1, :, :]

		return X_test, y_test

