import torch 
import torch.nn as nn
import numpy as np

class autoencoder(nn.Module):
	'''
	Pytorch autoencoder function

	padding:
	'same': padding = (input_size - filter_size)
	'''

	def __init__(self):
		super(autoencoder, self).__init__()
		self.encoder = nn.Sequential(
			nn.Conv1d(2, 16, kernel_size=4, padding=2, bias=True),
			nn.ReLU(True),
			nn.MaxPool1d(2, stride=2),
			nn.Conv1d(16, 32, kernel_size=4, padding=2, bias=True),
			nn.ReLU(True),
			nn.MaxPool1d(2, stride=2),
			nn.Conv1d(32, 64, kernel_size=4, padding=2, bias=True),
			nn.ReLU(True),
			nn.MaxPool1d(2, stride=2)
			)
		self.decoder = nn.Sequential(
			nn.ConvTranspose1d(64, 32, 4, stride=2),
			nn.ReLU(True),
			nn.ConvTranspose1d(32, 16, 4, stride=2),
			nn.ReLU(True),
			nn.ConvTranspose1d(16, 2, 4, stride=2)
			)

	def forward(self, x):
		x = self.encoder(x)
		x = self.decoder(x)
		return x



