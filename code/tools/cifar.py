import pickle
from numpy import concatenate, transpose, mean, zeros, asarray, uint16

def load(batches=[1], grayscale=False):
	"""
	Load CIFAR batches.
	"""

	data = zeros([32 * 32 * 3, 0])
	labels = []

	for b in batches:
		with open('data/cifar.{0}.pck'.format(b)) as handle:
			pck = pickle.load(handle)

		data = concatenate([data, pck['data'].T], 1)
		labels = concatenate([labels, pck['labels']])

	if grayscale:
		data = transpose(data.T.reshape(-1, 3, 32, 32), [0, 2, 3, 1])
		data = mean(data, 3)
		data = patch2vec(data)

	else:
		data = transpose(data.T.reshape(-1, 3, 32, 32), [0, 2, 3, 1])
		data = patch2vec(data)

	return asarray(data, order='F'), uint16(labels + 0.5)



def preprocess(data, dim=1024):
	"""
	Centers the data.
	"""

	# center data
	data = data - mean(data, 1).reshape(-1, 1)

	return data



def split(data):
	data = vec2patch(data)

	batch1 = patch2vec(data[:, :16, :16])
	batch2 = patch2vec(data[:, :16, 16:])
	batch3 = patch2vec(data[:, 16:, 16:])
	batch4 = patch2vec(data[:, 16:, :16])

	batch5 = patch2vec(data[:, 8:24, 8:24])
	
	batch6 = patch2vec(data[:, 8:24, :16])
	batch7 = patch2vec(data[:, 8:24, 16:])
	batch8 = patch2vec(data[:, :16, 8:24])
	batch9 = patch2vec(data[:, 16:, 8:24])

	return batch1, batch2, batch3, batch4, batch5, \
		batch6, batch7, batch8, batch9



def vec2patch(data):
	return data.T.reshape(data.shape[1], 32, 32, -1).squeeze()



def patch2vec(data):
	return data.reshape(data.shape[0], -1).T
