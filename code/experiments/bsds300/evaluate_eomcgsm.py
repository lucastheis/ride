"""
Create and evaluate a cheap ensemble of MCGSMs.
"""

import sys

sys.path.append('./code')

from argparse import ArgumentParser
from numpy import log, sum, hstack, sqrt, mean, transpose
from scipy.io import loadmat
from scipy.misc import logsumexp
from cmt.models import MCGSM, MoGSM
from cmt.utils import random_select
from tools import Experiment

def transform(data, horizontal=True, vertical=False, transp=False):
	"""
	Flips patches horizontally or vertically or both. This is a volume preserving
	transformation.
	"""

	# reconstruct patches
	patch_size = int(sqrt(data.shape[1] + 1) + .5)
	data = hstack([data, -sum(data, 1)[:, None]])
	data = data.reshape(-1, patch_size, patch_size)

	if transp:
		data = transpose(data, [0, 2, 1])
	if horizontal:
		data = data[:, :, ::-1]
	if vertical:
		data = data[:, ::-1]

	data = data.reshape(data.shape[0], -1)

	return data[:, :data.shape[1] - 1]

def main(argv):
	parser = ArgumentParser(argv[0], description=__doc__)
	parser.add_argument('model',            type=str)
	parser.add_argument('--data',     '-d', type=str, default='data/BSDS300_8x8.mat')
	parser.add_argument('--num_data', '-N', type=int, default=1000000)

	args = parser.parse_args(argv[1:])

	# load data
	data = loadmat(args.data)['patches_test']

	if args.num_data > 0 and args.num_data < data.shape[0]:
		# select random subset of data
		data = data[random_select(args.num_data, data.shape[0])]

	print 'Transforming data...'

	# transform data
	data_all = [
		data,
		transform(data,  True, False, False),
		transform(data, False,  True, False),
		transform(data,  True,  True, False),
		transform(data, False, False,  True),
		transform(data,  True, False,  True),
		transform(data, False,  True,  True),
		transform(data,  True,  True,  True)]

	# each entry corresponds to a different model/transformation
	loglik = [0.] * len(data_all)

	# load model
	experiment = Experiment(args.model)
	models = experiment['models'] # models for the different pixels
	preconditioners = experiment['preconditioners']

	if len(models) != data.shape[1]:
		print 'Wrong number of models!'
		return 0

	for i, model in enumerate(models):
		for n, data in enumerate(data_all):
			inputs, outputs = data.T[:i], data.T[[i]]

			# this sums over pixels
			if isinstance(model, MoGSM):
				loglik[n] = loglik[n] + model.loglikelihood(outputs)
			else:
				loglik[n] = loglik[n] + model.loglikelihood(*preconditioners[i](inputs, outputs)) \
					+ preconditioners[i].logjacobian(inputs, outputs)

		print '{0}/{1}'.format(i + 1, data.shape[1])

	# each row of loglik is a different model/transformation, each column a different data point
	loglik = logsumexp(loglik, 0) - log(len(loglik))

	print '{0:.4f} [nat]'.format(mean(loglik))

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
