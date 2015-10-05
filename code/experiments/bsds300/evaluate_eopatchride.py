"""
Compute average log-likelihood for ensemble of patch-based RIDE.
"""

import sys

sys.path.append('./code')

from argparse import ArgumentParser
from numpy import mean, ceil, log, hstack, sum, sqrt, transpose
from numpy.random import rand
from scipy.io import loadmat
from scipy.misc import logsumexp
from cmt.utils import random_select
from tools import Experiment

def transform(data, horizontal=True, vertical=False, transp=False):
	"""
	Flips patches horizontally or vertically or both. This is a volume preserving
	transformation.
	"""

	if transp:
		data = transpose(data, [0, 2, 1])
	if horizontal:
		data = data[:, :, ::-1]
	if vertical:
		data = data[:, ::-1]

	return data

def main(argv):
	parser = ArgumentParser(argv[0], description=__doc__)
	parser.add_argument('model',              type=str)
	parser.add_argument('--data',       '-d', type=str, default='data/BSDS300_8x8.mat')
	parser.add_argument('--num_data',   '-N', type=int, default=1000000)
	parser.add_argument('--batch_size', '-B', type=int, default=100000)

	args = parser.parse_args(argv[1:])

	data_test = loadmat(args.data)['patches_test']

	dim = data_test.shape[1]

	# reconstruct patches
	patch_size = int(sqrt(data_test.shape[1] + 1) + .5)
	data_test = hstack([data_test, -sum(data_test, 1)[:, None]])
	data_test = data_test.reshape(-1, patch_size, patch_size)

	if args.num_data > 0 and data_test.shape[0] > args.num_data:
		data_test = data_test[random_select(args.num_data, data_test.shape[0])]

	model = Experiment(args.model)['model']

	# container for average log-likelihoods of batches
	logliks = []

	try:
		for b in range(0, data_test.shape[0], args.batch_size):
			# select batch
			batch = data_test[b:b + args.batch_size]

			# compute average log-likelihood of different models
			loglik = []
			loglik.append(model.loglikelihood(batch))
			loglik.append(model.loglikelihood(transform(batch,  True, False, False)))
			loglik.append(model.loglikelihood(transform(batch, False,  True, False)))
			loglik.append(model.loglikelihood(transform(batch,  True,  True, False)))
			loglik.append(model.loglikelihood(transform(batch, False, False,  True)))
			loglik.append(model.loglikelihood(transform(batch,  True, False,  True)))
			loglik.append(model.loglikelihood(transform(batch, False,  True,  True)))
			loglik.append(model.loglikelihood(transform(batch,  True,  True,  True)))

			# compute average log-likelihood of mixture model
			loglik = logsumexp(loglik, 0) - log(len(loglik))

			logliks.append(loglik.mean())

			print '{0:.3f} [nat]'.format(mean(logliks))

	except KeyboardInterrupt:
		pass

	import pdb
	pdb.set_trace()



if __name__ == '__main__':
	sys.exit(main(sys.argv))
