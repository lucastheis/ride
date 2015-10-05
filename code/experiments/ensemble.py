"""
Compute average log-likelihood for a cheap ensemble of recurrent image density estimators.
"""

import sys
import caffe

sys.path.append('./code')

from argparse import ArgumentParser
from numpy import mean, ceil, log, asarray, transpose
from numpy.random import rand
from scipy.io import loadmat
from scipy.misc import logsumexp
from cmt.utils import random_select
from tools import Experiment

def main(argv):
	parser = ArgumentParser(argv[0], description=__doc__)
	parser.add_argument('model',              type=str)
	parser.add_argument('--data',       '-d', type=str, default='data/deadleaves_test.mat')
	parser.add_argument('--patch_size', '-p', type=int, default=64,
		help='Images are split into patches of this size and evaluated separately.')
	parser.add_argument('--fraction',   '-f', type=float, default=1.)
	parser.add_argument('--mode',       '-q', type=str,   default='CPU', choices=['CPU', 'GPU'])
	parser.add_argument('--device',     '-D', type=int,   default=0)
	parser.add_argument('--horizontal', '-H', type=int,   default=1)
	parser.add_argument('--vertical',   '-V', type=int,   default=1)
	parser.add_argument('--transpose',  '-T', type=int,   default=1)

	args = parser.parse_args(argv[1:])

	# select CPU or GPU for caffe
	if args.mode.upper() == 'GPU':
		caffe.set_mode_gpu()
		caffe.set_device(args.device)
	else:
		caffe.set_mode_cpu()

	# load data
	data = loadmat(args.data)['data']
	
	if data.ndim < 4:
		data = data[:, :, :, None]

	# load model
	experiment = Experiment(args.model)
	model = experiment['model']

	# apply model to data
	logloss = []

	def evaluate(patches):
		"""
		Returns a conditional log-likelihood for each pixel.
		"""
		# evaluate all reachable pixels
		loglik = model.loglikelihood(patches)

		# make sure the same pixels are evaluated with every transformation
		loglik = loglik[:, :-(model.input_mask.shape[0] - 1), :]

		# reshape into images x pixels
		return loglik.reshape(loglik.shape[0], -1) 

	for i in range(0, data.shape[1] - args.patch_size + 1, args.patch_size):
		for j in range(0, data.shape[2] - args.patch_size + 1, args.patch_size):
			# select random subset
			idx = random_select(int(ceil(args.fraction * data.shape[0]) + .5), data.shape[0])

			patches = data[:, i:i + args.patch_size, j:j + args.patch_size][idx]

			loglik = []
			loglik.append(evaluate(patches))
			if args.horizontal:
				loglik.append(evaluate(patches[:, :, ::-1]))
			if args.vertical:
				loglik.append(evaluate(patches[:, ::-1, :]))
			if args.horizontal and args.vertical:
				loglik.append(evaluate(patches[:, ::-1, ::-1]))
			if args.transpose:
				patches = transpose(patches, [0, 2, 1, 3])
				loglik.append(evaluate(patches))
				if args.horizontal:
					loglik.append(evaluate(patches[:, :, ::-1]))
				if args.vertical:
					loglik.append(evaluate(patches[:, ::-1, :]))
				if args.horizontal and args.vertical:
					loglik.append(evaluate(patches[:, ::-1, ::-1]))
			loglik = asarray(loglik)

			# compute log-likelihood for each image and model by summing over pixels
			num_pixels = loglik.shape[2]
			loglik = loglik.sum(2) # sum over pixels

			# compute log-likelihood for mixture model
			loglik = logsumexp(loglik, 0) - log(loglik.shape[0])

			# compute average log-loss in bit/px
			logloss.append(-mean(loglik) / num_pixels / log(2.))

			print 'Avg. log-likelihood: {0:.5f} [bit/px]'.format(-mean(logloss))

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
