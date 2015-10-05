"""
Compute average log-likelihood for recurrent image model.
"""

import sys
import caffe

sys.path.append('./code')

from argparse import ArgumentParser
from numpy import mean, ceil, std, inf, sqrt
from numpy.random import rand
from scipy.io import loadmat
from cmt.utils import random_select
from tools import Experiment
from ride import PatchRIDE

def main(argv):
	parser = ArgumentParser(argv[0], description=__doc__)
	parser.add_argument('model',              type=str)
	parser.add_argument('--data',       '-d', type=str, default='data/deadleaves_test.mat')
	parser.add_argument('--patch_size', '-p', type=int, default=64,
		help='Images are split into patches of this size and evaluated separately.')
	parser.add_argument('--fraction',   '-f', type=float, default=1.,
		help='Only use a fraction of the data for a faster estimate.')
	parser.add_argument('--mode',       '-q', type=str,   default='CPU', choices=['CPU', 'GPU'])
	parser.add_argument('--device',     '-D', type=int,   default=0)
	parser.add_argument('--stride',     '-S', type=int,   default=1)

	args = parser.parse_args(argv[1:])

	# select CPU or GPU for caffe
	if args.mode.upper() == 'GPU':
		caffe.set_mode_gpu()
		caffe.set_device(args.device)
	else:
		caffe.set_mode_cpu()

	# load data
	data = loadmat(args.data)['data']

	# load model
	experiment = Experiment(args.model)
	model = experiment['model']

	if isinstance(model, PatchRIDE):
		if args.patch_size != model.num_rows or args.patch_size != model.num_cols:
			print 'Model is for {0}x{1} patches but data is {2}x{2}.'.format(
				model.num_rows, model.num_cols, args.patch_size)
			return 0

	# apply model to data
	logloss = []

	for i in range(0, data.shape[1] - args.patch_size + 1, args.patch_size * args.stride):
		for j in range(0, data.shape[2] - args.patch_size + 1, args.patch_size * args.stride):
			# select random subset
			idx = random_select(int(ceil(args.fraction * data.shape[0]) + .5), data.shape[0])

			logloss.append(
				model.evaluate(
					data[:, i:i + args.patch_size, j:j + args.patch_size][idx]))

			loglik_avg = -mean(logloss)
			loglik_err = std(logloss, ddof=1) / sqrt(len(logloss)) if len(logloss) > 1 else inf
			print 'Avg. log-likelihood: {0:.5f} +- {1:.5f} [bit/px]'.format(loglik_avg, loglik_err)

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
