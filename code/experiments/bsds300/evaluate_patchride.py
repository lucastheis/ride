"""
Compute average log-likelihood for patch-based recurrent image model.
"""

import sys

sys.path.append('./code')

from argparse import ArgumentParser
from numpy import mean, ceil, log, hstack, sum, sqrt
from numpy.random import rand
from scipy.io import loadmat
from cmt.utils import random_select
from tools import Experiment

def main(argv):
	parser = ArgumentParser(argv[0], description=__doc__)
	parser.add_argument('model',              type=str)
	parser.add_argument('--data',       '-d', type=str, default='data/BSDS300_8x8.mat')
	parser.add_argument('--num_data',   '-N', type=int, default=1000000)

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

	print '{0:.3f} [nat]'.format(
		-model.evaluate(data_test) * dim * log(2.))



if __name__ == '__main__':
	sys.exit(main(sys.argv))
