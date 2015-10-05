"""
Estimate the entropy of the marginal distribution (equivalently, the negative log-likelihood
of a factorial model).
"""

import sys

sys.path.append('./code')

from argparse import ArgumentParser
from numpy import histogram, log, sum
from scipy.io import loadmat

def main(argv):
	parser = ArgumentParser(argv[0], description=__doc__)
	parser.add_argument('data_train', type=str)
	parser.add_argument('data_test', type=str)

	args = parser.parse_args(argv[1:])

	data_train = loadmat(args.data_train)['data']
	data_test = loadmat(args.data_test)['data']

	p_train, bins = histogram(data_train.ravel(), 100, density=True)
	h_test, bins = histogram(data_test.ravel(), bins, density=False)

	loglik = sum(h_test * log(p_train)) / sum(h_test) / log(2.)

	print 'Cross-entropy: {0:.4f} [bit/px]'.format(-loglik)

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
