"""
Computes correction factors needed to compute [bit/px] from [nat].
"""

import sys

sys.path.append('./code')

from argparse import ArgumentParser
from numpy import ones, zeros, histogram, mean, eye, vstack, linspace, exp, sum, log
from numpy.linalg import slogdet
from scipy.io import loadmat
from cmt.models import MoGSM
from cmt.tools import generate_data_from_image
from tools import mapp
from pgf import plot, hist, savefig, figure, axis

def dc_component(images, patch_size):
	input_mask = ones([patch_size, patch_size], dtype=bool)
	output_mask = zeros([patch_size, patch_size], dtype=bool)

	num_samples_per_image = int(1000000. / images.shape[0] + 1.)

	def extract(image):
		patches = generate_data_from_image(image,
			input_mask,
			output_mask,
			num_samples_per_image)[0]
		return patches
	patches = vstack(mapp(extract, images))
	patches = patches.reshape(patches.shape[0], -1)

	return mean(patches, 1)[None, :]

def main(argv):
	parser = ArgumentParser(argv[0], description=__doc__)
	parser.add_argument('--data_train', '-d', type=str, default='data/BSDS300_train.mat')
	parser.add_argument('--data_test', '-t', type=str, default='data/BSDS300_test.mat')
	parser.add_argument('--patch_size', '-p', type=int, default=8)

	args = parser.parse_args(argv[1:])

	A = eye(args.patch_size) - 1. / args.patch_size**2
	A[-1] = 1. / args.patch_size**2

	logjacobian = slogdet(A)[1]

	data_train = loadmat(args.data_train)['data']
	data_test = loadmat(args.data_test)['data']

	dc_train = dc_component(data_train, args.patch_size)
	dc_test = dc_component(data_test, args.patch_size)

	h_train, bins = histogram(dc_train, 60, density=True)
	h_test, bins = histogram(dc_test, bins, density=False)

	model = MoGSM(dim=1, num_components=16, num_scales=4)
	model.train(dc_train, parameters={'max_iter': 100})

	figure(sans_serif=True)
	t = linspace(0, 1, 100)
	hist(dc_train.ravel(), 100, density=True)
	plot(t, exp(model.loglikelihood(t[None]).ravel()), 'k', line_width=2)
	axis(width=5, height=5)
	savefig('dc_fit.tex')

	loglik = mean(model.loglikelihood(dc_test))

	print 'Add these two numbers to your results:'
	print 'Log-likelihood (MoGSM): {0:.4f} [nat]'.format(loglik)
	print 'Log-likelihood (histogram): {0:.4f} [nat]'.format(sum(h_test * log(h_train)) / sum(h_test))
	print 'Log-Jacobian: {0:.4f} [nat]'.format(logjacobian)

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
