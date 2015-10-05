"""
Evaluate information rate and log-likelihood of EoMCGSM.
"""

import sys

sys.path.append('./code')

from argparse import ArgumentParser
from cmt.tools import generate_data_from_image
from scipy.io import loadmat
from scipy.misc import logsumexp
from numpy import hstack, transpose, log, mean
from numpy.random import permutation
from tools import Experiment

def main(argv):
	parser = ArgumentParser(argv[0], description=__doc__)
	parser.add_argument('model',                  type=str)
	parser.add_argument('--data',           '-d', type=str, default='data/vanhateren_deq2_test.mat')
	parser.add_argument('--batch_size',     '-B', type=int, default=25)
	parser.add_argument('--verbosity',      '-v', type=int, default=1)

	args = parser.parse_args(argv[1:])

	### LOAD RESULTS

	experiment = Experiment(args.model)


	### DATA HANDLING

	if args.verbosity > 0:
		print 'Loading data...'

	# load data
	images = loadmat(args.data)['data']
	images = images[permutation(images.shape[0])]

	# causal neighborhood definition
	input_mask = experiment['input_mask']
	output_mask = experiment['output_mask']

	# extract causal neighborhoods
	def extract(images):
		data = []
		for i in range(images.shape[0]):
			data.append(generate_data_from_image(
				images[i], input_mask, output_mask))
		inputs, outputs = zip(*data)
		return hstack(inputs), hstack(outputs)


	### MODEL EVALUATION

	model = experiment['model']
	pre = experiment['preconditioner']

	def evaluate(images):
		print 'Extracting...'
		inputs, outputs = extract(images)
		print 'Evaluating...'
		loglik = model.loglikelihood(*pre(inputs, outputs)) \
			+ pre.logjacobian(inputs, outputs)
		return loglik.reshape(images.shape[0], -1)

	logliks = []

	for b in range(0, images.shape[0], args.batch_size):
		batch = images[b:b + args.batch_size]

		loglik = evaluate(batch)
		num_pixels = loglik.shape[1]
		loglik = [loglik.sum(1)]
		loglik.append(evaluate(batch[:, ::-1]).sum(1))
		loglik.append(evaluate(batch[:, :, ::-1]).sum(1))
		loglik.append(evaluate(batch[:, ::-1, ::-1]).sum(1))

		batch = transpose(batch, [0, 2, 1])

		loglik.append(evaluate(batch).sum(1))
		loglik.append(evaluate(batch[:, ::-1]).sum(1))
		loglik.append(evaluate(batch[:, :, ::-1]).sum(1))
		loglik.append(evaluate(batch[:, ::-1, ::-1]).sum(1))

		loglik = logsumexp(loglik, 0) - log(len(loglik))

		logliks.append(mean(loglik) / num_pixels / log(2.))

		print 'Cross-entropy: {0:.4f} [bit/px]'.format(-mean(logliks))

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
