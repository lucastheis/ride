"""
Evaluate information rate and log-likelihood of MCGSM.
"""

import sys

sys.path.append('./code')

from argparse import ArgumentParser
from cmt.tools import generate_data_from_image
from scipy.io import loadmat
from numpy import hstack
from tools import Experiment

def main(argv):
	parser = ArgumentParser(argv[0], description=__doc__)
	parser.add_argument('model',                  type=str)
	parser.add_argument('--num_data',       '-N', type=int, default=5000000)
	parser.add_argument('--data',           '-d', type=str, default='data/vanhateren_deq2_test.mat')
	parser.add_argument('--verbosity',      '-v', type=int, default=1)

	args = parser.parse_args(argv[1:])

	### LOAD RESULTS

	experiment = Experiment(args.model)


	### DATA HANDLING

	if args.verbosity > 0:
		print 'Loading data...'

	# load data
	images = loadmat(args.data)['data']

	# causal neighborhood definition
	input_mask = experiment['input_mask']
	output_mask = experiment['output_mask']

	# extract causal neighborhoods
	num_samples = args.num_data // images.shape[0]
	data = []
	for i in range(images.shape[0]):
		data.append(generate_data_from_image(
			images[i], input_mask, output_mask, num_samples))
	inputs, outputs = zip(*data)
	inputs = hstack(inputs)
	outputs = hstack(outputs)


	### MODEL EVALUATION

	crossentropy = experiment['model'].evaluate(inputs, outputs, experiment['preconditioner'])

	print 'Cross-entropy: {0:.4f} [bit/px]'.format(crossentropy)

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
