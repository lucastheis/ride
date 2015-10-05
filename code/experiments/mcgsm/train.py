"""
Extract causal neighborhoods and train MCGSM.
"""

import sys

sys.path.append('./code')

from argparse import ArgumentParser
from numpy import hstack
from scipy.io import loadmat
from tools import Experiment, mapp
from cmt.transforms import WhiteningPreconditioner
from cmt.models import MCGSM
from cmt.tools import generate_masks, generate_data_from_image

def main(argv):
	experiment = Experiment()

	parser = ArgumentParser(argv[0], description=__doc__)
	parser.add_argument('--data',           '-d', type=str, default='data/vanhateren_deq2_train.mat')
	parser.add_argument('--num_data',       '-N', type=int, default=1000000)
	parser.add_argument('--num_valid',      '-V', type=int, default=200000)
	parser.add_argument('--input_size',     '-i', type=int, default=9)
	parser.add_argument('--max_iter',       '-I', type=int, default=3000)
	parser.add_argument('--num_components', '-c', type=int, default=128)
	parser.add_argument('--num_features',   '-f', type=int, default=48)
	parser.add_argument('--num_scales',     '-s', type=int, default=4)
	parser.add_argument('--verbosity',      '-v', type=int, default=1)
	parser.add_argument('--output',         '-o', type=str, default='results/vanhateren_deq2/mcgsm.{0}.{1}.xpck')

	args = parser.parse_args(argv[1:])


	### DATA HANDLING

	if args.verbosity > 0:
		print 'Loading data...'

	# load data
	images = loadmat(args.data)['data']

	# define causal neighborhood
	input_mask, output_mask = generate_masks(input_size=args.input_size, output_size=1)

	# extract causal neighborhoods
	num_samples = int((args.num_data + args.num_valid) / images.shape[0] + .9)

	def extract(image):
		return generate_data_from_image(
			image, input_mask, output_mask, num_samples)

	inputs, outputs = zip(*mapp(extract, images))
	inputs, outputs = hstack(inputs), hstack(outputs)

	inputs_train = inputs[:, :args.num_data]
	outputs_train = outputs[:, :args.num_data]
	inputs_valid = inputs[:, args.num_data:]
	outputs_valid = outputs[:, args.num_data:]

	if inputs_valid.size < 100:
		print 'Not enough data for validation.'
		inputs_valid = None
		outputs_valid = None


	### MODEL TRAINING

	if args.verbosity > 0:
		print 'Preconditioning...'

	preconditioner = WhiteningPreconditioner(inputs_train, outputs_train)

	inputs_train, outputs_train = preconditioner(inputs_train, outputs_train)
	if inputs_valid is not None:
		inputs_valid, outputs_valid = preconditioner(inputs_valid, outputs_valid)

	# free memory
	del inputs
	del outputs

	if args.verbosity > 0:
		print 'Training model...'

	model = MCGSM(
		dim_in=inputs_train.shape[0],
		dim_out=outputs_train.shape[0],
		num_components=args.num_components,
		num_features=args.num_features,
		num_scales=args.num_scales)

	def callback(i, mcgsm):
		experiment['args'] = args
		experiment['model'] = mcgsm
		experiment['preconditioner'] = preconditioner
		experiment['input_mask'] = input_mask
		experiment['output_mask'] = output_mask
		experiment.save(args.output)

	model.train(
		inputs_train, outputs_train,
		inputs_valid, outputs_valid,
		parameters={
			'verbosity': args.verbosity,
			'cb_iter': 500,
			'callback': callback,
			'max_iter': args.max_iter})


	### SAVE RESULTS

	experiment['args'] = args
	experiment['model'] = model
	experiment['preconditioner'] = preconditioner
	experiment['input_mask'] = input_mask
	experiment['output_mask'] = output_mask
	experiment.save(args.output)

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))

