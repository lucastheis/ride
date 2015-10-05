import os
import sys

sys.path.append('./code')

from argparse import ArgumentParser
from numpy import hstack, sum, sqrt
from numpy.random import permutation
from scipy.io import loadmat
from ride import PatchRIDE, RIDE_BSDS300
from tools import Experiment
from copy import deepcopy

def main(argv):
	parser = ArgumentParser(argv[0], description=__doc__)
	parser.add_argument('--data',           '-d', type=str, default='data/BSDS300_8x8.mat')
	parser.add_argument('--nb_size',        '-b', type=int, default=5,
		help='Size of the causal neighborhood of pixels.')
	parser.add_argument('--num_train',      '-N', type=int, default=1000000)
	parser.add_argument('--num_valid',      '-V', type=int, default=200000)
	parser.add_argument('--num_hiddens',    '-n', type=int, default=64)
	parser.add_argument('--num_components', '-c', type=int, default=32)
	parser.add_argument('--num_scales',     '-s', type=int, default=4)
	parser.add_argument('--num_features',   '-f', type=int, default=32)
	parser.add_argument('--add_layer',      '-a', type=int,   default=[0], nargs='+')
	parser.add_argument('--learning_rate',  '-l', type=float, nargs='+', default=[.5, .1, .05, .01, .005, .001, 0.0005])
	parser.add_argument('--batch_size',     '-B', type=int, nargs='+', default=[50])
	parser.add_argument('--num_epochs',     '-e', type=int, default=[1], nargs='+')
	parser.add_argument('--finetune',       '-F', type=int, default=[1], nargs='+',
		help='Indicate iterations in which to finetune MCGSM with L-BFGS.')
	parser.add_argument('--precondition',   '-Q', type=int, default=1)
	parser.add_argument('--output',         '-o', type=str, default='results/BSDS300/')

	args = parser.parse_args(argv[1:])

	experiment = Experiment()

	print 'Loading data...'

	data_train = loadmat(args.data)['patches_train']
	data_valid = loadmat(args.data)['patches_valid']

	# reconstruct patches
	data_train = hstack([data_train, -sum(data_train, 1)[:, None]])
	data_valid = hstack([data_valid, -sum(data_valid, 1)[:, None]])
	patch_size = int(sqrt(data_train.shape[1]) + .5)
	data_train = data_train.reshape(-1, patch_size, patch_size)
	data_valid = data_valid.reshape(-1, patch_size, patch_size)

	print 'Creating model...'

	model = PatchRIDE(
		num_rows=8,
		num_cols=8,
		model_class=RIDE_BSDS300, # ensures the bottom-right pixel will be ignored
		nb_size=args.nb_size,
		num_hiddens=args.num_hiddens,
		num_components=args.num_components,
		num_scales=args.num_scales,
		num_features=args.num_features)

	print 'Evaluating...'

	loss = []
	loss_valid = []
	loss_valid.append(model.evaluate(data_valid))

	for i, learning_rate in enumerate(args.learning_rate):
		print 'Training...'

		if i < len(args.add_layer):
			for _ in range(args.add_layer[i]):
				# add spatial LSTM to the network
				model.add_layer()

		# randomize patch order
		data_train = data_train[permutation(data_train.shape[0])]

		# store current parameters
		model_copy = deepcopy(model)

		# train
		loss.append(
			model.train(data_train,
				learning_rate=learning_rate,
				precondition=args.precondition > 0,
				batch_size=args.batch_size[i % len(args.batch_size)],
				num_epochs=args.num_epochs[i % len(args.num_epochs)]))

		print 'Evaluating...'

		# evaluate model
		loss_valid.append(model.evaluate(data_valid))

		if loss_valid[-1] > loss_valid[-2]:
			# restore previous parameters
			model = model_copy

			print 'Performance got worse... Stopping optimization.'
			break

		# fine-tune
		if args.finetune[i % len(args.finetune)]:
			print 'Finetuning...'

			# store current parameters
			model_copy = deepcopy(model)

			model.finetune(data_train, num_samples_train=1000000, max_iter=500)

			print 'Evaluating...'

			loss_valid.append(model.evaluate(data_valid))

			if loss_valid[-1] > loss_valid[-2]:
				print 'Performance got worse... Restoring parameters.'

				model = model_copy
				loss_valid[-1] = loss_valid[-2]

		experiment['args'] = args
		experiment['loss'] = loss
		experiment['loss_valid'] = loss_valid
		experiment['model'] = model
		experiment.save(os.path.join(args.output, 'patchrim.{0}.{1}.xpck'))

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
