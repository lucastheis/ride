"""
Train recurrent image density estimator on images.
"""

import os
import sys
import caffe

sys.path.append('./code')

from argparse import ArgumentParser
from copy import deepcopy
from numpy import sqrt, vstack, hstack, std, min, asarray
from numpy.random import permutation, rand, randn
from scipy.io import loadmat
from cmt.utils import random_select
from cmt.tools import imread
from ride import RIDE, PatchRIDE, MultiscaleRIDE, ColorRIDE
from tools import Experiment

def main(argv):
	parser = ArgumentParser(argv[0], description=__doc__)
	parser.add_argument('--patch_size',      '-p', type=int,   default=[8, 10, 12, 14, 16, 18, 20, 22], nargs='+')
	parser.add_argument('--row_multiplier',  '-R', type=int, default=[1], nargs='+',
		help='Can be used to train on elongated patches.')
	parser.add_argument('--col_multiplier',  '-C', type=int, default=[1], nargs='+',
		help='Can be used to train on elongated patches.')
	parser.add_argument('--num_patches',     '-P', type=int,   default=None,
		help='If given, subsample training data.')
	parser.add_argument('--num_valid',       '-V', type=int,   default=0,
		help='Number of training images used for validation error based early stopping.')
	parser.add_argument('--finetune',        '-F', type=int,   default=[1], nargs='+',
		help='Indicate iterations in which to finetune MCGSM with L-BFGS.')
	parser.add_argument('--learning_rate',   '-l', type=float, default=[1., .5, .1, .05, .01, 0.005, 0.001, 0.0005], nargs='+')
	parser.add_argument('--momentum',        '-m', type=float, default=[.9], nargs='+')
	parser.add_argument('--batch_size',      '-B', type=int,   default=[50], nargs='+')
	parser.add_argument('--nb_size',         '-b', type=int,   default=5,
		help='Size of the causal neighborhood of pixels.')
	parser.add_argument('--num_hiddens',     '-n', type=int,   default=64)
	parser.add_argument('--num_components',  '-c', type=int,   default=32)
	parser.add_argument('--num_scales',      '-s', type=int,   default=4)
	parser.add_argument('--num_features',    '-f', type=int,   default=32)
	parser.add_argument('--num_epochs',      '-e', type=int,   default=[1], nargs='+')
	parser.add_argument('--precondition',    '-Q', type=int,   default=1)
	parser.add_argument('--method',          '-M', type=str,   default=['SGD'], nargs='+')
	parser.add_argument('--data',            '-d', type=str,   default='data/deadleaves_train.mat')
	parser.add_argument('--noise',           '-N', type=float, default=None,
		help='Standard deviation of Gaussian noise added to data before training (as fraction of data standard deviation).')
	parser.add_argument('--model',           '-I', type=str,   default='',
		help='Start with this model as initialization. Other flags will be ignored.')
	parser.add_argument('--add_layer',       '-a', type=int,   default=[0], nargs='+')
	parser.add_argument('--train_top_layer', '-T', type=int,   default=[0], nargs='+')
	parser.add_argument('--train_means',     '-S', type=int,   default=[0], nargs='+')
	parser.add_argument('--mode',            '-q', type=str,   default='CPU', choices=['CPU', 'GPU'])
	parser.add_argument('--device',          '-D', type=int,   default=0)
	parser.add_argument('--augment',         '-A', type=int,   default=1,
		help='Increase training set size by transforming data.')
	parser.add_argument('--overlap',         '-O', type=int,   default=[1], nargs='+')
	parser.add_argument('--output',          '-o', type=str,   default='results/deadleaves/')
	parser.add_argument('--patch_model',           type=int,   default=0,
		help='Train a patch-based model instead of a stochastic process.')
	parser.add_argument('--extended',        '-X', type=int,   default=0,
		help='Use extended version of spatial LSTM.')
	parser.add_argument('--multiscale',      '-Y', type=int,   default=0,
		help='Apply recurrent image model to multiscale representation of images.')
	parser.add_argument('--color',           '-Z', type=int,   default=0,
		help='Use separate models to model color and grayscale values.')

	args = parser.parse_args(argv[1:])

	experiment = Experiment()

	if args.mode.upper() == 'GPU':
		caffe.set_mode_gpu()
		caffe.set_device(args.device)
	else:
		caffe.set_mode_cpu()

	# load data
	if args.data.lower()[-4:] in ['.gif', '.png', '.jpg', 'jpeg']:
		data = imread(args.data)[None]
		data += rand(*data.shape)
	else:
		data = loadmat(args.data)['data']

	if args.augment > 0:
		data = vstack([data, data[:, :, ::-1]])
	if args.augment > 1:
		data = vstack([data, data[:, ::-1, :]])

	if args.noise is not None:
		# add noise as a means for regularization
		data += randn(*data.shape) * (std(data, ddof=1) / args.noise)

	if args.num_valid > 0:
		if args.num_valid >= data.shape[0]:
			print 'Cannot use {0} for validation, there are only {1} training images.'.format(
					args.num_valid, data.shape[0])
			return 1

		# select subset for validation
		idx = random_select(args.num_valid, data.shape[0])
		data_valid = data[idx]
		data = asarray([image for i, image in enumerate(data) if i not in idx])

		print '{0} training images'.format(data.shape[0])
		print '{0} validation images'.format(data_valid.shape[0])

		patches_valid = []
		patch_size = min([64, data.shape[1], data.shape[2]])
		for i in range(0, data_valid.shape[1] - patch_size + 1, patch_size):
			for j in range(0, data_valid.shape[2] - patch_size + 1, patch_size):
				patches_valid.append(data_valid[:, i:i + patch_size, j:j + patch_size])
		patches_valid = vstack(patches_valid)

	if args.model:
		# load pretrained model
		results = Experiment(args.model)
		model = results['model']
		loss = [results['loss']]

		if args.patch_model and not isinstance(model, PatchRIDE):
			model = PatchRIDE(
				model=model,
				num_rows=args.patch_size[0],
				num_cols=args.patch_size[0])

	else:
		# create recurrent image model
		if args.patch_model:
			model = PatchRIDE(
				num_rows=args.patch_size[0],
				num_cols=args.patch_size[0],
				num_channels=data.shape[-1] if data.ndim > 3 else 1,
				num_hiddens=args.num_hiddens,
				nb_size=args.nb_size,
				num_components=args.num_components,
				num_scales=args.num_scales,
				num_features=args.num_features,
				model_class=ColorRIDE if args.color else RIDE)

			if args.extended:
				print 'Extended patch model not supported.'
				return 0

			if args.multiscale:
				print 'Multiscale patch model not supported.'
				return 0

		else:
			if args.multiscale:
				if data.ndim > 3 and data.shape[-1] > 1:
					print 'Multiscale color model not supported.'
					return 0

				model = MultiscaleRIDE(
					num_hiddens=args.num_hiddens,
					nb_size=args.nb_size,
					num_components=args.num_components,
					num_scales=args.num_scales,
					num_features=args.num_features,
					extended=args.extended > 0)

			elif args.color:
				if data.ndim < 4 or data.shape[-1] != 3:
					print 'These images don\'t look like RGB images.'
					return 0
				model = ColorRIDE(
					num_hiddens=args.num_hiddens,
					nb_size=args.nb_size,
					num_components=args.num_components,
					num_scales=args.num_scales,
					num_features=args.num_features,
					extended=args.extended > 0)

			else:
				model = RIDE(
					num_channels=data.shape[-1] if data.ndim > 3 else 1,
					num_hiddens=args.num_hiddens,
					nb_size=args.nb_size,
					num_components=args.num_components,
					num_scales=args.num_scales,
					num_features=args.num_features,
					extended=args.extended > 0)

		loss = []

	# compute initial performance
	loss_valid = []

	if args.num_valid > 0:
		print 'Computing validation loss...'
		loss_valid.append(model.evaluate(patches_valid))
		model_copy = deepcopy(model)

	for k, patch_size in enumerate(args.patch_size):
		if args.multiscale:
			patch_size *= 2

		if k < len(args.add_layer):
			for _ in range(args.add_layer[k]):
				# add spatial LSTM to the network
				model.add_layer()

		# extract patches of given patch size
		patches = []

		row_size = patch_size * args.row_multiplier[k % len(args.row_multiplier)]
		col_size = patch_size * args.col_multiplier[k % len(args.col_multiplier)]

		if isinstance(model, PatchRIDE):
			model.num_rows = row_size
			model.num_cols = col_size

		for i in range(0, data.shape[1] - row_size + 1, row_size / args.overlap[k % len(args.overlap)]):
			for j in range(0, data.shape[2] - col_size + 1, col_size / args.overlap[k % len(args.overlap)]):
				patches.append(data[:, i:i + row_size, j:j + col_size])
		patches = vstack(patches)

		# randomize order of patches
		if args.num_patches is not None and args.num_patches < len(patches):
			patches = patches[random_select(args.num_patches, len(patches))]
		else:
			patches = patches[permutation(len(patches))]

		# determine batch size
		if args.method[k % len(args.method)].upper() == 'SFO':
			num_batches = int(max([25, sqrt(patches.shape[0]) / 5.]))
			batch_size = patches.shape[0] // num_batches
		else:
			batch_size = args.batch_size[k % len(args.batch_size)]

		if batch_size < 1:
			raise RuntimeError('Too little data.')

		print 'Patch size: {0}x{1}'.format(row_size, col_size)
		print 'Number of patches: {0}'.format(patches.shape[0])
		print 'Batch size: {0}'.format(batch_size)

		# train recurrent image model
		print 'Training...'
		loss.append(
			model.train(patches,
				batch_size=batch_size,
				method=args.method[k % len(args.method)],
				num_epochs=args.num_epochs[k % len(args.num_epochs)],
				learning_rate=args.learning_rate[k % len(args.learning_rate)],
				momentum=args.momentum[k % len(args.momentum)],
				precondition=args.precondition > 0,
				train_top_layer=args.train_top_layer[k % len(args.train_top_layer)] > 0,
				train_means=args.train_means[k % len(args.train_means)] > 0))

		if args.finetune[k % len(args.finetune)]:
			print 'Finetuning...'
			model.finetune(patches, num_samples_train=1000000, max_iter=500)

		if args.num_valid > 0:
			print 'Computing validation loss...'
			loss_valid.append(model.evaluate(patches_valid))

			if loss_valid[-1] > loss_valid[-2]:
				print 'Performance got worse. Stopping optimization.'
				model = model_copy
				break

			print 'Copying model...'

			model_copy = deepcopy(model)

		experiment['batch_size'] = batch_size
		experiment['args'] = args
		experiment['model'] = model
		experiment['loss_valid'] = loss_valid
		experiment['loss'] = hstack(loss) if len(loss) > 0 else []
		experiment.save(os.path.join(args.output, 'rim.{0}.{1}.xpck'))

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
