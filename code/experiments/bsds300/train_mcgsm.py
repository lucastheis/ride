"""
Train MCGSM on BSDS300 dataset.
"""

import sys

sys.path.append('./code')

from argparse import ArgumentParser
from scipy.io import loadmat
from cmt.models import MCGSM, MoGSM
from cmt.transforms import WhiteningPreconditioner
from cmt.utils import random_select
from tools import Experiment

def main(argv):
	parser = ArgumentParser(argv[0], description=__doc__)
	parser.add_argument('--data',           '-d', type=str, default='data/BSDS300_8x8.mat')
	parser.add_argument('--num_train',      '-N', type=int, default=1000000)
	parser.add_argument('--num_valid',      '-V', type=int, default=200000)
	parser.add_argument('--num_components', '-n', type=int, default=128)
	parser.add_argument('--num_scales',     '-s', type=int, default=4)
	parser.add_argument('--num_features',   '-f', type=int, default=48)
	parser.add_argument('--train_means',    '-M', type=int, default=0)
	parser.add_argument('--indices',        '-I', type=int, default=[], nargs='+')
	parser.add_argument('--initialize',     '-i', type=str, default=None)
	parser.add_argument('--verbosity',      '-v', type=int, default=1)
	parser.add_argument('--max_iter',       '-m', type=int, default=2000)

	args = parser.parse_args(argv[1:])

	experiment = Experiment()

	data_train = loadmat(args.data)['patches_train']
	data_valid = loadmat(args.data)['patches_valid']

	if args.initialize:
		results = Experiment(args.initialize)
		models = results['models']
		preconditioners = results['preconditioners']
	else:
		models = [None] * data_train.shape[1]
		preconditioners = [None] * data_train.shape[1]

	def preprocess(data, i, N):
		if N > 0 and N < data.shape[0]:
			# select subset of data
			idx = random_select(N, data.shape[0])
			return data[idx, :i].T, data[idx, i][None, :]
		return data.T[:i], data.T[[i]]

	for i in range(data_train.shape[1]):
		if args.indices and i not in args.indices:
			# skip this one
			continue

		print 'Training model {0}/{1}...'.format(i + 1, data_train.shape[1])

		inputs_train, outputs_train = preprocess(data_train, i, args.num_train)
		inputs_valid, outputs_valid = preprocess(data_valid, i, args.num_valid)

		if i > 0:
			if preconditioners[i] is None:
				preconditioners[i] = WhiteningPreconditioner(inputs_train, outputs_train)

			inputs_train, outputs_train = preconditioners[i](inputs_train, outputs_train)
			inputs_valid, outputs_valid = preconditioners[i](inputs_valid, outputs_valid)

			if models[i] is None:
				models[i] = MCGSM(
					dim_in=i,
					dim_out=1,
					num_components=args.num_components,
					num_features=args.num_features,
					num_scales=args.num_scales)
			models[i].train(
				inputs_train, outputs_train,
				inputs_valid, outputs_valid,
				parameters={
					'verbosity': 1,
					'max_iter': args.max_iter,
					'train_means': args.train_means > 0})
		else:
			preconditioners[i] = None

			if models[i] is None:
				models[i] = MoGSM(
					dim=1,
					num_components=4,
					num_scales=8)
			models[i].train(
				outputs_train,
				outputs_valid,
				parameters={
					'verbosity': 1,
					'threshold': -1.,
					'train_means': 1,
					'max_iter': 100})

		experiment['args'] = args
		experiment['models'] = models
		experiment['preconditioners'] = preconditioners
		experiment.save('results/BSDS300/snapshots/mcgsm_{0}_{1}.{{0}}.{{1}}.xpck'.format(i, args.num_components))

	if not args.indices:
		experiment['args'] = args
		experiment['models'] = models
		experiment['preconditioners'] = preconditioners
		experiment.save('results/BSDS300/mcgsm.{0}.{1}.xpck')

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
