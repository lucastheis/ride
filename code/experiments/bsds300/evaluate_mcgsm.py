import sys

sys.path.append('./code')

from argparse import ArgumentParser
from numpy import log
from scipy.io import loadmat
from cmt.models import MCGSM, MoGSM
from cmt.utils import random_select
from tools import Experiment

def main(argv):
	parser = ArgumentParser(argv[0], description=__doc__)
	parser.add_argument('model',            type=str)
	parser.add_argument('--data',     '-d', type=str, default='data/BSDS300_8x8.mat')
	parser.add_argument('--num_data', '-N', type=int, default=1000000)

	args = parser.parse_args(argv[1:])

	# load data
	data = loadmat(args.data)['patches_test']

	# load model
	experiment = Experiment(args.model)
	models = experiment['models']
	preconditioners = experiment['preconditioners']

	def preprocess(data, i, N):
		if N > 0 and N < data.shape[0]:
			# select subset of data
			idx = random_select(N, data.shape[0])
			return data[idx, :i].T, data[idx, i][None, :]
		return data.T[:i], data.T[[i]]

	logloss = 0.

	for i, model in enumerate(models):
		inputs, outputs = preprocess(data, i, args.num_data)

		if isinstance(model, MoGSM):
			logloss += model.evaluate(outputs)
		else:
			logloss += model.evaluate(inputs, outputs, preconditioners[i])

		print '{0}/{1} {2:.3f} [nat]'.format(
			i + 1,
			data.shape[1],
			-logloss * log(2.) / (i + 1.) * data.shape[1])

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
