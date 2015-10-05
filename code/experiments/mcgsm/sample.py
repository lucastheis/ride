"""
Sample image from MCGSM.
"""

import sys

sys.path.append('./code')

from argparse import ArgumentParser
from numpy import empty, power, exp, savez, percentile
from scipy.io import loadmat
from cmt.tools import sample_image, imwrite, imformat
from cmt.utils import random_select
from tools import Experiment

def main(argv):
	experiment = Experiment()

	parser = ArgumentParser(argv[0], description=__doc__)
	parser.add_argument('--model',  '-m', type=str, required=True)
	parser.add_argument('--data',   '-d', type=str, default='data/deadleaves_train.mat')
	parser.add_argument('--width',  '-W', type=int, default=512)
	parser.add_argument('--height', '-H', type=int, default=512)
	parser.add_argument('--crop',   '-C', type=int, default=16)
	parser.add_argument('--log',    '-L', type=int, default=0)
	parser.add_argument('--output', '-o', type=str, default='results/sample.png')

	args = parser.parse_args(argv[1:])

	images = loadmat(args.data)['data']
	vmin = percentile(images, 0.02)
	vmax = percentile(images, 98.)

	experiment = Experiment(args.model)

	img = empty([args.height + args.crop, args.width + 2 * args.crop])
	img.ravel()[:] = images.ravel()[random_select(img.size, images.size)]
	img = sample_image(
		img,
		experiment['model'],
		experiment['input_mask'],
		experiment['output_mask'],
		experiment['preconditioner'],
		min_value=vmin,
		max_value=vmax)

	if args.log:
		# linearize and gamma-correct
		img = power(exp(img), .45)
		vmin = power(exp(vmin), .45)
		vmax = power(exp(vmax), .45)

	imwrite(args.output, imformat(img[args.crop:, args.crop:-args.crop], vmin=vmin, vmax=vmax, symmetric=False))
	savez('sample.npz', sample=img)

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))

