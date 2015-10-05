"""
Sample image from recurrenti image model.
"""

import os
import sys

sys.path.append('./code')

from argparse import ArgumentParser
from numpy import asarray, percentile, exp, power, savez, sum, concatenate
from numpy.random import randn, randint
from scipy.io import loadmat
from tools import Experiment
from ride import MultiscaleRIDE, ColorRIDE, PatchRIDE
from cmt.utils import random_select
from cmt.tools import imread, imwrite, imformat

def main(argv):
	parser = ArgumentParser(argv[0], description=__doc__)
	parser.add_argument('model',            type=str)
	parser.add_argument('--num_rows', '-r', type=int, default=256)
	parser.add_argument('--num_cols', '-c', type=int, default=256)
	parser.add_argument('--data',     '-d', type=str, default=None)
	parser.add_argument('--log',      '-L', type=int, default=0)
	parser.add_argument('--output',   '-o', type=str, default='sample.png')
	parser.add_argument('--margin',   '-M', type=int, default=8)

	args = parser.parse_args(argv[1:])

	model = Experiment(args.model)['model']

	if isinstance(model, PatchRIDE):
		img = model.sample()[0]
		imwrite(args.output, imformat(img, vmin=0, vmax=255, symmetric=False))

	else:
		if args.data is None:
			# initialize image with white noise
			img_init = randn(1,
				args.num_rows + args.margin * 2,
				args.num_cols + args.margin * 2,
				sum(model.num_channels)) / 10.
			img = model.sample(img_init)

			if args.log:
				# linearize and gamma-correct
				img = power(exp(img), .45)

			if args.margin > 0:
				img = img[:, args.margin:-args.margin, args.margin:-args.margin]

			if img.shape[-1] == 3:
				img[img > 255.] = 255.
				img[img < 0.] = 0.
				imwrite(args.output, asarray(img[0, :, :, :], dtype='uint8'))
			else:
				imwrite(args.output, imformat(img[0, :, :, 0], perc=99))

		else:
			if args.data.lower()[-4:] in ['.gif', '.png', '.jpg', 'jpeg']:
				data = imread(args.data)[None]
				vmin, vmax = 0, 255
			else:
				data = loadmat(args.data)['data']
				vmin = percentile(data, 0.02)
				vmax = percentile(data, 98.)

			if data.ndim < 4:
				data = data[:, :, :, None]

			if isinstance(model, MultiscaleRIDE):
				num_channels = 1
			elif isinstance(model, ColorRIDE):
				num_channels = 3
			else:
				num_channels = model.num_channels

			num_pixels = (args.num_rows + args.margin) * (args.num_cols + args.margin * 2)

			# initialize image with white noise (but correct marginal distribution)
			img_init = []
			for c in range(num_channels):
				indices = randint(data.size // num_channels, size=num_pixels)
				img_init.append(
					asarray(data[:, :, :, c].ravel()[indices], dtype=float).reshape(
					1,
					args.num_rows + args.margin,
					args.num_cols + args.margin * 2, 1))
			img_init = concatenate(img_init, 3)

			img_init[img_init < vmin] = vmin
			img_init[img_init > vmax] = vmax

			if isinstance(model, MultiscaleRIDE) or isinstance(model, ColorRIDE):
				data = model._transform(data)
				idx = randint(data.shape[0])
				img = model.sample(img_init,
	#				min_values=data[idx].min(1).min(0),
	#				max_values=data[idx].max(1).max(0))
					min_values=data.min(2).min(1).min(0),
					max_values=data.max(2).max(1).max(0))
			else:
	#			img_init[:] = img_init.mean()
				img = model.sample(img_init,
					min_values=percentile(data, .1),
					max_values=percentile(data, 99.8))
	#				min_values=percentile(data, 1.),
	#				max_values=percentile(data, 96.))

			if args.log:
				# linearize and gamma-correct
				img = power(exp(img), .45)
				vmin = power(exp(vmin), .45)
				vmax = power(exp(vmax), .45)

			try:
				savez(args.output.split('.')[0] + '.npz', sample=img)
			except:
				pass

			if args.margin > 0:
				img = img[:, args.margin:, args.margin:-args.margin]

			if num_channels == 1:
				imwrite(args.output,
					imformat(img[0, :, :, 0], vmin=vmin, vmax=vmax, symmetric=False))
			else:
				imwrite(args.output,
					imformat(img[0], vmin=vmin, vmax=vmax, symmetric=False))

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
