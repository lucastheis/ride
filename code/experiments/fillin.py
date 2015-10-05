"""
Replace a square region in an image with inpainted pixels.
"""

import os
import sys

sys.path.append('./code')

from argparse import ArgumentParser
from numpy import zeros, asarray, dot, hstack, savez, load, sqrt, exp, argmax
from numpy.random import rand, randn
from copy import deepcopy
from scipy.io import loadmat
from cmt.tools import imread, imwrite, imformat, generate_data_from_image
from tools import Experiment, mapp

def main(argv):
	parser = ArgumentParser(argv[0], description=__doc__)
	parser.add_argument('image',                    type=str)
	parser.add_argument('model',                    type=str)
	parser.add_argument('--init',             '-I', type=str, default=None)
	parser.add_argument('--index',            '-x', type=int, default=0,
		help='Determines which image is used when whole dataset is given instead of image.')
	parser.add_argument('--fill_region',      '-f', type=int, default=71)
	parser.add_argument('--outer_patch_size', '-p', type=int, default=19)
	parser.add_argument('--inner_patch_size', '-i', type=int, default=5)
	parser.add_argument('--stride',           '-s', type=int, default=3)
	parser.add_argument('--candidates',       '-C', type=int, default=5,
		help='The best initialization is taken out of this many initializations.')
	parser.add_argument('--num_epochs',       '-e', type=int, default=1000)
	parser.add_argument('--method',           '-m', type=str, default='SAMPLE', choices=['SAMPLE', 'MAP'])
	parser.add_argument('--step_width',       '-l', type=float, default=100.)
	parser.add_argument('--output',           '-o', type=str, default='results/inpainting/')
	parser.add_argument('--flip',             '-F', type=int, default=0,
		help='If > 0, assume horizontal symmetry. If > 1, assume vertical symmetry.')

	args = parser.parse_args(argv[1:])


	### DATA

	# load image
	if args.image.lower()[-4:] in ['.gif', '.png', '.jpg', 'jpeg']:
		image = imread(args.image)[None]
		vmin, vmax = 0, 255
	else:
		image = loadmat(args.image)['data'][[args.index]]
		vmin, vmax = image.min(), image.max()
	if image.ndim < 4:
		image = image[:, :, :, None]

	image = asarray(image, dtype=float)

	imwrite(os.path.join(args.output, 'original.png'),
		imformat(image[0, :, :, 0], vmin=vmin, vmax=vmax, symmetric=False))

	# remove center portion
	i_start = (image.shape[1] - args.fill_region) // 2
	j_start = (image.shape[2] - args.fill_region) // 2
	image[0,
		i_start:i_start + args.fill_region,
		j_start:j_start + args.fill_region, 0] = vmin + rand(args.fill_region, args.fill_region) * (vmax - vmin)

	imwrite(os.path.join(args.output, 'start.png'),
		imformat(image[0, :, :, 0], vmin=vmin, vmax=vmax, symmetric=False))


	### MODEL

	# load model
	model = Experiment(args.model)['model']
	model.verbosity = False

	# use different models for sampling and likelihoods because of SLSTM caching
	model_copy = deepcopy(model)

	# create mask indicating pixels to replace
	M = args.outer_patch_size
	N = args.inner_patch_size
	m = (M - N) // 2
	n = M - N - m
	patch_mask = zeros([M, M], dtype=bool)
	patch_mask[m:-n, m:-n] = True

	if args.init is None:
		candidates = []
		logliks = []

		for _ in range(args.candidates):
			# replace missing pixels by ancestral sampling
			patch = image[:,
				i_start - M:i_start + args.fill_region,
				j_start - M:j_start + args.fill_region + M]
			sample_mask = zeros([patch.shape[1], patch.shape[2]], dtype=bool)
			sample_mask[M:, M:-M] = True
			image[:,
				i_start - M:i_start + args.fill_region,
				j_start - M:j_start + args.fill_region + M] = model.sample(patch, mask=sample_mask,
					min_values=vmin, max_values=vmax)

			candidates.append(image.copy())
			logliks.append(model.loglikelihood(image).sum())

		image = candidates[argmax(logliks)]

		imwrite(os.path.join(args.output, 'fillin.0.png'),
			imformat(image[0, :, :, 0], vmin=vmin, vmax=vmax, symmetric=False))

		start_epoch = 0

	else:
		init = load(args.init)
		image = init['image']
		start_epoch = init['epoch']


	### INPAINTING

	try:
		for epoch in range(start_epoch, args.num_epochs):
			print epoch
			
			h_flipped = False
			if args.flip > 0 and rand() < .5:
				print 'Horizontal flip.'
				# flip image horizontally
				image = image[:, :, ::-1]
				j_start = image.shape[2] - j_start - args.fill_region
				h_flipped = True

			v_flipped = False
			if args.flip > 0 and rand() < .5:
				print 'Vertical flip.'
				# flip image vertically
				image = image[:, ::-1, :]
				i_start = image.shape[1] - i_start - args.fill_region
				v_flipped = True

			for i in range(i_start - m, i_start - m + args.fill_region - N + 1, args.stride):
				for j in range(j_start - m, j_start - m + args.fill_region - N + 1, args.stride):
					patch = image[:, i:i + M, j:j + M]

					if args.method == 'SAMPLE':
						# proposal
						patch_pr, logq_pr = model.sample(patch.copy(), mask=patch_mask,
							min_values=vmin, max_values=vmax, return_loglik=True)

						# conditional log-density
						logq = model_copy._logq(patch, patch_mask)

						# joint log-densities
						logp = model_copy.loglikelihood(patch).sum()
						logp_pr = model_copy.loglikelihood(patch_pr).sum()

						if rand() < exp(logp_pr - logp - logq_pr + logq):
							# accept proposal
							patch[:] = patch_pr

					else:
						# gradient step
						grad = model.gradient(patch)[1]
						patch[:, patch_mask] += grad[:, patch_mask] * args.step_width

			# flip back
			if h_flipped:
				image = image[:, :, ::-1]
				j_start = image.shape[2] - j_start - args.fill_region
			if v_flipped:
				image = image[:, ::-1, :]
				i_start = image.shape[1] - i_start - args.fill_region

			imwrite(os.path.join(args.output, 'fillin.{0}.png'.format(epoch + 1)),
				imformat(image[0, :, :, 0], vmin=vmin, vmax=vmax, symmetric=False))

	except KeyboardInterrupt:
		pass

	savez(os.path.join(args.output, 'final.npz'), image=image, epoch=epoch)

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
