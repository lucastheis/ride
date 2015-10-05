import sys
import unittest

sys.path.append('./code')

from tempfile import mkstemp
from copy import deepcopy
from collections import defaultdict
from pickle import load, dump
from numpy import all, max, abs, empty_like, square, sum
from numpy.random import randn
from ride import SLSTM

class Tests(unittest.TestCase):
	def test_forward(self):
		slstm0 = SLSTM(batch_size=31)
		slstm1 = SLSTM(batch_size=30, slstm=slstm0)

		inputs = randn(31, slstm0.num_rows, slstm0.num_cols, slstm0.num_channels)

		outputs0 = slstm0.forward(inputs)
		outputs1 = slstm1.forward(inputs)

		self.assertLess(max(abs(outputs0 - outputs1)), 1e-7)

		slstm0 = SLSTM(batch_size=4)
		slstm1 = SLSTM(batch_size=30, slstm=slstm0)

		inputs = randn(29, slstm0.num_rows, slstm0.num_cols, slstm0.num_channels)

		outputs0 = slstm0.forward(inputs)
		outputs1 = slstm1.forward(inputs)

		self.assertLess(max(abs(outputs0 - outputs1)), 1e-7)



	def test_manual_forward(self):
		slstm0 = SLSTM(
			num_rows=4,
			num_cols=5,
			num_channels=2,
			num_hiddens=10,
			batch_size=5)
		slstm1 = SLSTM(
			num_rows=1,
			num_cols=1,
			num_channels=2,
			num_hiddens=10,
			batch_size=5,
			slstm=slstm0)

		inputs = randn(
			slstm0.batch_size,
			slstm0.num_rows,
			slstm0.num_cols,
			slstm0.num_channels)

		outputs0 = slstm0.forward(inputs)
		outputs1 = empty_like(outputs0)

		h1 = defaultdict(lambda: 0.)
		c1 = defaultdict(lambda: 0.)

		for i in range(slstm0.num_rows):
			for j in range(slstm0.num_cols):
				# set inputs
				slstm1.net.blobs['h_init_i_jm1'].data[:] = h1[i, j - 1]
				slstm1.net.blobs['h_init_im1_j'].data[:] = h1[i - 1, j]
				slstm1.net.blobs['c_init_i_jm1'].data[:] = c1[i, j - 1]
				slstm1.net.blobs['c_init_im1_j'].data[:] = c1[i - 1, j]

				slstm1.forward(inputs[:, [[i]], [[j]], :])

				h1[i, j] = slstm1.net.blobs['outputs'].data.copy()
				c1[i, j] = slstm1.net.blobs['c_0_0'].data.copy()

				outputs1[:, i, j, :] = h1[i, j].reshape(slstm0.batch_size, slstm0.num_hiddens)

		self.assertLess(max(abs(outputs0 - outputs1)), 1e-8)



	def test_gradient(self):
		h = 0.01

		slstm = SLSTM(
			num_rows=4,
			num_cols=3,
			batch_size=10,
			num_channels=3)

		# generate data
		inputs = randn(slstm.batch_size, slstm.num_rows, slstm.num_cols, slstm.num_channels)

		# assume we want to minimize squared hidden unit activity
		grad = slstm.backward(slstm.forward(inputs))

		params = deepcopy(slstm.parameters())
		params_copy = deepcopy(params)

		# numerical gradient
		grad_n = {}
		for key in grad:
			grad_n[key] = empty_like(grad[key])

			for i in range(grad_n[key].size):
				# increase parameter value
				params_copy[key].ravel()[i] = params[key].ravel()[i] + h

				slstm.set_parameters(params_copy)
				loss_p = sum(square(slstm.forward(inputs))) / 2.

				# decrease parameter value
				params_copy[key].ravel()[i] = params[key].ravel()[i] - h

				slstm.set_parameters(params_copy)
				loss_n = sum(square(slstm.forward(inputs))) / 2.

				# reset parameter value
				params_copy[key].ravel()[i] = params[key].ravel()[i]

				# estimate gradient
				grad_n[key].ravel()[i] = (loss_p - loss_n) / (2. * h)

			self.assertLess(max(abs(grad_n[key] - grad[key])), 1e-3)



	def test_pickle(self):
		filepath = mkstemp()[1]

		slstm0 = SLSTM(8, 12, 2, 13, 42)

		# store model
		with open(filepath, 'w') as handle:
			dump(slstm0, handle)

		# load model
		with open(filepath) as handle:
			slstm1 = load(handle)

		self.assertEqual(slstm0.num_rows, slstm1.num_rows)
		self.assertEqual(slstm0.num_cols, slstm1.num_cols)
		self.assertEqual(slstm0.num_channels, slstm1.num_channels)
		self.assertEqual(slstm0.num_hiddens, slstm1.num_hiddens)
		self.assertEqual(slstm0.batch_size, slstm1.batch_size)

		params0 = slstm0.parameters()
		params1 = slstm1.parameters()

		for key in params0:
			self.assertTrue(all(params0[key] == params1[key]))



if __name__ == '__main__':
	unittest.main()
