import sys
import unittest

sys.path.append('./code')

from numpy import zeros_like, max, sum
from numpy.linalg import norm
from numpy.random import randn
from ride import RIDE

class Tests(unittest.TestCase):
	def test_gradient(self):
		model = RIDE(nb_size=3, verbosity=0)

		# set preconditioner
		model.train(
			randn(3, 8, 8, model.num_channels), num_epochs=0, precondition=True)

		X = model.sample(randn(1, 5, 5, model.num_channels))

		# compute analytical gradient
		f0, dfdX = model.gradient(X)

		# following the gradient should increase f at the expected rate
		h = 0.001
		N = norm(dfdX)
		f1 = model.gradient(X + h * dfdX / N)[0]
		self.assertAlmostEqual((f1 - f0) / (h * N), 1., 2)

		# compute numerical gradient
		h = 0.01
		dfdXn = zeros_like(dfdX)
		Xn = X.copy()
		for i in range(X.shape[1]):
			for j in range(X.shape[2]):
				Xn[0, i, j] = X[0, i, j] + h
				fp = model.gradient(Xn)[0]

				Xn[0, i, j] = X[0, i, j] - h
				fn = model.gradient(Xn)[0]

				dfdXn[0, i, j] = (fp - fn) / (2. * h)

				Xn[0, i, j] = X[0, i, j]

		# compare gradients
		self.assertLess(max(abs(dfdX - dfdXn)), 1e-5)



if __name__ == '__main__':
	unittest.main()
