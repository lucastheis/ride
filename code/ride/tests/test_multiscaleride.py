import sys
import unittest

sys.path.append('./code')

from numpy import abs, max
from numpy.random import randn
from numpy.linalg import slogdet
from ride import MultiscaleRIDE
from ride.multiscaleride import MS_BASIS

class Tests(unittest.TestCase):
	def test_basis(self):
		# transformation should be volume preserving
		self.assertLess(slogdet(MS_BASIS)[1], 1e-10)



	def test_transform(self):
		model = MultiscaleRIDE()

		images = randn(5, 24, 24)
		images_rec = model._transform_inverse(model._transform(images)).squeeze()

		self.assertLess(max(abs(images_rec - images)), 1e-8)



if __name__ == '__main__':
	unittest.main()
