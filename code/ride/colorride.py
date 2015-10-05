from numpy import empty, concatenate, power, array, dot, log, mean, sum
from numpy.linalg import slogdet, inv
from cmt.tools import YCbCr
from .stackedride import StackedRIDE

Kr = .2989
Kg = .5870
Kb = 1. - Kr - Kg

# color matrix
YCC_BASIS = array([
	[Kr, -Kr / (1. - Kb) / 2.,              1. / 2.],
	[Kg, -Kg / (1. - Kb) / 2., -Kg / (1. - Kb) / 2.],
	[Kb,              1. / 2., -Kb / (1. - Kr) / 2.]])

class ColorRIDE(StackedRIDE):
	"""
	Models RGB images in YCC space using two RIDE models.
	"""

	def __init__(self, **kwargs):
		kwargs['num_channels'] = [1, 2]
		StackedRIDE.__init__(self, **kwargs)



	def _transform(self, images):
		return dot(images, YCC_BASIS)



	def _transform_inverse(self, images):
		return dot(images, inv(YCC_BASIS))



	def train(self, images, **kwargs):
		return StackedRIDE.train(self, self._transform(images), **kwargs)



	def finetune(self, images, **kwargs):
		return StackedRIDE.finetune(self, self._transform(images), **kwargs)



	def loglikelihood(self, images, **kwargs):
		return StackedRIDE.loglikelihood(self, self._transform(images), **kwargs) \
			+ slogdet(YCC_BASIS)[1]



	def evaluate(self, images, **kwargs):
		return -mean(self.loglikelihood(images)) / log(2.)



	def sample(self, images, **kwargs):
		results = StackedRIDE.sample(self, self._transform(images), **kwargs)
		if isinstance(results, tuple):
			return self._transform_inverse(results[0]).reshape(images.shape), results[1]
		else:
			return self._transform_inverse(results).reshape(images.shape)



	def _logq(self, images, **kwargs):
		return StackedRIDE._logq(self, self._transform(images), **kwargs)
