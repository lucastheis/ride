from numpy import empty, concatenate, power, asarray, dot, log, mean, sum
from .stackedride import StackedRIDE

# orthonormal Haar basis
MS_BASIS = asarray([
	[1.,  1.,  1.,  1.],
	[1., -1.,  1., -1.],
	[1.,  1., -1., -1.],
	[1., -1., -1.,  1.]]) / 2.

class MultiscaleRIDE(StackedRIDE):
	def __init__(self, **kwargs):
		kwargs['num_channels'] = [1, 3]
		StackedRIDE.__init__(self, **kwargs)



	def _transform(self, images):
		if images.ndim == 3:
			images = images[:, :, :, None]

		if images.shape[1] % 2:
			images = images[:, :-1]
		if images.shape[2] % 2:
			images = images[:, :, :-1]

		images = concatenate([
			images[:, 0::2, 0::2],
			images[:, 0::2, 1::2],
			images[:, 1::2, 0::2],
			images[:, 1::2, 1::2]], 3)

		return dot(images, MS_BASIS)



	def _transform_inverse(self, images):
		images = dot(images, MS_BASIS.T)

		images_ = empty([images.shape[0],
			images.shape[1] * 2,
			images.shape[2] * 2])
		images_[:, 0::2, 0::2] = images[:, :, :, 0]
		images_[:, 0::2, 1::2] = images[:, :, :, 1]
		images_[:, 1::2, 0::2] = images[:, :, :, 2]
		images_[:, 1::2, 1::2] = images[:, :, :, 3]

		return images_



	def train(self, images, **kwargs):
		return StackedRIDE.train(self, self._transform(images), **kwargs)



	def finetune(self, images, **kwargs):
		return StackedRIDE.finetune(self, self._transform(images), **kwargs)



	def loglikelihood(self, images, **kwargs):
		return StackedRIDE.loglikelihood(self, self._transform(images), **kwargs)



	def evaluate(self, images, **kwargs):
		return -mean(self.loglikelihood(images)) / log(2.) / sum(self.num_channels)



	def sample(self, images, **kwargs):
		return self._transform_inverse(
			StackedRIDE.sample(self, self._transform(images), **kwargs)).reshape(images.shape)
