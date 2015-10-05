from numpy import any, zeros, ones, concatenate, dstack, zeros_like, sum
from .ride import RIDE

class PatchRIDE(object):
	"""
	Convenience class which makes it easier to apply L{RIDE} to
	image patches of a fixed size.
	"""

	def __init__(self, num_rows=8, num_cols=8, model=None, model_class=RIDE, indicators=False, **kwargs):
		self.num_rows = num_rows
		self.num_cols = num_cols
		self.indicators = indicators

		if model:
			self.ride = model
		else:
			self.ride = model_class(**kwargs)

			if self.indicators:
				# TODO: this only works for single-channel images
				if self.ride.input_mask.ndim == 2:
					self.ride.input_mask = self.ride.input_mask[:, :, None]
					self.ride.output_mask = self.ride.output_mask[:, :, None]
				self.ride.input_mask = dstack([
					self.ride.input_mask,
					self.ride.input_mask])
				self.ride.output_mask = dstack([
					self.ride.output_mask,
					zeros([
						self.ride.input_mask.shape[0],
						self.ride.input_mask.shape[1],
						1], dtype=self.ride.output_mask.dtype)])

		# let RIDE know that some of its inputs will be indicators
		self.ride._indicators = indicators



	def add_layer(self):
		self.ride.add_layer()



	def _pad(self, images):
		if images.ndim == 3:
			images = images[:, :, :, None]

		# locate output pixel
		for i_off, j_off in zip(
				range(self.ride.output_mask.shape[0]),
				range(self.ride.output_mask.shape[1])):
			if any(self.ride.output_mask[i_off, j_off]):
				break

		pad_r = self.ride.output_mask.shape[1] - j_off - 1
		pad_b = self.ride.output_mask.shape[0] - i_off - 1

		# zero-padding of image to compensate for neighborhoods
		images = concatenate([
			zeros([
				images.shape[0],
				i_off,
				images.shape[2],
				images.shape[3]], images.dtype),
			images,
			zeros([
				images.shape[0],
				pad_b,
				images.shape[2],
				images.shape[3]], images.dtype)], 1)
		images = concatenate([
			zeros([
				images.shape[0],
				images.shape[1],
				j_off,
				images.shape[3]], images.dtype),
			images,
			zeros([
				images.shape[0],
				images.shape[1],
				pad_r,
				images.shape[3]], images.dtype)], 2)

		if self.indicators:
			indicators = zeros_like(images)
			if indicators.shape[-1] > 1:
				indicators = indicators[:, :, :, [0]]
			indicators[:, :i_off] = 1.
			indicators[:, :, :j_off] = 1.
			indicators[:, :, -pad_r:] = 1.
			if pad_b > 0:
				indicators[:, -pad_b:, :] = 1.

			images = concatenate([images, indicators], 3)

		return images



	def _unpad(self, images):
		if self.indicators:
			# remove indicators
			images = images[:, :, :, :-1]

		# locate output pixel
		for i_off, j_off in zip(
				range(self.ride.output_mask.shape[0]),
				range(self.ride.output_mask.shape[1])):
			if any(self.ride.output_mask[i_off, j_off]):
				break

		pad_r = self.ride.output_mask.shape[1] - j_off - 1
		pad_b = self.ride.output_mask.shape[0] - i_off - 1

		# remove padding
		if pad_r > 0:
			images = images[:, :, :-pad_r]
		if pad_b > 0:
			images = images[:, :-pad_b, :]
		return images[:, i_off:, j_off:]



	def train(self, images, **kwargs):
		return self.ride.train(self._pad(images), **kwargs)



	def loglikelihood(self, images, **kwargs):
		"""
		Return a log-likelihood for every image patch.
		"""
		loglik = self.ride.loglikelihood(self._pad(images), **kwargs)
		loglik = loglik.reshape(loglik.shape[0], -1)
		return loglik.sum(1)



	def evaluate(self, images, **kwargs):
		"""
		Return an average negative log-likelihood in bits per pixel.
		"""
		return self.ride.evaluate(self._pad(images), **kwargs)



	def finetune(self, images, **kwargs):
		return self.ride.finetune(self._pad(images), **kwargs)



	def hidden_states(self, images, **kwargs):
		return self.ride.hidden_states(self._pad(images), **kwargs)



	def sample(self, num_samples=1, **kwargs):
		images = zeros([num_samples, self.num_rows, self.num_cols, sum(self.ride.num_channels)])
		return self._unpad(self.ride.sample(self._pad(images), **kwargs))



	def __setstate__(self, state):
		self.__dict__ = state

		if not hasattr(self, 'indicators'):
			self.indicators = False
