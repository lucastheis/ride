__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@theis.io>'
__docformat__ = 'epytext'

from cmt.tools import generate_masks
from numpy import sum, mean, pad
from .ride import RIDE

class StackedRIDE(object):
	"""
	Allows modeling of multi-channel data using multiple recurrent image models.
	"""

	def __init__(self, num_channels=[1], nb_size=5, **kwargs):
		"""
		Create RIDE models.

		If C{num_channels} is C{[1, 2]}, for example, then the first channel of
		a three-channel image is modeled by one RIDE, and the next two channels are
		modeled conditioned on the first channel.

		Additional arguments will be passed on to the constructor of L{RIDE}.

		@type  num_channels: C{list}
		@param num_channels: a number of channels for each RIDE model

		@type  nb_size: C{int}
		@param nb_size: controls the neighborhood of pixels read from an image
		"""

		self.models = []
		self.num_channels = num_channels

		n_sum = 0

		for i, n in enumerate(num_channels):
			# generate masks
			input_mask, output_mask = generate_masks(
				input_size=[nb_size] * (n_sum + n),
				output_size=1,
				observed=[1] * n_sum + [0] * n)

			kwargs['num_channels'] = n
			kwargs['input_mask'] = input_mask
			kwargs['output_mask'] = output_mask

			self.models.append(RIDE(**kwargs))

			n_sum += n

		# double-check that all masks have the same size
		self._match_masks()



	def add_layer(self):
		"""
		Add another spatial LSTM layer to each RIDE.
		"""

		for model in self.models:
			model.add_layer()



	def train(self, images, **kwargs):
		"""
		Train all RIDEs.

		Arguments are passed on to the training method of L{RIDE}.

		@type  images: C{ndarray}/C{list}
		@param images: array or list of images to process
		"""

		loss = []
		for i, model in enumerate(self.models):
			loss.extend(
				model.train(images[:, :, :, :sum(self.num_channels[:i + 1])], **kwargs))
		return loss



	def finetune(self, images, **kwargs):
		"""
		Finetune all RIDEs.

		Arguments are passed on to the finetuning method of L{RIDE}.

		@type  images: C{ndarray}/C{list}
		@param images: array or list of images to process
		"""

		for i, model in enumerate(self.models):
			model.finetune(images[:, :, :, :sum(self.num_channels[:i + 1])], **kwargs)
		return True



	def loglikelihood(self, images, **kwargs):
		"""
		Returns a log-likelihood for each reachable pixel (in nats).

		@type  images: C{ndarray}/C{list}
		@param images: array or list of images for which to evaluate log-likelihood

		@rtype: C{ndarray}
		@return: an array of log-likelihoods for each image and predicted pixel
		"""

		loglik = []
		for i, model in enumerate(self.models):
			loglik.append(
				model.loglikelihood(images[:, :, :, :sum(self.num_channels[:i + 1])], **kwargs))
		return sum(loglik, 0)



	def evaluate(self, images, **kwargs):
		"""
		Computes the average negative log-likelihood in bits per pixel and channel.

		@type  images: C{ndarray}/C{list}
		@param images: an array or list of test images

		@rtype: C{float}
		@return: average negative log-likelihood in bits per pixel
		"""

		return -mean(self.loglikelihood(images)) / log(2.) / sum(self.num_channels)



	def sample(self, images, min_values=None, max_values=None, **kwargs):
		"""
		Sample one or several images.

		@type  images: C{ndarray}/C{list}
		@param images: an array or a list of images to initialize pixels at boundaries

		@type  min_values: C{ndarray}/C{list}
		@param min_values: list of lower bounds for each channel (for increased stability)

		@type  max_values: C{ndarray}/C{list}
		@param max_values: list of upper bounds for each channel (for increased stability)

		@rtype: C{ndarray}
		@return: sampled images of the size of the images given as input
		"""

		if 'return_loglik' in kwargs and kwargs['return_loglik']:
			logq = [0.] * len(self.models)

			for i, model in enumerate(self.models):
				j = sum(self.num_channels[:i + 1])

				if min_values is not None:
					kwargs['min_values'] = min_values[:j]
				if max_values is not None:
					kwargs['max_values'] = max_values[:j]

				images[:, :, :, :j], logq[j] = model.sample(images[:, :, :, :j], **kwargs)
			return images, sum(logq)

		else:
			for i, model in enumerate(self.models):
				j = sum(self.num_channels[:i + 1])

				if min_values is not None:
					kwargs['min_values'] = min_values[:j]
				if max_values is not None:
					kwargs['max_values'] = max_values[:j]

				images[:, :, :, :j] = model.sample(images[:, :, :, :j], **kwargs)
			return images



	def _logq(self, images, mask):
		logq = 0.
		for i, model in enumerate(self.models):
			logq += model._logq(images[:, :, :, :sum(self.num_channels[:i + 1])], mask)
		return logq



	def __setstate__(self, state):
		self.__dict__ = state
		self._match_masks()



	def _match_masks(self):
		"""
		Make sure masks all have the same shape.
		"""

		max_size = max([model.input_mask.shape[0] for model in self.models])

		for model in self.models:
			if model.input_mask.shape[0] < max_size:
				padding = [(0, max_size - model.input_mask.shape[0])] + [(0, 0)] * (model.input_mask.ndim - 1)
				model.input_mask = pad(model.input_mask, padding, 'constant')
				model.output_mask = pad(model.output_mask, padding, 'constant')

		# used by PatchRIDE
		self.input_mask = self.models[0].input_mask
		self.input_mask.flags.writeable = False
		self.output_mask = self.models[0].output_mask
		self.output_mask.flags.writeable = False
