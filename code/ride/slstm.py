__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@theis.io>'
__docformat__ = 'epytext'

import os
import jinja2
import caffe

from numpy import zeros, zeros_like, vstack
from tempfile import mkstemp

class SLSTM(object):
	"""
	An implementation of the spatial long short term memory recurrent neural network.
	"""

	def generate_prototxt(
		self,
		num_rows=11,
		num_cols=11,
		num_channels=3,
		num_hiddens=10,
		batch_size=50,
		nonlinearity='TanH',
		name='slstm'):
		"""
		Generates a Caffe protocol buffer description of a spatial LSTM network.

		@type  num_rows: C{int}
		@param num_rows: number of rows of spatial LSTM

		@type  num_cols: C{int}
		@param num_cols: number of columns of spatial LSTM

		@type  num_channels: C{int}
		@param num_channels: dimensionality of each input pixel

		@type  num_hiddens: C{int}
		@param num_hiddens: number of hidden units

		@type  batch_size: C{int}
		@param batch_size: number of inputs processed together

		@type  nonlinearity: C{str}
		@param nonlinearity: nonlinearity used by LSTM units

		@type  name: C{str}
		@param name: an arbitrary name for the network

		@rtype: C{str}
		@return: the protocol buffer (prototxt) description of the network
		"""

		layers = []

		for i in range(num_rows):
			for j in range(num_cols):
				layers.append({
					'i': i,
					'j': j,
					'i_jm1': '{0}_{1}'.format(i, j - 1) if j > 0 else 'init_i_jm1',
					'im1_j': '{0}_{1}'.format(i - 1, j) if i > 0 else 'init_im1_j'})

		with open(self.PROTOTXT_TPL) as handle:
			prototxt = jinja2.Template(handle.read()).render(
				name=name,
				batch_size=batch_size,
				num_pixels=num_rows * num_cols,
				num_channels=num_channels,
				num_hiddens=num_hiddens,
				layers=layers,
				nonlinearity=nonlinearity)

		if num_rows * num_cols < 2:
			prototxt = prototxt.replace('x_0_0', 'inputs')
			prototxt = prototxt.replace('h_0_0', 'outputs')

		# remove empty lines
		prototxt = '\n'.join([s for s in prototxt.splitlines() if s.strip()])

		return prototxt



	def __init__(self,
			num_rows=11,
			num_cols=11,
			num_channels=3,
			num_hiddens=10,
			batch_size=50,
			nonlinearity='TanH',
			extended=False,
			slstm=None,
			verbosity=1):
		"""
		Create Caffe network.

		@type  num_rows: C{int}
		@param num_rows: number of rows of spatial LSTM

		@type  num_cols: C{int}
		@param num_cols: number of columns of spatial LSTM

		@type  num_channels: C{int}
		@param num_channels: dimensionality of each input pixel

		@type  num_hiddens: C{int}
		@param num_hiddens: number of hidden units

		@type  batch_size: C{int}
		@param batch_size: number of inputs processed together

		@type  slstm: L{SLSTM}
		@param slstm: use parameters of this spatial LSTM for initialization

		@type  nonlinearity: C{str}
		@param nonlinearity: nonlinearity used by LSTM units

		@type  verbosity: C{int}
		@param verbosity: caffe output suppressed if C{verbosity < 2}

		@type  extended: C{int}
		@param extended: give control units access to memory units
		"""

		# location of prototxt template file
		if extended:
			self.PROTOTXT_TPL = os.path.join(
				os.path.dirname(os.path.realpath(__file__)), 'slstm_extended.prototxt')
		else:
			self.PROTOTXT_TPL = os.path.join(
				os.path.dirname(os.path.realpath(__file__)), 'slstm.prototxt')

		if slstm is not None:
			if slstm.num_hiddens != num_hiddens:
				raise ValueError('`slstm` has wrong number of hidden units.')
			if slstm.num_channels != num_channels:
				raise ValueError('`slstm` has wrong number of channels.')

		self.num_rows = num_rows
		self.num_cols = num_cols
		self.num_channels = num_channels
		self.num_hiddens = num_hiddens
		self.batch_size = batch_size
		self.nonlinearity = nonlinearity
		self.verbosity = verbosity
		self.extended = extended

		# create prototxt file
		prototxt = self.generate_prototxt(
			num_rows=self.num_rows,
			num_cols=self.num_cols,
			num_channels=self.num_channels,
			num_hiddens=self.num_hiddens,
			nonlinearity=self.nonlinearity,
			batch_size=self.batch_size)

		filepath = mkstemp()[1]
		with open(filepath, 'w') as handle:
			handle.write(prototxt)

		if self.verbosity < 2:
			# silence stdout and stderr
			stdout = os.dup(1)
			stderr = os.dup(2)

			devnull1 = os.open(os.devnull, os.O_RDWR)
			devnull2 = os.open(os.devnull, os.O_RDWR)

			os.dup2(devnull1, 1)
			os.dup2(devnull2, 2)

		# create Caffe network
		self.net = caffe.Net(filepath, caffe.TEST)

		if self.verbosity < 2:
			# restore stdout and stderr
			os.dup2(stdout, 1)
			os.dup2(stderr, 2)

			os.close(devnull1)
			os.close(devnull2)

		if slstm is not None:
			# copy parameters
			self.set_parameters(slstm.parameters())



	def forward(self, inputs):
		"""
		Compute hidden unit activations.

		@type  inputs: C{ndarray}
		@param inputs: a four-dimensional array (num_batches x num_rows x num_cols x num_channels)

		@rtype: C{ndarray}
		@return: a four-dimensional array storing hidden activations for each input pixel
		"""

		if inputs.shape[1] != self.num_rows or inputs.shape[2] != self.num_cols:
			raise ValueError('`inputs` has wrong dimensions.')
		if inputs.shape[3] != self.num_channels:
			raise ValueError('`inputs` has wrong number of channels.')

		outputs = []

		for k in range(0, inputs.shape[0], self.batch_size):
			if inputs.shape[0] - k >= self.batch_size:
				# remaining number of data points is large enough
				self.net.blobs['inputs'].data[:] = inputs[k:k + self.batch_size].reshape(
					self.batch_size, -1, self.num_channels)
				self.net.forward()

				outputs.append(
					self.net.blobs['outputs'].data.reshape(
						self.batch_size,
						inputs.shape[1],
						inputs.shape[2],
						self.num_hiddens).copy())

			else:
				# remaining number of data points is smaller than batch size
				lstm = self.__class__(
					num_rows=self.num_rows,
					num_cols=self.num_cols,
					num_channels=self.num_channels,
					num_hiddens=self.num_hiddens,
					batch_size=inputs.shape[0] - k,
					nonlinearity=self.nonlinearity,
					slstm=self,
					verbosity=self.verbosity,
					extended=self.extended)

				outputs.append(lstm.forward(inputs[k:]))

		return vstack(outputs) if len(outputs) > 1 else outputs[0]



	def backward(self, gradients, force_backward=False):
		"""
		Compute gradients of parameters.

		@type  gradients: C{ndarray}
		@param gradients: derivatives of some loss with respect to hidden unit activations

		@type  force_backward: C{ndarray}
		@param force_backward: also return gradient of input units

		@rtype: C{dict}
		@return: backpropagated gradients of parameters and inputs
		"""

		if gradients.shape[0] != self.batch_size:
			raise ValueError('`gradients` has wrong batch size.')
		if gradients.shape[1] != self.num_rows or gradients.shape[2] != self.num_cols:
			raise ValueError('`gradients` has wrong dimensions.')
		if gradients.shape[3] != self.num_hiddens:
			raise ValueError('`gradients` has wrong number of channels.')

		# backpropagate gradients
		self.net.blobs['outputs'].diff[:] = gradients.reshape(
			self.batch_size, -1, self.num_hiddens)
		self.net.backward()

		grad = {}

		if force_backward:
			# store gradient of input pixels
			grad['inputs'] = self.net.blobs['inputs'].diff.reshape(
				-1, self.num_rows, self.num_cols, self.num_channels)

		grad['W_g'] = self.net.params['g_0_0'][0].diff
		grad['b_g'] = self.net.params['g_0_0'][1].diff
		grad['W_i'] = self.net.params['i_0_0'][0].diff
		grad['b_i'] = self.net.params['i_0_0'][1].diff
		grad['W_f_i'] = self.net.params['f_0_0_i'][0].diff
		grad['b_f_i'] = self.net.params['f_0_0_i'][1].diff
		grad['W_f_j'] = self.net.params['f_0_0_j'][0].diff
		grad['b_f_j'] = self.net.params['f_0_0_j'][1].diff
		grad['W_o'] = self.net.params['o_0_0'][0].diff
		grad['b_o'] = self.net.params['o_0_0'][1].diff

		return grad



	def parameters(self):
		"""
		Return all parameters of the spatial LSTM.

		@rtype: C{dict}
		@return: a dictionary storing the model parameters in arrays

		@see: L{set_parameters}
		"""

		return {
			'W_g': self.net.params['g_0_0'][0].data,
			'b_g': self.net.params['g_0_0'][1].data,
			'W_i': self.net.params['i_0_0'][0].data,
			'b_i': self.net.params['i_0_0'][1].data,
			'W_f_i': self.net.params['f_0_0_i'][0].data,
			'b_f_i': self.net.params['f_0_0_i'][1].data,
			'W_f_j': self.net.params['f_0_0_j'][0].data,
			'b_f_j': self.net.params['f_0_0_j'][1].data,
			'W_o': self.net.params['o_0_0'][0].data,
			'b_o': self.net.params['o_0_0'][1].data}



	def set_parameters(self, params):
		"""
		Set parameters of the spatial LSTM.

		@type  params: C{dict}
		@param params: a dictionary storing the model parameters in arrays

		@see: L{parameters}
		"""

		self.net.params['g_0_0'][0].data[:] = params['W_g']
		self.net.params['g_0_0'][1].data[:] = params['b_g']
		self.net.params['i_0_0'][0].data[:] = params['W_i']
		self.net.params['i_0_0'][1].data[:] = params['b_i']
		self.net.params['f_0_0_i'][0].data[:] = params['W_f_i']
		self.net.params['f_0_0_i'][1].data[:] = params['b_f_i']
		self.net.params['f_0_0_j'][0].data[:] = params['W_f_j']
		self.net.params['f_0_0_j'][1].data[:] = params['b_f_j']
		self.net.params['o_0_0'][0].data[:] = params['W_o']
		self.net.params['o_0_0'][1].data[:] = params['b_o']



	def __reduce__(self):
		"""
		Used by pickle to store model.
		"""
		return (self.__class__, (
				self.num_rows,
				self.num_cols,
				self.num_channels,
				self.num_hiddens,
				self.batch_size,
				self.nonlinearity,
				self.extended),
			self.parameters())



	def __setstate__(self, params):
		"""
		Used by pickle to load model.
		"""

		self.set_parameters(params)
