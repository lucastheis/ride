from PIL import Image as PILImage
from numpy import ndarray, array, repeat, min, max, real
from utils import indent
from axes import Axes
from settings import Settings
from os import path
from colormap import colormaps

class Image(object):
	"""
	Represents images.
	"""

	_counter = 0

	def __init__(self, image, **kwargs):
		"""
		@type  image: string/array_like/PIL Image
		@param image: a filepath or an image in grayscale or RGB

		@param vmin:

		@param vmax:

		@param cmap:

		@param xmin:
		@param xmax:
		@param ymin:
		@param ymax:

		@param limits:
		"""

		self._cmap = kwargs.get('cmap', 'gray')

		if isinstance(image, str):
			self.image = PILImage.open(image)

		elif isinstance(image, PILImage.Image):
			self.image = image.copy()

		else:
			if isinstance(image, ndarray):
				# copy array
				image = array(image)

				if image.dtype.kind not in ['u', 'i']:
					self.vmin = kwargs.get('vmin', min(image))
					self.vmax = kwargs.get('vmax', max(image))

					# rescale image
					image = (image - self.vmin) / (self.vmax - self.vmin)
					image = array(image * 256., dtype='int32')

				image[image < 0] = 0
				image[image > 255] = 255
				image = array(image, dtype='uint8')

				if image.ndim < 3:
					image = repeat(image.reshape(image.shape[0], -1, 1), 3, 2)
					for i in range(image.shape[0]):
						for j in range(image.shape[1]):
							image[i, j, :] = colormaps[self._cmap][image[i, j, 0]]

			self.image = PILImage.fromarray(image)

		# specify pixel coordinates 
		self.xmin = kwargs.get('xmin', 0)
		self.xmax = kwargs.get('xmax', self.image.size[0])
		self.ymin = kwargs.get('ymin', 0)
		self.ymax = kwargs.get('ymax', self.image.size[1])

		if 'limits' in kwargs:
			self.xmin, self.xmax, \
			self.ymin, self.ymax = kwargs['limits']

		# add image to axis
		self.axes = kwargs.get('axes', Axes.gca())
		self.axes.children.append(self)

		self.idx = Image._counter
		Image._counter += 1


	def filename(self):
		return \
			str(self.axes.figure._session) + '_' + \
			str(self.idx) + '.' + Settings.image_format.lower()


	def save(self, filepath=''):
		self.image.save(path.join(filepath, self.filename()))


	def render(self):
		"""
		Produces LaTeX code for this image.

		@rtype: string
		@return: LaTeX code for this plot
		"""

		tex = '\\addplot[point meta min={0:5f}, point meta max={1:5f}] graphics\n'.format(self.vmin, self.vmax)
		tex += indent('[xmin={0},xmax={1},ymin={2},ymax={3}]\n'.format(*self.limits()))
		tex += indent('{' + path.join(Settings.image_folder, self.filename()) + '};\n')

		return tex


	def limits(self, limits=None):
		if limits is not None:
			self.xmin, self.xmax, \
			self.ymin, self.ymax = limits
		return [self.xmin, self.xmax, self.ymin, self.ymax]


	def  width(self):
		return self.image.size[0]


	def  height(self):
		return self.image.size[1]
