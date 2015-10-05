"""
Tools for sampling and visualizing image patches.
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@tuebingen.mpg.de>'
__docformat__ = 'epytext'
__version__ = '0.2.0'

from numpy import asarray, abs, min, max, zeros, ones, vstack, hstack, percentile
try:
	from PIL import Image
except:
	pass

def stitch(patches, **kwargs):
	"""
	Stiches patches together in a grid. Returns a single image containing all patches.

	@type  patches: array_like
	@param patches: KxMxN or KxMxNx3 array of K patches

	@type  margin: integer
	@param margin: the pixel spacing between patches

	@type  bgcolor: float/array_like
	@param bgcolor: gray-scale intensity or RGB color used for the background

	@type  num_rows: integer
	@param num_rows: the number of rows of the grid

	@type  num_cols: integer
	@param num_cols: the number of columns of the grid

	@rtype: ndarray
	@return: an AxB or AxBx3 image containing the image patches 
	"""

	# parameters
	num_rows = kwargs.get('num_rows', min(patches.shape[:2]))
	num_cols = kwargs.get('num_cols', patches.shape[0] / num_rows + (patches.shape[0] % num_rows > 0))
	margin = kwargs.get('margin', 1)
	bgcolor = kwargs.get('bgcolor', 0)

	if 'num_cols' in kwargs and not 'num_rows' in kwargs:
		num_rows = patches.shape[0] / num_cols + (patches.shape[0] % num_cols > 0)

	patches = asarray(patches)
	bgcolor = asarray(bgcolor)

	if not patches.ndim in [3, 4]:
		raise ValueError('Image patch array has wrong dimensionality.')

	# reshape some variables for easier handling
	if patches.ndim > 3 and patches.shape[-1] == 3 and bgcolor.size < 3:
		bgcolor = bgcolor * ones(3)

	patches = patches.reshape(patches.shape[0], patches.shape[1], patches.shape[2], -1)
	bgcolor = bgcolor.reshape(1, 1, patches.shape[-1])

	# stitch patches together in a grid
	grid = []

	for i in range(num_rows):
		row = []
		row.append(zeros([patches.shape[1], margin, patches.shape[-1]]) + bgcolor)

		for j in range(num_cols):
			if not patches.size:
				row.append(zeros([patches.shape[1], 
					patches.shape[2] + margin, patches.shape[-1]]) + bgcolor)
				continue

			row.append(patches[0])
			row.append(zeros([patches.shape[1], margin, patches.shape[-1]]) + bgcolor)
			patches = patches[1:]

		row = hstack(row)

		grid.append(zeros([margin, row.shape[1], row.shape[-1]]) + bgcolor)
		grid.append(row)

	grid.append(zeros([margin, row.shape[1], row.shape[-1]]) + bgcolor)
	grid = vstack(grid)

	if patches.dtype == 'uint8' and grid.dtype != 'uint8':
		grid = asarray(grid + 0.5, 'uint8')

	# remove last dimension if gray-scale
	if grid.shape[-1] == 1:
		return grid[:, :, 0]

	return grid



def imsave(filename, img):
	"""
	A convenient wrapper for saving images using the PIL package.

	@type  filename: string
	@param filename: the place where the image shall be stored

	@type  img: array_like
	@param img: a gray-scale or RGB image
	"""

	Image.fromarray(imformat(img)).save(filename)



def imformat(img, symmetric=True, perc=100):
	"""
	Rescales and converts images to uint8.

	@type  img: array_like
	@param img: any image

	@type  symmetric: boolean
	@param symmetric: if true, 0. will be mapped to 128

	@type  perc: int
	@param perc: can be used to clip intensity values

	@rtype: ndarray
	@return: the converted image
	"""

	img = asarray(img)

	if 'float' in str(img.dtype) or max(img) > 255 or min(img) < 0:
		# rescale
		if symmetric:
			a = float(percentile(abs(img), perc))
			img = (img + a) / (2. * a) * 255. + 0.5
		else:
			a, b = float(percentile(img, 100 - perc)), float(percentile(img, perc))
			img = (img - a) / (b - a) * 255. + 0.5

	img[img < 0] = 0
	img[img > 255] = 255

	return asarray(img, 'uint8')



def sample(img, patch_size, num_samples):
	"""
	Generates a random sample of image patches from an image.

	@type  img: array_like
	@param img: a grayscale image
	
	@type  patch_size: tuple
	@param patch_size: height and width of patches

	@type  num_samples: integer
	@param num_samples: number of samples

	@rtype: ndarray
	@return: patches stored in an KxMxN array
	"""

	# uniformly sample patch locations
	xpos = floor(uniform(0, img.shape[0] - patch_size[0] + 1, num_samples))
	ypos = floor(uniform(0, img.shape[1] - patch_size[1] + 1, num_samples))

	# collect sample patches
	samples = []
	for i in range(num_samples):
		samples.append(img[xpos[i]:xpos[i] + patch_size[0], ypos[i]:ypos[i] + patch_size[1]])

	return array(samples)
