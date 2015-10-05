from axes import Axes
from figure import Figure
from numpy import max, sum, cumsum, asarray

class AxesGrid(object):
	"""
	@type figure: L{Figure}
	@ivar figure: figure controlled by this AxesGrid instance

	@type width: float (read-only)
	@ivar width: width of grid in cm
	
	@type height: float (read-only)
	@ivar height: height of grid in cm
	"""

	def __init__(self, fig=None, *args, **kwargs):
		self.figure = fig

		# grid position within figure
		self.at = kwargs.get('at', [0, 0])

		# container for axes
		self.grid = {}

		# distance between axes
		self.spacing = kwargs.get('spacing', 2.)

		# add grid to figure
		if not self.figure:
			self.figure = Figure.gcf()
		self.figure.axes = [self]

	
	def __getitem__(self, key):
		return self.grid[key]

	
	def __setitem__(self, key, value):
		self.grid[key] = value


	def __len__(self):
		return len(self.grid)


	def keys(self):
		return self.grid.keys()


	@property
	def width(self):
		widths = self.widths()
		return sum(widths) + self.spacing * (len(widths) - 1)


	@property
	def height(self):
		heights = self.heights()
		return sum(heights) + self.spacing * (len(heights) - 1)


	def widths(self):
		if not len(self):
			return [0.]

		widths = [0.] * (max(self.keys(), 0)[1] + 1)

		for i, j in self.keys():
			widths[j] = max([self[i, j].width, widths[j]])
		return widths


	def heights(self):
		if not len(self):
			return [0.]

		heights = [0.] * (max(self.keys(), 0)[0] + 1)

		for i, j in self.keys():
			heights[i] = max([self[i, j].height, heights[i]])
		return heights


	def render(self):
		# compute axis positions
		x_pos, y_pos = [0.], [0.]
		x_pos.extend(cumsum(asarray(self.widths()) + self.spacing))
		y_pos.extend(cumsum(asarray(self.heights()) + self.spacing))

		tex = ''

		for i, j in self.keys():
			# position axis
			self[i, j].at = [x_pos[j], y_pos[i]]
			
			# render axis
			tex += self[i, j].render()

		return tex
