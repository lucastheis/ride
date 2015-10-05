from axes import Axes
from numpy import meshgrid, arange

class SurfPlot(object):
	"""
	Renders basic 3D surfaces.
	"""

	def __init__(self, *args, **kwargs):
		if len(args) == 1:
			self.xvalues, self.yvalues = meshgrid(
				arange(args[0].shape[0]),
				arange(args[0].shape[1]))
			self.zvalues = args[0]
		else:
			self.xvalues, self.yvalues, self.zvalues = args

		# shading
		self.shading = kwargs.get('shading', None)

		# custom plot options
		self.pgf_options = kwargs.get('pgf_options', [])

		# catch common mistakes
		if not isinstance(self.pgf_options, list):
			raise TypeError('pgf_options should be a list.')

		# add plot to axis
		self.axes = kwargs.get('axes', Axes.gca())
		self.axes.children.append(self)

		# adjust default behavior for 3D plots
		self.axes.axis_on_top = False
		self.axes.xlabel_near_ticks = False
		self.axes.ylabel_near_ticks = False
		self.axes.zlabel_near_ticks = False


	def render(self):
		options = ['surf']
		options.append('mesh/rows={0}'.format(self.zvalues.shape[0]))

		if self.shading:
			options.append('shader={0}'.format(self.shading))

		options_string = ','.join(options)

		tex = '\\addplot3[{0}] coordinates {{\n'.format(options_string)
		for i in range(self.zvalues.shape[0]):
			for j in range(self.zvalues.shape[1]):
				tex += '\t({0}, {1}, {2})\n'.format(
					self.xvalues[i, j], self.yvalues[i, j], self.zvalues[i, j])
		tex += '};\n'

		return tex
