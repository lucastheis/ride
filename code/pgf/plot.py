from numpy import asarray, arange, min, max, shape, zeros
from axes import Axes
from string import replace
from re import match
from rgb import RGB
from utils import indent

class Plot(object):
	"""
	Represents line plots.

	@type axes: L{Axes}
	@ivar axes: add plot to these axes

	@type xvalues: array_like
	@ivar xvalues: x-coordinates of data points

	@type yvalues: array_like
	@ivar yvalues: y-coordinates of data points

	@type labels: list/None
	@ivar labels: a list of strings labeling each data point 

	@type line_style: string/None
	@ivar line_style: solid, dashed, dotted or other line styles

	@type line_width: float/None
	@ivar line_width: line width in points (pt)

	@type opacity: float/None
	@ivar opacity: opacity between 0.0 and 1.0

	@type color: string/L{RGB}/None
	@ivar color: line and marker color

	@type fill: boolean/string/L{RGB}
	@ivar fill: fill color; if true, a color is picked automatically

	@type marker: string/None
	@ivar marker: marker style in PGFPlots notation

	@type marker_size: float/None
	@ivar marker_size: marker size

	@type marker_edge_color: string/L{RGB}/None
	@ivar marker_edge_color: marker border color

	@type marker_face_color: string/L{RGB}/None
	@ivar marker_face_color: marker color

	@type marker_opacity: float/None
	@ivar marker_opacity: marker opacity between 0.0 and 1.0

	@type xvalues_error: array_like
	@ivar xvalues_error: size of error bars in x-direction

	@type yvalues_error: array_like
	@ivar yvalues_error: size of error bars in y-direction

	@type error_marker: string/None
	@ivar error_marker: marker used for error bars, e.g. 'none'

	@type error_color: string/L{RGB}/None
	@ivar error_color: color of error bars

	@type error_style: float/None
	@ivar error_style: error bar style, e.g. solid, dashed or dotted

	@type error_width: float/None
	@ivar error_width: error bar line width in points (pt)

	@type ycomb: boolean
	@ivar ycomb: enables comb (or stem) plot

	@type xcomb: boolean
	@ivar xcomb: enables horizontal comb (or stem) plot

	@type closed: boolean
	@ivar closed: if enabled, the last point is connected to the first

	@type const_plot: boolean
	@ivar const_plot: if true, values will no longer be linearly interpolated

	@type pattern: string/None
	@ivar pattern: a PGF pattern to fill bar and area plots

	@type legend_entry: string/None
	@ivar legend_entry: legend entry for this plot

	@type pgf_options: list
	@ivar pgf_options: custom PGFPlots plot options

	@type comment: string
	@ivar comment: can be used to put a comment into the LaTeX code
	"""

	def __init__(self, *args, **kwargs):
		"""
		Initializes plot properties.
		"""

		# data points
		if len(args) < 1:
			self.xvalues = asarray([])
			self.yvalues = asarray([])
		elif len(args) < 2:
			self.yvalues = asarray(args[0]).flatten()
			self.xvalues = arange(1, len(self.yvalues) + 1)
		else:
			self.xvalues = asarray(args[0]).flatten()
			self.yvalues = asarray(args[1]).flatten()

		# labels for each data point
		self.labels = kwargs.get('labels', None)

		if isinstance(self.labels, str):
			self.labels = [self.labels]

		if self.labels and len(self.labels) != len(self.xvalues):
			raise ValueError('The number of labels should correspond to the number of data points.')

		# line style
		self.line_style = kwargs.get('line_style', None)
		self.line_width = kwargs.get('line_width', None)
		self.opacity = kwargs.get('opacity', None)
		self.color = kwargs.get('color', None)

		self.fill = kwargs.get('fill', False)

		# marker style
		self.marker = kwargs.get('marker', None)
		self.marker_size = kwargs.get('marker_size', None)
		self.marker_edge_color = kwargs.get('marker_edge_color', None)
		self.marker_face_color = kwargs.get('marker_face_color', None)
		self.marker_opacity = kwargs.get('marker_opacity', None)

		# error bars
		self.xvalues_error = asarray(kwargs.get('xvalues_error', [])).flatten()
		self.yvalues_error = asarray(kwargs.get('yvalues_error', [])).flatten()
		self.error_marker = kwargs.get('error_marker', None)
		self.error_color = kwargs.get('error_color', None)
		self.error_style = kwargs.get('error_style', None)
		self.error_width = kwargs.get('error_width', None)

		# PGF pattern to fill area and bar plots
		self.pattern = kwargs.get('pattern', None)

		# comb (or stem) plots
		self.ycomb = kwargs.get('ycomb', False)
		self.xcomb = kwargs.get('xcomb', False)

		self.closed = kwargs.get('closed', False)

		self.const_plot = kwargs.get('const_plot', False)

		# legend entry for this plot
		self.legend_entry = kwargs.get('legend_entry', None)

		# custom plot options
		self.pgf_options = kwargs.get('pgf_options', [])

		# catch common mistakes
		if not isinstance(self.pgf_options, list):
			raise TypeError('pgf_options should be a list.')

		# comment LaTeX code
		self.comment = kwargs.get('comment', '')

		# add plot to axes
		self.axes = kwargs.get('axes', Axes.gca())
		self.axes.children.append(self)


	def render(self):
		"""
		Produces LaTeX code for this plot.

		@rtype: string
		@return: LaTeX code for this plot
		"""

		options = []
		marker_options = []
		error_options = []

		if not self.axes.cycle_list and not self.axes.cycle_list_name:
			# default settings
			marker_options.append('solid')
			
			if not self.marker:
				options.append('no marks')
			elif not self.line_style:
				options.append('only marks')

		# basic properties
		if self.line_style:
			options.append(self.line_style)
		if self.line_width is not None:
			options.append('line width={0}pt'.format(self.line_width))
		if isinstance(self.color, RGB):
			options.append('color={0}'.format(self.color))
		elif isinstance(self.color, str):
			options.append(self.color)
		if self.fill:
			if isinstance(self.fill, str) or isinstance(self.fill, RGB):
				options.append('fill={0}'.format(self.fill))
			elif hasattr(self.fill, '__len__') and len(self.fill) == 3:
				print self.fill
				options.append('fill={0}'.format(RGB(*self.fill)))
			else:
				options.append('fill')
		if self.opacity is not None:
			options.append('opacity={0}'.format(self.opacity))
		if self.marker:
			options.append('mark={0}'.format(replace(self.marker, '.', '*')))

		# marker properties
		if self.marker_edge_color:
			marker_options.append(str(self.marker_edge_color))
		if self.marker_face_color:
			marker_options.append('fill={0}'.format(self.marker_face_color))
		if self.marker_size is not None:
			marker_options.append('scale={0}'.format(self.marker_size))
		if self.marker_opacity is not None:
			if self.opacity is not None:
				marker_options.append('opacity={0}'.format(self.opacity))
			marker_options.append('fill opacity={0}'.format(self.marker_opacity))
		elif self.opacity is not None:
			marker_options.append('fill opacity={0}'.format(self.opacity))
		if marker_options:
			options.append('mark options={{{0}}}'.format(', '.join(marker_options)))

		# custom properties
		options.extend(list(self.pgf_options))

		# PGF pattern
		if self.pattern:
			options.append('pattern={{{0}}}'.format(self.pattern))

		# error bar properties
		if len(self.xvalues_error) or len(self.yvalues_error):
			options.append('error bars/.cd')
		if len(self.xvalues_error):
			options.append('x dir=both')
			options.append('x explicit')
		if len(self.yvalues_error):
			options.append('y dir=both')
			options.append('y explicit')
		if self.error_marker:
			options.append('error mark={0}'.format(self.error_marker))
		if self.error_color:
			error_options.append('color={0}'.format(self.error_color))
		if self.error_style:
			error_options.append(self.error_style)
		if self.error_width:
			error_options.append('line width={0}pt'.format(self.error_width))
		if error_options:
			options.append('error bar style={{{0}}}'.format( ', '.join(error_options)))

		# comb plots
		if self.ycomb:
			options.append('ycomb')
		elif self.xcomb:
			options.append('xcomb')

		# linear interpolation
		if self.const_plot:
			options.append('const plot')

		if self.labels:
			options.append('nodes near coords')
			options.append('point meta=explicit symbolic')

		# summarize options into one string
		options_string = ', '.join(options)
		if len(options_string) > 70:
			options_string = '\n' + indent(',\n'.join(options))

		tex = '% ' + self.comment + '\n' if self.comment else ''
		if options_string:
			tex += '\\addplot+[{0}] coordinates {{\n'.format(options_string)
		else:
			tex += '\\addplot coordinates {\n'

		if len(self.xvalues_error) or len(self.yvalues_error):
			x_error = self.xvalues_error if len(self.xvalues_error) \
				else zeros(shape(self.yvalues_error))
			y_error = self.yvalues_error if len(self.yvalues_error) \
				else zeros(shape(self.xvalues_error))

			# render plot with error bars
			for x, y, e, f in zip(self.xvalues, self.yvalues, x_error, y_error):
				tex += '\t({0}, {1}) +- ({2}, {3})\n'.format(x, y, e, f)
		else:
			# render plot coordinates
			if self.labels:
				for x, y, l in zip(self.xvalues, self.yvalues, self.labels):
					tex += '\t({0}, {1}) [{2}]\n'.format(x, y, l)
			else:
				for x, y in zip(self.xvalues, self.yvalues):
					tex += '\t({0}, {1})\n'.format(x, y)
		if self.closed:
			tex += '} \\closedcycle;\n'
		else:
			tex += '};\n'

		if self.legend_entry is not None:
			tex += '\\addlegendentry{{{0}}};\n'.format(
				self.legend_entry.replace('_', '\\_'))

		return tex


	def limits(self):
		"""
		Returns data point limits as [xmin, xmax, ymin, ymax].

		@rtype: list
		@return: data point limits
		"""

		return [
			min(self.xvalues),
			max(self.xvalues),
			min(self.yvalues),
			max(self.yvalues)]
