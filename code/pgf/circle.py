from axes import Axes
from utils import indent

class Circle(object):
	def __init__(self, x, y, r, **kwargs):
		self.x = x
		self.y = y
		self.r = r

		# properties
		self.color = kwargs.get('color', None)
		self.line_style = kwargs.get('line_style', None)
		self.line_width = kwargs.get('line_width', None)

		# custom plot options
		self.pgf_options = kwargs.get('pgf_options', [])

		# catch common mistakes
		if not isinstance(self.pgf_options, list):
			raise TypeError('pgf_options should be a list.')

		# add rectangle to axes
		self.axes = kwargs.get('axes', Axes.gca())
		self.axes.children.append(self)


	def render(self):
		options = []

		if self.line_style:
			options.append(self.line_style)
		if self.line_width is not None:
			options.append('line width={0}'.format(self.line_width))
		if self.color:
			options.append('color={0}'.format(self.color))

		options.extend(self.pgf_options)

		# summarize options into one string
		options_string = ', '.join(options)
		if len(options_string) > 70:
			options_string = '\n' + indent(',\n'.join(options))

		return '\\draw[{0}] (axis cs:{1},{2}) circle[radius={3}];\n'.format(
			options_string, self.x, self.y, self.r)


	def limits(self):
		return [self.x - self.r, self.x + self.r, self.y - self.r, self.y + self.r]
