from axes import Axes
from utils import indent

class Arrow(object):
	def __init__(self, x, y, dx, dy, **kwargs):
		self.x = x
		self.y = y
		self.dx = dx
		self.dy = dy

		# properties
		self.color = kwargs.get('color', None)
		self.arrow_style = kwargs.get('arrow_style', '-latex')
		self.line_style = kwargs.get('line_style', None)
		self.line_width = kwargs.get('line_width', None)

		# custom plot options
		self.pgf_options = kwargs.get('pgf_options', [])

		# catch common mistakes
		if not isinstance(self.pgf_options, list):
			raise TypeError('pgf_options should be a list.')

		# add arrow to axes
		self.axes = kwargs.get('axis', Axes.gca())
		self.axes.children.append(self)


	def render(self):
		options = [self.arrow_style]

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

		return '\\draw[{0}] (axis cs:{1},{2}) -- (axis cs:{3},{4});\n'.format(
			options_string, self.x, self.y, self.x + self.dx, self.y + self.dy)


	def limits(self):
		return [self.x, self.x + self.dx, self.y, self.y + self.dy]
