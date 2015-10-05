from axes import Axes

class Text(object):
	def __init__(self, x, y, text, **kwargs):
		self.x = x
		self.y = y
		self.text = text

		# properties
		self.color = kwargs.get('color', None)

		# custom plot options
		self.pgf_options = kwargs.get('pgf_options', [])

		# catch common mistakes
		if not isinstance(self.pgf_options, list):
			raise TypeError('pgf_options should be a list.')

		# add text to axes
		self.axes = kwargs.get('axes', Axes.gca())
		self.axes.children.append(self)


	def render(self):
		options = []

		if self.color:
			options.append('color={0}'.format(self.color))

		options.extend(self.pgf_options)

		# summarize options into one string
		options_string = ', '.join(options)
		if len(options_string) > 70:
			options_string = '\n' + indent(',\n'.join(options))

		sans_serif = '\\sffamily ' if self.axes.figure.sans_serif else ''

		if options_string:
			return '\\node[{0}] at (axis cs:{1},{2}) {{{4}{3}}};\n'.format(
				options_string, self.x, self.y, self.text, sans_serif)
		else:
			return '\\node at (axis cs:{0},{1}) {{{3}{2}}};\n'.format(
				self.x, self.y, self.text, sans_serif)


	def limits(self):
		return [self.x, self.x, self.y, self.y]
