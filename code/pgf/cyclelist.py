from rgb import RGB

class CycleList(object):
	def __init__(self, styles=None):
		self.entries = []

		for style in styles:
			if isinstance(style, dict):
				self.append(**style)
			elif isinstance(style, list):
				self.append(*style)
			else:
				self.append(style)


	def __getitem__(self, key):
		return self.entries[key]


	def __getslice__(self, start, end):
		return self.entries[start:end]


	def __delitem__(self, key):
		del self.entries[key]

	
	def __delslice__(self, start, end):
		del self.entries[start:end]


	def append(self, *args, **kwargs):
		"""
		B{Examples:}

			>>> cl.append('red')
			>>> cl.append(RGB(0.5, 0.2, 1.0))
			>>> cl.append(color='black', fill='red')
			>>> cl.append('blue', 'dotted')
		"""

		if not args and not kwargs:
			raise TypeError('append() takes at least 1 argument')

		self.entries.append((list(args), kwargs))


	def render(self):
		def parse(arg):
			# makes sure that RGB values are represented correctly
			if isinstance(arg, RGB):
				return 'color=' + str(arg)
			return str(arg)

		# string representations of entries
		entries = []

		# generate string representations
		for entry in self.entries:
			for key, value in entry[1].items():
				entry[0].append('{0}={1}'.format(key, value))
			entries.append('\t{' + ', '.join(parse(arg) for arg in entry[0]) + '}')

		return 'cycle list={\n' + ',\n'.join(entries) + '}'


# predefined cycle lists
cycle_lists = {
	'fancy': CycleList([
		{'color': RGB( 10,  80, 230)},
		{'color': RGB(255,  83, 204)},
		{'color': RGB(170, 250, 120)},
		{'color': RGB(  0,   0,   0)},
		{'color': RGB(255, 200,   0)},
		]),
	'fruity': CycleList([
		{'color': 'orange!10!blue!80!black'},
		{'color': 'orange!5!red!90!black'},
		{'color': 'orange!90!yellow!85!black'},
		{'color': 'orange!30!green!90!black'},
		{'color': 'black'},
		]),
	'icy': CycleList([
		{'color': 'green!20!blue!30!white'},
		{'color': 'green!20!blue!30!white!80!black'},
		{'color': 'green!20!blue!30!white!60!black'},
		{'color': 'green!20!blue!30!white!40!black'},
		{'color': 'green!20!blue!30!white!20!black'},
		{'color': 'black'},
		]),
}
