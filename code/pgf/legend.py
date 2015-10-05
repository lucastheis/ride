from utils import indent, escape
from axes import Axes
from numpy import round

class Legend(object):
	def __init__(self, *args, **kwargs):
		# legend entries
		self.legend_entries = args

		# legend properties
		self.at = kwargs.get('at', None)
		self.anchor = kwargs.get('anchor', None)
		self.location = kwargs.get('location', None)
		self.align = kwargs.get('align', 'left')
		self.box = kwargs.get('box', True)
		self.font_size = kwargs.get('font_size', None)

		# assign legend to axis
		self.axes = kwargs.get('axes', Axes.gca())
		self.axes.legend = self



	def render(self):
		options = [
			'legend entries={{{0}}}'.format(','.join(escape(self.legend_entries))),
			'legend cell align={0}'.format(self.align)]
		cell_options = []

		if self.location:
			anchor = []
			left, bottom = 0.5, 0.5
			loc = self.location.lower()

			# position legend according to location hint
			if 'north' in loc:
				bottom = 1. - 0.1 / self.axes.height
				anchor.append('north')

			elif 'south' in loc:
				bottom = 0. + 0.1 / self.axes.height
				anchor.append('south')

			if 'outer' in loc and ('north' in loc or 'south' in loc):
				bottom = round(bottom)

			if 'west' in loc:
				if 'outer' in self.location.lower():
					left = 0. - 0.1 / self.axes.width
					anchor.append('east')
				else:
					left = 0. + 0.1 / self.axes.width
					anchor.append('west')

			elif 'east' in loc:
				if 'outer' in loc:
					left = 1. + 0.1 / self.axes.width
					anchor.append('west')
				else:
					left = 1. - 0.1 / self.axes.width
					anchor.append('east')

			left, bottom = round(left, 4), round(bottom, 4)

			sans_serif = '\\sansmath\\sffamily' if self.axes.figure.sans_serif else ''

			cell_options.append('at={{({0},{1})}}'.format(left, bottom))
			cell_options.append('anchor={0}'.format(' '.join(anchor)))
			if self.font_size is not None:
				cell_options.append('font=\\fontsize{{{0:d}pt}}{{{0:d}pt}}{1}\\selectfont'.format(
					self.font_size, sans_serif))
			elif sans_serif:
				cell_options.append('font={0}\\selectfont'.format(sans_serif))

		else:
			if self.at:
				cell_options.append('at={{({0},{1})}}'.format(*self.at))
			if self.anchor:
				cell_options.append('anchor={0}'.format(self.anchor))

		if self.box is False or self.box == 'off':
			cell_options.append('draw=none')

		if cell_options:
			options.append(
				'legend style={\n' + \
				indent(',\n'.join(cell_options)) + '\n}')

		return ',\n'.join(options)
