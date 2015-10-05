from axes import Axes
from plot import Plot
from numpy import asarray, arange, shape, percentile, min, max, logical_or, any
from numpy import where

class BoxPlot(object):
	def __init__(self, *args, **kwargs):
		# data points
		if len(args) < 1:
			self.xvalues = asarray([])
			self.yvalues = asarray([])
		elif len(args) < 2:
			self.yvalues = asarray(args[0]).reshape(shape(args[0])[0], -1)
			self.xvalues = arange(1, self.yvalues.shape[-1] + 1)
		else:
			self.xvalues = asarray(args[0]).flatten()
			self.yvalues = asarray(args[1]).reshape(shape(args[1])[0], -1)

		self.box_width = kwargs.get('box_width', 0.5);

		# custom plot options
		self.pgf_options = kwargs.get('pgf_options', [])

		# catch common mistakes
		if not isinstance(self.pgf_options, list):
			raise TypeError('pgf_options should be a list.')

		# add plot to axes
		self.axes = kwargs.get('axes', Axes.gca())
		self.axes.children.append(self)


	def render(self):
		"""
		Produces LaTeX code for this boxplot.

		@rtype: string
		@return: LaTeX code for this plot
		"""

		options = []
		marker_options = []

		tex = ''

		for k in range(len(self.xvalues)):
			qu1 = percentile(self.yvalues[:, k], 25)
			med = percentile(self.yvalues[:, k], 50)
			qu2 = percentile(self.yvalues[:, k], 75)
			iqr = qu2 - qu1

			outlier = logical_or(
				self.yvalues[:, k] > qu2 + 1.5 * iqr,
				self.yvalues[:, k] < qu1 - 1.5 * iqr)

			# median
			tex += '\\draw[red] (axis cs:{0},{1}) -- (axis cs:{2},{3});\n'.format(
				self.xvalues[k] - self.box_width / 2., med,
				self.xvalues[k] + self.box_width / 2., med)

			# box
			tex += '\\draw[blue] (axis cs:{0},{1}) rectangle (axis cs:{2},{3});\n'.format(
				self.xvalues[k] - self.box_width / 2., qu1,
				self.xvalues[k] + self.box_width / 2., qu2)

			# whiskers
			tex += '\\draw[|-, densely dashed] (axis cs:{0},{1}) -- (axis cs:{2},{3});\n'.format(
				self.xvalues[k], min(self.yvalues[-outlier, k]),
				self.xvalues[k], qu1)
			tex += '\\draw[-|, densely dashed] (axis cs:{0},{1}) -- (axis cs:{2},{3});\n'.format(
				self.xvalues[k], qu2,
				self.xvalues[k], max(self.yvalues[-outlier, k]))

			if any(outlier):
				tex += '\\addplot[red, mark=+, only marks] coordinates {\n'
				for i in where(outlier)[0]:
					tex += '\t({0}, {1})\n'.format(self.xvalues[k], self.yvalues[i, k])
				tex += '};\n'

		return tex


	def limits(self):
		return [
			min(self.xvalues) - self.box_width,
			max(self.xvalues) + self.box_width,
			min(self.yvalues) - self.box_width,
			max(self.yvalues) + self.box_width]
