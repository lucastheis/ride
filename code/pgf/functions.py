from axes import Axes
from axesgrid import AxesGrid
from cyclelist import CycleList, cycle_lists
from figure import Figure
from settings import Settings
from legend import Legend
from plot import Plot
from boxplot import BoxPlot
from surfplot import SurfPlot
from arrow import Arrow
from text import Text
from rectangle import Rectangle
from circle import Circle
from numpy import asmatrix, inf, min, copy, arange, repeat, isscalar, sum, ndarray
from numpy import histogram, append, ceil
from image import Image

def gcf():
	"""
	Returns currently active figure.
	"""
	return Figure.gcf()


def gca():
	"""
	Returns currently active axis.
	"""

	return Axes.gca()


def draw():
	"""
	Draws the currently active figure.
	"""

	gcf().draw()


def figure(idx=None, *args, **kwargs):
	"""
	Creates a new figure or moves the focus to an existing figure.

	@type  idx: integer
	@param idx: a number identifying a figure

	@rtype: Figure
	@return: currently active figure
	"""

	return Figure(idx, *args, **kwargs)


def plot(*args, **kwargs):
	"""
	Plot lines or markers.

	B{Examples:}

		>>> plot(y)           # plot y using values 1 to len(y) for x
		>>> plot(x, y)        # plot x and y using default line style and color
		>>> plot(x, y, 'r.')  # plot red markers at positions x and y
	"""

	# split formatting information from data points
	format_string = ''.join([arg for arg in args if isinstance(arg, str)])
	args = [asmatrix(arg) for arg in args if not isinstance(arg, str)]

	if not len(args):
		# no data is given, don't create a plot
		return None

	# parse format string into keyword arguments
	if 'color' not in kwargs:
		if 'r' in format_string:
			kwargs['color'] = 'red'
		elif 'g' in format_string:
			kwargs['color'] = 'green'
		elif 'b' in format_string:
			kwargs['color'] = 'blue'
		elif 'c' in format_string:
			kwargs['color'] = 'cyan'
		elif 'm' in format_string:
			kwargs['color'] = 'magenta'
		elif 'y' in format_string:
			kwargs['color'] = 'yellow'
		elif 'k' in format_string:
			kwargs['color'] = 'black'
		elif 'w' in format_string:
			kwargs['color'] = 'white'

	if 'marker' not in kwargs:
		if '.' in format_string:
			kwargs['marker'] = '*'
		elif 'o' in format_string:
			kwargs['marker'] = 'o'
		elif '+' in format_string:
			kwargs['marker'] = '+'
		elif '|' in format_string:
			kwargs['marker'] = '|'
		elif '*' in format_string:
			kwargs['marker'] = 'asterisk'
		elif 'x' in format_string:
			kwargs['marker'] = 'x'
		elif 'd' in format_string:
			kwargs['marker'] = 'diamond'
		elif '^' in format_string:
			kwargs['marker'] = 'triangle'
		elif 'p' in format_string:
			kwargs['marker'] = 'pentagon'

	if 'line_style' not in kwargs:
		if '---' in format_string:
			kwargs['line_style'] = 'densely dashed'
		elif '--' in format_string:
			kwargs['line_style'] = 'dashed'
		elif '-' in format_string:
			kwargs['line_style'] = 'solid'
		elif ':' in format_string:
			kwargs['line_style'] = 'densely dotted'

	# error bar shorthands
	if 'xerr' in kwargs:
		kwargs['xvalues_error'] = kwargs['xerr']
		kwargs.pop('xerr')
	if 'yerr' in kwargs:
		kwargs['yvalues_error'] = kwargs['yerr']
		kwargs.pop('yerr')

	if 'xvalues_error' in kwargs:
		xvalues_error = copy(kwargs['xvalues_error'])
	if 'yvalues_error' in kwargs:
		yvalues_error = copy(kwargs['yvalues_error'])

	if len(args) > 1:
		# heuristics to use if arguments differ in size
		if args[0].shape[0] < args[1].shape[0]:
			args[0] = repeat(args[0], ceil(args[1].shape[0] / float(args[0].shape[0])), 0)
		if args[0].shape[0] > args[1].shape[0]:
			args[1] = repeat(args[1], ceil(args[0].shape[0] / float(args[1].shape[0])), 0)

		if args[0].shape[1] == 1:
			args[0] = repeat(args[0], args[1].shape[1], 1)
		if args[1].shape[1] == 1:
			args[1] = repeat(args[1], args[0].shape[1], 1)

	# if arguments contain multiple rows, create multiple plots
	if len(args) and args[0].shape[0] > 1:
		plots = []

		for i in range(len(args[0])):
			if 'yvalues_error' in kwargs and yvalues_error.shape[0] > 1:
				kwargs['yvalues_error'] = yvalues_error[i]
			if 'xvalues_error' in kwargs and xvalues_error.shape[0] > 1:
				kwargs['xvalues_error'] = xvalues_error[i]
			plots.append(plot(*[arg[i] for arg in args], **kwargs))

		return plots

	return Plot(*args, **kwargs)



def stem(*args, **kwargs):
	"""
	Plot data points as stems from the x-axis. Takes the same arguments as the
	L{plot} function.

	B{Examples:}

		>>> stem(y)
		>>> stem(x, y)
		>>> stem(x, y, 'r', marker='none')
	"""

	kwargs['ycomb'] = True

	if 'marker' not in kwargs:
		kwargs['marker'] = 'o'

	return plot(*args, **kwargs)


def semilogx(*args, **kwargs):
	gca().axes_type = 'semilogxaxis'
	return plot(*args, **kwargs)



def semilogy(*args, **kwargs):
	gca().axes_type = 'semilogyaxis'
	return plot(*args, **kwargs)


def loglog(*args, **kwargs):
	gca().axes_type = 'loglogaxis'
	return plot(*args, **kwargs)


def bar(*args, **kwargs):
	gca().ybar = True

	if 'bar_width' in kwargs:
		gca().bar_width = kwargs['bar_width']
	if 'stacked' in kwargs and kwargs['stacked']:
		gca().stacked = True

	return plot(*args, **kwargs)


def barh(*args, **kwargs):
	# split formatting information from data points
	format_string = ''.join([arg for arg in args if isinstance(arg, str)])
	args = [asmatrix(arg) for arg in args if not isinstance(arg, str)]

	gca().xbar = True
	gca().stacked = kwargs.get('stacked', None)

	if len(args) == 1:
		args = [args[0], arange(1, args[0].shape[1] + 1)]

	return plot(format_string, *args, **kwargs)


def hist(values, bins=10, format_string='', range=None, normed=False, density=False, **kwargs):
	"""
	Computes and plots a histogram of the data provided.

	B{Examples:}
		>>> hist(x, 20, 'k')

	@type  values: array_like
	@param values: values from which to compute a histogram

	@type  bins: int
	@param bins: number of bins

	@type  range: tuple
	@param range: lower and upper range of bins

	@type  normed: boolean
	@param normed: if true, the histogram is normalized to sum to 1

	@type  density: boolean
	@param density: if true, the histogram is normalized to yield a density

	@type  log: boolean
	@param log: if true, the log-histogram is plotted instead of the histogram

	@rtype: L{Plot}
	@return: a reference to the plot
	"""

	# correct arguments if necessary
	if isinstance(bins, str):
		if not format_string:
			format_string = bins
		bins = 10

	if isinstance(format_string, tuple) or \
	   isinstance(format_string, list) or \
	   isinstance(format_string, ndarray):
		if range is None:
			range = format_string
		format_string = ''

	try:
		hist, bin_edges = histogram(values, bins, range, density=density)
	except:
		# use deprecated keyword with older versions of NumPy
		hist, bin_edges = histogram(values, bins, range, normed=density)

	if normed:
		hist = hist / sum(hist, dtype=float)

	hist = append(hist, hist[-1])

	kwargs['const_plot'] = True

	if not 'closed' in kwargs:
		kwargs['closed'] = True

	if not 'fill' in kwargs:
		kwargs['fill'] = True

	return plot(bin_edges, hist, format_string, **kwargs)


def errorbar(*args, **kwargs):
	"""
	Plot lines or markers with error bars.

	B{Examples:}

		>>> errorbar(y, y_err)
		>>> errorbar(y, y_err, 'r.')
		>>> errorbar(x, y, y_err)
		>>> errorbar(x, y, x_err, y_err)
	"""

	# split formatting information from data points
	format_string = ''.join([arg for arg in args if isinstance(arg, str)])
	args = [asmatrix(arg) for arg in args if not isinstance(arg, str)]

	if len(args) > 3:
		kwargs['xvalues_error'] = args[-2]
		kwargs['yvalues_error'] = args[-1]
		args = args[:-2]
	if len(args) > 1:
		kwargs['yvalues_error'] = args[-1]
		args = args[:-1]

	return plot(format_string, *args, **kwargs)


def boxplot(*args, **kwargs):
	return BoxPlot(*args, **kwargs)


def surf(*args, **kwargs):
	return SurfPlot(*args, **kwargs)


def title(title):
	gca().title = title


def xlabel(xlabel):
	gca().xlabel = xlabel


def ylabel(ylabel):
	gca().ylabel = ylabel


def zlabel(zlabel):
	gca().zlabel = zlabel


def xtick(xtick, labels=None, rotation=None):
	gca().xtick = list(xtick)

	if rotation:
		gca().xticklabel_rotation = rotation

	if labels is not None:
		xticklabels(labels, rotation)


def ytick(ytick, labels=None, rotation=None):
	gca().ytick = list(ytick)

	if rotation:
		gca().yticklabel_rotation = rotation

	if labels is not None:
		yticklabels(labels)


def ztick(ztick, labels=None, rotation=None):
	gca().ztick = list(ztick)

	if rotation:
		gca().zticklabel_rotation = rotation

	if labels is not None:
		zticklabels(labels)


def xticklabels(xticklabels, rotation=None):
	if not isinstance(xticklabels[0], str):
		xticklabels = ['{0}'.format(label) for label in xticklabels]

	gca().xticklabels = xticklabels

	if gca().xtick is None:
		xtick(range(1, len(xticklabels) + 1))


def yticklabels(yticklabels, rotation=None):
	if not isinstance(yticklabels[0], str):
		yticklabels = ['{0}'.format(label) for label in yticklabels]

	gca().yticklabels = yticklabels

	if rotation:
		gca().yticklabel_rotation = rotation


def zticklabels(zticklabels, rotation=None):
	if not isinstance(zticklabels[0], str):
		zticklabels = ['{0}'.format(label) for label in zticklabels]

	gca().zticklabels = zticklabels

	if rotation:
		gca().zticklabel_rotation = rotation


def axis(*args, **kwargs):
	if len(args) > 0:
		if isinstance(args[0], str):
			ax = gca()

			if args[0] == 'off':
				ax.hide_axis = True

			elif args[0] == 'on':
				ax.hide_axis = False

			elif args[0] == 'equal':
				ax.equal = True

			elif args[0] == 'square':
				ax.width = ax.height = min([gca().width, gca().height])

			elif args[0] == 'auto':
				ax.xmin = ax.xmax = ax.ymin = ax.ymax = None

			elif args[0] == 'tight':
				if not ax.children:
					return
				ax.xmin, ax.xmax, ax.ymin, ax.ymax = ax.limits()

			elif (args[0] == 'center') or (args[0] == 'origin'):
				ax.axis_x_line = 'center'
				ax.axis_y_line = 'middle'

		elif isinstance(args[0], list) or isinstance(args[0], ndarray):
			if len(args[0]) == 4:
				gca().xmin, gca().xmax, gca().ymin, gca().ymax = args[0]
			elif len(args[0]) == 6:
				gca().xmin, gca().xmax, \
				gca().ymin, gca().ymax, \
				gca().zmin, gca().zmax = args[0]

	for key, value in kwargs.items():
		gca().__dict__[key] = value

	return gca()



def xlim(xmin, xmax):
	gca().xmin = xmin
	gca().xmax = xmax



def ylim(ymin, ymax):
	gca().ymin = ymin
	gca().ymax = ymax



def zlim(zmin, zmax):
	gca().zmin = zmin
	gca().zmax = zmax



def grid(value=None):
	if value == 'off':
		gca().grid = False
	elif value == 'on':
		gca().grid = True
	else:
		gca().grid = not gca().grid


def render():
	return gcf().render()


def legend(*args, **kwargs):
	return Legend(*args, **kwargs)


def savefig(filename, format=None):
	gcf().save(filename, format)


def box(value=None):
	box_on = (gca().axis_x_line is None) and (gca().axis_y_line is None)
	if value == 'off' or (value is None and box_on):
		gca().axis_x_line = 'bottom'
		gca().axis_y_line = 'left'
	elif value == 'on' or (value is None and not box_on):
		gca().axis_x_line = None
		gca().axis_y_line = None


def arrow(x, y, dx, dy, format_string='', **kwargs):
	if 'color' not in kwargs:
		if 'r' in format_string:
			kwargs['color'] = 'red'
		elif 'g' in format_string:
			kwargs['color'] = 'green'
		elif 'b' in format_string:
			kwargs['color'] = 'blue'
		elif 'c' in format_string:
			kwargs['color'] = 'cyan'
		elif 'm' in format_string:
			kwargs['color'] = 'magenta'
		elif 'y' in format_string:
			kwargs['color'] = 'yellow'
		elif 'k' in format_string:
			kwargs['color'] = 'black'
		elif 'w' in format_string:
			kwargs['color'] = 'white'

	if 'line_style' not in kwargs:
		if '---' in format_string:
			kwargs['line_style'] = 'densely dashed'
		elif '--' in format_string:
			kwargs['line_style'] = 'dashed'
		elif '-' in format_string:
			kwargs['line_style'] = 'solid'
		elif ':' in format_string:
			kwargs['line_style'] = 'densely dotted'

	return Arrow(x, y, dx, dy, **kwargs)



def rectangle(x, y, dx, dy, format_string='', **kwargs):
	if 'color' not in kwargs:
		if 'r' in format_string:
			kwargs['color'] = 'red'
		elif 'g' in format_string:
			kwargs['color'] = 'green'
		elif 'b' in format_string:
			kwargs['color'] = 'blue'
		elif 'c' in format_string:
			kwargs['color'] = 'cyan'
		elif 'm' in format_string:
			kwargs['color'] = 'magenta'
		elif 'y' in format_string:
			kwargs['color'] = 'yellow'
		elif 'k' in format_string:
			kwargs['color'] = 'black'
		elif 'w' in format_string:
			kwargs['color'] = 'white'

	if 'line_style' not in kwargs:
		if '---' in format_string:
			kwargs['line_style'] = 'densely dashed'
		elif '--' in format_string:
			kwargs['line_style'] = 'dashed'
		elif '-' in format_string:
			kwargs['line_style'] = 'solid'
		elif ':' in format_string:
			kwargs['line_style'] = 'densely dotted'

	return Rectangle(x, y, dx, dy, **kwargs)



def circle(x, y, r, format_string='', **kwargs):
	if 'color' not in kwargs:
		if 'r' in format_string:
			kwargs['color'] = 'red'
		elif 'g' in format_string:
			kwargs['color'] = 'green'
		elif 'b' in format_string:
			kwargs['color'] = 'blue'
		elif 'c' in format_string:
			kwargs['color'] = 'cyan'
		elif 'm' in format_string:
			kwargs['color'] = 'magenta'
		elif 'y' in format_string:
			kwargs['color'] = 'yellow'
		elif 'k' in format_string:
			kwargs['color'] = 'black'
		elif 'w' in format_string:
			kwargs['color'] = 'white'

	if 'line_style' not in kwargs:
		if '---' in format_string:
			kwargs['line_style'] = 'densely dashed'
		elif '--' in format_string:
			kwargs['line_style'] = 'dashed'
		elif '-' in format_string:
			kwargs['line_style'] = 'solid'
		elif ':' in format_string:
			kwargs['line_style'] = 'densely dotted'

	return Circle(x, y, r, **kwargs)


def text(x, y, text, **kwargs):
	return Text(x, y, text, **kwargs)


def colormap(colormap):
	gca().colormap = colormap


def colorbar(colorbar=None):
	if colorbar is None:
		# toggle color bar
		if gca().colorbar:
			gca().colorbar = False
		else:
			gca().colorbar = True
		return

	gca().colorbar = colorbar


def cyclelist(cyclelist):
	if isinstance(cyclelist, str):
		if cyclelist in cycle_lists:
			# use predefined cycle list
			gca().cycle_list = cycle_lists[cyclelist]
		else:
			# use cycle list predefined by PGFPlots
			gca().cycle_list_name = cyclelist
	else:
		if isinstance(cyclelist, CycleList):
			# use given cycle list
			gca().cycle_list = cyclelist
		else:
			# create cycle list from given styles
			gca().cycle_list = CycleList(cyclelist)


def subplot(i, j, **kwargs):
	fig = gcf()

	if not (fig.axes and isinstance(fig.axes[0], AxesGrid)):
		# create new axis grid
		fig.axes = [AxesGrid(**kwargs)]

	# get axis grid
	grid = fig.axes[0]

	if not (i, j) in grid.keys():
		# create new axis in axis grid
		grid[i, j] = Axes(fig=fig, **kwargs)

	# make axis active; TODO: find a less hacky solution
	fig._ca = grid[i, j]


def imshow(image, **kwargs):
	img = Image(image, **kwargs)
	dpi = kwargs.get('dpi', 150.)
	gca().width = 2.54 / dpi * img.width()
	gca().height = 2.54 / dpi * img.height()
	gca().xtick_align = 'outside'
	gca().ytick_align = 'outside'
	axis('tight')
	return img
