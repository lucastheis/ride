from os import path, system, mkdir
from utils import min_free, indent
from settings import Settings
from numpy.random import randint

class Figure(object):
	"""
	Represents and renders figures.

	@type margin: float
	@ivar margin: space around axis

	@type sans_serif: boolean
	@ivar sans_serif: if true, use Helvetica instead of serif Computer Modern
	"""

	# references to all figures
	_figures = {}

	# currently active figure
	_cf = None

	# identifier for above set of figures
	_session = randint(1E8)

	@staticmethod
	def gcf():
		"""
		Returns currently active figure.

		@rtype: Figure
		@return: the currently active figure
		"""

		if not Figure._cf:
			Figure()
		return Figure._cf


	def __new__(cls, idx=None, **kwargs):
		if idx in Figure._figures:
			# move focus
			Figure._cf = Figure._figures[idx]

			# figure with specified ID already exists
			return Figure._cf
		else:
			# create new figure
			fig = object.__new__(cls, **kwargs)
			fig._idx = min_free(Figure._figures.keys())

			# store figure reference and move focus
			Figure._figures[fig._idx] = fig
			Figure._cf = fig

			return fig


	def __init__(self, *args, **kwargs):
		"""
		Initializes figure properties.
		"""

		if '_initialized' not in self.__dict__:
			self.axes = []

			# width and height of the figure
			self.width = None
			self.height = None

			# space around axes
			self.margin = kwargs.get('margin', 2.)

			# whether to use Helvetica or Computer Modern
			self.sans_serif = kwargs.get('sans_serif', False)

			# currently active axes
			self._ca = None

			# make sure figure will not be initialized twice
			self._initialized = True


	def render(self):
		"""
		Creates and returns LaTeX code for this figure.

		@rtype: string
		@return: LaTeX code for this figure
		"""

		# figure width and height
		width, height = self.width, self.height

		if self.axes:
			if not width:
				width = self.margin * 2. \
					+ max(ax.at[0] + ax.width for ax in self.axes)
			if not height:
				height = self.margin * 2. \
					+ max(ax.at[1] + ax.height for ax in self.axes)
		else:
			if not width:
				width = self.margin * 2. + 1.
			if not height:
				height = self.margin * 2. + 1.

		preamble = Settings.preamble

		if self.sans_serif:
		   preamble = preamble + \
			'\\usepackage[T1]{fontenc}\n' + \
			'\\usepackage{helvet}\n' + \
			'\\renewcommand{\\familydefault}{\\sfdefault}\n' + \
			'\\usepackage{sansmath}\n' + \
			'\\usepackage{sfmath}\n'

		tex = \
			'\\documentclass{article}\n' + \
			'\n' + \
			preamble + \
			'\n' + \
			'\\usepackage[\n' + \
			'\tmargin=0cm,\n' + \
			'\tpaperwidth={0}cm,\n'.format(width) + \
			'\tpaperheight={0}cm]{{geometry}}\n'.format(height) + \
			'\n' + \
			'\\begin{document}\n' + \
			'\t\\thispagestyle{empty}\n' + \
			'\n'
		if self.axes:
			tex += \
				'\t\\begin{figure}\n' + \
				'\t\t\\centering\n' + \
				'\t\t\\begin{tikzpicture}\n'
			for ax in self.axes:
				tex += indent(ax.render(), 3)
			tex += \
				'\t\t\\end{tikzpicture}\n' + \
				'\t\\end{figure}\n'
		else:
			tex += '\t\\mbox{}\n'
		tex += '\\end{document}'

		return tex


	def compile(self):
		"""
		Generates LaTeX code and tries to compile it into a PDF file.

		@rtype: string
		@return: path to PDF file
		"""

		self.save_images(Settings.tmp_dir)

		tex_file = path.join(Settings.tmp_dir, 'pgf_{0}_{1}.tex'.format(Figure._session, self._idx))
		pdf_file = path.join(Settings.tmp_dir, 'pgf_{0}_{1}.pdf'.format(Figure._session, self._idx))

		command = Settings.pdf_compile.format('-output-directory {0} {1}')
		command = command.format(Settings.tmp_dir, tex_file)

		# write LaTeX file
		with open(tex_file, 'w') as handle:
			handle.write(self.render())

		# compile
		if system('cd "{0}" && {1}'.format(Settings.tmp_dir, command)):
			raise RuntimeError('Compiling TeX source file to PDF failed.')

		return pdf_file


	def draw(self):
		"""
		Compiles LaTeX code and tries to open the resulting PDF file.
		"""

		if system(Settings.pdf_view.format(self.compile())):
			raise RuntimeError('Could not open PDF file.')


	def save(self, filename, format=None):
		"""
		Saves figure to specified file. If no file format is given, the
		file format is guessed based on the filename extension.

		@type  filename: string
		@param filename: file location

		@type  format: string/None
		@param format: currently either 'pdf' or 'tex'
		"""

		# figure out which file format to use
		if format is None:
			format = path.splitext(filename)[1][1:]
		format = format.lower()

		if format not in ['pdf', 'tex']:
			raise ValueError('Unknown format \'{0}\'.'.format(format))

		if format == 'pdf':
			# save PDF file
			system('cp {0} {1}'.format(self.compile(), filename))

		elif format == 'tex':
			self.save_images(path.dirname(filename))

			# save TeX file
			with open(filename, 'w') as handle:
				handle.write(self.render())



	def save_images(self, filepath):
		# make sure directory for images exists
		filepath = path.join(filepath, Settings.image_folder)
		if not path.exists(filepath):
			mkdir(filepath)

		# save figures
		from image import Image
		from axesgrid import AxesGrid
		from axes import Axes
		for ax in self.axes:
			if isinstance(ax, AxesGrid):
				for _, axs in ax.grid.iteritems():
					for child in axs.children:
						if isinstance(child, Image):
							child.save(filepath)

			elif isinstance(ax, Axes):
				for child in ax.children:
					if isinstance(child, Image):
						child.save(filepath)
