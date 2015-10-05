import os

class Settings(object):
	# where to store and compile *.tex files
	tmp_dir = '/tmp/'

	# how to compile LaTeX code into PDFs
	pdf_compile = 'pdflatex -halt-on-error -interaction batchmode {0} > /dev/null'

	# how to open and show PDFs
	pdf_view_mac = 'open {0}'
	pdf_view_linux = 'evince --preview {0} &'
	pdf_view = pdf_view_mac if os.uname()[0] in ['Darwin'] \
		else pdf_view_linux

	# where and how images used in the figures will be stored
	image_folder = 'images'
	image_format = 'PNG'

	# will be included in header of LaTeX file
	preamble = \
		'\\usepackage[utf8]{inputenc}\n' + \
		'\\usepackage{amsmath}\n' + \
		'\\usepackage{amssymb}\n' + \
		'\\usepackage{pgfplots}\n' + \
		'\\usepgflibrary{arrows}\n' + \
		'\\usetikzlibrary{patterns}\n'
