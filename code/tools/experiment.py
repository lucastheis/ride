#!/usr/bin/env python

"""
Manage and display experimental results.
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@theis.io>'
__docformat__ = 'epytext'
__version__ = '0.4.4'

import sys
import os
import numpy
import scipy
import socket

sys.path.append('./code')

from argparse import ArgumentParser
from pickle import Unpickler, dump
from subprocess import Popen, PIPE
from os import path
from warnings import warn
from time import time, strftime, localtime
from numpy import random, ceil, argsort
from numpy.random import rand, randint
from distutils.version import StrictVersion
from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler
from httplib import HTTPConnection
from getopt import getopt

class FileCache(type):
	"""
	This meta class is used to cache object creation.
	"""

	def __init__(cls, name, bases, dict):
		super(FileCache, cls).__init__(name, bases, dict)

		cls._CACHE = {}
		cls._CACHE_ENABLED = True



	def __call__(cls, filename='', *args, **kwargs):
		if not cls._CACHE_ENABLED:
			return super(FileCache, cls).__call__(filename, *args, **kwargs)

		if filename in cls._CACHE:
			# make sure file has not been modified
			mtime = os.path.getmtime(filename)
			if mtime == cls._CACHE[filename][0]:
				# return from cache
				return cls._CACHE[filename][1]

		instance = super(FileCache, cls).__call__(filename, *args, **kwargs)

		if os.path.exists(filename):
			# cache instance
			mtime = os.path.getmtime(filename)
			cls._CACHE[filename] = (mtime, instance)

		return instance



class Experiment(object):
	"""
	@type time: float
	@ivar time: time at initialization of experiment

	@type duration: float
	@ivar duration: time in seconds between initialization and saving

	@type script: string
	@ivar script: stores the content of the main Python script

	@type platform: string
	@ivar platform: information about operating system

	@type processors: string
	@ivar processors: some information about the processors

	@type environ: string
	@ivar environ: environment variables at point of initialization

	@type hostname: string
	@ivar hostname: hostname of server running the experiment

	@type cwd: string
	@ivar cwd: working directory at execution time

	@type comment: string
	@ivar comment: a comment describing the experiment

	@type results: dictionary
	@ivar results: container to store experimental results

	@type commit: string
	@ivar commit: git commit hash

	@type modified: boolean
	@ivar modified: indicates uncommited changes

	@type filename: string
	@ivar filename: path to stored results

	@type seed: int
	@ivar seed: random seed used through the experiment

	@type versions: dictionary
	@ivar versions: versions of Python, numpy and scipy
	"""

	__metaclass__ = FileCache

	def __init__(self, filename='', comment='', seed=None, server=None, port=8000):
		"""
		If the filename is given and points to an existing experiment, load it.
		Otherwise store the current timestamp and try to get commit information
		from the repository in the current directory.

		@type  filename: string
		@param filename: path to where the experiment will be stored
		
		@type comment: string
		@param comment: a comment describing the experiment

		@type  seed: integer
		@param seed: random seed used in the experiment
		"""

		self.id = 0
		self.time = time()
		self.comment = comment
		self.filename = filename
		self.results = {}
		self.seed = seed
		self.script = ''
		self.cwd = ''
		self.platform = ''
		self.processors = ''
		self.environ = ''
		self.duration = 0
		self.versions = {}
		self.server = ''

		if self.seed is None:
			self.seed = int((time() + 1e6 * rand()) * 1e3) % 4294967295

		# set random seed
		random.seed(self.seed)
		numpy.random.seed(self.seed)

		if self.filename:
			# load given experiment
			self.load()

		else:
			# identifies the experiment
			self.id = randint(1E8)

			# check if a comment was passed via the command line
			parser = ArgumentParser(add_help=False)
			parser.add_argument('--comment')
			optlist, argv = parser.parse_known_args(sys.argv[1:])
			optlist = vars(optlist)

			# remove comment command line argument from argument list
			sys.argv[1:] = argv

			# comment given as command line argument
			self.comment = optlist.get('comment', '')

			# get OS information
			self.platform = sys.platform

			# arguments to the program
			self.argv = sys.argv
			self.script_path = sys.argv[0]

			try:
				with open(sys.argv[0]) as handle:
					# store python script
					self.script = handle.read()
			except:
				warn('Unable to read Python script.')

			# environment variables
			self.environ = os.environ
			self.cwd = os.getcwd()
			self.hostname = socket.gethostname()

			# store some information about the processor(s)
			if self.platform == 'linux2':
				cmd = 'egrep "processor|model name|cpu MHz|cache size" /proc/cpuinfo'
				with os.popen(cmd) as handle:
					self.processors = handle.read()
			elif self.platform == 'darwin':
				cmd = 'system_profiler SPHardwareDataType | egrep "Processor|Cores|L2|Bus"'
				with os.popen(cmd) as handle:
					self.processors = handle.read()

			# version information
			self.versions['python'] = sys.version
			self.versions['numpy'] = numpy.__version__
			self.versions['scipy'] = scipy.__version__

			# store information about git repository
			if path.isdir('.git'):
				# get commit hash
				pr1 = Popen(['git', 'log', '-1'], stdout=PIPE)
				pr2 = Popen(['head', '-1'], stdin=pr1.stdout, stdout=PIPE)
				pr3 = Popen(['cut', '-d', ' ', '-f', '2'], stdin=pr2.stdout, stdout=PIPE)
				self.commit = pr3.communicate()[0][:-1]

				# check if project contains uncommitted changes
				pr1 = Popen(['git', 'status', '--porcelain'], stdout=PIPE)
				pr2 = Popen(['egrep', '^.M'], stdin=pr1.stdout, stdout=PIPE)
				self.modified = pr2.communicate()[0]

				if self.modified:
					warn('Uncommitted changes.')
			else:
				# no git repository
				self.commit = None
				self.modified = False

			# server managing experiments 
			self.server = server
			self.port = port
			self.status('running')



	def __del__(self):
		self.status(None)



	def __str__(self):
		"""
		Summarize information about the experiment.

		@rtype: string
		@return: summary of the experiment
		"""

		strl = []

		# date and duration of experiment
		strl.append(strftime('date \t\t %a, %d %b %Y %H:%M:%S', localtime(self.time)))
		strl.append('duration \t ' + str(int(self.duration)) + 's')
		strl.append('hostname \t ' + self.hostname)

		# commit hash
		if self.commit:
			if self.modified:
				strl.append('commit \t\t ' + self.commit + ' (modified)')
			else:
				strl.append('commit \t\t ' + self.commit)

		# results
		strl.append('results \t {' + ', '.join(map(str, self.results.keys())) + '}')

		# comment
		if self.comment:
			strl.append('\n' + self.comment)

		return '\n'.join(strl)



	def __getitem__(self, key):
		return self.results[key]



	def __setitem__(self, key, value):
		self.results[key] = value



	def __delitem__(self, key):
		del self.results[key]



	def keys(self):
		return self.results.keys()



	def status(self, status, **kwargs):
		if self.server:
			try:
				conn = HTTPConnection(self.server, self.port)
				conn.request('GET', '/version/')
				resp = conn.getresponse()

				if not resp.read().startswith('Experiment'):
					raise RuntimeError()

				HTTPConnection(self.server, self.port).request('POST', '', str(dict({
						'id': self.id,
						'version': __version__,
						'status': status,
						'hostname': self.hostname,
						'cwd': self.cwd,
						'script_path': self.script_path,
						'script': self.script,
						'comment': self.comment,
						'time': self.time,
					}, **kwargs)))
			except:
				warn('Unable to connect to \'{0}:{1}\'.'.format(self.server, self.port))



	def progress(self, progress):
		self.status('PROGRESS', progress=progress)



	def save(self, filename=None, overwrite=False):
		"""
		Store results. If a filename is given, the default is overwritten.

		@type  filename: string
		@param filename: path to where the experiment will be stored

		@type  overwrite: boolean
		@param overwrite: overwrite existing files
		"""

		self.duration = time() - self.time

		if filename is None:
			filename = self.filename
		else:
			# replace {0} and {1} by date and time
			tmp1 = strftime('%d%m%Y', localtime(time()))
			tmp2 = strftime('%H%M%S', localtime(time()))
			filename = filename.format(tmp1, tmp2)

			self.filename = filename

		# make sure directory exists
		try:
			os.makedirs(path.dirname(filename))
		except OSError:
			pass

		# make sure filename is unique
		counter = 0
		pieces = path.splitext(filename)

		if not overwrite:
			while path.exists(filename):
				counter += 1
				filename = pieces[0] + '.' + str(counter) + pieces[1]

			if counter:
				warn(''.join(pieces) + ' already exists. Saving to ' + filename + '.')

		# store experiment
		with open(filename, 'wb') as handle:
			dump({
				'version': __version__,
				'id': self.id,
				'time': self.time,
				'seed': self.seed,
				'duration': self.duration,
				'environ': self.environ,
				'hostname': self.hostname,
				'cwd': self.cwd,
				'argv': self.argv,
				'script': self.script,
				'script_path': self.script_path,
				'processors': self.processors,
				'platform': self.platform,
				'comment': self.comment,
				'commit': self.commit,
				'modified': self.modified,
				'versions': self.versions,
				'results': self.results}, handle, 1)

		self.status('SAVE', filename=filename, duration=self.duration)



	def load(self, filename=None):
		"""
		Loads experimental results from the specified file.

		@type  filename: string
		@param filename: path to where the experiment is stored
		"""

		if filename:
			self.filename = filename

		with open(self.filename, 'rb') as handle:
			res = load(handle)

			self.time = res['time']
			self.seed = res['seed']
			self.duration = res['duration']
			self.processors = res['processors']
			self.environ = res['environ']
			self.platform = res['platform']
			self.comment = res['comment']
			self.commit = res['commit']
			self.modified = res['modified']
			self.versions = res['versions']
			self.results = res['results']
			self.argv = res['argv'] \
				if StrictVersion(res['version']) >= '0.3.1' else None
			self.script = res['script'] \
				if StrictVersion(res['version']) >= '0.4.0' else None
			self.script_path = res['script_path'] \
				if StrictVersion(res['version']) >= '0.4.0' else None
			self.cwd = res['cwd'] \
				if StrictVersion(res['version']) >= '0.4.0' else None
			self.hostname = res['hostname'] \
				if StrictVersion(res['version']) >= '0.4.0' else None
			self.id = res['id'] \
				if StrictVersion(res['version']) >= '0.4.0' else None



class ExperimentRequestHandler(BaseHTTPRequestHandler):
	"""
	Renders HTML showing running and finished experiments.
	"""

	xpck_path = ''
	running = {}
	finished = {}

	def do_GET(self):
		"""
		Renders HTML displaying running and saved experiments.
		"""

		# number of bars representing progress
		max_bars = 20

		if self.path == '/version/':
			self.send_response(200)
			self.send_header('Content-type', 'text/plain')
			self.end_headers()

			self.wfile.write('Experiment {0}'.format(__version__))

		elif self.path.startswith('/running/'):
			id = int([s for s in self.path.split('/') if s != ''][-1])

			# display running experiment
			if id in ExperimentRequestHandler.running:
				self.send_response(200)
				self.send_header('Content-type', 'text/html')
				self.end_headers()

				self.wfile.write(HTML_HEADER)
				self.wfile.write('<h2>Experiment</h2>')

				instance = ExperimentRequestHandler.running[id]

				num_bars = int(instance['progress']) * max_bars / 100

				self.wfile.write('<table>')
				self.wfile.write('<tr><th>Experiment:</th><td>{0}</td></tr>'.format(
					os.path.join(instance['cwd'], instance['script_path'])))
				self.wfile.write('<tr><th>Hostname:</th><td>{0}</td></tr>'.format(instance['hostname']))
				self.wfile.write('<tr><th>Status:</th><td class="running">{0}</td></tr>'.format(instance['status']))
				self.wfile.write('<tr><th>Progress:</th><td class="progress"><span class="bars">{0}</span>{1}</td></tr>'.format(
					'|' * num_bars, '|' * (max_bars - num_bars)))
				self.wfile.write('<tr><th>Start:</th><td>{0}</td></tr>'.format(
					strftime('%a, %d %b %Y %H:%M:%S', localtime(instance['time']))))
				self.wfile.write('<tr><th>Comment:</th><td>{0}</td></tr>'.format(
					instance['comment']  if instance['comment'] else '-'))
				self.wfile.write('</table>')

				self.wfile.write('<h2>Script</h2>')
				self.wfile.write('<pre>{0}</pre>'.format(instance['script']))
				self.wfile.write(HTML_FOOTER)

			elif id in ExperimentRequestHandler.finished:
				self.send_response(302)
				self.send_header('Location', '/finished/{0}/'.format(id))
				self.end_headers()

			else:
				self.send_response(200)
				self.send_header('Content-type', 'text/html')
				self.end_headers()

				self.wfile.write(HTML_HEADER)
				self.wfile.write('<h2>404</h2>')
				self.wfile.write('Requested experiment not found.')
				self.wfile.write(HTML_FOOTER)

		elif self.path.startswith('/finished/'):
			self.send_response(200)
			self.send_header('Content-type', 'text/html')
			self.end_headers()

			self.wfile.write(HTML_HEADER)

			id = int([s for s in self.path.split('/') if s != ''][-1])

			# display finished experiment
			if id in ExperimentRequestHandler.finished:
				instance = ExperimentRequestHandler.finished[id]

				if id in ExperimentRequestHandler.running:
					progress = ExperimentRequestHandler.running[id]['progress']
				else:
					progress = 100

				num_bars = int(progress) * max_bars / 100

				self.wfile.write('<h2>Experiment</h2>')
				self.wfile.write('<table>')
				self.wfile.write('<tr><th>Experiment:</th><td>{0}</td></tr>'.format(
					os.path.join(instance['cwd'], instance['script_path'])))
				self.wfile.write('<tr><th>Results:</th><td>{0}</td></tr>'.format(
					os.path.join(instance['cwd'], instance['filename'])))
				self.wfile.write('<tr><th>Status:</th><td class="finished">{0}</td></tr>'.format(instance['status']))
				self.wfile.write('<tr><th>Progress:</th><td class="progress"><span class="bars">{0}</span>{1}</td></tr>'.format(
					'|' * num_bars, '|' * (max_bars - num_bars)))
				self.wfile.write('<tr><th>Start:</th><td>{0}</td></tr>'.format(
					strftime('%a, %d %b %Y %H:%M:%S', localtime(instance['time']))))
				self.wfile.write('<tr><th>End:</th><td>{0}</td></tr>'.format(
					strftime('%a, %d %b %Y %H:%M:%S', localtime(instance['duration']))))
				self.wfile.write('<tr><th>Comment:</th><td>{0}</td></tr>'.format(
					instance['comment']  if instance['comment'] else '-'))
				self.wfile.write('</table>')

				self.wfile.write('<h2>Results</h2>')

				try:
					experiment = Experiment(os.path.join(instance['cwd'], instance['filename']))
				except:
					self.wfile.write('Could not open file.')
				else:
					self.wfile.write('<table>')
					for key, value in experiment.results.items():
						self.wfile.write('<tr><th>{0}</th><td>{1}</td></tr>'.format(key, value))
					self.wfile.write('</table>')

				self.wfile.write('<h2>Script</h2>')
				self.wfile.write('<pre>{0}</pre>'.format(instance['script']))


			else:
				self.wfile.write('<h2>404</h2>')
				self.wfile.write('Requested experiment not found.')

			self.wfile.write(HTML_FOOTER)

		else:
			files = []

			if 'xpck_path' in ExperimentRequestHandler.__dict__:
				if ExperimentRequestHandler.xpck_path != '':
					for path in ExperimentRequestHandler.xpck_path.split(':'):
						files += [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.xpck')]
				
			if 'XPCK_PATH' in os.environ:
				for path in os.environ['XPCK_PATH'].split(':'):
					files += [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.xpck')]

			self.send_response(200)
			self.send_header('Content-type', 'text/html')
			self.end_headers()

			self.wfile.write(HTML_HEADER)
			self.wfile.write('<h2>Running</h2>')

			# display running experiments
			if ExperimentRequestHandler.running:
				self.wfile.write('<table>')
				self.wfile.write('<tr>')
				self.wfile.write('<th>Experiment</th>')
				self.wfile.write('<th>Hostname</th>')
				self.wfile.write('<th>Status</th>')
				self.wfile.write('<th>Progress</th>')
				self.wfile.write('<th>Start</th>')
				self.wfile.write('<th>Comment</th>')
				self.wfile.write('</tr>')

				# sort ids by start time of experiment 
				times = [instance['time'] for instance in ExperimentRequestHandler.running.values()]
				ids = ExperimentRequestHandler.running.keys()
				ids = [ids[i] for i in argsort(times)][::-1]

				for id in ids:
					instance = ExperimentRequestHandler.running[id]
					num_bars = int(instance['progress']) * max_bars / 100

					self.wfile.write('<tr>')
					self.wfile.write('<td class="filepath"><a href="/running/{1}/">{0}</a></td>'.format(
						instance['script_path'], instance['id']))
					self.wfile.write('<td>{0}</td>'.format(instance['hostname']))
					self.wfile.write('<td class="running">{0}</td>'.format(instance['status']))
					self.wfile.write('<td class="progress"><span class="bars">{0}</span>{1}</td>'.format(
						'|' * num_bars, '|' * (max_bars - num_bars)))
					self.wfile.write('<td>{0}</td>'.format(strftime('%a, %d %b %Y %H:%M:%S',
						localtime(instance['time']))))
					self.wfile.write('<td class="comment">{0}</td>'.format(
						instance['comment']  if instance['comment'] else '-'))
					self.wfile.write('</tr>')

				self.wfile.write('</table>')

			else:
				self.wfile.write('No running experiments.')

			self.wfile.write('<h2>Saved</h2>')

			# display saved experiments
			if ExperimentRequestHandler.finished:
				self.wfile.write('<table>')
				self.wfile.write('<tr>')
				self.wfile.write('<th>Results</th>')
				self.wfile.write('<th>Status</th>')
				self.wfile.write('<th>Progress</th>')
				self.wfile.write('<th>Start</th>')
				self.wfile.write('<th>End</th>')
				self.wfile.write('<th>Comment</th>')
				self.wfile.write('</tr>')

				# sort ids by start time of experiment 
				times = [instance['time'] + instance['duration']
					for instance in ExperimentRequestHandler.finished.values()]
				ids = ExperimentRequestHandler.finished.keys()
				ids = [ids[i] for i in argsort(times)][::-1]

				for id in ids:
					instance = ExperimentRequestHandler.finished[id]

					if id in ExperimentRequestHandler.running:
						progress = ExperimentRequestHandler.running[id]['progress']
					else:
						progress = 100

					num_bars = int(progress) * max_bars / 100

					self.wfile.write('<tr>')
					self.wfile.write('<td class="filepath"><a href="/finished/{1}/">{0}</a></td>'.format(
						instance['filename'], instance['id']))
					self.wfile.write('<td class="finished">saved</td>')
					self.wfile.write('<td class="progress"><span class="bars">{0}</span>{1}</td>'.format(
						'|' * num_bars, '|' * (max_bars - num_bars)))
					self.wfile.write('<td>{0}</td>'.format(strftime('%a, %d %b %Y %H:%M:%S',
						localtime(instance['time']))))
					self.wfile.write('<td>{0}</td>'.format(strftime('%a, %d %b %Y %H:%M:%S',
						localtime(instance['time'] + instance['duration']))))
					self.wfile.write('<td class="comment">{0}</td>'.format(
						instance['comment']  if instance['comment'] else '-'))
					self.wfile.write('</tr>')

				self.wfile.write('</table>')

			else:
				self.wfile.write('No saved experiments.')

			self.wfile.write(HTML_FOOTER)



	def do_POST(self):
		instances = ExperimentRequestHandler.running
		instance = eval(self.rfile.read(int(self.headers['Content-Length'])))
		
		if instance['status'] is 'PROGRESS':
			if instance['id'] not in instances:
				instances[instance['id']] = instance
				instances[instance['id']]['status'] = 'running'
			instances[instance['id']]['progress'] = instance['progress']

		elif instance['status'] is 'SAVE':
			ExperimentRequestHandler.finished[instance['id']] = instance
			ExperimentRequestHandler.finished[instance['id']]['status'] = 'saved'

		else:
			if instance['id'] in instances:
				progress = instances[instance['id']]['progress']
			else:
				progress = 0
			instances[instance['id']] = instance
			instances[instance['id']]['progress'] = progress

		if instance['status'] is None:
			try:
				del instances[instance['id']]
			except:
				pass



class XUnpickler(Unpickler):
	"""
	An extension of the Unpickler class which resolves some backwards
	compatibility issues of Numpy.
	"""

	def find_class(self, module, name):
		"""
		Helps Unpickler to find certain Numpy modules.
		"""

		try:
			numpy_version = StrictVersion(numpy.__version__)

			if numpy_version >= '1.5.0':
				if module == 'numpy.core.defmatrix':
					module = 'numpy.matrixlib.defmatrix'

		except ValueError:
			pass

		return Unpickler.find_class(self, module, name)



def load(file):
	return XUnpickler(file).load()
		


def main(argv):
	"""
	Load and display experiment information.
	"""

	if len(argv) < 2:
		print 'Usage:', argv[0], '[--server] [--port=<port>] [--path=<path>] [filename]'
		return 0

	optlist, argv = getopt(argv[1:], '', ['server', 'port=', 'path='])
	optlist = dict(optlist)

	if '--server' in optlist:
		try:
			ExperimentRequestHandler.xpck_path = optlist.get('--path', '')
			port = optlist.get('--port', 8000)

			# start server
			server = HTTPServer(('', port), ExperimentRequestHandler)
			server.serve_forever()

		except KeyboardInterrupt:
			server.socket.close()

		return 0

	# load experiment
	experiment = Experiment(sys.argv[1])

	if len(argv) > 1:
		# print arguments
		for arg in argv[1:]:
			try:
				print experiment[arg]
			except:
				try:
					print experiment[int(arg)]
				except:
					print experiment.__dict__[arg]
		return 0

	# print summary of experiment
	print experiment

	return 0



HTML_HEADER = '''<html>
	<head>
		<title>Experiments</title>
		<style type="text/css">
			body {
				font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
				font-size: 11pt;
				color: black;
				background: white;
				padding: 0pt 20pt;
			}

			h2 {
				margin-top: 20pt;
				font-size: 16pt;
			}

			table {
				border-collapse: collapse;
			}

			tr:nth-child(even) {
				background: #f4f4f4;
			}

			th {
				font-size: 12pt;
				text-align: left;
				padding: 2pt 10pt 3pt 0pt;
			}

			td {
				font-size: 10pt;
				padding: 3pt 10pt 2pt 0pt;
			}

			pre {
				font-size: 10pt;
				background: #f4f4f4;
				padding: 5pt;
			}

			a {
				text-decoration: none;
				color: #04a;
			}

			.running {
				color: #08b;
			}

			.finished {
				color: #390;
			}

			.comment {
				min-width: 200pt;
				font-style: italic;
			}

			.progress {
				color: #ccc;
			}

			.progress .bars {
				color: black;
			}
		</style>
	</head>
	<body>'''

HTML_FOOTER = '''
	</body>
</html>'''



if __name__ == '__main__':
	sys.exit(main(sys.argv))
