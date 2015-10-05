__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@theis.io>'
__docformat__ = 'epytext'
__version__ = '0.4.0'

from multiprocessing import Process, Queue, cpu_count
from random import randint, seed as py_seed

try:
	from numpy.random import seed as np_seed
except ImportError:
	np_seed = lambda rseed: None

def mapp(function, *args):
	"""
	A parallel implementation of map. Example:

		>>> mapp(lambda x, y: x + y, range(10), range(10))

	@type  function: function
	@param function: the function that will be applied to the given arguments

	@rtype: list
	@return: an ordered list of the function's return values
	"""

	if mapp.max_processes < 2:
		return map(function, *args)

	if len(args) < 1:
		raise TypeError('mapp() takes at least 2 arguments')

	def run(function, queue, indices, rseed, *args):
		# randomize
		np_seed(rseed)
		py_seed(rseed)

		# compute and store results
		queue.put(dict(zip(indices, map(function, *args))))

	# number of arguments and number of processes
	num_args = len(args[0])
	num_jobs = min(num_args, mapp.max_processes)

	# used later to identify results
	indices = range(num_args)

	# queue for storing return values
	queue = Queue(num_args)

	# start processes
	processes = []
	for j in range(num_jobs):
		# prepare job
		job_args = [function, queue, indices[j::num_jobs], randint(0, 1E7)] + \
			[arg[j::num_jobs] for arg in args]

		# start process
		processes.append(Process(target=run, args=job_args))
		processes[-1].start()

	# collect and store results
	results = {}
	for _ in range(num_jobs):
		results.update(queue.get())

	# wait for processes to finish
	for process in processes:
		process.join()

	# return sorted results
	return [results[idx] for idx in range(num_args)]

mapp.max_processes = cpu_count()
