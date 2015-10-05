"""
Collect snapshots and merge into a single model.
"""

import os
import sys

sys.path.append('./code')

from glob import glob
from tools import Experiment

def main(argv):
	preconditioners = []
	models = []

	for i in range(63):
		files = glob('results/BSDS300/snapshots/mcgsm_{0}_128.*.xpck'.format(i))
		files.sort(key=os.path.getmtime)

		if len(files) == 0:
			print 'Could not find snapshot for model {0}.'.format(i)
			files = glob('results/BSDS300/snapshots/mcgsm_{0}_64.*.xpck'.format(i))

		filepath = files[-1]

		print 'Using {0}.'.format(filepath)

		experiment = Experiment(filepath)

		preconditioners.append(experiment['preconditioners'][i])
		models.append(experiment['models'][i])

	experiment = Experiment()
	experiment['models'] = models
	experiment['preconditioners'] = preconditioners
	experiment.save('results/BSDS300/mcgsm_128_merged.xpck', overwrite=True)

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
