"""
Tools for dealing with images stored in the IML format.
"""

import numpy
import array

def loadiml(filename):
	with open(filename, 'rb') as handle:
		data = handle.read()

	arr = array.array('H', data)
	arr.byteswap()

	return numpy.array(arr, dtype='uint16').reshape(1024, 1536)
