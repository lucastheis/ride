from numpy import min, max, iterable
from string import rstrip

def indent(text, times=1, ind='\t'):
	"""
	Indent text with multiple lines.

	@type  text: string
	@param text: some text

	@type  times: integer
	@param times: number of indentations

	@type  ind: string
	@param ind: string inserted at the beginning of each line
	"""

	if times > 1:
		text = indent(text, times - 1)
	return '\n'.join([rstrip(ind + line) for line in text.split('\n')])


def min_free(indices):
	if not indices:
		return 0
	return min(list(set(range(max(indices) + 2)).difference(indices)))


def escape(string):
	"""
	If string contains commas, put curly braces around the string. If string
	contains underscore but no formulas, escape underscores.

	@type  string: string/list
	@param string: e.g. a legend entry or axis label

	@rtype: string/list
	@return: processed string(s)
	"""

	if not isinstance(string, str) and iterable(string):
		return [escape(s) for s in string]

	if '$' not in string and r'\_' not in string:
		string = str(string).replace('_', r'\_')

	if r'\%' not in string:
		string = str(string).replace('%', r'\%')

	if ',' in string:
		return '{' + str(string) + '}'

	return string
