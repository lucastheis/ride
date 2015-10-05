class RGB(object):
	"""
	Generates RGB colors in PGF format.

	@type red: integer/float
	@ivar red: value between 0 and 255 or 0. and 1.

	@type green: integer/float
	@ivar green: value between 0 and 255 or 0. and 1.

	@type blue: integer/float
	@ivar blue: value between 0 and 255 or 0. and 1.
	"""

	def __init__(self, red, green, blue):
		if not isinstance(red, (int, float)) or \
		   not isinstance(green, (int, float)) or \
		   not isinstance(blue, (int, float)):
			raise TypeError('Colors should be specified as integer or floats.')

		if isinstance(red, int):
			self.type = int
			if not (0 <= red <= 255) or \
			   not (0 <= green <= 255) or \
			   not (0 <= blue <= 255):
				raise ValueError('Integer color values should be between 0 and 255.')
		else:
			self.type = float
			if not (0. <= red <= 1.) or \
			   not (0. <= green <= 1.) or \
			   not (0. <= blue <= 1.):
				raise ValueError('Real Color values should be between 0 and 1.')

		self.red = red
		self.green = green
		self.blue = blue


	def __str__(self):
		if self.type == int:
			return '{{rgb,255:red,{0};green,{1};blue,{2}}}'.format(
				self.red, self.green, self.blue)
		else:
			return '{{rgb,1:red,{0:.2f};green,{1:.2f};blue,{2:.2f}}}'.format(
				self.red, self.green, self.blue)
