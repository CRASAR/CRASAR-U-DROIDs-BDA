import random

def get_line_id():
	return "%032x" % random.getrandbits(128)