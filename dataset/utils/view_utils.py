import random

def get_view_id():
	return "%032x" % random.getrandbits(128)