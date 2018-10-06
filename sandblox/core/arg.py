from typing import Callable


class LateBoundArg(object):
	def __init__(self, resolver: Callable):
		self.resolver = resolver

	def resolve(self):
		return self.resolver()

def arg(resolver: Callable):
	return LateBoundArg(resolver)

def resolve(*args):
	resolved = [arg.resolve() if isinstance(arg, LateBoundArg) else arg for arg in args]
	return resolved if len(resolved) > 1 else resolved[0]
