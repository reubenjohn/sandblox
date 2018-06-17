import tensorflow as tf

from .tf_util import scope_name as get_scope_name


def absolute_scope_name(relative_scope_name):
	"""Appends parent scope name to `relative_scope_name`"""
	base = get_scope_name()
	base = base + "/" if len(base) > 0 else base
	return base + relative_scope_name


def infer_scope_name(self, scope_name):
	return scope_name if scope_name is not None else type(self).__name__


infer_rel_scope_name = infer_scope_name


def infer_abs_scope_name(self, scope_name: str = None):
	scope_name = infer_rel_scope_name(self, scope_name)
	return absolute_scope_name(scope_name)


class Scope(object):
	def __init__(self, obj, scope_name: str):
		self.rel = self.abs = None
		self.setup(obj, scope_name)

	def setup(self, obj, scope_name: str = None):
		self.rel = infer_rel_scope_name(obj, scope_name)
		self.abs = absolute_scope_name(self.rel)

	def make_unique(self, graph=None):
		if graph is None:
			graph = tf.get_default_graph()
		graph.unique_name(self.rel)

	@property
	def exact_rel_pattern(self) -> str:
		return self.abs + '/'

	@property
	def exact_abs_pattern(self) -> str:
		return '^' + self.abs + '/'


class UninitializedScope(Scope):
	# noinspection PyMissingConstructor
	def __init__(self):
		pass

	def __getattribute__(self, item):
		raise AttributeError('The scope is only available after you call super constructor __init__.\n'
							 'Alternatively, manually setup the scope with self.setup_scope(scope_name)')
