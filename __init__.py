import tensorflow as tf


def scope_name():
	"""Returns the name of current scope as a string, e.g. deepq/q_func"""
	return tf.get_variable_scope().name


def absolute_scope_name(relative_scope_name):
	"""Appends parent scope name to `relative_scope_name`"""
	base = scope_name()
	base = base + "/" if len(base) > 0 else base
	return base + relative_scope_name


class TFObject(object):
	def __init__(self, scope_name: str = None, **kwargs):
		self.scope_name = type(self).__name__ if scope_name is None else scope_name
		super(TFObject, self).__init__(**kwargs)

	def absolute_scope_name(self):
		return absolute_scope_name(self.scope_name)

	def exact_scope_name(self):
		return "%s/" % self.scope_name

	def exact_absolute_scope_name(self):
		return "%s/" % self.absolute_scope_name()
