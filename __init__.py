import inspect
from types import SimpleNamespace

import tensorflow as tf

from pythonic_tf.util import zipsame
from . import util as U


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


class TFFunction(TFObject):
	def __init__(self, scope_name: str, func, *args, **kwargs):
		super(TFFunction, self).__init__(scope_name)
		self.exact_abs_scope_name = self.scope_name

		# self.placeholders = kwargs['placeholders'] if 'placeholders' in kwargs else []
		inp = SimpleNamespace()
		params = list(inspect.signature(func).parameters.keys())
		for arg_i, arg in enumerate(args):
			inp.__dict__[params[arg_i]] = arg
		inp.__dict__.update(kwargs)
		self.inp = inp

		with tf.variable_scope(self.scope_name + "/" + type(func).__name__, reuse=None):
			self.out = func(*args, **kwargs)
		self.eval = lambda feed_dict: U.get_session().run(self.out, feed_dict)
		super(TFFunction, self).__init__(scope_name, **kwargs)

	def get_variables(self):
		return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.exact_abs_scope_name)

	def get_trainable_variables(self):
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.exact_abs_scope_name)

	# TODO Return tensorflow operation instead
	def assign_trainable_vars(self, source_policy_graph):
		# THIS_POLICY = POLICY
		weight_update = tf.group(tf.assign(new, old)
								 for (new, old) in
								 zipsame(self.get_trainable_variables(),
										 source_policy_graph.get_trainable_variables()))
		return weight_update

	def eval(self, feed_dict=None, options=None, run_metadata=None):
		U.get_session().run(self.out, feed_dict, options, run_metadata)


def tf_function(scope_name: str = None, tf_func_class=TFFunction):
	def tf_fn(func):
		def custom_fn(*args, **kwargs):
			return tf_func_class(scope_name, func, *args, **kwargs)

		return custom_fn

	return tf_fn


# def tf_class(tf_class, scope_name:str):
# 	tf_class_ref = tf_class
#
# 	def tf_cls(cls):
#

class TFMethod(TFFunction):
	def __init__(self, scope_name: str, cls, method, *args, **kwargs):
		super(TFMethod, self).__init__(scope_name, method, *args, **kwargs)
		self.cls = cls
		self.method = method
		self.method_kargs = (args, kwargs)

	def freeze(self):
		setattr(self.cls, self.method.__name__, self)

	def is_frozen(self):
		return isinstance(getattr(self.cls, self.method.__name__), TFMethod)

	def assert_frozen(self):
		assert self.is_frozen()


def tf_method(scope_name: str = None, tf_method_class=TFMethod):
	def tf_m(method):
		def custom_meth(*args, **kwargs):
			return tf_method_class(scope_name, args[0], method, *args, **kwargs)

		return custom_meth

	return tf_m
