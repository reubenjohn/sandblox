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
	def __init__(self, scope_name: str, func, override_inputs, decorator_args: [None, dict] = None, *args, **kwargs):
		if scope_name is None:
			scope_name = func.__name__
		super(TFFunction, self).__init__(scope_name)
		self.exact_abs_scope_name = self.scope_name

		with tf.variable_scope(self.scope_name, reuse=None):
			ret = func(*args, **kwargs)
			if override_inputs:
				self.inp, self.out = ret
			else:
				self.inp, self.out = self.args_to_inputs(func, *args, **kwargs), ret
		super(TFFunction, self).__init__(scope_name, **kwargs)

	def eval(self, feed_dict):
		return U.get_session().run(list(self.out.__dict__.values()), feed_dict)

	@staticmethod
	def args_to_inputs(func, *args, **kwargs):
		inp = SimpleNamespace()
		params = list(inspect.signature(func).parameters.keys())
		for arg_i, arg in enumerate(args):
			inp.__dict__[params[arg_i]] = arg
		return inp

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


def tf_function(scope_name: str = None, tf_func_class=TFFunction, override_inputs=False):
	def tf_fn(func):
		def custom_fn(*args, **kwargs):
			return tf_func_class(scope_name, func, override_inputs, *args, **kwargs)

		return custom_fn

	return tf_fn


# def tf_class(tf_class, scope_name:str):
# 	tf_class_ref = tf_class
#
# 	def tf_cls(cls):
#

class TFMethod(TFFunction):
	def __init__(self, scope_name: str, cls, method, override_inputs, decorator_args: [None, dict] = None, *args,
				 **kwargs):
		super(TFMethod, self).__init__(scope_name, method, override_inputs, decorator_args, *args, **kwargs)
		self.cls = cls
		self.method = method
		self.method_kargs = (args, kwargs)

	@staticmethod
	def args_to_inputs(method, *args, **kwargs):
		inp = SimpleNamespace()
		args = args[1:]
		params = list(inspect.signature(method).parameters.keys())[1:]
		for arg_i, arg in enumerate(args):
			inp.__dict__[params[arg_i]] = arg
		return inp

	def freeze(self):
		setattr(self.cls, self.method.__name__, self)

	def is_frozen(self):
		return isinstance(getattr(self.cls, self.method.__name__), TFMethod)

	def assert_frozen(self):
		assert self.is_frozen()


def tf_method(scope_name: str = None, tf_method_class=TFMethod, override_inputs=False, *args, **kwargs):
	decorator_args = dict(args=args, kwargs=kwargs)

	def tf_m(method):
		def custom_meth(*args, **kwargs):
			return tf_method_class(scope_name, args[0], method, override_inputs, decorator_args, *args, **kwargs)

		return custom_meth

	return tf_m
