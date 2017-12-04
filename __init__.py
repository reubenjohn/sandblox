import inspect

import tensorflow as tf

from pythonic_tf.util import zipsame
from . import util as U


def get_scope_name():
	"""Returns the name of current scope as a string, e.g. deepq/q_func"""
	return tf.get_variable_scope().name


def absolute_scope_name(relative_scope_name):
	"""Appends parent scope name to `relative_scope_name`"""
	base = get_scope_name()
	base = base + "/" if len(base) > 0 else base
	return base + relative_scope_name


class TFObject(object):
	def __init__(self, scope_name: str = None, **kwargs):
		self.rel_scope_name = scope_name if scope_name is not None else type(self).__name__
		self.abs_scope_name = absolute_scope_name(self.rel_scope_name)
		# noinspection PyArgumentList
		super(TFObject, self).__init__(**kwargs)

	def exact_scope_name(self):
		return "%s/" % self.abs_scope_name

	def exact_absolute_scope_name(self):
		return "^%s/" % self.abs_scope_name


class InpOutBase(object):
	def __repr__(self):
		items = ("{}={!r}".format(k, self.__dict__[k]) for k in self.__dict__.keys())
		return "{}({})".format(type(self).__name__, ", ".join(items))

	def __eq__(self, other):
		return self.__dict__ == other.__dict__

	def __getitem__(self, item):
		return list(self.__dict__.values())[item]

	def keys(self):
		return self.__dict__.keys()

	def values(self):
		return self.__dict__.values()


class InpOutImplicitOrder(object):
	def __init__(self, **kwargs):
		self.__dict__.update(kwargs)
		pass


class InpOutExplicitOrder(object):
	def __init__(self, *args):
		self.keys = [kwarg[0] for kwarg in args]
		for kwarg in args:
			self.__dict__[kwarg[0]] = kwarg[1]
		pass

	def __eq__(self, other):
		raise NotImplementedError

	def __getitem__(self, item):
		return self.__dict__[self.keys[item]]


import platform

version = platform.python_version_tuple()
if version[0] > '3' and version[1] > '6':
	InpOut = InpOutImplicitOrder
	print('InpOut(InpOutImplicitOrder) has not been tested for python version 3.6+.'
	      'Proceed at own risk, and consider contributing results')
else:
	InpOut = InpOutExplicitOrder


class Inp(InpOut):
	pass


class Out(InpOut):
	pass


class TFSubGraph(TFObject):
	# TODO Add basic TFFunction-ality here and have it extend this
	pass


class TFFunction(TFObject):
	def __init__(self, name: str, func, override_inputs, *args, **kwargs):
		if name is None:
			name = func.__name__
		super(TFFunction, self).__init__(name)

		with tf.variable_scope(self.rel_scope_name, reuse = None):
			ret = func(*args, **kwargs)
			if override_inputs:
				self.inp, self.out = ret
			else:
				self.inp, self.out = self.args_to_inputs(func, *args, **kwargs), ret
		super(TFFunction, self).__init__(name, **kwargs)

	@staticmethod
	def args_to_inputs(func, *args, **kwargs):
		inp = Inp()
		params = list(inspect.signature(func).parameters.keys())
		for arg_i, arg in enumerate(args):
			inp.__dict__[params[arg_i]] = arg
		return inp

	def get_variables(self):
		return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.exact_absolute_scope_name())

	def get_trainable_variables(self):
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.exact_absolute_scope_name())

	def assign_trainable_vars(self, source_policy_graph):
		weight_update = tf.group(*(tf.assign(new, old) for (new, old) in zipsame(self.get_trainable_variables(),
		                                                                         source_policy_graph.get_trainable_variables())))
		return weight_update

	def eval(self, feed_dict = None, options = None, run_metadata = None):
		return U.get_session().run(list(self.out), feed_dict, options, run_metadata)


class MetaTFFunction(object):
	def __new__(cls, func, scope_name: str = None, tf_func_class = TFFunction, override_inputs = False):
		def custom_fn(*args, **kwargs):
			return tf_func_class(scope_name, func, override_inputs, *args, **kwargs)

		return custom_fn


def tf_function(scope_name: str = None, tf_func_class = TFFunction, override_inputs = False,
                meta_tf_function = MetaTFFunction):
	def tf_fn(func):
		return meta_tf_function(func, scope_name, tf_func_class, override_inputs)

	return tf_fn


# def tf_class(tf_class, scope_name:str):
# 	tf_class_ref = tf_class
#
# 	def tf_cls(cls):
#

class TFMethod(TFFunction):
	def __init__(self, name: str, cls, method, override_inputs, *args,
	             **kwargs):
		method_name = name if name is not None else method.__name__
		object_name = cls.rel_scope_name if isinstance(cls, TFObject) else type(cls).__name__
		super(TFMethod, self).__init__(object_name + "/" + method_name, method, override_inputs, *args, **kwargs)
		self.cls = cls
		self.method = method
		self.method_kargs = (args, kwargs)

	@staticmethod
	def args_to_inputs(method, *args, **kwargs):
		inp = Inp()
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


class MetaTFMethod(MetaTFFunction):
	def __new__(cls, method, scope_name: str = None, tf_method_class = TFMethod, override_inputs = False):
		def custom_meth(*args, **kwargs):
			return tf_method_class(scope_name, args[0], method, override_inputs, *args, **kwargs)

		return custom_meth


def tf_method(name: str = None, tf_method_class = TFMethod, override_inputs = False,
              meta_method_class = MetaTFMethod):
	def tf_m(method):
		return meta_method_class(method, name, tf_method_class, override_inputs)

	return tf_m
