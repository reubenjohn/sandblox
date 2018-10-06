import inspect

from sandblox.util import tf_util as U
from sandblox.util.scope import *


# THIS FILE IS DEPRECATED AND WILL BE REMOVED IN A FUTURE RELEASE

class TFObject(object):
	scope = UninitializedScope()

	def __init__(self, scope_name: str = None, **kwargs):
		if isinstance(self.scope, UninitializedScope):
			self.scope = Scope(scope_name, self)
		# noinspection PyArgumentList
		super(TFObject, self).__init__(**kwargs)

	def setup_scope(self, scope_name):
		self.scope = Scope(scope_name, self)


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


class Outs(InpOut):
	pass


class TFSubGraph(TFObject):
	# TODO Add basic PythonicTFFunction-ality here and have it extend this
	pass


class PythonicTFFunction(TFObject):
	def __init__(self, name: str, func, override_inputs, *args, **kwargs):
		if name is None:
			name = func.__name__
		super(PythonicTFFunction, self).__init__(name)

		with tf.variable_scope(self.scope.rel, reuse=None):
			ret = func(*args, **kwargs)
			if override_inputs:
				self.i, self.o = ret
			else:
				self.i, self.o = self.args_to_inputs(func, *args, **kwargs), ret
		super(PythonicTFFunction, self).__init__(name, **kwargs)

	@staticmethod
	def args_to_inputs(func, *args, **kwargs):
		inp = Inp()
		params = list(inspect.signature(func).parameters.keys())
		for arg_i, arg in enumerate(args):
			inp.__dict__[params[arg_i]] = arg
		return inp

	def get_variables(self):
		return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope.exact_abs_pattern)

	def get_trainable_variables(self):
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope.exact_abs_pattern)

	def assign_trainable_vars(self, source_hypothesis_graph):
		weight_update = [tf.assign(new, old) for (new, old) in U.zipsame(self.get_trainable_variables(),
																		 source_hypothesis_graph.get_trainable_variables())]
		return weight_update

	def eval(self, feed_dict=None, options=None, run_metadata=None):
		return U.get_session().run(list(self.o), feed_dict, options, run_metadata)


class MetaTFFunction(object):
	def __new__(cls, func, scope_name: str = None, tf_func_class=PythonicTFFunction, override_inputs=False):
		def custom_fn(*args, **kwargs):
			return tf_func_class(scope_name, func, override_inputs, *args, **kwargs)

		return custom_fn


def pythonic_tf_function(scope_name: str = None, tf_func_class=PythonicTFFunction, override_inputs=False,
						 meta_tf_function=MetaTFFunction):
	def tf_fn(func):
		return meta_tf_function(func, scope_name, tf_func_class, override_inputs)

	return tf_fn


# def tf_class(tf_class, scope_name:str):
# 	tf_class_ref = tf_class
#
# 	def tf_cls(cls):
#

class TFMethod(PythonicTFFunction):
	def __init__(self, name: str, obj: [object, TFObject], method, override_inputs, *args, **kwargs):
		method_name = name if name is not None else method.__name__
		object_name = obj.scope.rel if isinstance(obj, TFObject) else type(obj).__name__
		super(TFMethod, self).__init__(object_name + "/" + method_name, method, override_inputs, *args, **kwargs)
		self.cls = obj
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
	def __new__(cls, method, scope_name: str = None, tf_method_class=TFMethod, override_inputs=False):
		def custom_meth(*args, **kwargs):
			return tf_method_class(scope_name, args[0], method, override_inputs, *args, **kwargs)

		return custom_meth


def tf_method(name: str = None, tf_method_class=TFMethod, override_inputs=False,
			  meta_method_class=MetaTFMethod):
	def tf_m(method):
		return meta_method_class(method, name, tf_method_class, override_inputs)

	return tf_m
