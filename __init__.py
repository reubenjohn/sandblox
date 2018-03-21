import inspect
import sys
from collections import OrderedDict

import tensorflow as tf

from pythonic_tf.util import zipsame
from . import util as U


class FlatBoundArguments(object):
	def __init__(self, fn):
		self.signature = inspect.signature(fn)

	@staticmethod
	def _flatten_kwargs(bound_args: OrderedDict):
		if 'kwargs' in bound_args:
			bound_args.update(bound_args['kwargs'])
			bound_args.pop('kwargs')

	def __call__(self, *args, **kwargs) -> OrderedDict:
		bound_args = self.signature.bind(*args, **kwargs)
		bound_args.apply_defaults()
		arguments = bound_args.arguments
		self._flatten_kwargs(arguments)
		return arguments


class DictAttrs(object):
	def __init__(self, dic: dict = None):
		if dic is not None:
			self.__dict__.update(dic)

	def __iter__(self):
		return self.__dict__.__iter__()

	def __setitem__(self, key, value):
		self.__dict__.__setitem__(key, value)

	def __str__(self):
		return self.__dict__.__str__()


class AttrFactory(object):
	def __init__(self, attr_builder_class):
		self.cls = attr_builder_class

	def __getattr__(self, item):
		if item == 'cls':
			return self.cls
		return self.cls().__getattr__(item)


class AttrBuilder:
	def _on_new_attr_val(self, key, val):
		raise NotImplementedError

	def _new_attr_val(self, key, val):
		self._on_new_attr_val(key, val)
		return self

	def __getattr__(self, item):
		return lambda val: self._new_attr_val(item, val)


class BlockOutsBase(AttrBuilder):
	def add_output(self, key, val):
		if key in self.out:
			print('Warning an output named %s already exists with value: %s' % (key, self.out[key]))
		self.out[key] = val
		self.outs += (val,)

	def _on_new_attr_val(self, key, val):
		self.add_output(key, val)


class BlockOutsKwargs(BlockOutsBase):
	def __init__(self, **kwargs):
		self.out = kwargs
		if sys.version_info[0] < 3 or (sys.version_info[0] == 3 and sys.version_info[1] < 6):
			print('WARNING: keyword arguments constructor will not preserve output order before Python 3.6!\n' +
				  'Please use the empty constructor approach provided for backward compatibility:\n' +
				  'Eg: ' + type(self).__name__ + '().a(a_val).b(b_val)')
		self.outs = tuple(val for val in kwargs.values())


class BlockOutsAttrs(BlockOutsBase):
	def __init__(self):
		self.out = DictAttrs()
		self.outs = ()


Out = AttrFactory(BlockOutsAttrs) if sys.version_info[0] < 3 or (
	sys.version_info[0] == 3 and sys.version_info[1] < 6) else BlockOutsKwargs


def dynamic(arg):
	arg.__is_d_inp = True
	return arg


def is_dynamic_arg(arg):
	return hasattr(arg, '__is_d_inp')


def soft_initialize(self, *args):
	"""
	Useful to avoid false positive warnings from IDEs during __init__
	:param self: the object on which to perform the soft initialization
	:param a: the attribute key for which to probe based on which the assignment may or may not be performed
	:return:
	"""
	for arg in args:
		yield arg if hasattr(self, arg) and getattr(self, arg) is not None else None


def flattened_dynamic_arguments(inp: dict) -> list:
	result = []
	for key in inp:
		if is_dynamic_arg(inp[key]):
			result.append(inp[key])
		elif isinstance(inp[key], Mold):
			result.extend(inp[key].d_inps)
	return result


class Mold(object):
	def __init__(self, *args, **kwargs):
		self.out, self.outs = soft_initialize(self, 'out', 'outs')
		self.inp, self.inps = soft_initialize(self, 'inp', 'inps')
		self.d_inps, = soft_initialize(self, 'd_inps')
		super(Mold, self).__init__()
		self.build(*args, **kwargs)
		self.eval = util.function(self.d_inps, self.outs)

	def on_build(self, *args, **kwargs):
		raise NotImplementedError

	def build(self, *args, **kwargs):
		inp = FlatBoundArguments(self.on_build)(*args, **kwargs)
		self.inp = DictAttrs(inp)
		self.inps = list(inp.values())
		self.d_inps = flattened_dynamic_arguments(inp)

		ret = self.on_build(*args, **kwargs)

		self.out = ret.out
		self.outs = ret.outs
		self.set_out(ret)
		return ret

	def set_out(self, outs: Out = None):
		if isinstance(outs, Out.cls):
			block_outputs = outs
		elif isinstance(outs[0], Out):
			block_outputs = outs[0]
		else:
			raise AssertionError(
				'A SandBlock must either return only a ' + type(Out).__name__
				+ ' or it must be the first element of what is returned'
			)
		self.out = block_outputs.out
		self.outs = block_outputs.outs


def mold(fn):
	class MoldFn(Mold):
		on_build = fn

		def __init__(self, *args, **kwargs):
			self.on_build = fn
			super(MoldFn, self).__init__(*args, **kwargs)

	return MoldFn


def get_scope_name():
	"""Returns the name of current scope as a string, e.g. deepq/q_func"""
	return tf.get_variable_scope().name


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


class TFObject(object):
	def __init__(self, scope_name: str = None, **kwargs):
		self._rel_scope_name = self._abs_scope_name = None
		self.setup_scope(scope_name)
		# noinspection PyArgumentList
		super(TFObject, self).__init__(**kwargs)

	def setup_scope(self, scope_name: str = None):
		self._rel_scope_name = infer_rel_scope_name(self, scope_name)
		self._abs_scope_name = absolute_scope_name(self._rel_scope_name)

	def make_scope_unique(self, graph=None):
		if graph is None:
			graph = tf.get_default_graph()
		graph.unique_name(self.rel_scope_name)

	@property
	def rel_scope_name(self):
		if self._rel_scope_name is None:
			raise AttributeError('rel_scope_name is only available after you call super constructor __init__.\n'
								 'You may instead also use self.infer_rel_scope_name(self, scope_name)')
		return self._rel_scope_name

	@property
	def abs_scope_name(self):
		if self._abs_scope_name is None:
			raise AttributeError('abs_scope_name is only available after you call super constructor __init__.\n'
								 'You may instead also use self.infer_abs_scope_name(self, scope_name)')
		return self._abs_scope_name

	@property
	def exact_rel_scope_name_pattern(self):
		return "%s/" % self.abs_scope_name

	@property
	def exact_absolute_scope_name_pattern(self):
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


class Outs(InpOut):
	pass


class TFSubGraph(TFObject):
	# TODO Add basic TFFunction-ality here and have it extend this
	pass


class TFFunction(TFObject):
	def __init__(self, name: str, func, override_inputs, *args, **kwargs):
		if name is None:
			name = func.__name__
		super(TFFunction, self).__init__(name)

		with tf.variable_scope(self.rel_scope_name, reuse=None):
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
		return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.exact_absolute_scope_name_pattern)

	def get_trainable_variables(self):
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.exact_absolute_scope_name_pattern)

	def assign_trainable_vars(self, source_hypothesis_graph):
		weight_update = [tf.assign(new, old) for (new, old) in zipsame(self.get_trainable_variables(),
																	   source_hypothesis_graph.get_trainable_variables())]
		return weight_update

	def eval(self, feed_dict=None, options=None, run_metadata=None):
		return U.get_session().run(list(self.out), feed_dict, options, run_metadata)


class MetaTFFunction(object):
	def __new__(cls, func, scope_name: str = None, tf_func_class=TFFunction, override_inputs=False):
		def custom_fn(*args, **kwargs):
			return tf_func_class(scope_name, func, override_inputs, *args, **kwargs)

		return custom_fn


def tf_function(scope_name: str = None, tf_func_class=TFFunction, override_inputs=False,
				meta_tf_function=MetaTFFunction):
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
	def __new__(cls, method, scope_name: str = None, tf_method_class=TFMethod, override_inputs=False):
		def custom_meth(*args, **kwargs):
			return tf_method_class(scope_name, args[0], method, override_inputs, *args, **kwargs)

		return custom_meth


def tf_method(name: str = None, tf_method_class=TFMethod, override_inputs=False,
			  meta_method_class=MetaTFMethod):
	def tf_m(method):
		return meta_method_class(method, name, tf_method_class, override_inputs)

	return tf_m
