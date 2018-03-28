import inspect
import sys
from collections import OrderedDict
from typing import Type, Any

import numpy
import tensorflow as tf

from sandblox.util import zipsame
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

	def __getitem__(self, item):
		return self.__dict__.__getitem__(item)

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
		if key in self.o:
			print('Warning an output named %s already exists with value: %s' % (key, self.o[key]))
		self.o[key] = val
		self.oz.append(val)

	def _on_new_attr_val(self, key, val):
		self.add_output(key, val)


class BlockOutsKwargs(BlockOutsBase):
	def __init__(self, **kwargs):
		self.o = kwargs
		if sys.version_info[0] < 3 or (sys.version_info[0] == 3 and sys.version_info[1] < 6):
			print('WARNING: keyword arguments constructor will not preserve output order before Python 3.6!\n' +
				  'Please use the empty constructor approach provided for backward compatibility:\n' +
				  'Eg: ' + type(self).__name__ + '().a(a_val).b(b_val)')
		self.oz = tuple(val for val in kwargs.values())


class BlockOutsAttrs(BlockOutsBase):
	def __init__(self):
		self.o = DictAttrs()
		self.oz = []


Out = AttrFactory(BlockOutsAttrs) if sys.version_info[0] < 3 or (
	sys.version_info[0] == 3 and sys.version_info[1] < 6) else BlockOutsKwargs


def dynamic(*args):
	if len(args) == 1:
		args[0].__is_d_inp = True
		return args[0]
	for arg in args:
		arg.__is_d_inp = True
	return args


def is_dynamic_arg(arg):
	return hasattr(arg, '__is_d_inp') or (isinstance(arg, tf.Tensor) and arg.op.type == 'Placeholder')


def soft_assign(self, *args):
	"""
	Useful to avoid false positive warnings from IDEs during __init__
	:param self: the object on which to perform the soft initialization
	:param args: the attribute key (or tuple key value pair) for which to probe based on which the assignment may or may not be performed
	:return:
	"""
	for arg in args:
		if isinstance(arg, tuple):
			val = arg[1]
			arg = arg[0]
		else:
			val = None
		if hasattr(self, arg) and getattr(self, arg) is not None:
			yield getattr(self, arg)
		else:
			yield val


def flattened_dynamic_arguments(inps: dict) -> list:
	result = []
	for key in inps:
		inp = inps[key]
		if is_dynamic_arg(inp):
			result.append(inp)
		elif isinstance(inp, Block):
			result.extend(inp.di)
	return result


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


class Block(object):
	default_scope_name = None
	scope = UninitializedScope()

	def __init__(self, *args, **kwargs):
		if isinstance(self.scope, UninitializedScope):
			self.scope = Scope(self, kwargs.pop('scope_name', self.default_scope_name))
		self.o, self.oz, self.i, self.iz = None, None, None, None
		self.di = []
		self._build(*args, **kwargs)
		self.built_fn = None
		super(Block, self).__init__()

	def setup_scope(self, scope_name):
		self.scope = Scope(self, scope_name)

	def eval(self, *args, **kwargs):
		raise NotImplementedError

	def process_inputs(self, *args, **kwargs):
		pass

	def run(self, *args, **kwargs):
		self.process_inputs(*args, **kwargs)
		dynamic_oz = self.eval(*args, **kwargs) if self.is_dynamic() else self.built_fn(*args, **kwargs)
		self.process_outputs(dynamic_oz)
		return dynamic_oz

	def process_outputs(self, outputs):
		for inp in self.iz:
			if isinstance(inp, Block):
				inp.process_outputs(outputs)
		self.process_my_outputs(outputs)

	def process_my_outputs(self, outputs):
		pass

	def is_dynamic(self):
		return self.built_fn is None or any(b.is_dynamic() for b in self.iz if isinstance(b, Block))

	def build(self, *args, **kwargs):
		raise NotImplementedError

	def _build(self, *args, **kwargs):
		i = FlatBoundArguments(self.build)(*args, **kwargs)
		self.i = DictAttrs(i)
		self.iz = list(i.values())
		self.di.extend(flattened_dynamic_arguments(i))

		with tf.variable_scope(self.scope.rel):
			print('building in scope %s' % self.scope.rel)
			ret = self.build(*args, **kwargs)

		if isinstance(ret, Out.cls):
			block_outputs = ret
		elif hasattr(ret, '__len__') and len(ret) > 1 and isinstance(ret[0], Out.cls):
			block_outputs = ret[0]
		else:
			raise AssertionError(
				'A SandBlock must either return only a ' + type(Out).__name__
				+ ' or it must be the first element of what is returned'
			)
		self.o = block_outputs.o
		self.oz = block_outputs.oz

		return ret


# TODO Support passing session and name_scope as kwarg
class TFBlock(Block):
	def build(self, *args, **kwargs):
		raise NotImplementedError

	def __init__(self, *args, **kwargs):
		sess = kwargs.pop('session', None)
		assert sess is None or isinstance(sess, tf.Session), 'Specified session must be of type tf.Session'
		super(TFBlock, self).__init__(*args, **kwargs)
		self.givens = {}
		self.options = None
		self.run_metadata = None
		self.built_fn = util.function(self.di, self.oz, session=sess)

	def set_session(self, session: tf.Session):
		self.built_fn.sess = session

	@property
	def sess(self):
		return self.built_fn.sess if self.built_fn.sess is not None else U.get_session()

	def process_inputs(self, *args, **kwargs):
		super(TFBlock, self).process_inputs(*args, **kwargs)
		if not self.is_dynamic():
			self.built_fn.givens = self.get_all_givens()
			self.built_fn.using(self.options, self.run_metadata)

	def get_all_givens(self) -> dict:
		givens = {}
		for inp in self.iz:
			if isinstance(inp, TFBlock):
				child_givens = inp.get_all_givens()
				givens.update(child_givens)
		my_givens = self.get_my_givens()
		givens.update(my_givens)
		return givens

	def get_my_givens(self):
		return self.givens

	def using(self, options=None, run_metadata=None):
		self.options = options
		self.run_metadata = run_metadata
		return self

	def eval(self, *args, **kwargs):
		pass


class StateShape(object):
	def __init__(self, shape):
		self._shape = shape

	def shape(self):
		return self._shape

	def batch_shape(self, batch_size: [int, list] = 1):
		batch_size = batch_size if isinstance(batch_size, list) else [batch_size]
		return batch_size + self.shape()

	def new(self, batch_size: [int, list] = 1):
		return numpy.random.uniform(-1, 1, self.batch_shape(batch_size))

	def new_placeholder(self, batch_size: [None, int, list] = None):
		return tf.placeholder(tf.float32, self.batch_shape(batch_size), 'state')

	def new_variable(self, batch_size: [int, list] = 1):
		return tf.Variable(self.new(batch_size), name='state')

	def assign(self, dest_state, src_state):
		return tf.assign(dest_state, src_state)


class StatefulTFBlock(TFBlock):
	STATE = None  # type: StateShape

	def __init__(self, *args, **kwargs):
		self.prev_state = self.next_state = self.state_index = self.state = None
		super(StatefulTFBlock, self).__init__(*args, **kwargs)

	def _build(self, *args, **kwargs):
		super(StatefulTFBlock, self)._build(*args, **kwargs)
		tuple_state = self.o.state
		if isinstance(tuple_state, tuple):
			self.prev_state, self.next_state = self.o.state
			if not is_dynamic_arg(self.prev_state):
				dependencies = [self.o[key] for key in self.o if key != 'state']
				with tf.control_dependencies(dependencies):
					updated_state = self.STATE.assign(self.prev_state, self.next_state)
			else:
				updated_state = self.next_state
			self.o.state = updated_state
			oz_index = self.oz.index(tuple_state)
			self.oz[oz_index] = self.o.state
			if is_dynamic_arg(self.prev_state):
				self.state_index = oz_index

	def build(self, *args, **kwargs):
		raise NotImplementedError

	def get_my_givens(self):
		givens = super(StatefulTFBlock, self).get_my_givens()
		if is_dynamic_arg(self.prev_state):
			givens.update({self.prev_state: self.state})
		return givens

	def process_my_outputs(self, outputs):
		super(StatefulTFBlock, self).process_my_outputs(outputs)
		if self.state_index is not None:
			self.state = outputs[self.state_index]


def cast_to_stateful_tf_block(ob) -> StatefulTFBlock:
	return ob


def get_class_for_block_fn(fn, base_cls):
	class BlockFn(base_cls):
		build = fn

		default_scope_name = fn.__name__

		def __init__(self, *args, **kwargs):
			self.build = fn
			super(BlockFn, self).__init__(*args, **kwargs)

		def eval(self, *args, **kwargs):
			raise NotImplementedError

	return BlockFn


class Decorators(object):
	@staticmethod
	def block_decorator(fn) -> Type[Block]:
		return get_class_for_block_fn(fn, Block)

	@staticmethod
	def block_meta_decorator(cls: Type[Any]) -> '(fn: Any) -> Type[Block]':
		def block_decorator(fn) -> Type[Block]:
			return get_class_for_block_fn(fn, cls)

		return block_decorator

	@staticmethod
	def tf_block_decorator(fn) -> Type[TFBlock]:
		return get_class_for_block_fn(fn, TFBlock)

	@staticmethod
	def tf_block_meta_decorator(cls: Type[Any]) -> '(fn: Any) -> Type[TFBlock]':
		def tf_block_decorator(fn) -> Type[TFBlock]:
			return get_class_for_block_fn(fn, cls)

		return tf_block_decorator


def block(cls) -> Type[Block]:
	is_meta_decorator = not inspect.isfunction(cls)
	return Decorators.block_meta_decorator(cls) if is_meta_decorator else Decorators.block_decorator(cls)


def tf_block(tf_block_cls) -> Type[TFBlock]:
	is_meta_decorator = not inspect.isfunction(tf_block_cls)
	return Decorators.tf_block_meta_decorator(tf_block_cls) if is_meta_decorator else Decorators.tf_block_decorator(
		tf_block_cls)


def stateful_tf_block(state_shape, cls: Type[Any] = StatefulTFBlock) -> '(fn: Any) -> Type[StatefulTFBlock]':
	assert isinstance(state_shape, StateShape)
	assert issubclass(cls, StatefulTFBlock)

	def stateful_tf_block_decorator(fn) -> Type[StatefulTFBlock]:
		class StatefulTFBlockFn(cls):
			build = fn
			default_scope_name = fn.__name__
			STATE = state_shape

			def __init__(self, *args, **kwargs):
				self.build = fn
				super(StatefulTFBlockFn, self).__init__(*args, **kwargs)

		# noinspection PyTypeChecker
		return StatefulTFBlockFn

	return stateful_tf_block_decorator


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
	scope = UninitializedScope()

	def __init__(self, scope_name: str = None, **kwargs):
		if isinstance(self.scope, UninitializedScope):
			self.scope = Scope(self, scope_name)
		# noinspection PyArgumentList
		super(TFObject, self).__init__(**kwargs)

	def setup_scope(self, scope_name):
		self.scope = Scope(self, scope_name)


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

		with tf.variable_scope(self.scope.rel, reuse=None):
			ret = func(*args, **kwargs)
			if override_inputs:
				self.i, self.o = ret
			else:
				self.i, self.o = self.args_to_inputs(func, *args, **kwargs), ret
		super(TFFunction, self).__init__(name, **kwargs)

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
		weight_update = [tf.assign(new, old) for (new, old) in zipsame(self.get_trainable_variables(),
																	   source_hypothesis_graph.get_trainable_variables())]
		return weight_update

	def eval(self, feed_dict=None, options=None, run_metadata=None):
		return U.get_session().run(list(self.o), feed_dict, options, run_metadata)


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
