import inspect
import sys
from typing import Type, Any

import numpy
import tensorflow as tf

from sandblox.util import zipsame
from . import util as U

# TODO Introduce DesignViolation escalation system
# TODO Implement sandblox saving mechanism

TFGraphKeys = [val for key, val in zip(tf.GraphKeys.__dict__.keys(), tf.GraphKeys.__dict__.values()) if
			   isinstance(val, str) and '__' not in key]


class DictAttrs(object):
	def __init__(self, **dic):
		self.__dict__.update(dic)

	def __iter__(self):
		return self.__dict__.__iter__()

	def __getitem__(self, item):
		return self.__dict__.__getitem__(item)

	def __setitem__(self, key, value):
		self.__dict__.__setitem__(key, value)

	def __str__(self):
		return self.__dict__.__str__()


class DictAttrBuilder:
	def _on_new_attr_val(self, key, val):
		raise NotImplementedError

	def _new_attr_val(self, key, val):
		self._on_new_attr_val(key, val)
		return self

	def __getattr__(self, item):
		return lambda val: self._new_attr_val(item, val)


class BlockOutsBase(DictAttrBuilder):
	def _on_new_attr_val(self, key, val):
		if key in self.o:
			print('Warning an output named %s already exists with value: %s' % (key, self.o[key]))
		self.o[key] = val
		self.oz.append(val)


class BlockOutsKwargs(BlockOutsBase):
	_you_were_warned = False  # TODO Use DesignViolation implementation instead

	def __init__(self, **kwargs):
		self.o = kwargs
		if not BlockOutsKwargs._you_were_warned and (
						sys.version_info[0] < 3 or (sys.version_info[0] == 3 and sys.version_info[1] < 6)):
			print('WARNING: keyword arguments constructor will not preserve output order before Python 3.6!\n' +
				  'Please use the empty constructor approach provided for backward compatibility:\n' +
				  'Eg: ' + type(self).__name__ + '().a(a_val).b(b_val)')
		self.oz = tuple(val for val in kwargs.values())


class BlockOutsAttrs(BlockOutsBase):
	def __init__(self):
		self.o = DictAttrs()
		self.oz = []


class DictAttrBuilderFactory(object):
	def __init__(self, attr_builder_class):
		self.cls = attr_builder_class

	def __getattr__(self, item):
		if item == 'cls':
			return self.cls
		return self.cls().__getattr__(item)


Out = DictAttrBuilderFactory(BlockOutsAttrs) if sys.version_info[0] < 3 or (
	sys.version_info[0] == 3 and sys.version_info[1] < 6) else BlockOutsKwargs

Props = DictAttrs  # Don't need DictAttrBuilderFactory since prop order does not need to be maintained


# TODO Add tests for dynamic arg concept
def dynamic(*args):
	if len(args) == 1:
		args[0].is_d_inp = True
		return args[0]
	for arg in args:
		arg.is_d_inp = True
	return args


def is_dynamic_arg(arg):
	return hasattr(arg, 'is_d_inp') or (isinstance(arg, tf.Tensor) and arg.op.type == 'Placeholder')


class OptionalDynamicArg(object):
	__slots__ = 'default_arg'
	is_d_inp = True

	def __init__(self, default_arg=None):
		self.default_arg = default_arg

	def resolve(self, *args, **kwargs):
		return self.default_arg


def resolve(arg: OptionalDynamicArg, *args, **kwargs):
	return arg.resolve(*args, **kwargs) if isinstance(arg, OptionalDynamicArg) else arg


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
			result.append(inp.default_arg if isinstance(inp, OptionalDynamicArg) else inp)
		elif isinstance(inp, BlockBase):  # Do subclasses also evaluate to True?
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


class BlockBase(object):
	scope = UninitializedScope()
	default_scope_name = None
	props = None

	def __init__(self, **props):
		self.i = self.o = self.iz = self.oz = self.di = self.built_fn = self.scope = None
		props = Props(**props)
		if self.props:
			props.__dict__.update(self.props.__dict__)
		self.props = props
		dic = props.__dict__
		self.scope = Scope(self, dic.get('scope_name', None))
		# TODO Test name collision when explicitly specified names for two blocks are the same, and the lack thereof
		if dic.get('make_scope_unique', True):
			graph = dic.get('graph', tf.get_default_graph())
			assert graph is not None, 'Could not find a default graph, so a graph must be provided since make_scope_unique is True'
			self.scope.make_unique(graph)

	def __call__(self, *args, **kwargs):
		self.i, self.iz, self.di = self._bind(*args, **kwargs)
		self._build(*args, **kwargs)
		return self

	def is_dynamic(self):
		return self.built_fn is None or any(b.is_dynamic() for b in self.iz if isinstance(b, BlockBase))

	# TODO Add test case
	def setup_scope(self, scope_name):
		self.scope = Scope(self, scope_name)

	def _bind(self, *args, **kwargs):
		bound_i = util.FlatArgumentsBinder(self.build)(*args, **kwargs)
		i = DictAttrs(**bound_i)
		iz = list(bound_i.values())
		di = flattened_dynamic_arguments(bound_i)
		return i, iz, di

	def _build(self, *args, **kwargs):
		if len(self.get_all_ops()) > 0:
			print('WARNING: Building ops into pollute d name scope')  # TODO Implement DesignViolation here
		with tf.variable_scope(self.scope.rel):
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

	def build(self, *args, **kwargs):
		raise NotImplementedError

	def run(self, *args, **kwargs):
		self.process_inputs(*args, **kwargs)
		dynamic_oz = self.eval(*args, **kwargs)
		self.post_eval(dynamic_oz)
		return dynamic_oz

	def process_inputs(self, *args, **kwargs):
		pass

	def eval(self, *args, **kwargs):
		raise NotImplementedError

	def post_eval(self, outputs):
		for inp in self.iz:
			if isinstance(inp, BlockBase):
				inp.post_eval(outputs)
		self.post_my_eval(outputs)

	# TODO Fix this disgusting design :(
	def post_my_eval(self, outputs):
		pass

	def get_all_ops(self) -> list:
		raise NotImplementedError

	def get_variables(self):
		raise NotImplementedError

	def assign_vars(self, source_block: 'BlockBase'):
		raise NotImplementedError

	def get_trainable_variables(self):
		raise NotImplementedError

	def assign_trainable_vars(self, source_block: 'BlockBase'):
		raise NotImplementedError


# TODO Shift all Tensorflow logic to TF subclass
class Function(BlockBase):
	def __init__(self, *args, **kwargs):
		super(Function, self).__init__(**kwargs)

	# self.__call__(*args, **kwargs)

	def get_all_ops(self) -> list:
		raise NotImplementedError

	def get_variables(self):
		raise NotImplementedError

	def assign_vars(self, source_block: BlockBase):
		raise NotImplementedError

	def get_trainable_variables(self):
		raise NotImplementedError

	def assign_trainable_vars(self, source_block: BlockBase):
		raise NotImplementedError


class TFFunction(Function):
	def __init__(self, *args, **kwargs):
		self.givens = {}
		self.options = None
		self.run_metadata = None
		self.built_fn = None
		super(TFFunction, self).__init__(*args, **kwargs)
		sess = self.props.__dict__.get('session', None)
		assert sess is None or isinstance(sess, tf.Session), 'Specified session must be of type tf.Session'
		self.props.__dict__['session'] = sess
		self.props.session = sess

	def __call__(self, *args, **kwargs):
		super(TFFunction, self).__call__(*args, **kwargs)
		self.built_fn = util.function(self.di, self.oz, session=self.props.session)
		return self

	def build(self, *args, **kwargs):
		raise NotImplementedError

	def set_session(self, session: tf.Session):
		self.props.session = session
		if self.built_fn is not None:
			self.built_fn.set_session(session)

	@property
	def sess(self):
		if self.built_fn is not None:
			return self.built_fn.sess
		elif self.props.session is not None:
			return self.props.session
		else:
			return U.get_session()

	def process_inputs(self, *args, **kwargs):
		super(TFFunction, self).process_inputs(*args, **kwargs)
		if not self.is_dynamic():
			self.built_fn.givens = self.get_all_givens()
			self.built_fn.using(self.options, self.run_metadata)

	def get_all_givens(self) -> dict:
		givens = {}
		for inp in self.iz:
			if isinstance(inp, TFFunction):
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
		return self.built_fn(*args, **kwargs)

	def get_all_ops(self) -> list:
		all_ops = set()
		for key in TFGraphKeys:
			collect = tf.get_collection(key, self.scope.exact_abs_pattern)
			if len(collect) > 0:
				list(map(all_ops.add, collect))  # TODO Add coverage for this line
		return list(all_ops)

	# TODO Add test case
	def get_variables(self):
		return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope.exact_abs_pattern)

	# TODO Add test case
	def assign_vars(self, source_block: 'TFFunction'):
		weight_update = [tf.assign(new, old) for (new, old) in zipsame(self.get_variables(),
																	   source_block.get_variables())]
		return weight_update

	# TODO Add test case
	def get_trainable_variables(self):
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope.exact_abs_pattern)

	# TODO Add test case
	def assign_trainable_vars(self, source_block: 'TFFunction'):
		weight_update = [tf.assign(new, old) for (new, old) in zipsame(self.get_trainable_variables(),
																	   source_block.get_trainable_variables())]
		return weight_update


class StateManager(object):
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


class StatefulTFFunction(TFFunction):
	state_manager = None  # type: StateManager

	def __init__(self, *args, **kwargs):
		self.prev_state = self.next_state = self.dynamic_state_index = self.state = None
		super(StatefulTFFunction, self).__init__(*args, **kwargs)

	def _build(self, *args, **kwargs):
		super(StatefulTFFunction, self)._build(*args, **kwargs)
		# TODO any output which is a tuple should be inferred as stateful
		state = self.o.state
		oz_index = self.oz.index(state)
		if isinstance(state, tuple):
			self.prev_state, self.state_manager, self.next_state = state
			if is_dynamic_arg(self.prev_state):
				next_state = self.next_state
				self.dynamic_state_index = oz_index
			else:
				dependencies = [self.o[key] for key in self.o if key != 'state']
				with tf.control_dependencies(dependencies):
					next_state = self.state_manager.assign(self.prev_state, self.next_state)

			self.o.state = next_state
			self.oz[oz_index] = next_state

	def build(self, *args, **kwargs):
		raise NotImplementedError

	def get_my_givens(self):
		binds = super(StatefulTFFunction, self).get_my_givens()
		if is_dynamic_arg(self.prev_state):
			binds[self.prev_state] = self.state
		return binds

	def post_my_eval(self, outputs):
		super(StatefulTFFunction, self).post_my_eval(outputs)
		if self.dynamic_state_index is not None:
			self.state = outputs[self.dynamic_state_index]


def cast_to_stateful_tf_block(ob) -> StatefulTFFunction:
	return ob


def get_class_for_block_fn(fn, base_cls, default_props: Props = None):
	class BlockFn(base_cls):
		build = fn
		props = default_props

		default_scope_name = fn.__name__

		def __init__(self, *args, **kwargs):
			self.build = fn
			props = kwargs.pop('props', Props())
			super(BlockFn, self).__init__(**props.__dict__)
			self.__call__(*args, **kwargs)

	return BlockFn


class Decorators(object):
	@staticmethod
	def function_decorator(fn) -> Type[Function]:
		return get_class_for_block_fn(fn, Function)

	@staticmethod
	def function_meta_decorator(cls: Type[Any], default_props: Props = None) -> '(fn: Any) -> Type[Function]':
		def block_decorator(fn) -> Type[Function]:
			return get_class_for_block_fn(fn, cls, default_props)

		return block_decorator

	@staticmethod
	def tf_block_decorator(fn) -> Type[TFFunction]:
		return get_class_for_block_fn(fn, TFFunction)

	@staticmethod
	def tf_block_meta_decorator(cls: Type[Any] = None, default_props: Props = None) -> '(fn: Any) -> Type[TFFunction]':
		def tf_block_decorator(fn) -> Type[TFFunction]:
			return get_class_for_block_fn(fn, cls, default_props)

		return tf_block_decorator


# noinspection PyShadowingBuiltins
def function(cls=Function) -> Type[Function]:
	is_meta_decorator = not inspect.isfunction(cls)
	return Decorators.function_meta_decorator(cls) if is_meta_decorator else Decorators.function_decorator(cls)


def tf_function(tf_function_cls=TFFunction, default_props: Props = None) -> Type[TFFunction]:
	is_meta_decorator = not inspect.isfunction(tf_function_cls)
	return Decorators.tf_block_meta_decorator(tf_function_cls, default_props) if is_meta_decorator \
		else Decorators.tf_block_decorator(tf_function_cls)


def stateful_tf_function(default_state_type: StateManager,
						 cls: Type[Any] = StatefulTFFunction,
						 default_props: Props = None) -> '(fn: Any) -> Type[StatefulTFFunction]':
	if default_state_type is not None:
		assert isinstance(default_state_type, StateManager)
	assert issubclass(cls, StatefulTFFunction)

	def stateful_tf_block_decorator(fn) -> Type[StatefulTFFunction]:
		class StatefulTFBlockFn(cls):
			build = fn
			default_scope_name = fn.__name__
			state_manager = default_state_type
			props = default_props

			def __init__(self, *args, **kwargs):
				self.build = fn
				super(StatefulTFBlockFn, self).__init__(*args, **kwargs)
				self.__call__(*args, **kwargs)

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
		weight_update = [tf.assign(new, old) for (new, old) in zipsame(self.get_trainable_variables(),
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
