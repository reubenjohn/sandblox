import inspect
from typing import Type, Union, Callable

import tensorflow as tf

from sandblox.core.block import BlockBase
from sandblox.core.io import Props
from sandblox.util import tf_util as U


# TODO Shift all Tensorflow logic to TF subclass
class Function(BlockBase):
	def __init__(self, **default_props):
		self.default_props_dict = default_props
		super(Function, self).__init__(**default_props)

	# self.__call__(*args, **kwargs)

	def __call__(self, *args, **kwargs):
		props = dict(**self.default_props_dict)
		props.update(kwargs.pop('props', Props()).__dict__)
		block = type(self)(**props)
		return block.build_graph(*args, **kwargs)

	def eval(self, *args, **kwargs):
		raise NotImplementedError

	def build(self, *args, **kwargs):
		raise NotImplementedError

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
	def __init__(self, **default_props):
		self.givens = {}
		self.options = None
		self.run_metadata = None
		self.built_fn = None
		super(TFFunction, self).__init__(**default_props)
		sess = self.props.__dict__.get('session', None)
		assert sess is None or isinstance(sess, tf.Session), 'Specified session must be of type tf.Session'
		self.props.__dict__['session'] = sess
		self.props.session = sess

	def __call__(self, *args, **kwargs):
		block = super(TFFunction, self).__call__(*args, **kwargs)
		block.built_fn = U.function(block.di, block.oz, session=block.props.session)
		return block

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
		for key in U.TFGraphKeys:
			collect = tf.get_collection(key, self.scope.exact_abs_pattern)
			if len(collect) > 0:
				list(map(all_ops.add, collect))  # TODO Add coverage for this line
		return list(all_ops)

	# TODO Add test case
	def get_variables(self):
		return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope.exact_abs_pattern)

	# TODO Add test case
	def assign_vars(self, source_block: 'TFFunction'):
		weight_update = [tf.assign(new, old) for (new, old) in U.zipsame(self.get_variables(),
																		 source_block.get_variables())]
		return weight_update

	# TODO Add test case
	def get_trainable_variables(self):
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope.exact_abs_pattern)

	# TODO Add test case
	def assign_trainable_vars(self, source_block: 'TFFunction'):
		weight_update = [tf.assign(new, old) for (new, old) in U.zipsame(self.get_trainable_variables(),
																		 source_block.get_trainable_variables())]
		return weight_update


def instantiate_sandblox_function(cls: Type[Function], fn_name, def_props: Props = None):
	if def_props is None:
		def_props = Props()
	if def_props.scope_name is None:
		def_props.scope_name = fn_name
	block_fn_instance = cls(**def_props.__dict__)  # type: Type[Function]

	return block_fn_instance


def to_sandblox_function(fn: Callable, base_cls: Type[Function], def_props: Props = None):
	# noinspection PyAbstractClass
	class DecoratedFunction(base_cls):
		build = fn

		def __init__(self, **default_props):
			self.build = fn
			super(DecoratedFunction, self).__init__(**default_props)

	return instantiate_sandblox_function(DecoratedFunction, fn.__name__, def_props)


def _tf_block_decorator(fn: Callable) -> Type[TFFunction]:
	return to_sandblox_function(fn, TFFunction)


def _tf_block_meta_decorator(cls: Type[Function] = None,
							 default_props: Props = None) -> '(fn: Any) -> Type[TFFunction]':
	def tf_block_decorator(fn) -> Type[TFFunction]:
		return to_sandblox_function(fn, cls, default_props)

	return tf_block_decorator


def tf_function(fn_or_cls: Union[Callable, TFFunction] = TFFunction, default_props: Props = None) -> Type[TFFunction]:
	is_meta_decorator = not inspect.isfunction(fn_or_cls)
	return _tf_block_meta_decorator(fn_or_cls, default_props) if is_meta_decorator else _tf_block_decorator(fn_or_cls)
