import inspect
from typing import Callable, Type, Union

import tensorflow as tf

from sandblox.core.context import StaticContext
from sandblox.core.function import Props, fn_to_built_block
from sandblox.core.mold import Mold
from sandblox.util import tf_util as U


class TFStaticContext(StaticContext):
	def __init__(self, block):
		super().__init__(block)
		self.var_scope = tf.variable_scope(self.block.scope.rel, reuse=self.block.reuse_var_scope)

	def __enter__(self, *args, **kwargs):
		super().__enter__(*args, **kwargs)
		self.var_scope.__enter__()
		if len(self.block.get_all_ops(tf.get_variable_scope().name)) > 0:
			print('WARNING: Building ops into pollute d name scope')  # TODO Implement DesignViolation here

	def __exit__(self, *args, **kwargs):
		super().__exit__()
		self.var_scope.__exit__(*args, **kwargs)


class TFMold(Mold):
	def __init__(self, **default_props):
		self.givens = {}
		self.options = None
		self.run_metadata = None
		self.built_fn = None
		super(TFMold, self).__init__(**default_props)
		sess = self.props.__dict__.get('session', None)
		assert sess is None or isinstance(sess, tf.Session), 'Specified session must be of type tf.Session'
		self.props.session = sess
		# TODO Test name collision when explicitly specified names for two blocks are the same, and the lack thereof
		if default_props.get('make_scope_unique', not self.reuse_var_scope):
			graph = default_props.get('graph', tf.get_default_graph())
			assert graph is not None, 'Could not find a default graph, so a graph must be provided since make_scope_unique is True'
			self.scope.make_unique(graph)

	def __call__(self, *args, **kwargs):
		block = super(TFMold, self).__call__(*args, **kwargs)
		with block.graph.as_default():
			block.built_fn = U.function(block.di, block.oz, session=block.props.session)
		return block

	@property
	def graph(self):
		return self.sess.graph if self.sess is not None else tf.get_default_graph()

	@property
	def sess(self):
		if self.built_fn is not None:
			return self.built_fn.sess
		elif self.props.session is not None:
			return self.props.session
		else:
			return U.get_session()

	def static_context(self):
		return TFStaticContext(self)

	def build_graph(self, *args, **kwargs):
		with self.graph.as_default():
			super(TFMold, self).build_graph(*args, **kwargs)
			return self

	def build(self, *args, **kwargs):
		raise NotImplementedError

	def set_session(self, session: tf.Session):
		self.props.session = session
		if self.built_fn is not None:
			self.built_fn.set_session(session)

	def static_run(self, *args, **kwargs):
		if self.built_fn:
			self.built_fn.givens = self.get_all_givens()
			self.built_fn.using(self.options, self.run_metadata)
		return self.built_fn(*args, **kwargs)

	def dynamic_run(self, dynamic_outputs, *args, **kwargs):
		return super().dynamic_run(dynamic_outputs, *args, **kwargs)

	def get_all_givens(self) -> dict:
		givens = {}
		for inp in self.iz:
			if isinstance(inp, TFMold):
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

	def get_all_ops(self, scope_name: str = None) -> list:
		if scope_name is None:
			scope_name = self.scope.exact_abs_pattern
		all_ops = set()
		for key in U.TFGraphKeys:
			collect = tf.get_collection(key, scope_name)
			if len(collect) > 0:
				list(map(all_ops.add, collect))  # TODO Add coverage for this line
		return list(all_ops)

	# TODO Add test case
	def get_variables(self):
		return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope.exact_abs_pattern)

	# TODO Add test case
	def assign_vars(self, source_block: 'TFMold'):
		weight_update = [tf.assign(new, old) for (new, old) in U.zipsame(self.get_variables(),
																		 source_block.get_variables())]
		return weight_update

	# TODO Add test case
	def get_trainable_variables(self):
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope.exact_abs_pattern)

	# TODO Add test case
	def assign_trainable_vars(self, source_block: 'TFMold'):
		weight_update = [tf.assign(new, old) for (new, old) in U.zipsame(self.get_trainable_variables(),
																		 source_block.get_trainable_variables())]
		return weight_update


def _tf_block_decorator(fn: Callable) -> Type[TFMold]:
	return fn_to_built_block(fn, TFMold)


def _tf_block_meta_decorator(cls: Type[TFMold] = None,
							 default_props: Props = None) -> '(fn: Any) -> Type[TFMold]':
	def tf_block_decorator(fn) -> Type[TFMold]:
		return fn_to_built_block(fn, cls, default_props)

	return tf_block_decorator


def tf_block(fn_or_cls: Union[Callable, TFMold] = TFMold, default_props: Props = None) -> Type[TFMold]:
	is_meta_decorator = not inspect.isfunction(fn_or_cls)
	return _tf_block_meta_decorator(fn_or_cls, default_props) if is_meta_decorator else _tf_block_decorator(fn_or_cls)
