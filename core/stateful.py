from typing import Type, Callable

import numpy as np

from sandblox.core.function import TFFunction, instantiate_sandblox_function
from sandblox.core.io import *
from sandblox.core.io import BlockOutsBase
from sandblox.util import *


class StateManager(object):
	def __init__(self, *args, **kwargs):
		pass

	def batch_shape(self, batch_size: [int, list] = 1):
		raise NotImplementedError

	def new(self, batch_size: [int, list] = 1):
		raise NotImplementedError

	def new_placeholder(self, batch_size: [None, int, list] = None):
		raise NotImplementedError

	def new_variable(self, batch_size: [int, list] = 1):
		raise NotImplementedError

	def assign(self, dest_state, src_state):
		raise NotImplementedError


class TFStateManager(StateManager):
	def __init__(self, shape, dtype=np.float32, *args, **kwargs):
		super(TFStateManager, self).__init__(*args, **kwargs)
		self._shape = shape
		self._dtype = dtype

	def shape(self):
		return self._shape

	def batch_shape(self, batch_size: [int, list] = 1):
		batch_size = batch_size if isinstance(batch_size, list) else [batch_size]
		return batch_size + self.shape()

	def new(self, batch_size: [int, list] = 1):
		return np.random.uniform(-1, 1, self.batch_shape(batch_size))

	def new_placeholder(self, batch_size: [None, int, list] = None):
		return tf.placeholder(tf.as_dtype(self._dtype), self.batch_shape(batch_size), 'state')

	def new_variable(self, batch_size: [int, list] = 1):
		return tf.Variable(self.new(batch_size), dtype=tf.as_dtype(self._dtype), name='state')

	# TODO Investigate implications of making this static
	def assign(self, dest_state, src_state):
		return tf.assign(dest_state, src_state)


class StatefulTFFunction(TFFunction):
	state_manager = None  # type: TFStateManager

	def __init__(self, **props):
		self.prev_state = self.next_state = self.dynamic_state_index = self.state = None
		super(StatefulTFFunction, self).__init__(**props)

	def build_wrapper(self, *args, **kwargs) -> BlockOutsBase:
		out = super(StatefulTFFunction, self).build_wrapper(*args, **kwargs)
		# TODO any output which is a tuple should be inferred as stateful
		state = out.o.state
		oz_index = out.oz.index(state)
		if isinstance(state, tuple):
			self.prev_state, self.state_manager, self.next_state = state
			if is_dynamic_arg(self.prev_state):
				next_state = self.next_state
				self.dynamic_state_index = oz_index
			else:
				dependencies = [out.o[key] for key in out.o if key != 'state']
				with tf.control_dependencies(dependencies):
					next_state = self.state_manager.assign(self.prev_state, self.next_state)

			out.state(next_state)
		return out

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


def to_stateful_sandblox_function(fn: Callable, default_state_manager: TFStateManager,
								  base_cls: Type[StatefulTFFunction],
								  def_props: Props) -> Type[StatefulTFFunction]:
	# noinspection PyAbstractClass
	class StatefulDecoratedFunction(base_cls):
		build = fn
		state_manager = default_state_manager

		def __init__(self, **props):
			self.build = fn
			super(StatefulDecoratedFunction, self).__init__(**props)

	return instantiate_sandblox_function(StatefulDecoratedFunction, fn.__name__, def_props)


def stateful_tf_function(default_state_type: TFStateManager,
						 cls: Type[StatefulTFFunction] = StatefulTFFunction,
						 default_props: Props = None) -> '(fn: Any) -> Type[StatefulTFFunction]':
	if default_state_type is not None:
		assert isinstance(default_state_type, TFStateManager)

	def stateful_tf_block_decorator(fn: Callable) -> StatefulTFFunction:
		return to_stateful_sandblox_function(fn, default_state_type, cls, default_props)

	return stateful_tf_block_decorator
