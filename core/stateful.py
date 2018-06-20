from typing import Type, Callable

import numpy as np

from sandblox.core.block import is_dynamic_arg
from sandblox.core.function import TFFunction, instantiate_sandblox_function
from sandblox.util import *


class StateManager(object):
	def __init__(self, shape):
		self._shape = shape

	def shape(self):
		return self._shape

	def batch_shape(self, batch_size: [int, list] = 1):
		batch_size = batch_size if isinstance(batch_size, list) else [batch_size]
		return batch_size + self.shape()

	def new(self, batch_size: [int, list] = 1):
		return np.random.uniform(-1, 1, self.batch_shape(batch_size))

	def new_placeholder(self, batch_size: [None, int, list] = None):
		return tf.placeholder(tf.float32, self.batch_shape(batch_size), 'state')

	def new_variable(self, batch_size: [int, list] = 1):
		return tf.Variable(self.new(batch_size), name='state')

	# TODO Investigate implications of making this static
	def assign(self, dest_state, src_state):
		return tf.assign(dest_state, src_state)


class StatefulTFFunction(TFFunction):
	state_manager = None  # type: StateManager

	def __init__(self, *args, **kwargs):
		self.prev_state = self.next_state = self.dynamic_state_index = self.state = None
		super(StatefulTFFunction, self).__init__(**kwargs)

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


def to_stateful_sandblox_function(fn: Callable, default_state_manager: StateManager, base_cls: Type[StatefulTFFunction],
								  def_props: Props) -> Type[StatefulTFFunction]:
	# noinspection PyAbstractClass
	class StatefulDecoratedFunction(base_cls):
		build = fn
		state_manager = default_state_manager

		def __init__(self, **props):
			self.build = fn
			super(StatefulDecoratedFunction, self).__init__(**props)

	return instantiate_sandblox_function(StatefulDecoratedFunction, fn.__name__, def_props)


def stateful_tf_function(default_state_type: StateManager,
						 cls: Type[StatefulTFFunction] = StatefulTFFunction,
						 default_props: Props = None) -> '(fn: Any) -> Type[StatefulTFFunction]':
	if default_state_type is not None:
		assert isinstance(default_state_type, StateManager)

	def stateful_tf_block_decorator(fn: Callable) -> StatefulTFFunction:
		return to_stateful_sandblox_function(fn, default_state_type, cls, default_props)

	return stateful_tf_block_decorator
