from typing import Type, Callable, List

import numpy as np

from sandblox import errors
from sandblox.core.function import instantiate_block
from sandblox.core.io import *
from sandblox.core.io import BlockOutsBase
from sandblox.tf.tf_mold import TFMold


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
	def __init__(self, shape: List, dtype=np.float32, *args, **kwargs):
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


class State(object):
	def __init__(self, prev, state_manager: StateManager, next):
		self.prev, self.state_manager, self.next = prev, state_manager, next

	@property
	def static_val(self):
		return self.prev


class DynamicStateBinder(State):
	def __init__(self, output_index, prev, state_manager: StateManager, next):
		super(DynamicStateBinder, self).__init__(prev, state_manager, next)
		self.__sandblox_dynamic__ = True
		self.dynamic_output_index = output_index
		self.dynamic_val = state_manager.new()


class StatefulTFMold(TFMold):
	state_manager = None  # type: TFStateManager

	def __init__(self, **props):
		self.states = DictAttrs()
		super(StatefulTFMold, self).__init__(**props)

	def _wrap_static(self, *args, **kwargs) -> BlockOutsBase:
		out = super(StatefulTFMold, self)._wrap_static(*args, **kwargs)

		for index, key in enumerate(out.o):
			output = out.o[key]
			if isinstance(output, tuple):
				if len(output) == 3:
					prev_op, state_manager, next_op = output
				elif len(output) == 2:
					prev_op, next_op = output
					assert self.state_manager is not None
					state_manager = self.state_manager
				else:
					raise ValueError(
						'Observed static output is a tuple. It must thus be of form (prev_op, state_manager, next_op) '
						'or (prev_op, next_op) if a default state manager has been specified in the props')

				out.__getattr__(key)(next_op)
				if is_dynamic_input(prev_op):
					self.states[key] = DynamicStateBinder(index, prev_op, state_manager, next_op)
				else:
					self.states[key] = State(prev_op, state_manager, next_op)
					dependencies = out.oz
					with tf.control_dependencies(dependencies):
						next_op = state_manager.assign(prev_op, next_op)

				out.__getattr__(key)(next_op)
		return out

	def static(self, *args, **kwargs):
		return Out

	def compute_is_dynamic(self):
		return len(self.dynamic_states) > 0 or super().compute_is_dynamic()

	def dynamic(self, *args, **kwargs):
		if self.is_dynamic:
			return self._static_run(*args, **kwargs)

	@property
	def dynamic_states(self) -> List[DynamicStateBinder]:
		return [state for state in [self.states[key] for key in self.states] if is_dynamic_input(state)]

	def self_givens(self):
		binds = super(StatefulTFMold, self).self_givens()
		for dynamic_state_binder in self.dynamic_states:
			binds[dynamic_state_binder.prev] = dynamic_state_binder.dynamic_val
		return binds

	def _post_dynamic(self, dynamic_oz):
		if self.is_dynamic:
			for dynamic_state_binder in self.dynamic_states:
				dynamic_state_binder.dynamic_val = dynamic_oz[dynamic_state_binder.dynamic_output_index]
		else:
			raise errors.BlockNotDynamicError(self)


def to_stateful_sandblox_function(fn: Callable, default_state_manager: TFStateManager,
								  base_cls: Type[StatefulTFMold],
								  def_props: Props) -> Type[StatefulTFMold]:
	# noinspection PyAbstractClass
	class StatefulDecoratedFunction(base_cls):
		static = fn
		state_manager = default_state_manager

		def __init__(self, **props):
			self.static = fn
			super(StatefulDecoratedFunction, self).__init__(**props)

	return instantiate_block(StatefulDecoratedFunction, fn.__name__, def_props)


def stateful_tf_static(default_state_type: TFStateManager,
					   cls: Type[StatefulTFMold] = StatefulTFMold,
					   default_props: Props = None) -> '(fn: Any) -> Type[StatefulTFMold]':
	if default_state_type is not None:
		assert isinstance(default_state_type, TFStateManager)

	def stateful_tf_block_decorator(fn: Callable) -> StatefulTFMold:
		return to_stateful_sandblox_function(fn, default_state_type, cls, default_props)

	return stateful_tf_block_decorator
