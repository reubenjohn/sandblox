from contextlib import contextmanager
from unittest import TestCase

import numpy as np

from sandblox import *
from sandblox import TFMold
from sandblox.tf.tf_function import tf_block


class Suppress(object):
	# TODO Handle default and implicit dynamic_val management
	# TODO Lifecycle that fuses dynamic_val & static graph based computing

	class TestHierarchicalBase(TestCase):
		__slots__ = 'state_tensor', 'mean_selector', 'hypo'

		def assertEqual(self, first, second, msg=None):
			first, second = U.core_op_name(first), U.core_op_name(second)
			super(Suppress.TestHierarchicalBase, self).assertEqual(first, second, msg)

		def setUp(self):
			super(Suppress.TestHierarchicalBase, self).setUp()
			self.sess = tf.Session(graph=tf.Graph())
			with self.sess.graph.as_default():
				self.setUp_graph(*self.get_graph_params())

		def setUp_graph(self, state_tensor, accumulate_selected_mean_cls):
			self.state_tensor = state_tensor
			self.mean_selector = accumulate_selected_mean_cls(
				selected_index=tf.placeholder(tf.float32, [None], 'selected_index'),
				mean_evaluators=[MeanEvaluators.reduce_mean, MeanEvaluators.MapReduceMean(func=lambda x: x * x)],
				stateful_input=increase_input(
					tf.placeholder(tf.float32, [None, 2], 'offset'),
					state_tensor,
					props=Props(scope_name='hypo')
				),
				props=Props(scope_name='agent')
			)

		@contextmanager
		def ctx(self):
			with self.sess.graph.as_default():
				init = tf.global_variables_initializer()
			with self.sess.as_default() as ctx:
				self.sess.run(init)
				yield ctx

		def test_inputs(self):
			ai = self.mean_selector.i
			self.assertEqual(self.mean_selector.iz, [ai.selected_index, ai.mean_evaluators, ai.stateful_input])
			hi = ai.stateful_input.i
			expected_di = [ai.selected_index, hi.offset]
			if is_dynamic_input(self.state_tensor):
				expected_di.append(hi.accumulation)
			self.assertEqual(self.mean_selector.di, expected_di)

		def test_eval(self):
			with self.ctx():
				agent_result = self.mean_selector.run([0], [[.1, .2]])
				selections = agent_result[0]
				self.assertTrue(
					isinstance(selections, np.ndarray) and
					selections.dtype == np.float32 and
					selections.shape == (1,)
				)
				self.assertTrue(agent_result[1].shape == (1, 2) and
								agent_result[1].dtype == np.float32)

		def get_graph_params(self):
			raise NotImplementedError

	class TestPlaceholderStateHierarchicalBase(TestHierarchicalBase):
		__slots__ = 'state_tensor', 'mean_selector', 'hypo'

		def get_graph_params(self):
			raise NotImplementedError

		def test_state_update(self):
			# TODO Make sandblox handle global variable initialization
			with self.ctx():
				self.assertEqual(self.mean_selector.states.accumulation.dynamic_val.shape, (1, 2))
				batch_accumulator_shape = self.mean_selector.states.accumulation.state_manager.batch_shape(2)
				self.mean_selector.states.accumulation.dynamic_val = np.zeros(batch_accumulator_shape)
				agent_result = self.mean_selector.run([0, 1], [[0, 2], [0, 2]])
				means = agent_result[0]
				new_states = agent_result[1]
				self.assertTrue(np.array_equal(new_states, [[0, 2], [0, 2]]))
				self.assertTrue(np.array_equal(means, [1, 2]))


@tf_block(props=Props(state_manager=TFStateManager([2], np.float32)))
def increase_input(offset, accumulation):
	next_state = accumulation + offset
	return Out.offset(offset).accumulation(next_state)


class MeanEvaluators:
	@staticmethod
	@tf_block
	def reduce_mean(elems):
		return Out.mean(tf.reduce_mean(elems, axis=1))

	class MapReduceMean(TFMold):
		def build(self, elems):
			mapped = tf.map_fn(self.props.func, elems)
			return Out.mean(tf.reduce_mean(mapped, axis=1))


def evaluate_selected_mean(selected_index, mean_evaluators, stateful_input) -> Out:
	means = [mean_evaluator(stateful_input.o.offset).o.mean for mean_evaluator in mean_evaluators]
	cast_indices = tf.cast(selected_index, tf.int32, 'cast_indices')
	selected_op = tf.gather(means, cast_indices, name='indexed_elements')[:, 0]
	return selected_op


@stateful_tf_function(None)
def accumulate_selected_mean(selected_index, mean_evaluators, stateful_input) -> StatefulTFBlock:
	mean = evaluate_selected_mean(selected_index, mean_evaluators, stateful_input)
	return Out.mean(mean).accumulation(
		(stateful_input.i.accumulation, stateful_input.props.state_manager, stateful_input.o.accumulation))


class TestPlaceholderStateBlockHierarchy(Suppress.TestPlaceholderStateHierarchicalBase):
	def get_graph_params(self):
		return increase_input.props.state_manager.new_placeholder(), accumulate_selected_mean


class TestVariableStateBlockHierarchy(Suppress.TestHierarchicalBase):
	def get_graph_params(self):
		return increase_input.props.state_manager.new_variable(), accumulate_selected_mean


@stateful_tf_function(increase_input.props.state_manager)
def accumulate_selected_mean_with_default_state_manager(selected_index, mean_evaluators,
														stateful_input) -> StatefulTFBlock:
	mean = evaluate_selected_mean(selected_index, mean_evaluators, stateful_input)
	return Out.mean(mean).accumulation((stateful_input.i.accumulation, stateful_input.o.accumulation))


class TestPlaceholderDefaultStateManagerBlockHierarchy(Suppress.TestPlaceholderStateHierarchicalBase):
	def get_graph_params(self):
		return increase_input.props.state_manager.new_placeholder(), accumulate_selected_mean_with_default_state_manager


class TestVariableDefaultStateManagerBlockHierarchy(Suppress.TestHierarchicalBase):
	def get_graph_params(self):
		return increase_input.props.state_manager.new_variable(), accumulate_selected_mean_with_default_state_manager
