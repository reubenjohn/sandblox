from unittest import TestCase

import numpy as np

from sandblox import *
from sandblox import TFMold
from sandblox.tf.tf_mold import tf_static


@tf_static(props=Props(state_manager=TFStateManager([2], np.float32)))
def add(a, b):
	next_state = b + a
	return Out.offset(a).b(next_state)


@tf_static
def reduce_mean(elems):
	return Out.mean(tf.reduce_mean(elems, axis=1))


class MapReduceMean(TFMold):
	def static(self, elems):
		mapped = tf.map_fn(self.props.func, elems)
		return Out.mean(tf.reduce_mean(mapped, axis=1))


def _evaluate_selected_mean(selected_index, mean_evaluators, add) -> Out:
	means = [mean_evaluator(add.i.a).o.mean for mean_evaluator in mean_evaluators]
	cast_indices = tf.cast(selected_index, tf.int32, 'cast_indices')
	selected_op = tf.gather(means, cast_indices, name='indexed_elements')[:, 0]
	return selected_op


@stateful_tf_static(None)
def accumulate_selected_mean(selected_index, mean_evaluators, add) -> StatefulTFBlock:
	mean = _evaluate_selected_mean(selected_index, mean_evaluators, add)
	return Out.mean(mean).b((add.i.b, add.props.state_manager, add.o.b))


@stateful_tf_static(add.props.state_manager)
def accumulate_selected_mean_with_default_state_manager(selected_index, mean_evaluators, add) -> StatefulTFBlock:
	mean = _evaluate_selected_mean(selected_index, mean_evaluators, add)
	return Out.mean(mean).b((add.i.b, add.o.b))


class Suppress(object):
	# TODO Handle default and implicit dynamic_val management
	# TODO Lifecycle that fuses dynamic_val & static graph based computing
	class TestHierarchicalBase(TestCase):
		__slots__ = 'state_tensor', 'mean_selector', 'hypo'

		def assertEqual(self, first, second, msg=None):
			first, second = U.core_op_name(first), U.core_op_name(second)
			super(Suppress.TestHierarchicalBase, self).assertEqual(first, second, msg)

		def __init__(self, methodName='runTest'):
			super().__init__(methodName)

		def setUp(self):
			super().setUp()
			self.sess = tf.Session(graph=tf.Graph())
			self.graph_context = self.sess.graph.as_default()
			self.graph_context.__enter__()
			self.sess_context = self.sess.as_default()
			self.sess_context.__enter__()

		def setUp_graph(self, state_tensor, accumulate_selected_mean_cls):
			self.state_tensor = state_tensor
			self.mean_selector = accumulate_selected_mean_cls(
				selected_index=tf.placeholder(tf.float32, [None], 'selected_index'),
				mean_evaluators=[reduce_mean, MapReduceMean(func=lambda x: x * x)],
				add=add(
					tf.placeholder(tf.float32, [None, 2], 'offset'),
					state_tensor,
					props=Props(scope_name='hypo')
				),
				props=Props(scope_name='agent')
			)
			# TODO Make sandblox handle global variable initialization
			init = tf.global_variables_initializer()
			self.sess.run(init)

		def tearDown(self):
			super().tearDown()
			self.sess_context.__exit__(None, None, None)
			self.graph_context.__exit__(None, None, None)

		def test_inputs(self):
			ai = self.mean_selector.i
			self.assertEqual(self.mean_selector.iz, [ai.selected_index, ai.mean_evaluators, ai.add])
			hi = ai.add.i
			expected_di = [ai.selected_index, hi.a]
			if is_dynamic_input(self.state_tensor):
				expected_di.append(hi.b)
			self.assertEqual(self.mean_selector.di, expected_di)

		def test_eval(self):
			agent_result = self.mean_selector.run([0], [[.1, .2]])
			selections = agent_result[0]
			self.assertTrue(
				isinstance(selections, np.ndarray) and
				selections.dtype == np.float32 and
				selections.shape == (1,)
			)
			self.assertTrue(agent_result[1].shape == (1, 2) and
							agent_result[1].dtype == np.float32)

	class TestPlaceholderStateHierarchicalBase(TestHierarchicalBase):
		__slots__ = 'state_tensor', 'mean_selector', 'hypo'

		def test_state_update(self):
			self.assertEqual(self.mean_selector.states.b.dynamic_val.shape, (1, 2))
			batch_accumulator_shape = self.mean_selector.states.b.state_manager.batch_shape(2)
			self.mean_selector.states.b.dynamic_val = np.zeros(batch_accumulator_shape)

			agent_result = self.mean_selector.run([0, 1], [[0, 2], [0, 2]])

			means = agent_result[0]
			self.assertTrue(np.array_equal(means, [1, 2]))

			new_states = agent_result[1]
			self.assertTrue(np.array_equal(new_states, [[0, 2], [0, 2]]))


class TestPlaceholderStateBlockHierarchy(Suppress.TestPlaceholderStateHierarchicalBase):
	def setUp(self):
		super().setUp()
		self.setUp_graph(add.props.state_manager.new_placeholder(), accumulate_selected_mean)


class TestVariableStateBlockHierarchy(Suppress.TestHierarchicalBase):
	def setUp(self):
		super().setUp()
		self.setUp_graph(add.props.state_manager.new_variable(), accumulate_selected_mean)


class TestPlaceholderDefaultStateManagerBlockHierarchy(Suppress.TestPlaceholderStateHierarchicalBase):
	def setUp(self):
		super().setUp()
		super().setUp_graph(add.props.state_manager.new_placeholder(),
							accumulate_selected_mean_with_default_state_manager)


class TestVariableDefaultStateManagerBlockHierarchy(Suppress.TestHierarchicalBase):
	def setUp(self):
		super().setUp()
		super().setUp_graph(add.props.state_manager.new_variable(),
							accumulate_selected_mean_with_default_state_manager)
