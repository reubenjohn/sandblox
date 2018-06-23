from contextlib import contextmanager
from unittest import TestCase

from sandblox import *


# TODO Handle default and implicit dynamic_val management
# TODO Lifecycle that fuses dynamic_val & static graph based computing
class Supress(object):
	class TestHierarchicalBase(TestCase):
		__slots__ = 'state_tensor', 'mean_selector', 'hypo'

		def setUp(self):
			super(Supress.TestHierarchicalBase, self).setUp()
			self.sess = tf.Session(graph=tf.Graph())
			with self.sess.graph.as_default():
				self.setUp_graph(*self.get_graph_params())

		def setUp_graph(self, state_tensor, accumulator_mean_selector_cls):
			self.state_tensor = state_tensor
			self.mean_selector = accumulator_mean_selector_cls(
				tf.placeholder(tf.float32, [None], 'selected_index'),
				[reduce_mean, MapReduceMean(func=lambda x: x * x)],
				offset_accumulator(
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
			self.assertEqual(self.mean_selector.iz, [ai.selected_index, ai.mean_evaluators, ai.hypothesis])
			hi = ai.hypothesis.i
			expected_di = [ai.selected_index, hi.offset]
			if is_dynamic_arg(self.state_tensor):
				expected_di.append(hi.accumulator)
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
				self.assertEqual(self.mean_selector.states.accumulator.dynamic_val.shape, (1, 2))
				batch_accumulator_shape = self.mean_selector.states.accumulator.state_manager.batch_shape(2)
				self.mean_selector.states.accumulator.dynamic_val = np.zeros(batch_accumulator_shape)
				agent_result = self.mean_selector.run([0, 1], [[0, 2], [0, 2]])
				means = agent_result[0]
				new_states = agent_result[1]
				self.assertTrue(np.array_equal(new_states, [[0, 2], [0, 2]]))
				self.assertTrue(np.array_equal(means, [1, 2]))


@tf_function(default_props=Props(state_manager=TFStateManager([2], np.float32)))
def offset_accumulator(offset, accumulator):
	next_state = accumulator + offset
	return Out.offset(offset).accumulator(next_state)


@tf_function
def reduce_mean(elems):
	return Out.mean(tf.reduce_mean(elems, axis=1))  # Just to have the dimensions match with k_greedy = 2


class MapReduceMean(TFFunction):
	def build(self, elems):
		mapped = tf.map_fn(self.props.func, elems)
		return Out.mean(tf.reduce_mean(mapped, axis=1))


def select_and_evaluate_mean(selected_index, mean_evaluators, hypothesis) -> Out:
	means = [selector(hypothesis.o.offset).o.mean for selector in mean_evaluators]
	cast_indices = tf.cast(selected_index, tf.int32, 'cast_indices')
	selected_op = tf.gather(means, cast_indices, name='indexed_elements')[:, 0]
	return selected_op


@stateful_tf_function(None)
def accumulator_mean_selector(selected_index, mean_evaluators, hypothesis) -> StatefulTFFunction:
	selected_op = select_and_evaluate_mean(selected_index, mean_evaluators, hypothesis)
	return Out.mean(selected_op).accumulator(
		(hypothesis.i.accumulator, hypothesis.props.state_manager, hypothesis.o.accumulator))


class TestPlaceholderStateBlockHierarchy(Supress.TestPlaceholderStateHierarchicalBase):
	def get_graph_params(self):
		return offset_accumulator.props.state_manager.new_placeholder(), accumulator_mean_selector


class TestVariableStateBlockHierarchy(Supress.TestHierarchicalBase):
	def get_graph_params(self):
		return offset_accumulator.props.state_manager.new_variable(), accumulator_mean_selector


@stateful_tf_function(offset_accumulator.props.state_manager)
def default_state_accumulator_mean_selector(selected_index, mean_evaluators, hypothesis) -> StatefulTFFunction:
	selected_op = select_and_evaluate_mean(selected_index, mean_evaluators, hypothesis)
	return Out.mean(selected_op).accumulator((hypothesis.i.accumulator, hypothesis.o.accumulator))


class TestPlaceholderDefaultStateManagerBlockHierarchy(Supress.TestPlaceholderStateHierarchicalBase):
	def get_graph_params(self):
		return offset_accumulator.props.state_manager.new_placeholder(), default_state_accumulator_mean_selector


class TestVariableDefaultStateManagerBlockHierarchy(Supress.TestHierarchicalBase):
	def get_graph_params(self):
		return offset_accumulator.props.state_manager.new_variable(), default_state_accumulator_mean_selector
