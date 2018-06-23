from contextlib import contextmanager
from unittest import TestCase

from sandblox import *


# TODO Handle default and implicit state management
# TODO Lifecycle that fuses dynamic & static graph based computing
class Supress(object):
	class TestHierarchicalBase(TestCase):
		__slots__ = 'state_tensor', 'agnt', 'hypo'

		def setUp(self):
			super(Supress.TestHierarchicalBase, self).setUp()
			self.sess = tf.Session(graph=tf.Graph())
			with self.sess.graph.as_default():
				self.setUp_graph(*self.get_graph_params())

		def setUp_graph(self, state_tensor, agent_cls):
			self.state_tensor = state_tensor
			self.agnt = agent_cls(
				tf.placeholder(tf.float32, (), 'selected_index'),
				[greedy, KGreedy(k=2)],
				dense_hypothesis(
					tf.placeholder(tf.float32, [None, 2], 'ob'),
					state_tensor,
					props=Props(scope_name='hypo')
				),
				props=Props(scope_name='agent')
			)
			if is_dynamic_arg(self.state_tensor):
				self.agnt.state = dense_hypothesis.props.state_manager.new()

		@contextmanager
		def ctx(self):
			with self.sess.graph.as_default():
				init = tf.global_variables_initializer()
			with self.sess.as_default() as ctx:
				self.sess.run(init)
				yield ctx

		def test_inputs(self):
			ai = self.agnt.i
			self.assertEqual(self.agnt.iz, [ai.selected_index, ai.selectors, ai.hypothesis])
			hi = ai.hypothesis
			expected_di = [ai.selected_index, hi.i.ob]
			if is_dynamic_arg(self.state_tensor):
				expected_di.append(hi.i.state)
			self.assertEqual(self.agnt.di, expected_di)

		def test_eval(self):
			with self.ctx():
				agent_result = self.agnt.run(0, [[.1, .2]])
				selections = agent_result[0]
				self.assertTrue(
					isinstance(selections, np.ndarray) and
					selections.dtype == np.int32 and
					selections.shape == (1, 2)
				)
				self.assertTrue(agent_result[1].shape == (1, 4) and
								agent_result[1].dtype == np.float64)

		def get_graph_params(self):
			raise NotImplementedError

	class TestPlaceholderStateHierarchicalBase(TestHierarchicalBase):
		__slots__ = 'state_tensor', 'agnt', 'hypo'

		def get_graph_params(self):
			raise NotImplementedError

		def test_state_update(self):
			# TODO Make sandblox handle global variable initialization
			with self.ctx():
				old_state = self.agnt.state
				agent_result = self.agnt.run(0, [[.1, .2]])
				self.assertEqual(agent_result[1].shape, (1, 4))
				# State should be updated: Small possibility of a false negative here
				self.assertTrue(not np.alltrue(np.equal(self.agnt.state, old_state)))


@tf_function(default_props=Props(state_manager=TFStateManager([4], np.float64)))
def dense_hypothesis(ob, state):
	logits = tf.layers.dense(ob, 2)
	next_state = tf.layers.dense(state, 4, name='next_state')
	return Out.logits(logits).state(next_state)


class KGreedy(TFFunction):
	def build(self, logits):
		return Out.action(tf.nn.top_k(logits, self.props.k)[1])


@tf_function
def greedy(logits):
	argmax = tf.expand_dims(tf.argmax(logits, axis=1, output_type=tf.int32), axis=0)
	return Out.action(tf.concat([argmax, argmax], axis=1))  # Just to have the dimentions match with k_greedy = 2


def agent_logic(selected_index, selectors, hypothesis) -> Out:
	actions = [selector(hypothesis.o.logits).o.action for selector in selectors]
	selected_action_op = tf.gather(
		actions,
		tf.cast(selected_index, tf.int32, "action_selected_index"),
		name="action"
	)
	return Out.action(selected_action_op).state(
		(hypothesis.i.state, hypothesis.props.state_manager, hypothesis.o.state))


@stateful_tf_function(None)
def agent(selected_index, selectors, hypothesis) -> StatefulTFFunction:
	return agent_logic(selected_index, selectors, hypothesis)


class TestPlaceholderStateBlockHierarchy(Supress.TestPlaceholderStateHierarchicalBase):
	def get_graph_params(self):
		return dense_hypothesis.props.state_manager.new_placeholder(), agent


class TestVariableStateBlockHierarchy(Supress.TestHierarchicalBase):
	def get_graph_params(self):
		return dense_hypothesis.props.state_manager.new_variable(), agent


@stateful_tf_function(dense_hypothesis.props.state_manager)
def default_state_manager_agent(selected_index, selectors, hypothesis) -> StatefulTFFunction:
	return agent_logic(selected_index, selectors, hypothesis)


class TestPlaceholderDefaultStateManagerBlockHierarchy(Supress.TestPlaceholderStateHierarchicalBase):
	def get_graph_params(self):
		return dense_hypothesis.props.state_manager.new_placeholder(), default_state_manager_agent


class TestVariableDefaultStateManagerBlockHierarchy(Supress.TestHierarchicalBase):
	def get_graph_params(self):
		return dense_hypothesis.props.state_manager.new_variable(), default_state_manager_agent
