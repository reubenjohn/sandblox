from unittest import TestCase

from sandblox import *


# TODO Handle default and implicit state management
# TODO Lifecycle that fuses dynamic & static graph based computing
class Suppress2(object):
	class TestHierarchicalBase(TestCase):
		__slots__ = 'state_tensor', 'agnt', 'hypo'

		def __init__(self, method_name: str = 'runTest'):
			super(Suppress2.TestHierarchicalBase, self).__init__(method_name)

		def test_inputs(self):
			ai = self.agnt.i
			self.assertEqual(self.agnt.iz, [ai.selected_index, ai.selectors, ai.hypothesis])
			hi = ai.hypothesis.i
			expected_di = [ai.selected_index, hi.ob]
			if is_dynamic_arg(self.state_tensor):
				expected_di.append(hi.state)
			self.assertEqual(self.agnt.di, expected_di)

		def test_eval(self):
			with tf.Session() as sess:
				sess.run(tf.global_variables_initializer())
				agent_result = self.agnt.run(0, [[.1, .2]])
				selections = agent_result[0]
				self.assertTrue(
					isinstance(selections, numpy.ndarray) and
					selections.dtype == numpy.int32 and selections.shape == (1, 2)
				)
				self.assertEqual(agent_result[1].shape, (1, 4))


@tf_function(default_props=Props(state_manager=StateManager([4])))
def dense_hypothesis(ob, state):
	logits = tf.layers.dense(ob, 2)
	next_state = tf.layers.dense(state, 4)
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


def build_hypothesis(state_tensor, scope_name):
	return dense_hypothesis(tf.placeholder(tf.float32, [None, 2], 'ob'), state_tensor,
							props=Props(scope_name=scope_name))


def build_agent(agent_cls, hypothesis):
	return agent_cls(
		tf.placeholder(tf.float32, (), 'selected_index'),
		[greedy, KGreedy(k=2)],
		hypothesis
	)


class TestPlaceholderStateHierarchicalBlock(Suppress2.TestHierarchicalBase):
	state_tensor = dense_hypothesis.props.state_manager.new_placeholder()
	hypo = build_hypothesis(state_tensor, 'placeholder_hypothesis')
	agnt = build_agent(agent, hypo)
	agnt.state = dense_hypothesis.props.state_manager.new()

	def test_state_update(self):
		with tf.Session() as sess:
			# TODO Make sandblox handle global variable initialization
			sess.run(tf.global_variables_initializer())
			old_state = self.agnt.state
			agent_result = self.agnt.run(0, [[.1, .2]])
			self.assertEqual(agent_result[1].shape, (1, 4))
			# State should be updated: Small possibility of a false failure here
			self.assertTrue(
				not numpy.alltrue(numpy.equal(self.agnt.state, old_state)))


class TestVariableStateHierarchicalBlock(Suppress2.TestHierarchicalBase):
	state_tensor = dense_hypothesis.props.state_manager.new_variable()
	hypo = build_hypothesis(state_tensor, 'variable_hypothesis2')
	agnt = build_agent(agent, hypo)


@stateful_tf_function(dense_hypothesis.props.state_manager)
def default_state_manager_agent(selected_index, selectors, hypothesis) -> StatefulTFFunction:
	return agent_logic(selected_index, selectors, hypothesis)


class TestPlaceholderDefaultStateManagerHierarchicalBlock(Suppress2.TestHierarchicalBase):
	state_tensor = dense_hypothesis.props.state_manager.new_placeholder()
	hypo = build_hypothesis(state_tensor, 'placeholder_hypothesis2')
	agnt = build_agent(default_state_manager_agent, hypo)
	agnt.state = dense_hypothesis.props.state_manager.new()

	def test_state_update(self):
		with tf.Session() as sess:
			# TODO Make sandblox handle global variable initialization behind the scenes
			sess.run(tf.global_variables_initializer())
			old_state = self.agnt.state
			agent_result = self.agnt.run(0, [[.1, .2]])
			self.assertEqual(agent_result[1].shape, (1, 4))
			# State should be updated: Small possibility of a false failure here
			self.assertTrue(
				not numpy.alltrue(numpy.equal(self.agnt.state, old_state)))


class TestVariableDefaultStateManagerHierarchicalBlock(Suppress2.TestHierarchicalBase):
	state_tensor = dense_hypothesis.props.state_manager.new_variable()
	hypo = build_hypothesis(state_tensor, 'variable_hypothesis')
	agnt = build_agent(default_state_manager_agent, hypo)
