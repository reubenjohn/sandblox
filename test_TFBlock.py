import time
from typing import Type
from unittest import TestCase

import numpy
import tensorflow as tf

import sandblox as sx
from sandblox import Out


class FooLogic(object):
	call_cache = dict()
	di = [tf.placeholder(tf.float32, (), 'y')]
	args = [tf.ones((), tf.float32), di[0]]
	kwargs = dict(extra=10)

	@staticmethod
	def args_call(fn, **expansion):
		expansion.update(FooLogic.kwargs)
		return fn(*FooLogic.args, **expansion)

	@staticmethod
	def internal_args_call(fn, **expansion):
		expansion.update(FooLogic.kwargs)
		return fn(0, *FooLogic.args, **expansion)

	cached_args_call = lambda fn: FooLogic.cached_call(fn, *FooLogic.args, **FooLogic.kwargs)

	@staticmethod
	def call(x, y, param_with_default=-5, **kwargs):
		return x + y + kwargs['extra'] + param_with_default, tf.random_uniform(())

	@staticmethod
	def cached_call(fn, *args, **kwargs):
		bound_args = sx.util.FlatArgumentsBinder(fn)(*args, **kwargs)
		key = fn.__name__ + ':' + str(bound_args)
		if key not in FooLogic.call_cache:
			FooLogic.call_cache[key] = fn(*args, **kwargs)
		return FooLogic.call_cache[key]


@sx.tf_function
def foo(x, y, param_with_default=-5, **kwargs):
	# noinspection PyTypeChecker
	b, a = FooLogic.cached_call(FooLogic.call, x, y, param_with_default, **kwargs)

	if Out == sx.BlockOutsKwargs:
		return Out(b=b, a=a)
	else:
		return Out.b(b).a(a)


@sx.tf_function
def bad_foo(x, y, param_with_default=-5, **kwargs):
	# noinspection PyTypeChecker
	b, a = FooLogic.cached_call(FooLogic.call, x, y, param_with_default, **kwargs)
	return b, a


# noinspection PyClassHasNoInit
class Foo(sx.TFFunction):
	def build(self, x, y, param_with_default=-5, **kwargs):
		# noinspection PyTypeChecker
		b, a = FooLogic.cached_call(FooLogic.call, x, y, param_with_default, **kwargs)

		# TODO Test both cases for python 3.6+
		if Out == sx.BlockOutsKwargs:
			return Out(b=b, a=a)
		else:
			return Out.b(b).a(a)


# noinspection PyClassHasNoInit
class BadFoo(sx.TFFunction):
	def build(self, x, y, param_with_default=-5, **kwargs):
		# noinspection PyTypeChecker
		b, a = FooLogic.cached_call(FooLogic.call, x, y, param_with_default, **kwargs)

		return b, a


class FooWithProps(Foo):
	def __init__(self, *args, **kwargs):
		super(Foo, self).__init__(*args, **kwargs)
		assert self.props.my_prop == 0


class BadFooWithProps(BadFoo):
	def __init__(self, *args, **kwargs):
		super(BadFoo, self).__init__(*args, **kwargs)
		assert self.props.my_prop == 0


class Suppress1(object):
	# Wrapped classes don't get tested themselves
	# noinspection PyCallByClass
	class TestBlockBase(TestCase):
		target = None  # type: Type[sx.TFFunction]
		bad_target = None  # type: Type[sx.TFFunction]
		block_foo_ob = None  # type: sx.TFFunction

		def create_block_ob(self, **props) -> sx.TFFunction:
			raise NotImplementedError

		ELAPSE_LIMIT = 25000  # usec To accommodate slowness during debugging
		ELAPSE_TARGET = 2500  # usec

		def __init__(self, method_name: str = 'runTest'):
			super(Suppress1.TestBlockBase, self).__init__(method_name)
			# TODO Use variable instead
			self.bound_flattened_logic_arguments = FooLogic.args_call(sx.util.FlatArgumentsBinder(FooLogic.call))
			self.logic_outputs = list(FooLogic.cached_args_call(FooLogic.call))

			self.options = tf.RunOptions()
			self.options.output_partition_graphs = True
			self.options.trace_level = tf.RunOptions.FULL_TRACE

		def setUp(self):
			super(Suppress1.TestBlockBase, self).setUp()

		def test_block_inputs(self):
			self.assertEqual(self.block_foo_ob.i.__dict__, self.bound_flattened_logic_arguments)

		def test_block_dynamic_inputs(self):
			self.assertEqual(self.block_foo_ob.di, FooLogic.di)

		def test_block_out(self):
			self.assertEqual(self.block_foo_ob.o.a, self.logic_outputs[1])
			self.assertEqual(self.block_foo_ob.o.b, self.logic_outputs[0])

		def test_block_out_order(self):
			self.assertEqual(self.block_foo_ob.oz, self.logic_outputs)

		def test_eval(self):
			with tf.Session():
				eval_100 = self.block_foo_ob.run(100)

				metadata = tf.RunMetadata()
				eval_0 = self.block_foo_ob.using(self.options, metadata).run(0)
				self.assertTrue(hasattr(metadata, 'partition_graphs') and len(metadata.partition_graphs) > 0)

				self.assertEqual(eval_100[0], eval_0[0] + 100)
				self.assertNotEqual(eval_100[1], eval_0[1])  # Boy aren't you unlucky if you fail this test XD

		def test_bad_foo_assertion(self):
			with self.assertRaises(AssertionError) as bad_foo_context:
				FooLogic.args_call(self.bad_target)
			self.assertTrue('must either return' in str(bad_foo_context.exception))

		def test_overhead(self):
			# TODO Measure relative execution time of empty eval block to full eval block instead of absolute time
			self.block_foo_ob.eval = lambda *args: ()
			built_fn = self.block_foo_ob.built_fn
			self.block_foo_ob.built_fn = None
			with tf.Session():
				then = time.time()
				[self.block_foo_ob.run(100) for _ in range(1000)]
				elapse = int((time.time() - then) * 1e9 / 1000)
				if elapse > Suppress1.TestBlockBase.ELAPSE_LIMIT:
					overdue_percentage = \
						(Suppress1.TestBlockBase.ELAPSE_LIMIT - elapse) * 100 / Suppress1.TestBlockBase.ELAPSE_LIMIT
					self.fail('Overdue by %.1f%% (%3dns elapsed)' % (overdue_percentage, elapse))
				else:
					goal_progress = 100 + ((Suppress1.TestBlockBase.ELAPSE_TARGET - elapse) * 100 / elapse)
					print('Efficiency goal progress %.1f%% (%3dns elapsed) - %s' % (
						goal_progress, elapse, type(self).__name__))

			self.block_foo_ob.built_fn = built_fn

		# TODO Test scope_name
		def test_session_specification(self):
			sess = tf.Session()
			with tf.Session():
				block = self.create_block_ob(session=sess)
				self.assertEqual(block.sess, sess)
				block.set_session(tf.Session())
				self.assertNotEqual(block.sess, sess)
				with self.assertRaises(AssertionError) as bad_foo_context:
					self.create_block_ob(session='some_invalid_session')
				self.assertTrue('must be of type tf.Session' in str(bad_foo_context.exception))


class TestBlockFunction(Suppress1.TestBlockBase):
	target = foo
	bad_target = bad_foo
	block_foo_ob = FooLogic.args_call(target)

	def create_block_ob(self, **props):
		return FooLogic.args_call(TestBlockFunction.target, props=sx.Props(**props))

	def __init__(self, method_name: str = 'runTest'):
		super(TestBlockFunction, self).__init__(method_name)


class TestBlockClass(Suppress1.TestBlockBase):
	target = Foo
	bad_target = BadFoo()
	block_foo_ob = FooLogic.args_call(target())

	def create_block_ob(self, **props):
		return TestBlockClass.target(**props)

	def setUp(self):
		super(TestBlockClass, self).setUp()


# noinspection PyCallByClass
class TestBlockClassWithProps(Suppress1.TestBlockBase):
	target = FooWithProps
	bad_target = BadFooWithProps(my_prop=0)
	block_foo_ob = FooLogic.args_call(target(my_prop=0))

	def create_block_ob(self, **props):
		return TestBlockClass.target(**props)

	def setUp(self):
		super(TestBlockClassWithProps, self).setUp()

	def test_bad_foo_assertion(self):
		with self.assertRaises(AssertionError) as bad_foo_context:
			FooLogic.internal_args_call(self.bad_target)
		self.assertTrue('must either return' in str(bad_foo_context.exception))

	def test_session_specification(self):
		sess = tf.Session()
		block = self.target(my_prop=0, session=sess)
		block = FooLogic.args_call(block)
		self.assertEqual(block.sess, sess)


# TODO Handle default and implicit state management
# TODO Lifecycle that fuses dynamic & static graph based computing


@sx.tf_function(default_props=sx.Props(state_manager=sx.StateManager([4])))
def dense_hypothesis(ob, state):
	logits = tf.layers.dense(ob, 2)
	next_state = tf.layers.dense(state, 4)
	return Out.logits(logits).state(next_state)


class KGreedy(sx.TFFunction):
	def build(self, logits):
		return sx.Out.action(tf.nn.top_k(logits, self.props.k)[1])


@sx.tf_function
def greedy(logits):
	argmax = tf.expand_dims(tf.argmax(logits, axis=1, output_type=tf.int32), axis=0)
	return sx.Out.action(tf.concat([argmax, argmax], axis=1))  # Just to have the dimentions match with k_greedy = 2


def agent_logic(selected_index, selectors, hypothesis) -> sx.Out:
	actions = [selector(hypothesis.o.logits).o.action for selector in selectors]
	selected_action_op = tf.gather(
		actions,
		tf.cast(selected_index, tf.int32, "action_selected_index"),
		name="action"
	)
	return Out.action(selected_action_op).state(
		(hypothesis.i.state, hypothesis.props.state_manager, hypothesis.o.state))


@sx.stateful_tf_function(None)
def agent(selected_index, selectors, hypothesis) -> sx.StatefulTFFunction:
	return agent_logic(selected_index, selectors, hypothesis)


def build_hypothesis(state_tensor, scope_name):
	return dense_hypothesis(tf.placeholder(tf.float32, [None, 2], 'ob'), state_tensor,
							props=sx.Props(scope_name=scope_name))


def build_agent(agent_cls, hypothesis):
	return agent_cls(
		tf.placeholder(tf.float32, (), 'selected_index'),
		[greedy, KGreedy(k=2)],
		hypothesis
	)


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
			if sx.is_dynamic_arg(self.state_tensor):
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


class TestPlaceholderStateHierarchicalBlock(Suppress2.TestHierarchicalBase):
	state_tensor = dense_hypothesis.props.state_manager.new_placeholder()
	hypo = build_hypothesis(state_tensor, scope_name='placeholder_hypothesis')
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
	hypo = build_hypothesis(state_tensor, scope_name='variable_hypothesis2')
	agnt = build_agent(agent, hypo)


@sx.stateful_tf_function(dense_hypothesis.props.state_manager)
def default_state_manager_agent(selected_index, selectors, hypothesis) -> sx.StatefulTFFunction:
	return agent_logic(selected_index, selectors, hypothesis)


class TestPlaceholderDefaultStateManagerHierarchicalBlock(Suppress2.TestHierarchicalBase):
	state_tensor = dense_hypothesis.props.state_manager.new_placeholder()
	hypo = build_hypothesis(state_tensor, scope_name='placeholder_hypothesis2')
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
	hypo = build_hypothesis(state_tensor, scope_name='variable_hypothesis')
	agnt = build_agent(default_state_manager_agent, hypo)
