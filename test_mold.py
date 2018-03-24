from unittest import TestCase

import numpy
import tensorflow as tf

import sandblox as sx
from reils.hypothesis import action_selector
from reils.hypothesis.action_selector import ActionSelector
from reils.utils.distributions import make_pdtype
from reils_gym import GymEnvironment
from sandblox import Out


class FooLogic(object):
	call_cache = dict()
	args = [tf.ones((), tf.float32), tf.placeholder(tf.float32, (), 'y')]
	kwargs = dict(extra=10)
	args_call = lambda fn: fn(*FooLogic.args, **FooLogic.kwargs)
	cached_args_call = lambda fn: FooLogic.cached_call(fn, *FooLogic.args, **FooLogic.kwargs)
	internal_args_call = lambda fn: fn(0, *FooLogic.args, **FooLogic.kwargs)

	@staticmethod
	def call(x, y, param_with_default=-5, **kwargs):
		return x + y + kwargs['extra'] + param_with_default, tf.random_uniform(())

	@staticmethod
	def cached_call(fn, *args, **kwargs):
		bound_args = sx.FlatBoundArguments(fn)(*args, **kwargs)
		key = fn.__name__ + ':' + str(bound_args)
		if key not in FooLogic.call_cache:
			FooLogic.call_cache[key] = fn(*args, **kwargs)
		return FooLogic.call_cache[key]


@sx.block
def foo(x, y, param_with_default=-5, **kwargs):
	# noinspection PyTypeChecker
	b, a = FooLogic.cached_call(FooLogic.call, x, y, param_with_default, **kwargs)

	if Out == sx.BlockOutsKwargs:
		return Out(b=b, a=a)
	else:
		return Out.b(b).a(a)


@sx.block
def bad_foo(x, y, param_with_default=-5, **kwargs):
	# noinspection PyTypeChecker
	b, a = FooLogic.cached_call(FooLogic.call, x, y, param_with_default, **kwargs)
	return b, a


class Foo(sx.Block):
	def build(self, x, y, param_with_default=-5, **kwargs):
		# noinspection PyTypeChecker
		b, a = FooLogic.cached_call(FooLogic.call, x, y, param_with_default, **kwargs)

		# TODO Test both cases for python 3.6+
		if Out == sx.BlockOutsKwargs:
			return Out(b=b, a=a)
		else:
			return Out.b(b).a(a)


class BadFoo(sx.Block):
	def build(self, x, y, param_with_default=-5, **kwargs):
		# noinspection PyTypeChecker
		b, a = FooLogic.cached_call(FooLogic.call, x, y, param_with_default, **kwargs)

		return b, a


class FooWithInternalArgs(Foo):
	def __init__(self, exclusive_constructor_arg, x, y, param_with_default=-5, **kwargs):
		super(Foo, self).__init__(x, y, param_with_default, **kwargs)
		self.internal_arg = exclusive_constructor_arg


class BadFooWithInternalArgs(BadFoo):
	def __init__(self, exclusive_constructor_arg, x, y, param_with_default=-5, **kwargs):
		super(BadFoo, self).__init__(x, y, param_with_default, **kwargs)
		self.internal_arg = exclusive_constructor_arg


class BaseTestCases(object):
	# Wrapped classes don't get tested themselves
	# noinspection PyCallByClass
	class TestBlockBase(TestCase):
		def __init__(self, method_name: str = 'runTest'):
			super(BaseTestCases.TestBlockBase, self).__init__(method_name)
			# TODO Use variable instead
			self.bound_flattened_logic_arguments = FooLogic.args_call(sx.FlatBoundArguments(FooLogic.call))
			self.logic_outputs = list(FooLogic.cached_args_call(FooLogic.call))
			self.block_foo_ob = None
			self.bad_foo_context = None

			self.options = tf.RunOptions()
			self.options.output_partition_graphs = True
			self.options.trace_level = tf.RunOptions.FULL_TRACE

		def setUp(self):
			super(BaseTestCases.TestBlockBase, self).setUp()

		def test_block_inputs(self):
			self.assertEqual(self.block_foo_ob.i.__dict__, self.bound_flattened_logic_arguments)

		def test_block_out(self):
			self.assertEqual(self.block_foo_ob.o.a, self.logic_outputs[1])
			self.assertEqual(self.block_foo_ob.o.b, self.logic_outputs[0])

		def test_block_out_order(self):
			self.assertEqual(self.block_foo_ob.oz, self.logic_outputs)

		def test_eval(self):
			with tf.Session():
				eval_100 = self.block_foo_ob.eval(100)

				metadata = tf.RunMetadata()
				eval_0 = self.block_foo_ob.using(self.options, metadata).eval(0)
				self.assertTrue(hasattr(metadata, 'partition_graphs') and len(metadata.partition_graphs) > 0)

				self.assertEqual(eval_100[0], eval_0[0] + 100)
				self.assertNotEqual(eval_100[1], eval_0[1])  # Boy aren't you unlucky if you fail this test XD

		def test_bad_foo_assertion(self):
			self.assertTrue('must either return' in str(self.bad_foo_context.exception))


class TestBlockFunction(BaseTestCases.TestBlockBase):
	def __init__(self, method_name: str = 'runTest'):
		super(TestBlockFunction, self).__init__(method_name)
		self.block_foo_ob = FooLogic.args_call(foo)
		with self.assertRaises(AssertionError) as self.bad_foo_context:
			FooLogic.args_call(bad_foo)


class TestBlockClass(BaseTestCases.TestBlockBase):
	def setUp(self):
		super(TestBlockClass, self).setUp()
		self.block_foo_ob = FooLogic.args_call(Foo)
		with self.assertRaises(AssertionError) as self.bad_foo_context:
			FooLogic.args_call(BadFoo)


# noinspection PyCallByClass
class TestBlockClassWithInternals(BaseTestCases.TestBlockBase):
	def setUp(self):
		super(TestBlockClassWithInternals, self).setUp()
		self.block_foo_ob = FooLogic.internal_args_call(FooWithInternalArgs)
		with self.assertRaises(AssertionError) as self.bad_foo_context:
			FooLogic.internal_args_call(BadFooWithInternalArgs)


# TODO Handle default and implicit state management
# TODO Lifecycle that fuses dynamic & static graph based computing


class TestBlockUsage(TestCase):
	ob_space, ac_space = GymEnvironment.get_spaces('CartPole-v0')

	def test_lifecycle(self):
		class Hypothesis(sx.Block):
			def build(self, ob, state):
				logits = tf.layers.dense(ob, 2)
				next_state = tf.layers.dense(state, 4)
				return Out.logits(logits).state(next_state)

			@staticmethod
			def state_shape():
				return [4]

			@staticmethod
			def state_batch_shape(batch_size):
				batch_size = batch_size if isinstance(batch_size, list) else [batch_size]
				return batch_size + Hypothesis.state_shape()

			@staticmethod
			def new_state_placeholder(batch_size: [None, int, list] = None):
				return tf.placeholder(tf.float32, Hypothesis.state_batch_shape(batch_size), 'state')

			@staticmethod
			def new_state_variable(batch_size: [int, list] = 1):
				return tf.Variable(Hypothesis.new_state(batch_size), name='state')

			@staticmethod
			def new_state(batch_size: [int, list] = 1):
				return numpy.zeros(Hypothesis.state_batch_shape(batch_size), numpy.float32)

			@staticmethod
			def assign_state(dest_state, src_state):
				pass

		@sx.block
		def agent(selected_index, selectors: [ActionSelector], hypothesis) -> sx.Block:
			selected_action_op = tf.gather(
				[selector(hypothesis.o.logits) for selector in selectors],
				tf.cast(selected_index, tf.int32, "action_selected_index"),
				name="action"
			)
			return Out.action(selected_action_op).state(hypothesis.o.state)

		hypo = Hypothesis(tf.placeholder(tf.float32, [None, 2], 'ob'), Hypothesis.new_state_placeholder())
		pdtype = make_pdtype(TestBlockUsage.ac_space)
		agnt = agent(
			tf.placeholder(tf.float32, (), 'selected_index'),
			[action_selector.Greedy(pdtype), action_selector.Stochastic(pdtype)],
			hypo
		)
		ai = agnt.i
		self.assertEqual(agnt.iz,
						 [
							 ai.selected_index,
							 ai.selectors,
							 ai.hypothesis
						 ])
		hi = ai.hypothesis.i
		self.assertEqual(agnt.d_inps,
						 [
							 ai.selected_index,
							 hi.ob,
							 hi.state
						 ])
		# TODO test options and run_metadata
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			agent_result = agnt.eval(0, [[.1, .2]], [[.1, .2, .3, .4]])
			self.assertTrue(agent_result[0] in [[0], [1]])
			self.assertEqual(agent_result[1].shape, (1, 4))
