import time
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
	di = [tf.placeholder(tf.float32, (), 'y')]
	args = [tf.ones((), tf.float32), di[0]]
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


@sx.tf_block
def foo(x, y, param_with_default=-5, **kwargs):
	# noinspection PyTypeChecker
	b, a = FooLogic.cached_call(FooLogic.call, x, y, param_with_default, **kwargs)

	if Out == sx.BlockOutsKwargs:
		return Out(b=b, a=a)
	else:
		return Out.b(b).a(a)


@sx.tf_block
def bad_foo(x, y, param_with_default=-5, **kwargs):
	# noinspection PyTypeChecker
	b, a = FooLogic.cached_call(FooLogic.call, x, y, param_with_default, **kwargs)
	return b, a


class Foo(sx.TFBlock):
	def build(self, x, y, param_with_default=-5, **kwargs):
		# noinspection PyTypeChecker
		b, a = FooLogic.cached_call(FooLogic.call, x, y, param_with_default, **kwargs)

		# TODO Test both cases for python 3.6+
		if Out == sx.BlockOutsKwargs:
			return Out(b=b, a=a)
		else:
			return Out.b(b).a(a)


class BadFoo(sx.TFBlock):
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


class Suppress1(object):
	# Wrapped classes don't get tested themselves
	# noinspection PyCallByClass
	class TestBlockBase(TestCase):
		def __init__(self, method_name: str = 'runTest'):
			super(Suppress1.TestBlockBase, self).__init__(method_name)
			# TODO Use variable instead
			self.bound_flattened_logic_arguments = FooLogic.args_call(sx.FlatBoundArguments(FooLogic.call))
			self.logic_outputs = list(FooLogic.cached_args_call(FooLogic.call))
			self.block_foo_ob = sx.cast_to_block(None)  # To help IDEs help us ;)
			self.bad_foo_context = None

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
			self.assertTrue('must either return' in str(self.bad_foo_context.exception))

		def test_overhead(self):
			self.block_foo_ob.eval = lambda *args: ()
			built_fn = self.block_foo_ob.built_fn
			self.block_foo_ob.built_fn = None
			with tf.Session():
				then = time.time()
				for _ in range(100):
					self.block_foo_ob.run(100)
				elapse = int((time.time() - then) * 1e6 / 10)
				print(elapse)
				self.assertTrue(elapse < 100)
			self.block_foo_ob.built_fn = built_fn


class TestBlockFunction(Suppress1.TestBlockBase):
	def __init__(self, method_name: str = 'runTest'):
		super(TestBlockFunction, self).__init__(method_name)
		self.block_foo_ob = FooLogic.args_call(foo)
		with self.assertRaises(AssertionError) as self.bad_foo_context:
			FooLogic.args_call(bad_foo)


class TestBlockClass(Suppress1.TestBlockBase):
	def setUp(self):
		super(TestBlockClass, self).setUp()
		self.block_foo_ob = FooLogic.args_call(Foo)
		with self.assertRaises(AssertionError) as self.bad_foo_context:
			FooLogic.args_call(BadFoo)


# noinspection PyCallByClass
class TestBlockClassWithInternals(Suppress1.TestBlockBase):
	def setUp(self):
		super(TestBlockClassWithInternals, self).setUp()
		self.block_foo_ob = FooLogic.internal_args_call(FooWithInternalArgs)
		with self.assertRaises(AssertionError) as self.bad_foo_context:
			FooLogic.internal_args_call(BadFooWithInternalArgs)


# TODO Handle default and implicit state management
# TODO Lifecycle that fuses dynamic & static graph based computing


class Hypothesis(sx.StatefullTFBlock):
	__slots__ = 'state',

	STATE = sx.StateShape([4])

	def build(self, ob, state):
		logits = tf.layers.dense(ob, 2)
		next_state = tf.layers.dense(state, 4)
		return Out.logits(logits).state(next_state)


class Agent(sx.TFBlock):
	def __init__(self, *args, **kwargs):
		super(Agent, self).__init__(*args, **kwargs)

	def build(self, selected_index, selectors: [ActionSelector], hypothesis) -> sx.TFBlock:
		selected_action_op = tf.gather(
			[selector(hypothesis.o.logits) for selector in selectors],
			tf.cast(selected_index, tf.int32, "action_selected_index"),
			name="action"
		)
		if not sx.is_dynamic_arg(hypothesis.i.state):
			with tf.control_dependencies([selected_action_op]):
				updated_state = Hypothesis.STATE.assign(hypothesis.i.state, hypothesis.o.state)
		else:
			updated_state = hypothesis.o.state
		return Out.action(selected_action_op).state(updated_state)

	def get_my_givens(self):
		givens = super(Agent, self).get_my_givens()
		if sx.is_dynamic_arg(self.i.hypothesis.i.state):
			givens.update({self.i.hypothesis.i.state: self.state})
		return givens

	def process_my_outputs(self, outputs):
		super(Agent, self).process_my_outputs(outputs)
		if sx.is_dynamic_arg(self.i.hypothesis.i.state):
			state_index = self.oz.index(self.o.state)
			self.state = outputs[state_index]


ob_space, ac_space = GymEnvironment.get_spaces('CartPole-v0')
pdtype = make_pdtype(ac_space)


class Suppress2(object):
	class TestHierarchicalBase(TestCase):
		__slots__ = 'state_tensor',

		def __init__(self, method_name: str = 'runTest'):
			super(Suppress2.TestHierarchicalBase, self).__init__(method_name)
			self.hypo = Hypothesis(tf.placeholder(tf.float32, [None, 2], 'ob'), self.state_tensor)
			self.agnt = Agent(
				tf.placeholder(tf.float32, (), 'selected_index'),
				[action_selector.Greedy(pdtype), action_selector.Stochastic(pdtype)],
				self.hypo
			)

		def setUp(self):
			self.agnt.state = Hypothesis.STATE.new()

		def test_inputs(self):
			ai = self.agnt.i
			self.assertEqual(self.agnt.iz, [ai.selected_index, ai.selectors, ai.hypothesis])
			hi = ai.hypothesis.i
			expected_d_inps = [ai.selected_index, hi.ob]
			if sx.is_dynamic_arg(self.state_tensor):
				expected_d_inps.append(hi.state)
			self.assertEqual(self.agnt.di, expected_d_inps)

		def test_eval(self):
			with tf.Session() as sess:
				sess.run(tf.global_variables_initializer())
				agent_result = self.agnt.run(0, [[.1, .2]])
				self.assertTrue(agent_result[0] in [[0], [1]])
				self.assertEqual(agent_result[1].shape, (1, 4))


class TestPlaceholderStateHierarchicalBlock(Suppress2.TestHierarchicalBase):
	def __init__(self, method_name: str = 'runTest'):
		self.state_tensor = Hypothesis.STATE.new_placeholder()
		super(TestPlaceholderStateHierarchicalBlock, self).__init__(method_name)

	def test_state_update(self):
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			old_state = self.agnt.state
			agent_result = self.agnt.run(0, [[.1, .2]])
			self.assertEqual(agent_result[1].shape, (1, 4))
			self.assertTrue(
				not numpy.alltrue(
					numpy.equal(self.agnt.state, old_state)))  # Small possibility of a false positive here


class TestVarialbleStateHierarchicalBlock(Suppress2.TestHierarchicalBase):
	def __init__(self, method_name: str = 'runTest'):
		self.state_tensor = Hypothesis.STATE.new_variable()
		super(TestVarialbleStateHierarchicalBlock, self).__init__(method_name)
