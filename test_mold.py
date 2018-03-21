from unittest import TestCase

import numpy
import tensorflow as tf

import pythonic_tf as sb
from pythonic_tf import Out, dynamic
from reils.hypothesis import action_selector
from reils.hypothesis.action_selector import ActionSelector
from reils.utils.distributions import make_pdtype
from reils_gym import GymEnvironment


class FooLogic(object):
	call_cache = dict()
	args = [tf.ones((), tf.float32), dynamic(tf.placeholder(tf.float32, (), 'y'))]
	kwargs = dict(extra=10)
	args_call = lambda fn: fn(*FooLogic.args, **FooLogic.kwargs)
	cached_args_call = lambda fn: FooLogic.cached_call(fn, *FooLogic.args, **FooLogic.kwargs)
	internal_args_call = lambda fn: fn(0, *FooLogic.args, **FooLogic.kwargs)

	@staticmethod
	def call(x, y, param_with_default=-5, **kwargs):
		return x + y + kwargs['extra'] + param_with_default, tf.random_uniform(())

	@staticmethod
	def cached_call(fn, *args, **kwargs):
		bound_args = sb.FlatBoundArguments(fn)(*args, **kwargs)
		key = fn.__name__ + ':' + str(bound_args)
		if key not in FooLogic.call_cache:
			FooLogic.call_cache[key] = fn(*args, **kwargs)
		return FooLogic.call_cache[key]


# noinspection PyTypeChecker
@sb.mold
def foo(x, y, param_with_default=-5, **kwargs):
	b, a = FooLogic.cached_call(FooLogic.call, x, y, param_with_default, **kwargs)

	if Out == sb.BlockOutsKwargs:
		return Out(b=b, a=a)
	else:
		return Out.b(b).a(a)


# noinspection PyTypeChecker
class Foo(sb.Mold):
	def on_build(self, x, y, param_with_default=-5, **kwargs):
		b, a = FooLogic.cached_call(FooLogic.call, x, y, param_with_default, **kwargs)

		# TODO Test both cases for python 3.6+
		if Out == sb.BlockOutsKwargs:
			out = Out(b=b, a=a)
		else:
			out = Out.b(b).a(a)

		return out


class FooWithInternalArgs(Foo):
	def __init__(self, exclusive_constructor_arg, x, y, param_with_default=-5, **kwargs):
		super(Foo, self).__init__(x, y, param_with_default, **kwargs)
		self.internal_arg = exclusive_constructor_arg


class BaseTestCases(object):
	# Wrapped classes don't get tested themselves
	# noinspection PyCallByClass
	class TestMoldBase(TestCase):
		def __init__(self, method_name: str = 'runTest'):
			super(BaseTestCases.TestMoldBase, self).__init__(method_name)
			# TODO Use variable instead
			self.bound_flattened_logic_arguments = FooLogic.args_call(sb.FlatBoundArguments(FooLogic.call))
			self.logic_outputs = FooLogic.cached_args_call(FooLogic.call)
			self.block_foo_ob = None

		def setUp(self):
			super(BaseTestCases.TestMoldBase, self).setUp()

		def test_mold_inputs(self):
			self.assertEqual(self.block_foo_ob.inp.__dict__, self.bound_flattened_logic_arguments)

		def test_mold_out(self):
			self.assertEqual(self.block_foo_ob.out.a, self.logic_outputs[1])
			self.assertEqual(self.block_foo_ob.out.b, self.logic_outputs[0])

		def test_mold_out_order(self):
			self.assertEqual(self.block_foo_ob.outs, self.logic_outputs)

		def test_eval(self):
			with tf.Session():
				eval_100 = self.block_foo_ob.eval(100)
				eval_0 = self.block_foo_ob.eval(0)
				self.assertEqual(eval_100[0], eval_0[0] + 100)
				self.assertNotEqual(eval_100[1], eval_0[1])  # Boy aren't you unlucky if you fail this test XD


class TestMoldFunction(BaseTestCases.TestMoldBase):
	def __init__(self, method_name: str = 'runTest'):
		super(TestMoldFunction, self).__init__(method_name)
		self.block_foo_ob = FooLogic.args_call(foo)


class TestMoldClass(BaseTestCases.TestMoldBase):
	def setUp(self):
		super(TestMoldClass, self).setUp()
		self.block_foo_ob = FooLogic.args_call(Foo)


# noinspection PyCallByClass
class TestMoldClassWithInternals(BaseTestCases.TestMoldBase):
	def setUp(self):
		super(TestMoldClassWithInternals, self).setUp()
		self.block_foo_ob = FooLogic.internal_args_call(FooWithInternalArgs)


class TestMoldUsage(TestCase):
	class Hypothesis(sb.Mold):
		def on_build(self, ob, prev_state):
			logits = tf.layers.dense(ob, 2)
			next_state = tf.layers.dense(prev_state, 4)
			return Out.logits(logits).state(next_state)

		@staticmethod
		def new_state_placeholder(batch_shape: [None, int, list] = None):
			batch_shape = batch_shape if isinstance(batch_shape, list) else [batch_shape]
			return tf.placeholder(tf.float32, batch_shape)

		@staticmethod
		def new_state_variable(batch_shape: [int, list] = 1):
			return tf.Variable(TestMoldUsage.Hypothesis.new_state(batch_shape))

		@staticmethod
		def new_state(batch_shape: [int, list] = 1):
			batch_shape = batch_shape if isinstance(batch_shape, list) else [batch_shape]
			return numpy.zeros(batch_shape + [4], numpy.float32)

		@staticmethod
		def assign_state(dest_state, src_state):
			pass

	class Agent(sb.Mold):
		def on_build(self, selected_index, selectors: [ActionSelector], hypothesis, hypothesis_class=None, ob=None):
			selector_ops = [selector(hypothesis.out.logits) for selector in selectors]
			selected_action_op = tf.gather(selector_ops,
										   tf.cast(selected_index, tf.int32, "action_selected_index"),
										   name="action")
			return Out.act(selected_action_op)

	def test_lifecycle(self):
		# TODO Handle default and implicit state management
		# TODO Lifecycle that fuses dynamic & static graph based computing
		ob_space, ac_space = GymEnvironment.get_spaces('CartPole-v0')
		hypo = TestMoldUsage.Hypothesis(dynamic(tf.placeholder(tf.float32, [None, 2])),
										TestMoldUsage.Hypothesis.new_state_variable())
		self.assertTrue(hypo.inp.ob is not None)
		pdtype = make_pdtype(ac_space)
		agent = TestMoldUsage.Agent(dynamic(tf.placeholder(tf.float32, (), 'selected_index')),
									[action_selector.Greedy(pdtype),
									 action_selector.LinDecayEpsilonGreedy(pdtype, .95, 10000)], hypo)
		self.assertTrue(agent.out.act is not None)
		self.assertEqual(agent.inps, [agent.inp.selected_index, agent.inp.selectors, agent.inp.hypothesis, None, None])
		self.assertEqual(agent.d_inps, [agent.inp.selected_index, agent.inp.hypothesis.inp.ob])
