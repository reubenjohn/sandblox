import time
from typing import List, Type
from unittest import TestCase

import tensorflow as tf

import sandblox as sx


class FooLogic(object):
	call_cache = dict()
	di = [tf.placeholder(tf.float32, (), 'y')]
	# TODO Have variables initialized for each call
	args = [tf.ones((), tf.float32), di[0]]
	kwargs = dict(extra=10)
	foo_var = tf.get_variable('foo_var', initializer=-5.0)
	foo_var_initializer = tf.variables_initializer([foo_var])

	@classmethod
	def initialize(cls):
		tf.get_default_session().run(cls.foo_var_initializer)

	@staticmethod
	def args_call(fn, **expansion):
		expansion.update(FooLogic.kwargs)
		return fn(*FooLogic.args, **expansion)

	@staticmethod
	def internal_args_call(fn, **expansion):
		expansion.update(FooLogic.kwargs)
		return fn(0, *FooLogic.args, **expansion)

	@staticmethod
	def call(x, y, param_with_default=-5, **kwargs):
		res = tf.add(
			tf.add(
				tf.add(x, y, 'logic_add_1'),
				kwargs['extra'], 'logic_add_2'),
			param_with_default, 'logic_add_3'
		), tf.add(
			FooLogic.foo_var,
			tf.random_uniform(()), 'logic_2_add'
		)
		return res


@sx.tf_function
def foo(x, y, param_with_default=-5, **kwargs):
	# noinspection PyTypeChecker
	b, a = FooLogic.call(x, y, param_with_default, **kwargs)

	if sx.Out == sx.BlockOutsKwargs:
		return sx.Out(b=b, a=a)
	else:
		return sx.Out.b(b).a(a)


@sx.tf_function
def bad_foo(x, y, param_with_default=-5, **kwargs):
	# noinspection PyTypeChecker
	b, a = FooLogic.call(x, y, param_with_default, **kwargs)
	return b, a


# noinspection PyClassHasNoInit
class Foo(sx.TFFunction):
	def build(self, x, y, param_with_default=-5, **kwargs):
		# noinspection PyTypeChecker
		b, a = FooLogic.call(x, y, param_with_default, **kwargs)

		# TODO Test both cases for python 3.6+
		if sx.Out == sx.BlockOutsKwargs:
			return sx.Out(b=b, a=a)
		else:
			return sx.Out.b(b).a(a)


# noinspection PyClassHasNoInit
class BadFoo(sx.TFFunction):
	def build(self, x, y, param_with_default=-5, **kwargs):
		# noinspection PyTypeChecker
		b, a = FooLogic.call(x, y, param_with_default, **kwargs)

		return b, a


class FooWithProps(Foo):
	def __init__(self, **default_props):
		super(Foo, self).__init__(**default_props)
		assert self.props.my_prop == 0


class BadFooWithProps(BadFoo):
	def __init__(self, **default_props):
		super(BadFoo, self).__init__(**default_props)
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

		def create_bad_block_ob(self, **props) -> sx.TFFunction:
			raise NotImplementedError

		ELAPSE_LIMIT = 25000  # usec To accommodate slowness during debugging
		ELAPSE_TARGET = 2500  # usec

		def __init__(self, method_name: str = 'runTest'):
			super(Suppress1.TestBlockBase, self).__init__(method_name)
			with tf.variable_scope(self.block_foo_ob.scope.rel, reuse=True):
				self.bound_flattened_logic_args = FooLogic.args_call(sx.U.FlatArgumentsBinder(FooLogic.call))
				self.logic_outs = list(FooLogic.args_call(FooLogic.call))

			self.options = tf.RunOptions()
			self.options.output_partition_graphs = True
			self.options.trace_level = tf.RunOptions.FULL_TRACE

		def test_block_inputs(self):
			self.assertEqual(self.block_foo_ob.i.__dict__, self.bound_flattened_logic_args)

		def test_block_dynamic_inputs(self):
			self.assertEqual(self.block_foo_ob.di, FooLogic.di)

		@staticmethod
		def core_op_name(op) -> str:
			return op.name.split('/')[-1]

		@staticmethod
		def core_op_names(ops) -> List[str]:
			return list(map(Suppress1.TestBlockBase.core_op_name, ops))

		def test_block_out(self):
			self.assertEqual(self.core_op_name(self.block_foo_ob.o.a), self.core_op_name(self.logic_outs[1]))
			self.assertEqual(self.core_op_name(self.block_foo_ob.o.b), self.core_op_name(self.logic_outs[0]))

		def test_block_out_order(self):
			self.assertEqual(self.core_op_names(self.block_foo_ob.oz), self.core_op_names(self.logic_outs))

		def test_eval(self):
			with tf.Session() as sess:
				FooLogic.initialize()
				sess.run(tf.variables_initializer(self.block_foo_ob.get_variables()))
				eval_100 = self.block_foo_ob.run(100)

				metadata = tf.RunMetadata()
				eval_0 = self.block_foo_ob.using(self.options, metadata).run(0)
				self.assertTrue(hasattr(metadata, 'partition_graphs') and len(metadata.partition_graphs) > 0)

				self.assertEqual(eval_100[0], eval_0[0] + 100)
				self.assertNotEqual(eval_100[1], eval_0[1])  # Boy aren't you unlucky if you fail this test XD

		def test_bad_foo_assertion(self):
			with self.assertRaises(AssertionError) as bad_foo_context:
				self.create_bad_block_ob(reuse=None)
			self.assertTrue('must either return' in str(bad_foo_context.exception))

		def test_run_overhead(self):
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

		def test_variable_assignment(self):
			# with tf.Graph().as_default() as graph:
			# 	block1 = self.create_block_ob()
			# 	block2 = self.create_block_ob()
			# 	vars1 = block1.get_variables()
			# 	vars2 = block2.get_variables()
			# 	vars = vars1.expand(*vars2)
			# 	init = tf.variables_initializer(vars)
			# 	with tf.Session() as sess:
			# 		init.eval()
			# 	block2.assign_vars(block1)
			pass


class TestBlockFunction(Suppress1.TestBlockBase):
	target = foo
	bad_target = bad_foo
	block_foo_ob = FooLogic.args_call(target)

	def create_block_ob(self, **props) -> sx.BlockBase:
		return FooLogic.args_call(TestBlockFunction.target, props=sx.Props(**props))

	def create_bad_block_ob(self, **props) -> sx.BlockBase:
		return FooLogic.args_call(TestBlockFunction.bad_target, props=sx.Props(**props))

	def __init__(self, method_name: str = 'runTest'):
		super(TestBlockFunction, self).__init__(method_name)


class TestBlockClass(Suppress1.TestBlockBase):
	target = Foo()
	bad_target = BadFoo()
	block_foo_ob = FooLogic.args_call(target)

	def create_block_ob(self, **props) -> sx.BlockBase:
		return FooLogic.args_call(TestBlockClass.target, props=sx.Props(**props))

	def create_bad_block_ob(self, **props) -> sx.BlockBase:
		return FooLogic.args_call(TestBlockClass.bad_target, props=sx.Props(**props))


# noinspection PyCallByClass
class TestBlockClassWithProps(TestBlockClass):
	target = FooWithProps
	bad_target = BadFooWithProps
	block_foo_ob = FooLogic.args_call(target(my_prop=0))

	def create_block_ob(self, **props):
		return super(TestBlockClassWithProps, self).create_block_ob(my_prop=0, **props)

	def create_bad_block_ob(self, **props):
		return super(TestBlockClassWithProps, self).create_bad_block_ob(my_prop=0, **props)
