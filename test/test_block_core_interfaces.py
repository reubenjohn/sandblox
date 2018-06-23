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
			tf.get_variable('foo_var', initializer=tf.random_uniform((), -1, 1)),
			tf.random_uniform(()), 'logic_2_add'
		)
		return res


@sx.tf_function
def foo(x, y, param_with_default=-5, **kwargs):
	b, a = FooLogic.call(x, y, param_with_default, **kwargs)

	if sx.Out == sx.BlockOutsKwargs:
		return sx.Out(b=b, a=a)
	else:
		return sx.Out.b(b).a(a)


@sx.tf_function
def bad_foo(x, y, param_with_default=-5, **kwargs):
	b, a = FooLogic.call(x, y, param_with_default, **kwargs)
	return b, a


# noinspection PyClassHasNoInit
class Foo(sx.TFFunction):
	def build(self, x, y, param_with_default=-5, **kwargs):
		b, a = FooLogic.call(x, y, param_with_default, **kwargs)

		# TODO Test both cases for python 3.6+
		if sx.Out == sx.BlockOutsKwargs:
			return sx.Out(b=b, a=a)
		else:
			return sx.Out.b(b).a(a)


# noinspection PyClassHasNoInit
class BadFoo(sx.TFFunction):
	def build(self, x, y, param_with_default=-5, **kwargs):
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


class Suppress(object):
	# Wrapped classes don't get tested themselves
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
			super(Suppress.TestBlockBase, self).__init__(method_name)
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
			return list(map(Suppress.TestBlockBase.core_op_name, ops))

		def test_block_out(self):
			self.assertEqual(self.core_op_name(self.block_foo_ob.o.a), self.core_op_name(self.logic_outs[1]))
			self.assertEqual(self.core_op_name(self.block_foo_ob.o.b), self.core_op_name(self.logic_outs[0]))

		def test_block_out_order(self):
			self.assertEqual(self.core_op_names(self.block_foo_ob.oz), self.core_op_names(self.logic_outs))

		def test_eval(self):
			with tf.Session() as sess:
				sess.run(tf.variables_initializer(self.block_foo_ob.get_variables()))
				eval_100 = self.block_foo_ob.run(100)

				metadata = tf.RunMetadata()
				eval_0 = self.block_foo_ob.using(self.options, metadata).run(0)
				self.assertTrue(hasattr(metadata, 'partition_graphs') and len(metadata.partition_graphs) > 0)

				self.assertEqual(eval_100[0], eval_0[0] + 100)
				self.assertNotEqual(eval_100[1], eval_0[1])  # Boy aren't you unlucky if you fail this test XD

		def test_bad_foo_assertion(self):
			with self.assertRaises(AssertionError) as bad_foo_context:
				with tf.Session(graph=tf.Graph()):
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
				if elapse > Suppress.TestBlockBase.ELAPSE_LIMIT:
					overdue_percentage = \
						(Suppress.TestBlockBase.ELAPSE_LIMIT - elapse) * 100 / Suppress.TestBlockBase.ELAPSE_LIMIT
					self.fail('Overdue by %.1f%% (%3dns elapsed)' % (overdue_percentage, elapse))
				else:
					goal_progress = 100 + ((Suppress.TestBlockBase.ELAPSE_TARGET - elapse) * 100 / elapse)
					print('Efficiency goal progress %.1f%% (%3dns elapsed) - %s' % (
						goal_progress, elapse, type(self).__name__))

			self.block_foo_ob.built_fn = built_fn

		# TODO Test scope_name
		def test_session_specification(self):
			sess = tf.Session()
			with tf.Session(graph=tf.Graph()):
				block = self.create_block_ob(session=sess)
				self.assertEqual(block.sess, sess)
				block.set_session(tf.Session())
				self.assertNotEqual(block.sess, sess)
				with self.assertRaises(AssertionError) as bad_foo_context:
					self.create_block_ob(session='some_invalid_session')
				self.assertTrue('must be of type tf.Session' in str(bad_foo_context.exception))

		def test_variable_assignment(self):
			with tf.Graph().as_default():
				block1 = self.create_block_ob(scope_name='source')
				block2 = self.create_block_ob(scope_name='target')
				vars1 = block1.get_variables()
				vars2 = block2.get_variables()
				init = tf.variables_initializer(vars1 + vars2)
				assignment_op = block2.assign_vars(block1)
				eq_op = tf.equal(vars1, vars2)
				with tf.Session() as sess:
					sess.run(init)
					self.assertTrue(not sess.run(eq_op))
					sess.run(assignment_op)
					self.assertTrue(sess.run(eq_op))

		def test_reuse(self):
			with tf.Graph().as_default():
				block1 = self.create_block_ob(scope_name='reuse_me')
				block2 = self.create_block_ob(scope_name='reuse_me', reuse=True)
				vars1 = block1.get_variables()
				tf.get_collection(tf.GraphKeys.UPDATE_OPS, block1.scope.exact_abs_pattern)
				vars2 = block2.get_variables()
				init = tf.variables_initializer(vars1 + vars2)
				eq_op = tf.equal(vars1, vars2)
				update_vars_1 = [tf.assign(var, 2) for var in vars1]
				with tf.Session() as sess:
					sess.run(init)
					self.assertTrue(sess.run(eq_op))
					sess.run(update_vars_1)
					self.assertTrue(sess.run(eq_op))


class TestBlockFunction(Suppress.TestBlockBase):
	target = foo
	bad_target = bad_foo
	block_foo_ob = FooLogic.args_call(target)

	def create_block_ob(self, **props) -> sx.BlockBase:
		return FooLogic.args_call(TestBlockFunction.target, props=sx.Props(**props))

	def create_bad_block_ob(self, **props) -> sx.BlockBase:
		return FooLogic.args_call(TestBlockFunction.bad_target, props=sx.Props(**props))

	def __init__(self, method_name: str = 'runTest'):
		super(TestBlockFunction, self).__init__(method_name)


class TestBlockClass(Suppress.TestBlockBase):
	target = Foo()
	bad_target = BadFoo()
	block_foo_ob = FooLogic.args_call(target)

	def create_block_ob(self, **props) -> sx.BlockBase:
		return FooLogic.args_call(TestBlockClass.target, props=sx.Props(**props))

	def create_bad_block_ob(self, **props) -> sx.BlockBase:
		return FooLogic.args_call(TestBlockClass.bad_target, props=sx.Props(**props))


class TestBlockClassWithProps(TestBlockClass):
	target = FooWithProps
	bad_target = BadFooWithProps
	block_foo_ob = FooLogic.args_call(target(my_prop=0))

	def create_block_ob(self, **props):
		return super(TestBlockClassWithProps, self).create_block_ob(my_prop=0, **props)

	def create_bad_block_ob(self, **props):
		return super(TestBlockClassWithProps, self).create_bad_block_ob(my_prop=0, **props)
