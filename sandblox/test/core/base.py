from timeit import timeit
from typing import Type

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.errors_impl import FailedPreconditionError

import sandblox as sx
import sandblox.util.tf_util as U
from sandblox.core.io import bind_resolved
from sandblox.test.core.foo import FooLogic


class Suppressed(object):
	# Wrapped classes don't get tested themselves
	class TestBlockBase(object):
		mold_cls = None  # type: Type[sx.TFMold]
		bad_mold_cls = None  # type: Type[sx.TFMold]

		def create_block(self, **props):
			return self.mold_cls(**props)

		def create_bad_block(self, **props):
			return self.bad_mold_cls(**props)

		def build_block(self, block=None, **props) -> sx.TFMold:
			if block is None:
				block = self.create_block()
			return FooLogic.args_call(block, props=sx.Props(**props))

		def create_bad_built_block(self, block=None, **props) -> sx.TFMold:
			if block is None:
				block = self.create_bad_block()
			return FooLogic.args_call(block, props=sx.Props(**props))

		OVERHEAD_RATIO_LIMIT = 15

		def __init__(self, method_name: str = 'runTest'):
			super(Suppressed.TestBlockBase, self).__init__(method_name)
			built_block = self.build_block()
			with tf.variable_scope(built_block.scope.rel, reuse=True):
				self.bound_flattened_logic_args = bind_resolved(FooLogic.call, *FooLogic.args,
																**FooLogic.kwargs)
				self.logic_outs = list(FooLogic.resolved_args_call(FooLogic.call))

			self.options = tf.RunOptions()
			self.options.output_partition_graphs = True

		def test_block_inputs(self):
			built_block = self.build_block()
			self.assertEqual(built_block.i.__dict__, self.bound_flattened_logic_args)

		def test_block_dynamic_inputs(self):
			built_block = self.build_block()
			self.assertEqual(built_block.di, [sx.resolve(*FooLogic.di)])

		def assertEqual(self, first, second, msg=None):
			first, second = U.core_op_name(first), U.core_op_name(second)
			super(Suppressed.TestBlockBase, self).assertEqual(first, second, msg)

		def test_block_out(self):
			built_block = self.build_block()
			self.assertEqual(U.core_op_name(built_block.o.a), U.core_op_name(self.logic_outs[1]))
			self.assertEqual(U.core_op_name(built_block.o.b), U.core_op_name(self.logic_outs[0]))

		def test_block_out_order(self):
			built_block = self.build_block()
			self.assertEqual(U.core_op_name(built_block.oz), U.core_op_name(self.logic_outs))

		def test_run(self):
			with tf.Session() as sess:
				built_block = self.build_block()
				sess.run(tf.variables_initializer(built_block.get_variables()))
				eval_100 = built_block.run(100)

				metadata = tf.RunMetadata()
				eval_0 = built_block.using(self.options, metadata).run(0)
				self.assertTrue(hasattr(metadata, 'partition_graphs') and len(metadata.partition_graphs) > 0)

				self.assertEqual(eval_100[0], eval_0[0] + 100)
				self.assertNotEqual(eval_100[1], eval_0[1])  # Boy aren't you unlucky if you fail this test XD

		def test_non_Out_return_assertion(self):
			with self.assertRaises(AssertionError) as bad_foo_context:
				with tf.Session(graph=tf.Graph()):
					self.create_bad_built_block(reuse=None)
			self.assertTrue('must either return' in str(bad_foo_context.exception))

		def test_run_overhead(self):
			with tf.Session() as sess:
				built_block = self.build_block()
				sess.run(tf.variables_initializer(built_block.get_variables()))

				run_backup = built_block.built_fn.sess.run
				built_block.built_fn.sess.run = no_op_fn

				actual_elapse = timeit(lambda: built_block.run(100), number=1000)
				stub_elapse = timeit(lambda: built_block.built_fn.sess.run(), number=1000)

				built_block.built_fn.sess.run = run_backup

				overhead_ratio = (actual_elapse - stub_elapse) / stub_elapse

				if overhead_ratio > Suppressed.TestBlockBase.OVERHEAD_RATIO_LIMIT:
					self.fail('Overhead factor of %.1f exceeded limit of %.1f' % (
						overhead_ratio, Suppressed.TestBlockBase.OVERHEAD_RATIO_LIMIT))
				elif overhead_ratio / Suppressed.TestBlockBase.OVERHEAD_RATIO_LIMIT > 0.8:
					print('WARNING %s: Overhead factor of %.1f approaching limit of %.1f' % (
						type(self).__name__, overhead_ratio, Suppressed.TestBlockBase.OVERHEAD_RATIO_LIMIT))

		def test_session_specification(self):
			sess = tf.Session(graph=tf.Graph())
			with tf.Session(graph=tf.Graph()):
				block = self.build_block(session=sess)
				with sess.graph.as_default():
					sess.run(tf.initialize_variables(block.get_variables()))
				self.assertEqual(block.sess, sess)
				block.run(100)
				block.set_session(tf.Session())
				self.assertNotEqual(block.sess, sess)
				with self.assertRaises(RuntimeError) as ctx:
					block.run(100)
				self.assertTrue('graph is empty' in str(ctx.exception))
				with self.assertRaises(AssertionError) as ctx:
					self.build_block(session='some_invalid_session')
				self.assertTrue('must be of type tf.Session' in str(ctx.exception))

		def test_get_variables(self):
			with tf.Graph().as_default():
				block1 = self.build_block(scope_name='source')
				vars1 = block1.get_variables()
				self.assertEqual([var.name for var in vars1], ['source/foo_var:0'])

				init = tf.variables_initializer(vars1)
				with tf.Session() as sess:
					with self.assertRaises(FailedPreconditionError) as ctx:
						sess.run(vars1)
					self.assertTrue('source/foo_var' in ctx.exception.message)
					sess.run(init)
					vals1 = sess.run(vars1)
					self.assertEqual(len(vals1), 1)
					self.assertEqual(vals1[0], np.float32)

		def test_variable_assignment(self):
			with tf.Graph().as_default():
				block1 = self.build_block(scope_name='source')
				block2 = self.build_block(scope_name='block')
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

		def test_make_scope_unique(self):
			with tf.Graph().as_default():
				block1 = self.build_block(scope_name='make_me_unique')
				block2 = self.build_block(scope_name='make_me_unique')
				vars1 = block1.get_variables()
				vars2 = block2.get_variables()
				self.assertTrue(all([var1.name != var2.name for var1, var2 in zip(vars1, vars2)]))
				init = tf.variables_initializer(vars1 + vars2)
				eq_op = tf.equal(vars1, vars2)
				var1_eq_var2 = [tf.assign(var1, var2) for var1, var2 in zip(vars1, vars2)]
				with tf.Session() as sess:
					sess.run(init)
					sess.run(var1_eq_var2)
					self.assertTrue(sess.run(eq_op))
					sess.run(init)
					self.assertTrue(not sess.run(eq_op))

		def test_reuse(self):
			with tf.Graph().as_default():
				block1 = self.build_block(scope_name='reuse_me')
				block2 = self.build_block(scope_name='reuse_me', reuse=True)
				vars1 = block1.get_variables()
				vars2 = block2.get_variables()
				init = tf.variables_initializer(vars1 + vars2)
				eq_op = tf.equal(vars1, vars2)
				update_vars_1 = [tf.assign(var, 2) for var in vars1]
				with tf.Session() as sess:
					sess.run(init)
					self.assertTrue(sess.run(eq_op))
					sess.run(update_vars_1)
					self.assertTrue(sess.run(eq_op))


def no_op_fn(*_args, **_kwargs):
	return ()
