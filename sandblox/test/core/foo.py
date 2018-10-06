import tensorflow as tf

import sandblox as sx


class FooLogic(object):
	call_cache = dict()
	di = [sx.arg(lambda: tf.placeholder(tf.float32, (), 'y'))]
	# TODO Have variables initialized for each call
	args = [sx.arg(lambda: tf.ones((), tf.float32)), di[0]]
	kwargs = dict(extra=10)

	@staticmethod
	def args_call(fn, **expansion):
		expansion.update(FooLogic.kwargs)
		return fn(*FooLogic.args, **expansion)

	@staticmethod
	def resolved_args_call(fn, **expansion):
		expansion.update(FooLogic.kwargs)
		return fn(*sx.resolve(*FooLogic.args), **expansion)

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