from typing import Callable, Union, Any
from unittest import TestCase

import tensorflow as tf

import sandblox as sx


class Adder(sx.TFFunction):
	def build(self, a, b):
		return sx.Out.result(tf.add(a, b))


@sx.tf_function
def square(calculator):
	return sx.Out.result(tf.square(calculator.o.result))


class TestInputsAreBuiltCheck(TestCase):
	def test_inputs_are_built_check(self):
		with tf.Session(graph=tf.Graph()) as sess:
			adder = Adder()  # type: Union[Any, Callable]
			adder_a_b = adder(1, 2)
			square_a_plus_b = square(adder_a_b)
			self.assertEqual(9, sess.run(square_a_plus_b.o.result))

			with self.assertRaises(sx.errors.NotBuiltError) as ctx:
				square(adder)
			self.assertTrue('not been built' in str(ctx.exception))
