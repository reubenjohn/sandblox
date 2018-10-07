from unittest import TestCase

import tensorflow as tf

import sandblox as sx
from sandblox.core.io import BlockOutsKwargs
from . import TestBlockBase, FooLogic


class Foo(sx.TFMold):
	def static(self, x, y, param_with_default=-5, **kwargs):
		b, a = FooLogic.call(x, y, param_with_default, **kwargs)

		# TODO Test both cases for python 3.6+
		if sx.Out == BlockOutsKwargs:
			return sx.Out(b=b, a=a)
		else:
			return sx.Out.b(b).a(a)


class BadFoo(sx.TFMold):
	def static(self, x, y, param_with_default=-5, **kwargs):
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


class TestBlockClass(TestBlockBase, TestCase):
	mold_cls = Foo
	bad_mold_cls = BadFoo

	def test_is_built(self):
		with tf.Session(graph=tf.Graph()).as_default():
			block = self.create_block()
			self.assertTrue(not block.is_built())

			built_block = self.build_block()
			self.assertTrue(built_block.is_built())

			with self.assertRaises(AssertionError) as ctx:
				FooLogic.args_call(built_block.setup_static)
			self.assertTrue('already built' in ctx.exception.args[0])
