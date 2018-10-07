from unittest import TestCase

import tensorflow as tf

import sandblox as sx
from sandblox.core.io import BlockOutsKwargs
from . import TestBlockBase, FooLogic


@sx.tf_block
def foo(x, y, param_with_default=-5, **kwargs):
	b, a = FooLogic.call(x, y, param_with_default, **kwargs)

	if sx.Out == BlockOutsKwargs:
		return sx.Out(b=b, a=a)
	else:
		return sx.Out.b(b).a(a)


@sx.tf_block
def bad_foo(x, y, param_with_default=-5, **kwargs):
	b, a = FooLogic.call(x, y, param_with_default, **kwargs)
	return b, a


class TestBlockFunction(TestBlockBase, TestCase):
	mold_cls = None
	bad_mold_cls = None

	def create_block(self, **props):
		raise AssertionError('Block decorations are designed to share share one block instance')

	def create_bad_block(self, **props):
		raise AssertionError('Block decorations are designed to share share one block instance')

	def build_block(self, block=None, **props) -> sx.TFMold:
		return super().build_block(foo, **props)

	def create_bad_built_block(self, block=None, **props) -> sx.TFMold:
		return super().build_block(bad_foo, **props)

	def test_is_built(self):
		with tf.Graph().as_default():
			self.assertTrue(not foo.is_built())

			block = self.build_block(scope_name='make_me_unique')
			self.assertTrue(block.is_built())

			with self.assertRaises(AssertionError) as ctx:
				FooLogic.args_call(block.build_graph)
			self.assertTrue('already built' in ctx.exception.args[0])
