from unittest import TestCase

from sandblox.core.io import BlockOutsKwargs

import sandblox as sx

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
	target = foo
	bad_target = bad_foo
	block_foo_ob = FooLogic.args_call(target)

	def create_block_ob(self, **props) -> sx.Block:
		return FooLogic.args_call(TestBlockFunction.target, props=sx.Props(**props))

	def create_bad_block_ob(self, **props) -> sx.Block:
		return FooLogic.args_call(TestBlockFunction.bad_target, props=sx.Props(**props))
