from unittest import TestCase

import sandblox as sx

from . import TestBlockBase, FooLogic


class Foo(sx.TFMold):
	def build(self, x, y, param_with_default=-5, **kwargs):
		b, a = FooLogic.call(x, y, param_with_default, **kwargs)

		# TODO Test both cases for python 3.6+
		if sx.Out == sx.BlockOutsKwargs:
			return sx.Out(b=b, a=a)
		else:
			return sx.Out.b(b).a(a)


class BadFoo(sx.TFMold):
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



class TestBlockClass(TestBlockBase, TestCase):
	target = Foo()
	bad_target = BadFoo()
	block_foo_ob = FooLogic.args_call(target)

	def create_block_ob(self, **props) -> sx.Block:
		return FooLogic.args_call(TestBlockClass.target, props=sx.Props(**props))

	def create_bad_block_ob(self, **props) -> sx.Block:
		return FooLogic.args_call(TestBlockClass.bad_target, props=sx.Props(**props))
