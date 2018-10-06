from unittest import TestCase

from . import FooLogic
from .test_class import TestBlockClass, Foo, BadFoo


class FooWithProps(Foo):
	def __init__(self, **default_props):
		super(Foo, self).__init__(**default_props)
		assert self.props.my_prop == 0


class BadFooWithProps(BadFoo):
	def __init__(self, **default_props):
		super(BadFoo, self).__init__(**default_props)
		assert self.props.my_prop == 0

class TestBlockClassWithProps(TestBlockClass, TestCase):
	target = FooWithProps
	bad_target = BadFooWithProps
	block_foo_ob = FooLogic.args_call(target(my_prop=0))

	def create_block_ob(self, **props):
		return super(TestBlockClassWithProps, self).create_block_ob(my_prop=0, **props)

	def create_bad_block_ob(self, **props):
		return super(TestBlockClassWithProps, self).create_bad_block_ob(my_prop=0, **props)
