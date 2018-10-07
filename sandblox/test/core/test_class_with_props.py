from unittest import TestCase

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
	mold_cls = FooWithProps
	bad_mold_cls = BadFooWithProps

	def create_block(self, **props):
		return super(TestBlockClassWithProps, self).create_block(my_prop=0, **props)

	def create_bad_block(self, **props):
		return super(TestBlockClassWithProps, self).create_bad_block(my_prop=0, **props)
