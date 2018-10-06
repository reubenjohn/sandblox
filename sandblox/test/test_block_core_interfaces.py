import sandblox as sx
from sandblox.test.core.foo import FooLogic


@sx.tf_block
def foo(x, y, param_with_default=-5, **kwargs):
	b, a = FooLogic.call(x, y, param_with_default, **kwargs)

	if sx.Out == sx.BlockOutsKwargs:
		return sx.Out(b=b, a=a)
	else:
		return sx.Out.b(b).a(a)


@sx.tf_block
def bad_foo(x, y, param_with_default=-5, **kwargs):
	b, a = FooLogic.call(x, y, param_with_default, **kwargs)
	return b, a

