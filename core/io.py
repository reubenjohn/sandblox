import sys

import tensorflow as tf

from sandblox.util.misc import DictAttrBuilder, DictAttrs, DictAttrBuilderFactory


class _BlockOutsBase(DictAttrBuilder):
	def _on_new_attr_val(self, key, val):
		if key in self.o:
			print('Warning an output named %s already exists with value: %s' % (key, self.o[key]))
		self.o[key] = val
		self.oz.append(val)


class BlockOutsKwargs(_BlockOutsBase):
	_you_were_warned = False  # TODO Use DesignViolation implementation instead

	def __init__(self, **kwargs):
		self.o = kwargs
		if not BlockOutsKwargs._you_were_warned and (
						sys.version_info[0] < 3 or (sys.version_info[0] == 3 and sys.version_info[1] < 6)):
			print('WARNING: keyword arguments constructor will not preserve output order before Python 3.6!\n' +
				  'Please use the empty constructor approach provided for backward compatibility:\n' +
				  'Eg: ' + type(self).__name__ + '().a(a_val).b(b_val)')
		self.oz = tuple(val for val in kwargs.values())


class BlockOutsAttrs(_BlockOutsBase):
	def __init__(self):
		self.o = DictAttrs()
		self.oz = []


Out = DictAttrBuilderFactory(BlockOutsAttrs) if sys.version_info[0] < 3 or (
	sys.version_info[0] == 3 and sys.version_info[1] < 6) else BlockOutsKwargs

Props = DictAttrs  # Don't need DictAttrBuilderFactory since prop order does not need to be maintained


# TODO Add tests for dynamic arg concept
def dynamic(*args):
	for arg in args:
		arg.is_d_inp = True
	if len(args) == 1:
		return args[0]
	return args


def is_dynamic_arg(arg):
	return hasattr(arg, 'is_d_inp') or (isinstance(arg, tf.Tensor) and arg.op.type == 'Placeholder')
