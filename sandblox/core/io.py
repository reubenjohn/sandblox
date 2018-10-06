import sys
from collections.__init__ import OrderedDict

import tensorflow as tf

from sandblox.core import resolve
from sandblox.util import U
from sandblox.util.misc import DictAttrBuilder, DictAttrs, DictAttrBuilderFactory


class BlockOutsBase(DictAttrBuilder):
	def _on_new_attr_val(self, key, val):
		if key in self.o:
			# TODO Use DesignViolation implementation instead
			print('Warning an existing output named %s with value %s will be overwritten' % (key, self.o[key]))
			self.oz[self.oz.index(self.o[key])] = val
		else:
			self.oz.append(val)
		self.o[key] = val


class BlockOutsKwargs(BlockOutsBase):
	_you_were_warned = False  # TODO Use DesignViolation implementation instead

	def __init__(self, **kwargs):
		self.o = kwargs
		if not BlockOutsKwargs._you_were_warned and (
				sys.version_info[0] < 3 or (sys.version_info[0] == 3 and sys.version_info[1] < 6)):
			print('WARNING: keyword arguments constructor will not preserve output order before Python 3.6!\n' +
				  'Please use the empty constructor approach provided for backward compatibility:\n' +
				  'Eg: ' + type(self).__name__ + '().a(a_val).b(b_val)')
		self.oz = tuple(val for val in kwargs.values())


class BlockOutsAttrs(BlockOutsBase):
	def __init__(self):
		self.o = DictAttrs()
		self.oz = []


Out = DictAttrBuilderFactory(BlockOutsAttrs) if sys.version_info[0] < 3 or (
			sys.version_info[0] == 3 and sys.version_info[1] < 6) else BlockOutsKwargs

Props = DictAttrs  # Don't need DictAttrBuilderFactory since prop order does not need to be maintained


# TODO Add test for dynamic arg concept
def dynamic(*args):
	for arg in args:
		arg.__sandblox_dynamic__ = True
	if len(args) == 1:
		return args[0]
	return args


def is_dynamic_input(arg):
	return (hasattr(arg, '__sandblox_dynamic__') and getattr(arg, '__sandblox_dynamic__')) or (
			isinstance(arg, tf.Tensor) and arg.op.type == 'Placeholder')


def bind_resolved(fn, *args, **kwargs):
	bound_args = U.FlatArgumentsBinder(fn)(*args, **kwargs)
	resolved_args = OrderedDict([(key, resolve(value)) for key, value in bound_args.items()])
	return resolved_args
