import sys


# TODO Introduce DesignViolation escalation system

class DictAttrs(object):
	def __init__(self, **dic):
		self.__dict__.update(dic)

	def __iter__(self):
		return self.__dict__.__iter__()

	def __getitem__(self, item):
		return self.__dict__.__getitem__(item)

	# To prevent unhelpful lint warnings
	def __getattr__(self, item):
		pass

	def __setitem__(self, key, value):
		self.__dict__.__setitem__(key, value)

	def __str__(self):
		return self.__dict__.__str__()


class DictAttrBuilder:
	def _on_new_attr_val(self, key, val):
		raise NotImplementedError

	def _new_attr_val(self, key, val):
		self._on_new_attr_val(key, val)
		return self

	def __getattr__(self, item):
		return lambda val: self._new_attr_val(item, val)


class BlockOutsBase(DictAttrBuilder):
	def _on_new_attr_val(self, key, val):
		if key in self.o:
			print('Warning an output named %s already exists with value: %s' % (key, self.o[key]))
		self.o[key] = val
		self.oz.append(val)


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


class DictAttrBuilderFactory(object):
	def __init__(self, attr_builder_class):
		self.cls = attr_builder_class

	def __getattr__(self, item):
		if item == 'cls':
			return self.cls
		return self.cls().__getattr__(item)


Out = DictAttrBuilderFactory(BlockOutsAttrs) if sys.version_info[0] < 3 or (
	sys.version_info[0] == 3 and sys.version_info[1] < 6) else BlockOutsKwargs

Props = DictAttrs  # Don't need DictAttrBuilderFactory since prop order does not need to be maintained
