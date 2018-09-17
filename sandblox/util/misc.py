# TODO Introduce DesignViolation escalation system

class DictAttrs(object):
	def __init__(self, **dic):
		self.__dict__.update(dic)

	def __iter__(self):
		return self.__dict__.__iter__()

	def __getitem__(self, item):
		return self.__dict__.__getitem__(item)

	def __getattr__(self, item):
		if not item in self.__dict__:
			raise LookupError

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


class DictAttrBuilderFactory(object):
	def __init__(self, attr_builder_class):
		self.cls = attr_builder_class

	def __getattr__(self, item):
		if item == 'cls':
			return self.cls
		return self.cls().__getattr__(item)
