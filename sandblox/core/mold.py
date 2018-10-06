from sandblox.core.block import Block
from sandblox.core.io import Props


# noinspection PyAbstractClass
class Mold(Block):
	def __init__(self, **default_props):
		self.default_props_dict = default_props
		super(Mold, self).__init__(**default_props)

	def __call__(self, *args, **kwargs):
		props = dict(**self.default_props_dict)
		props.update(kwargs.pop('props', Props()).__dict__)
		block = type(self)(**props)
		block.build_graph(*args, **kwargs)
		return block
