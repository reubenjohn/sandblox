from typing import Type, Callable

from sandblox.core.block import Block
from sandblox.core.io import Props
from sandblox.core.mold import Mold


class Function(Mold):
	def static(self, *args, **kwargs):
		raise NotImplementedError

	def get_all_ops(self, scope_name: str = None) -> list:
		raise NotImplementedError

	def get_variables(self):
		raise NotImplementedError

	def assign_vars(self, source_block: Block):
		raise NotImplementedError

	def get_trainable_variables(self):
		raise NotImplementedError

	def assign_trainable_vars(self, source_block: Block):
		raise NotImplementedError


def instantiate_block(cls: Type[Mold], fn_name, default_props: Props = None):
	if default_props is None:
		default_props = Props()
	if 'scope_name' in default_props:
		default_props.scope_name = fn_name
	block_fn_instance = cls(**default_props.__dict__)  # type: Type[Mold]

	return block_fn_instance


def fn_to_built_block(fn: Callable, base_cls: Type[Mold], def_props: Props = None):
	# noinspection PyAbstractClass
	class FnBuiltBlock(base_cls):
		static = fn

		def __init__(self, **default_props):
			self.static = fn
			super(FnBuiltBlock, self).__init__(**default_props)

	return instantiate_block(FnBuiltBlock, fn.__name__, def_props)
