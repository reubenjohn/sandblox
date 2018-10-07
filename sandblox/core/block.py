from inspect import signature

import sandblox.errors as errors
from sandblox.constants.internal import python_less_than_3_3
from sandblox.core.arg import resolve
from sandblox.core.context import StaticContext
from sandblox.core.io import Props, Out, is_dynamic_input
from sandblox.core.io import bind_resolved, BlockOutsBase
from sandblox.util.misc import DictAttrs
from sandblox.util.scope import Scope
from sandblox.util.scope import UninitializedScope


def flattened_dynamic_inputs(inps: dict) -> list:
	result = []
	for key in inps:
		inp = inps[key]
		if is_dynamic_input(inp):
			result.append(resolve(inp))
		elif isinstance(inp, Block):
			if not inp.is_built():
				raise errors.NotBuiltError(block=inp)
			result.extend(inp.di)
	return result


class Block(object):
	# TODO Support forwarding of arguments to variable_scope
	# TODO Implement sandblox saving mechanism
	scope = UninitializedScope()

	def __init__(self, **props_dict):
		self._is_dynamic = None
		self._is_built = False
		self._self_givens = {}
		self.i = self.o = self.iz = self.oz = self.di = self.built_fn = None
		self.props = Props(**props_dict)
		self.scope = Scope(props_dict.get('scope_name'), self)
		self.reuse_var_scope = props_dict.get('reuse', None)

	# TODO Add test case
	def setup_scope(self, scope_name):
		self.scope = Scope(scope_name, self)

	@property
	def is_dynamic(self):
		if self._is_dynamic is None:
			self._is_dynamic = self.compute_is_dynamic()
		return self._is_dynamic or any(b.is_dynamic for b in self.iz if isinstance(b, Block))

	def compute_is_dynamic(self):
		self._is_dynamic = True
		try:
			sig = signature(self.dynamic)
			self.dynamic([None] * len(sig.parameters))
		except errors.BlockNotDynamicError:
			self._is_dynamic = False
		except Exception:
			self._is_dynamic = True
		return self.is_dynamic or any(b.is_dynamic for b in self.iz if isinstance(b, Block))

	def is_built(self):
		return self._is_built

	def _static_context(self):
		return StaticContext(self)

	def setup_static(self, *args, **kwargs):
		assert not self._is_built, 'Block already built'

		with self._static_context():
			self.i, self.iz, self.di, bound_args = self._bind(*args, **kwargs)
			out = self._wrap_static(**bound_args)

		self.o = out.o
		self.oz = out.oz

		self._is_built = True

	def _bind(self, *args, **kwargs):
		input_args = bind_resolved(self.static, *args, **kwargs)
		i = DictAttrs(**input_args)
		iz = list(input_args.values())
		di = flattened_dynamic_inputs(input_args)
		return i, iz, di, input_args

	def _wrap_static(self, *args, **kwargs) -> BlockOutsBase:
		ret = self.static(*args, **kwargs)

		if isinstance(ret, Out.cls):
			return ret
		else:
			raise AssertionError('A SandBlock must either return only a ' + type(Out).__name__)

	def static(self, *args, **kwargs):
		raise NotImplementedError

	def run(self, *args, **kwargs):
		dynamic_outputs = self._static_run(self.givens(), *args, **kwargs)
		if self.is_dynamic:
			dynamic_outputs = self._dynamic_run(dynamic_outputs, *args, **kwargs)
		return dynamic_outputs

	def givens(self) -> dict:
		givens = {}
		for inp in self.iz:
			if isinstance(inp, Block):
				givens.update(inp.givens())
		givens.update(self.self_givens())
		return givens

	def self_givens(self):
		return self._self_givens

	def _static_run(self, givens, *args, **kwargs):
		return None

	def _dynamic_run(self, dynamic_outputs, *args, **kwargs):
		dynamic_outputs = self.dynamic(dynamic_outputs, *args, **kwargs)
		self._recurse_dynamic_results(dynamic_outputs)
		return dynamic_outputs

	def dynamic(self, static_outputs, *args, **kwargs):
		raise errors.BlockNotDynamicError(self)

	def _recurse_dynamic_results(self, outputs):
		for inp in self.iz:
			if isinstance(inp, Block):
				inp._recurse_dynamic_results(outputs)
		self._post_dynamic(outputs)

	# TODO Fix this disgusting design :(
	def _post_dynamic(self, outputs):
		pass

	def get_all_ops(self, scope_name: str = None) -> list:
		raise NotImplementedError

	def get_variables(self):
		raise NotImplementedError

	def assign_vars(self, source_block: 'Block'):
		raise NotImplementedError

	def get_trainable_variables(self):
		raise NotImplementedError

	def assign_trainable_vars(self, source_block: 'Block'):
		raise NotImplementedError

	def __str__(self):
		return '%s.%s:/%s' % (
			self.__module__, type(self).__name__ if python_less_than_3_3 else type(self).__qualname__, self.scope.abs)
