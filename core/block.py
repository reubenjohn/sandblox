from collections import OrderedDict
from typing import Any, Callable

from sandblox.core.io import *
from sandblox.core.io import BlockOutsBase
from sandblox.util import *

python_less_than_3_3 = sys.version_info[0] < 3 and sys.version_info[1] < 3


class NotBuiltError(AssertionError):
	def __init__(self, *args: Any, block=None) -> None:
		self.block = block
		if len(args) > 0:
			msg = args[0]
			args = args[1:]
			msg = msg + ': ' + str(self.block)
			super(NotBuiltError, self).__init__(msg, *args)
		else:
			msg = 'Block "{}" has not yet been built'.format(self.block)
			super(NotBuiltError, self).__init__(msg)


class LateBoundArg(object):
	def __init__(self, resolver: Callable):
		self.resolver = resolver

	def resolve(self):
		return self.resolver()


def resolve(*args):
	resolved = [arg.resolve() if isinstance(arg, LateBoundArg) else arg for arg in args]
	return resolved if len(resolved) > 1 else resolved[0]


def flattened_dynamic_arguments(inps: dict) -> list:
	result = []
	for key in inps:
		inp = inps[key]
		if is_dynamic_arg(inp):
			result.append(resolve(inp))
		elif isinstance(inp, BlockBase):
			if not inp.is_built():
				raise NotBuiltError(block=inp)
			result.extend(inp.di)
	return result


# TODO Support forwarding of arguments to variable_scope
# TODO Implement sandblox saving mechanism


def bind_resolved(fn, *args, **kwargs):
	bound_args = U.FlatArgumentsBinder(fn)(*args, **kwargs)
	resolved_args = OrderedDict([(key, resolve(value)) for key, value in bound_args.items()])
	return resolved_args


class BlockBase(object):
	scope = UninitializedScope()

	def __init__(self, **props_dict):
		self._is_built = False
		self.i = self.o = self.iz = self.oz = self.di = self.built_fn = None
		self.props = Props(**props_dict)
		self.scope = Scope(self, props_dict.get('scope_name'))
		self.reuse_var_scope = props_dict.get('reuse', None)
		# TODO Test name collision when explicitly specified names for two blocks are the same, and the lack thereof
		if props_dict.get('make_scope_unique', True):
			graph = props_dict.get('graph', tf.get_default_graph())
			assert graph is not None, 'Could not find a default graph, so a graph must be provided since make_scope_unique is True'
			self.scope.make_unique(graph)

	def build_graph(self, *args, **kwargs):
		self.i, self.iz, self.di, bound_args = self._bind(*args, **kwargs)

		with tf.variable_scope(self.scope.rel, reuse=self.reuse_var_scope):
			if len(self.get_all_ops(tf.get_variable_scope().name)) > 0:
				print('WARNING: Building ops into pollute d name scope')  # TODO Implement DesignViolation here
			out = self.build_wrapper(**bound_args)
		self.o = out.o
		self.oz = out.oz
		self._is_built = True

		return self

	def is_built(self):
		return self._is_built

	def is_dynamic(self):
		return self.built_fn is None or any(b.is_dynamic() for b in self.iz if isinstance(b, BlockBase))

	# TODO Add test case
	def setup_scope(self, scope_name):
		self.scope = Scope(self, scope_name)

	def _bind(self, *args, **kwargs):
		input_args = bind_resolved(self.build, *args, **kwargs)
		i = DictAttrs(**input_args)
		iz = list(input_args.values())
		di = flattened_dynamic_arguments(input_args)
		return i, iz, di, input_args

	def build_wrapper(self, *args, **kwargs) -> BlockOutsBase:
		ret = self.build(*args, **kwargs)

		if isinstance(ret, Out.cls):
			return ret
		elif hasattr(ret, '__len__') and len(ret) > 1 and isinstance(ret[0], Out.cls):
			return ret[0]
		else:
			raise AssertionError(
				'A SandBlock must either return only a ' + type(Out).__name__
				+ ' or it must be the first element of what is returned'
			)

	def build(self, *args, **kwargs):
		raise NotImplementedError

	def run(self, *args, **kwargs):
		self.process_inputs(*args, **kwargs)
		dynamic_oz = self.eval(*args, **kwargs)
		self.post_eval(dynamic_oz)
		return dynamic_oz

	def process_inputs(self, *args, **kwargs):
		pass

	def eval(self, *args, **kwargs):
		raise NotImplementedError

	def post_eval(self, outputs):
		for inp in self.iz:
			if isinstance(inp, BlockBase):
				inp.post_eval(outputs)
		self.post_my_eval(outputs)

	# TODO Fix this disgusting design :(
	def post_my_eval(self, outputs):
		pass

	def get_all_ops(self, scope_name: str = None) -> list:
		raise NotImplementedError

	def get_variables(self):
		raise NotImplementedError

	def assign_vars(self, source_block: 'BlockBase'):
		raise NotImplementedError

	def get_trainable_variables(self):
		raise NotImplementedError

	def assign_trainable_vars(self, source_block: 'BlockBase'):
		raise NotImplementedError

	def __str__(self):
		return '%s.%s:/%s' % (
			self.__module__, type(self).__name__ if python_less_than_3_3 else type(self).__qualname__, self.scope.abs)
