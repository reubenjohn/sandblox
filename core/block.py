from sandblox.util import *


# TODO Add tests for dynamic arg concept
def dynamic(*args):
	for arg in args:
		arg.is_d_inp = True
	if len(args) == 1:
		return args[0]
	return args


def is_dynamic_arg(arg):
	return hasattr(arg, 'is_d_inp') or (isinstance(arg, tf.Tensor) and arg.op.type == 'Placeholder')


class OptionalDynamicArg(object):
	__slots__ = 'default_arg'
	is_d_inp = True

	def __init__(self, default_arg=None):
		self.default_arg = default_arg

	def resolve(self, *args, **kwargs):
		return self.default_arg


def resolve(arg: OptionalDynamicArg, *args, **kwargs):
	return arg.resolve(*args, **kwargs) if isinstance(arg, OptionalDynamicArg) else arg


def flattened_dynamic_arguments(inps: dict) -> list:
	result = []
	for key in inps:
		inp = inps[key]
		if is_dynamic_arg(inp):
			result.append(inp.default_arg if isinstance(inp, OptionalDynamicArg) else inp)
		elif isinstance(inp, BlockBase):  # Do subclasses also evaluate to True?
			result.extend(inp.di)
	return result


# TODO Implement sandblox saving mechanism

# TODO Support forwarding of arguments to variable_scope
class BlockBase(object):
	scope = UninitializedScope()

	def __init__(self, **props_dict):
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
		self.i, self.iz, self.di = self._bind(*args, **kwargs)
		self._build(*args, **kwargs)
		return self

	def is_dynamic(self):
		return self.built_fn is None or any(b.is_dynamic() for b in self.iz if isinstance(b, BlockBase))

	# TODO Add test case
	def setup_scope(self, scope_name):
		self.scope = Scope(self, scope_name)

	def _bind(self, *args, **kwargs):
		input_args = U.FlatArgumentsBinder(self.build)(*args, **kwargs)
		i = DictAttrs(**input_args)
		iz = list(input_args.values())
		di = flattened_dynamic_arguments(input_args)
		return i, iz, di

	def _build(self, *args, **kwargs):
		if len(self.get_all_ops()) > 0:
			print('WARNING: Building ops into pollute d name scope')  # TODO Implement DesignViolation here
		with tf.variable_scope(self.scope.rel, reuse=self.reuse_var_scope):
			ret = self.build(*args, **kwargs)

		if isinstance(ret, Out.cls):
			block_outputs = ret
		elif hasattr(ret, '__len__') and len(ret) > 1 and isinstance(ret[0], Out.cls):
			block_outputs = ret[0]
		else:
			raise AssertionError(
				'A SandBlock must either return only a ' + type(Out).__name__
				+ ' or it must be the first element of what is returned'
			)
		self.o = block_outputs.o
		self.oz = block_outputs.oz

		return ret

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

	def get_all_ops(self) -> list:
		raise NotImplementedError

	def get_variables(self):
		raise NotImplementedError

	def assign_vars(self, source_block: 'BlockBase'):
		raise NotImplementedError

	def get_trainable_variables(self):
		raise NotImplementedError

	def assign_trainable_vars(self, source_block: 'BlockBase'):
		raise NotImplementedError
