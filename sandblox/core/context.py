class Context:
	def __init__(self, block):
		self.block = block

	def __enter__(self, *args, **kwargs):
		pass

	def __exit__(self, *args, **kwargs):
		pass


class StaticContext(Context):
	pass