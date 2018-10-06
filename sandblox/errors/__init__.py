from typing import Any


class NotBuiltError(AssertionError):
	def __init__(self, *args: Any, block=None) -> None:
		self.block = block
		if len(args) > 0:
			msg = args[0]
			args = args[1:]
			msg = msg + ': ' + str(self.block)
			super(NotBuiltError, self).__init__(msg, *args)
		else:
			msg = 'Mold "{}" has not been built'.format(self.block)
			super(NotBuiltError, self).__init__(msg)


class BlockNotDynamicException(NotImplementedError):
	def __init__(self, block=None) -> None:
		self.block = block
		msg = 'Mold "{}" is not dynamic'.format(self.block)
		super(BlockNotDynamicException, self).__init__(msg)
