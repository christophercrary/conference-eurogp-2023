"""Generic linear program node."""

class Node:
    """Class for generic linear program node."""

    __slots__ = (
        'opcode', 'depth', 'size', 'parent', 'value', 'name', 'arity',
        'function', 'terminal', 'variable', 'constant')

    def __init__(
        self, opcode=0, depth=0, size=1, parent=-1, value=0, 
            name='', arity=0, function=False, terminal=False, 
            variable=False, constant=False):
        # Core attributes.
        self.opcode = opcode
        self.depth = depth
        self.size = size
        self.parent = parent
        self.value = value

        # Extra attributes.
        self.name = name
        self.arity = arity
        self.function = function
        self.terminal = terminal
        self.variable = variable
        self.constant = constant

    def __str__(self):
        return self.name