"""Extension for generic linear program."""
from gp.core import node, program
from gp.core.math import clog
from .node import Node

class Program(program.Program):
    """Class extension for generic linear program.
    
    It is expected that node objects within the
    program expression will be of type `gp.hw.node.Node`,
    not `gp.core.node.Node`.
    """
    def machine_code(self, w_opcode=16, w_depth=16, w_value=32, form='x'):
        """Return machine codes for `Program` object.
        
        A "null word" is appended to the end of the program.
        """
        return [node.machine_code(w_opcode, w_depth, w_value, form) 
            for node in self + [Node()]]

    @staticmethod
    def min_depth(s, primitive_set):
        """Return minimum program depth for program of size `s`."""
        return clog(1 + s * (primitive_set.m - 1), primitive_set.m) - 1

    @staticmethod
    def max_terminal_nodes(s, primitive_set):
        """Return maximum number of terminal nodes for program."""
        # Ary-ness of primitive set.
        m = primitive_set.m

        # Counter for the maximum number of terminal nodes.
        n = 0

        while s != 0:
            # Minimum depth for an m-ary tree consisting of `s` nodes.
            d = Program.min_depth(s, primitive_set)

            if d > 0:
                # There are between `m**(d-1)` and `m**d` additional 
                # possible terminal nodes. In other words, there are
                # between 1 and `m` full subtrees of size `(1-m**d)/(1-m)`
                # that contribute to the maximum number of terminal nodes, 
                # where some nonzero remainder of size may exist after 
                # considering these full subtrees. Compute this number of 
                # subtrees, `c`, and the relevant size remainder, `r`.
                c, r = divmod(s - 1, (1 - m ** d) / (1 - m))
                # Update the maximum possible number of terminal
                # nodes and the program size variable `s`.
                n += int(c * (m ** (d - 1)))
                s = r
            else:
                # There is exactly one additional possible terminal
                # node, with no remainder of size.
                n += 1
                s = 0
        return n

    @staticmethod
    def from_str(s, ps):
        """Construct `Program` object from program string.
        
        It is assumed that the string gives the program in 
        a prefix (i.e., Polish) notation.
        """
        # Extract a `gp.core.program.Program` object, and convert
        # all `gp.core.node.Node` objects within the program to
        # `gp.hw.node.Node` objects.
        nodes = program.Program.from_str(s, ps)[:]
        return Program([Node(
            opcode=n.opcode, depth=n.depth, size=n.size, parent=n.parent, 
            value=n.value, name=n.name, arity=n.arity, function=n.function, 
            terminal=n.terminal, variable=n.variable, constant=n.constant) 
                for n in nodes])

    @staticmethod
    def generate(primitive_set, d_max, s_max, d_min=0, s_min=1, trait='size',
        n_programs=1, n_threads=1):
        """Generate some amount of random program expressions."""
        programs = []
        programs_ = program.Program.generate(
            primitive_set=primitive_set, d_max=d_max, s_max=s_max, 
            d_min=d_min, s_min=s_min, trait=trait, n_programs=n_programs,
            n_threads=n_threads)
        for program_ in programs_:
            # Convert all `gp.core.node.Node` objects within the program to
            # `gp.hw.node.Node` objects.
            nodes = program_[:]
            programs.append(Program([Node(
                opcode=n.opcode, depth=n.depth, size=n.size, parent=n.parent, 
                value=n.value, name=n.name, arity=n.arity, function=n.function, 
                terminal=n.terminal, variable=n.variable, constant=n.constant) 
                for n in nodes]))
        return programs