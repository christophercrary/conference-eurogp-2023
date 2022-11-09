"""Generic linear program."""
import random
import re
import shlex

from pathos.pools import ProcessPool

from .node import Node

class Program(list):
    """Class for generic linear program."""
    __slots__ = ('nodes', 'code')

    def __init__(self, nodes=[]):
        super().__init__(nodes)
        self.code = None

    def subprogram(self, i):
        """Return subprogram rooted at the node whose index is `i`."""
        return Program(self[i : i + self[i].size])

    @property
    def depth(self):
        """Return depth (i.e., height) of program."""
        return self[0].depth

    @property
    def size(self):
        """Return size of (i.e., number of nodes within) program."""
        return len(self)

    @property
    def preorder(self):
        """Return list of nodes given by pre-order traversal."""
        return [n for n in self]

    @property
    def preorder_str(self):
        """Return program string given by pre-order traversal.
        
        Pre-order traversal infers prefix (i.e., Polish) notation.
        Spacing is used to separate node elements.
        """
        return ' '.join([n.name for n in self.preorder])

    @property
    def inorder(self):
        """Return tuple of node names given by in-order traversal."""
        stack = []
        for node in reversed(self):
            if node.function:
                operands = [stack.pop() for _ in range(node.arity)]
                stack.append([op_ for op in operands[:-1] for op_ in op] 
                    + [node] + operands[-1:])
            else:
                stack.append([node])
        return stack.pop() if stack != [] else []

    @property
    def inorder_str(self):
        """Return program string given by in-order traversal.
        
        In-order traversal infers infix notation. Parentheses 
        are added where it is necessary, and some spacing is 
        added in places where it may enhance readability.
        """
        stack = []
        for node in reversed(self):
            if node.function:
                operands = [stack.pop() for _ in range(node.arity)]
                L = ' '.join(operands[:-1])
                R = operands[-1]
                if L == '':
                    # Only one child exists.
                    stack.append(f'({node.name} {R})')
                else:
                    # More than one child exists.
                    stack.append(f'({L} {node.name} {R})')
            else:
                stack.append(node.name)
        s = stack.pop() if stack != [] else ''
        # Remove any outer parentheses.
        return s[1:-1] if s != '' and s[0] == '(' else s

    @property
    def postorder(self):
        """Return tuple of node names given by post-order traversal."""
        stack = []
        for node in reversed(self):
            if node.function:
                operands = [stack.pop() for _ in range(node.arity)]
                stack.append([op_ for op in operands for op_ in op] + [node])
            else:
                stack.append([node])
        return stack.pop() if stack != [] else []

    @property
    def postorder_str(self):
        """Return program string given by post-order traversal.
        
        Post-order traversal infers postfix (i.e., Reverse Polish) 
        notation. Spacing is used to separate node elements.
        """
        return ' '.join([n.name for n in self.postorder])

    def __str__(self):
        """Return program string in prefix (i.e., Polish) notation.
        
        A minimal amount of parentheses are added to denote tree 
        structure, so that the following `compile` method can more 
        easily create a Python code object, and spacing is added 
        where it may enhance readability.
        """
        stack = []
        for node in reversed(self):
            if node.function:
                operands = [stack.pop() for _ in range(node.arity)]
                stack.append((')' + ' ,'.join(
                    [op[::-1] for op in operands[::-1]]) 
                    + '(' + node.name[::-1])[::-1])
            else:
                stack.append(node.name)
        return stack.pop() if stack != [] else ''

    def __call__(self, *args):
        """Evaluate program."""
        if self.code is None:
            raise ValueError('Program has not yet been compiled '
                             'by the `compile` method.')
        return self.code(*args)

    def compile(self, primitive_set):
        """Compile program to a Python code object.

        The attribute `self.code` is set to a Python lambda expression
        that utilizes the variable terminals of the given primitive 
        set as arguments and the relevant program expression as a body.
        """
        # Retrieve program string.
        code = str(self)

        if len(primitive_set.variables) > 0:
            # Create a string representation of a code object.
            args = ', '.join(primitive_set.variables)
            code = f'lambda {args}: {code}'
        try:
            # Attempt to parse the program string constructed above.
            self.code = eval(
                code, {'__builtins__': None} | primitive_set.namespace)
        except MemoryError:
            print(f'Depth of program, {self.depth}, is too large to be '
                  f'evaluated by the Python interpreter.')
        return self.code

    @staticmethod
    def max_size(m, d):
        """Return maximum size for an `m`-ary program of depth `d`."""
        if m == 0 and d > 0:
            raise ValueError('For `m==0`, no program of depth f`{d}` exists.') 
        elif m < 0 or d < 0:
            return 0
        elif m == 0:
            return 1
        elif m == 1:
            return d + 1
        else:
            return (1 - m ** (d + 1)) // (1 - m)

    @staticmethod
    def from_str(s, ps):
        """Construct `Program` object from program string.
        
        It is assumed that the string gives the program in 
        a prefix (i.e., Polish) notation.
        
        Keyword arguments:
        s -- Prefix program string.
        ps -- `PrimitiveSet` object.
        """
        # Remove any extra whitespace, parentheses, and commas from
        # string except for substrings given between quotes, and 
        # split the resulting string into a list of node expressions.
        s = re.sub(r'([(,)])', r' \1 ', s)
        s = [' '.join(re.sub(r'[(,)]', r' ', s_).split())
            if s_[0] != '"' else s_ for s_ in shlex.split(s, posix=False)]
        s = [s_ for s_ in s if s_ != '']

        # Size of program.
        n = len(s)

        # Initialize list of `Node` objects.
        nodes = [Node() for _ in range(n)]

        for i, name in reversed(list(enumerate(s))):
            # Update the relevant `Node` element.
            node = nodes[i]
            node.name = name
            if name in ps.functions:
                # The string element represents a function node.
                arity = ps.arity(name)
                node.arity = arity
                node.opcode = 1 + list(ps.functions.keys()).index(name)
                node.function = True
                # Extract all child nodes associated with the function, 
                # and update some relevant node attributes.
                k = i + 1
                for _ in range(arity):
                    child = nodes[k]
                    child.parent = i
                    node.size += child.size
                    node.depth = max(node.depth, child.depth)
                    k += child.size
                node.depth += 1
            elif name in ps.variables:
                # The string element represents a variable node.
                node.opcode = (1 + len(ps.functions) + 1 
                    + list(ps.variables.keys()).index(name))
                node.terminal = True
                node.variable = True
            elif name in ps.constants:
                # The string element represents a constant function node.
                node.opcode = 1 + len(ps.functions)
                node.value = ps[name]()
                node.name = str(node.value)
                node.terminal = True
                node.constant = True
            else:
                # The string element represents a constant that is 
                # meant to be evaluated as a Python code expression.
                #
                # Remove outer quotations, if needed.
                name = name[1:-1] if name[0] == '"' else name
                node.opcode = 1 + len(ps.functions)
                node.value = eval(name, ps.namespace)
                node.name = str(node.value)
                node.terminal = True
                node.constant = True

        return Program(nodes)

    @staticmethod
    def _generate(primitive_set, d_max, s_max, d_min=0, s_min=1, trait='size'):
        """Generate a random program expression."""
        ps = primitive_set

        # Validate parameters.
        if len(ps.terminals) == 0:
            raise ValueError('Invalid primitive set.')
        elif d_min < 0 or d_max < 0 or d_max < d_min:
            raise ValueError('Invalid depth constraints.')
        elif (s_min < 1 or s_max < 1 or s_max < s_min or s_max < d_min or
            (len(ps.functions) == 0 and s_min > 1) or
            (s_min_possible := Program.max_size(ps.a_min, d_min)) > s_max or 
            (s_max_possible := Program.max_size(ps.a_max, d_max)) < s_min or
            s_max_possible < s_max):
            raise ValueError(
                f'Invalid size constraints. For the minimum/maximum '
                f'depth values {d_min} and {d_max}, respectively, '
                f'the minimum/maximum possible program sizes are '
                f'{s_min_possible} and {s_max_possible}, respectively.')
        elif trait != 'depth' and trait != 'size':
            raise ValueError('Invalid trait.')

        # The given parameters are valid; choose a desired 
        # depth/size value, depending on the specified trait.
        if trait == 'depth':
            desired = random.randint(d_min, d_max)
        elif trait == 'size':
            desired = random.randint(s_min, s_max)
            
        # Chosen primitives.
        names = []

        # Stack for constructing the program.
        stack = [(0, 1)]

        # Overall depth of the program.
        depth = 0

        while len(stack) != 0:
            # Retrieve the depth of the next relevant node
            # and the current size of the overall program.
            d, s = stack.pop()

            # Eliminate from consideration functions that will cause 
            # the program to violate given size constraints.
            # Additionally, if `trait == 'size'`, eliminate functions 
            # that will cause the program not to meet the desired 
            # size value; however, if after such an elimination 
            # there exist no valid functions, default to using the 
            # set of functions considered valid before this point 
            # (which may also be empty), so that at least the 
            # minimum/maximum size constraints may ultimately be met.

            # Eliminate functions that would cause the maximum
            # size constraint to be violated.
            valid_functions = [f for f in list(ps.functions.keys())
                if s + ps.arity(f) <= s_max]

            if trait == 'size':
                # Eliminate functions that would cause the desired
                # size constraint not to be met.
                temp_functions = [f for f in valid_functions 
                    if s + ps.arity(f) <= desired]
                # Update the set of valid functions if the current 
                # `if` block did not eliminate all functions. (The 
                # set of valid functions might already be empty.)
                if temp_functions != []:
                    valid_functions = temp_functions

            # Maximum arity for the current valid functions.
            m = (max([ps.arity(f) for f in valid_functions]) if
                valid_functions != [] else 0)

            # Eliminate functions that would cause the minimum
            # size constraint to be violated.
            temp_functions = valid_functions[:]
            for f in valid_functions:
                # Maximum possible size of subprogram rooted 
                # at the current node, excluding this node, 
                # if the current node is given to be `f`.
                # (For simplicity, we assume that all functions
                # with an arity not too big for the current node
                # could be continually be used.)
                s_ = ps.arity(f) * Program.max_size(m, d_max-(d+1))
                # Maximum possible program size if the relevant 
                # node under consideration was chosen to be the 
                # function `f`.
                s_ = s + s_ + sum([Program.max_size(m, d_max-d)-1 
                    for (d, *_) in stack]) if stack != [] else s + s_
                if s_ < s_min:
                    # The current function is not acceptable.
                    temp_functions.remove(f)
            valid_functions = temp_functions

            if trait == 'size':
                # Eliminate functions that would cause the desired
                # size constraint not to be met.
                temp_functions = valid_functions[:]
                for f in valid_functions:
                    # Maximum possible size of subprogram rooted 
                    # at the current node, excluding this node, 
                    # if the current node is given to be `f`.
                    # (For simplicity, we assume that all functions
                    # with an arity not too big for the current node
                    # could be continually be used.)
                    s_ = ps.arity(f) * Program.max_size(m, d_max-(d+1))
                    # Maximum possible program size if the relevant 
                    # node under consideration was chosen to be the 
                    # function `f`.
                    s_ = s + s_ + sum([Program.max_size(m, d_max-d)-1 
                        for (d, *_) in stack]) if stack != [] else s + s_
                    if s_ < desired:
                        # The current function is not acceptable.
                        temp_functions.remove(f)
                # Update the set of valid functions if the current 
                # `if` block did not eliminate all functions. (The 
                # set of valid functions might already be empty.)
                if temp_functions != []:
                    valid_functions = temp_functions

            # Maximum possible program size if the relevant node 
            # under consideration is chosen to be a terminal. This 
            # maximum size would occur if every outstanding node 
            # within the current stack is made to be the root of a 
            # full `m`-ary subtree such that the sum of the depth 
            # of this subtree and the depth of the root node within 
            # the overall program is equal to `d_max`.
            s_max_possible = s + sum([Program.max_size(m, d_max-d) - 1 
                for (d, *_) in stack]) if stack != [] and m > 0 else s

            # Determine if the current node should be a terminal.
            choose_terminal = (valid_functions == [] or 
                d == d_max or s == s_max or 
                (trait == 'depth' and d == desired) or
                (trait == 'size' and s >= desired) or 
                    (d >= d_min and s >= s_min and (trait == 'depth' or 
                        (trait == 'size' and s_max_possible >= desired)) and
                            random.random() < ps.terminal_proportion))

            if choose_terminal:
                # A random terminal is to be chosen.
                name = random.choice(list(ps.terminals.keys()))
                if name in ps.constants:
                    # Replace name with result of constant.
                    name = str(ps[name]())
                names.append(name)
                # If the stack is nonempty, update the size of the
                # next element to be equivalent to the size of the
                # current node under consideration, so that, upon
                # considering this upcoming element, the `s` value
                # specified by this element will accurately represent 
                # the current program size.
                if len(stack) != 0:
                    d, _ = stack.pop()
                    stack.append((d, s))
            else:
                # A random valid function is to be chosen.
                name = random.choice(valid_functions) 
                names.append(name)
                # Add a placeholder stack element for each 
                # argument needed by the chosen function.
                arity = ps.arity(name)
                for _ in range(arity):
                    stack.append((d + 1, s + arity))
                # Update the overall program depth if appropriate.
                if d + 1 > depth:
                    depth = d + 1

        # All program nodes have been chosen; determine if the 
        # program constraints have been met.
        if depth < d_min or depth > d_max or s < s_min or s > s_max:
            raise ValueError('No program exists for the given constraints.')
        else:
            # Construct a `Program` object from the chosen nodes.
            return Program.from_str(' '.join(names), ps)

    @staticmethod
    def generate(primitive_set, d_max, s_max, d_min=0, s_min=1, 
        trait='size', n_programs=1, n_threads=1):
        """Generate some number of random program expressions.
        
        Parallel processing is used based on the `n_threads` parameter.
        """
        if n_threads == -1:
            # Use all available threads.
            n_threads = None
        with ProcessPool(n_threads) as pool:
            programs = pool.map(lambda _ : Program._generate(
                primitive_set=primitive_set, d_max=d_max, s_max=s_max, 
                d_min=d_min, s_min=s_min, trait=trait), range(n_programs))
        return programs
