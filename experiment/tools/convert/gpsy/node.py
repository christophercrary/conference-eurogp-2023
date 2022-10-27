"""Define module for a generic m-ary GP program node.

Most importantly, the `Node` class is defined.
Although named "Node", this class in fact operates 
on arbitrary m-ary program trees by considering a 
given `Node` as the root of some program tree and 
by recursively considering the `Node` instances 
accessible by way of the `children` class attribute 
as child subprograms of the overall program. Note 
that this is a conventional construction for a tree 
data structure.

Using the notation given above ultimately allows 
all `Node` methods to be directly applied to any 
`Node`, which is more flexible than having an 
explicit "Program" data structure, which would 
likely need an explicity attribute for "root node", 
among other things.
"""

from collections.abc import Hashable, Sequence
import inspect
import keyword
import re

from gpsy.primitive_set import PrimitiveSet

class Node:
    """Define generic m-ary GP program node."""

    __slots__ = ('name', 'children', 'code')

    def __init__(self, name=None, children=[]):
        """Initialize node.
        
        Keyword arguments:
        name -- Name of primitive. Must be hashable.
        children -- Sequence of children nodes, each of 
            class 'Node', given in a depth-first, 
            left-to-right ordering. (default: [])
        """
        if not isinstance(name, Hashable):
            raise TypeError(f'Name `{name}` is not hashable.')
        if not isinstance(children, Sequence):
            raise TypeError(f'Value provided for argument `children`, '
                            f'`{children}`, is not of type `Sequence`.')
        for child in children:
            if not isinstance(child, Node):
                raise TypeError(f'Child `{child}` is not of type `Node`.')
                  
        self.name = name
        self.children = children
        self.code = None

    @property
    def preorder(self):
        """Return tuple of node names given by pre-order traversal."""
        if not self.children:
            # Terminal node.
            return (self.name,)
        else:
            # Function node.
            return ((self.name,) 
                + tuple(c for _ in self.children for c in _.preorder))

    @property
    def preorder_str(self):
        """Return program string given by pre-order traversal.
        
        Pre-order traversal infers prefix (i.e., Polish) notation.
        Spacing is used instead of parentheses.
        """
        return ' '.join(self.preorder)

    @property
    def inorder(self):
        """Return tuple of node names given by in-order traversal."""
        if not self.children:
            return (self.name,)
        else:
            return (tuple(c for _ in self.children[:-1] for c in _.inorder)
                + (self.name,) 
                + tuple(c for c in self.children[-1].inorder))

    @property
    def inorder_str(self):
        """Return program string given by in-order traversal.
        
        In-order traversal infers infix notation. Parentheses 
        are added where it is necessary, and some spacing is 
        added in places where it may enhance readability.
        """
        if not self.children:
            return self.name
        else:
            # Strings for left children and right child, respectively.
            L = f'{" ".join([c.inorder_str for c in self.children[:-1]])}'
            R = f'{self.children[-1].inorder_str}'
            if L == '':
                # Only one child exists.
                return (f'({self.name} {R})')
            else:
                # More than one child exists.
                return (f'({L} {self.name} {R})')

    @property
    def postorder(self):
        """Return tuple of node names given by post-order traversal."""
        if not self.children:
            return (self.name,)
        else:
            return (tuple(c for _ in self.children for c in _.postorder)
                + (self.name,))

    @property
    def postorder_str(self):
        """Return program string given by post-order traversal.
        
        Post-order traversal infers postfix (i.e., Reverse Polish) 
        notation. Spacing is added instead of parentheses.
        """
        return ' '.join(self.postorder)

    @property
    def depth(self):
        """Return depth (i.e., height) of program."""
        if not self.children:
            return 0
        else:
            return max([1+child.depth for child in self.children])

    @property
    def size(self):
        """Return size of (i.e., number of nodes within) program."""
        return len(self.preorder)

    def __contains__(self, name):
        """Return whether or not a node with the given name exists."""
        return (name in self.preorder)

    def __len__(self):
        """Return length of program.
        
        The "length" of a program is defined to be the size
        of the program.
        """
        return self.size

    def __str__(self):
        """Return program string in prefix (i.e., Polish) notation.
        
        A minimal amount of parentheses are added to denote tree 
        structure, so that the following `compile` method can more 
        easily create a Python code object, and spacing is added 
        where it may enhance readability. All terminal nodes are
        given by just their name; if some name represents a function
        terminal, opening and closing parentheses to denote a call
        of the relevant function are not added until the program
        is compiled by the `compile` method. 
        
        It is primarily intended that this method be used by the 
        `compile` method, although it can be utilized whenever the 
        chosen notation is useful.
        """
        if not self.children:
            return self.name
        else:
            return (f'{self.name}('
                    f'{", ".join([str(c) for c in self.children])})')

    def __call__(self, *args):
        """Evaluate program rooted at `self`."""
        if not self.code:
            raise ValueError('Program has not yet been compiled '
                             'by the `compile` method.')
        return self.code(*args)

    def is_valid(self, primitive_set):
        """Determine if program is valid for given primitive set.
        
        For a given primitive set, a program is considered 
        "valid (in the context of the primitive set)" if all 
        symbols within the program exist within the set of 
        names given by the primitive set, and if every function 
        symbol within the program is associated with an 
        appropriate number of arguments.

        Keyword arguments:
        primitive_set -- Primitive set, of type `PrimitiveSet`.
        """
        if not isinstance(primitive_set, PrimitiveSet):
            raise TypeError(f'Value given for argument `primitive_set`, '
                            f'`{primitive_set}`, is not of type '
                            f'`PrimitiveSet`.')

        # Symbols given by the program. A postfix ordering is
        # used to be able to emulate program execution via a 
        # theoretical stack-based computer, as described below.
        symbols = self.postorder

        # The following emulates how a general stack-based computer 
        # might execute the relevant program expression. 
        # 
        # If the whole program is parsed without encountering 
        # an invalid symbol and only a single argument/result 
        # is left outstanding, then the program can be executed, 
        # which implies that the program is valid; otherwise, 
        # the program is invalid.

        # Number of arguments/results outstanding.
        num_args = 0

        for s in symbols:
            if s in primitive_set.terminals:
                # Terminal symbol encountered.
                num_args += 1
            elif s in primitive_set.functions:
                # Function symbol encountered. 
                # Get arity information regarding function symbol.
                a, va, ka, *_ = inspect.getfullargspec(
                    primitive_set.namespace[s])

                # Minimum arity of function.
                min_ = len(a)

                # Maximum arity of function, which depends on whether
                # or not there exist variable arguments.
                max_ = len(a) if va == None and ka == None else None

                if num_args < min_:
                    # Not enough arguments for function.
                    return False

                # Remove an appropriate number of arguments for the 
                # function, but add one back to account for the result 
                # of the function. (If the function accepts a variable 
                # number of arguments, all previously outstanding 
                # arguments/results are removed.)
                num_args = 1 if max_ is None else num_args - max_ + 1
            else:
                # Undefined symbol encountered.
                return False

        # Expression was fully parsed; Return whether or not 
        # there exists exactly one argument/result outstanding, 
        # which, at this point, is equivalent to whether or not 
        # the program is valid.
        return (num_args == 1)

    def compile(self, primitive_set):
        """Compile program rooted at `self` to a Python code object.

        If the argument `primitive_set` is of class `PrimitiveSet`, 
        and if the program is valid in the context of `primitive_set`
        (i.e., if all symbols within the program have meaning via 
        `primitive_set`), the attribute `self.code` is set to a 
        compiled code object that utilizes the variable terminals 
        of the given primitive set as arguments and the relevant 
        program expression as a body. All variable terminals are
        included as arguments, even if the program does not utilize 
        all of them, since this leads to more efficient evaluations 
        in the average context of GP; there becomes no need to 
        determine which particular variables should be passed when 
        a particular program is to be called.

        Keyword arguments:
        primitive_set -- Primitive set, of type `PrimitiveSet`.
        """
        if not isinstance(primitive_set, PrimitiveSet):
            raise TypeError(f'Value given for argument `primitive_set`, '
                            f'`{primitive_set}`, is not of type '
                            f'`PrimitiveSet`.')
        
        # Retrieve program string.
        code = str(self)
        
        # Determine if the program is valid in the context of
        # the given primitive set.
        if not self.is_valid(primitive_set):
            raise ValueError(f'Program `{code}` is not valid in the '
                             f'context of the given primitive set. The '
                             f'symbols defined by the primitive set are '
                             f'`{primitive_set.primitives}`.')

        # Append "()" to all terminal functions nodes, so that 
        # they are valid function calls.
        for ft in primitive_set.function_terminals:
            code = re.sub(ft, f'{ft}()', code)

        if len(primitive_set.variable_terminals) > 0:
            # Create a string representation of a code object.
            args = ','.join(primitive_set.variable_terminals)
            code = f'lambda {args}: {code}'
        try:
            # Attempt to parse the program string constructed above.
            self.code = eval(code, primitive_set.namespace)
        except MemoryError:
            print(f'Depth of program, {self.depth}, is too large to be '
                  f'evaluated by the Python interpreter.')

    @classmethod
    def from_str(cls, s):
        """Return `Node` object for given program string.
        
        The program string must be given in the same format as 
        that which is returned by the `__str__` method of the 
        `Node` class, although extra whitespace is permissible.
        A function terminal with name `name` can be given by 
        either `name` or `name()`.

        The aforementioned format was chosen so that program tree
        structure can be distinguished simply from string syntax, 
        and not from the semantics of any particular primitive set. 
        (For a format with no parentheses and just spacing, like
        given by the `preorder.str` and `postorder.str` methods of 
        the `Node` class, the semantics of a particular primitive 
        set would be needed to infer tree structure.)

        If the given string can represent a program within *some*
        (i.e., at least one) GP language, then a corresponding 
        `Node` object will be returned; otherwise, some exception 
        will be raised.

        Keyword arguments:
        s -- Program string, given in an appropriate format.
        """
        if not isinstance(s, str):
            raise TypeError(f'Value provided for the argument `s`, {s}, '
                            f'is not a string.')

        # Starting and ending indices, respectively, to track
        # certain substrings of `s`.
        i = 0
        j = 0

        # Two-dimensional stack (i.e., a stack of stacks) to 
        # construct a tree of `Node` objects.
        stack = [[]]

        while j < len(s):
            # Retrieve next character.
            c = s[j]

            if c == '(' or c == ',' or c == ')':
                # Consider the substring `s[i:j]` with all extra
                # whitespace removed. 
                substr = ' '.join(s[i:j].split())

                if c == '(':
                    # The string `substr` is to represent the name 
                    # of a `Node` object, and, thus, must be a Python 
                    # identifier that is not also a reserved keyword.
                    if not substr.isidentifier():
                        raise ValueError(f'Within string `{s}`, substring '
                                        f'`{substr}` is not a valid Python '
                                        f'identifier.')
                    if keyword.iskeyword(substr):
                        raise ValueError(f'Within string `{s}`, substring '
                                        f'`{substr}` is a Python keyword.')
                    # Assume that some amount of subprograms are to 
                    # follow, with each representing a child of the 
                    # aforementioned `Node` object with name `substr`. 
                    # Push the name of this potential parent node onto 
                    # the current one-dimensional stack.
                    stack[-1].append(substr) 

                    # Add a new one-dimensional stack for parsing the 
                    # supposed subprograms.
                    stack.append([])

                elif c == ',':
                    # Assume that another child subprogram is to follow. 
                    # Ensure that this is reasonable.
                    if (len(stack) == 1) or (substr == '' 
                        and (len(stack[-1]) == 0 or s[i-1] == ',')):
                        # Comma either (i) does not follow any open 
                        # parenthesis or (ii) immediately follows 
                        # either an open parenthesis or comma.
                        raise ValueError(f'Within string `{s}`, syntax error '
                                        f'at index {j}, character `{c}`.')
                    if substr != '':
                        # The substring `substr` is to represent the 
                        # name of a `Node` object.
                        if keyword.iskeyword(substr):
                            raise ValueError(f'Within string `{s}`, substring '
                                            f'`{substr}` is a Python keyword.')
                        if not substr.isidentifier():
                            try:
                                # Check if `substr` is an immutable 
                                # (i.e., constant) value.
                                eval(substr)
                            except SyntaxError:
                                raise ValueError(
                                    f'Within string `{s}`, substring '
                                    f'`{substr}` is neither a valid '
                                    f'Python identifier nor an '
                                    f'immutable (i.e., constant) '
                                    f'value.')
                            # The string `substr` is an immutable value.

                        # Within valid program strings, this case 
                        # only occurs when a terminal node has been 
                        # encountered. Thus, we can construct and 
                        # push a `Node` object to the relevant stack, 
                        # rather than just a name.
                        stack[-1].append(cls(name=substr, children=[]))

                    # Else, if `substr` is the empty string, then the 
                    # comma encountered follows some valid subprogram,
                    # and, thus, nothing needs to be done.

                else:
                    # Assume that some series of subprograms is to be 
                    # terminated. Ensure that this is reasonable.
                    if (len(stack) == 1) or (substr == '' and s[i-1] == ','):
                        # Closing parenthesis either (i) does not 
                        # followany open parenthesis or (ii) imm-
                        # ediately follows a comma.
                        raise ValueError(f'Within string `{s}`, syntax error '
                                        f'at index {j}, character `{c}`.')
                    if substr != '':
                        # The substring `substr` is to represent the 
                        # name of a `Node` object.
                        if keyword.iskeyword(substr):
                            raise ValueError(f'Within string `{s}`, substring '
                                            f'`{substr}` is a Python keyword.')
                        if not substr.isidentifier():
                            try:
                                # Check if `substr` is an immutable 
                                # (i.e., constant) value.
                                eval(substr)
                            except SyntaxError:
                                raise ValueError(
                                    f'Within string `{s}`, substring '
                                    f'`{substr}` is neither a valid '
                                    f'Python identifier nor an '
                                    f'immutable (i.e., constant) '
                                    f'value.')
                            # The string `substr` is an immutable value.

                        # Within valid program strings, this case 
                        # only occurs when a terminal node has been 
                        # encountered. Thus, we can construct and 
                        # push a `Node` object to the relevant stack, 
                        # rather than just a name.
                        stack[-1].append(cls(name=substr, children=[]))

                    # Terminate the latest series of programs by 
                    # replacing the last element of the previous 
                    # one-dimensional stack by a `Node` object 
                    # whose name is specified by this last element 
                    # of the previous one-dimensional stack and 
                    # whose children are the node objects specified 
                    # by the contents of the current one-dimensional 
                    # stack.
                    children = stack.pop()
                    name = stack[-1].pop()
                    stack[-1].append(cls(name=name, children=children))

                    if len(stack) == 1:
                        # A complete program string has been specified;
                        # break out of `while` loop.
                        break

                # Prepare for a new substring sequence.
                i = j+1

            else:
                # Some non-delimiter character was encountered.
                if (i-1 > 0 and s[i-1] == ')'):
                    # Closing parenthesis immediately precedes some 
                    # character other than a comma, opening paren-
                    # thesis, and closing parenthesis.
                    raise ValueError(f'Within string `{s}`, syntax error '
                                    f'at index {j}, character `{c}`.')

            # Move on to the following character, if one exists.
            j += 1

        # String parsing has completed.
        if j < len(s) - 1:
            # Some amount of characters follow the end 
            # of a valid program.
            raise ValueError(f'Within string `{s}`, syntax error '
                            f'at index {j+1}, character `{s[j+1]}`.')
        elif j == len(s):
            if len(stack) == 1:
                # The entire string `s` consists of no characters in 
                # the set ',()'. Assume that the string is to represent 
                # a program consisting of a single Node with name `s`.
                if keyword.iskeyword(s):
                    raise ValueError(f'String `{s}` is a Python keyword.')
                if not s.isidentifier():
                    try:
                        # Check if `substr` is an immutable value.
                        eval(s)
                    except SyntaxError:
                        raise ValueError(
                            f'String `{s}` is neither a valid Python '
                            f'identifier nor an immutable (i.e., '
                            f'constant) value.')
                    # The string `substr` is an immutable value.

                stack[0].append(cls(name=s, children=[]))
            else:
                # No character within the given string caused a syntax
                # error, but the program string does not have enough
                # characters to be a program. 
                raise ValueError(f'The string `{s}` is incomplete.')
                
        # The string represents a program in *some* language.
        return (stack[0].pop())

    def tensorgp_str(self, primitive_set):
        """Return program string for TensorGP.
        
        Constant terminal values `t` must be rewritten as 
        the string `scalar(t)`.

        This function will soon be deprecated.
        """
        if not self.children:
            if self.name in primitive_set.constant_terminals:
                return f'scalar({self.name})'
            else:
                return self.name
        else:
            ps = primitive_set
            return (
                f'{self.name}('
                f'{", ".join([c.tensorgp_str(ps) for c in self.children])})')