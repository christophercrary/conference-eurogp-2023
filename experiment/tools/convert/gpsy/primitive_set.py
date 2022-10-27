"""Define module for generic GP primitive set."""

from collections.abc import Hashable, Sequence
import inspect
from itertools import count, filterfalse
import keyword
import math
import random
import re

class PrimitiveSet:
    """Define generic GP primitive set."""

    __slots__ = (
        'functions',
        'variable_terminals',
        'constant_terminals',
        'function_terminals',
        'namespace')

    def __init__(self):
        """Initialize primitive set."""
        # Lists to store names for certain primitive types.
        self.functions = []
        self.variable_terminals = []
        self.constant_terminals = []
        self.function_terminals = []
    
        # Namespace dictionary to map the name of each function, 
        # constant terminal, and function terminal primitive to 
        # an appropriate Python object. Such a dictionary will 
        # provide context for how to evaluate certain GP program 
        # expressions created by the `Node` class. 
        #
        # Note that variable terminals are not included within 
        # the namespace dictionary, since these will instead be 
        # parameters for any Python code object associated with a 
        # GP program. (Refer to the `compile` method of the
        # `Node` class for more details.)
        #
        # Also, note that the `namespace` attribute explicitly
        # includes a mapping for built-in Python functions, where
        # this mapping in fact omits these functions from the 
        # primitive set namespace, so that the built-in `eval` 
        # function can execute faster when evaluating GP programs. 
        # (Refer to documentation for the `eval` function for more 
        # details.)
        self.namespace = {'__builtins__': None}

    @property
    def terminals(self):
        """Return the list of names for all terminal primitives.
        
        In the returned list, variable terminals are given
        before constant terminals, which are given before
        function terminals.
        """
        return (tuple(self.variable_terminals)
            + tuple(self.constant_terminals)
            + tuple(self.function_terminals))

    @property
    def primitives(self):
        """Return the list of names for all primitives.
        
        In the returned list, functions are given before
        variable terminals, which are given before constant
        terminals, which are given before function terminals.
        """
        return (tuple(self.functions) + self.terminals)

    def __contains__(self, name):
        """Return whether or not `name` exists in the primitive set.
        
        Returns `True` if there exists a primitive with name `name`,
        returns `False` otherwise.
        """
        return (name in self.primitives)

    def __len__(self):
        """Return length of primitive set.
        
        The "length" of a primitive set is defined to be 
        the length of the list of all primitive names.
        """
        return len(self.primitives)

    def add_function(self, function, name=None):
        """Add function to primitive set.

        If `name` is `None` and `function` is callable,
        the name `function.__name__` is provided; if 
        `function` is not callable, a `TypeError` exception
        is raised.

        Any value other than `None` given for `name`  must 
        be a valid Python identifier that is not also a
        reserved keyword. If the provided name already exists 
        within the the primitive set, a `ValueError` exception 
        is raised.
        
        Keyword arguments:
        function -- Python function with arity greater than 
            zero to implement primitive.
        name -- Name for function. Must be either `None` 
            or a valid Python identifier that is not a
            reserved keyword. (default: None)
        """
        try:
            args, *_ = inspect.getfullargspec(function)
        except TypeError:
            print(f'Value provided for argument `function`, `{function}`, '
                  f'is invalid.')
            raise

        if len(args) == 0:
            raise ValueError(f'Function `{function}` has zero arguments.')

        if name is None:
            name = function.__name__
        else:
            if not isinstance(name, str):
                raise TypeError(f'Name `{name}` is not a string.')
            if not name.isidentifier():
                raise ValueError(f'Name `{name}` is not a Python identifier.')
            if keyword.iskeyword(name):
                raise ValueError(f'Name `{name}` is a Python keyword.')

        if name in self.primitives:
            raise ValueError(f'Name `{name}` already exists within '
                             f'the primitive set.')

        self.functions.append(name)
        self.namespace[name] = function    

    def add_variable_terminal(self, name=None):
        """Add variable terminal.
        
        If `name` is `None`, a name of the form `f'v{m}'` 
        will be provided, with `m` being the smallest 
        nonnegative integer such that `f'v{m}'` is not 
        in the set of primitive names.

        Any value other than `None` given for `name`  must 
        be a valid Python identifier that is not also a
        reserved keyword. If the provided name already exists 
        within the the primitive set, a `ValueError` exception 
        is raised.

        Keyword arguments:
        name -- Name for variable terminal. Must be either 
            `None` or a valid Python identifier that is not 
            a reserved keyword. (default: None)
        """
        if name is None:
            # Relevant regular expression.
            r = re.compile('v[0-9]+')

            # Set of integers `m` such that `f'v{m}'` exists
            # in `self.primitives`.
            M = set([int(''.join(m)) for _,*m in 
                list(filter(r.match, self.primitives))])

            # Smallest nonnegative integer `m` such that
            # `f'v{m}'` does not exist in `self.primitives`.
            m = (next(filterfalse(M.__contains__, count(0))))

            # Construct name.
            name = (f'v{str(m)}')
        else:
            if not isinstance(name, str):
                raise TypeError(f'Name `{name}` is not a string.')
            if not name.isidentifier():
                raise ValueError(f'Name `{name}` is not a Python identifier.')
            if keyword.iskeyword(name):
                raise ValueError(f'Name `{name}` is a Python keyword.')
            if name in self.primitives:
                raise ValueError(f'Name `{name}` already exists in '
                                f'the primitive set.')
        
        self.variable_terminals.append(name)

    def add_constant_terminal(self, constant, name=None):
        """Add constant terminal.

        If `name` is `None` and `constant` is both immutable (i.e., 
        hashable) and not callable, then a string representation of 
        `constant` will be used as the name. Any value other than 
        `None` given for `name`  must be a valid Python identifier 
        that is not also a reserved keyword. If the provided name 
        already exists within the the primitive set, a `ValueError` 
        exception is raised.
        
        Keyword arguments:
        constant -- Constant value for terminal. Must be 
            both immutable (i.e., hashable) and not callable.
        names -- Name for constant terminal. Must be either 
            `None` or a valid Python identifier that is not 
            a reserved keyword. (default: None)
        """
        if not isinstance(constant, Hashable):
            raise TypeError(f'Value provided for argument `constant`, '
                            f'`{constant}`, is not hashable.')
        if callable(constant):
            raise TypeError(f'Value provided for argument `constant`, '
                            f'`{constant}`, is callable.')
                        
        if name is None:
            name = str(constant) 
        else:
            if not isinstance(name, str):
                raise TypeError(f'Name `{name}` is not a string.')
            if not name.isidentifier():
                raise ValueError(f'Name `{name}` is not a Python identifier.')
            if keyword.iskeyword(name):
                raise ValueError(f'Name `{name}` is a Python keyword.')

        if name in self.primitives:
            raise ValueError(f'Name `{name}` already exists in '
                             f'the primitive set.')

        self.constant_terminals.append(name)
        self.namespace[name] = constant

    def add_function_terminal(self, function, name=None):
        """Add function terminal.

        If `name` is `None` and `function` is callable,
        the name `function.__name__` is provided; if 
        `function` is not callable, a `TypeError` exception
        is raised.

        Any value other than `None` given for `name`  must 
        be a valid Python identifier that is not also a
        reserved keyword. If the provided name already exists 
        within the the primitive set, a `ValueError` exception 
        is raised.
        
        Keyword arguments:
        function -- Zero-arity Python function to implement 
            primitive.
        name -- Name for function terminal. Must be either 
            `None` or a valid Python identifier that is not 
            a reserved keyword. (default: None)
        """
        try:
            args, *_ = inspect.getfullargspec(function)
        except TypeError:
            print(f'Value provided for argument `function`, `{function}`, '
                  f'is not callable.')
            raise

        if len(args) != 0:
            raise ValueError(f'Function `{function}` has more '
                             f'than zero arguments.')

        if name is None:
            name = function.__name__
        else:
            if not isinstance(name, str):
                raise TypeError(f'Name `{name}` is not a string.')
            if not name.isidentifier():
                raise ValueError(f'Name `{name}` is not a Python identifier.')
            if keyword.iskeyword(name):
                raise ValueError(f'Name `{name}` is a Python keyword.')

        if name in self.primitives:
            raise ValueError(f'Name `{name}` already exists in '
                             f'the primitive set.')

        self.function_terminals.append(name)
        self.namespace[name] = function        

    def add_random_constant(self, seq=None, rand=None):
        """Add random constant.

        A constant can be drawn uniformly from a given sequence
        if `seq` is a Python sequence, or drawn via `rand` if 
        `seq` is `None` and `rand` is callable. If both `seq` 
        and `rand` are `None`, then a constant is drawn via the
        built-in `random.random` Python method.

        No attempt is made to ensure valid functionality of 
        `rand`. Also, no attempt is made to ensure that any 
        resulting constant is distinct from those that already 
        exist within the primitive set namespace.
        
        Keyword arguments:
        seq -- Python sequence from which to draw constants. 
            (default: None)
        rand -- Function through which to draw constants.
            (default: None)
        """
        if seq is None and rand is None:
            rand = random.random
        elif isinstance(seq, Sequence):
            rand = lambda: random.choice(seq)

        # Else, the value given for `rand` is used.

        try:
            # Retrieve random constant by way of `rand`.
            constant = rand()
        except TypeError:
            raise TypeError(f'Value provided for argument `rand`, '
                            f'`{rand}`, is not callable.')

        # Add random constant to the primitive set.
        self.add_constant_terminal(constant)

    def remove_function(self, name, default=None):
        """Remove function, if it exists.
        
        If a function with name `name` exists, the value 
        `None` is returned.

        If a function with name `name` does not exist
        and `default` is not `None`, then `default` is
        returned; otherwise, a `ValueError` exception is
        raised.

        Keyword arguments:
        name -- Name of function.
        default -- Default return value. (default: None)
        """
        try:
            self.functions.remove(name)
            self.namespace.pop(name)
        except ValueError:
            if default is None:
                print(f'Name `{name}` is not in primitive set.')
                raise
            return default
        else:
            return None
                
    def remove_variable_terminal(self, name, default=None):
        """Remove variable terminal, if it exists.
        
        If a variable terminal with name `name` exists, 
        the value `None` is returned.

        If a variable terminal with name `name` does not 
        exist and `default` is not `None`, then `default` 
        is returned; otherwise, a `ValueError` exception is
        raised.

        Keyword arguments:
        name -- Name of variable terminal.
        default -- Default return value. (default: None)
        """
        try:
            self.variable_terminals.remove(name)
        except ValueError:
            if default is None:
                print(f'Name `{name}` is not in primitive set.')
                raise
            return default
        else:
            return None

    def remove_constant_terminal(self, name, default=None):
        """Remove constant terminal, if it exists.
        
        If a constant terminal with name `name` exists, 
        the value `None` is returned.

        If a constant terminal with name `name` does not 
        exist and `default` is not `None`, then `default` 
        is returned; otherwise, a `ValueError` exception is
        raised.

        Keyword arguments:
        name -- Name of constant terminal.
        default -- Default return value. (default: None)
        """
        try:
            self.constant_terminals.remove(name)
            self.namespace.pop(name)
        except ValueError:
            if default is None:
                print(f'Name `{name}` is not in primitive set.')
                raise
            return default
        else:
            return None          

    def remove_function_terminal(self, name, default=None):
        """Remove function terminal, if it exists.
        
        If a function terminal with name `name` exists, 
        the value `None` is returned.

        If a function terminal with name `name` does not 
        exist and `default` is not `None`, then `default` 
        is returned; otherwise, a `ValueError` exception is
        raised.

        Keyword arguments:
        name -- Name of function terminal.
        default -- Default return value. (default: None)
        """
        try:
            self.function_terminals.remove(name)
            self.namespace.pop(name)
        except ValueError:
            if default is None:
                print(f'Name `{name}` is not in primitive set.')
                raise
            return default
        else:
            return None

    @property
    def assembly_language(self):
        """Return tuple of assembly language words.
        
        For every `PrimitiveSet` object, an assembly language is 
        inferred in the following manner, for the purposes of a 
        specialized hardware architecture:

        (1) An end-of-program word, which is used to separate 
            consecutive programs within the memory of the hardware 
            architecture, is mapped to the first opcode, `0`.
            This word of the language is simply denoted 'null'.
        (2) The functions, whose names are given by `self.functions`, 
            are mapped to the next `len(self.functions)` opcodes, 
            where function `i` for `0 <= i <= len(self.functions) - 1`
            is mapped to opcode `1 + i`.
        (3) The variable terminals, whose names are given by `self.
            variable_terminals`, are mapped to the next `len(self.
            variable_terminals)` opcodes, where variable terminal 
            `i` for `0 <= i <= len(self.variable_terminals) - 1` 
            is mapped to opcode `1 + len(self.functions) + i`.
        (4) The constant terminals, whose names are given by `self.
            constant_terminals`, are mapped to the next `len(self.
            constant_terminals)` opcodes, where constant terminal `i` 
            for `0 <= i <= len(self.constant_terminals) - 1` is mapped 
            to opcode `1 + len(self.functions) + len(self.variable_
            terminals) + i`.
        (5) The function terminals, whose names are given by `self.
            function_terminals`, are mapped to the next `len(self.
            function_terminals)` opcodes, where function terminal `i` 
            for `0 <= i <= len(self.function_terminals) - 1` is mapped 
            to opcode `1 + len(self.functions) + len(self.variable_
            terminals) + len(self.constant_terminals) + i`.

        With exception to 'null', each assembly language word is
        denoted by the name of the primitive to which it corresponds.
        As such, we will commonly refer to each word of the assembly 
        language as a "name."
        """
        return ('null',) + self.primitives

    def opcode(self, name, form='b'):
        """Return opcode string for given assembly language word.

        An opcode is relevant to the assembly language inferred 
        from the primitive set.

        If the specified name does not exist within the assembly
        language, a `ValueError` exception is raised.

        Keyword arguments:
        name -- Name of word within assembly language.
        form -- Format specification for the opcode string. 
            This can be one of the following standard Python 
            "integer presentation types": 'b', for binary, 
            'd', for decimal, 'x', for lowercase hexadecimal, 
            or 'X', for uppercase hexadecimal. For each format 
            type, the minimum number of digits needed to 
            represent all opcodes within the relevant assembly 
            language is utilized with leading zeros. (default: 'b')
        """
        if form not in 'bdxX':
            # Invalid format specification.
            raise ValueError(f'Value provided for argument `form`, '
                             f'`{form}`, is invalid.')
        try:
            opcode = self.assembly_language.index(name)
        except ValueError:
            print(f'Name `{name}` does not exist.')
            raise

        if form == 'b':
            width = int(math.ceil(math.log(len(self.assembly_language), 2)))
        if form == 'd':
            width = int(math.ceil(math.log(len(self.assembly_language), 10)))
        else:
            width = int(math.ceil(math.log(len(self.assembly_language), 16)))

        return f'{opcode:0{width}{form}}'