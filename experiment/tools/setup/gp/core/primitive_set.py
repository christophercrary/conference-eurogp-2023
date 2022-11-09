"""Primitive set."""
from collections import OrderedDict
import inspect
from itertools import count, filterfalse
import keyword
from operator import itemgetter
import re

class PrimitiveSet:
    """Class for generic primitive set."""
    __slots__ = ('functions', 'variables', 'constants', 'namespace')

    def __init__(
        self, functions=OrderedDict(), variables=OrderedDict(), 
            constants=OrderedDict()):
        self.functions = functions
        self.variables = variables
        self.constants = constants
        self.namespace = functions | variables | constants

    @property
    def terminals(self):
        """Return all terminal primitives."""
        return self.variables | self.constants

    @property
    def primitives(self):
        """Return all primitives."""
        return self.namespace

    @property
    def function_proportion(self):
        """Return proportion of primitives that are functions."""
        return len(self.functions) / len(self.primitives)

    @property
    def terminal_proportion(self):
        """Return proportion of primitives that are terminals."""
        return len(self.terminals) / len(self.primitives)

    @property
    def variable_proportion(self):
        """Return proportion of primitives that are variables."""
        return len(self.variables) / len(self.primitives)
        
    @property
    def constant_proportion(self):
        """Return proportion of primitives that are constants."""
        return len(self.constants) / len(self.primitives)

    @property
    def a_min(self):
        """Return minimum arity of function set."""
        return (min([self.arity(f) for f in self.functions.keys()])
            if len(self.functions) != 0 else 0)

    @property
    def a_max(self):
        """Return maximum arity of function set."""
        return (max([self.arity(f) for f in self.functions.keys()])
            if len(self.functions) != 0 else 0)

    @property
    def m(self):
        """Return the "ary-ness" of the primitive set.
        
        Ary-ness is defined to be the maximum arity.
        """
        return self.a_max

    def __getitem__(self, names):
        """Return objects from namespace of primitive set."""
        return itemgetter(names)(self.namespace)

    def __contains__(self, name):
        """Return whether or not a primitive exists."""
        return name in self.namespace

    def __len__(self):
        """Return the number of primitives."""
        return len(self.namespace)

    def arity(self, name, default=None):
        """Return arity of primitive, if it exists.

        If a primitive with name `name` exists, the 
        arity of that primitive is returned.

        If a primitive with name `name` does not exist
        and `default` is not `None`, then `default` is
        returned; otherwise, a `KeyError` exception is
        raised.

        Keyword arguments:
        name -- Name of primitive.
        default -- Default return value. (default: None)
        """
        try:
            primitive = self.namespace[name]
        except KeyError:
            if default is None:
                print(f'Name `{name}` is not in primitive set.')
                raise
            return default
        else:
            if name in self.functions:
                args, *_ = inspect.getfullargspec(primitive)
                arity = len(args)
            else:
                arity = 0
            return arity

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

        if name in self:
            raise ValueError(f'Name `{name}` already exists within '
                             f'the primitive set.')
        self.namespace[name] = function
        self.functions[name] = function

    def add_variable(self, name=None):
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
            # in `self.namespace.keys()`.
            M = set([int(''.join(m)) for _,*m in 
                list(filter(r.match, list(self.namespace.keys())))])

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
        self.namespace[name] = None
        self.variables[name] = None

    def add_constant(self, constant, name=None):
        """Add constant terminal.

        If `name` is `None` and `constant` is callable,
        the name `constant.__name__` is provided; if 
        `constant` is not callable, a `TypeError` exception
        is raised.

        Any value other than `None` given for `name`  must 
        be a valid Python identifier that is not also a
        reserved keyword. If the provided name already exists 
        within the the primitive set, a `ValueError` exception 
        is raised.
        
        Keyword arguments:
        constant -- Zero-arity Python function to implement 
            constant.
        name -- Name for constant terminal. Must be either 
            `None` or a valid Python identifier that is not 
            a reserved keyword. (default: None)
        """
        try:
            args, *_ = inspect.getfullargspec(constant)
        except TypeError:
            print(f'Value provided for argument `constant`, `{constant}`, '
                  f'is not callable.')
            raise

        if len(args) != 0:
            raise ValueError(f'constant `{constant}` has more '
                             f'than zero arguments.')
                             
        if name is None:
            name = constant.__name__
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
        self.namespace[name] = constant        
        self.constants[name] = constant

    def remove(self, name, default=None):
        """Remove primitive, if it exists.
        
        If an primitive with name `name` exists, the value 
        `None` is returned.

        If a primitive with name `name` does not exist
        and `default` is not `None`, then `default` is
        returned; otherwise, a `KeyError` exception is
        raised.

        Keyword arguments:
        name -- Name of primitive.
        default -- Default return value. (default: None)
        """
        try:
            self.namespace.pop(name)
        except KeyError:
            if default is None:
                print(f'Name `{name}` is not in primitive set.')
                raise
            return default
        else:
            if name in self.functions:
                self.functions.pop(name)
            elif name in self.variables:
                self.variables.pop(name)
            else:
                self.constants.pop(name)
            return None