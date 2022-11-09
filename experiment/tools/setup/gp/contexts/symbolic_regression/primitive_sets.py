"""GP primitive sets."""
from collections import OrderedDict

from . import constants as c
from . import functions as f
from gp.core.primitive_set import PrimitiveSet

nicolau_a = PrimitiveSet(
    functions=OrderedDict(
        {'add' : f.add, 'sub' : f.sub, 'mul' : f.mul, 'aq' : f.aq}),
    variables=OrderedDict(
        {name : None for name in [f'v{i}' for i in range(3)]}),
    constants=OrderedDict({'rand' : c.rand}))

nicolau_b = PrimitiveSet(
    functions=OrderedDict(
        {'sin' : f.sin, 'tanh' : f.tanh, 'add' : f.add, 'sub' : f.sub, 
        'mul' : f.mul, 'aq' : f.aq}),
    variables=OrderedDict(
        {name : None for name in [f'v{i}' for i in range(5)]}),
    constants=OrderedDict({'rand' : c.rand}))

nicolau_c = PrimitiveSet(
    functions=OrderedDict(
        {'sin' : f.sin, 'tanh' : f.tanh, 'exp' : f.exp, 'log' : f.log, 
        'sqrt' : f.sqrt, 'add' : f.add, 'sub' : f.sub, 'mul' : f.mul, 
        'aq' : f.aq}),
    variables=OrderedDict(
        {name : None for name in [f'v{i}' for i in range(8)]}),
    constants=OrderedDict({'rand' : c.rand}))