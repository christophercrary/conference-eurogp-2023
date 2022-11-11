"""Convert program strings for each relevant GP tool."""
# Some relevant imports and initializations.
import datetime as dt
import os
import pickle

from gp.hw.program import Program
from gp.contexts.symbolic_regression.primitive_sets import \
    nicolau_a, nicolau_b, nicolau_c

# Useful file path.
root_dir = f'{os.getcwd()}/../../results/programs'

########################################################################

def tensorgp_str(self):
    """Return program string for TensorGP.
    
    Constant terminal values `t` must be rewritten as 
    the string `scalar(t)`.
    """
    stack = []
    for node in reversed(self):
        if node.function:
            operands = [stack.pop() for _ in range(node.arity)]
            stack.append((')' + ' ,'.join(
                [op[::-1] for op in operands[::-1]]) 
                + '(' + node.name[::-1])[::-1])
        elif node.variable:
            stack.append(node.name)
        else:
            stack.append(f'scalar({str(node.name)})')
    return stack.pop() if stack != [] else ''

# Add method to `Program` class.
setattr(Program, 'tensorgp_str', property(tensorgp_str))

########################################################################

# Primitive sets.
primitive_sets = {
    'nicolau_a' : nicolau_a, 
    'nicolau_b' : nicolau_b,
    'nicolau_c' : nicolau_c,
}

# Load programs and input/target data.
with open(f'{root_dir}/../programs.pkl', 'rb') as f:
    programs = pickle.load(f)

print(f'\n')

for name, ps in primitive_sets.items():
    ps = primitive_sets[name]
    print(f'({dt.datetime.now().ctime()}) Converting data for '
        f'primitive set `{name}`...')

    # Convert programs to a representation relevant to TensorGP.
    with open(f'{root_dir}/{name}/programs_tensorgp.txt', 'w+') as f:
        for i, program_bin in enumerate(programs[name]):
            for j, program in enumerate(program_bin):
                f.write(f'{program.tensorgp_str}')
                if j < len(program_bin) - 1:
                    f.write(f'\n')
            if i < len(programs[name]) - 1:
                f.write(f'\n')

    # Convert programs to a representation relevant to Operon.
    with open(f'{root_dir}/{name}/programs_operon.txt', 'w+') as f:
        for i, program_bin in enumerate(programs[name]):
            for j, program in enumerate(program_bin):
                f.write(f'{program.inorder_str}')
                if j < len(program_bin) - 1:
                    f.write(f'\n')
            if i < len(programs[name]) - 1:
                f.write(f'\n')