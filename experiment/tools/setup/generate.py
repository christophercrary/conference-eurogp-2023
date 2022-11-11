# Some relevant imports and initializations.
import datetime as dt
import math
import os
import pickle
import random

import numpy as np

from gp.hw.program import Program
from gp.contexts.symbolic_regression.primitive_sets import \
    nicolau_a, nicolau_b, nicolau_c

# Random seed for reproducibility.
random.seed(42)

# Useful file path.
root_dir = f'{os.getcwd()}/../../results/programs'

########################################################################

# Primitive sets.
primitive_sets = {
    'nicolau_a' : nicolau_a, 
    'nicolau_b' : nicolau_b,
    'nicolau_c' : nicolau_c,
}

# Numbers of fitness cases relevant to each primitive set.
n_fitness_cases = (10, 100, 1000, 10000, 100000)
# n_fitness_cases = (10, 100,)

# Program tree depth constraints for each primitive set.
d = (9, 7, 7)

# Number of program bins.
n_bins = 32

# Number of programs per bin.
n_programs = 512

# Numbers of variables relevant to each primitive set.
n_variables = [len(primitive_sets[name].variables) for name in primitive_sets]

print(f'({dt.datetime.now().ctime()}) Generating random '
      f'input/target data...')

# Generate random input/target data.
inputs = np.array(
    [[random.random() for _ in range(max(n_variables))] 
        for _ in range(max(n_fitness_cases))])
target = np.array(
    [random.random() for _ in range(max(n_fitness_cases))])

# Dictionary of programs for each primitive set.
programs = {name : [] for name in primitive_sets}

print(f'\n')

for (name, ps), d_ in zip(primitive_sets.items(), d):
    print(f'({dt.datetime.now().ctime()}) Generating random programs '
          f'for primitive set `{name}`...')

    # Maximum possible program size.
    s_max_possible = Program.max_size(ps.m, d_)

    # Number of unique sizes per program bin (except possibly the last).
    n_sizes = math.ceil(s_max_possible / n_bins)

    for i in range(n_bins):
        # Minimum/maximum sizes of programs for the current bin.
        s_min = i * n_sizes + 1
        s_max = s_max_possible if i == n_bins - 1 else (i + 1) * n_sizes

        # Construct random program expressions for bin `i`.
        programs[name].append(Program.generate(
            primitive_set=ps, d_max=d_, s_max=s_max, d_min=0, 
            s_min=s_min, n_programs=n_programs, n_threads=-1))

    # Preserve information about program expressions.
    with open(f'{root_dir}/{name}/programs.txt', 'w+') as f:
        for i, program_bin in enumerate(programs[name]):
            for j, program in enumerate(program_bin):
                f.write(f'{program}')
                if j < len(program_bin) - 1:
                    f.write(f'\n')
            if i < len(programs[name]) - 1:
                f.write(f'\n')
    with open(f'{root_dir}/{name}/program_memory.txt', 'w+') as f:
        for i, program_bin in enumerate(programs[name]):
            for j, program in enumerate(program_bin):
                machine_code = program.machine_code()
                for code in machine_code[:-1]:
                    f.write(f'{code}\n')
                f.write(f'{machine_code[-1]}')
                if j < len(program_bin) - 1:
                    f.write(f'\n')
            if i < len(programs[name]) - 1:
                f.write(f'\n')

    # For each number of fitness cases, preserve the relevant 
    # subset of input/target data.
    for nfc in n_fitness_cases:
        input_ = inputs[:nfc, :len(ps.variables)]
        target_ = target[:nfc]
        data = np.column_stack((input_, target_))
        with open(f'{root_dir}/{name}/{nfc}/data.csv', 'w+') as f:
            # Write header information.
            for name_ in ps.variables:
                f.write(f'{name_},')
            f.write(f'y\n')
            # Write data.
            for i, row in enumerate(data):
                for value in row[:-1]:
                    f.write(f'{str(value)},')
                f.write(f'{str(row[-1])}\n')
                # NOTE: We need a newline at the end of the CSV file 
                # for Operon to be able to parse it.

# Pickle the programs and input/target data.
with open(f'{root_dir}/../programs.pkl', 'wb') as f:
    pickle.dump(programs, f)
with open(f'{root_dir}/../inputs.pkl', 'wb') as f:
    pickle.dump(inputs, f)
with open(f'{root_dir}/../target.pkl', 'wb') as f:
    pickle.dump(target, f)