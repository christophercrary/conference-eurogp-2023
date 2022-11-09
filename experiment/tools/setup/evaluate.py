# Some relevant imports and initializations.
import datetime as dt
import os
import pickle

import numpy as np

from gp.core.evaluation import standard as evaluate
from gp.contexts.symbolic_regression.primitive_sets import \
    nicolau_a, nicolau_b, nicolau_c
from gp.contexts.symbolic_regression.fitness import rmse as fitness

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

# Number of program bins.
n_bins = 32

# Number of programs per bin.
n_programs = 1024

# Numbers of variables relevant to each primitive set.
n_variables = [len(primitive_sets[name].variables) for name in primitive_sets]

# Load programs and input/target data.
with open(f'{root_dir}/../setup.pkl', 'rb') as f:
    inputs, target, programs = pickle.load(f)
inputs = np.asarray(inputs)
target = np.asarray(target)

# Dictionary to contain fitness results relevant to each primitive set.
results = {name : [[] for _ in range(len(n_fitness_cases))] 
    for name in primitive_sets}

for name, ps in primitive_sets.items():
    for i, nfc in enumerate(n_fitness_cases):
        # For number of fitness cases `nfc`...

        for j, program_bin in enumerate(programs[name]):
            # For program bin `i + 1`...
            print(f'({dt.datetime.now().ctime()}) Evaluating programs for '
                f'primitive set `{name}`, bin {j+1}, {nfc} fitness cases...')
            
            # Extract the relevant input/target data.
            input_ = inputs[:nfc, :len(ps.variables)]
            target_ = target[:nfc]

            # Compute fitness values for each program.
            outputs = evaluate(program_bin, input_, ps, n_threads=-1)
            results[name][i].append([fitness(target_, output) 
                for output in outputs])

        # Preserve output/fitness data.
        with open(f'{root_dir}/{name}/{nfc}/fitness.csv', 'w+') as f:
            for j, result_bin in enumerate(results[name][i]):
                for k, value in enumerate(result_bin):
                    f.write(f'{str(value)}')
                    if k < len(result_bin) - 1:
                        f.write(f'\n')
                if j < len(results[name][i]) - 1:
                    f.write(f'\n')