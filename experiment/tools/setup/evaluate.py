# Some relevant imports and initializations.
import datetime as dt
import os
import pickle

import numpy as np

from gp.core.evaluation import standard as evaluate
from gp.hw.program import Program
from gp.contexts.symbolic_regression.primitive_sets import \
    nicolau_a, nicolau_b, nicolau_c
from gp.contexts.symbolic_regression.fitness import rmse

# Useful file path.
# root_dir = f'{os.getcwd()}/experiment/results/programs'
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
n_programs = 512

# Numbers of variables relevant to each primitive set.
n_variables = [len(primitive_sets[name].variables) for name in primitive_sets]

# Load programs and input/target data.
# with open(f'{root_dir}/../programs.pkl', 'rb') as f:
#     programs = pickle.load(f)
with open(f'{root_dir}/../inputs.pkl', 'rb') as f:
    inputs = pickle.load(f)
with open(f'{root_dir}/../target.pkl', 'rb') as f:
    target = pickle.load(f)
inputs = np.asarray(inputs)
target = np.asarray(target)

# Dictionary to contain fitness results relevant to each primitive set.
results = {name : [[] for _ in range(len(n_fitness_cases))] 
    for name in primitive_sets}

for name, ps in primitive_sets.items():
    # Read in the programs relevant to the primitive set from file.
    # This file contains `num_size_bins * n_programs` programs.
    with open(f'{root_dir}/{name}/programs.txt', 'r') as f:
        programs = f.readlines()

    for i, nfc in enumerate(n_fitness_cases):
        # For number of fitness cases `nfc`...

        for j in range(n_bins):
            # For program bin `j + 1`...
            program_bin = [Program.from_str(p, ps) for 
                p in programs[n_programs * (j) : n_programs * (j + 1)]]

            print(f'({dt.datetime.now().ctime()}) Evaluating programs for '
                f'primitive set `{name}`, bin {j+1}, {nfc} fitness cases...')
            
            # Extract the relevant input/target data.
            input_ = inputs[:nfc, :len(ps.variables)]
            target_ = target[:nfc]

            # Compute fitness values for each program.
            _, fitnesses = evaluate(program_bin, input_, target_, rmse,
                ps, n_threads=-1)
            results[name][i].append(fitnesses)
            # outputs = evaluate(program_bin, input_, ps, n_threads=-1)
            # results[name][i].append([fitness(target_, output) 
            #     for output in outputs])

        # # Preserve fitness data.
        # with open(f'{root_dir}/{name}/{nfc}/fitness.csv', 'w+') as f:
        #     for j, result_bin in enumerate(results[name][i]):
        #         for k, value in enumerate(result_bin):
        #             f.write(f'{str(value)}')
        #             if k < len(result_bin) - 1:
        #                 f.write(f'\n')
        #         if j < len(results[name][i]) - 1:
        #             f.write(f'\n')