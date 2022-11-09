import datetime as dt
import math
import os
import pickle
import timeit
import sys

import deap.gp
import numpy as np
from pathos.pools import ProcessPool
from sklearn.metrics import mean_squared_error

sys.path.insert(1, '../setup/')
from gp.contexts.symbolic_regression.primitive_sets import \
    nicolau_a, nicolau_b, nicolau_c

# Useful directory path.
root_dir = f'{os.getcwd()}/../../results/programs'

########################################################################

def evaluate(primitive_set, trees, X, target):
    """Return list of fitness scores for programs.
    
    Root-mean square error is used as the fitness function.
    """
    def evaluate_(tree):
        try:
            # Transform `PrimitiveTree` object into a callable function.
            program = deap.gp.compile(tree, primitive_set)

            # Calculate program outputs.
            estimated = tuple(program(*X_) for X_ in X)

            # Calculate and return fitness.
            return math.sqrt(mean_squared_error(target, estimated))
        except ValueError:
            return float("inf")

    # Calculate fitness scores for the set of trees in parallel.
    fitness = ProcessPool().map(evaluate_, trees)

    return fitness

########################################################################

# Primitive sets.
primitive_sets = {
    'nicolau_a': nicolau_a,
    'nicolau_b': nicolau_b,
    'nicolau_c': nicolau_c,
}

# Numbers of fitness cases.
n_fitness_cases = (10, 100,)
# n_fitness_cases = (10, 100, 1000, 10000, 100000)

# Number of program bins.
n_bins = 32

# Number of programs per bin.
n_programs = 1

# Number of times in which experiments are run.
n_runs = 11

# Runtimes for programs within each size bin, for each number 
# of fitness cases, for each function set.
runtimes = []

# Load input/target data.
with open(f'{root_dir}/../setup.pkl', 'rb') as f:
    inputs, target, *_ = pickle.load(f)
inputs = np.asarray(inputs)
target = np.asarray(target)

for name, ps in primitive_sets.items():
    # Prepare for statistics relevant to the primitive set.
    runtimes.append([])

    # Read in the programs relevant to the primitive set from file.
    # This file contains `num_size_bins * n_programs` programs.
    with open(f'{root_dir}/{name}/programs.txt', 'r') as f:
        programs = f.readlines()

    # Primitive set object for DEAP tool.
    primitive_set = deap.gp.PrimitiveSet("main", len(ps.variables), prefix="v")

    # Add functions to primitive set.
    for name_, f in ps.functions.items():
        primitive_set.addPrimitive(f, ps.arity(name_))

    # For each amount of fitness cases, and for each size bin, 
    # calculate the relevant statistics.
    for nfc in n_fitness_cases:
        # Extract the relevant input/target data.
        input_ = inputs[:nfc, :len(ps.variables)]
        target_ = target[:nfc]

        # Prepare for statistics relevant to the 
        # numbers of fitness cases and size bins.
        runtimes[-1].append([[] for _ in range(n_bins)])

        for i in range(n_bins):
            # For each size bin...
            print(f'({dt.datetime.now().ctime()}) DEAP: evaluating programs '
                f'for primitive set `{name}`, bin {i + 1}, {nfc} fitness '
                f'cases...')

            # `PrimitiveTree` objects for size bin `i`.
            trees = [deap.gp.PrimitiveTree.from_string(p, primitive_set) for 
                p in programs[n_programs * (i) : n_programs * (i + 1)]]

            # Raw runtimes after running the `evaluate`
            # function a total of `n_runs` times.
            runtimes[-1][-1][i] = timeit.Timer(
                'evaluate(primitive_set, trees, input_, target_)',
                globals=globals()).repeat(repeat=n_runs, number=1)

# Preserve results.
with open(f'{root_dir}/../runtimes/deap/results.pkl', 'wb') as f:
    pickle.dump(runtimes, f)