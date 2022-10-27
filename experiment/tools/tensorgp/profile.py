import datetime as dt
import math
import os
import pickle
import random
import sys
import timeit

import numpy as np
from scipy.stats import iqr
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join('./', 'tensorgp')))
from tensorgp.engine import *

# Useful path directory.
root_dir = (f'{os.path.dirname(os.path.abspath(__file__))}/../../results/'
            f'programs')

# Random seed, for reproducibility.
seed = 37
random.seed(seed)

########################################################################
# Some helper function(s).
########################################################################

def get_max_size(m, d):
    """Return the maximum possible size for a program.
    
    The program is considered to be `m`-ary, of depth `d`.
    """
    if (m==1):
        return d+1
    else:
        return int((1-m**(d+1))/(1-m))

######################################################################## Profiling of program evaluation mechanism given by TensorGP.
########################################################################

def r2(**kwargs):
    """R-squared fitness function."""
    population = kwargs.get('population')
    tensors = kwargs.get('tensors')
    target = kwargs.get('target'
    )

    # print('Population size:', len(population))
    # print('Shape of tensors:', tf.shape(tensors))
    # print('Shape of target:', tf.shape(target))

    fitness = []
    best_ind = 0

    max_fit = float('0')

    for i in range(len(tensors)):

        # print(f'Tensors[{i}]:', tensors[i])
        # print(f'Target:', target)

        # fit = tf_r2(target, tensors[i]).numpy()
        fit = tf_rmse(target, tensors[i]).numpy()
        # print('Fitness:', fit)

        if fit > max_fit:
            max_fit = fit
            best_ind = i

        fitness.append(fit)
        population[i]['fitness'] = fit

        # print(f'Tensor[{i}]: {tensors[i]}')
        # print(f'Target: {target}')

    # print(f'`Length of `fitness: {len(fitness)}')

    return population, best_ind


# Parameter for debug logging within TensorGP.
debug = 0

# Computing devices to utilize.
devices = ('/cpu:0', '/gpu:0')

# Overall set of functions.
functions = {'add', 'aq', 'exp', 'log', 'mul', 
             'sin', 'sqrt', 'sub', 'tanh'}

# Dictionary for particular function set criteria.
function_sets = {
    'nicolau_a': (4, 2, 9, 32),
    'nicolau_b': (6, 2, 7, 8),
    'nicolau_c': (9, 2, 6, 4)
}

# Number of programs per size bin.
num_programs_per_size_bin = 128

# Numbers of fitness cases.
num_fitness_cases = (10, 100, 1000, 10000, 100000)

# Overall set of input vectors.
with open(f'{root_dir}/inputs.pkl', 'rb') as f:
    inputs_ = pickle.load(f)

# Overall target vector.
with open(f'{root_dir}/target.pkl', 'rb') as f:
    target_ = pickle.load(f)

# Infer `NumPy` arrays for input/target vectors.
inputs_ = np.array(inputs_)
target_ = np.array(target_)

# Number of times in which the `timeit.repeat` function is
# called, in order to generate a list of median average
# runtimes.
num_epochs = 1

# Value for the `repeat` argument of the `timeit.repeat` method.
repeat = 3

# Value for the `number` argument of the `timeit.repeat` method.
number = 1

# Median average runtimes for programs within each size bin,
# for each number of fitness cases, for each function set.
med_avg_runtimes = []

for device in devices:
    # For each device...

    # Prepare for statistics relevant to function set.
    med_avg_runtimes.append([])

    for name, (num_functions, max_arity, 
        max_depth, bin_size) in function_sets.items():
        # For each function set...
        print(f'Function set `{name}`:')

        # Number of variables for given function set.
        num_variables = num_functions - 1

        # Maximum program size for function set.
        max_possible_size = get_max_size(max_arity, max_depth)

        # Number of size bins.
        num_size_bins = int(math.ceil(max_possible_size/bin_size))

        # Prepare for statistics relevant to function set.
        med_avg_runtimes[-1].append([])

        # Read in the programs relevant to the function set from file.
        # This file contains `population_size * num_size_bins` programs,
        # representing the `population_size` programs for each of the
        # `num_size_bins` size bins.
        with open(f'{root_dir}/{name}/programs_tensorgp.txt', 'r') as f:
            programs = f.readlines()

        for nfc in num_fitness_cases:
            # For each number of fitness cases...
            print(f'Number of fitness cases: `{nfc}`')

            # Create a terminal set relevant to the function set.

            # Tensor dimensions for the current number of fitness cases.
            target_dims = (nfc,)

            # Number of fitness case dimensions. Note that this 
            # is *not* the same thing as the number of variable
            # terminals.
            num_dimensions = 1

            terminal_set = Terminal_Set(num_dimensions, target_dims)

            # Add custom terminals and remove default terminal.
            for i in range(num_variables):
                terminal_set.add_to_set(
                    f'v{i}', tf.cast(inputs_[:nfc, i], tf.float32))
            terminal_set.remove_from_set('x')

            # Target for given number of fitness cases.
            target = tf.cast(tf.convert_to_tensor(target_[:nfc]), tf.float32)

            # Prepare for statistics relevant to the 
            # numbers of fitness cases and size bins.
            med_avg_runtimes[-1][-1].append([[] for _ in range(num_size_bins)])

            # Create an appropriate GP engine.
            engine = Engine(debug=debug,
                            seed=seed,
                            device=device,
                            operators=functions,
                            terminal_set=terminal_set,
                            target_dims=target_dims,
                            target=target,
                            fitness_func=r2,
                            population_size=num_programs_per_size_bin,
                            min_domain=-10000,
                            max_domain=10000)

            for i in range(num_size_bins):
                # For each size bin, calculate the relevant statistics.
                print(f'({dt.datetime.now().ctime()}) Size bin `{i+1}`...')

                # Population relevant to the current size bin.
                population, *_ = engine.generate_pop_from_expr(
                    programs[(i) * num_programs_per_size_bin:
                             (i+1) * num_programs_per_size_bin])

                for _ in range(num_epochs):
                    # For each epoch...

                    # Raw runtimes after running the `fitness_func_wrap` 
                    # function a total of `repeat * number` times. 
                    # The resulting object is a list of `repeat` values,
                    # where each represents a raw runtime after running
                    # the relevant code `number` times.
                    runtimes = timeit.Timer(
                        'engine.fitness_func_wrap(population=population)',
                        globals=globals()).repeat(repeat=repeat, number=number)

                    # Average runtimes, taking into account `number`.
                    avg_runtimes = [runtime/number for runtime in runtimes]

                    # Calculate and append median average runtime.
                    med_avg_runtimes[-1][-1][-1][i].append(
                        np.median(avg_runtimes))

# Preserve results.
with open(f'{root_dir}/../results_tensorgp.pkl', 'wb') as f:
    pickle.dump(med_avg_runtimes, f)