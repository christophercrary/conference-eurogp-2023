import datetime as dt
import os
import pickle
import sys
import timeit

import numpy as np
import tensorflow as tf

# sys.path.insert(1, './experiment/tools/setup/')
sys.path.insert(1, '../setup/')
from gp.contexts.symbolic_regression.primitive_sets import \
    nicolau_a, nicolau_b, nicolau_c

# sys.path.insert(1, './experiment/tools/tensorgp/tensorgp')
sys.path.insert(1, './tensorgp')
from tensorgp.engine import *

# Useful path directory.
# root_dir = (f'{os.getcwd()}/experiment/results/programs')
root_dir = (f'{os.getcwd()}/../../results/programs')

########################################################################

def rmse(**kwargs):
    """RMSE fitness function."""
    population = kwargs.get('population')
    tensors = kwargs.get('tensors')
    target = kwargs.get('target')
    output_file_path = kwargs.get('f_path')

    fitness = []
    best_ind = 0

    max_fit = float('0')

    for i in range(len(tensors)):
        fit = tf_rmse(target, tensors[i]).numpy()
        # print(f'Fitness: {str(fit)}')

        if fit > max_fit:
            max_fit = fit
            best_ind = i

        fitness.append(fit)
        population[i]['fitness'] = fit

    # Write fitness values to the relevant file.
    with open(f'{output_file_path}', 'a+') as f:
        for i in range(len(tensors)):
            f.write(f'{str(population[i]["fitness"])}\n')

    return population, best_ind

########################################################################

# Parameter for debug logging within TensorGP.
debug = 0

# Computing devices to utilize.
devices = ('/cpu:0', '/gpu:0')
# devices = ('/cpu:0',)
# devices = ('/gpu:0',)

# Primitive sets.
primitive_sets = {
    'nicolau_a': nicolau_a,
    'nicolau_b': nicolau_b,
    'nicolau_c': nicolau_c,
}

# Numbers of fitness cases.
n_fitness_cases = (10, 100, 1000, 10000, 100000)
# n_fitness_cases = (10,)

# Number of program bins.
n_bins = 32

# Number of programs per bin.
n_programs = 512

# Number of times in which experiments are run.
n_runs = 11

# Runtimes for programs within each size bin, for each number 
# of fitness cases, for each primitive set, for each device.
runtimes = []

# Load input/target data.
with open(f'{root_dir}/../inputs.pkl', 'rb') as f:
    inputs = pickle.load(f)
with open(f'{root_dir}/../target.pkl', 'rb') as f:
    target = pickle.load(f)
inputs = np.asarray(inputs)
target = np.asarray(target)

for device in devices:
    # Prepare for statistics relevant to the device.
    runtimes.append([])

    for name, ps in primitive_sets.items():
        # Prepare for statistics relevant to primitive set.
        runtimes[-1].append([])

        # Read in the programs relevant to the primitive set from file.
        # This file contains `n_bins * n_programs` programs.
        with open(f'{root_dir}/{name}/programs_tensorgp.txt', 'r') as f:
            programs = f.readlines()

        for nfc in n_fitness_cases:
            # Create a terminal set relevant to the primitive set.

            # Tensor dimensions for the current number of fitness cases.
            target_dims = (nfc,)

            # Number of fitness case dimensions. Note that this 
            # is *not* the same thing as the number of variable
            # terminals.
            num_dimensions = 1

            terminal_set = Terminal_Set(num_dimensions, target_dims)

            # Add custom terminals and remove default terminal.
            for i in range(len(ps.variables)):
                terminal_set.add_to_set(
                    f'v{i}', tf.cast(inputs[:nfc, i], tf.float32))

            # Target for given number of fitness cases.
            target_ = tf.cast(tf.convert_to_tensor(target[:nfc]), tf.float32)

            # Prepare for statistics relevant to the 
            # numbers of fitness cases and size bins.
            runtimes[-1][-1].append([[] for _ in range(n_bins)])

            # Create an appropriate TensorGP engine.
            engine = Engine(
                debug=debug,
                seed=42,
                device=device,
                operators=ps.functions.keys(),
                terminal_set=terminal_set,
                target_dims=target_dims,
                target=target_,
                fitness_func=rmse,
                population_size=n_programs,
                domain = [-1e1000, 1e1000],
                codomain = [-1e1000, 1e1000])

            # Remove any pre-existing fitness output file relevant
            # to the current primitive set and number of fitness
            # cases, to prepare for new fitness outputs.
            output_file_path = f'{root_dir}/{name}/{nfc}/fitness_tensorgp.csv'
            if os.path.exists(output_file_path):
                os.remove(output_file_path)

            for i in range(n_bins):
                # For each size bin, calculate the relevant statistics.
                print(f'({dt.datetime.now().ctime()}) TensorGP: evaluating '
                    f'programs for primitive set `{name}`, bin {i + 1}, {nfc} ' 
                    f'fitness cases...')

                # Population relevant to the current size bin.
                population, *_ = engine.generate_pop_from_expr(
                    programs[n_programs * (i) : n_programs * (i + 1)])

                # Raw runtimes after running the `evaluate`
                # function a total of `n_runs` times.
                runtimes[-1][-1][-1][i] = timeit.Timer(
                    f'engine.fitness_func_wrap(population=population,'
                    f'f_path="{output_file_path}")',
                    globals=globals()).repeat(repeat=n_runs, number=1)

# Preserve results.
for i, name in enumerate(devices):
    if 'cpu' in name:
        with open(
            f'{root_dir}/../runtimes/tensorgp/results_cpu.pkl', 'wb') as f:
            pickle.dump(runtimes[i], f)
    elif 'gpu' in name:
        with open(
            f'{root_dir}/../runtimes/tensorgp/results_gpu.pkl', 'wb') as f:
            pickle.dump(runtimes[i], f)