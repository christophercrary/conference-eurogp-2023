import datetime as dt
from inspect import isclass
import math
import os
import pickle
# import pygraphviz as pgv
import random
import sys
import timeit

from deap import gp
import numpy as np
from pathos.pools import ProcessPool
from sklearn.metrics import mean_squared_error, r2_score


# Useful directory path.
root_dir = f'{os.getcwd()}/../results/programs'


# Seed the relevant random number generator, for reproducibility.
random.seed(37)

########################################################################
# Some helper functions.
########################################################################

def get_max_size(m, d):
  """Return the maximum possible size for a `m`-ary program of depth `d`."""

  if (m==1):
    return d+1
  else:
    return int((1-m**(d+1))/(1-m))

def generate_primitive_set(
    function_set, num_variables, num_constants, erc_name):
    """Return tuple containing (i) the primitive set based on 
    the given function set (`function_set`), number of terminal 
    variables (`num_variables`), and number of terminal constants
    (`num_constants`), and (ii) the fixed ephemeral constant
    list utilized by the primitive set."""

    primitive_set = gp.PrimitiveSet("main", num_variables, prefix="v")

    # Add functions to primitive set.
    for op, arity in function_set:
        primitive_set.addPrimitive(op, arity)

    # Create a list of fixed ephemeral random constants.
    ephemeral_constants = []
    for i in range(num_constants):
        ephemeral_constants.append(random.uniform(1,2))
    
    # Add an ephemeral constant to the DEAP primitive set that
    # returns a random value from the `ephemeral_constants` list 
    # of random constants. This is done so that there can only
    # exist a particular set of random constants, which is unlike
    # how DEAP would typically implement ephemeral constants.
    primitive_set.addEphemeralConstant(erc_name, lambda: ephemeral_constants[
        random.randint(0, num_constants-1)])

    return (primitive_set, ephemeral_constants)

def generate_program_(primitive_set, min_depth, max_depth, min_size, 
    max_size, desired_trait, ret_type, terminal_condition):
    """Generate a program as done by the `generate` function of the `deap.gp`
    library, except take into account minimum and maximum program sizes,
    by `min_size` and `max_size, respectively.
    """

    if ret_type is None:
        ret_type = primitive_set.ret

    program = []

    if desired_trait == 'depth':
        desired_value = random.randint(min_depth, max_depth)
    else:
        desired_value = random.randint(min_size, max_size)

    # Initial stack for constructing the relevant program.
    stack = [(0, 1)]

    while len(stack) != 0:

        # Retrieve the next relevant node within the stack.
        # The value `depth` represents the depth of this
        # node within the overall random program, the value
        # `size` represents the current size of the overall
        # program, and the value `ret_type` represents the 
        # return type of the current node.
        depth, size = stack.pop()

        if (terminal_condition(stack, depth, size, ret_type,
            primitive_set, min_depth, max_depth, min_size, 
            max_size, desired_trait, desired_value)):

            # A random terminal node is to be chosen.

            terminal = random.choice(primitive_set.terminals[ret_type])

            if isclass(terminal):
                terminal = terminal()

            program.append(terminal)

            # If the stack is nonempty, update the size of the
            # next element to be equivalent to the size of the
            # current node under consideration, so that, upon
            # considering this upcoming element, the `size` value
            # specified by this element will accurately represent 
            # the current program size. (The size of the upcoming
            # element may already be equal to `size`, but it will
            # never be greater than `size`.)
            if len(stack) != 0:
                depth, _ = stack.pop()
                stack.append((depth, size))

        else:

            # A random (valid) function node is to be chosen,
            # if one exists.

            # Valid functions for the current node, 
            # based on function arity.
            valid_functions = [f for f in primitive_set.primitives[ret_type] 
                if size+f.arity <= max_size]

            if valid_functions != [] and desired_trait == 'size':
                # Determine the subset of valid functions that are also
                # valid for potentially constructing a program of the 
                # specified size constraints, given the current program. 
                # (The choice of some functions may preclude the desired 
                # program size.)

                # Maximum function arity for the set of functions that
                # are valid for the current node.
                max_arity = max([f.arity for f in valid_functions])

                temp_functions = []

                for f in valid_functions:

                    # Maximum possible size of subprogram rooted at the 
                    # current node, excluding this node, if the current 
                    # node is given to be `f`.
                    max_possible_size = f.arity * get_max_size(
                        max_arity, max_depth-(depth+1))

                    # Maximum possible program size if the relevant node 
                    # under consideration was chosen to be the function 
                    # `f`. This maximum size would occur if every out-
                    # standing node within the current stack is made to 
                    # be the root of a full `max_arity`-ary subtree such 
                    # that the sum of the depth of this subtree and the 
                    # depth of the root node within the overall program 
                    # is equal to `max_depth`.
                    max_possible_size = (size + max_possible_size if stack==[]
                        else size + max_possible_size + sum([get_max_size(
                            max_arity, max_depth-d)-1 for (d,*_) in stack]))

                    if max_possible_size >= desired_value:
                        temp_functions.append(f)

                valid_functions = temp_functions

            
            function = (None if valid_functions == [] else 
                random.choice(valid_functions))

            if function == None:
                # The current program cannot be made to meet the 
                # specified size constraints.
                return None
            else:
                program.append(function)

            for _ in reversed(function.args):
                # Add a placeholder stack element for each 
                # argument needed by the chosen function.
                stack.append((depth+1, size+function.arity))

    return program

def generate_grow(primitive_set, min_depth, max_depth, min_size, max_size,
    desired_trait, ret_type=None):

    def terminal_condition(stack, depth, size, ret_type,
        primitive_set, min_depth, max_depth, min_size, max_size,
        desired_trait, desired_value):
        """Expression generation stops when the depth is equal to the desired
        depth or when it is randomly determined that a node should be a terminal.
        """

        # List of valid terminals for the current node.
        valid_terminals = [t for t in primitive_set.terminals[ret_type]] 

        # List of valid functions for the current node.
        valid_functions = [f for f in primitive_set.primitives[ret_type] 
            if size+f.arity <= max_size]

        # Maximum function arity for the set of functions that
        # are valid for the current node.
        max_arity = (1 if valid_functions == [] else 
            max([f.arity for f in valid_functions]))

        # Maximum possible program size if the relevant node under
        # consideration is chosen to be a terminal. This maximum size 
        # would occur if every outstanding node within the current stack 
        # is made to be the root of a full `max_arity`-ary subtree such 
        # that the sum of the depth of this subtree and the depth of the 
        # root node within the overall program is equal to `max_depth`.
        # (Note that when `valid_functions` is empty, the value of this 
        # variable is arbitrary.)
        max_possible_size = (size if stack == [] else size + 
            sum([get_max_size(max_arity, max_depth-d)-1
                for (d,*_) in stack]))

        ret = ((valid_terminals != []) and
                ((valid_functions == []) or 
                    (depth == max_depth) or
                    (size == max_size) or
                    ((desired_trait == 'depth') and 
                        (depth == desired_value)) or
                    ((desired_trait == 'size') and 
                        (size >= desired_value)) or
                    ((depth >= min_depth) and 
                        (size >= min_size) and
                        ((desired_trait == 'depth') or
                            ((desired_trait == 'size') and
                                (max_possible_size >= desired_value))) and
                        (random.random() < primitive_set.terminalRatio))))

        return ret

    return generate_program_(primitive_set, min_depth, max_depth, 
        min_size, max_size, desired_trait, ret_type, terminal_condition)
    
def generate_program(gen_strategy, primitive_set, min_depth, max_depth,
    min_size, max_size, desired_trait):
    """Generate a program expression based on the specified initialization
    strategy (`gen_strategy`), primitive set (`primitive_set`), and maximum
    program depth (`max_depth`)."""

    if gen_strategy == 'rhh':
        return gp.genHalfAndHalf(primitive_set, min_=min_depth, max_=max_depth)
    elif gen_strategy == 'full':
        return gp.genFull(primitive_set, min_=max_depth, max_=max_depth)
    else:
        return generate_grow(primitive_set, min_depth, max_depth, min_size,
            max_size, desired_trait)


# def generate_program(pset, d_max, s_desired):
#     """Attempt to generate a program of the specified size.
    
#     The program is generated based on the given primitive set 
#     and maximum depth constraints. Different tree shapes are
#     randomly sampled based on the `terminalRatio` attribute
#     of the primitive set.

#     It is not guaranteed that a program of the specified size
#     will be found, even if one exists in theory.

#     Keyword arguments:
#     pset -- Primitive set, of type `deap.gp.PrimitiveSet`.
#     d_max -- Maximum allowable depth of program.
#     s_desired -- Desired size of program.
#     """
#     program = []  # Prefix representation of program.
#     depth = 0     # Depth of program.
#     size = 1      # Size of program.
#     stack = [0]   # Stack of depths for nodes outstanding.

#     while len(stack) != 0:
#         # Depth of the next relevant node.
#         d = stack.pop()

#         # Functions with an arity that is not too large.
#         fn = [f for f in pset.primitives[pset.ret] if size+f.arity <= s_desired]

#         # Maximum arity of the above functions.
#         a_max = 0 if fn == [] else max([f.arity for f in fn])

#         # Determine which of the above functions have an arity 
#         # that is not too small.
#         temp = []
#         for f in fn:
#             # Maximum possible size of the subprogram rooted at the 
#             # current node, excluding this node, if the current 
#             # node is given to be `f`. (For simplicity, it can be 
#             # assumed that `a_max` is valid when choosing children 
#             # nodes.)
#             s = f.arity * get_max_size(a_max, d_max-(d+1))

#             # Maximum possible size of the overall program if the 
#             # relevant node under consideration is chosen to be 
#             # the function `f`. This maximum size would occur if 
#             # every outstanding node within the current stack is 
#             # made to be the root of a full `a_max`-ary subtree 
#             # such that the sum of (i) the depth of this subtree 
#             # and (ii) the depth of the root node within the 
#             # overall program is equal to `d_max`. (Again, for 
#             # simplicity, we assume that `a_max` is valid for 
#             # choosing children nodes.)
#             s_max = (size + s if stack == [] else size + s + sum(
#                 [get_max_size(a_max, d_max-d)-1 for d in stack]))

#             if s_max >= s_desired:
#                 temp.append(f)

#         # Functions with an arity that is not too small.
#         fn = temp
#         a_max = 0 if fn == [] else max([f.arity for f in fn])

#         if fn == [] and size < s_desired:
#             # Invalid program.
#             return None

#         # Maximum possible program size if the node currently
#         # under consideration is chosen to be a terminal.
#         s_max = (size if stack == [] or fn == [] else 
#             size + sum([get_max_size(a_max, d_max-d)-1 for d in stack]))

#         # Boolean to determine if the current node should be a terminal.
#         choose_terminal = (
#             (fn == []) or (d == d_max) or (size == s_desired) or (
#             random.random() < pset.terminalRatio and s_max >= s_desired))

#         if choose_terminal:
#             # A random terminal node is to be chosen.
#             terminal = random.choice(pset.terminals[pset.ret])
#             terminal = terminal() if isclass(terminal) else terminal
#             program.append(terminal)
#         else:
#             # A random (valid) function node is to be chosen.
#             f = random.choice(fn)
#             program.append(f)

#             for _ in range(f.arity):
#                 # Add a stack element for each function argument.
#                 stack.append(d + 1)

#             if (d + 1) > depth:
#                 # Update overall depth.
#                 depth += 1

#             # Update program size.
#             size += f.arity

#     if size != s_desired:
#         # Invalid program.
#         return None
#     else:
#         return program


########################################################################
# Some user-defined GP functions.
########################################################################

def add(x1, x2):
    """Return result of addition."""
    return x1 + x2

def aq(x1, x2):
    """Return result of analytical quotient.
    
    The analytical quotient is as defined by Ni et al. in their paper 
    'The use of an analytic quotient operator in genetic programming':  
    `aq(x1, x2) = (x1)/(sqrt(1+x2^(2)))`.
    """
    try:
        return (x1) / (math.sqrt(1 + x2 ** (2)))
    except OverflowError:
        return float("inf")

def exp(x): 
    """Return result of exponentiation, base `e`."""
    try:
        return math.exp(x)
    except OverflowError:
        return float("inf")

def log(x):
    """Return result of protected logarithm, base `e`."""
    return math.log(abs(x)) if x != 0 else 0

def mul(x1, x2):
    """Return result of multiplication."""
    return x1 * x2

def sin(x):
    """Return result of sine."""
    return math.sin(x)

def sqrt(x):
    """Return result of protected square root."""
    return math.sqrt(abs(x))

def sub(x1, x2):
    """Return result of subtraction."""
    return x1 - x2

def tanh(x):
    """Return result of hyperbolic tangent."""
    return math.tanh(x)


########################################################################
# Some user-defined GP function sets.
########################################################################

nicolau_a = [
  [add, 2], [sub, 2], [mul, 2], [aq, 2]]

nicolau_b = [
  [sin, 1], [tanh, 1], [add, 2], [sub, 2], [mul, 2], [aq, 2]]

nicolau_c = [
  [sin, 1], [tanh, 1], [exp, 1], [log, 1], [sqrt, 1], 
  [add, 2], [sub, 2], [mul, 2], [aq, 2]]


########################################################################
# Generation of random programs.
########################################################################

# Program opcode width, used to calculate the number of constants.
opcode_width = 8

# Function sets.
function_sets = {
    'nicolau_a': (nicolau_a, 9, 32),
    'nicolau_b': (nicolau_b, 7, 8),
    'nicolau_c': (nicolau_c, 7, 8)
}

# Maximum arity for each function set.
max_arities = [max([arity for (_, arity) in function_set])
    for _, (function_set, *_) in function_sets.items()]

# Maximum number of size bins.
max_num_size_bins = max([int(math.ceil(
    get_max_size(max_arity, max_depth)/bin_size))
    for (_, (_, max_depth, bin_size)), max_arity in 
        zip(function_sets.items(), max_arities)])

# Desired number of programs to be stored within each size bin.
num_programs_per_size_bin = 128

# Program initialization strategy.
gen_strategy = 'grow'

# Dictionary to map each function set name to the list of files
# that track the programs generated for each size bin associated
# with the function set specified by this name.
program_dict = {name: [([], [], [], [], [], []) 
    for i in range(max_num_size_bins)] for name in function_sets}

# Dictionaries to store `PrimitiveSet` and `PrimitiveTree` 
# objects generated by DEAP.
primitive_sets = {name: None for name in function_sets}
primitive_trees = {name: [[] for _ in range(max_num_size_bins)]
    for name in function_sets}

########################################################################

# Generate random programs for all function sets.
for name, (function_set, max_depth, bin_size) in function_sets.items():

    # Number of functions within function set.
    num_functions = len(function_set)

    # Name strings for functions.
    function_names = [function_set[i][0].__name__ 
        for i in range(num_functions)]

    # Maximum arity for function set.
    max_arity = max([arity for _, arity in function_set])

    # Maximum program size for function set.
    max_possible_size = get_max_size(max_arity, max_depth)

    # Number of size bins.
    num_size_bins = int(math.ceil(max_possible_size/bin_size))

    # Number of variables within primitive set.
    num_variables = num_functions-1

    # Name strings for variables.
    variable_names = ['v'+str(i) for i in range(num_variables)]

    # Number of constants within primitive set.
    num_constants = 2**(opcode_width)-(num_functions+1)-num_variables

    # Desired "generation trait" for random programs.
    desired_trait = 'size'

    # Primitive set and list of (name strings for) constants.
    primitive_set, constant_names = generate_primitive_set(
        function_set, num_variables, num_constants, 'erc_'+name)

    primitive_sets[name] = primitive_set

    # Preserve random constants.
    with open(
        f'{root_dir}/{name}/constants.txt', 'w') as f:
        for constant in constant_names:
            f.write(f'{constant}\n')

    # print('Opcode width: ', opcode_width)
    # print('Function set: ', function_set)
    # print('Maximum function arity: ', max_arity)
    # print('Number of functions: ', num_functions)
    # print('Number of variables: ', num_variables)
    # print('Number of constants: ', num_constants)
    # print('Maximum program depth: ', max_depth)
    # print('Maximum program size: ', max_possible_size)
    # print('Program generation, strategy: ', gen_strategy)
    # print('Program generation, desired trait:', desired_trait)
    # print('\n')

    ####################################################################

    # Create a uniform distribution of programs based on size bins.

    # Minimum depth for size-constrained programs.
    min_depth = 1

    for i in range(num_size_bins):

        # Minimum/maximum sizes of programs for size bin `i`.
        min_size = i*bin_size+1
        max_size = max_possible_size if i==num_size_bins-1 else (i+1)*bin_size

        # Number of distinct random programs generated for size bin `i`.
        j = 0

        while j < num_programs_per_size_bin:

            program = generate_program(
                gen_strategy, primitive_set, min_depth, max_depth,
                min_size, max_size, desired_trait)
            # program = generate_program(
            #     primitive_set, max_depth, random.randint(min_size, max_size))

            if program is None: 
                continue

            # Extract some information about the program.

            nodes, edges, labels = gp.graph(program)

            # Programs of interest (POI).
            # poi = (('nicolau_c', 5, 63), ('nicolau_c', 9, 60), 
            #        ('nicolau_c', 10, 18), ('nicolau_c', 11, 111),
            #        ('nicolau_c', 13, 18), ('nicolau_c', 14, 1), 
            #        ('nicolau_c', 16, 73), ('nicolau_c', 17, 69), 
            #        ('nicolau_c', 18, 96), ('nicolau_c', 19, 70), 
            #        ('nicolau_c', 19, 87), ('nicolau_c', 20, 64), 
            #        ('nicolau_c', 21, 16), ('nicolau_c', 21, 51),
            #        ('nicolau_c', 27, 72))
            
            # for name_, i_, j_ in poi:
            #     if (name == name_) and (i == i_) and (j == j_):
            #         # Print graphical representation of specified program.
            #         g = pgv.AGraph()
            #         g.add_nodes_from(nodes)
            #         g.add_edges_from(edges)
            #         g.layout(prog='dot')

            #         for i_ in nodes:
            #             n = g.get_node(i_)
            #             n.attr["label"] = labels[i_]

            #         g.draw(f'{root_dir}/../graphics/tree_'
            #                f'{name}_bin_{i}_program_{j}.pdf')

            # Size of program.
            size = nodes[-1]+1

            # A tree representation of the program.
            program = gp.PrimitiveTree(program)

            # String representation of program.
            program_str = str(program)

            # Depth of program.
            depth = program.height

            # Ensure that the program depth and size are permissible.
            if (depth > max_depth) or (size > max_size): 
                print('Uh-oh...')
                continue

            # Retrieve the relevant program dictionary tuple.
            (programs, depths, sizes, function_counts,
                variable_counts, constant_counts) = program_dict[name][i]

            # Number of programs currently stored in bin with index `i`.
            num_programs = len(programs)

            # Write the newly generated program and its relevant 
            # information to the dictionary if this new program is 
            # syntactically (not semantically) distinct from all 
            # other programs stored in the relevant bin.
            if programs.count(program_str) == 0:

                # Increment number of distinct random programs.
                j += 1

                # Preserve `PrimitiveTree` object.
                primitive_trees[name][i].append(program)

                # Extract some additional information about the program.

                # Numbers of instances for each type of function,
                # variable terminal, and constant terminal.
                function_count = [0]*(num_functions)
                variable_count = [0]*(num_variables)
                constant_count = [0]*(num_constants)
                
                for node in nodes:

                    node_name = labels[node]

                    if (node_name in function_names):
                        index = function_names.index(node_name)
                        function_count[index] += 1
                    elif (node_name in variable_names):
                        index = variable_names.index(node_name)
                        variable_count[index] += 1
                    else:
                        index = constant_names.index(node_name)
                        constant_count[index] += 1


                # Update the elements of the relevant dictionary tuple.

                programs.append(program_str)
                depths.append(depth)
                sizes.append(size)

                function_counts = function_count if (
                    function_counts == []) else ([sum(i) for i in 
                        list(zip(function_counts, function_count))])
                    
                variable_counts = variable_count if (
                    variable_counts == []) else ([sum(i) for i in 
                        list(zip(variable_counts, variable_count))])

                constant_counts = constant_count if (
                    constant_counts == []) else ([sum(i) for i in 
                        list(zip(constant_counts, constant_count))])

                program_dict[name][i] = (programs, depths, sizes,
                    function_counts, variable_counts, constant_counts)
            
            
# Pickle the relevant dictionary, so that it can be used by
# other scripts (e.g., `stats.py`).
with open(f'{root_dir}/programs.pkl', 'wb') as f:
    pickle.dump(program_dict, f)


# For each function set, print the number of programs currently 
# within each size bin, and if each bin has been filled with 
# `num_programs_per_size_bin` programs, create a file that 
# contains the programs from every bin, if such a file does 
# not already exist.

# print('Numbers of programs:')

for name, (_, max_depth, bin_size) in function_sets.items():

    # Maximum program size for function set.
    max_size = get_max_size(max_arity, max_depth)

    # Number of size bins.
    num_size_bins = int(math.ceil(max_size/bin_size))

    # Number of programs per size bin.
    num_programs = [len(programs) for programs,*_ in program_dict[name]]

    # print(name, '=', num_programs, '\n')

    all_bins_are_filled = True if (min(num_programs[0:num_size_bins]) == 
        num_programs_per_size_bin) else False

    if (all_bins_are_filled):
        with open(f'{root_dir}/{name}/programs_deap.txt', 'w+') as f:
            for programs, *_ in program_dict[name]:
                for program in programs:
                    f.write(f'{program}\n')


######################################################################## 
# Profiling of program evaluation mechanism given by DEAP.
########################################################################

# Note that the following was not relegated to another 
# script, since it was seemingly not trivial to preserve 
# the primitive sets and namespaces created by DEAP.

# Fitness outputs.


def evaluate(primitive_set, trees, inputs, target):
    """Return list of fitness scores for programs.
    
    The "R-squared" function implemented by the `sklearn.metrics` 
    module is used as a fitness function.

    Keyword arguments:
    primitive_set -- Primitive set, of type `PrimitiveSet`, used to 
        compile each `PrimitiveTree` object given by `trees`.
    trees -- Tuple of `PrimitiveTree` objects.
    inputs -- Tuple of input vectors.
    target -- Tuple of target vectors.
    """

    def evaluate_(tree):
        try:
            # Transform `PrimitiveTree` object into a callable function.
            program = gp.compile(tree, primitive_set)

            # Calculate program outputs, i.e., estimations of target vector.
            estimated = tuple(program(*input) for input in inputs)

            # Calculate and return fitness.
            return math.sqrt(mean_squared_error(target, estimated))
        except ValueError:
            return float("inf")

    # Calculate fitness scores for the set of trees in parallel, by way 
    # of the `pathos.pools` module. Note that this module is utilized 
    # instead of the standard `multiprocessing` module since the latter 
    # does not readily support lambda functions. (All available logical 
    # CPU cores are utilized by default. To utilize a different amount, 
    # specify the `nodes` attribute via the `ProcessPool` constructor.
    # This property can also be printed out, if need be.)
    fitness = ProcessPool().map(evaluate_, trees)

    # fitness = []
    # for tree in trees:
    #     fitness.append(evaluate_(tree))

    return fitness


# Maximum number of variables across all functions sets.
max_num_variables = max([len(function_set)-1 
    for (function_set, *_) in function_sets.items()])

# Numbers of fitness cases.
num_fitness_cases = (10, 100, 1000, 10000, 100000)

# Random fitness case vector for maximum amount of fitness cases.
inputs_ = np.array(
    [[random.random() for _ in range(max_num_variables)] 
    for _ in range(max(num_fitness_cases))])

# Preserve fitness cases for reference.
with open(f'{root_dir}/inputs.pkl', 'wb') as f:
    pickle.dump(inputs_, f)

# Random target vector for maximum amount of fitness cases.
target_ = np.array([random.random() for _ in range(max(num_fitness_cases))])

# Preserve target for reference.
with open(f'{root_dir}/target.pkl', 'wb') as f:
    pickle.dump(target_, f)

# Number of times in which the `timeit.repeat` function is
# called, in order to generate a list of median average
# runtimes.
num_epochs = 1

# Value for the `repeat` argument of the `timeit.repeat` method.
repeat = 1

# Value for the `number` argument of the `timeit.repeat` method.
number = 1

# Median average runtimes for programs within each size bin,
# for each number of fitness cases, for each function set.
med_avg_runtimes = []

# Fitness outputs.
fitnesses = []

# for name, (function_set, max_depth, bin_size) in list(
#     function_sets.items())[2:3]:
for name, (function_set, max_depth, bin_size) in function_sets.items():
    # For each function set...
    print(f'Function set `{name}`:')

    # Number of functions within function set.
    num_functions = len(function_set)

    # Maximum arity for function set.
    max_arity = max([arity for _, arity in function_set])

    # Maximum program size for function set.
    max_possible_size = get_max_size(max_arity, max_depth)

    # Number of size bins.
    num_size_bins = int(math.ceil(max_possible_size/bin_size))

    # Primitive set relevant to function set.
    primitive_set = primitive_sets[name]

    # Number of variables for primitive set.
    num_variables = num_functions - 1

    # Prepare for statistics relevant to function set.
    # sizes.append([])
    med_avg_runtimes.append([])
    fitnesses.append([])

    # For each amount of fitness cases, and for each size bin, 
    # calculate the relevant statistics.

    for nfc in num_fitness_cases:
    # for nfc in (100000,):
    # for nfc in (10,):
        # For each number of fitness cases...
        print(f'Number of fitness cases: `{nfc}`')

        # Fitness cases relevant to function set and `nfc`.
        inputs = inputs_[:nfc, :num_variables]

        # Target relevant to `nfc`.
        target = target_[:nfc]

        # Prepare for statistics relevant to the 
        # numbers of fitness cases and size bins.
        med_avg_runtimes[-1].append([[] for _ in range(num_size_bins)])
        fitnesses[-1].append([[] for _ in range(num_size_bins)])

        for i in range(num_size_bins):
            # For each size bin...
            print(f'({dt.datetime.now().ctime()}) Size bin `{i+1}`...')

            # `PrimitiveTree` objects for size bin `i`.
            trees = tuple(primitive_trees[name][i])

            # Calculate and append fitness values for current size bin.
            fitnesses[-1][-1][i] = evaluate(
                primitive_set, trees, inputs, target)

            for _ in range(num_epochs):
                # For each epoch...

                # Raw runtimes after running the `fitness_func_wrap` 
                # function a total of `repeat * number` times. 
                # The resulting object is a list of `repeat` values,
                # where each represents a raw runtime after running
                # the relevant code `number` times.
                runtimes = timeit.Timer(
                    'evaluate(primitive_set, trees, inputs, target)',
                    globals=globals()).repeat(repeat=repeat, number=number)

                # Average runtimes, taking into account `number`.
                avg_runtimes = [runtime/number for runtime in runtimes]

                # Calculate and append median average runtime.
                med_avg_runtimes[-1][-1][i].append(
                    np.median(avg_runtimes))

# Preserve results.
results = [fitnesses, med_avg_runtimes]
with open(f'{root_dir}/../results_deap.pkl', 'wb') as f:
    pickle.dump(results, f)
        

