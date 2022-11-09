"""Core evaluation algorithms."""
from pathos.pools import ProcessPool

def standard(programs, X, primitive_set, n_threads=1):
    """Evaluate programs on given set of inputs."""
    if n_threads == -1:
        # Use all available threads.
        n_threads = None

    def evaluate(program, X=X, primitive_set=primitive_set):
        """Evaluate a single program on given set of inputs."""
        # Transform the program expression into a callable object.
        program.compile(primitive_set)
        # Evaluate the program on each input.
        return tuple(program(*X_) for X_ in X)

    # Perform a map operation for evaluation.
    # return list(map(evaluate, programs))
    fitness = ProcessPool(n_threads).map(evaluate, programs)
    return fitness