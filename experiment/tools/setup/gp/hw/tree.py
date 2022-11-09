"""Program tree."""
from gp.core.primitive_set import PrimitiveSet
from .program import Program
from .node import Node

class Tree:
    """Class for program tree."""
    __slots__ = ('primitive_set', 'd', 'd_p')

    def __init__(self, primitive_set=PrimitiveSet(), d=1, d_p=0):
        self.primitive_set = primitive_set
        self.d = d
        self.d_p = d_p

    @property
    def m(self):
        """Return ary-ness of tree."""
        return self.primitive_set.m

    @property
    def d_s(self):
        """Return depth of sequential function tree."""
        return self.d - 1 - self.d_p - 1

    @staticmethod
    def _n_nodes(m, d):
        """Return number of nodes in full `m`-ary tree of depth `d`."""
        return Program.max_size(m, d)

    def max_size(self):
        """Return maximum program size for tree."""
        return Tree._n_nodes(self.m, self.d)

    def n_function_nodes(self):
        """Return number of function nodes."""
        return (self.d_s + 1) + Tree._n_nodes(self.m, self.d_p)

    def n_terminal_nodes(self):
        """Return number of terminal nodes in a compacted tree."""
        return self.m ** (self.d_p + 1)

    @staticmethod
    def _depth(m, d):
        """Generate list of node depths, specified by node index.

        Specifically, generate a list in which element `i` represents the 
        depth of node `i` within a full `m`-ary tree of depth `d`, where 
        node indices correspond to a preorder traversal of the tree.

        Note that, in this context, depth increases from zero, starting 
        with terminal nodes.
        """
        if d == 0:
            return [0]
        else:
            return [d] + (Tree._depth(m, d - 1) * m)

    def depth(self):
        """Generate list of node depths for a compacted tree."""
        return (list(reversed(range(self.d_p + 1, self.d))) 
            + Tree._depth(self.m, self.d_p))

    @staticmethod
    def _next_function(m, d, _d=0):
        """Generate list with next relevant function node indices.
        
        Specifically, generate list in which certain elements `i` contain 
        the index of the immediately following child of the parent of the 
        node whose index is `i` within a full `m`-ary function tree of 
        depth `d`, where node indices correspond to a preorder traversal 
        of the tree. More specifically, all elements `i` such that there 
        exists a node whose index is `i` and the aforementioned condition 
        holds will contain the index of the next child of the parent
        of node `i` within the relevant tree; all other elements within 
        the list will contain some arbitrary value.

        Note that the argument `_d` is meant for internal use only, 
        to build the relevant recursion---a user is not meant to specify 
        this value.
        """
        if _d == d:
            # Return list containing the next function node index for
            # the leftmost leaf node within the overall `m`-ary tree.
            return [_d + 1]
        else:
            # Number of nodes within the current subtree.
            n_nodes_ = Tree._n_nodes(m, d - _d)

            # Number of nodes within each child subtree.
            n_nodes_child = Tree._n_nodes(m, d - (_d + 1))

            # List of relevant indices for the subtree rooted at the
            # relevant leftmost child.
            leftmost_child = Tree._next_function(m, d, _d + 1)

            # List of relevant indices for the subtrees rooted at the
            # rightmost `m-1` children.
            right_children = [j + (i * n_nodes_child) for i in range(1, m) 
                for j in leftmost_child]

            # Return relevant list for the current subtree.
            return [_d + n_nodes_] + leftmost_child + right_children

    def next_function(self):
        """Generate next function node indices for a compacted tree."""
        n = self.n_function_nodes()
        return (list(range(self.d_s + 2)) 
            + [i + (self.d_s + 1) if i + (self.d_s + 1) < n else n - 1
                for i in Tree._next_function(self.m, self.d_p)[1:]])

    @staticmethod
    def _next_terminal(m, d):
        """Generate list with next relevant terminal node indices.
        
        Specifically, generate a list in which certain elements `i` contain
        the index of the first terminal node that corresponds to the next 
        child of the parent of the node whose index is `i` within a full 
        `m`-ary function tree of depth `d`, where node indices correspond 
        to a preorder traversal of the tree. More specifically, all elements
        `i` such that there exists a node whose index is `i` and the 
        aforementioned condition holds will contain the index of the first 
        terminal node that corresponds to the next child of the parent of 
        node `i` within the tree; all other elements within the list will
        contain some arbitrary value.
        """
        if d == 0:
            return [m]
        else:
            # Number of terminal nodes corresponding to each child 
            # subtree of the root node.
            n_terminal_nodes = m ** d

            # List of relevant indices for the subtree rooted at the
            # leftmost child.
            leftmost_child = Tree._next_terminal(m, d-1)

            # List of relevant indices for the subtrees rooted at the
            # rightmost `m-1` children.
            right_children = [j + (i * n_terminal_nodes) for i in range(1, m) 
                for j in leftmost_child]

            # Return relevant list for the current subtree.
            return [m ** (d + 1)] + leftmost_child + right_children

    def next_terminal(self):
        """Generate next terminal node indices for a compacted tree."""
        n = self.n_terminal_nodes()
        return ([0] * (self.d_s + 2) + [i if i < n else n - 1 
            for i in Tree._next_terminal(self.m, self.d_p)[1:]])

    def machine_code(self, program):
        """Compile `gp.hw.program.Program` object to tree.
        
        Specifically, generate the terminal select, constant, and 
        function machine codes relevant to the program tree.
        """
        if len(program) > self.max_size():
            raise ValueError('The given program is too large.')
        program = program + [Node()]
        s = len(program)

        # Number of functions in opcode set.
        n_functions = len(self.primitive_set.functions)

        # Function arities.
        a = [self.primitive_set.arity(f) for f in self.primitive_set.functions]

        # Number of terminal nodes.
        n_terminal_nodes = self.n_terminal_nodes()

        # Number of function nodes.
        n_function_nodes = self.n_function_nodes()

        # Look-up table for node depth, in relation to a given 
        # function node index.
        depth = self.depth()

        # Look-up table for the next function node index, 
        # in relation to a given function node index.
        next_function = self.next_function()

        # Look-up table for the next terminal node index, 
        # in relation to a given function node index.
        next_terminal = self.next_terminal()

        # Temporary counter to track the number of operands needed
        # until an operation or program is fully handled.
        arity = 1

        # Temporary counter for the number of operations outstanding.
        n_operations = 0

        # Stack to preserve the number of right subtrees needed
        # to handle some previously encountered operation(s).
        n_right_subtrees = [0] * (self.d-1)

        # Stacks to contain the next function/terminal node index
        # when jumping to a right subtree of a previous parent.
        next_function_ = [0] * (self.d-1)
        next_terminal_ = [0] * (self.d-1)

        # Terminal select, constant, and function select values.
        terminal_sel = [[-1] for _ in range(n_terminal_nodes)]
        constants = [[-1] for _ in range(n_terminal_nodes)]
        function_sel = [[-1] for _ in range(n_function_nodes)]

        # Linear program node pointer.
        i_l = 0

        # Terminal and function node pointers, respectively.
        i_t, i_f = 0, 0

        # By default, bypass the leftmost leaf function node in
        # the overall function tree, to handle the case in which 
        # the program consists of only a single terminal node.
        function_sel[self.d - 1][-1] = 0

        while i_l < s - 1:
            # Until the `null` function code is encountered...

            # Extract the next two adjacent nodes, and translate 
            # just the first.
            node, node_n = program[i_l], program[i_l + 1]

            # Until an appropriate depth is encountered, bypass nodes 
            # in the function tree. Note that if bypasses must occur 
            # and the current program node is a terminal, the relevant 
            # leftmost leaf node in the function subtree rooted at 
            # function node `i_f` will not be bypassed. Such leaf
            # node bypasses are handled elsewhere---see below.
            while depth[i_f] > 0 and (depth[i_f] + 1) != node.depth:
                function_sel[i_f][-1] = 0
                i_f += 1

            if node.function:
                # The current node is a function.
                function_sel[i_f][-1] = node.opcode
                arity = a[node.opcode - 1]
                if depth[i_f] > 0:
                    # The current function tree node has a depth 
                    # greater than zero; thus, either a terminal 
                    # or function node may be encountered next 
                    # within the program string. Regardless, advance 
                    # the function node index to point to the next 
                    # left child within the function tree.
                    i_f += 1
                    if arity > 1:
                        # The now previous function node (`i_f - 1`) is 
                        # meant to have some right subtree(s) for which
                        # arguments have yet to be encountered; consider 
                        # this program node outstanding and preserve the 
                        # next relevant node indices for the upcoming 
                        # right subtree.
                        n_right_subtrees[n_operations] = arity - 1
                        next_function_[n_operations] = next_function[i_f]
                        next_terminal_[n_operations] = next_terminal[i_f]
                        n_operations += 1
                    if node_n.terminal:
                        # The next node is a terminal and the depth 
                        # of the current function node is greater than 
                        # zero. Thus, the next terminal must be bypassed
                        # through at least the leftmost leaf node of the 
                        # function subtree rooted at function node 
                        # `i_f`. The `while` loop for bypassing given 
                        # above will not bypass this leaf node; we 
                        # perform the relevant bypass here.
                        function_sel[i_f + depth[i_f]][-1] = 0
                        # Set an appropriate temporary arity value 
                        # before encountering the upcoming terminal.
                        arity = 1
            elif node.terminal:
                # The current node is a terminal.
                terminal_sel[i_t][-1] = node.opcode - (1 + n_functions)
                if node.constant:
                    constants[i_t][-1] = node.value
                i_t += 1
                arity -= 1
                if arity == 0 and n_operations > 0:
                    # All operands for the most recent operation have 
                    # been encountered and some operation that was 
                    # encountered before the most recent operation has
                    # additional arguments outstanding; jump to the next
                    # relevant right subtree for this operation.
                    i_f = next_function_[n_operations-1]
                    i_t = next_terminal_[n_operations-1]
                    if depth[i_f] >= self.d_p:
                        # We jumped out of the parallel tree; append a 
                        # window to the new function node and to every 
                        # node that is a descendant of this new node,
                        # including terminals.
                        for i in range(i_f, n_function_nodes):
                            function_sel[i].append(-1)
                        for i in range(n_terminal_nodes):
                            terminal_sel[i].append(-1)
                            constants[i].append(-1)

                    if n_right_subtrees[n_operations-1] > 1:
                        # More right subtrees must eventually be visited
                        # for the most recent outstanding operation 
                        # (which is the parent of the newly encountered
                        # function node); preserve the next relevant 
                        # function/terminal indices for the upcoming 
                        # right subtree.
                        next_function_[n_operations-1] = next_function[i_f]
                        next_terminal_[n_operations-1] = next_terminal[i_f]
                        # Consider the new right subtree handled.
                        n_right_subtrees[n_operations-1] -= 1
                    else:
                        # All relevant right subtrees have been
                        # encountered for the most recent outstanding
                        # operation; consider it no longer outstanding.
                        n_operations -= 1

                    if node_n.terminal:
                        # The first node in the new right subtree is 
                        # a terminal. Thus, this terminal must be 
                        # bypassed through at least the leftmost 
                        # leaf node of the function subtree rooted 
                        # at function node `i_f`. The `while` loop 
                        # for bypassing given above will not bypass 
                        # this leaf node; we perform the relevant 
                        # bypass here.
                        function_sel[i_f + depth[i_f]][-1] = 0
                        # Set an appropriate temporary arity value 
                        # before encountering the upcoming terminal.
                        arity = 1

            # Advance the linear program index.
            i_l += 1

        return terminal_sel, constants, function_sel

    def max_windows(self, s):
        """Return maximum number(s) of windows for compacted tree.
        
        Specifically, in the context of a program of size `s`, 
        return a list containing the maximum number of windows 
        for the parallel subtree and the maximum number of windows 
        for each level of the sequential subtree, if one exists. 
        The first element of the list is the maximum number of 
        parallel windows, and if a sequential subtree exists (i.e., 
        if 0 <= d_p < d), then every element `i` such that `i > 0` 
        is the maximum number of windows for level `i` of the 
        sequential tree.
        """
        # Validate parameters.
        s = self.d + 1 if s < self.d + 1 else s
        s = self.max_size() if s > self.max_size() else s

        # Maximum number of parallel/sequential windows.
        windows = []

        # Compute the maximum number of windows for the parallel tree.
        s_ = s - (1 + self.d_p)
        d_ = Program.min_depth(s_, self.primitive_set)
        if d_ > (self.d - (1 + self.d_p)):
            # A tree of depth `d_` cannot extend from the root
            # node of the parallel tree, and the maximum number 
            # of parallel windows is however many can fit in the
            # overall tree.
            windows.append(self.m ** (self.d_s + 1))
        else:
            # A tree of depth `d_` can extend from the root node
            # of the parallel tree, and the maximum number of 
            # parallel windows is equivalent to the maximum number 
            # of terminal nodes for an `m`-ary tree consisting of 
            # `s_` nodes.
            windows.append(Program.max_terminal_nodes(s_, self.primitive_set))

        # Compute the maximum number of windows for each level
        # of the sequential tree, if one exists.
        for i in range(self.d_s + 1):
            s_ = s - (1 + self.d_p + 1 + i)
            d_ = Program.min_depth(s_, self.primitive_set)
            if d_ > (self.d - (1 + self.d_p + 1 + i)):
                windows.append(self.m ** (self.d_s - i))
            else:
                windows.append(
                    Program.max_terminal_nodes(s_, self.primitive_set))

        return windows