from random import choice, randint, random, sample, choices

from individual import Individual
import parameter

from numpy import isnan
import cache

def crossover(parents):
    """
    Perform crossover on a population of individuals. The size of the crossover
    population is defined as params['GENERATION_SIZE'] rather than params[
    'POPULATION_SIZE']. This saves on wasted evaluations and prevents search
    from evaluating too many individuals.

    :param parents: A population of parent individuals on which crossover is to
    be performed.
    :return: A population of fully crossed over individuals.
    """

    # Initialise an empty population.
    cross_pop = []

    while len(cross_pop) < (parameter.Population - parameter.Tournament_size + 1):
        inds_in = sample(parents, 2)

        inds_out = crossover_inds(inds_in[0], inds_in[1])

        if inds_out is None:
            # Crossover failed.
            pass

        else:

            # Extend the new population.
            cross_pop.extend(inds_out)

    return cross_pop

def crossover_inds(parent_0, parent_1):
    """
    Perform crossover on two selected individuals.

    :param parent_0: Parent 0 selected for crossover.
    :param parent_1: Parent 1 selected for crossover.
    :return: Two crossed-over individuals.
    """

    # Create copies of the original parents. This is necessary as the
    # original parents remain in the parent population and changes will
    # affect the originals unless they are cloned.
    ind_0 = parent_0.deep_copy()
    ind_1 = parent_1.deep_copy()

    crossover_method = cross_methods(parameter.Crossover)
    # Perform crossover on ind_0 and ind_1.
    inds = crossover_method(ind_0, ind_1)

    return inds

def cross_methods(method_name):
    # Based on method name return specific method
    methods = {
        "variable_onepoint": variable_onepoint,
        "subtree": subtree,
        "target_subtree": target_subtree
    }
    return methods.get(method_name, variable_onepoint)

def variable_onepoint(parent_0, parent_1, within_used=True):
    # Crossover cannot be performed on invalid individuals.

    ind_0 = parent_0.genome
    ind_1 = parent_1.genome

    if within_used:
        max_p_0, max_p_1 = parent_0.used_codons, parent_1.used_codons
    else:
        max_p_0, max_p_1 = len(ind_0), len(ind_1)

    pt_p_0, pt_p_1 = randint(1, max_p_0), randint(1, max_p_1)

    if random() < parameter.Crossover_probability:
        c_0 = ind_0[:pt_p_0] + ind_1[pt_p_1:]
        c_1 = ind_1[:pt_p_1] + ind_0[pt_p_0:]
    else:
        c_0, c_1 = ind_0[:], ind_1[:]

    return [Individual(c_0, None), Individual(c_1, None)]


def subtree(p_0, p_1):
    """
    Given two individuals, create two children using subtree crossover and
    return them. Candidate subtrees are selected based on matching
    non-terminal nodes rather than matching terminal nodes.

    :param p_0: Parent 0.
    :param p_1: Parent 1.
    :return: A list of crossed-over individuals.
    """

    def do_crossover(tree0, tree1, shared_nodes):
        """
        Given two instances of the representation.tree.Tree class (
        derivation trees of individuals) and a list of intersecting
        non-terminal nodes across both trees, performs subtree crossover on
        these trees.

        :param tree0: The derivation tree of individual 0.
        :param tree1: The derivation tree of individual 1.
        :param shared_nodes: The sorted list of all non-terminal nodes that are
        in both derivation trees.
        :return: The new derivation trees after subtree crossover has been
        performed.
        """

        # Randomly choose a non-terminal from the set of permissible
        # intersecting non-terminals.
        crossover_choice = choice(shared_nodes)

        # Find all nodes in both trees that match the chosen crossover node.
        nodes_0 = tree0.get_target_nodes([], target=[crossover_choice])
        nodes_1 = tree1.get_target_nodes([], target=[crossover_choice])

        # Randomly pick a node.
        t0, t1 = choice(nodes_0), choice(nodes_1)

        # Check the parents of both chosen subtrees.
        p0 = t0.parent
        p1 = t1.parent

        if not p0 and not p1:
            # Crossover is between the entire tree of both tree0 and tree1.

            return t1, t0

        elif not p0:
            # Only t0 is the entire of tree0.
            tree0 = t1

            # Swap over the subtrees between parents.
            i1 = [id(i) for i in p1.children].index(id(t1))
            p1.children[i1] = t0

            # Set the parents of the crossed-over subtrees as their new
            # parents. Since the entire tree of t1 is now a whole
            # individual, it has no parent.
            t0.parent = p1
            t1.parent = None

        elif not p1:
            # Only t1 is the entire of tree1.
            tree1 = t0

            # Swap over the subtrees between parents.
            i0 = [id(i) for i in p0.children].index(id(t0))
            p0.children[i0] = t1

            # Set the parents of the crossed-over subtrees as their new
            # parents. Since the entire tree of t0 is now a whole
            # individual, it has no parent.
            t1.parent = p0
            t0.parent = None

        else:
            # The crossover node for both trees is not the entire tree.

            # For the parent nodes of the original subtrees, get the indexes
            # of the original subtrees.
            i0 = [id(i) for i in p0.children].index(id(t0))
            i1 = [id(i) for i in p1.children].index(id(t1))

            # Swap over the subtrees between parents.
            p0.children[i0] = t1
            p1.children[i1] = t0

            # Set the parents of the crossed-over subtrees as their new
            # parents.
            t1.parent = p0
            t0.parent = p1

        return tree0, tree1

    def intersect(l0, l1):
        """
        Returns the intersection of two sets of labels of nodes of
        derivation trees. Only returns matching non-terminal nodes across
        both derivation trees.

        :param l0: The labels of all nodes of tree 0.
        :param l1: The labels of all nodes of tree 1.
        :return: The sorted list of all non-terminal nodes that are in both
        derivation trees.
        """

        # Find all intersecting elements of both sets l0 and l1.
        shared_nodes = l0.intersection(l1)

        # Find only the non-terminals present in the intersecting set of
        # labels.
        shared_nodes = [i for i in shared_nodes if i in parameter.Grammar_file.non_terminals]

        return sorted(shared_nodes)

    if random() > parameter.Crossover_probability:
        # Crossover is not to be performed, return entire individuals.
        ind0 = p_1
        ind1 = p_0

    else:
        # Crossover is to be performed.

        if p_0.invalid:
            # The individual is invalid.
            tail_0 = []

        else:
            # Save tail of each genome.
            tail_0 = p_0.genome[p_0.used_codons:]

        if p_1.invalid:
            # The individual is invalid.
            tail_1 = []

        else:
            # Save tail of each genome.
            tail_1 = p_1.genome[p_1.used_codons:]

        # Get the set of labels of non terminals for each tree.
        labels1 = p_0.tree.get_node_labels(set())
        labels2 = p_1.tree.get_node_labels(set())

        # Find overlapping non-terminals across both trees.
        shared_nodes = intersect(labels1, labels2)
        shared_nodes.remove(parameter.Grammar_file.start_rule['symbol'])

        if len(shared_nodes) != 0:
            # There are overlapping NTs, cross over parts of trees.
            ret_tree0, ret_tree1 = do_crossover(p_0.tree, p_1.tree,
                                                shared_nodes)

        else:
            # There are no overlapping NTs, cross over entire trees.
            ret_tree0, ret_tree1 = p_1.tree, p_0.tree

        # Initialise new individuals using the new trees.
        ind0 = Individual(None, ret_tree0)
        ind1 = Individual(None, ret_tree1)

        # Preserve tails.
        ind0.genome = ind0.genome + tail_0
        ind1.genome = ind1.genome + tail_1

    return [ind0, ind1]


def target_subtree(p_0, p_1):
    """
    Given two individuals, create two children using subtree crossover and
    return them. Candidate subtrees are selected based on matching
    non-terminal nodes rather than matching terminal nodes.

    :param p_0: Parent 0.
    :param p_1: Parent 1.
    :return: A list of crossed-over individuals.
    """

    def do_crossover(tree0, tree1, shared_nodes):
        """
        Given two instances of the representation.tree.Tree class (
        derivation trees of individuals) and a list of intersecting
        non-terminal nodes across both trees, performs subtree crossover on
        these trees.

        :param tree0: The derivation tree of individual 0.
        :param tree1: The derivation tree of individual 1.
        :param shared_nodes: The sorted list of all non-terminal nodes that are
        in both derivation trees.
        :return: The new derivation trees after subtree crossover has been
        performed.
        """

        # Randomly choose a non-terminal from the set of permissible
        # intersecting non-terminals.
        # Dynamicly adjust the weights through generation

        generations = parameter.Generation
        current_gen = cache.generation[0]
        p_struct = 1.0 - (current_gen / generations)
        struct_list = parameter.Grammar_file.struct_nts

        S = sum(1 for n in shared_nodes if n in struct_list)
        U = len(shared_nodes) - S

        if S == 0 or U == 0:
            weights = [1.0] * len(shared_nodes)
        else:
            w_struct = p_struct / S
            w_unstruct = (1.0 - p_struct) / U
            weights = [w_struct if n in struct_list else w_unstruct for n in shared_nodes]

        crossover_choice = choices(shared_nodes, weights=weights, k=1)[0]

        # Find all nodes in both trees that match the chosen crossover node.
        nodes_0 = tree0.get_target_nodes([], target=[crossover_choice])
        nodes_1 = tree1.get_target_nodes([], target=[crossover_choice])

        # Randomly pick a node.
        t0, t1 = choice(nodes_0), choice(nodes_1)

        # Check the parents of both chosen subtrees.
        p0 = t0.parent
        p1 = t1.parent

        if not p0 and not p1:
            # Crossover is between the entire tree of both tree0 and tree1.

            return t1, t0

        elif not p0:
            # Only t0 is the entire of tree0.
            tree0 = t1

            # Swap over the subtrees between parents.
            i1 = [id(i) for i in p1.children].index(id(t1))
            p1.children[i1] = t0

            # Set the parents of the crossed-over subtrees as their new
            # parents. Since the entire tree of t1 is now a whole
            # individual, it has no parent.
            t0.parent = p1
            t1.parent = None

        elif not p1:
            # Only t1 is the entire of tree1.
            tree1 = t0

            # Swap over the subtrees between parents.
            i0 = [id(i) for i in p0.children].index(id(t0))
            p0.children[i0] = t1

            # Set the parents of the crossed-over subtrees as their new
            # parents. Since the entire tree of t0 is now a whole
            # individual, it has no parent.
            t1.parent = p0
            t0.parent = None

        else:
            # The crossover node for both trees is not the entire tree.

            # For the parent nodes of the original subtrees, get the indexes
            # of the original subtrees.
            i0 = [id(i) for i in p0.children].index(id(t0))
            i1 = [id(i) for i in p1.children].index(id(t1))

            # Swap over the subtrees between parents.
            p0.children[i0] = t1
            p1.children[i1] = t0

            # Set the parents of the crossed-over subtrees as their new
            # parents.
            t1.parent = p0
            t0.parent = p1

        return tree0, tree1

    def intersect(l0, l1):
        """
        Returns the intersection of two sets of labels of nodes of
        derivation trees. Only returns matching non-terminal nodes across
        both derivation trees.

        :param l0: The labels of all nodes of tree 0.
        :param l1: The labels of all nodes of tree 1.
        :return: The sorted list of all non-terminal nodes that are in both
        derivation trees.
        """

        # Find all intersecting elements of both sets l0 and l1.
        shared_nodes = l0.intersection(l1)

        # Find only the non-terminals present in the intersecting set of
        # labels.
        shared_nodes = [i for i in shared_nodes if i in parameter.Grammar_file.non_terminals]

        return sorted(shared_nodes)

    if random() > parameter.Crossover_probability:
        # Crossover is not to be performed, return entire individuals.
        ind0 = p_1
        ind1 = p_0

    else:
        # Crossover is to be performed.

        if p_0.invalid:
            # The individual is invalid.
            tail_0 = []

        else:
            # Save tail of each genome.
            tail_0 = p_0.genome[p_0.used_codons:]

        if p_1.invalid:
            # The individual is invalid.
            tail_1 = []

        else:
            # Save tail of each genome.
            tail_1 = p_1.genome[p_1.used_codons:]

        # Get the set of labels of non terminals for each tree.
        labels1 = p_0.tree.get_node_labels(set())
        labels2 = p_1.tree.get_node_labels(set())

        # Find overlapping non-terminals across both trees.
        shared_nodes = intersect(labels1, labels2)
        shared_nodes.remove(parameter.Grammar_file.start_rule['symbol'])

        if len(shared_nodes) != 0:
            # There are overlapping NTs, cross over parts of trees.
            ret_tree0, ret_tree1 = do_crossover(p_0.tree, p_1.tree,
                                                shared_nodes)

        else:
            # There are no overlapping NTs, cross over entire trees.
            ret_tree0, ret_tree1 = p_1.tree, p_0.tree

        # Initialise new individuals using the new trees.
        ind0 = Individual(None, ret_tree0)
        ind1 = Individual(None, ret_tree1)

        # Preserve tails.
        ind0.genome = ind0.genome + tail_0
        ind1.genome = ind1.genome + tail_1

    return [ind0, ind1]