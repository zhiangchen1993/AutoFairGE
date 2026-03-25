from random import choice, randint, random, choices

from individual import Individual
import parameter
import math
from derivation import generate_tree

import cache

def mutation(parents):
    """
    Perform mutation on a population of individuals. Calls mutation operator as
    specified in params dictionary.

    :param pop: A population of individuals to be mutated.
    :return: A fully mutated population.
    """

    # Initialise empty pop for mutated individuals.

    new_pop = []

    mutation_method = mutate_methods(parameter.Mutation)
    for ind in parents:
        new_ind = mutation_method(ind)
        new_pop.append(new_ind)

    return new_pop

def mutate_methods(method_name):
    # Based on method name return specific method
    methods = {
        "int_flip_per_codon": int_flip_per_codon,
        "subtree": subtree,
        "target_subtree": target_subtree
    }
    return methods.get(method_name, int_flip_per_codon)

def int_flip_per_codon(ind, Codon_size=parameter.Codon_size):
    eff_length = ind.used_codons

    if not eff_length or math.isnan(eff_length):
        return ind

    p_mut = parameter.Mutation_probability

    for i in range(eff_length):
        if random() < p_mut:
            ind.genome[i] = randint(0, Codon_size)

    new_ind = Individual(ind.genome, None)

    return new_ind


def subtree(ind, mutation_events=1):
    """
    Mutate the individual by replacing a randomly selected subtree with a
    new randomly generated subtree. Guaranteed one event per individual, unless
    params['MUTATION_EVENTS'] is specified as a higher number.

    :param ind: An individual to be mutated.
    :return: A mutated individual.
    """

    def subtree_mutate(ind_tree):
        """
        Creates a list of all nodes and picks one node at random to mutate.
        Because we have a list of all nodes, we can (but currently don't)
        choose what kind of nodes to mutate on. Handy.

        :param ind_tree: The full tree of an individual.
        :return: The full mutated tree and the associated genome.
        """

        # Find the list of nodes we can mutate from.
        targets = ind_tree.get_target_nodes([], target=parameter.Grammar_file.non_terminals)

        # Pick a node.
        new_tree = choice(targets)

        # Set the depth limits for the new subtree.
        if parameter.Max_tree_depth:
            # Set the limit to the tree depth.
            max_depth = parameter.Max_tree_depth - new_tree.depth

        else:
            # There is no limit to tree depth.
            max_depth = None

        # Mutate a new subtree.
        generate_tree(new_tree, [], [], "random", 0, 0, 0, max_depth)

        return ind_tree

    if ind.invalid:
        # The individual is invalid.
        tail = []

    else:
        # Save the tail of the genome.
        tail = ind.genome[ind.used_codons:]

    # Allows for multiple mutation events should that be desired.
    for i in range(mutation_events):
        ind.tree = subtree_mutate(ind.tree)

    # Re-build a new individual with the newly mutated genetic information.
    ind = Individual(None, ind.tree)

    # Add in the previous tail.
    ind.genome = ind.genome + tail

    return ind

def target_subtree(ind, mutation_events=1):
    """
    Mutate the individual by replacing a randomly selected subtree with a
    new randomly generated subtree. Guaranteed one event per individual, unless
    params['MUTATION_EVENTS'] is specified as a higher number.

    :param ind: An individual to be mutated.
    :return: A mutated individual.
    """

    def subtree_mutate(ind_tree):
        """
        Creates a list of all nodes and picks one node at random to mutate.
        Because we have a list of all nodes, we can (but currently don't)
        choose what kind of nodes to mutate on. Handy.

        :param ind_tree: The full tree of an individual.
        :return: The full mutated tree and the associated genome.
        """


        # Find the list of nodes we can mutate from.
        targets = ind_tree.get_target_nodes([], target=parameter.Grammar_file.non_terminals)

        generations = parameter.Generation
        current_gen = cache.generation[0]
        p_struct = 1.0 - (current_gen / generations)
        struct_list = parameter.Grammar_file.struct_nts

        S = sum(1 for n in targets if n.root in struct_list)
        U = len(targets) - S

        if S == 0 or U == 0:
            weights = [1.0] * len(targets)
        else:
            w_struct = p_struct / S
            w_unstruct = (1.0 - p_struct) / U
            weights = [w_struct if (n.root in struct_list) else w_unstruct for n in targets]

        # Pick a node.
        new_tree = choices(targets, weights=weights, k=1)[0]

        # Set the depth limits for the new subtree.
        if parameter.Max_tree_depth:
            # Set the limit to the tree depth.
            max_depth = parameter.Max_tree_depth - new_tree.depth

        else:
            # There is no limit to tree depth.
            max_depth = None

        # Mutate a new subtree.
        generate_tree(new_tree, [], [], "random", 0, 0, 0, max_depth)

        return ind_tree

    if ind.invalid:
        # The individual is invalid.
        tail = []

    else:
        # Save the tail of the genome.
        tail = ind.genome[ind.used_codons:]

    # Allows for multiple mutation events should that be desired.
    for i in range(mutation_events):
        ind.tree = subtree_mutate(ind.tree)

    # Re-build a new individual with the newly mutated genetic information.
    ind = Individual(None, ind.tree)

    # Add in the previous tail.
    ind.genome = ind.genome + tail

    return ind