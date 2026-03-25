from random import randint
from individual import Individual
from datetime import datetime
from socket import gethostname
from os import getpid
import time
import parameter
from tree import Tree
from random import choice, randint, randrange
from derivation import generate_tree
from evaluate_fitness import evaluate_fitness

def initialization(population_size):

    # Time stamp
    start = datetime.now()
    hms = "%02d%02d%02d" % (start.hour, start.minute, start.second)
    time_stamp = "_".join([gethostname(),
                           str(start.year)[2:],
                           str(start.month),
                           str(start.day),
                           hms,
                           str(start.microsecond),
                           str(getpid())])

    # Initialize experiment name
    experiment_name = getattr(parameter, 'Experiment_name', time_stamp)

    # Initialise empty population.
    init_method = initialization_methods(parameter.Initilization)
    individuals = init_method(population_size)

    # Evaluate initial population
    individuals = evaluate_fitness(individuals, initial=parameter.Unique_ind_fitness)

    return individuals,time.time(),experiment_name, time_stamp

def initialization_methods(method_name):
    # Based on method name return specific method
    methods = {
        "uniform_genome": uniform_genome,
        "uniform_tree": uniform_tree
    }
    return methods.get(method_name, uniform_genome)




def sample_genome(Codon_size=parameter.Codon_size, Genotype_length=parameter.Genotype_length):
    return [randint(0, Codon_size) for _ in range(Genotype_length)]

def uniform_genome(size):
    return [Individual(sample_genome(), None) for _ in range(size)]


def uniform_tree(size):
    return [generate_ind_tree(parameter.Max_tree_depth,
                              "random") for _ in range(size)]

def generate_ind_tree(max_depth, method):
    """
    Generate an individual using a given subtree initialisation method.

    :param max_depth: The maximum depth for the initialised subtree.
    :param method: The method of subtree initialisation required.
    :return: A fully built individual.
    """

    # Initialise an instance of the tree class
    ind_tree = Tree(str(parameter.Grammar_file.start_rule["symbol"]), None)

    # Generate a tree
    genome, output, nodes, _, depth = generate_tree(ind_tree, [], [], method,
                                                    0, 0, 0, max_depth)

    # Get remaining individual information
    phenotype, invalid, used_cod = "".join(output), False, len(genome)

    if parameter.Grammar_file.python_mode:
        # Grammar contains python code

        phenotype = python_filter(phenotype)

    # Initialise individual
    ind = Individual(genome, ind_tree, map_ind=False)

    # Set individual parameters
    ind.phenotype, ind.nodes = phenotype, nodes
    ind.depth, ind.used_codons, ind.invalid = depth, used_cod, invalid

    # Generate random tail for genome.
    ind.genome = genome + [randint(0, parameter.Codon_size) for
                           _ in range(int(ind.used_codons / 2))]

    return ind


def python_filter(txt):
    """ Create correct python syntax.

    We use {: and :} as special open and close brackets, because
    it's not possible to specify indentation correctly in a BNF
    grammar without this type of scheme."""

    indent_level = 0
    tmp = txt[:]
    i = 0
    while i < len(tmp):
        tok = tmp[i:i + 2]
        if tok == "{:":
            indent_level += 1
        elif tok == ":}":
            indent_level -= 1
        tabstr = "\n" + "  " * indent_level
        if tok == "{:" or tok == ":}":
            tmp = tmp.replace(tok, tabstr, 1)
        i += 1
    # Strip superfluous blank lines.
    txt = "\n".join([line for line in tmp.split("\n")
                     if line.strip() != ""])
    return txt