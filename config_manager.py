import argparse
import importlib
import random
import numpy as np
import parameter


def str2bool(v):
    """Convert string to boolean value for argparse"""
    if isinstance(v, bool):
        return v
    if v is None:
        return True  # Default behavior when flag is provided without value
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_arguments():
    """Parameters from command code"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--Fitness_file', type=str, default=parameter.Fitness_file)
    parser.add_argument('--Grammar_file', type=str, default=parameter.Grammar_file)
    parser.add_argument('--Random_seed', type=int, default=parameter.Random_seed)
    parser.add_argument('--Codon_size', type=int, default=parameter.Codon_size)
    parser.add_argument('--Genotype_length', type=int, default=parameter.Genotype_length)
    parser.add_argument('--Max_tree_depth', type=int, default=parameter.Max_tree_depth)
    parser.add_argument('--Tournament_size', type=int, default=parameter.Tournament_size)
    parser.add_argument('--Generation', type=int, default=parameter.Generation)
    parser.add_argument('--Population', type=int, default=parameter.Population)
    parser.add_argument('--Initilization', type=str, default=parameter.Initilization)
    parser.add_argument('--Crossover', type=str, default=parameter.Crossover)
    parser.add_argument('--Crossover_probability', type=float, default=parameter.Crossover_probability)
    parser.add_argument('--Mutation', type=str, default=parameter.Mutation)
    parser.add_argument('--Mutation_probability', type=float, default=parameter.Mutation_probability)
    parser.add_argument('--Selection', type=str, default=parameter.Selection)
    parser.add_argument('--Replacement', type=str, default=parameter.Replacement)
    parser.add_argument('--Max_dup_fitness', type=int, default=parameter.Max_dup_fitness)
    parser.add_argument('--Cache', type=str2bool, default=parameter.Cache)
    parser.add_argument('--Lookup_fitness', type=str2bool, default=parameter.Lookup_fitness)
    parser.add_argument('--Lookup_bad_fitness', type=str2bool, default=parameter.Lookup_bad_fitness)
    parser.add_argument('--Mutate_duplicates', type=str2bool, default=parameter.Mutate_duplicates)
    parser.add_argument('--Unique_ind_fitness', type=str2bool, default=parameter.Unique_ind_fitness)
    parser.add_argument('--Experiment_name', type=str, default=parameter.Experiment_name)
    parser.add_argument('--Dataset_train', type=str, default=parameter.Dataset_train)
    parser.add_argument('--Dataset_val', type=str, default=parameter.Dataset_val)
    parser.add_argument('--Dataset_test', type=str, default=parameter.Dataset_test)

    return parser.parse_args()


def apply_parameters(args):
    """Apply parameters"""
    parameter.Random_seed = args.Random_seed
    parameter.Codon_size = args.Codon_size
    parameter.Genotype_length = args.Genotype_length
    parameter.Max_tree_depth = args.Max_tree_depth
    parameter.Tournament_size = args.Tournament_size
    parameter.Generation = args.Generation
    parameter.Population = args.Population
    parameter.Initilization = args.Initilization
    parameter.Crossover = args.Crossover
    parameter.Crossover_probability = args.Crossover_probability
    parameter.Mutation = args.Mutation
    parameter.Mutation_probability = args.Mutation_probability
    parameter.Selection = args.Selection
    parameter.Replacement = args.Replacement
    parameter.Max_dup_fitness = args.Max_dup_fitness
    parameter.Cache = args.Cache
    parameter.Lookup_fitness = args.Lookup_fitness
    parameter.Lookup_bad_fitness = args.Lookup_bad_fitness
    parameter.Mutate_duplicates = args.Mutate_duplicates
    parameter.Unique_ind_fitness = args.Unique_ind_fitness
    parameter.Experiment_name = args.Experiment_name
    parameter.Dataset_train = args.Dataset_train
    parameter.Dataset_val = args.Dataset_val
    parameter.Dataset_test = args.Dataset_test

    # Add Fitness
    parameter.Fitness_file = args.Fitness_file
    parameter.Fitness = importlib.import_module(args.Fitness_file).Fitness
    parameter.Fitness_function = parameter.Fitness()

    # Add Grammar
    from grammar import Grammar
    parameter.Grammar_filename = args.Grammar_file
    parameter.Grammar_file = Grammar(args.Grammar_file)

def setup_random_seed():
    """Set random seed"""
    random.seed(parameter.Random_seed)
    np.random.seed(parameter.Random_seed)


def initialize_config():

    args = parse_arguments()
    apply_parameters(args)
    setup_random_seed()