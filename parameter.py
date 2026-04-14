# Name of the experiment
Experiment_name = "Census_race_acc_aod_CASH"

Dataset_train = 'data/Census_race/Census_race_train.csv'
Dataset_val = 'data/Census_race/Census_race_val.csv'
Dataset_test = 'data/Census_race/Census_race_test.csv'

# Basic parameters
Random_seed = 0

Codon_size=99999
Genotype_length=2000

# Grammar, and Fitness
Grammar_file = 'grammar/CASH.bnf'
Fitness_file = 'fitness.fitness_acc_aod'

Max_tree_depth=90
Tournament_size=2

Generation = 300
Population = 300

# uniform_tree/uniform_genome
Initilization = "uniform_tree"
# Unique_ind_fitness (ensure unique fitness for each individual during initialization)
Unique_ind_fitness = True

# int_flip_per_codon/subtree
Crossover = 'subtree'
Crossover_probability = 0.7

# variable_onepoint/subtree
Mutation = 'subtree'
Mutation_probability = 0.1

# tournament/nsga2_selection
Selection = 'nsga2_selection'

# generational/nsga2_replacement/nsga2_lim_replacement
Replacement = 'nsga2_lim_replacement'

# Max_dup_fitness for nsga2_lim_replacement (limit number of individuals with same fitness)
Max_dup_fitness = 3

# Cache
Cache = True
Lookup_fitness = True
Lookup_bad_fitness = False
Mutate_duplicates = False
