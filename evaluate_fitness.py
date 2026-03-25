import numpy as np
import parameter
from cache import cache
from mutation import mutation

def evaluate_fitness(individuals, initial=False):
    results = []

    for name, ind in enumerate(individuals):
        ind.name = name

        if ind.invalid or ind.phenotype == None or ind.used_codons == 0:
            # Invalid individuals cannot be evaluated and are given a bad
            # default fitness.
            if not isinstance(parameter.Fitness_function.maximise, list):
                ind.fitness = np.nan
            else:
                ind.fitness = [np.nan] * len(parameter.Fitness_function.maximise)

        else:
            eval_ind = True

            # cache part
            if parameter.Cache and ind.phenotype in cache:
                # The individual has been encountered before in
                # the utilities.trackers.cache.

                if parameter.Lookup_fitness:
                    # Set the fitness as the previous fitness from the
                    # cache.
                    ind.fitness = cache[ind.phenotype]
                    eval_ind = False

                elif parameter.Lookup_bad_fitness:
                    # Give the individual a bad default fitness.
                    if not isinstance(parameter.Fitness_function.maximise, list):
                        ind.fitness = np.nan
                    else:
                        ind.fitness = [np.nan] * len(parameter.Fitness_function.maximise)
                    eval_ind = False

                elif parameter.Mutate_duplicates:
                    # Mutate the individual to produce a new phenotype
                    # which has not been encountered yet.
                    while (not ind.phenotype) or ind.phenotype in cache:
                        ind = mutation([ind])[0]

                    # Need to overwrite the current individual in the pop.
                    individuals[name] = ind
                    ind.name = name

            if eval_ind:
                # Check for duplicate fitness in cache when initializing
                if initial and parameter.Cache:
                    # Build set of existing fitness values BEFORE evaluating current individual
                    existing_fitness = {
                        tuple(fit_val) if isinstance(fit_val, list) else fit_val
                        for fit_val in cache.values()
                    }
                    
                    # Evaluate current individual
                    results = eval_or_append(ind, results)
                    
                    # Convert current fitness to comparable format
                    current_fit = tuple(ind.fitness) if isinstance(ind.fitness, list) else ind.fitness
                    
                    # Check if fitness is valid (not NaN)
                    if isinstance(ind.fitness, list):
                        is_valid = not any([np.isnan(f) for f in ind.fitness])
                    else:
                        is_valid = not np.isnan(ind.fitness)
                    
                    # Mutate until unique and valid fitness is found
                    while (not is_valid) or (current_fit in existing_fitness):
                        ind = mutation([ind])[0]
                        results = eval_or_append(ind, results)
                        individuals[name] = ind
                        ind.name = name
                        
                        # Update current fitness
                        current_fit = tuple(ind.fitness) if isinstance(ind.fitness, list) else ind.fitness
                        if isinstance(ind.fitness, list):
                            is_valid = not any([np.isnan(f) for f in ind.fitness])
                        else:
                            is_valid = not np.isnan(ind.fitness)
                else:
                    # Normal evaluation without uniqueness check
                    results = eval_or_append(ind, results)

    return individuals
def eval_or_append(ind, results):

    ind.evaluate()

    # cache
    if (isinstance(ind.fitness, list) and not
    any([np.isnan(i) for i in ind.fitness])) or \
            (not isinstance(ind.fitness, list) and not
            np.isnan(ind.fitness)):
        # All fitnesses are valid.
        cache[ind.phenotype] = ind.fitness