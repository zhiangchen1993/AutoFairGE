import parameter
from NSGA2 import compute_pareto_metrics
import numpy as np
from collections import defaultdict
import random
def replacement(new_pop, old_pop):
    replacement_method = replacement_methods(parameter.Replacement)

    return replacement_method(new_pop, old_pop)

def replacement_methods(method_name):
    methods = {
        "generational": generational,
        "nsga2_replacement": nsga2_replacement,
        "nsga2_lim_replacement": nsga2_lim_replacement
    }
    return methods.get(method_name, generational)

def generational(new_pop, old_pop):
    if hasattr(parameter, 'Elite_size'):
        Elite_size = parameter.Elite_size
    else:
        Elite_size = round(parameter.Population * 0.01)

    old_pop = [i for i in old_pop if not np.isnan(i.fitness)]
    new_pop = [i for i in old_pop if not np.isnan(i.fitness)]

    old_pop.sort(reverse=True)
    new_pop.sort(reverse=True)

    for ind in old_pop[:Elite_size]:
        new_pop.insert(0, ind)

    return new_pop[:parameter.Population]

def nsga2_replacement(new_pop, old_pop):
    """
        Replaces the old population with the new population using NSGA-II
        replacement. Both new and old populations are combined, pareto fronts
        and crowding distance are calculated, and the replacement population is
        computed based on crowding distance per pareto front.

        :param new_pop: The new population (e.g. after selection, variation, &
                        evaluation).
        :param old_pop: The previous generation population.
        :return: The 'POPULATION_SIZE' new population.
        """

    # Combine both populations (R_t = P_t union Q_t)
    new_pop.extend(old_pop)

    available = [i for i in new_pop if not i.invalid and not any([np.isnan(fit) for fit in i.fitness])]

    # Compute the pareto fronts and crowding distance
    pareto = compute_pareto_metrics(available)

    # Size of the new population
    pop_size = parameter.Population

    # New population to replace the last one
    temp_pop, i = [], 0

    while len(temp_pop) < pop_size:
        # Populate the replacement population

        if len(pareto.fronts[i]) <= pop_size - len(temp_pop):
            temp_pop.extend(pareto.fronts[i])

        else:
            # Sort the current pareto front with respect to crowding distance.
            pareto.fronts[i] = sorted(pareto.fronts[i],
                                      key=lambda item:
                                      pareto.crowding_distance[item],
                                      reverse=True)

            # Get number of individuals to add in temp to achieve the pop_size
            diff_size = pop_size - len(temp_pop)

            # Extend the replacement population
            temp_pop.extend(pareto.fronts[i][:diff_size])

        # Increment counter.
        i += 1

    return temp_pop

def nsga2_lim_replacement(new_pop, old_pop):
    """
        Replaces the old population with the new population using NSGA-II
        replacement. Both new and old populations are combined, pareto fronts
        and crowding distance are calculated, and the replacement population is
        computed based on crowding distance per pareto front.

        :param new_pop: The new population (e.g. after selection, variation, &
                        evaluation).
        :param old_pop: The previous generation population.
        :return: The 'POPULATION_SIZE' new population.
        """

    # Get limit from parameter
    limit = parameter.Max_dup_fitness

    # Combine both populations (R_t = P_t union Q_t)
    new_pop.extend(old_pop)

    available = [i for i in new_pop if not i.invalid and not any([np.isnan(fit) for fit in i.fitness])]

    fitness_groups = defaultdict(list)

    for ind in available:
        key = tuple(ind.fitness)
        group = fitness_groups[key]
        n = len(group)
        if n < limit:
            group.append(ind)
        else:
            j = random.randint(0, n)
            if j < limit:
                group[j] = ind

    # Extract final results
    final_available = [ind for group in fitness_groups.values() for ind in group]

    # Compute the pareto fronts and crowding distance
    pareto = compute_pareto_metrics(final_available)

    # Size of the new population
    pop_size = parameter.Population

    # New population to replace the last one
    temp_pop, i = [], 0

    while len(temp_pop) < pop_size and i < len(pareto.fronts):
        # Populate the replacement population

        if len(pareto.fronts[i]) <= pop_size - len(temp_pop):
            temp_pop.extend(pareto.fronts[i])

        else:
            # Sort the current pareto front with respect to crowding distance.
            pareto.fronts[i] = sorted(pareto.fronts[i],
                                      key=lambda item:
                                      pareto.crowding_distance[item],
                                      reverse=True)

            # Get number of individuals to add in temp to achieve the pop_size
            diff_size = pop_size - len(temp_pop)

            # Extend the replacement population
            temp_pop.extend(pareto.fronts[i][:diff_size])

        # Increment counter.
        i += 1

    return temp_pop

