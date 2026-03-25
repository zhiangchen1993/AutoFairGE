import numpy as np
from random import sample
import parameter
from NSGA2 import compute_pareto_metrics, \
    crowded_comparison_operator

def selection(individuals):
    """
    Perform selection on a population in order to select a population of
    individuals for variation.

    :param population: input population
    :return: selected population
    """
    selection_method = selection_methods(parameter.Selection)

    return selection_method(individuals)

def selection_methods(method_name):
    methods = {
        "tournament": tournament,
        "nsga2_selection": nsga2_selection
    }
    return methods.get(method_name, tournament)

def tournament(individuals):
    """
    Given an entire population, draw <tournament_size> competitors randomly and
    return the best. Only valid individuals can be selected for tournaments.

    :param population: A population from which to select individuals.
    :return: A population of the winners from tournaments.
    """

    winners = []
    available = [i for i in individuals if not np.isnan(i.fitness)]

    while len(winners) < (parameter.Population - parameter.Tournament_size):
        competitors = sample(available, parameter.Tournament_size)
        winners.append(max(competitors))

    return winners

def nsga2_selection(population):
    """Apply NSGA-II selection operator on the *population*. Usually, the
        size of *population* will be larger than *k* because any individual
        present in *population* will appear in the returned list at most once.
        Having the size of *population* equals to *k* will have no effect other
        than sorting the population according to their front rank. The
        list returned contains references to the input *population*. For more
        details on the NSGA-II operator see [Deb2002]_.

        :param population: A population from which to select individuals.
        :returns: A list of selected individuals.
        .. [Deb2002] Deb, Pratab, Agarwal, and Meyarivan, "A fast elitist
           non-dominated sorting genetic algorithm for multi-objective
           optimization: NSGA-II", 2002.
        """

    selection_size = parameter.Population - parameter.Tournament_size + 1
    tournament_size = parameter.Tournament_size

    winners = []

    available = [i for i in population if not i.invalid and not any([np.isnan(fit) for fit in i.fitness])]

    pareto = compute_pareto_metrics(available)

    while len(winners) < selection_size:
        # Return the single best competitor.
        winners.append(pareto_tournament(available, pareto, tournament_size))

    return winners

def pareto_tournament(population, pareto, tournament_size):
    """
    The Pareto tournament selection uses both the pareto front of the
    individual and the crowding distance.

    :param population: A population from which to select individuals.
    :param pareto: The pareto front information.
    :param tournament_size: The size of the tournament.
    :return: The selected individuals.
    """

    # Initialise no best solution.
    best = None

    # Randomly sample *tournament_size* participants.
    participants = sample(population, tournament_size)

    for participant in participants:
        if best is None or crowded_comparison_operator(participant, best,
                                                       pareto):
            best = participant

    return best
