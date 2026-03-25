from config_manager import initialize_config
import parameter
from initialization import initialization
from search_loop import step
from get_stats import get_stats

import cache

def main():
    # Initialise configurations
    initialize_config()

    # Initialise population
    individuals, start_time, experiment_name, time_stamp = initialization(parameter.Population)

    # Print review for initial population
    get_stats(individuals, generation_number=parameter.Generation, start=start_time, experiment_name=experiment_name, time_stamp=time_stamp)

    # GE evolution
    for generation in range(1, (parameter.Generation + 1)):

        cache.generation[0] = generation
        # New generation
        individuals = step(individuals)

        # Print review
        get_stats(individuals, gen=generation, start=start_time, generation_number=parameter.Generation, experiment_name=experiment_name, time_stamp=time_stamp)


if __name__ == "__main__":
    main()