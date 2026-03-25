
import numpy as np
from mapper import Mapper
import parameter
class Individual():

    def __init__(self, genome, ind_tree, map_ind=True):

        if map_ind:
            # The individual needs to be mapped from the given input
            # parameters.
            self.phenotype, self.genome, self.tree, self.nodes, self.invalid, \
                self.depth, self.used_codons = Mapper(genome, ind_tree)

        else:
            # The individual does not need to be mapped.
            self.genome, self.tree = genome, ind_tree

        self.fitness_function = parameter.Fitness_function
        if not isinstance(self.fitness_function.maximise, list):
            self.fitness = np.nan
        else:
            self.fitness = [np.nan] * len(self.fitness_function.maximise)

        self.runtime_error = False
        self.name = None

    def __lt__(self, other):
        if np.isnan(self.fitness):
            return True
        elif np.isnan(other.fitness):
            return False
        else:
            return self.fitness < other.fitness if self.fitness_function.maximise else other.fitness < self.fitness

    def __le__(self, other):
        if np.isnan(self.fitness):
            return True
        elif np.isnan(other.fitness):
            return False
        else:
            return self.fitness <= other.fitness if self.fitness_function.maximise else other.fitness <= self.fitness

    def __str__(self):
        return ("Individual: " +
                str(self.phenotype) + "; " + str(self.fitness))

    def deep_copy(self):
        """
        Copy an individual and return a unique version of that individual.

        :return: A unique copy of the individual.
        """
        new_tree = self.tree.__copy__()

        # Create a copy of self by initialising a new individual.
        new_ind = Individual(self.genome.copy(), new_tree, map_ind=False)

        # Set new individual parameters (no need to map genome to new
        # individual).
        new_ind.phenotype, new_ind.invalid = self.phenotype, self.invalid
        new_ind.depth, new_ind.nodes = self.depth, self.nodes
        new_ind.used_codons = self.used_codons
        new_ind.runtime_error = self.runtime_error

        return new_ind

    def evaluate(self):
        if self.phenotype is None:
            if not isinstance(self.fitness_function.maximise, list):
                self.fitness = np.nan
            else:
                self.fitness = [np.nan] * len(self.fitness_function.maximise)
        else:
            self.fitness = self.fitness_function(self.phenotype)