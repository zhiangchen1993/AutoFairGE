import time
import numpy as np
from sys import stdout
from os import getpid, getcwd, path, makedirs
from socket import gethostname
import csv
import parameter
import inspect
from NSGA2 import compute_pareto_metrics
import shutil

def generate_folders_and_files(Experiment_name, time_stamp):
    path_1 = path.join(getcwd(), "results")
    if not path.isdir(path_1):
        # Create results folder.
        makedirs(path_1, exist_ok=True)

    file_path = path.join(path_1, Experiment_name)

    if not path.isdir(file_path):
        makedirs(file_path, exist_ok=True)

    file_path1 = path.join(file_path, time_stamp)
    if not path.isdir(file_path1):
        makedirs(file_path1, exist_ok=True)

def save_params_to_file(experiment_name, time_stamp):

    path_0 = path.join(getcwd(), "results", experiment_name)
    path_1 = path.join(path_0, time_stamp)

    file_path = path.join(path_1, "parameters.txt")

    with open(file_path, "w") as f:
        for name, value in inspect.getmembers(parameter):
            if not name.startswith("__") and not inspect.ismodule(value) and not inspect.isfunction(value):
                try:
                    f.write(f"{name} = {repr(value)}\n")
                except Exception as e:
                    f.write(f"{name} = <unserializable: {e}>\n")


def get_stats(individuals, gen=0, start=None, generation_number=None, experiment_name=None, time_stamp=None):

    # MOO
    if isinstance(parameter.Fitness_function.maximise, list):

        used_time = time.time() - start

        available = [i for i in individuals if not i.invalid and not any([np.isnan(fit) for fit in i.fitness])]
        pareto = compute_pareto_metrics(available)
        first_front_size = len(pareto.fronts[0])

        # generate file
        if gen == 0:
            print("\nStart:\t", start, "\n")

            generate_folders_and_files(experiment_name, time_stamp)
            save_params_to_file(experiment_name, time_stamp)

            # save the grammar
            path_0 = path.join(getcwd(), "results", experiment_name)
            path_1 = path.join(path_0, time_stamp)
            current_bnf_file = parameter.Grammar_filename if isinstance(parameter.Grammar_filename, str) else "test.bnf"
            src_bnf_path = path.join(getcwd(), current_bnf_file)
            dst_bnf_path = path.join(path_1, "grammar.bnf")
            if path.isfile(src_bnf_path):
                shutil.copy(src_bnf_path, dst_bnf_path)

        path_0 = path.join(getcwd(), "results", experiment_name)
        path_1 = path.join(path_0, time_stamp)

        # save the best results
        best_fitness_per_objective = []
        best_fitness_inds = []

        for m in range(parameter.Fitness_function.num_obj):
            sorted_pop = sorted(available, key=lambda ind: ind.fitness[m],
                                reverse=parameter.Fitness_function.maximise[m])

            best_fitness_per_objective.append(sorted_pop[0].fitness[m])
            best_fitness_inds.append(sorted_pop[0].phenotype)

        multi_result_path = path.join(path_1, "results.csv")

        with (open(multi_result_path, "a", newline='') as csvfile):
            writer = csv.writer(csvfile)

            if gen == 0:
                header = ["Generation"] + [f"Best_Fitness_Obj_{m}" for m in range(parameter.Fitness_function.num_obj)] + [f"Best_Phenotype_Obj_{m}" for m in range(parameter.Fitness_function.num_obj)] + ['Used_time']
                writer.writerow(header)

            writer.writerow([gen] + best_fitness_per_objective + best_fitness_inds + [used_time])

        # save all individuals in one generation
        generation_path = path.join(path_1, "generation.csv")
        with open(generation_path, "a", newline='') as genfile:
            writer = csv.writer(genfile)

            if gen == 0:
                writer.writerow(["Generation", "Phenotype", "Fitness"])

            for idx, individual in enumerate(individuals):
                writer.writerow([gen, individual.phenotype, individual.fitness])

        # save the pareto front in one generation
        pareto_front = path.join(path_1, "pareto_front.csv")
        with open(pareto_front, "a", newline='') as paretofile:
            writer = csv.writer(paretofile)

            if gen == 0:
                writer.writerow(["Generation", "Phenotype", "Fitness"])

            for idx, individual in enumerate(pareto.fronts[0]):
                writer.writerow([gen, individual.phenotype, individual.fitness])

        # print states
        print("=" * 50)
        perc = gen / (generation_number + 1) * 100
        print(f"Evolution: {perc}% complete")
        print(f"Generation {gen}:")
        print("=" * 50)

        # Save the testing results
        if gen == generation_number:
            test_results = path.join(path_1, "test_results.csv")
            with open(test_results, "a", newline='') as testresults:
                writer = csv.writer(testresults)
                writer.writerow(['Time_stamp', "Generation", "Phenotype", 'Val_Fitness', "Test_Fitness"])
                for idx, individual in enumerate(pareto.fronts[0]):
                    test_fitness = parameter.Fitness_function.evaluate_on_test(individual.phenotype)
                    writer.writerow([time_stamp, gen, individual.phenotype, individual.fitness, test_fitness])

            print("\n" + "=" * 50)
            print("TEST SET EVALUATION COMPLETED (Multi-Objective Optimization)")
            print(f"Evaluated {len(pareto.fronts[0])} individuals from Pareto front")
            print(f"Test results saved to: test_results.csv")
            print("=" * 50)
    # SOO
    else:
        # generate results
        available = [i for i in individuals if not i.invalid and not any([np.isnan(fit) for fit in i.fitness])]

        best = max(available)
        best_genotype = best.genome[:best.used_codons]
        best_phenotype = best.phenotype
        best_fitness = best.fitness

        fitnesses = [i.fitness for i in available]
        ave_fitness = np.nanmean(fitnesses, axis=0)
        std_dev = np.nanstd(fitnesses)

        used_time = time.time() - start

        # generate file
        if gen == 0:
            print("\nStart:\t", start, "\n")

            generate_folders_and_files(experiment_name, time_stamp)
            save_params_to_file(experiment_name, time_stamp)

            # save the grammar
            path_0 = path.join(getcwd(), "results", experiment_name)
            path_1 = path.join(path_0, time_stamp)
            src_bnf_path = path.join(getcwd(), "test.bnf")
            dst_bnf_path = path.join(path_1, "grammar.bnf")
            if path.isfile(src_bnf_path):
                shutil.copy(src_bnf_path, dst_bnf_path)

        path_0 = path.join(getcwd(), "results", experiment_name)
        path_1 = path.join(path_0, time_stamp)

        # save the results
        result_path = path.join(path_1, "result.csv")

        with open(result_path, "a", newline='') as resultfile:
            writer = csv.writer(resultfile)

            if gen == 0:
                writer.writerow(
                    ["Generation", "Best_Genotype", "Best_Phenotype", "Best_Fitness", "Average_Fitness", "Std_Fitness", "Used_time"])

            writer.writerow([gen, best_genotype, best_phenotype, best_fitness, ave_fitness, std_dev, used_time])

        # save all individuals in one generation
        generation_path = path.join(path_1, "generation.csv")

        with open(generation_path, "a", newline='') as genfile:
            writer = csv.writer(genfile)

            if gen == 0:
                writer.writerow(["Generation", "Phenotype", "Fitness"])

            for idx, individual in enumerate(individuals):
                writer.writerow([gen, individual.phenotype, individual.fitness])

        # Save the testing results
        if gen == generation_number:
            test_results = path.join(path_1, "test_results.csv")
            with open(test_results, "w", newline='') as testresults:
                writer = csv.writer(testresults)
                writer.writerow(['Time_stamp', "Generation", "Phenotype", 'Val_Fitness', "Test_Fitness"])

                test_fitness = parameter.Fitness_function.evaluate_on_test(best_phenotype)
                writer.writerow([time_stamp, gen, best_phenotype, best_fitness, test_fitness])

            print("\n" + "=" * 50)
            print("TEST SET EVALUATION COMPLETED (Single-Objective Optimization)")
            print("Best Individual Performance:")
            print(f"  Phenotype      : {best_phenotype}")
            print(f"  Validation Set : {best_fitness}")
            print(f"  Test Set       : {test_fitness}")
            print(f"Test results saved to: test_results.csv")
            print("=" * 50)

        # print states
        print("=" * 50)
        perc = gen / (generation_number + 1) * 100
        print(f"Evolution: {perc}% complete")
        print(f"Generation {gen}:")
        print(f"  Best Genotype   : {best_genotype}")
        print(f"  Best Phenotype  : {best_phenotype}")
        print(f"  Best Fitness    : {best_fitness}")
        print("=" * 50)
