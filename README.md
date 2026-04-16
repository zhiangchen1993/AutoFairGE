# AutoFairGE

Fairness-aware AutoML framework based on Multi-Objective Grammatical Evolution

## Introduction

AutoFairGE is a fairness-aware AutoML framework that incorporates both accuracy and fairness from a multi-objective perspective, automatically exploring diverse model configurations and machine learning pipelines. It is built upon the Grammatical Evolution implementation of [PonyGE2](https://github.com/PonyGE/PonyGE2).

## Features

- **Fairness Definitions**: AutoFairGE flexibly supports arbitrary fairness definitions. The library currently provides widely used fairness metrics, including Average Odds Difference (AOD), Statistical Parity Difference (SPD), and Theil Index.

- **ML Pipeline**: AutoFairGE currently supports Combined Algorithm Selection and Hyperparameter Optimization (CASH); ensemble construction; preprocessing optimization; and traditional unfairness mitigation methods (in progress).

- **Search Space**: AutoFairGE uses grammars (BNF) to define user-specified search spaces. By adjusting the grammar, the search space can be focused on model configuration-level exploration or extended to full ML pipeline optimization.

- **GE Functionality**: The library provides two individual representations — integer genome and derivation tree — along with their corresponding GE operators. Variants targeting search efficiency (e.g., population diversity) are also incorporated.

## Requirements

- Python >= 3.11
- numpy
- pandas
- scikit-learn
- imbalanced-learn
- aif360 (for unfairness mitigation methods)

## Project Structure

```
AutoFairGE/
├── AutoFairGE.py          # Main entry point
├── parameter.py            # Default hyperparameters and configurations
├── config_manager.py       # Command-line argument parser
├── grammar/                # BNF grammar files defining search spaces
├── fitness/                # Fitness functions (accuracy + fairness metrics)
├── data/                   # Datasets
├── results/                # Experiment outputs
├── initialization.py       # Population initialization
├── search_loop.py          # Evolutionary search loop
├── NSGA2.py                # NSGA-II multi-objective optimization
├── selection.py            # Selection operators
├── crossover.py            # Crossover operators
├── mutation.py             # Mutation operators
├── replacement.py          # Replacement strategies
├── individual.py           # Individual representation
├── grammar.py              # Grammar parser
├── mapper.py               # Genotype-to-phenotype mapping
├── tree.py                 # Derivation tree representation
├── derivation.py           # Derivation tree operations
├── evaluate_fitness.py     # Fitness evaluation
├── get_stats.py            # Statistics and logging
└── cache.py                # Fitness caching
```

## Quick Start

Run with default settings:

```bash
python AutoFairGE.py
```

Or customize via command-line arguments:

```bash
python AutoFairGE.py \
    --Dataset_train data/Compas_race/compas_train_race.csv \
    --Dataset_val data/Compas_race/compas_val_race.csv \
    --Dataset_test data/Compas_race/compas_test_race.csv \
    --Random_seed 0 \
    --Experiment_name my_experiment \
    --Fitness_file fitness.fitness_acc_aod \
    --Grammar_file grammar/CASH.bnf
```

## Publications

- Chen, Z., Connor, M., Pant, S. and O'Neill, M., 2025, July. Investigating Combined Algorithm Selection and Hyperparameter Optimization for Fairness. In *Proceedings of the Genetic and Evolutionary Computation Conference Companion* (pp. 255-258).

- (In Review) Zhiang Chen, Mark Connor, Sudarshan Pant, and Michael O'Neill. AutoFairGE: Towards Fairness-Aware AutoML with Grammatical Evolution. *Genetic Programming and Evolvable Machines*.

- (In Review) Zhiang Chen, Mark Connor, and Michael O'Neill, 2026, August. Building Fairness-Aware Ensemble-based AutoML with AutoFairGE. *Parallel Problem Solving From Nature*.
