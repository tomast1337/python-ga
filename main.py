from dataclasses import dataclass
import time
from typing import Protocol, TypedDict, Union, Callable, Optional, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Chromosome import *

SIZE_POP = 100  # size of population
N_GENS = 40  # number of generations
N_EXPS = 20

# set np seed
np.random.seed(42)  # 42 is the answer to everything


# time it decorator
def timeit(func, name=None):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(
            f"Time taken: {end - start} seconds for {name if name else func.__name__}"
        )
        return result

    return wrapper


@timeit
def run_experiments():
    @timeit
    def fitness_without_windowing():
        print("Running fitness_without_windowing")
        experiment = ExperimentSet(
            populations_param={
                "mutation_rate": 0.1,
                "crossover_rate": 0.8,
                "size": SIZE_POP,
                "elitism_number": 0,
                "steady_state": False,
                "duplicate_selection": False,
                "windowing": False,
            },
            n_experiments=N_EXPS,
            n_generations=N_GENS,
            fitness_func=f6,
        )
        experiment.run()
        mean_history = experiment.get_mean_history()
        return mean_history

    @timeit
    def fitness_windowing():
        print("Running fitness_windowing")
        experiment = ExperimentSet(
            populations_param={
                "mutation_rate": 0.1,
                "crossover_rate": 0.8,
                "size": SIZE_POP,
                "elitism_number": 0,
                "steady_state": False,
                "duplicate_selection": False,
                "windowing": True,
            },
            n_experiments=N_EXPS,
            n_generations=N_GENS,
            fitness_func=f6,
        )
        experiment.run()
        mean_history = experiment.get_mean_history()
        return mean_history

    @timeit
    def fitness_linear_scaling():
        print("Running fitness_linear_scaling")
        experiment = ExperimentSet(
            populations_param={
                "mutation_rate": 0.1,
                "crossover_rate": 0.8,
                "size": SIZE_POP,
                "elitism_number": 0,
                "steady_state": False,
                "duplicate_selection": False,
                "linear_scaling": True,
            },
            n_experiments=N_EXPS,
            n_generations=N_GENS,
            fitness_func=f6,
        )
        experiment.run()
        mean_history = experiment.get_mean_history()
        return mean_history

    @timeit
    def elitism_no_steady_state():
        print("Running elitism_no_steady_state")
        experiment = ExperimentSet(
            populations_param={
                "mutation_rate": 0.1,
                "crossover_rate": 0.8,
                "size": SIZE_POP,
                "elitism_number": 5,
                "steady_state": False,
                "duplicate_selection": False,
            },
            n_experiments=N_EXPS,
            n_generations=N_GENS,
            fitness_func=f6,
        )
        experiment.run()
        mean_history = experiment.get_mean_history()
        return mean_history

    @timeit
    def steady_state_without_dupes():
        print("Running steady_state_without_dupes")
        experiment = ExperimentSet(
            populations_param={
                "mutation_rate": 0.1,
                "crossover_rate": 0.8,
                "size": SIZE_POP,
                "elitism_number": 5,
                "steady_state": True,
                "duplicate_selection": False,
                "windowing": False,
            },
            n_experiments=N_EXPS,
            n_generations=N_GENS,
            fitness_func=f6,
        )
        experiment.run()
        mean_history = experiment.get_mean_history()
        return mean_history

    @timeit
    def steady_state_with_dupes():
        print("Running steady_state_with_dupes")
        experiment = ExperimentSet(
            populations_param={
                "mutation_rate": 0.1,
                "crossover_rate": 0.8,
                "size": SIZE_POP,
                "elitism_number": 5,
                "steady_state": True,
                "duplicate_selection": True,
                "windowing": False,
            },
            n_experiments=N_EXPS,
            n_generations=N_GENS,
            fitness_func=f6,
        )
        experiment.run()
        mean_history = experiment.get_mean_history()
        return mean_history

    return {
        "fitness_without_windowing": fitness_without_windowing(),
        "fitness_windowing": fitness_windowing(),
        "fitness_linear_scaling": fitness_linear_scaling(),
        "steady_state_with_dupes": steady_state_with_dupes(),
        "steady_state_without_dupes": steady_state_without_dupes(),
        "elitism_no_steady_state": elitism_no_steady_state(),
    }


def main():
    results = run_experiments()
    fitness_without_windowing = results["fitness_without_windowing"]
    fitness_windowing = results["fitness_windowing"]
    fitness_linear_scaling = results["fitness_linear_scaling"]
    steady_state_with_dupes = results["steady_state_with_dupes"]
    steady_state_without_dupes = results["steady_state_without_dupes"]
    elitism_no_steady_state = results["elitism_no_steady_state"]

    def nines_after_decimal(df, name):
        plt.plot(df["Generation"], df["9s after the decimal"], label=name)
        plt.legend()
        plt.xlabel("Generation")
        plt.ylabel("9s after the decimal")
        plt.title(name)
        plt.show()

    nines_after_decimal(fitness_without_windowing, "Fitness is The Evaluation")
    nines_after_decimal(fitness_windowing, "Fitness Windowing")
    nines_after_decimal(fitness_linear_scaling, "Fitness Linear Scaling")
    nines_after_decimal(elitism_no_steady_state, "Elitism Without Steady State")
    nines_after_decimal(
        steady_state_without_dupes, "Steady State Without Dupes With Elitism"
    )
    nines_after_decimal(steady_state_with_dupes, "Steady StateWith Dupes With Elitism")

    def plot_mean_fitness(df, name, plot):
        # mean_fitness
        plot.plot(df["Generation"], df["Mean Fitness"], label="mean of best fitness")
        plot.legend()
        # min_fitness
        plot.plot(df["Generation"], df["Min fitness"], label="mean of min fitness")
        plot.legend()
        # std_fitness
        plot.plot(df["Generation"], df["Mean of Means"], label="mean of means fitness")
        plot.legend()
        plot.set_xlabel("Generation")
        plot.set_ylabel("Mean Fitness")
        plot.set_title(name)

    # 1 by 3 plot
    figure, axis = plt.subplots(3, 1)
    # plot of elitism_no_steady_state steady_state_without_dupes steady_state_with_dupes mean fitness
    plot_mean_fitness(elitism_no_steady_state, "Elitism Without Steady State", axis[0])
    plot_mean_fitness(
        steady_state_without_dupes, "Steady State Without Dupes With Elitism", axis[1]
    )
    plot_mean_fitness(
        steady_state_with_dupes, "Steady StateWith Dupes With Elitism", axis[2]
    )

    plt.tight_layout()
    plt.show()

    # 1 by 3 plot
    figure, axis = plt.subplots(3, 1)

    plot_mean_fitness(fitness_without_windowing, "Fitness is The Evaluation", axis[0])
    plot_mean_fitness(fitness_windowing, "Fitness Windowing", axis[1])
    plot_mean_fitness(fitness_linear_scaling, "Fitness Linear Scaling", axis[2])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
