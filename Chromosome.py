from dataclasses import dataclass
from typing import Protocol, TypedDict, Union, Callable, Optional, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


Bit = Union["0", "1"]

class PopulationParams(TypedDict):
    mutation_rate: float
    crossover_rate: float
    size: int
    elitism_number: int
    steady_state: bool
    windowing: bool
    duplicate_elitism: bool


GENE_SIZE = 22
CHROMOSOME_SIZE = 44


def f6(x: float, y: float) -> float:
    return 0.5 - ((np.sin(np.sqrt(x**2 + y**2))) ** 2 - 0.5) / (
        (1 + 0.001 * (x**2 + y**2)) ** 2
    )


@dataclass
class Chromosome:
    genes: List[Bit]
    size: int = CHROMOSOME_SIZE

    def __init__(self, genes: List[Bit], size: int = CHROMOSOME_SIZE):
        super().__init__()
        if len(genes) != size:
            raise ValueError(
                f"Chromosome must have at least {size} gene. {len(genes)}")
        if not all([gene in ["0", "1"] for gene in genes]):
            raise ValueError("Chromosome genes must be represented by 0 or 1.")
        self.genes = genes
        self.size = size

    @classmethod
    def from_bit_str(cls, bits: str):
        """
        Gera um cromossomo a partir de uma string contendo caracteres 0 e 1.
        """
        return cls([bit for bit in bits])

    @classmethod
    def random(cls, size: int = CHROMOSOME_SIZE):
        """
        Gera um cromossomo aleatório.
        """
        return cls([str(np.random.randint(0, 2)) for _ in range(size)], size)

    def mutate(self) -> None:
        """
        Realiza a operação de mutação no cromossomo.
        """
        index = np.random.randint(0, len(self.genes))
        self.genes[index] = "0" if self.genes[index] == "1" else "1"

    def crossover(self, other: "Chromosome", split_point: int = None) -> None:
        """
        Realiza a operação de crossover entre este e outro cromossomo.
        Se fornecido, a divisão será feita em `split_point`; caso contrário,
        será gerado um ponto de corte aleatoriamente.
        """
        if split_point is None:
            split_point = np.random.randint(0, len(self.genes))

        temp = self.genes[split_point:]
        self.genes[split_point:] = other.genes[split_point:]
        other.genes[split_point:] = temp

    def copy(self) -> "Chromosome":
        """
        Retorna uma cópia do cromossomo.
        """
        return Chromosome(self.genes.copy(), self.size)

    def __repr__(self) -> str:
        return "".join(self.genes)

    def _to_int(self) -> Tuple[int, int]:
        """
        Converte a sequência de bits dos genes para um par de números inteiros.
        Usado na conversão para coordenadas.
        """
        x = int("".join(self.genes[:GENE_SIZE]), 2)
        y = int("".join(self.genes[GENE_SIZE:]), 2)
        return x, y

    def to_coords(self, max: float = 100, min: float = -100) -> Tuple[float, float]:
        """
        Usando para o fitness
        """
        x, y = self._to_int()
        return (
            x * (max - min) / (2**GENE_SIZE - 1) + min,
            y * (max - min) / (2**GENE_SIZE - 1) + min,
        )

    @property
    def fitness(self, func: Callable[[float, float], float] = f6) -> float:
        """
        Calcula a aptidão do cromossomo de acordo com a função de avaliação `func`.
        """
        coords = self.to_coords()
        return func(*coords)


def random_call(chance: float, callback: Optional[Callable] = None) -> bool:
    """
    Faz um sorteio aleatório com probabilidade `chance`. De acordo com o resultado,
    chama ou não a função `callback`. Retorna o resultado do sorteio.
    """
    result = np.random.random() < chance
    if callback is not None and result:
        callback()
    return result


class Population:
    individuals: List[Chromosome]
    mutation_rate: float
    crossover_rate: float
    elitism_number: int
    steady_state: bool
    duplicate_selection: bool
    windowing: bool
    size: int
    history: pd.DataFrame

    def __init__(
        self,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        size: int = 100,
        elitism_number: int = 1,
        steady_state: bool = False,
        duplicate_selection: bool = False,
        windowing: bool = False,
    ):
        # history of best individuals
        self.history = pd.DataFrame(
            columns=[
                "generation",
                "best_fitness",
                "best_x",
                "best_y",
                "best_chromosome",
            ]
        )

        self.mutation_rate = mutation_rate  # mutation rate
        self.crossover_rate = crossover_rate  # crossover rate
        # number of best individuals to keep each generation
        self.elitism_number = elitism_number
        self.steady_state = steady_state  # use steady state instead of generational
        self.duplicate_selection = duplicate_selection  # allow duplicate elitism
        self.windowing = windowing  # use windowing instead of truncation
        self.size = size  # population size

        # generate random individuals
        self.individuals = []
        for _ in range(size):
            self.individuals.append(Chromosome.random())
        self.register_gen(0)

    def get_best_n(self, count: int) -> List[Chromosome]:
        """
        Retorna os `n` melhores indivíduos da população.
        """
        sorted_pop = sorted(self.individuals, key=lambda x: x.fitness, reverse=True)[
            :count
        ]

        # print([x.fitness for x in sorted_pop])
        # clone the best ones
        return [x.copy() for x in sorted_pop[:count]]

    def train(self, n_generations: int = 40):
        for i in range(0, n_generations):
            self.next_gen()
            self.n_generation = i
            self.register_gen(i + 1)

    def register_gen(self, i):
        data = {
            "generation": i,
            "best_fitness": self.get_best_n(1)[0].fitness,
            "best_x": self.get_best_n(1)[0].to_coords()[0],
            "best_y": self.get_best_n(1)[0].to_coords()[1],
            "best_chromosome": self.get_best_n(1)[0].__repr__(),
        }
        self.history = pd.concat(
            [self.history, pd.DataFrame(data, index=[0])], ignore_index=True
        )

    def _steady_state_selection(self):
        # based on the fitness of each individual, select the best ones
        # step 1: sort the population by fitness
        sorted_pop = sorted(
            self.individuals, key=lambda x: x.fitness, reverse=True)
        # step 2: do a non uniform random selection of the individuals, the best ones have a higher chance of being selected
        weights = [i.fitness for i in sorted_pop]
        
        # if windowing is enabled, normalize the weights push the higher weights to the top and lower weights to the bottom
        if self.windowing:
            max_weight = max(weights)
            min_weight = min(weights)
            weights = [(weight - min_weight) / (max_weight - min_weight) for weight in weights]

        # normalize the weights
        weights = [i/sum(weights) for i in weights]

        # step 3: select the individuals
        # p is the probability of each individual to be selected
        selected = np.random.choice(sorted_pop, size=self.size//4, p=weights, replace=self.duplicate_selection)
        
        # step 4: form the selected individuals, generate the new population
        self.individuals = []
        for _ in range(4):
            self.individuals.extend(selected)
        self.crossover_population()
        
        # step 5: mutate the new population
        self.mutate_population()

    def _non_steady_state_selection(self):
        self.crossover_population()
        self.mutate_population()

    def next_gen(self):
        the_elite = self.get_best_n(self.elitism_number)
        if self.steady_state:
            self._steady_state_selection()
        else:
            self._non_steady_state_selection()
        self.individuals = the_elite + \
            self.individuals[self.elitism_number:]

    def mutate_population(self):
        for individual in self.individuals:
            random_call(self.mutation_rate, callback=individual.mutate)

    def crossover_population(self):
        # shuffle individuals
        np.random.shuffle(self.individuals)
        # crossover
        for i in range(0, len(self.individuals), 2):

            def crossover():
                self.individuals[i].crossover(self.individuals[i + 1])

            random_call(self.crossover_rate, callback=crossover)


class ExperimentSet:
    n_experiments: int
    n_generations: int
    populations_param: PopulationParams
    fitness_func: Callable[[float, float], float] = f6

    def __init__(
        self,
        populations_param: List[dict],
        fitness_func: Callable[[float, float], float] = f6,
        n_experiments: int = 20,
        n_generations: int = 40,
    ):
        self.populations = []
        self.n_experiments = n_experiments
        self.populations_param = populations_param
        self.fitness_func = fitness_func
        self.n_experiments = n_experiments
        self.n_generations = n_generations

    def run(self):
        self.populations = [
            Population(**self.populations_param) for _ in range(self.n_experiments)
        ]
        print("=" * 25)
        print("POPULATION PARAMS")
        print("=" * 25)
        print("Mutation rate:         ",
              self.populations_param["mutation_rate"])
        print("Crossover rate:        ",
              self.populations_param["crossover_rate"])
        print("Population size:       ", self.populations_param["size"])
        print("Elitism number:        ",
              self.populations_param["elitism_number"])
        print("Steady state:          ",
              self.populations_param["steady_state"])
        print("Duplicate elitism:     ",
              self.populations_param["duplicate_elitism"])
        print("Fitness function:      ", self.fitness_func.__name__)
        print("Number of generations: ", self.n_generations)
        print("Number of experiments: ", self.n_experiments)
        print("=" * 25)

        for i in range(self.n_experiments):
            print(
                f"Running experiment {i + 1} of {self.n_experiments}, "
                f"best fitness: {self.populations[i].get_best_n(1)[0].fitness:.2f}",
            )
            self.populations[i].train(self.n_generations)

    def get_mean_history(self):
        df = pd.DataFrame(columns=["Generation", "Mean Fitness"])

        # Get best_fitness from each population in each generation
        historys = [population.history for population in self.populations]
        for gen in range(self.n_generations):
            data = {
                "Generation": gen,
                "Mean Fitness": np.mean(
                    [
                        historys[i].iloc[gen]["best_fitness"]
                        for i in range(len(historys))
                    ]
                ),
            }
            df = pd.concat([df, pd.DataFrame(data, index=[0])],
                           ignore_index=True)
        return df


def run_experiments():
    populations_param = {
        "mutation_rate": 0.1,
        "crossover_rate": 0.8,
        "size": 100,
        "elitism_number": 1,
        "steady_state": True,
        "duplicate_elitism": False,
    }
    ExperimentSet(
        populations_param=populations_param,
        fitness_func=f6,
    ).run()

def main():
    es = ExperimentSet(
        populations_param={
            "mutation_rate": 0.1,
            "crossover_rate": 0.8,
            "size": 100,
            "elitism_number": 1,
            "steady_state": True,
            "duplicate_elitism": False,
        },
        n_experiments=20,
        n_generations=40,
        fitness_func=f6,
        windowing=False,
    )

    es.run()
    x = es.get_mean_history()
    plt.plot(x["Generation"], x["Mean Fitness"])
    plt.show()


if __name__ == "__main__":
    main()
