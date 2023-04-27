'''
Construa um algoritmo genético para encontrar o máximo da função f6 descrita nas notas de aula.

- Implemente o gráfico da média das melhores soluções por experimento.
- Realize um total de 20 experimentos, com tamanho da população de 100 cromossomas por geração e número de gerações igual a 40.
- Utilize as mesmas taxas de crossover e mutação descritas nas notas de aula, assim como tamanho de cromossoma. Use aptidão igual a avaliação.
- Use aptidão igual a Windowing
- Use aptidão igual a normalização linear (1 a 100).
- Utilize Elitismo.
- Use steady state com e sem duplicados.
- Para a implementação do gráfico, o eixo y, deverá mostrar o número de 9 encontrados após a casa decimal e o eixo x o número de gerações. Compare os 3 métodos de aptidão, sem elitismo.

Compare os resultados encontrados com normalização linear com elitismo,
com os resultados encontrados com normalização linear com elitismo e steady state sem duplicadas e com os
resultados encontrados com normalização linear com elitismo e steady state sem duplicadas.

Entregue um relatório com o que foi implementado e com todos os resultados encontrados.
Realize uma conclusão crítica de todos os resultados encontrados. 
'''
# /bin/python "/home/tomast1337/dados-1tb/python ga/test.py"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Chromosome = np.ndarray
Population = list[Chromosome]

MIN_VALUE = -100
MAX_VALUE = 100

MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
POPULATION_SIZE = 100
GENERATION_COUNT = 40
CHROMOSOME_SIZE = 44

# Aliases
CHR_SIZE = CHROMOSOME_SIZE
GENE_SIZE = CHROMOSOME_SIZE // 2
POP_SIZE = POPULATION_SIZE


def f6(x: tuple[float, float]) -> float:
    return 0.5 - ((np.sin(np.sqrt(x[0] ** 2 + x[1] ** 2)))
                  ** 2 - 0.5) / ((1 + 0.001 * (x[0] ** 2 + x[1] ** 2)) ** 2)


def clamp_value(individual: Chromosome) -> tuple[float, float]:
    x, y = individual
    x = MIN_VALUE + (MAX_VALUE - MIN_VALUE) * x / (2**GENE_SIZE - 1)
    y = MIN_VALUE + (MAX_VALUE - MIN_VALUE) * y / (2**GENE_SIZE - 1)
    return x, y


def fitness(individual: Chromosome) -> float:
    x, y = clamp_value(individual)
    return f6((x, y))


def select_best_individual(population: Population, count: int) -> Chromosome:
    """
    Seleciona o indivíduo com maior aptidão (fitness) presente
    na população de entrada.
    """

    sorted_population = sorted(population, key=fitness)
    return sorted_population[-count:]


def slip_to_bits(chromosome: Chromosome) -> tuple[np.array, np.array]:
    """
    Converte um cromossomo para sua representação binária,
    retornando duas sequências de bits (uma para cada gene).
    """
    x_bits = np.binary_repr(chromosome[0], width=GENE_SIZE)
    y_bits = np.binary_repr(chromosome[1], width=GENE_SIZE)
    x_bits = np.array([int(i) for i in x_bits])
    y_bits = np.array([int(i) for i in y_bits])
    return x_bits, y_bits


def mutation(individual: Chromosome) -> Chromosome:
    """ Recebe um indivíduo de entrada e aplica uma mutação aleatória nos bits do cromossoma."""
    if np.random.uniform(0, 1) > MUTATION_RATE:
        return individual
    x_bits, y_bits = slip_to_bits(individual)
    random_index_x = np.random.randint(0, GENE_SIZE)
    random_index_y = np.random.randint(0, GENE_SIZE)
    x_bits[random_index_x] = 0 if x_bits[random_index_x] else 1
    y_bits[random_index_y] = 0 if y_bits[random_index_y] else 1
    x = int("".join(str(i) for i in x_bits), 2)
    y = int("".join(str(i) for i in y_bits), 2)
    return x, y


def crossover(a: Chromosome, b: Chromosome) -> tuple[Chromosome, Chromosome]:
    """
    Recebe 2 indivíduos de entrada e aplica o crossover nos bits no cromossoma de cada indivíduo.
    """
    if np.random.uniform(0, 1) > CROSSOVER_RATE:
        return a, b
    a_x_bits, a_y_bits = slip_to_bits(a)
    b_x_bits, b_y_bits = slip_to_bits(b)
    split_point = np.random.randint(1, GENE_SIZE - 1)
    a_x_bits[split_point:], b_x_bits[split_point:] = b_x_bits[split_point:], a_x_bits[split_point:]
    a_y_bits[split_point:], b_y_bits[split_point:] = b_y_bits[split_point:], a_y_bits[split_point:]

    a_x = int("".join(str(i) for i in a_x_bits), 2)
    a_y = int("".join(str(i) for i in a_y_bits), 2)
    b_x = int("".join(str(i) for i in b_x_bits), 2)
    b_y = int("".join(str(i) for i in b_y_bits), 2)

    return (a_x, a_y), (b_x, b_y)


def generate_random_population(size: int) -> Population:
    """Gera uma população de tamanho `size` com indivíduos aleatórios."""

    bits = np.random.randint(0, 2, size=(size, CHR_SIZE))
    len(bits)
    population = []
    for bit_x_y in bits:
        x = int("".join(str(i) for i in bit_x_y[:GENE_SIZE]), 2)
        y = int("".join(str(i) for i in bit_x_y[GENE_SIZE:]), 2)
        population.append(np.array([x, y], dtype=int))
    population = np.array(population)
    return population


def print_gen_info(population: Population, generation: int):
    df_final_population = pd.DataFrame(population)
    # rename columns to x and y
    df_final_population.rename(columns={0: 'x', 1: 'y'}, inplace=True)
    # for each row, clamp x and y values
    df_final_population['x'] = df_final_population['x'].apply(
        lambda x: MIN_VALUE + (MAX_VALUE - MIN_VALUE) * x / (2**GENE_SIZE - 1))

    df_final_population['y'] = df_final_population['y'].apply(
        lambda x: MIN_VALUE + (MAX_VALUE - MIN_VALUE) * x / (2**GENE_SIZE - 1))

    # create fitness column
    df_final_population['fitness'] = df_final_population.apply(fitness, axis=1)

    # sort by fitness
    df_final_population.sort_values(
        by=['fitness'], inplace=True, ascending=False)

    return df_final_population


def next_generation(population: Population) -> Population:
    """Recebe uma população de entrada e retorna a próxima geração, aplicando
    os operadores de crossover e mutação."""
    # Elitism
    bests = select_best_individual(population, 5)
    print(bests)
    # Crossover
    random_indices = list(range(POP_SIZE))
    np.random.shuffle(random_indices)
    for i in range(0, POP_SIZE, 2):
        i_a, i_b = random_indices[i], random_indices[i + 1]
        a = population[i_a]
        b = population[i_b]
        population[i_a], population[i_b] = crossover(a, b)

    # Mutation
    for i in range(POP_SIZE):
        population[i] = mutation(population[i])

    # Reinsere o melhor indivíduo
    population[0] = bests[0]
    population[1] = bests[1]
    population[2] = bests[2]
    population[3] = bests[3]
    population[4] = bests[4]

    return population


population = generate_random_population(POP_SIZE)

fitness_history = []
for gen in range(1, GENERATION_COUNT+1):
    print(f'Generation {gen}')
    population = next_generation(population)
    population_df = print_gen_info(population, gen)
    fitness_history.append(population_df['fitness'].max())

plt.plot(fitness_history)
plt.show()


population_df

# scatter plot
plt.plot(population_df['x'], population_df['y'], 'o')


plt.show()

real_fitness = np.array([fitness([x, y])
                        for x, y in zip(population_df['x'], population_df['y'])])

plt.plot(real_fitness)
plt.show()

# box plot
population_df.plot.box()

print(
    population_df.describe())
print(population_df.head())
