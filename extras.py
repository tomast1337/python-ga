
def mutation(individual: Chromosome) -> Population:
    """ Recebe um indivíduo de entrada e aplica uma mutação aleatória."""
    random_value = np.random.uniform(0, 1)
    if random_value <= mutation_rate:
        return individual + np.random.uniform(-1, 1, chromosome_size)
    return individual


def crossover(individual_a: Chromosome, individual_b: Chromosome) -> tuple[Chromosome, Chromosome]:
    """Recebe 2 indivíduos a e b e realiza uma troca de genes entre os indivíduos."""
    random_value = np.random.uniform(0, 1)
    if random_value <= crossover_rate:
        novo_a = individual_a.copy()
        novo_b = individual_b.copy()
        novo_a[0], novo_b[0] = novo_b[0], novo_a[0]
        return novo_a, novo_b
    return individual_a, individual_b
