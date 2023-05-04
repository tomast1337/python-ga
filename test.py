
from Chromosome import *


def test():
    print("Hello World")
    # Test mutation
    a = Chromosome.from_bit_str("00000000000000000000000000000000000000000000")
    b = Chromosome.from_bit_str("11111111111111111111111111111111111111111111")
    print("a:", str(a))
    a.mutate()
    print("a:", str(a), "mutated")
    print("b:", str(b))
    b.mutate()
    print("b:", str(b), "mutated")

    # Test crossover
    c = Chromosome.from_bit_str("01010101010101010101010101010101010101010101")
    d = Chromosome.from_bit_str("11111111111111111111111111111111111111111111")
    print("c:", str(c))
    print("d:", str(d))
    c.crossover(d)
    print("c:", str(c), "crossovered")
    print("d:", str(d), "crossovered")

    # Test to_coords
    print("a coords:", a.to_coords())
    print("b coords:", b.to_coords())
    print("c coords:", c.to_coords())
    print("d coords:", d.to_coords())

    # Test fitness
    print("a fitness:", a.fitness)
    print("b fitness:", b.fitness)
    print("c fitness:", c.fitness)
    print("d fitness:", d.fitness)

def test_pop():
    # population
    population = Population(
        size=100,
        elitism_number=10,
    )
    population.train(40)
    population_history = population.history
    print(population.history)

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        population_history["best_x"],
        population_history["best_y"],
        population_history["best_fitness"],
        c="r",
        marker="x",
    )

    # add f6(x,y) surface
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = f6(X, Y)
    ax.plot_surface(X, Y, Z, alpha=0.3)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f6(x,y)")

    plt.show()

    # fitness history
    plt.plot(population_history["generation"],
             population_history["best_fitness"])
    plt.show()

