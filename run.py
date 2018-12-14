from numpy import matrix
from algorithm import *

if __name__ != 'main':
    pass


with open('input.txt', 'r') as f:
    products, factories = map(lambda x: int(x), f.readline().split(' '))

    def read_ints():
        return list(map(int, next(f).split(' ')))

    def read_matrix():
        return matrix([read_ints() for _ in range(products)])

    costs = read_matrix() + read_matrix()
    price, number = matrix(read_ints()), matrix(read_ints())

graph = matrix(np.multiply(
    matrix(np.repeat(price, factories, axis=0)).T - costs,
    matrix(np.repeat(number, factories, axis=0)).T))
print(graph)

matching = extended_hungarian_algorithm(graph, log=True)
expected = brute_force(graph)
print('Matching:', matching)
print('Weight:', sum(graph[i, j] for i, j in matching))
print('Expected:', expected)
