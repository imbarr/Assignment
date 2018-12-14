import numpy as np
from queue import Queue
import math
from itertools import permutations


def reduce(m, axis=None):
    return m - np.repeat(m.min(axis), m.shape[axis], axis=axis)\
        if(axis is not None)\
        else reduce(reduce(m, axis=0), axis=1)


def egervary_operation(graph, x, y):
    d = graph[list(x), :][:, list(set(range(graph.shape[1])) - set(y))].min()
    gsd = graph[list(x), :][:, list(set(range(graph.shape[1])) - set(y))]
    s = set(range(graph.shape[1])) - set(y)
    return graph - \
        np.matrix(
            np.repeat(
                [[(d if i in x else 0) for i in range(graph.shape[0])]], graph.shape[1], axis=0)).T + \
        np.matrix(
            np.repeat(
                [[(d if i in y else 0) for i in range(graph.shape[1])]], graph.shape[0], axis=0))


def find_chain(graph, matching, start):
    parent = {}
    shift = graph.shape[0]
    x, y = {start}, set()
    current = Queue()
    while True:
        for i in x:
            current.put(i)
        while not current.empty():
            a = current.get()
            if a < shift:
                for b in range(graph.shape[1]):
                    if graph[a, b] == 0 and b not in y and (a, b) not in matching:
                        current.put(b + shift)
                        parent[b + shift] = a
                        y.add(b)
            elif a - shift not in {edge[1] for edge in matching}:
                result = set()
                i = a
                while parent[i] in parent:
                    result.add((parent[i], i - shift))
                    result.add((parent[i], parent[parent[i]] - shift))
                    i = parent[parent[i]]
                result.add((parent[i], i - shift))
                return graph, result
            else:
                for b in range(graph.shape[0]):
                    if graph[b, a - shift] == 0 and b not in x and (b, a - shift) in matching:
                        current.put(b)
                        parent[b] = a
                        x.add(b)

        graph = egervary_operation(graph, x, y)


def hungarian_algorithm(graph, log=False):
    graph = reduce(graph)
    matching = set()
    for start in range(graph.shape[0]):
        graph, chain = find_chain(graph, matching, start)
        if log:
            print('Pogress:', start, '/', graph.shape[0])
        matching = matching ^ chain
    return matching


def extended_hungarian_algorithm(graph, log=False):
    n = graph.shape[0]
    for _ in range(graph.shape[1] - n):
        graph = np.append(graph, [[0 for _ in range(graph.shape[1])]], axis=0)
    matching = hungarian_algorithm(graph, log)
    return list(filter(lambda edge: edge[0] < n, matching))


def brute_force(graph):
    weight = None
    total = math.factorial(graph.shape[1])
    for i, combination in enumerate(permutations(range(graph.shape[1]))):
        print('Pogress:', i, '/', total)
        current = sum(graph[i, combination[i]] for i in range(graph.shape[0]))
        weight = current if weight is None else min(weight, current)
    return weight
