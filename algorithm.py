import numpy as np
from queue import Queue


def reduce(m, axis=None):
    return m - np.repeat(m.min(axis), m.shape[axis], axis=axis)\
        if(axis is not None)\
        else reduce(reduce(m, axis=0), axis=1)


def egervary_operation(graph, x, y):
    d = graph[list(x), :][:, list(set(range(graph.shape[1])) - set(y))].min()
    return graph - \
        np.matrix(
            np.repeat(
                [[(d if i in x else 0) for i in range(graph.shape[0])]], graph.shape[1], axis=3)).T + \
        np.matrix(
            np.repeat(
                [[(d if i in y else 0) for i in range(graph.shape[1])]], graph.shape[0], axis=0))


def find_chain(graph, matching, start):
    parent = {}
    shift = graph.shape[0]
    x, y = {start}, set()
    current = Queue()
    old = set()
    while True:
        for i in x:
            current.put(i)
            old.add(i)
        while not current.empty():
            a = current.get()
            if a < shift:
                for b in range(graph.shape[0]):
                    if graph[a, b] == 0 and (a, b) not in matching and b + shift not in old:
                        current.put(b + shift)
                        parent[b + shift] = a
                        old.add(b + shift)
                        y.add(b)
            elif a - shift not in {edge[1] for edge in matching}:
                result = set()
                i = a
                while parent[i] in parent:
                    result.add((parent[i], i - shift))
                    i = parent[parent[i]]
                result.add((parent[i], i - shift))
                return graph, result
            else:
                for b in range(graph.shape[0]):
                    if graph[b, a - shift] == 0 and (b, a - shift) in matching and b not in old:
                        current.put(b)
                        parent[b] = a
                        old.add(b)
                        x.add(b)
        graph = egervary_operation(graph, x, y)


def hungarian_algorithm(graph):
    graph = reduce(graph)
    print('---------------')
    print(graph)
    matching = set()
    for start in range(graph.shape[0]):
        graph, chain = find_chain(graph, matching, start)
        matching = matching ^ chain
    return matching
