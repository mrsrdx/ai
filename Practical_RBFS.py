import math

h = {
    'A': 10,
    'B': 8,
    'C': 5,
    'D': 7,
    'E': 3,
    'G': 0
}

graph = {
    'A': {'B': 2, 'C': 4},
    'B': {'D': 7, 'E': 3},
    'C': {'E': 1},
    'D': {'G': 1},
    'E': {'G': 5},
    'G': {}
}

def rbfs(node, goal, g, limit, path):

    if node == goal:
        return path, g, g + h[node]

    children = []

    for n, cost in graph[node].items():
        if n not in path:
            new_g = g + cost
            children.append([new_g + h[n], n, new_g])

    if not children:
        return None, math.inf, math.inf

    while True:
        children.sort()

        best = children[0]

        if best[0] > limit:
            return None, math.inf, best[0]

        alt = children[1][0] if len(children) > 1 else math.inf

        result, cost, best[0] = rbfs(
            best[1],
            goal,
            best[2],
            min(limit, alt),
            path + [best[1]]
        )

        if result:
            return result, cost, best[0]


path, cost, _ = rbfs('A', 'G', 0, math.inf, ['A'])

print("Path:", path)
print("Cost:", cost)
