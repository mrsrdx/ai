import queue as Q

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

start = 'A'
goal = 'G'

q = Q.PriorityQueue()
q.put((h[start], 0, start, [start]))

while not q.empty():
    f, g, node, path = q.get()

    if node == goal:
        print("Path:", path)
        print("Cost:", g)
        break

    for next_node, cost in graph[node].items():
        new_g = g + cost
        new_f = new_g + h[next_node]

        q.put((new_f, new_g, next_node, path + [next_node]))
