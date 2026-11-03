# Practical 2: A* Search and Recursive Best-First Search

import heapq
import math
import time

graph = {
    'Arad': {'Zerind': 75, 'Sibiu': 140, 'Timisoara': 118},
    'Zerind': {'Arad': 75, 'Oradea': 71},
    'Oradea': {'Zerind': 71, 'Sibiu': 151},
    'Sibiu': {'Arad': 140, 'Oradea': 151, 'Fagaras': 99, 'Rimnicu Vilcea': 80},
    'Timisoara': {'Arad': 118, 'Lugoj': 111},
    'Lugoj': {'Timisoara': 111, 'Mehadia': 70},
    'Mehadia': {'Lugoj': 70, 'Dobreta': 75},
    'Dobreta': {'Mehadia': 75, 'Craiova': 120},
    'Craiova': {'Dobreta': 120, 'Rimnicu Vilcea': 146, 'Pitesti': 138},
    'Rimnicu Vilcea': {'Sibiu': 80, 'Craiova': 146, 'Pitesti': 97},
    'Fagaras': {'Sibiu': 99, 'Bucharest': 211},
    'Pitesti': {'Rimnicu Vilcea': 97, 'Craiova': 138, 'Bucharest': 101},
    'Bucharest': {'Fagaras': 211, 'Pitesti': 101, 'Giurgiu': 90, 'Urziceni': 85},
    'Giurgiu': {'Bucharest': 90},
    'Urziceni': {'Bucharest': 85, 'Hirsova': 98, 'Vaslui': 142},
    'Hirsova': {'Urziceni': 98, 'Eforie': 86},
    'Eforie': {'Hirsova': 86},
    'Vaslui': {'Urziceni': 142, 'Iasi': 92},
    'Iasi': {'Vaslui': 92, 'Neamt': 87},
    'Neamt': {'Iasi': 87}
}

h = {
    'Arad': 366, 'Zerind': 374, 'Oradea': 380, 'Sibiu': 253,
    'Timisoara': 329, 'Lugoj': 244, 'Mehadia': 241, 'Dobreta': 242,
    'Craiova': 160, 'Rimnicu Vilcea': 193, 'Fagaras': 176,
    'Pitesti': 100, 'Bucharest': 0, 'Giurgiu': 77, 'Urziceni': 80,
    'Hirsova': 151, 'Eforie': 161, 'Vaslui': 199, 'Iasi': 226, 'Neamt': 234
}

def astar(start, goal):
    queue = [(h[start], 0, start, [start])]
    best_cost = {start: 0}

    while queue:
        f, cost, city, path = heapq.heappop(queue)

        if city == goal:
            return path, cost

        if cost > best_cost.get(city, math.inf):
            continue

        for neighbour, distance in graph[city].items():
            new_cost = cost + distance
            if new_cost < best_cost.get(neighbour, math.inf):
                best_cost[neighbour] = new_cost
                heapq.heappush(
                    queue,
                    (new_cost + h[neighbour], new_cost,
                     neighbour, path + [neighbour])
                )
    return None, math.inf

class Node:
    def __init__(self, city, parent=None, cost=0, f=None):
        self.city = city
        self.parent = parent
        self.cost = cost
        self.f = max(cost + h[city], parent.f if parent else 0) if f is None else f

def rbfs(node, goal, f_limit):
    if node.city == goal:
        return node, node.f

    successors = []
    for neighbour, distance in graph[node.city].items():
        if node.parent and neighbour == node.parent.city:
            continue
        child_cost = node.cost + distance
        child = Node(neighbour, node, child_cost)
        successors.append(child)

    if not successors:
        return None, math.inf

    while True:
        successors.sort(key=lambda item: item.f)
        best = successors[0]

        if best.f > f_limit:
            return None, best.f

        alternative = successors[1].f if len(successors) > 1 else math.inf
        result, best.f = rbfs(best, goal, min(f_limit, alternative))

        if result is not None:
            return result, best.f

def get_path(node):
    path = []
    while node:
        path.append(node.city)
        node = node.parent
    return path[::-1]

start, goal = 'Arad', 'Bucharest'

begin = time.time()
a_path, a_cost = astar(start, goal)
a_time = time.time() - begin

begin = time.time()
r_result, r_f = rbfs(Node(start), goal, math.inf)
r_time = time.time() - begin

print("A* Path:", " -> ".join(a_path))
print("A* Cost:", a_cost)
print("A* Time:", a_time)

if r_result:
    print("RBFS Path:", " -> ".join(get_path(r_result)))
    print("RBFS Cost:", r_result.cost)
else:
    print("RBFS: No path found")
print("RBFS Time:", r_time)
