# Practical 1: Breadth First Search and Iterative Depth First Search

from collections import deque
import time

graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

def bfs(start, goal):
    visited = []
    queue = deque([start])

    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.append(node)
            if node == goal:
                return visited
            for neighbour in graph[node]:
                if neighbour not in visited:
                    queue.append(neighbour)
    return visited

def iterative_dfs(start, goal):
    visited = []
    stack = [start]

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.append(node)
            if node == goal:
                return visited
            for neighbour in reversed(graph[node]):
                if neighbour not in visited:
                    stack.append(neighbour)
    return visited

start, goal = 'A', 'F'

begin = time.time()
bfs_path = bfs(start, goal)
bfs_time = time.time() - begin

begin = time.time()
dfs_path = iterative_dfs(start, goal)
dfs_time = time.time() - begin

print("BFS Traversal:", bfs_path)
print("Iterative DFS Traversal:", dfs_path)
print("BFS Time:", bfs_time)
print("DFS Time:", dfs_time)
