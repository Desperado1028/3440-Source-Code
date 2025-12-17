import random
import time
import matplotlib.pyplot as plt
from prim_opt import prim_optimized

def prim_unoptimized(n, matrix):
    visited = [False] * n
    key = [float('inf')] * n
    key[0] = 0
    mst_weight = 0

    for _ in range(n):
        u = -1
        for i in range(n):
            if not visited[i] and (u == -1 or key[i] < key[u]):
                u = i
        visited[u] = True
        mst_weight += key[u]

        for v in range(n):
            if matrix[u][v] != 0 and not visited[v] and matrix[u][v] < key[v]:
                key[v] = matrix[u][v]

    return mst_weight

def generate_sparse_graph_both(n, extra_edges=0, max_weight=100):
    graph_list = {i: [] for i in range(n)}
    matrix = [[0] * n for _ in range(n)]

    # spanning tree
    for i in range(1, n):
        parent = random.randint(0, i - 1)
        w = random.randint(1, max_weight)
        graph_list[parent].append((w, i))
        graph_list[i].append((w, parent))
        matrix[parent][i] = w
        matrix[i][parent] = w

    # extra edges
    for _ in range(extra_edges):
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u != v:
            w = random.randint(1, max_weight)
            graph_list[u].append((w, v))
            graph_list[v].append((w, u))
            matrix[u][v] = w
            matrix[v][u] = w

    return graph_list, matrix

def measure_time(func, *args):
    start = time.perf_counter()
    func(*args)
    return time.perf_counter() - start

sizes = [100, 200, 300, 400, 500]  # graph sizes
times_unopt = []
times_opt = []

for n in sizes:
    graph_list, matrix = generate_sparse_graph_both(n, extra_edges=10)

    # measure unoptimized
    t1 = measure_time(prim_unoptimized, n, matrix)
    times_unopt.append(t1)

    # measure optimized
    t2 = measure_time(prim_optimized, n, graph_list)
    times_opt.append(t2)

    print(f"n={n}: unoptimized={t1:.4f}s, optimized={t2:.4f}s")


plt.figure(figsize=(8, 5))
plt.plot(sizes, times_unopt, marker='o', label='Prim Unoptimized (Adj Matrix)')
plt.plot(sizes, times_opt, marker='o', label='Prim Optimized (Adj List + Heap)')

plt.xlabel("Number of Vertices (n)")
plt.ylabel("Execution Time (seconds)")
plt.title("Performance Comparison: Prim Unoptimized vs Optimized")
plt.legend()
plt.grid(True)
plt.show()