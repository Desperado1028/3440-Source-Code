import random
import time
import matplotlib.pyplot as plt
import heapq
import math
from collections import deque

def generate_sparse_graph(n, extra_edges=0, max_weight=100):
    graph = {i: [] for i in range(n)}
    for i in range(1, n):
        parent = random.randint(0, i - 1)
        weight = random.randint(1, max_weight)
        graph[parent].append((weight, i))
        graph[i].append((weight, parent))
    added = 0
    while added < extra_edges:
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u != v:
            weight = random.randint(1, max_weight)
            graph[u].append((weight, v))
            graph[v].append((weight, u))
            added += 1
    return graph

def generate_dense_graph(n, max_weight=100):
    graph = {i: [] for i in range(n)}
    for u in range(n):
        for v in range(u + 1, n):
            weight = random.randint(1, max_weight)
            graph[u].append((weight, v))
            graph[v].append((weight, u))
    return graph

def graph_to_edge_list(graph):
    edges = []
    for u, neighbors in graph.items():
        for weight, v in neighbors:
            if u < v:
                edges.append((weight, u, v))
    return edges

def kruskal_baseline(n, graph):
    edges = graph_to_edge_list(graph)
    edges.sort(key=lambda x: x[0])
    mst_edges = []
    mst_weight = 0
    visited = [False] * n
    for weight, u, v in edges:
        if visited[u] and visited[v]:
            continue
        if not is_connected(graph, u, v, visited):
            visited[u] = True
            visited[v] = True
            mst_edges.append((u, v, weight))
            mst_weight += weight
        if len(mst_edges) == n - 1:
            break
    return mst_weight

def is_connected(graph, u, v, visited):
    visited_temp = [False] * len(visited)
    stack = [u]
    visited_temp[u] = True
    while stack:
        node = stack.pop()
        if node == v:
            return True
        for _, neighbor in graph[node]:
            if not visited_temp[neighbor]:
                visited_temp[neighbor] = True
                stack.append(neighbor)
    return False

def find(parent, i):
    if parent[i] != i:
        parent[i] = find(parent, parent[i])
    return parent[i]

def union(parent, rank, x, y):
    xroot = find(parent, x)
    yroot = find(parent, y)
    if xroot == yroot:
        return False
    if rank[xroot] < rank[yroot]:
        parent[xroot] = yroot
    elif rank[xroot] > rank[yroot]:
        parent[yroot] = xroot
    else:
        parent[yroot] = xroot
        rank[xroot] += 1
    return True

def kruskal_optimized(n, graph):
    edges = graph_to_edge_list(graph)
    edges.sort(key=lambda x: x[0])
    parent = list(range(n))
    rank = [0] * n
    mst_edges = []
    mst_weight = 0
    for weight, u, v in edges:
        if union(parent, rank, u, v):
            mst_edges.append((u, v, weight))
            mst_weight += weight
            if len(mst_edges) == n - 1:
                break
    return mst_weight

def measure_time(func, *args):
    start = time.perf_counter()
    func(*args)
    return time.perf_counter() - start

def compare_kruskal_performance():
    sizes = [ 100, 150, 200, 250,300,350,400,450,500]
    extra_edges_sparse = 20
    max_weight = 100
    times_baseline_sparse = []
    times_optimized_sparse = []
    times_baseline_dense = []
    times_optimized_dense = []
    print("Starting performance test...")
    for n in sizes:
        print(f"\nTesting sparse graph (n={n}, extra_edges={extra_edges_sparse})")
        graph_sparse = generate_sparse_graph(n, extra_edges=extra_edges_sparse, max_weight=max_weight)
        t_baseline = measure_time(kruskal_baseline, n, graph_sparse)
        times_baseline_sparse.append(t_baseline)
        print(f"  Baseline implementation: {t_baseline:.4f}s")
        t_optimized = measure_time(kruskal_optimized, n, graph_sparse)
        times_optimized_sparse.append(t_optimized)
        print(f"  Optimized implementation: {t_optimized:.4f}s")
    for n in sizes:
        print(f"\nTesting dense graph (n={n})")
        graph_dense = generate_dense_graph(n, max_weight=max_weight)
        t_baseline = measure_time(kruskal_baseline, n, graph_dense)
        times_baseline_dense.append(t_baseline)
        print(f"  Baseline implementation: {t_baseline:.4f}s")
        t_optimized = measure_time(kruskal_optimized, n, graph_dense)
        times_optimized_dense.append(t_optimized)
        print(f"  Optimized implementation: {t_optimized:.4f}s")
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 1, 1)
    plt.plot(sizes, times_baseline_sparse, 'o-', label='Baseline (DFS/BFS)')
    plt.plot(sizes, times_optimized_sparse, 's-', label='Optimized (Union-Find)')
    plt.xlabel('Number of Vertices (n)')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Performance Comparison: Kruskal Baseline vs Optimized (Sparse Graphs)')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(sizes, times_baseline_dense, 'o-', label='Baseline (DFS/BFS)')
    plt.plot(sizes, times_optimized_dense, 's-', label='Optimized (Union-Find)')
    plt.xlabel('Number of Vertices (n)')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Performance Comparison: Kruskal Baseline vs Optimized (Dense Graphs)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('kruskal_performance_comparison.png', dpi=300)
    plt.show()
    print("\nSpeedup analysis:")
    for i, n in enumerate(sizes):
        speedup_sparse = times_baseline_sparse[i] / times_optimized_sparse[i]
        speedup_dense = times_baseline_dense[i] / times_optimized_dense[i]
        print(f"n={n}: Sparse Graph Speedup = {speedup_sparse:.2f}x, Dense Graph Speedup = {speedup_dense:.2f}x")

if __name__ == "__main__":
    compare_kruskal_performance()