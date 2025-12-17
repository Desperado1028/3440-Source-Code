import heapq
import random

def generate_sparse(n, extra_edges=0, max_weight=100):
    """

    :param n: number of vertices
    :param extra_edges:
    :param max_weight:
    :return: adjacent list
    """
    graph = {i: [] for i in range(n)}

    # 1. generate a tree which is connected
    for i in range(1, n):
        parent = random.randint(0, i - 1)
        weight = random.randint(1, max_weight)

        graph[parent].append((weight, i))
        graph[i].append((weight, parent))

    # 2. add extra edges
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

def generate_dense(n, max_weight=100):
    graph = {i: [] for i in range(n)}

    for u in range(n):
        for v in range(u + 1, n):
            weight = random.randint(1, max_weight)
            graph[u].append((weight, v))
            graph[v].append((weight, u))

    return graph

def prim_optimized(n, graph):
    """
        Prim's optimization algorithm using binary heap and adjacent list
    :param n: number of vertices
    :param graph: adjacent list
    :return: weight and MST
    """
    visited = [False] * n
    min_heap = []
    mst_edges = []
    mst_weight = 0

    visited[0] = True

    for weight, v in graph[0]:
        heapq.heappush(min_heap, (weight, 0, v))

    while min_heap and len(mst_edges) < n - 1:
        weight, u, v = heapq.heappop(min_heap)  # the edge with the smallest weight

        if visited[v]:
            continue

        visited[v] = True
        mst_edges.append((u, v, weight))
        mst_weight += weight

        for w, neighbor in graph[v]:
            if not visited[neighbor]:
                heapq.heappush(min_heap, (w, v, neighbor))

    return mst_weight, mst_edges

if __name__ == '__main__':
    n = 100
    graph_sparse = generate_sparse(n, extra_edges=20)
    graph_dense = generate_dense(n)

    mst_weight_sparse, mst_edges_sparse = prim_optimized(n, graph_sparse)
    mst_weight_dense, mst_edges_dense = prim_optimized(n, graph_dense)


    print("Sparse Graph: MST Weight =", mst_weight_sparse)
    print("Sparse Graph: MST Edges =", mst_edges_sparse)  # -> (u, v, weight)

    print("Dense Graph: MST Weight =", mst_weight_dense)
    print("Dense Graph: MST Edges =", mst_edges_dense)  # -> (u, v, weight)