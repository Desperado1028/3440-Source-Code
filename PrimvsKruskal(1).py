from time import perf_counter
import matplotlib.pyplot as plt

from prim_opt import generate_sparse, generate_dense, prim_optimized
from KruskalCompare import kruskal_optimized


# ===== 单次公平对比：同一张图，Prim vs Kruskal =====
def compare_once(n, graph):
    #edges = graph_to_edges(graph)

    # Prim
    t1 = perf_counter()
    prim_optimized(n, graph)
    t2 = perf_counter()

    # Kruskal
    t3 = perf_counter()
    kruskal_optimized(n, graph)
    t4 = perf_counter()

    return t2 - t1, t4 - t3

#=====多次试验取平均值 =====
def compare_avg(n, graph_func, trials=20):
    prim_total = 0.0
    kruskal_total = 0.0

    for _ in range(trials):
        graph = graph_func(n)
        tp, tk = compare_once(n, graph)
        prim_total += tp
        kruskal_total += tk

    return prim_total / trials, kruskal_total / trials

# ===== 主实验 =====
if __name__ == "__main__":

    # ---------- Sparse graph ----------
    sizes_sparse = [200, 400, 600, 800, 1000]
    prim_sparse = []
    kruskal_sparse = []
    
    def gen_sparse(n):
        return generate_sparse(n, extra_edges=n)

    for n in sizes_sparse:
        print(f"Running sparse graph, n = {n}")
        tp, tk = compare_avg(n, gen_sparse, trials=5)
        prim_sparse.append(tp)
        kruskal_sparse.append(tk)
       

    plt.figure()
    plt.plot(sizes_sparse, prim_sparse, marker='o', label="Prim (optimized)")
    plt.plot(sizes_sparse, kruskal_sparse, marker='s', label="Kruskal (optimized)")
    plt.xlabel("Number of vertices (n)")
    plt.ylabel("Execution time (seconds)")
    plt.title("Prim vs Kruskal on Sparse Graphs")
    plt.legend()
    plt.grid(True)
    plt.show()


    # ---------- Dense graph ----------
    sizes_dense = [300, 350, 400, 450, 500]
    prim_dense = []
    kruskal_dense = []

    for n in sizes_dense:
        print(f"Running dense graph (avg), n = {n}")
        tp, tk = compare_avg(n, generate_dense, trials=5)

        prim_dense.append(tp)
        kruskal_dense.append(tk)

    plt.figure()
    plt.plot(sizes_dense, prim_dense, marker='o', label="Prim (optimized)")
    plt.plot(sizes_dense, kruskal_dense, marker='s', label="Kruskal (optimized)")
    plt.xlabel("Number of vertices (n)")
    plt.ylabel("Execution time (seconds)")
    plt.title("Prim vs Kruskal on Dense Graphs")
    plt.legend()
    plt.grid(True)
    plt.show()