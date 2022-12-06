import numpy as np


class Samples:
    chain_graph = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], dtype=np.float64)

    triangle_graph = np.array([[0, 1, 0, 0], [1, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0]], dtype=np.float64)

    full_graph = np.array(
        [
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        ],
        dtype=np.float64,
    )

    tree_matrix = np.array(
        [
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )

    diploma_matrix = np.array(
        [
            [0, 1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [1, 1, 0, 1, 1, 0],
            [0, 0, 1, 0, 1, 1],
            [0, 0, 1, 1, 0, 1],
            [0, 0, 0, 1, 1, 0],
        ],
        dtype=np.float64,
    )

    big_chain = np.zeros((100, 100))
    for i in range(100):
        if i + 1 < 100:
            big_chain[i][i + 1] = 1
        if i - 1 >= 0:
            big_chain[i][i - 1] = 1

    weighted = np.array(
        [[0, 3, 2, 0, 4], [3, 0, 8, 0, 0], [2, 8, 0, 1, 0], [0, 0, 1, 0, 3], [4, 0, 0, 3, 0]], dtype=np.float64
    )

    weighted_sp = np.array([[0, 3, 2, 3, 4], [3, 0, 5, 6, 7], [2, 5, 0, 1, 4], [3, 6, 1, 0, 3], [4, 7, 4, 3, 0]])

    all = {
        "chain_graph": chain_graph,
        "triangle_graph": triangle_graph,
        "full_graph": full_graph,
        "tree_matrix": tree_matrix,
        "diploma_matrix": diploma_matrix,
        "big_chain": big_chain,
        "weighted": weighted,
        "weighted_sp": weighted_sp,
    }

    def __getitem__(self, item):
        return self.all[item]
