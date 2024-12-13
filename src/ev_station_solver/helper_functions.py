import base64

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial.distance import cdist, euclidean, pdist, squareform


def get_pdf(file_path):
    """
    Get the pdf file from the file path. Used to display the documentation in our web app.
    :param file_path:
    :return:
    """
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = (
        f'<iframe src="data:application/pdf;base64,{base64_pdf}#toolbar=0" '
        f'width="100%" height="1000" frameborder="0"></iframe>'
    )
    return pdf_display


def get_distance_matrix(
    locations_1: np.ndarray, locations_2: np.ndarray | None = None, metric="euclidean", symmetric: bool = False
) -> np.ndarray:
    """
    Compute a distance matrix using scipy, with optional support for returning
    only the upper triangular part when a single dataset is provided.

    Parameters:
        locations_1 (array-like): First dataset, where each row is a data point.
        locations_2 (array-like, optional): Second dataset, where each row is a data point. If None,
                                      distances are computed within locations_1.
        metric (str): The distance metric to use (default is 'euclidean').
        symmetric (bool): Whether to return only the upper triangular part.
                                      This is valid only when locations_2 is None.

    Returns:
        np.ndarray: A 2D distance matrix.
    """
    if symmetric:
        if locations_2 is not None:
            raise ValueError("symmetric is valid only when locations_2 is not provided.")

        # Compute pairwise distances within data1
        distances = pdist(locations_1, metric=metric)

        # Convert to a square-form distance matrix
        return squareform(distances)
    else:
        # Compute pairwise distances between data1 and data2
        return cdist(locations_1, locations_2, metric=metric)


def geometric_median(X, eps=1e-5):
    # source: https://stackoverflow.com/questions/30299267/geometric-median-of-multidimensional-points
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros / r
            y1 = max(0, 1 - rinv) * T + min(1, rinv) * y

        if euclidean(y, y1) < eps:
            return y1

        y = y1


def compute_maximum_matching(w: np.ndarray, queue_size: int, reachable: np.ndarray):
    w = w.astype(int)  # convert to int
    graph = np.repeat(reachable, queue_size * w, axis=1)
    result = maximum_bipartite_matching(csr_matrix(graph), perm_type="column")

    return np.mean(result >= 0)
