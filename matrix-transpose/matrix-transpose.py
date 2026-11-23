import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    rows, col = np.shape(A)
    transpose = np.zeros((col, rows))

    for i in range (rows):
        for j in range(col):
            transpose[j, i] = A[i][j]

    return transpose
    
    pass
