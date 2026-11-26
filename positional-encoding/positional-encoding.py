
import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """

    if seq_len < 1 or d_model < 1:
        raise ValueError("seq_len and d_model must be >= 1")

    pos = np.arange(seq_len)
    i = np.arange(0, d_model, 2)

    denominator_term = 1.0 / (base ** (i / d_model))

    angles = pos[:, np.newaxis] * denominator_term[np.newaxis, :]

    pe = np.zeros((seq_len, d_model), dtype=float)

    pe[:, 0::2] = np.sin(angles)

    pe[:, 1::2] = np.cos(angles[:, :d_model // 2])

    return pe
