from numpy import float64, inner
from numpy.linalg import norm
from numpy.typing import NDArray


def calculate_1d_cosine_similarity(a: NDArray[float64], b: NDArray[float64]) -> float:

    return inner(a, b) / (norm(a)*norm(b))