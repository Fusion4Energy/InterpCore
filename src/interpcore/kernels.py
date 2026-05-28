import numpy as np


# --- interpolation kernels ---
def FEM_interpolation_kernel(
    distances: np.ndarray,
    dest_idx: np.ndarray,
    src_idx: int,
    dest_coords: np.ndarray,
    src_coords: np.ndarray,
    vector_to_interp: np.ndarray,
    vector_interpolated: np.ndarray,
) -> bool:
    """FEM-based interpolation kernel"""
    roi = dest_coords[dest_idx] - src_coords[src_idx]
    vi = np.divide(roi, distances)

    if vi.shape[0] > 0:
        Nt = vi.shape[0]
        A = np.zeros([3, 3])
        for j in range(0, vi.shape[0]):
            dist = distances[j][0]
            l = vi[j, 0]
            m = vi[j, 1]
            n = vi[j, 2]
            A[0, 0] += (1.0 / dist) * l**2.0
            A[0, 1] += (1.0 / dist) * l * m
            A[0, 2] += (1.0 / dist) * l * n

            A[1, 0] += (1.0 / dist) * l * m
            A[1, 1] += (1.0 / dist) * m**2.0
            A[1, 2] += (1.0 / dist) * m * n
            A[2, 0] += (1.0 / dist) * l * n
            A[2, 1] += (1.0 / dist) * m * n
            A[2, 2] += (1.0 / dist) * n**2.0

        F = np.zeros([3, 1])
        F[0] = vector_to_interp[src_idx, 0]
        F[1] = vector_to_interp[src_idx, 1]
        F[2] = vector_to_interp[src_idx, 2]

        R = np.zeros([3 * vi.shape[0], 1])

        if abs(np.linalg.det(A)) > 1e-10:
            U = np.linalg.solve(A, F)

            for j in range(0, vi.shape[0]):
                dist = distances[j][0]
                l = vi[j, 0]
                m = vi[j, 1]
                n = vi[j, 2]
                A[0, 0] = (1.0 / dist) * l**2.0
                A[0, 1] = (1.0 / dist) * l * m
                A[0, 2] = (1.0 / dist) * l * n

                A[1, 0] = (1.0 / dist) * l * m
                A[1, 1] = (1.0 / dist) * m**2.0
                A[1, 2] = (1.0 / dist) * m * n
                A[2, 0] = (1.0 / dist) * l * n
                A[2, 1] = (1.0 / dist) * m * n
                A[2, 2] = (1.0 / dist) * n**2.0
                R[3 * j : 3 * j + 3] = np.dot(A, U)

        vector_interpolated[dest_idx, :] += R.reshape(Nt, 3)
    else:
        return False  # not possible to map
    return True


def dist_weight_kernel(
    distances: np.ndarray,
    dest_idx: np.ndarray,
    src_index: int,
    vector_to_interp: np.ndarray,
    vector_interpolated: np.ndarray,
) -> bool:
    """simply distribute the value components proportionally to the inverse of the
    distance"""
    max_distance = np.max(distances)
    weights = 1 / (distances / max_distance)  # inverse distance weights
    weights /= np.sum(weights)  # normalize the weights so that they sum to 1
    vector_interpolated[dest_idx, :] += weights * vector_to_interp[src_index]

    return True
