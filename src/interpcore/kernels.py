import numpy as np
from interpcore.config import InterpolationConfig
from sklearn.metrics.pairwise import euclidean_distances
import logging
from enum import Enum


class INTERPOLATION_KERNEL(Enum):
    """Kernels for interpolation, aka, how to distribute EM force on mech nodes."""

    DISTANCE_WEIGHTED = "Weighted by distance"
    FEM = "FEM system"
    AVERAGE = "Average"
    CLOSEST = "Closest"


# If true, the method tends to distribute each source point to the destination mesh
# If false, the method loops on each dest point and assigns a value depeding on neighbouring source points
DEST_SRC_MAP = {
    INTERPOLATION_KERNEL.DISTANCE_WEIGHTED: True,
    INTERPOLATION_KERNEL.FEM: True,
    INTERPOLATION_KERNEL.AVERAGE: False,
    INTERPOLATION_KERNEL.CLOSEST: False,
}


# --- interpolation kernels ---
def _FEM_interpolation_kernel(
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


def _dist_weight_kernel(
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


def _closest_node_kernel(
    distances: np.ndarray,
    dest_idx: np.ndarray,
    src_index: int,
    vector_to_interp: np.ndarray,
    vector_interpolated: np.ndarray,
) -> bool:
    """Assign the closest src value to each dest node"""
    closest_idx = dest_idx[np.argmin(distances)]
    vector_interpolated[closest_idx, :] += vector_to_interp[src_index]
    return True


def interpolate_block(
    chunked_coords: np.ndarray,
    neighbours_coords: np.ndarray,
    idx_query: np.ndarray,
    chunk_idx: slice,
    src_values: np.ndarray,
    config: InterpolationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a block of EM nodes to Mech nodes"""
    interpolated = np.zeros([neighbours_coords.shape[0], 3])
    unmapped = np.zeros([1, 3])

    i = chunk_idx.start
    for point in chunked_coords[chunk_idx]:
        neighbours_idx = idx_query[i]
        mapped = False

        # Ensure query is not empty
        if neighbours_idx.shape[0] > 0:
            distances = euclidean_distances(
                neighbours_coords[neighbours_idx], point.reshape(1, -1)
            )
            # clip the mech nodes based on max distance
            keep = distances.flatten() < config.max_distance
            distances = distances[keep]
            neighbours_idx = neighbours_idx[keep]

            # Ensure that after clipping something remains
            if neighbours_idx.shape[0] == 0:
                # TODO: this makes sense only when source are distributed and
                # not viceversa. If kernel does not find neighbours of a destination
                # point we should raise an error instead.
                unmapped = unmapped + src_values[i, :]
                i = i + 1
                continue

            # if coincident node found, assign directly
            if distances[0] < config.coincidence_tolerance:
                # TODO: also here be careful. neighbours idx may be src or dest
                # depending on the kernel.
                logging.debug(f"Coincident node found {chunked_coords[i, :]}")
                interpolated[neighbours_idx[0], :] = [
                    src_values[i, 0],
                    src_values[i, 1],
                    src_values[i, 2],
                ]
                i = i + 1
                continue

            # TODO: logic is not always summing, to be double checked.
            # If nothing above, perform the interpolation
            # This will adjourn the "interpolated" array
            if config.kernel == INTERPOLATION_KERNEL.DISTANCE_WEIGHTED:
                mapped = _dist_weight_kernel(
                    distances,
                    neighbours_idx,
                    i,
                    src_values,
                    interpolated,
                )
            elif config.kernel == INTERPOLATION_KERNEL.FEM:
                mapped = _FEM_interpolation_kernel(
                    distances,
                    neighbours_idx,
                    i,
                    neighbours_coords,
                    chunked_coords,
                    src_values,
                    interpolated,
                )
            else:
                raise ValueError(f"Unknown interpolation kernel {config.kernel}")
        if not mapped:
            unmapped = unmapped + src_values[i, :]
        i += 1
    return interpolated, unmapped
