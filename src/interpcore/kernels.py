import numpy as np
from interpcore.config import InterpolationConfig, INTERPOLATION_KERNEL
from sklearn.metrics.pairwise import euclidean_distances
import logging
from interpcore.errors import InterpolationError


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


def _average_kernel(
    neighbours_idx: np.ndarray,
    dest_index: int,
    vector_to_interp: np.ndarray,
    vector_interpolated: np.ndarray,
) -> bool:
    """Average the values from neighboring source points (dest-to-source mode)"""
    # Average all neighboring source values
    avg_value = np.mean(vector_to_interp[neighbours_idx, :], axis=0)
    vector_interpolated[dest_index, :] = avg_value
    return True


def _closest_source_kernel(
    distances: np.ndarray,
    neighbours_idx: np.ndarray,
    dest_index: int,
    vector_to_interp: np.ndarray,
    vector_interpolated: np.ndarray,
) -> bool:
    """Assign value from the closest source point (dest-to-source mode)"""
    closest_src_idx = neighbours_idx[np.argmin(distances)]
    vector_interpolated[dest_index, :] = vector_to_interp[closest_src_idx, :]
    return True


def interpolate_block(
    chunked_coords: np.ndarray,
    neighbours_coords: np.ndarray,
    idx_query: np.ndarray,
    chunk_idx: slice,
    src_values: np.ndarray,
    config: InterpolationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a block of points

    This function handles two different philosophies:
    1. Source-to-Destination (DEST_SRC_MAP[kernel] = True):
       - chunked_coords are source points, neighbours_coords are destination points
       - For each source point, distribute its value to neighboring destination points
       - Used by: DISTANCE_WEIGHTED, FEM

    2. Destination-to-Source (DEST_SRC_MAP[kernel] = False):
       - chunked_coords are destination points, neighbours_coords are source points
       - For each destination point, compute value from neighboring source points
       - Used by: AVERAGE, CLOSEST
    """
    is_src_to_dest = DEST_SRC_MAP[config.kernel]

    if is_src_to_dest:
        # Initialize output for destination points
        interpolated = np.zeros([neighbours_coords.shape[0], config.num_components])
    else:
        # Initialize output for destination points (same size as chunked_coords)
        interpolated = np.zeros([chunked_coords.shape[0], config.num_components])

    unmapped = np.zeros([1, config.num_components])

    i = chunk_idx.start
    for point in chunked_coords[chunk_idx]:
        neighbours_idx = np.asarray(idx_query[i], dtype=int)
        mapped = False

        # Ensure query is not empty
        if neighbours_idx.shape[0] > 0:
            distances = euclidean_distances(
                neighbours_coords[neighbours_idx], point.reshape(1, -1)
            )
            # clip the nodes based on max distance
            keep = distances.flatten() < config.max_distance
            distances = distances[keep]
            neighbours_idx = neighbours_idx[keep]

            # Ensure that after clipping something remains
            if neighbours_idx.shape[0] == 0:
                if is_src_to_dest:
                    # Source point has no destination neighbors - accumulate as unmapped
                    unmapped = unmapped + src_values[i, :]
                else:
                    # Destination point has no source neighbors - this is an error condition
                    raise InterpolationError(
                        f"Destination point at {point} has no source neighbors within max_distance"
                    )
                i = i + 1
                continue

            # if coincident node found, assign directly
            if distances[0] < config.coincidence_tolerance:
                logging.debug(f"Coincident node found {point}")
                if is_src_to_dest:
                    # Source point coincides with destination point
                    # neighbours_idx[0] is the destination index
                    interpolated[neighbours_idx[0], :] = src_values[i, :]
                else:
                    # Destination point coincides with source point
                    # neighbours_idx[0] is the source index, i is relative to chunk
                    dest_idx = i - chunk_idx.start
                    interpolated[dest_idx, :] = src_values[neighbours_idx[0], :]
                i = i + 1
                continue

            # Perform the interpolation based on kernel and philosophy
            if is_src_to_dest:
                index = i
            else:
                index = i - chunk_idx.start

            # Source-to-Destination: distribute source value to destination neighbors
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

            # Destination-to-Source: compute destination value from source neighbors
            if config.kernel == INTERPOLATION_KERNEL.AVERAGE:
                mapped = _average_kernel(
                    neighbours_idx,
                    index,
                    src_values,
                    interpolated,
                )
            elif config.kernel == INTERPOLATION_KERNEL.CLOSEST:
                mapped = _closest_source_kernel(
                    distances,
                    neighbours_idx,
                    index,
                    src_values,
                    interpolated,
                )

            if not mapped:
                # Should be possible only in source-to-destination
                unmapped = unmapped + src_values[i, :]

        else:
            # No neighbors found at all
            if is_src_to_dest:
                unmapped = unmapped + src_values[i, :]
            else:
                raise InterpolationError(
                    f"Destination point at {point} has no neighbors"
                )

        i += 1

    return interpolated, unmapped
