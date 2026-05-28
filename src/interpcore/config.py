from dataclasses import dataclass
from enum import Enum


class QUERY_TYPE(Enum):
    """Methods to query neighbors in the mechanical mesh."""

    RADIUS = "Radius"
    K = "K-Nearest Neighbors"


class INTERPOLATION_KERNEL(Enum):
    """Kernels for interpolation, aka, how to distribute EM force on mech nodes."""

    DISTANCE_WEIGHTED = "Weighted by distance"
    FEM = "FEM system"


@dataclass
class InterpolationConfig:
    """Class containing all needed parameters to customize the interpolation

    Parameters
    ----------
    method : QUERY_TYPE
        Method to query neighbors in the mechanical mesh.
    param : int | float
        Parameter for the chosen query method. Radius in meters for RADIUS method,
        number of neighbors for K method.
    max_distance : float
        Maximum distance [m] to consider a mechanical node as neighbor.
    coincidence_tolerance : float
        Tolerance [m] to consider two nodes as coincident.
    kernel : INTERPOLATION_KERNEL
        Kernel for interpolation.
    multithread : bool
        Whether to use multithreading for interpolation.

    Raises
    ------
    ValueError
        if parameter for K method is not an integer.
    """

    method: QUERY_TYPE
    param: int | float
    max_distance: float
    coincidence_tolerance: float
    kernel: INTERPOLATION_KERNEL
    multithread: bool

    def __post_init__(self):
        if self.method == QUERY_TYPE.K:
            try:
                self.param = int(self.param)
            except (TypeError, ValueError):
                raise ValueError("Parameter for K query must be an integer.")
