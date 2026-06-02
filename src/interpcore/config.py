from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from interpcore.kernels import INTERPOLATION_KERNEL


class QUERY_TYPE(Enum):
    """Methods to query neighbors in the mechanical mesh."""

    RADIUS = "Radius"
    K = "K-Nearest Neighbors"


class INTERPOLATED_LOAD_TYPE(Enum):
    """Types of loads that can be interpolated."""

    EM_FORCE = "EM-Force"
    HEAT_FLUX = "Heat Flux"
    HEAT_GEN = "Heat Generation"
    HTC = "Heat Transfer Coefficient"


class INTERPOLATION_KERNEL(Enum):
    """Kernels for interpolation, aka, how to distribute EM force on mech nodes."""

    DISTANCE_WEIGHTED = "Weighted by distance"
    FEM = "FEM system"
    AVERAGE = "Average"
    CLOSEST = "Closest"


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
    interpolated_load: INTERPOLATED_LOAD_TYPE
        Type of load to interpolate.
    accept_no_neighbor : bool, optional
        Whether to accept points with no neighbors within max_distance.
        If False, an error will be raised. If true, interpolated value will
        be set to zero for these points. By default False.

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
    interpolated_load: INTERPOLATED_LOAD_TYPE
    accept_no_neighbor: bool = False

    def __post_init__(self):
        if self.method == QUERY_TYPE.K:
            try:
                self.param = int(self.param)
            except (TypeError, ValueError):
                raise ValueError("Parameter for K query must be an integer.")

        # assign the correct number of components depeding on the load type
        if self.interpolated_load == INTERPOLATED_LOAD_TYPE.EM_FORCE:
            self.num_components = 3
        elif self.interpolated_load == INTERPOLATED_LOAD_TYPE.HEAT_FLUX:
            self.num_components = 1
        elif self.interpolated_load == INTERPOLATED_LOAD_TYPE.HEAT_GEN:
            self.num_components = 1
        elif self.interpolated_load == INTERPOLATED_LOAD_TYPE.HTC:
            self.num_components = 2
        else:
            raise ValueError(f"Unsupported load type: {self.interpolated_load}")
