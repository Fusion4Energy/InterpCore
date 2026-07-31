from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING
from interpcore.errors import ConfigurationError
from pathlib import Path

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
    AVERAGE_WEIGHTED = "Weighted Average"
    CLOSEST = "Closest"


# If true, the method tends to distribute each source point to the destination mesh
# If false, the method loops on each dest point and assigns a value depeding on neighbouring source points
DEST_SRC_MAP = {
    INTERPOLATION_KERNEL.DISTANCE_WEIGHTED: True,
    INTERPOLATION_KERNEL.FEM: True,
    INTERPOLATION_KERNEL.AVERAGE: False,
    INTERPOLATION_KERNEL.CLOSEST: False,
    INTERPOLATION_KERNEL.AVERAGE_WEIGHTED: False,
}


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
                raise ConfigurationError("Parameter for K query must be an integer.")

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
            raise ConfigurationError(f"Unsupported load type: {self.interpolated_load}")

        # ensure that kernels are not used in incompatible ways with the load type
        if not _is_compatible(self.kernel, self.interpolated_load):
            raise ConfigurationError(
                f"Incompatible kernel {self.kernel} for load type {self.interpolated_load}"
            )


@dataclass
class ColumnsConfig:
    """Describe the indices of the columns of both the source file(s) and of the
    destination mesh file. Indices are 0-based, i.e., the first column has index 0.

    Parameters
    ----------
    source_xyz : int
        Index of the first column containing the x coordinate of the source mesh.
        y, and z coordinates are expected to be in the next two columns.
    value : int
        Index of the column containing the values to be interpolated from the source mesh.
    dest_xyz : int
        Index of the first column containing the x coordinate of the destination mesh.
        y, and z coordinates are expected to be in the next two columns.
    id : int
        Index of the column containing the destination mesh node IDs.
    volume_area : int | None, optional
        Index of the column containing the volume or area of the destination mesh.
    """

    source_xyz: int
    value: int
    dest_xyz: int
    id: int
    volume_area: int | None = None


def _is_compatible(
    kernel: INTERPOLATION_KERNEL, load_type: INTERPOLATED_LOAD_TYPE
) -> bool:
    if DEST_SRC_MAP[kernel] and load_type != INTERPOLATED_LOAD_TYPE.EM_FORCE:
        return False
    if not DEST_SRC_MAP[kernel] and load_type == INTERPOLATED_LOAD_TYPE.EM_FORCE:
        return False
    return True
