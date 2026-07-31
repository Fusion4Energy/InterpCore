# InterpCore

A Python library for interpolating physical field data (electromagnetic forces, heat flux, etc.) between different mesh representations and exporting to ANSYS APDL format.

## Features

- **Multiple interpolation kernels**: Distance-weighted, FEM-based, K-nearest neighbors, closest point
- **Flexible query methods**: K-nearest neighbors or radius-based search
- **Support for multiple load types**: 
  - EM forces (3-component vector fields)
  - Heat flux (scalar fields)
  - Heat generation (volumetric)
  - Heat Transfer Coefficient + bulk fluid temperature (convection BCs)
- **Analysis tools**:
  - Scalar field integration for computing total heat generation, flux, etc.
  - EM force resultant computation for validating force/moment conservation
- **Export to ANSYS APDL**: Direct export of interpolated results in APDL format
- **Visualization**: Built-in VTK export for ParaView or PyVista visualization
- **Efficient**: KDTree-based spatial queries for fast neighbor searches

## Installation

```bash
pip install interpcore
```

## Quick Start

```python
from interpcore.interpolator import Interpolator
from interpcore.config import (
    InterpolationConfig,
    ColumnsConfig,
    QUERY_TYPE,
    INTERPOLATED_LOAD_TYPE,
)
from interpcore.kernels import INTERPOLATION_KERNEL

# Configure interpolation
config = InterpolationConfig(
    method=QUERY_TYPE.K,  # type of neighbour search
    param=5,  # parameter relative to the neighbour search (K or radius)
    max_distance=2.0,  # filter by a max radius of search (in case of K is used)
    coincidence_tolerance=0.01,  # tolerance to consider two nodes coincident
    kernel=INTERPOLATION_KERNEL.DISTANCE_WEIGHTED,  # How to interpolate
    multithread=False,  # use or not multithread
    interpolated_load=INTERPOLATED_LOAD_TYPE.EM_FORCE,  # type of load that is being interpolated
)

# Define column indices for the input files
columns = ColumnsConfig(id=0, dest_xyz=1, source_xyz=1, value=4)

# Create interpolator and run
interpolator = Interpolator(
    path_to_src_folder="source_data",
    path_to_dest_mesh="destination_mesh.txt",
    config=config,
    columns=columns,
)

# Interpolate all source files
interpolator.interpolate_all()

# Export to ANSYS format
interpolator.export_to_ansys("output_directory")

# Optional: Build VTK for visualization. If outdir=None they are not exported
interpolator.build_vtk_output(outdir="vtk_output")
```

## Interpolation Configuration

Two dataclass objects fully control how the interpolation is performed.

### `InterpolationConfig`

Controls the spatial search strategy, the interpolation kernel, and the target load type.

| Parameter | Type | Description |
|---|---|---|
| `method` | `QUERY_TYPE` | Neighbour search strategy: `QUERY_TYPE.K` (K-nearest) or `QUERY_TYPE.RADIUS` (radius-based). |
| `param` | `int \| float` | Parameter for the chosen method: number of neighbours for `K`, radius (same unit as coordinates) for `RADIUS`. |
| `max_distance` | `float` | Maximum distance (same unit as coordinates) beyond which a destination node is not considered a neighbour, even when using `QUERY_TYPE.K`. Acts as a hard cutoff. |
| `coincidence_tolerance` | `float` | Distance below which two nodes are treated as coincident and the source value is copied directly without interpolation. |
| `kernel` | `INTERPOLATION_KERNEL` | Algorithm used to assign a value from the neighbourhood. See [Interpolation Kernels](#interpolation-kernels). |
| `multithread` | `bool` | Whether to parallelise the interpolation loop using threads. |
| `interpolated_load` | `INTERPOLATED_LOAD_TYPE` | Physical quantity being interpolated. Determines the number of components and the APDL export format. |
| `accept_no_neighbor` | `bool` | If `False` (default), raise an error when a destination node has no neighbour within `max_distance`. If `True`, assign zero to those nodes silently. |

> **Note**: not every kernel is compatible with every load type. See [Kernel–Load Compatibility](#kernel-load-compatibility).

### `ColumnsConfig`

Describes the zero-based column indices in both the source data files and the destination mesh file. All three coordinate components (x, y, z) are expected in consecutive columns starting at the given index.

| Parameter | Type | Description |
|---|---|---|
| `source_xyz` | `int` | Index of the **x** column in the source file; y and z must follow immediately. |
| `value` | `int` | Index of the column containing the scalar (or first component) value to interpolate from the source file. |
| `dest_xyz` | `int` | Index of the **x** column in the destination mesh file; y and z must follow immediately. |
| `id` | `int` | Index of the node-ID column in the destination mesh file. |
| `volume_area` | `int \| None` | Index of the volume/area column in the destination mesh file. Required for scalar integrals and for EM force density inputs. |

## Interpolation checks and outputs

## Analysis Methods

After interpolation, InterpCore provides methods to analyze and validate results:

### Scalar Integrals

For scalar fields (heat flux, heat generation), you can compute the total integral over the destination mesh:

```python
# Requires volume or area data in the destination mesh
columns = ColumnsConfig(
    id=0, dest_xyz=1, source_xyz=1, value=4, volume_area=4
)  # volume_area points to the vol/area column

interpolator = Interpolator(
    path_to_src_folder="source_data",
    path_to_dest_mesh="destination_mesh.txt",
    config=config,
    columns=columns,
)

interpolator.interpolate_all()

# Compute integrals (e.g., total heat generation in W)
integrals = interpolator.compute_scalar_integrals()
# Returns: {"data_001": array([total_value])}
```

**Note**: The destination mesh must include volume (for 3D elements) or area (for 2D elements) data. Use the APDL scripts in [`apdl-scripts/`](apdl-scripts/) to export element centroids with volumes and areas.

### EM Force Resultants

For EM force fields, you can compute force and moment resultants to verify conservation:

```python
# After interpolation with EM_FORCE load type
resultants = interpolator.compute_EM_resultants(pole=np.array([0.0, 0.0, 0.0]))

# Returns for each source file:
# {
#     "force_001": {
#         "R_F_EM": [Fx, Fy, Fz],           # Total force from source data
#         "R_F_Mech": [Fx, Fy, Fz],         # Total force from interpolated data
#         "R_M_EM": [Mx, My, Mz],           # Total moment from source data
#         "R_M_Mech": [Mx, My, Mz],         # Total moment from interpolated data
#         "f_err_comp": [ex, ey, ez],       # Relative force error by component
#         "m_err_comp": [ex, ey, ez],       # Relative moment error by component
#         "Unmapped_EM_Force": float        # Norm of unmapped forces
#     }
# }
```

This is useful for validating that the interpolation preserves global force and moment equilibrium. Small errors indicate good interpolation quality.

## Examples

Complete working examples with sample data are available in the [`doc/`](doc/) folder:

- **[Heat Flux Example](doc/heat_flux/heat_flux.ipynb)**: Scalar field interpolation using the AVERAGE kernel
- **[Heat Generation Example](doc/heat_gen/heat_gen.ipynb)**: Volumetric heat generation using the CLOSEST kernel
- **[EM Force Example](doc/em_force/em_force.ipynb)**: Vector field interpolation with glyph visualization
- **[HTC Example](doc/htc/htc.ipynb)**: Convection boundary condition interpolation (HTC + bulk fluid temperature) 

Each example includes:
- Sample mesh files
- Sample data files
- Jupyter notebook with full workflow
- Visualization with PyVista

## Configuration Options

### Query Methods
- `QUERY_TYPE.K`: K-nearest neighbors (param = number of neighbors)
- `QUERY_TYPE.RADIUS`: Radius-based search (param = radius in same unit as coordinates)

### Interpolation Kernels

#### Source-to-target

Each source point is distributed to destination neighbours:

- `DISTANCE_WEIGHTED`: Weight by inverse distance
- `FEM`: FEM-based interpolation


#### Target-to-source 
A value is assigned to each destination point based on source neighbours

- `CLOSEST`: Use closest source point value
- `AVERAGE`: Simple average of neighbors
- `AVERAGE_WEIGHTED`: Average the neighbours values but weighting them by distance (the closer the more important).

### Load Types
- `EM_FORCE`: 3-component vector fields (Fx, Fy, Fz). If "vol" column is provided the forces are
    interpreted as force densities and will be multiplied by the volume.
- `HEAT_FLUX`: Scalar fields for surface heat flux
- `HEAT_GEN`: Scalar fields for volumetric heat generation
- `HTC`: 2-component convection boundary condition — Heat Transfer Coefficient and bulk fluid (reference) temperature. Exported as `SFE,,CONV,1` and `SFE,,CONV,2` in APDL.

### Kernel-Load Compatibility

Not every kernel can be used with every load type. The kernels are divided into two families based on the direction of the mapping:

| Kernel | Family | Compatible loads |
|---|---|---|
| `DISTANCE_WEIGHTED` | Source-to-target | `EM_FORCE` only |
| `FEM` | Source-to-target | `EM_FORCE` only |
| `AVERAGE` | Target-to-source | `HEAT_FLUX`, `HEAT_GEN`, `HTC` |
| `AVERAGE_WEIGHTED` | Target-to-source | `HEAT_FLUX`, `HEAT_GEN`, `HTC` |
| `CLOSEST` | Target-to-source | `HEAT_FLUX`, `HEAT_GEN`, `HTC` |

Source-to-target kernels distribute each source point's contribution onto its destination neighbours, which is suited to force conservation in EM applications. Target-to-source kernels assign a value to each destination node by aggregating its source neighbours, which is better suited to scalar field interpolation. Using an incompatible combination raises a `ConfigurationError` at construction time.

## File Format

The file format is pretty free, header, no header, commas, tabs....
The important part is that the correct index columns are specified when creating the interpolator.

Destination mesh input files can be created using the apdl scripts included in this repository
[here](/apdl-scripts/).

## Requirements

- Python ≥ 3.10
- scikit-learn
- pandas
- tqdm
- pyvista

## License

Licensed under the [European Union Public Licence (EUPL) 1.2](LICENSE)

## Authors

Developed by the F4E mechanical team
