# InterpCore

A Python library for interpolating physical field data (electromagnetic forces, heat flux, etc.) between different mesh representations and exporting to ANSYS APDL format.

## Features

- **Multiple interpolation kernels**: Distance-weighted, FEM-based, K-nearest neighbors, closest point
- **Flexible query methods**: K-nearest neighbors or radius-based search
- **Support for multiple load types**: 
  - EM forces (3-component vector fields)
  - Heat flux (scalar fields)
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
from interpcore.config import InterpolationConfig, QUERY_TYPE, INTERPOLATED_LOAD_TYPE
from interpcore.kernels import INTERPOLATION_KERNEL

# Configure interpolation
config = InterpolationConfig(
    method=QUERY_TYPE.K,
    param=5,
    max_distance=2.0,
    coincidence_tolerance=0.01,
    kernel=INTERPOLATION_KERNEL.DISTANCE_WEIGHTED,
    multithread=False,
    interpolated_load=INTERPOLATED_LOAD_TYPE.EM_FORCE
)

# Define file column indices
file_idx = {"ids": 0, "dest_x": 1, "src_x": 1, "val": 4}

# Create interpolator and run
interpolator = Interpolator(
    path_to_src_folder="source_data",
    path_to_dest_mesh="destination_mesh.txt",
    config=config,
    file_idx=file_idx
)

# Interpolate all source files
interpolator.interpolate_all()

# Export to ANSYS format
interpolator.export_to_ansys("output_directory")

# Optional: Build VTK for visualization
interpolator.build_vtk_output(outdir="vtk_output")
```

## Examples

Complete working examples with sample data are available in the [`doc/`](doc/) folder:

- **[Heat Flux Example](doc/heat_flux/)**: Scalar field interpolation using the AVERAGE kernel
- **[Heat Generation Example](doc/heat_gen/)**: Volumetric heat generation using the CLOSEST kernel
- **[EM Force Example](doc/em_force/)**: Vector field interpolation with glyph visualization

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

### Load Types
- `EM_FORCE`: 3-component vector fields (Fx, Fy, Fz). If "vol" column is provided the forces are
    interpreted as force densities and will be multiplied by the volume.
- `HEAT_FLUX`: Scalar fields for surface heat flux
- `HEAT_GEN`: Scalar fields for volumetric heat generation

## File Format

### Destination Mesh
```
Node_ID X Y Z
101 0.25 0.33 0.00
102 0.25 1.00 0.00
...
```

### Source Data (EM Forces)
```
Node_ID X Y Z Fx Fy Fz
1 0.50 0.50 0.00 10.50 5.20 2.10
2 0.60 1.50 0.00 12.30 6.40 2.35
...
```

### Source Data (Heat Flux)
```
Node_ID X Y Z HeatFlux
1 0.50 0.50 0.00 100.00
2 0.50 1.50 0.00 130.00
...
```

### Source Data (Heat Generation)
```
Node_ID X Y Z HeatGen
1 0.45 0.70 0.00 500.00
2 0.45 1.50 0.00 550.00
...
```

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
