# InterpCore

[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://fusion4energy.github.io/InterpCore/)

A Python library for interpolating physical field data (electromagnetic forces, heat
flux, heat generation, convection boundary conditions) between different mesh
representations and exporting the result to ANSYS APDL format.

**Full documentation: <https://fusion4energy.github.io/InterpCore/>**

## Features

- Multiple interpolation kernels: distance-weighted, FEM-based, average, weighted
  average, closest point
- K-nearest or radius-based neighbour search (KDTree)
- EM forces, heat flux, heat generation and HTC + bulk temperature
- Force/moment resultants and scalar integrals to validate the interpolation
- Direct export to ANSYS APDL and VTK output for ParaView/PyVista

## Installation

```bash
pip install interpcore
```

Requires Python ≥ 3.11.

## Quick Start

```python
from interpcore.interpolator import Interpolator
from interpcore.config import (
    InterpolationConfig,
    ColumnsConfig,
    QUERY_TYPE,
    INTERPOLATED_LOAD_TYPE,
    INTERPOLATION_KERNEL,
)

config = InterpolationConfig(
    method=QUERY_TYPE.K,
    param=5,
    max_distance=2.0,
    coincidence_tolerance=0.01,
    kernel=INTERPOLATION_KERNEL.DISTANCE_WEIGHTED,
    multithread=False,
    interpolated_load=INTERPOLATED_LOAD_TYPE.EM_FORCE,
)
columns = ColumnsConfig(id=0, dest_xyz=1, source_xyz=1, value=4)

interpolator = Interpolator(
    path_to_src_folder="source_data",
    path_to_dest_mesh="destination_mesh.txt",
    config=config,
    columns=columns,
)
interpolator.interpolate_all()
interpolator.export_to_ansys("output_directory")
```

See the [documentation](https://fusion4energy.github.io/InterpCore/) for the
configuration options, the kernels and load types, and the
[examples](docs/examples/) for complete Jupyter notebook workflows.

## License

Licensed under the [European Union Public Licence (EUPL) 1.2](LICENSE)

## Authors

Developed by the F4E mechanical team
