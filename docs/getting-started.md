# Getting started

A complete interpolation consists of four steps:

1. describe **how** to interpolate with an [`InterpolationConfig`](user-guide/configuration.md#interpolationconfig);
2. describe **where** the data are in the input files with a [`ColumnsConfig`](user-guide/configuration.md#columnsconfig);
3. create an [`Interpolator`](api/interpolator.md) and run it;
4. export the results to APDL and, optionally, to VTK.

```python
from interpcore.interpolator import Interpolator
from interpcore.config import (
    InterpolationConfig,
    ColumnsConfig,
    QUERY_TYPE,
    INTERPOLATED_LOAD_TYPE,
    INTERPOLATION_KERNEL,
)

# 1. How to interpolate
config = InterpolationConfig(
    method=QUERY_TYPE.K,  # type of neighbour search
    param=5,  # K for QUERY_TYPE.K, radius for QUERY_TYPE.RADIUS
    max_distance=2.0,  # hard cut-off on neighbour distance
    coincidence_tolerance=0.01,  # below this distance nodes are coincident
    kernel=INTERPOLATION_KERNEL.DISTANCE_WEIGHTED,
    multithread=False,
    interpolated_load=INTERPOLATED_LOAD_TYPE.EM_FORCE,
)

# 2. Where the data are (0-based column indices)
columns = ColumnsConfig(id=0, dest_xyz=1, source_xyz=1, value=4)

# 3. Interpolate every file found in the source folder
interpolator = Interpolator(
    path_to_src_folder="source_data",
    path_to_dest_mesh="destination_mesh.txt",
    config=config,
    columns=columns,
)
interpolator.interpolate_all()

# 4. Export
interpolator.export_to_ansys("output_directory")
interpolator.build_vtk_output(outdir="vtk_output")  # outdir=None: build only, no files
```

!!! tip "Multiple load cases"
    `path_to_src_folder` can be a folder containing many source files (e.g. one per
    time step or load case) sharing the same coordinates. The KDTree query is computed
    only once and re-used for every file. A single file path is also accepted.

## Next steps

- Understand the [interpolation workflow](user-guide/how-it-works.md).
- Pick the right [kernel](user-guide/kernels.md) for your [load type](user-guide/load-types.md).
- [Validate the results](user-guide/analysis.md) with integrals and resultants.
- Browse the [examples](examples/index.md).
