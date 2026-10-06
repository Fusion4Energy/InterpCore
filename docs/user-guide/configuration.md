# Configuration

Two dataclasses fully control how the interpolation is performed. Both are defined in
[`interpcore.config`](../api/config.md).

## `InterpolationConfig`

Controls the neighbour search strategy, the interpolation kernel and the target load
type.

| Parameter | Type | Description |
|---|---|---|
| `method` | `QUERY_TYPE` | Neighbour search strategy: `QUERY_TYPE.K` (K-nearest) or `QUERY_TYPE.RADIUS` (radius-based). See [Query methods](query-methods.md). |
| `param` | `int \| float` | Parameter of the chosen method: number of neighbours for `K`, radius (same unit as the coordinates) for `RADIUS`. |
| `max_distance` | `float` | Neighbours farther than this distance are discarded, whatever the query method. Acts as a hard cut-off. |
| `coincidence_tolerance` | `float` | Below this distance two points are considered coincident and the value is copied directly, without interpolation. |
| `kernel` | `INTERPOLATION_KERNEL` | Algorithm used to compute the value from the neighbourhood. See [Kernels](kernels.md). |
| `multithread` | `bool` | Split the interpolation loop into blocks processed in parallel threads (one per CPU). |
| `interpolated_load` | `INTERPOLATED_LOAD_TYPE` | Physical quantity being interpolated. Sets the number of components and the APDL export format. See [Load types](load-types.md). |
| `accept_no_neighbor` | `bool` | Only for target-to-source kernels. If `False` (default), raise an `InterpolationError` when a destination point has no neighbour within `max_distance`. If `True`, assign zero to it. |

The configuration is validated on creation; a `ConfigurationError` is raised if:

- `method` is `QUERY_TYPE.K` and `param` cannot be converted to an integer;
- the `kernel` is not compatible with the `interpolated_load`
  (see [kernel–load compatibility](kernels.md#kernel-load-compatibility)).

## `ColumnsConfig`

Describes the **0-based** column indices of the source data files and of the
destination mesh file. The three coordinates (x, y, z) must be in consecutive columns,
starting at the given index.

| Parameter | Type | File | Description |
|---|---|---|---|
| `source_xyz` | `int` | source | Index of the **x** column; y and z must follow. |
| `value` | `int` | source | Index of the (first) value column. Multi-component loads read the following columns too (3 for `EM_FORCE`, 2 for `HTC`). |
| `dest_xyz` | `int` | destination | Index of the **x** column; y and z must follow. |
| `id` | `int` | destination | Index of the node/element ID column. |
| `volume_area` | `int \| None` | both | Index of the volume or area column. Required for [scalar integrals](analysis.md#scalar-integrals). For `EM_FORCE`, if set, the source values are interpreted as force **densities** and multiplied by the volume read at this index in the source file. |

### Example

Given a source file and a destination mesh like

=== "Source file"

    ```text
    x, y, z, Fx, Fy, Fz
    0.10, 0.20, 0.30, 1.5, -0.2, 3.1
    ...
    ```

=== "Destination mesh"

    ```text
    Node_ID, X, Y, Z
    1, 0.11, 0.19, 0.31
    ...
    ```

the matching configuration is:

```python
columns = ColumnsConfig(source_xyz=0, value=3, dest_xyz=1, id=0)
```
