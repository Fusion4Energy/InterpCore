# Load types

The `interpolated_load` parameter of the
[`InterpolationConfig`](configuration.md#interpolationconfig) sets the number of
value columns read from the source files and the APDL commands written by
[`export_to_ansys`](../api/interpolator.md).

| Load type | Components | Destination points | Kernels | APDL command |
|---|---|---|---|---|
| `EM_FORCE` | 3 ($F_x, F_y, F_z$) | nodes | source-to-target | `F` |
| `HEAT_FLUX` | 1 | element centroids | target-to-source | `SFE,,HFLUX` |
| `HEAT_GEN` | 1 | element centroids | target-to-source | `BFE,,HGEN` |
| `HTC` | 2 (HTC, $T_{bulk}$) | element centroids | target-to-source | `SFE,,CONV` |

The ID column of the destination mesh is written as the node/element number of the
APDL command, so it must contain node IDs for `EM_FORCE` and element IDs for the
other loads. The [APDL scripts](file-formats.md#apdl-export-scripts) export
destination meshes in the right format.

## `EM_FORCE`

Nodal force vectors. The value columns are $F_x, F_y, F_z$, starting at
`ColumnsConfig.value`.

If `ColumnsConfig.volume_area` is set, the source values are interpreted as **force
densities** and multiplied by the volume column of the source file before
interpolation.

```text
F, 1234, Fx, 1.523000e+01
...
F, 1234, Fy, -2.100000e-01
...
F, 1234, Fz, 3.100000e+00
```

## `HEAT_FLUX`

Surface heat flux applied to element faces.

```text
SFE, 5678,, HFLUX,, 1.250000e+05
```

## `HEAT_GEN`

Volumetric heat generation applied to elements.

```text
BFE, 5678, HGEN,, 3.400000e+06
```

## `HTC`

Convection boundary condition. The first value column is the Heat Transfer
Coefficient, the second one (next column) is the bulk fluid temperature.

```text
SFE, 5678,, CONV, 1, 2.000000e+04
...
SFE, 5678,, CONV, 2, 1.500000e+02
```

!!! note
    The output file contains first all the lines of the first component, then all
    those of the second component, and so on. The number formatting shown above is
    illustrative.
