# Analysis and checks

After `interpolate_all()`, the [`Interpolator`](../api/interpolator.md) offers methods
to check the quality of the interpolation.

## Scalar integrals

For scalar loads (`HEAT_FLUX`, `HEAT_GEN`), `compute_scalar_integrals()` integrates
the interpolated field over the destination mesh:

$$
Q = \sum_{e} q_e\, V_e
$$

where $V_e$ is the volume (or area) of element $e$. For heat generation in W/m³ and
volumes in m³, the result is the total power in W. Compare it with the same quantity
computed in the source model to check the interpolation.

The `volume_area` column must be set in the
[`ColumnsConfig`](configuration.md#columnsconfig):

```python
columns = ColumnsConfig(id=0, dest_xyz=1, source_xyz=1, value=4, volume_area=4)

interpolator = Interpolator(
    path_to_src_folder="source_data",
    path_to_dest_mesh="destination_mesh.txt",
    config=config,
    columns=columns,
)
interpolator.interpolate_all()

integrals = interpolator.compute_scalar_integrals()
# {"data_001": array([total_value])}
```

An `IncompatibleResultsError` is raised for multi-component loads or when no volume or
area is available.

!!! tip
    Element centroids with areas and volumes can be exported with the
    [`export_centroids.apdl`](file-formats.md#apdl-export-scripts) script.

## EM force resultants

For `EM_FORCE`, `compute_EM_resultants(pole)` compares the total force and moment of
the source data with those of the interpolated nodal forces:

$$
\mathbf{R}_F = \sum_i \mathbf{F}_i,
\qquad
\mathbf{R}_M = \sum_i (\mathbf{x}_i - \mathbf{p}) \times \mathbf{F}_i
$$

where $\mathbf{p}$ is the pole (origin by default).

```python
import numpy as np

resultants = interpolator.compute_EM_resultants(pole=np.array([0.0, 0.0, 0.0]))
```

For each source file, the returned dictionary contains:

| Key | Description |
|---|---|
| `R_F_EM` | Total force of the source data $[F_x, F_y, F_z]$ |
| `R_F_Mech` | Total force of the interpolated data |
| `R_M_EM` | Total moment of the source data about the pole $[M_x, M_y, M_z]$ |
| `R_M_Mech` | Total moment of the interpolated data about the pole |
| `f_err_comp` | Relative force error per component, $(R_{F,EM} - R_{F,Mech}) / R_{F,EM}$ |
| `m_err_comp` | Relative moment error per component |
| `Unmapped_EM_Force` | Norm of the sum of the source forces that found no destination neighbour |

Both source-to-target kernels conserve the total force by construction, so a force
error usually means that part of the force was unmapped. The moment is not conserved
exactly, since forces are moved from the source points to the nodes: its error
depends on the distance between the two point sets and is a good indicator of the
interpolation quality.

## Visualisation

`build_vtk_output(outdir)` builds PyVista `PolyData` point clouds of both the source
and the interpolated data, stored in the `src_vtk` and `dest_vtk` dictionaries. If
`outdir` is given, they are also saved as `<name>_src.vtk` and
`<name>_interpolated.vtk` for ParaView.

```python
import pyvista as pv

interpolator.build_vtk_output()
name = "data_001"

pl = pv.Plotter(shape=(1, 2))
pl.subplot(0, 0)
pl.add_mesh(interpolator.src_vtk[name], scalars="Value")
pl.subplot(0, 1)
pl.add_mesh(interpolator.dest_vtk[name], scalars="Value")
pl.link_views()
pl.show()
```

For multi-component loads, each component is also stored as a separate
`Component_<i>` array.
