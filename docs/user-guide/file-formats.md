# File formats

## Input files

The input format is flexible. The parser:

- skips the header lines automatically (lines containing more letters than digits);
- detects the delimiter from the first line: `;`, `,`, `:` or, if none is found,
  any whitespace (spaces or tabs).

Only the column indices must be given, through the
[`ColumnsConfig`](configuration.md#columnsconfig).

### Source files

One file per load case, with at least the three coordinates and the value columns.
All source files of a folder must share the same coordinates, as the neighbour query
is computed only once.

```text
X [m]   Y [m]   Z [m]   Q [W/m3]
0.100   0.200   0.300   3.4e6
...
```

### Destination mesh

One file with the IDs, the coordinates and, optionally, the volume or area of the
destination points:

- **nodes** for `EM_FORCE`;
- **element centroids** for `HEAT_FLUX`, `HEAT_GEN` and `HTC`.

## APDL export scripts

The [`apdl-scripts/`](https://github.com/Fusion4Energy/InterpCore/tree/main/apdl-scripts)
folder contains two macros to export the destination mesh from an ANSYS `.cdb` file.
Edit the variables in the `USER INPUTS` section, then run the macro in ANSYS
Mechanical APDL.

=== "export_nodes.apdl"

    Exports the nodes of a component to a CSV file, for `EM_FORCE`.

    ```text
    Node_ID,X,Y,Z
    ```

    ```python
    columns = ColumnsConfig(id=0, dest_xyz=1, source_xyz=..., value=...)
    ```

=== "export_centroids.apdl"

    Exports the element centroids of a component, with their area and volume, to a
    CSV file, for `HEAT_FLUX`, `HEAT_GEN` and `HTC`.

    ```text
    Element_ID,X,Y,Z,Area,Volume
    ```

    ```python
    # volume_area=4 for areas (surface loads), 5 for volumes
    columns = ColumnsConfig(id=0, dest_xyz=1, source_xyz=..., value=..., volume_area=5)
    ```

Both macros use the same user inputs:

```text
cdb_file = 'CDB_name'                 ! CDB file name without extension
named_selection = 'component_name'    ! Name of the component/named selection
output_file = 'nodes_output'          ! Output CSV file name without extension
```

## Output files

`export_to_ansys(outdir)` writes one `interpolated_<name>.txt` file per source file,
containing APDL commands that can be read with `/INPUT`. See
[Load types](load-types.md) for the commands written for each load.
