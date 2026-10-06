# Examples

Complete workflows with sample data. Each example folder contains a source data
folder, a destination mesh and a Jupyter notebook; the pages below are rendered from
the notebooks.

<div class="grid cards" markdown>

-   **Heat flux**

    ---

    Scalar surface load interpolated with the `AVERAGE` kernel.

    [:octicons-arrow-right-24: Example](heat_flux/heat_flux.md) ·
    [:octicons-download-24: Notebook](heat_flux/heat_flux.ipynb)

-   **Heat generation**

    ---

    Volumetric heat generation interpolated with the `CLOSEST` kernel, with total
    power check.

    [:octicons-arrow-right-24: Example](heat_gen/heat_gen.md) ·
    [:octicons-download-24: Notebook](heat_gen/heat_gen.ipynb)

-   **EM force**

    ---

    Vector field interpolation with force/moment resultants and glyph
    visualisation.

    [:octicons-arrow-right-24: Example](em_force/em_force.md) ·
    [:octicons-download-24: Notebook](em_force/em_force.ipynb)

-   **HTC**

    ---

    Convection boundary condition: Heat Transfer Coefficient and bulk fluid
    temperature.

    [:octicons-arrow-right-24: Example](htc/htc.md) ·
    [:octicons-download-24: Notebook](htc/htc.ipynb)

</div>

To run the examples locally, clone the repository and open the notebooks from the
[`docs/examples/`](https://github.com/Fusion4Energy/InterpCore/tree/main/docs/examples)
folder, so that the relative paths to the data files are correct.
