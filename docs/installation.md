# Installation

InterpCore requires **Python ≥ 3.11**.

=== "pip"

    ```bash
    pip install interpcore
    ```

=== "From source"

    ```bash
    git clone https://github.com/Fusion4Energy/InterpCore.git
    cd InterpCore
    pip install -e .[dev]
    ```

## Dependencies

Installed automatically:

- [scikit-learn](https://scikit-learn.org/) (KDTree and distance computations)
- [pandas](https://pandas.pydata.org/) (file parsing)
- [tqdm](https://tqdm.github.io/) (progress bars)
- [PyVista](https://docs.pyvista.org/) (VTK output and visualisation)

## Building this documentation

The documentation is built with [Zensical](https://zensical.org/). The example pages
are generated from the Jupyter notebooks with `nbconvert`, using the outputs saved in
the notebooks (they are not re-executed).

```bash
pip install -e .[docs]
jupyter nbconvert --to markdown docs/examples/*/*.ipynb
zensical serve
```
