from pathlib import Path
import logging
import pandas as pd
import numpy as np
from interpcore.config import INTERPOLATED_LOAD_TYPE


def parse_mech_mesh(
    filepath: Path, col_mesh_ids: int = 0, col_mesh_x: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Parse a mechanical mesh and return the array of coordinates and the vector
    of node numbers

    Parameters
    ----------
    filepath : Path
        path to the mechanical mesh file
    col_mesh_ids : int, optional
        index of the column containing node/element numbers, by default 0
    col_mesh_x : int, optional
        index of the column containing x coordinates, by default 1

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        coordinates: np.ndarray
            point coordinates array of mechanical mesh nodes/element centroids
        id_numbers: np.ndarray
            array of mechanical mesh node/element numbers
    """
    logging.info("Loading Mechanical Mesh...")
    skip1 = _detect_lines_to_skip(filepath)
    Detailed_mesh = _detect_delimiter(filepath, skip1)
    # Detailed_mesh  =  df1.values

    coordinates = Detailed_mesh[:, col_mesh_x : col_mesh_x + 3]
    id_numbers = Detailed_mesh[:, col_mesh_ids]
    return coordinates, id_numbers


class CloudParser:
    def __init__(
        self,
        filepath: Path,
    ):
        """Object responsible for the parsing operation

        Parameters
        ----------
        filepath : Path
            path to the source file
        """
        # load the dataframe
        logging.info("Loading EM dataset...")
        skip = _detect_lines_to_skip(filepath)
        self.src_data = _detect_delimiter(filepath, skip)

    def get_coordinates(self, col_mesh_x: int = 1) -> np.ndarray:
        """Extract coordinates from the loaded data based on the specified columns and
        number of components.

        Parameters
        ----------
        col_mesh_x : int, optional
            index of the column containing x coordinates, by default 1
        Returns
        -------
        np.ndarray
            point coordinates array of source mesh points
        """
        return self.src_data[:, col_mesh_x : col_mesh_x + 3]

    def get_values(self, val_idx: int, n_components: int = 1) -> np.ndarray:
        """Extract values from the loaded data based on the specified column.

        Parameters
        ----------
        val_idx : int
            index of the column containing the values
        n_components : int, optional
            number of components to extract, by default 1
        Returns
        -------
        np.ndarray
            array of values from the specified columns
        """
        return self.src_data[:, val_idx : val_idx + n_components]


class EMCloudParser(CloudParser):
    def get_values(
        self, val_idx: int, n_components: int = 3, v_idx: int | None = None
    ) -> np.ndarray:
        """Extract force values from the loaded data based on the specified columns for
        forces and volumes.

        Parameters
        ----------
        val_idx : int
            index of the column containing the force components
        v_idx : int, optional
            index of the column containing volume information, by default None.
            If provided, the forces are interpreted as densities and scaled by the volume.

        Returns
        -------
        np.ndarray
            array of force values, scaled by volume if v_idx is provided
        """
        forces = self.src_data[:, val_idx : val_idx + n_components]
        if v_idx is not None:
            volumes = self.src_data[:, v_idx]
            forces = forces * volumes[:, np.newaxis]
        return forces


def _detect_lines_to_skip(csvFile: Path) -> int:
    with open(csvFile, "r") as myCsvfile:
        skip = -1
        letters = 1
        numbers = 0

        while letters > numbers:
            skip = skip + 1
            total_line = myCsvfile.readline()
            numbers = sum(c.isdigit() for c in total_line)
            letters = sum(c.isalpha() for c in total_line)
            # spaces  = sum(c.isspace() for c in total_line)
            # others  = len(total_line) - numbers - letters - spaces

    return skip


def _detect_delimiter(csvFile: Path, skip: int) -> np.ndarray:
    with open(csvFile, "r") as myCsvfile:
        header = myCsvfile.readline()
        delimiter = None
        for d in [";", ",", ":"]:
            if header.find(d) != -1:
                delimiter = d
                break
        if delimiter is None:
            delimiter = r"\s+"

        logging.info(f"EM Delimiter detected: '{delimiter}'")

        df = pd.read_csv(csvFile, sep=delimiter, skiprows=skip, header=None).values
        return df


def parse_values(
    filepath: Path,
    load_type: INTERPOLATED_LOAD_TYPE,
    file_idx: dict,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Parse values to be interpolated

    Parameters
    ----------
    filepath : Path
        path to the source file
    load_type : INTERPOLATED_LOAD_TYPE
        type of load to be interpolated
    file_idx : dict
        indices map to correctly read the source file
    n_components : int
        number of components to extract

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        coordinates: np.ndarray
            point coordinates array of source mesh points
        values: np.ndarray
            array of values to interpolate
    """
    if load_type == INTERPOLATED_LOAD_TYPE.EM_FORCE:
        parser = EMCloudParser(filepath)
        values = parser.get_values(
            val_idx=file_idx.get("val", 3),
            n_components=n_components,
            v_idx=file_idx.get("volume", None),
        )
    else:
        parser = CloudParser(filepath)
        values = parser.get_values(val_idx=file_idx["val"], n_components=n_components)

    coordinates = parser.get_coordinates(col_mesh_x=file_idx["src_x"])
    return coordinates, values
