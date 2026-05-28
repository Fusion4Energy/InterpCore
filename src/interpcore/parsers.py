from pathlib import Path
import logging
import pandas as pd
import numpy as np


def parse_mech_mesh(
    filepath: Path, col_mesh_nd: int = 0, col_mesh_x: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Parse a mechanical mesh and return the array of coordinates and the vector
    of node numbers

    Parameters
    ----------
    filepath : Path
        path to the mechanical mesh file
    col_mesh_nd : int, optional
        index of the column containing node numbers, by default 0
    col_mesh_x : int, optional
        index of the column containing x coordinates, by default 1

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        X_Mech: np.ndarray
            point coordinates array of mechanical mesh nodes
        Node_Number: np.ndarray
            array of mechanical mesh node numbers
    """
    logging.info("Loading Mechanical Mesh...")
    skip1 = _detect_lines_to_skip(filepath)
    Detailed_mesh = _detect_delimiter(filepath, skip1)
    # Detailed_mesh  =  df1.values

    X_Mech = Detailed_mesh[:, col_mesh_x : col_mesh_x + 3]
    Node_Number = Detailed_mesh[:, col_mesh_nd]
    return X_Mech, Node_Number


def parse_em_loads(
    filepath: Path, col_em_x: int = 0, col_em_f: int = 3, vol_col: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Parse an EM loads file and return the array of coordinates and the array of forces

    Parameters
    ----------
    filepath : Path
        path to the EM loads file
    col_em_x : int, optional
        index of the column containing x coordinates, by default 0
    col_em_f : int, optional
        index of the column containing force components, by default 3
    vol_col : int | None, optional
        index of the column containing volume information, by default None

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        X_EM: np.ndarray
            point coordinates array of electromagnetic mesh centroids
        F_EM: np.ndarray
            Nodal force array from EM analysis
    """
    logging.info("Loading EM dataset...")
    skip = _detect_lines_to_skip(filepath)
    EM_analysis = _detect_delimiter(filepath, skip)

    X_EM = EM_analysis[:, col_em_x : col_em_x + 3]
    F_EM = EM_analysis[:, col_em_f : col_em_f + 3]

    # if a force density, scale by the volume
    if vol_col is not None:
        volumes = EM_analysis[:, vol_col]
        F_EM = F_EM * volumes[:, np.newaxis]
    return X_EM, F_EM


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
