from interpcore.config import InterpolationConfig, INTERPOLATED_LOAD_TYPE
from pathlib import Path
from interpcore.parsers import parse_mech_mesh, parse_values
from interpcore.dest_tree import DestinationTree
from interpcore.errors import IncompatibleResultsError
import os
import pyvista as pv
import logging
import numpy as np
from tqdm import tqdm


class Interpolator:
    def __init__(
        self,
        path_to_src_folder: str,
        path_to_dest_mesh: str,
        config: InterpolationConfig,
        file_idx: dict,
    ):
        """Class to handle the interpolation operations

        Parameters
        ----------
        path_to_src_folder : str
            path where the input cloud points are stored
        path_to_dest_mesh : str
            path to the mechanical mesh file
        config : InterpolationConfig
            configuration for the interpolation problem
        file_idx : dict
            dictionary with file indices for parsing
        """
        # parse all necessary files
        if file_idx is None:
            mesh_data = parse_mech_mesh(Path(path_to_dest_mesh))
        else:
            mesh_data = parse_mech_mesh(
                Path(path_to_dest_mesh),
                col_mesh_ids=file_idx.get("ids", 0),
                col_mesh_x=file_idx.get("dest_x", 1),
                col_vol=file_idx.get("vol", None),
                col_area=file_idx.get("area", None),
            )

        mech_x = mesh_data["coordinates"]
        ids = mesh_data["id_numbers"]
        self.volumes = mesh_data.get("volumes", None)
        self.areas = mesh_data.get("areas", None)

        # parse all value files to interpolate
        src_values = {}

        # raise an error if folder does not exist or is empty
        if not os.path.exists(path_to_src_folder):
            raise FileNotFoundError(
                f"Source folder {path_to_src_folder} does not exist."
            )
        if len(os.listdir(path_to_src_folder)) == 0:
            raise FileNotFoundError(f"Source folder {path_to_src_folder} is empty.")

        for file in os.listdir(path_to_src_folder):
            file_path = Path(path_to_src_folder, file)
            name = file_path.stem
            src_coordinates, values = parse_values(
                file_path,
                load_type=config.interpolated_load,
                file_idx=file_idx,
                n_components=config.num_components,
            )
            src_values[name] = values

        self.src_values = src_values

        # Intialize the KDtree and precompute the queries (assumption that em_x is the same)
        self.tree = DestinationTree(
            dest_coordinates=mech_x,
            src_coordinates=src_coordinates,
            dest_ids=ids,
            config=config,
        )
        self.interpolated_results = None
        self.dest_vtk: dict[str, pv.PolyData] = {}
        self.src_vtk: dict[str, pv.PolyData] = {}

    def interpolate_all(self):
        """Go through all em forces file and interpolate them"""
        interpolated_results = {}
        for name, values in tqdm(self.src_values.items(), total=len(self.src_values)):
            interpolated, unmapped = self.tree.interpolate(values)
            interpolated_results[name] = {
                "interpolated": interpolated,
                "unmapped": unmapped,
            }
        self.interpolated_results = interpolated_results

    # def dump_interpolation_check(self, outfile: Path, pole: np.ndarray | None = None):
    #     """Dump a csv file with the interpolation check results
    #     Parameters
    #     ----------
    #     outfile : Path
    #         output csv file path
    #     pole : np.ndarray | None, optional
    #         reference pole for moment calculation, by default None (0,0,0)
    #     """
    #     rows = []
    #     for name in self.src_values.keys():
    #         row = self._compute_resultants(name, pole)
    #         rows.append(row)

    #     df = pd.DataFrame(rows)
    #     df.to_csv(outfile, index=False)
    #     logging.info(f"Interpolation check dumped to {outfile}")

    def export_to_ansys(self, outdir: Path):
        """Export each interpolated result to an ANSYS format

        Parameters
        ----------
        outdir : Path
            output directory for the interpolated files
        """
        if self.interpolated_results is None:
            raise ValueError(
                "No interpolated results found. Run interpolate_all() first."
            )

        templates = _select_template(self.tree.config.interpolated_load)

        for name, result in self.interpolated_results.items():
            outfile = Path(outdir, f"interpolated_{name}.txt")

            with open(outfile, "w") as f:
                for i, template in enumerate(templates):
                    for j, id in enumerate(self.tree.ids):
                        f.write(template.format(id, result["interpolated"][j, i]))

        logging.info(f"ANSYS files exported to {outdir}")

    def compute_scalar_integrals(self) -> dict[str, float]:
        """Compute the integrals of the interpolated results over the destination mesh"""
        if self.interpolated_results is None:
            raise ValueError(
                "No interpolated results found. Run interpolate_all() first."
            )

        integrals = {}
        for name, result in self.interpolated_results.items():
            if self.tree.config.num_components > 1:
                raise IncompatibleResultsError(
                    "Integral computation only supported for scalar results (num_components=1)."
                )
            if self.volumes is not None:
                integral = np.sum(
                    result["interpolated"] * self.volumes[:, np.newaxis], axis=0
                )
            elif self.areas is not None:
                integral = np.sum(
                    result["interpolated"] * self.areas[:, np.newaxis], axis=0
                )
            else:
                raise IncompatibleResultsError(
                    "No volume or area information found. Cannot compute integrals."
                )
            integrals[name] = integral

        return integrals

    def compute_EM_resultants(
        self, pole: np.ndarray | None = None
    ) -> dict[str, dict[str, float]]:
        """Compute the force and moment resultants of the EM forces and the interpolated
        forces, and their differences

        Parameters
        ----------
        pole : np.ndarray | None, optional
            reference pole for moment calculation, by default None (0,0,0)

        Returns
        -------
        dict[str, dict[str, float]]
            dictionary containing the resultants and their differences for each
            interpolated load
        """
        if self.interpolated_results is None:
            raise ValueError(
                "No interpolated results found. Run interpolate_all() first."
            )
        if pole is None:
            pole = np.array([0.0, 0.0, 0.0])

        resultants = {}
        for name in self.src_values.keys():
            F_EM = self.src_values[name]
            F_Mech = self.interpolated_results[name]["interpolated"]

            R_F_EM = np.sum(F_EM, axis=0)
            R_F_Mech = np.sum(F_Mech, axis=0)

            M_EM = np.cross(self.tree.src_coordinates - pole, F_EM)
            M_Mech = np.cross(self.tree.dest_coordinates - pole, F_Mech)

            R_M_EM = np.sum(M_EM, axis=0)
            R_M_Mech = np.sum(M_Mech, axis=0)

            f_err_comp = np.divide(R_F_EM - R_F_Mech, R_F_EM)
            m_err_comp = np.divide(R_M_EM - R_M_Mech, R_M_EM)

            resultants[name] = {
                "R_F_EM": R_F_EM,
                "R_F_Mech": R_F_Mech,
                "R_M_EM": R_M_EM,
                "R_M_Mech": R_M_Mech,
                "f_err_comp": f_err_comp,
                "m_err_comp": m_err_comp,
                "Unmapped_EM_Force": np.linalg.norm(
                    self.interpolated_results[name]["unmapped"]
                ),
            }

        return resultants

    def build_vtk_output(self, outdir: Path | None = None):
        """Build the VTK output files for visualization

        Parameters
        ----------
        oiutdir : Path | None, optional
            output directory for the vtk files, if None, files are not saved, by default None

        """
        if self.interpolated_results is None:
            raise ValueError(
                "No interpolated results found. Run interpolate_all() first."
            )
        # build and dump the vtks
        for name in self.src_values.keys():
            # Destination mesh
            pdata = pv.PolyData(self.tree.dest_coordinates)
            interp_matrix = self.interpolated_results[name]["interpolated"]

            # For scalar data (1 component), flatten to 1D array for better performance
            if interp_matrix.shape[1] == 1:
                pdata["Value"] = interp_matrix.ravel()
            else:
                pdata["Value"] = interp_matrix
                for i in range(interp_matrix.shape[1]):
                    pdata[f"Component_{i}"] = interp_matrix[:, i]
            self.dest_vtk[name] = pdata

            # Source mesh
            pdata_em = pv.PolyData(self.tree.src_coordinates)
            interp_matrix_em = self.src_values[name]

            # For scalar data (1 component), flatten to 1D array for better performance
            if interp_matrix_em.shape[1] == 1:
                pdata_em["Value"] = interp_matrix_em.ravel()
            else:
                pdata_em["Value"] = interp_matrix_em
                for i in range(interp_matrix_em.shape[1]):
                    pdata_em[f"Component_{i}"] = interp_matrix_em[:, i]

            self.src_vtk[name] = pdata_em

            if outdir is not None:
                outfile = Path(outdir, f"{name}_interpolated.vtk")
                self.dest_vtk[name].save(outfile)
                outfile_em = Path(outdir, f"{name}_EM.vtk")
                self.src_vtk[name].save(outfile_em)


def _select_template(interpolated_load: INTERPOLATED_LOAD_TYPE) -> list[str]:
    if interpolated_load == INTERPOLATED_LOAD_TYPE.EM_FORCE:
        return [
            "F, {}, Fx, {}\n",
            "F, {}, Fy, {}\n",
            "F, {}, Fz, {}\n",
        ]
    elif interpolated_load == INTERPOLATED_LOAD_TYPE.HEAT_FLUX:
        return [
            "SFE, {},, HFLUX,, {}\n",
        ]
    elif interpolated_load == INTERPOLATED_LOAD_TYPE.HEAT_GEN:
        return [
            "BFE, {}, HGEN,, {}\n",
        ]
    elif interpolated_load == INTERPOLATED_LOAD_TYPE.HTC:
        return [
            "SFE, {},, CONV, 1, {}\n",  # HTC
            "SFE, {},, CONV, 2, {}\n",  # TEMP
        ]
    else:
        raise NotImplementedError(f"Unsupported load type: {interpolated_load}")
