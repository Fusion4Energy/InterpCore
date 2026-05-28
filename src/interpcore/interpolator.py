from interpcore.config import InterpolationConfig
from pathlib import Path
from interpcore.parsers import parse_mech_mesh, parse_values
from interpcore.dest_tree import DestinationTree
import os
import numpy as np
import logging
import pandas as pd


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
            mech_x, ids = parse_mech_mesh(Path(path_to_dest_mesh))
        else:
            mech_x, ids = parse_mech_mesh(
                Path(path_to_dest_mesh),
                col_mesh_ids=file_idx.get("ids", 0),
                col_mesh_x=file_idx.get("dest_x", 1),
            )

        # parse all value files to interpolate
        src_values = {}

        # raise an error if folder does not exist or is empty
        if not os.path.exists(path_to_src_folder):
            raise FileNotFoundError(f"EM folder {path_to_src_folder} does not exist.")
        if len(os.listdir(path_to_src_folder)) == 0:
            raise FileNotFoundError(f"EM folder {path_to_src_folder} is empty.")

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
        self.dest_vtk = {}
        self.src_vtk = {}

    def interpolate_all(self, progress_callback=None):
        """Go through all em forces file and interpolate them
        Parameters
        ----------
        progress_callback : callable, optional
            Function to call with progress percentage (int 0-100)
        """
        interpolated_results = {}
        total = len(self.src_values)
        for idx, (name, values) in enumerate(self.src_values.items()):
            interpolated, unmapped = self.tree.interpolate(values)
            interpolated_results[name] = {
                "interpolated": interpolated,
                "unmapped": unmapped,
            }
            if progress_callback is not None:
                percent = int(100 * (idx + 1) / total)
                progress_callback(percent)
        self.interpolated_results = interpolated_results

    def dump_interpolation_check(self, outfile: Path, pole: np.ndarray | None = None):
        """Dump a csv file with the interpolation check results
        Parameters
        ----------
        outfile : Path
            output csv file path
        pole : np.ndarray | None, optional
            reference pole for moment calculation, by default None (0,0,0)
        """
        rows = []
        for name in self.src_values.keys():
            row = self._compute_resultants(name, pole)
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(outfile, index=False)
        logging.info(f"Interpolation check dumped to {outfile}")

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

        for name, result in self.interpolated_results.items():
            outfile = Path(outdir, f"interpolated_{name}.txt")

            with open(outfile, "w") as f:
                for i, label in enumerate(["Fx", "Fy", "Fz"]):
                    for j, nodeID in enumerate(self.node_numbers):
                        f.write(
                            f"F, {int(nodeID)}, {label}, {result['interpolated'][j][i]}\n"
                        )

        logging.info(f"ANSYS files exported to {outdir}")

    def _compute_resultants(self, name: str, pole: np.ndarray | None = None) -> dict:
        if pole is None:
            pole = np.array([0.0, 0.0, 0.0])

        F_EM = self.src_values[name]
        if self.interpolated_results is None:
            raise ValueError(
                "No interpolated results found. Run interpolate_all() first."
            )
        F_Mech = self.interpolated_results[name]["interpolated"]

        R_F_EM = np.sum(F_EM, axis=0)
        R_F_Mech = np.sum(F_Mech, axis=0)

        M_EM = np.cross(self.em_x - pole, F_EM)
        M_Mech = np.cross(self.mech_x - pole, F_Mech)

        R_M_EM = np.sum(M_EM, axis=0)
        R_M_Mech = np.sum(M_Mech, axis=0)

        f_err_comp = np.divide(R_F_EM - R_F_Mech, R_F_EM)
        m_err_comp = np.divide(R_M_EM - R_M_Mech, R_M_EM)

        # give some warning if differences are very high
        if np.any(np.abs(f_err_comp) > 0.2):
            logging.warning(f"High difference in force resultant for {name}")
        if np.any(np.abs(m_err_comp) > 0.2):
            logging.warning(f"High difference in moment resultant for {name}")

        row = {
            "Name": name,
            "Fx [N]": R_F_EM[0],
            "Fy [N]": R_F_EM[1],
            "Fz [N]": R_F_EM[2],
            "Mx [Nm]": R_M_EM[0],
            "My [Nm]": R_M_EM[1],
            "Mz [Nm]": R_M_EM[2],
            "dFx [%]": f_err_comp[0] * 100,
            "dFy [%]": f_err_comp[1] * 100,
            "dFz [%]": f_err_comp[2] * 100,
            "dMx [%]": m_err_comp[0] * 100,
            "dMy [%]": m_err_comp[1] * 100,
            "dMz [%]": m_err_comp[2] * 100,
            "Unmapped_EM_Force [N]": np.linalg.norm(
                self.interpolated_results[name]["unmapped"]
            ),
        }

        return row

    def build_vtk_output(self):
        """Build the VTK output files for visualization

        Parameters
        ----------
        outdir : Path
            output directory for the interpolated files

        """
        if self.interpolated_results is None:
            raise ValueError(
                "No interpolated results found. Run interpolate_all() first."
            )
        # build and dump the vtks
        for name in self.src_values.keys():
            # Mech
            pdata = pv.PolyData(self.mech_x)
            pdata["Fx [N]"] = self.interpolated_results[name]["interpolated"][:, 0]
            pdata["Fy [N]"] = self.interpolated_results[name]["interpolated"][:, 1]
            pdata["Fz [N]"] = self.interpolated_results[name]["interpolated"][:, 2]
            pdata["Force [N]"] = self.interpolated_results[name]["interpolated"]
            self.dest_vtk[name] = pdata

            # EM
            pdata_em = pv.PolyData(self.em_x)
            pdata_em["Fx [N]"] = self.src_values[name][:, 0]
            pdata_em["Fy [N]"] = self.src_values[name][:, 1]
            pdata_em["Fz [N]"] = self.src_values[name][:, 2]
            pdata_em["Force [N]"] = self.src_values[name]
            self.src_vtk[name] = pdata_em

    def export_forces_to_vtk(self, outdir: Path):
        """Export each interpolated result to a different VTK

        Parameters
        ----------
        outdir : Path
            output directory for the interpolated files

        """
        if self.interpolated_results is None:
            raise ValueError(
                "No interpolated results found. Run interpolate_all() first."
            )

        # dump the vtks
        # check first if vtk where built
        if not self.dest_vtk or not self.src_vtk:
            logging.warning("vtk were not built, building now...")
            self.build_vtk_output()

        for name in self.src_values.keys():
            # Mech
            outfile = Path(outdir, f"{name}_interpolated.vtk")
            self.dest_vtk[name].save(outfile)

            # EM
            outfile_em = Path(outdir, f"{name}_EM.vtk")
            self.src_vtk[name].save(outfile_em)

        logging.info(f"VTK files exported to {outdir}")
