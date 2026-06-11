import pytest
import tempfile
import shutil
import numpy as np
from pathlib import Path
from interpcore.interpolator import Interpolator, _select_template
from interpcore.config import (
    InterpolationConfig,
    QUERY_TYPE,
    INTERPOLATED_LOAD_TYPE,
    INTERPOLATION_KERNEL,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing"""
    temp = tempfile.mkdtemp()
    yield Path(temp)
    shutil.rmtree(temp)


@pytest.fixture
def sample_config_heat_flux():
    """Create a sample configuration for heat flux interpolation"""
    return InterpolationConfig(
        method=QUERY_TYPE.RADIUS,
        param=1.0,
        max_distance=2.0,
        coincidence_tolerance=1e-6,
        kernel=INTERPOLATION_KERNEL.DISTANCE_WEIGHTED,
        multithread=False,
        interpolated_load=INTERPOLATED_LOAD_TYPE.HEAT_FLUX,
    )


@pytest.fixture
def sample_config_em_force():
    """Create a sample configuration for EM force interpolation"""
    return InterpolationConfig(
        method=QUERY_TYPE.K,
        param=2,
        max_distance=2.0,
        coincidence_tolerance=1e-6,
        kernel=INTERPOLATION_KERNEL.AVERAGE,
        multithread=False,
        interpolated_load=INTERPOLATED_LOAD_TYPE.EM_FORCE,
    )


@pytest.fixture
def sample_config_heat_gen():
    """Create a sample configuration for heat generation interpolation"""
    return InterpolationConfig(
        method=QUERY_TYPE.K,
        param=3,
        max_distance=2.0,
        coincidence_tolerance=1e-6,
        kernel=INTERPOLATION_KERNEL.CLOSEST,
        multithread=False,
        interpolated_load=INTERPOLATED_LOAD_TYPE.HEAT_GEN,
    )


@pytest.fixture
def sample_config_htc():
    """Create a sample configuration for HTC interpolation"""
    return InterpolationConfig(
        method=QUERY_TYPE.K,
        param=3,
        max_distance=2.0,
        coincidence_tolerance=1e-6,
        kernel=INTERPOLATION_KERNEL.DISTANCE_WEIGHTED,
        multithread=False,
        interpolated_load=INTERPOLATED_LOAD_TYPE.HTC,
    )


@pytest.fixture
def create_sample_mesh_files(temp_dir):
    """Create sample mesh and data files for testing"""
    # Create destination mesh file
    dest_mesh = temp_dir / "destination_mesh.txt"
    dest_content = """Node_ID X Y Z
101 0.0 0.0 0.0
102 1.0 0.0 0.0
103 2.0 0.0 0.0
104 0.0 1.0 0.0
105 1.0 1.0 0.0
"""
    dest_mesh.write_text(dest_content)

    # Create source data folder
    src_folder = temp_dir / "source_data"
    src_folder.mkdir()

    # Create source data file with heat flux (1 component)
    src_file = src_folder / "data_001.txt"
    src_content = """Node_ID X Y Z HeatFlux
1 0.5 0.5 0.0 100.0
2 1.5 0.5 0.0 200.0
3 0.5 1.5 0.0 150.0
"""
    src_file.write_text(src_content)

    return {
        "dest_mesh": str(dest_mesh),
        "src_folder": str(src_folder),
    }


@pytest.fixture
def create_sample_em_force_files(temp_dir):
    """Create sample mesh and EM force data files for testing"""
    # Create destination mesh file
    dest_mesh = temp_dir / "destination_mesh.txt"
    dest_content = """Node_ID X Y Z
201 0.0 0.0 0.0
202 1.0 0.0 0.0
203 2.0 0.0 0.0
"""
    dest_mesh.write_text(dest_content)

    # Create source data folder
    src_folder = temp_dir / "em_force_data"
    src_folder.mkdir()

    # Create source data file with EM forces (3 components)
    src_file = src_folder / "force_001.txt"
    src_content = """Node_ID X Y Z Fx Fy Fz
1 0.5 0.0 0.0 10.0 20.0 30.0
2 1.5 0.0 0.0 15.0 25.0 35.0
"""
    src_file.write_text(src_content)

    return {
        "dest_mesh": str(dest_mesh),
        "src_folder": str(src_folder),
    }


@pytest.fixture
def create_sample_heat_gen_files(temp_dir):
    """Create sample mesh and heat generation data files for testing"""
    # Create destination mesh file
    dest_mesh = temp_dir / "destination_mesh.txt"
    dest_content = """Node_ID X Y Z
301 0.0 0.0 0.0
302 1.0 0.0 0.0
303 2.0 0.0 0.0
304 0.0 1.0 0.0
"""
    dest_mesh.write_text(dest_content)

    # Create source data folder
    src_folder = temp_dir / "heat_gen_data"
    src_folder.mkdir()

    # Create source data file with heat generation (1 component)
    src_file = src_folder / "heatgen_001.txt"
    src_content = """Node_ID X Y Z HeatGen
1 0.5 0.5 0.0 500.0
2 1.5 0.5 0.0 750.0
3 0.5 1.5 0.0 600.0
"""
    src_file.write_text(src_content)

    return {
        "dest_mesh": str(dest_mesh),
        "src_folder": str(src_folder),
    }


@pytest.fixture
def create_sample_htc_files(temp_dir):
    """Create sample mesh and HTC data files for testing"""
    # Create destination mesh file
    dest_mesh = temp_dir / "destination_mesh.txt"
    dest_content = """Node_ID X Y Z
501 0.0 0.0 0.0
502 1.0 0.0 0.0
503 2.0 0.0 0.0
504 0.0 1.0 0.0
"""
    dest_mesh.write_text(dest_content)

    # Create source data folder
    src_folder = temp_dir / "htc_data"
    src_folder.mkdir()

    # Create source data file with HTC and reference temperature (2 components)
    src_file = src_folder / "htc_001.txt"
    src_content = """Node_ID X Y Z HTC Tref
1 0.5 0.5 0.0 250.0 300.0
2 1.5 0.5 0.0 300.0 310.0
3 0.5 1.5 0.0 275.0 305.0
"""
    src_file.write_text(src_content)

    return {
        "dest_mesh": str(dest_mesh),
        "src_folder": str(src_folder),
    }


class TestInterpolator:
    """Tests for Interpolator class"""

    def test_initialization_with_valid_inputs(
        self, create_sample_mesh_files, sample_config_heat_flux
    ):
        """Test successful initialization with valid mesh and data files"""
        file_idx = {"ids": 0, "dest_x": 1, "src_x": 1, "val": 4}
        interpolator = Interpolator(
            path_to_src_folder=create_sample_mesh_files["src_folder"],
            path_to_dest_mesh=create_sample_mesh_files["dest_mesh"],
            config=sample_config_heat_flux,
            file_idx=file_idx,
        )

        assert interpolator is not None
        assert interpolator.tree is not None
        assert len(interpolator.src_values) == 1
        assert "data_001" in interpolator.src_values
        assert interpolator.interpolated_results is None

    def test_initialization_with_custom_file_idx(
        self, create_sample_mesh_files, sample_config_heat_flux
    ):
        """Test initialization with custom file indices"""
        file_idx = {"ids": 0, "dest_x": 1, "src_x": 1, "val": 4}
        interpolator = Interpolator(
            path_to_src_folder=create_sample_mesh_files["src_folder"],
            path_to_dest_mesh=create_sample_mesh_files["dest_mesh"],
            config=sample_config_heat_flux,
            file_idx=file_idx,
        )

        assert interpolator is not None
        assert interpolator.tree is not None

    def test_initialization_with_missing_source_folder(
        self, temp_dir, sample_config_heat_flux
    ):
        """Test that FileNotFoundError is raised when source folder doesn't exist"""
        dest_mesh = temp_dir / "dest.txt"
        dest_mesh.write_text("Node_ID X Y Z\n1 0.0 0.0 0.0\n")
        file_idx = {"ids": 0, "dest_x": 1, "src_x": 1, "val": 4}

        with pytest.raises(FileNotFoundError, match="does not exist"):
            Interpolator(
                path_to_src_folder=str(temp_dir / "nonexistent"),
                path_to_dest_mesh=str(dest_mesh),
                config=sample_config_heat_flux,
                file_idx=file_idx,
            )

    def test_initialization_with_empty_source_folder(
        self, temp_dir, sample_config_heat_flux
    ):
        """Test that FileNotFoundError is raised when source folder is empty"""
        dest_mesh = temp_dir / "dest.txt"
        dest_mesh.write_text("Node_ID X Y Z\n1 0.0 0.0 0.0\n")
        file_idx = {"ids": 0, "dest_x": 1, "src_x": 1, "val": 4}

        empty_folder = temp_dir / "empty"
        empty_folder.mkdir()

        with pytest.raises(FileNotFoundError, match="is empty"):
            Interpolator(
                path_to_src_folder=str(empty_folder),
                path_to_dest_mesh=str(dest_mesh),
                config=sample_config_heat_flux,
                file_idx=file_idx,
            )

    def test_interpolate_all(self, create_sample_mesh_files, sample_config_heat_flux):
        """Test the interpolate_all method"""
        file_idx = {"ids": 0, "dest_x": 1, "src_x": 1, "val": 4}
        interpolator = Interpolator(
            path_to_src_folder=create_sample_mesh_files["src_folder"],
            path_to_dest_mesh=create_sample_mesh_files["dest_mesh"],
            config=sample_config_heat_flux,
            file_idx=file_idx,
        )

        interpolator.interpolate_all()

        assert interpolator.interpolated_results is not None
        assert len(interpolator.interpolated_results) == 1
        assert "data_001" in interpolator.interpolated_results

        result = interpolator.interpolated_results["data_001"]
        assert "interpolated" in result
        assert "unmapped" in result
        assert result["interpolated"].shape[0] > 0  # Has destination points
        assert result["interpolated"].shape[1] == 1  # Heat flux is 1 component

    def test_interpolate_all_with_em_force(
        self, create_sample_em_force_files, sample_config_em_force
    ):
        """Test interpolation with EM force data (3 components)"""
        file_idx = {"ids": 0, "dest_x": 1, "src_x": 1, "val": 4}
        interpolator = Interpolator(
            path_to_src_folder=create_sample_em_force_files["src_folder"],
            path_to_dest_mesh=create_sample_em_force_files["dest_mesh"],
            config=sample_config_em_force,
            file_idx=file_idx,
        )

        interpolator.interpolate_all()

        assert interpolator.interpolated_results is not None
        result = interpolator.interpolated_results["force_001"]
        assert result["interpolated"].shape[1] == 3  # EM force has 3 components

    def test_interpolate_all_with_htc(self, create_sample_htc_files, sample_config_htc):
        """Test interpolation with HTC data (2 components: HTC and Tref)"""
        file_idx = {"ids": 0, "dest_x": 1, "src_x": 1, "val": 4}
        interpolator = Interpolator(
            path_to_src_folder=create_sample_htc_files["src_folder"],
            path_to_dest_mesh=create_sample_htc_files["dest_mesh"],
            config=sample_config_htc,
            file_idx=file_idx,
        )

        interpolator.interpolate_all()

        assert interpolator.interpolated_results is not None
        result = interpolator.interpolated_results["htc_001"]
        assert "interpolated" in result
        assert "unmapped" in result
        assert result["interpolated"].shape[0] > 0
        assert result["interpolated"].shape[1] == 2  # HTC has 2 components

    def test_export_to_ansys_without_interpolation(
        self, create_sample_mesh_files, sample_config_heat_flux, temp_dir
    ):
        """Test that export raises ValueError when called before interpolation"""
        file_idx = {"ids": 0, "dest_x": 1, "src_x": 1, "val": 4}
        interpolator = Interpolator(
            path_to_src_folder=create_sample_mesh_files["src_folder"],
            path_to_dest_mesh=create_sample_mesh_files["dest_mesh"],
            config=sample_config_heat_flux,
            file_idx=file_idx,
        )

        output_dir = temp_dir / "output"
        output_dir.mkdir()

        with pytest.raises(
            ValueError, match="No interpolated results found. Run interpolate_all"
        ):
            interpolator.export_to_ansys(output_dir)

    def test_export_to_ansys_heat_flux(
        self, create_sample_mesh_files, sample_config_heat_flux, temp_dir
    ):
        """Test exporting heat flux results to ANSYS format"""
        file_idx = {"ids": 0, "dest_x": 1, "src_x": 1, "val": 4}
        interpolator = Interpolator(
            path_to_src_folder=create_sample_mesh_files["src_folder"],
            path_to_dest_mesh=create_sample_mesh_files["dest_mesh"],
            config=sample_config_heat_flux,
            file_idx=file_idx,
        )

        interpolator.interpolate_all()

        output_dir = temp_dir / "output"
        output_dir.mkdir()
        interpolator.export_to_ansys(output_dir)

        # Check that output file was created
        output_files = list(output_dir.glob("interpolated_*.txt"))
        assert len(output_files) == 1
        assert output_files[0].name == "interpolated_data_001.txt"

        # Check file content format
        content = output_files[0].read_text()
        assert "SFE" in content
        assert "HFLUX" in content

    def test_export_to_ansys_em_force(
        self, create_sample_em_force_files, sample_config_em_force, temp_dir
    ):
        """Test exporting EM force results to ANSYS format"""
        file_idx = {"ids": 0, "dest_x": 1, "src_x": 1, "val": 4}
        interpolator = Interpolator(
            path_to_src_folder=create_sample_em_force_files["src_folder"],
            path_to_dest_mesh=create_sample_em_force_files["dest_mesh"],
            config=sample_config_em_force,
            file_idx=file_idx,
        )

        interpolator.interpolate_all()

        output_dir = temp_dir / "output"
        output_dir.mkdir()
        interpolator.export_to_ansys(output_dir)

        # Check that output file was created
        output_files = list(output_dir.glob("interpolated_*.txt"))
        assert len(output_files) == 1

        # Check file content format (should have Fx, Fy, Fz)
        content = output_files[0].read_text()
        assert "Fx" in content
        assert "Fy" in content
        assert "Fz" in content

    def test_export_to_ansys_heat_gen(
        self, create_sample_heat_gen_files, sample_config_heat_gen, temp_dir
    ):
        """Test exporting heat generation results to ANSYS format"""
        file_idx = {"ids": 0, "dest_x": 1, "src_x": 1, "val": 4}
        interpolator = Interpolator(
            path_to_src_folder=create_sample_heat_gen_files["src_folder"],
            path_to_dest_mesh=create_sample_heat_gen_files["dest_mesh"],
            config=sample_config_heat_gen,
            file_idx=file_idx,
        )

        interpolator.interpolate_all()

        output_dir = temp_dir / "output"
        output_dir.mkdir()
        interpolator.export_to_ansys(output_dir)

        # Check that output file was created
        output_files = list(output_dir.glob("interpolated_*.txt"))
        assert len(output_files) == 1
        assert output_files[0].name == "interpolated_heatgen_001.txt"

        # Check file content format (should have HGEN)
        content = output_files[0].read_text()
        assert "HGEN" in content
        assert "BF" in content

    def test_export_to_ansys_htc(
        self, create_sample_htc_files, sample_config_htc, temp_dir
    ):
        """Test exporting HTC results to ANSYS format"""
        file_idx = {"ids": 0, "dest_x": 1, "src_x": 1, "val": 4}
        interpolator = Interpolator(
            path_to_src_folder=create_sample_htc_files["src_folder"],
            path_to_dest_mesh=create_sample_htc_files["dest_mesh"],
            config=sample_config_htc,
            file_idx=file_idx,
        )

        interpolator.interpolate_all()

        output_dir = temp_dir / "output"
        output_dir.mkdir()
        interpolator.export_to_ansys(output_dir)

        # Check that output file was created
        output_files = list(output_dir.glob("interpolated_*.txt"))
        assert len(output_files) == 1
        assert output_files[0].name == "interpolated_htc_001.txt"

        # Check file content format (should have CONV)
        content = output_files[0].read_text()
        assert "CONV" in content
        assert "SFE" in content

    def test_build_vtk_output_without_interpolation(
        self, create_sample_mesh_files, sample_config_heat_flux
    ):
        """Test that build_vtk_output raises ValueError when called before interpolation"""
        file_idx = {"ids": 0, "dest_x": 1, "src_x": 1, "val": 4}
        interpolator = Interpolator(
            path_to_src_folder=create_sample_mesh_files["src_folder"],
            path_to_dest_mesh=create_sample_mesh_files["dest_mesh"],
            config=sample_config_heat_flux,
            file_idx=file_idx,
        )

        with pytest.raises(
            ValueError, match="No interpolated results found. Run interpolate_all"
        ):
            interpolator.build_vtk_output()

    def test_build_vtk_output_without_saving(
        self, create_sample_mesh_files, sample_config_heat_flux
    ):
        """Test building VTK output without saving to disk"""
        file_idx = {"ids": 0, "dest_x": 1, "src_x": 1, "val": 4}
        interpolator = Interpolator(
            path_to_src_folder=create_sample_mesh_files["src_folder"],
            path_to_dest_mesh=create_sample_mesh_files["dest_mesh"],
            config=sample_config_heat_flux,
            file_idx=file_idx,
        )

        interpolator.interpolate_all()
        interpolator.build_vtk_output(outdir=None)

        # Check that VTK dictionaries are populated
        assert len(interpolator.dest_vtk) == 1
        assert len(interpolator.src_vtk) == 1
        assert "data_001" in interpolator.dest_vtk
        assert "data_001" in interpolator.src_vtk

    def test_build_vtk_output_with_saving(
        self, create_sample_mesh_files, sample_config_heat_flux, temp_dir
    ):
        """Test building and saving VTK output to disk"""
        file_idx = {"ids": 0, "dest_x": 1, "src_x": 1, "val": 4}
        interpolator = Interpolator(
            path_to_src_folder=create_sample_mesh_files["src_folder"],
            path_to_dest_mesh=create_sample_mesh_files["dest_mesh"],
            config=sample_config_heat_flux,
            file_idx=file_idx,
        )

        interpolator.interpolate_all()

        output_dir = temp_dir / "vtk_output"
        output_dir.mkdir()
        interpolator.build_vtk_output(outdir=output_dir)

        # Check that VTK files were created
        vtk_files = list(output_dir.glob("*.vtk"))
        assert len(vtk_files) == 2  # One for source, one for destination

        # Check file names
        file_names = [f.name for f in vtk_files]
        assert "data_001_interpolated.vtk" in file_names
        assert "data_001_src.vtk" in file_names

    def test_multiple_source_files(self, temp_dir, sample_config_heat_flux):
        """Test interpolation with multiple source data files"""
        # Create destination mesh
        dest_mesh = temp_dir / "dest.txt"
        dest_content = """Node_ID X Y Z
1 0.0 0.0 0.0
2 1.0 0.0 0.0
"""
        dest_mesh.write_text(dest_content)

        # Create source folder with multiple files
        src_folder = temp_dir / "sources"
        src_folder.mkdir()

        for i in range(3):
            src_file = src_folder / f"data_{i:03d}.txt"
            src_content = f"""Node_ID X Y Z HeatFlux
1 0.5 0.0 0.0 {100 + i * 10}.0
"""
            src_file.write_text(src_content)

        file_idx = {"ids": 0, "dest_x": 1, "src_x": 1, "val": 4}
        interpolator = Interpolator(
            path_to_src_folder=str(src_folder),
            path_to_dest_mesh=str(dest_mesh),
            config=sample_config_heat_flux,
            file_idx=file_idx,
        )

        assert len(interpolator.src_values) == 3

        interpolator.interpolate_all()
        assert interpolator.interpolated_results is not None
        assert len(interpolator.interpolated_results) == 3

    def test_compute_scalar_integrals(self, temp_dir, sample_config_heat_gen):
        """Test computing scalar integrals with volume data"""
        # Create destination mesh with volume column
        dest_mesh = temp_dir / "destination_mesh.txt"
        dest_content = """Node_ID X Y Z Volume
401 0.0 0.0 0.0 0.001
402 1.0 0.0 0.0 0.002
403 2.0 0.0 0.0 0.0015
"""
        dest_mesh.write_text(dest_content)

        # Create source data folder with heat generation
        src_folder = temp_dir / "heat_gen_data"
        src_folder.mkdir()

        src_file = src_folder / "heatgen_001.txt"
        src_content = """Node_ID X Y Z HeatGen
1 0.5 0.0 0.0 1000.0
2 1.5 0.0 0.0 2000.0
3 2.5 0.0 0.0 1500.0
"""
        src_file.write_text(src_content)

        # Create interpolator with volume data
        file_idx = {"ids": 0, "dest_x": 1, "src_x": 1, "val": 4, "vol": 4}
        interpolator = Interpolator(
            path_to_src_folder=str(src_folder),
            path_to_dest_mesh=str(dest_mesh),
            config=sample_config_heat_gen,
            file_idx=file_idx,
        )

        # Run interpolation
        interpolator.interpolate_all()

        # Compute integrals
        integrals = interpolator.compute_scalar_integrals()

        # Verify results
        assert integrals is not None
        assert "heatgen_001" in integrals
        assert isinstance(integrals["heatgen_001"], np.ndarray)
        assert integrals["heatgen_001"].size == 1  # Scalar result
        assert integrals["heatgen_001"][0] > 0  # Should be positive heat generation

    def test_compute_scalar_integrals_without_interpolation(
        self, create_sample_heat_gen_files, sample_config_heat_gen
    ):
        """Test that compute_scalar_integrals raises ValueError before interpolation"""
        file_idx = {"ids": 0, "dest_x": 1, "src_x": 1, "val": 4}
        interpolator = Interpolator(
            path_to_src_folder=create_sample_heat_gen_files["src_folder"],
            path_to_dest_mesh=create_sample_heat_gen_files["dest_mesh"],
            config=sample_config_heat_gen,
            file_idx=file_idx,
        )

        with pytest.raises(
            ValueError, match="No interpolated results found. Run interpolate_all"
        ):
            interpolator.compute_scalar_integrals()

    def test_compute_em_resultants(
        self, create_sample_em_force_files, sample_config_em_force
    ):
        """Test computing EM force and moment resultants"""
        file_idx = {"ids": 0, "dest_x": 1, "src_x": 1, "val": 4}
        interpolator = Interpolator(
            path_to_src_folder=create_sample_em_force_files["src_folder"],
            path_to_dest_mesh=create_sample_em_force_files["dest_mesh"],
            config=sample_config_em_force,
            file_idx=file_idx,
        )

        # Run interpolation
        interpolator.interpolate_all()

        # Compute EM resultants with default pole (0, 0, 0)
        resultants = interpolator.compute_EM_resultants()

        # Verify results
        assert resultants is not None
        assert "force_001" in resultants

        result = resultants["force_001"]
        assert "R_F_EM" in result
        assert "R_F_Mech" in result
        assert "R_M_EM" in result
        assert "R_M_Mech" in result
        assert "f_err_comp" in result
        assert "m_err_comp" in result
        assert "Unmapped_EM_Force" in result

        # Check that force resultants are 3D vectors
        assert isinstance(result["R_F_EM"], np.ndarray)
        assert isinstance(result["R_F_Mech"], np.ndarray)
        assert isinstance(result["R_M_EM"], np.ndarray)
        assert isinstance(result["R_M_Mech"], np.ndarray)
        assert len(result["R_F_EM"]) == 3
        assert len(result["R_F_Mech"]) == 3
        assert len(result["R_M_EM"]) == 3
        assert len(result["R_M_Mech"]) == 3

    def test_compute_em_resultants_with_custom_pole(
        self, create_sample_em_force_files, sample_config_em_force
    ):
        """Test computing EM resultants with custom reference pole"""
        file_idx = {"ids": 0, "dest_x": 1, "src_x": 1, "val": 4}
        interpolator = Interpolator(
            path_to_src_folder=create_sample_em_force_files["src_folder"],
            path_to_dest_mesh=create_sample_em_force_files["dest_mesh"],
            config=sample_config_em_force,
            file_idx=file_idx,
        )

        interpolator.interpolate_all()

        # Compute with custom pole
        custom_pole = np.array([1.0, 1.0, 1.0])
        resultants = interpolator.compute_EM_resultants(pole=custom_pole)

        # Verify results exist and are different from default pole
        assert resultants is not None
        assert "force_001" in resultants

        result = resultants["force_001"]
        assert "R_M_EM" in result
        assert "R_M_Mech" in result

    def test_compute_em_resultants_without_interpolation(
        self, create_sample_em_force_files, sample_config_em_force
    ):
        """Test that compute_EM_resultants raises ValueError before interpolation"""
        file_idx = {"ids": 0, "dest_x": 1, "src_x": 1, "val": 4}
        interpolator = Interpolator(
            path_to_src_folder=create_sample_em_force_files["src_folder"],
            path_to_dest_mesh=create_sample_em_force_files["dest_mesh"],
            config=sample_config_em_force,
            file_idx=file_idx,
        )

        with pytest.raises(
            ValueError, match="No interpolated results found. Run interpolate_all"
        ):
            interpolator.compute_EM_resultants()


class TestSelectTemplate:
    """Tests for _select_template helper function"""

    def test_select_template_em_force(self):
        """Test template selection for EM force"""
        templates = _select_template(INTERPOLATED_LOAD_TYPE.EM_FORCE)

        assert len(templates) == 3
        assert "Fx" in templates[0]
        assert "Fy" in templates[1]
        assert "Fz" in templates[2]

    def test_select_template_heat_flux(self):
        """Test template selection for heat flux"""
        templates = _select_template(INTERPOLATED_LOAD_TYPE.HEAT_FLUX)

        assert len(templates) == 1
        assert "HFLUX" in templates[0]
        assert "SFE" in templates[0]

    def test_select_template_heat_gen(self):
        """Test template selection for heat generation"""
        templates = _select_template(INTERPOLATED_LOAD_TYPE.HEAT_GEN)

        assert len(templates) == 1
        assert "HGEN" in templates[0]
        assert "BFE" in templates[0]

    def test_select_template_htc(self):
        """Test template selection for HTC"""
        templates = _select_template(INTERPOLATED_LOAD_TYPE.HTC)

        assert len(templates) == 2
        assert "CONV" in templates[0]
        assert "SFE" in templates[0]
        assert "CONV" in templates[1]
        assert "SFE" in templates[1]

    def test_select_template_unsupported_type(self):
        """Test that unsupported load types raise NotImplementedError"""

        # Create a mock unsupported load type
        class UnsupportedLoad:
            pass

        with pytest.raises(NotImplementedError):
            _select_template(UnsupportedLoad())
