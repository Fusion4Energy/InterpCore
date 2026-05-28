import pytest
from interpcore.config import (
    QUERY_TYPE,
    INTERPOLATION_KERNEL,
    InterpolationConfig,
)


class TestInterpolationConfig:
    """Tests for InterpolationConfig dataclass"""

    def test_create_config_with_radius_method(self):
        """Test creating config with RADIUS query method"""
        config = InterpolationConfig(
            method=QUERY_TYPE.RADIUS,
            param=0.5,
            max_distance=1.0,
            coincidence_tolerance=1e-6,
            kernel=INTERPOLATION_KERNEL.DISTANCE_WEIGHTED,
            multithread=True,
        )

        assert config.method == QUERY_TYPE.RADIUS
        assert config.param == 0.5
        assert config.max_distance == 1.0
        assert config.coincidence_tolerance == 1e-6
        assert config.kernel == INTERPOLATION_KERNEL.DISTANCE_WEIGHTED
        assert config.multithread is True

    def test_create_config_with_k_method_integer_param(self):
        """Test creating config with K query method and integer parameter"""
        config = InterpolationConfig(
            method=QUERY_TYPE.K,
            param=5,
            max_distance=2.0,
            coincidence_tolerance=1e-8,
            kernel=INTERPOLATION_KERNEL.FEM,
            multithread=False,
        )

        assert config.method == QUERY_TYPE.K
        assert config.param == 5
        assert isinstance(config.param, int)
        assert config.max_distance == 2.0
        assert config.coincidence_tolerance == 1e-8
        assert config.kernel == INTERPOLATION_KERNEL.FEM
        assert config.multithread is False

    def test_create_config_with_k_method_float_param_converts_to_int(self):
        """Test that float parameter is converted to int for K query method"""
        config = InterpolationConfig(
            method=QUERY_TYPE.K,
            param=5.0,
            max_distance=1.5,
            coincidence_tolerance=1e-7,
            kernel=INTERPOLATION_KERNEL.DISTANCE_WEIGHTED,
            multithread=True,
        )

        assert config.method == QUERY_TYPE.K
        assert config.param == 5
        assert isinstance(config.param, int)

    def test_create_config_with_k_method_convertible_float_param(self):
        """Test that convertible float parameter (e.g., 8.7) is converted to int"""
        config = InterpolationConfig(
            method=QUERY_TYPE.K,
            param=8.7,
            max_distance=1.0,
            coincidence_tolerance=1e-6,
            kernel=INTERPOLATION_KERNEL.FEM,
            multithread=False,
        )

        assert config.method == QUERY_TYPE.K
        assert config.param == 8
        assert isinstance(config.param, int)

    def test_create_config_with_k_method_invalid_param_raises_error(self):
        """Test that invalid parameter for K query method raises ValueError"""
        with pytest.raises(
            ValueError, match="Parameter for K query must be an integer"
        ):
            InterpolationConfig(
                method=QUERY_TYPE.K,
                param="invalid",  # type: ignore[arg-type]
                max_distance=1.0,
                coincidence_tolerance=1e-6,
                kernel=INTERPOLATION_KERNEL.DISTANCE_WEIGHTED,
                multithread=True,
            )

    def test_create_config_with_radius_method_allows_float_param(self):
        """Test that RADIUS method allows float parameter without conversion"""
        config = InterpolationConfig(
            method=QUERY_TYPE.RADIUS,
            param=2.5,
            max_distance=5.0,
            coincidence_tolerance=1e-5,
            kernel=INTERPOLATION_KERNEL.FEM,
            multithread=True,
        )

        assert config.method == QUERY_TYPE.RADIUS
        assert config.param == 2.5
        assert isinstance(config.param, float)

    def test_config_with_different_kernels(self):
        """Test creating configs with different interpolation kernels"""
        config_distance = InterpolationConfig(
            method=QUERY_TYPE.RADIUS,
            param=1.0,
            max_distance=2.0,
            coincidence_tolerance=1e-6,
            kernel=INTERPOLATION_KERNEL.DISTANCE_WEIGHTED,
            multithread=True,
        )

        config_fem = InterpolationConfig(
            method=QUERY_TYPE.K,
            param=10,
            max_distance=3.0,
            coincidence_tolerance=1e-7,
            kernel=INTERPOLATION_KERNEL.FEM,
            multithread=False,
        )

        assert config_distance.kernel == INTERPOLATION_KERNEL.DISTANCE_WEIGHTED
        assert config_fem.kernel == INTERPOLATION_KERNEL.FEM

    def test_config_multithread_options(self):
        """Test creating configs with different multithread settings"""
        config_single = InterpolationConfig(
            method=QUERY_TYPE.RADIUS,
            param=1.0,
            max_distance=2.0,
            coincidence_tolerance=1e-6,
            kernel=INTERPOLATION_KERNEL.DISTANCE_WEIGHTED,
            multithread=False,
        )

        config_multi = InterpolationConfig(
            method=QUERY_TYPE.RADIUS,
            param=1.0,
            max_distance=2.0,
            coincidence_tolerance=1e-6,
            kernel=INTERPOLATION_KERNEL.DISTANCE_WEIGHTED,
            multithread=True,
        )

        assert config_single.multithread is False
        assert config_multi.multithread is True

    def test_config_with_various_tolerances(self):
        """Test creating configs with different tolerance values"""
        tolerances = [1e-4, 1e-6, 1e-8, 1e-10]

        for tol in tolerances:
            config = InterpolationConfig(
                method=QUERY_TYPE.RADIUS,
                param=1.0,
                max_distance=2.0,
                coincidence_tolerance=tol,
                kernel=INTERPOLATION_KERNEL.DISTANCE_WEIGHTED,
                multithread=True,
            )
            assert config.coincidence_tolerance == tol

    def test_config_with_various_max_distances(self):
        """Test creating configs with different max_distance values"""
        distances = [0.1, 1.0, 5.0, 10.0, 100.0]

        for dist in distances:
            config = InterpolationConfig(
                method=QUERY_TYPE.RADIUS,
                param=dist / 2,
                max_distance=dist,
                coincidence_tolerance=1e-6,
                kernel=INTERPOLATION_KERNEL.DISTANCE_WEIGHTED,
                multithread=True,
            )
            assert config.max_distance == dist
