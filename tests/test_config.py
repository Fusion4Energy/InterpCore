from interpcore.config import (
    INTERPOLATED_LOAD_TYPE,
    QUERY_TYPE,
    INTERPOLATION_KERNEL,
    InterpolationConfig,
)
from interpcore.errors import ConfigurationError
import pytest


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
            interpolated_load=INTERPOLATED_LOAD_TYPE.EM_FORCE,
        )

        assert config.method == QUERY_TYPE.RADIUS
        assert config.param == 0.5
        assert config.max_distance == 1.0
        assert config.coincidence_tolerance == 1e-6
        assert config.kernel == INTERPOLATION_KERNEL.DISTANCE_WEIGHTED
        assert config.multithread is True

    def test_kernel_incompatible_configuration_error(self):
        """Test incompatible kernel configuration raises ConfigurationError"""
        with pytest.raises(ConfigurationError):
            InterpolationConfig(
                method=QUERY_TYPE.K,
                param=1,
                max_distance=1.0,
                coincidence_tolerance=1e-6,
                kernel=INTERPOLATION_KERNEL.AVERAGE,  # Incompatible with EM_FORCE
                multithread=False,
                interpolated_load=INTERPOLATED_LOAD_TYPE.EM_FORCE,
            )

        with pytest.raises(ConfigurationError):
            InterpolationConfig(
                method=QUERY_TYPE.K,
                param=1,
                max_distance=1.0,
                coincidence_tolerance=1e-6,
                kernel=INTERPOLATION_KERNEL.DISTANCE_WEIGHTED,  # Incompatible with EM_FORCE
                multithread=False,
                interpolated_load=INTERPOLATED_LOAD_TYPE.HEAT_FLUX,
            )
