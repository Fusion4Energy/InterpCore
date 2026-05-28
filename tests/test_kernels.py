import numpy as np
from interpcore.kernels import FEM_interpolation_kernel, dist_weight_kernel


class TestFEMInterpolationKernel:
    """Tests for FEM-based interpolation kernel"""

    def test_basic_interpolation(self):
        """Test basic FEM interpolation with valid inputs"""
        # Setup source and destination coordinates
        src_coords = np.array([[0.0, 0.0, 0.0]])
        dest_coords = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        # Vector to interpolate
        vector_to_interp = np.array([[1.0, 2.0, 3.0]])
        vector_interpolated = np.zeros((3, 3))

        # Destination indices and distances
        dest_idx = np.array([0, 1, 2])
        distances = np.array([[1.0], [1.0], [1.0]])
        src_idx = 0

        result = FEM_interpolation_kernel(
            distances,
            dest_idx,
            src_idx,
            dest_coords,
            src_coords,
            vector_to_interp,
            vector_interpolated,
        )

        assert result is True
        # Check that interpolated values are non-zero
        assert np.any(vector_interpolated != 0)

    def test_single_destination_point(self):
        """Test interpolation to a single destination point"""
        src_coords = np.array([[0.0, 0.0, 0.0]])
        dest_coords = np.array([[1.0, 1.0, 1.0]])

        vector_to_interp = np.array([[5.0, 10.0, 15.0]])
        vector_interpolated = np.zeros((1, 3))

        dest_idx = np.array([0])
        distances = np.array([[np.sqrt(3)]])
        src_idx = 0

        result = FEM_interpolation_kernel(
            distances,
            dest_idx,
            src_idx,
            dest_coords,
            src_coords,
            vector_to_interp,
            vector_interpolated,
        )

        assert result is True
        assert vector_interpolated.shape == (1, 3)

    def test_empty_destination_indices(self):
        """Test that empty destination indices returns False"""
        src_coords = np.array([[0.0, 0.0, 0.0]])
        dest_coords = np.array([[1.0, 0.0, 0.0]])

        vector_to_interp = np.array([[1.0, 2.0, 3.0]])
        vector_interpolated = np.zeros((1, 3))

        dest_idx = np.array([], dtype=int)
        distances = np.array([]).reshape(0, 1)
        src_idx = 0

        result = FEM_interpolation_kernel(
            distances,
            dest_idx,
            src_idx,
            dest_coords,
            src_coords,
            vector_to_interp,
            vector_interpolated,
        )

        assert result is False

    def test_multiple_destination_points(self):
        """Test interpolation to multiple destination points"""
        src_coords = np.array([[0.0, 0.0, 0.0]])
        dest_coords = np.array(
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [1.0, 1.0, 0.0]]
        )

        vector_to_interp = np.array([[10.0, 20.0, 30.0]])
        vector_interpolated = np.zeros((4, 3))

        dest_idx = np.array([0, 1, 2, 3])
        distances = np.array([[1.0], [2.0], [3.0], [np.sqrt(2)]])
        src_idx = 0

        result = FEM_interpolation_kernel(
            distances,
            dest_idx,
            src_idx,
            dest_coords,
            src_coords,
            vector_to_interp,
            vector_interpolated,
        )

        assert result is True
        assert vector_interpolated.shape == (4, 3)
        # Each destination point should have received some interpolated value
        assert np.any(vector_interpolated != 0)

    def test_colinear_points(self):
        """Test with colinear destination points (may result in singular matrix)"""
        src_coords = np.array([[0.0, 0.0, 0.0]])
        # All points along x-axis
        dest_coords = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])

        vector_to_interp = np.array([[1.0, 2.0, 3.0]])
        vector_interpolated = np.zeros((3, 3))

        dest_idx = np.array([0, 1, 2])
        distances = np.array([[1.0], [2.0], [3.0]])
        src_idx = 0

        result = FEM_interpolation_kernel(
            distances,
            dest_idx,
            src_idx,
            dest_coords,
            src_coords,
            vector_to_interp,
            vector_interpolated,
        )

        # Should still return True even if matrix might be ill-conditioned
        assert result is True

    def test_vector_accumulation(self):
        """Test that interpolated values are accumulated (+=)"""
        src_coords = np.array([[0.0, 0.0, 0.0]])
        dest_coords = np.array([[1.0, 0.0, 0.0]])

        vector_to_interp = np.array([[1.0, 1.0, 1.0]])
        vector_interpolated = np.array([[5.0, 5.0, 5.0]])  # Pre-existing values

        dest_idx = np.array([0])
        distances = np.array([[1.0]])
        src_idx = 0

        initial_values = vector_interpolated.copy()

        result = FEM_interpolation_kernel(
            distances,
            dest_idx,
            src_idx,
            dest_coords,
            src_coords,
            vector_to_interp,
            vector_interpolated,
        )

        assert result is True
        # Values should be accumulated (added to initial values)
        # At least some component should be greater than initial
        assert np.any(vector_interpolated >= initial_values)


class TestDistWeightKernel:
    """Tests for distance-weighted interpolation kernel"""

    def test_basic_distance_weighting(self):
        """Test basic distance-weighted interpolation"""
        distances = np.array([[1.0], [2.0], [3.0]])
        dest_idx = np.array([0, 1, 2])
        src_index = 0

        vector_to_interp = np.array([[10.0, 20.0, 30.0]])
        vector_interpolated = np.zeros((3, 3))

        result = dist_weight_kernel(
            distances, dest_idx, src_index, vector_to_interp, vector_interpolated
        )

        assert result is True
        # Check that all destination points received some value
        assert np.all(vector_interpolated != 0)

        # Closer points should receive more weight
        # First point (closest) should have largest values
        assert np.all(vector_interpolated[0] >= vector_interpolated[1])
        assert np.all(vector_interpolated[1] >= vector_interpolated[2])

    def test_single_destination(self):
        """Test with a single destination point"""
        distances = np.array([[5.0]])
        dest_idx = np.array([0])
        src_index = 0

        vector_to_interp = np.array([[100.0, 200.0, 300.0]])
        vector_interpolated = np.zeros((1, 3))

        result = dist_weight_kernel(
            distances, dest_idx, src_index, vector_to_interp, vector_interpolated
        )

        assert result is True
        # With single point, it should receive the full value
        np.testing.assert_array_almost_equal(
            vector_interpolated[0], vector_to_interp[0]
        )

    def test_equal_distances(self):
        """Test with equal distances - should distribute equally"""
        distances = np.array([[2.0], [2.0], [2.0]])
        dest_idx = np.array([0, 1, 2])
        src_index = 0

        vector_to_interp = np.array([[30.0, 60.0, 90.0]])
        vector_interpolated = np.zeros((3, 3))

        result = dist_weight_kernel(
            distances, dest_idx, src_index, vector_to_interp, vector_interpolated
        )

        assert result is True
        # All points at equal distance should receive equal weights
        np.testing.assert_array_almost_equal(
            vector_interpolated[0], vector_interpolated[1]
        )
        np.testing.assert_array_almost_equal(
            vector_interpolated[1], vector_interpolated[2]
        )

    def test_weight_normalization(self):
        """Test that weights are properly normalized"""
        distances = np.array([[1.0], [2.0], [4.0]])
        dest_idx = np.array([0, 1, 2])
        src_index = 0

        vector_to_interp = np.array([[10.0, 10.0, 10.0]])
        vector_interpolated = np.zeros((3, 3))

        result = dist_weight_kernel(
            distances, dest_idx, src_index, vector_to_interp, vector_interpolated
        )

        assert result is True
        # Sum of distributed values should equal the original vector
        # (within numerical precision)
        total = np.sum(vector_interpolated, axis=0)
        np.testing.assert_array_almost_equal(total, vector_to_interp[0], decimal=10)

    def test_varying_distances(self):
        """Test with varying distances"""
        distances = np.array([[0.5], [1.5], [3.0], [6.0]])
        dest_idx = np.array([0, 1, 2, 3])
        src_index = 0

        vector_to_interp = np.array([[5.0, 10.0, 15.0]])
        vector_interpolated = np.zeros((4, 3))

        result = dist_weight_kernel(
            distances, dest_idx, src_index, vector_to_interp, vector_interpolated
        )

        assert result is True
        # Verify inverse distance relationship
        # Closest point gets most weight
        assert np.all(vector_interpolated[0] > vector_interpolated[1])
        assert np.all(vector_interpolated[1] > vector_interpolated[2])
        assert np.all(vector_interpolated[2] > vector_interpolated[3])

    def test_vector_accumulation(self):
        """Test that values are accumulated (+=)"""
        distances = np.array([[1.0], [2.0]])
        dest_idx = np.array([0, 1])
        src_index = 0

        vector_to_interp = np.array([[2.0, 4.0, 6.0]])
        vector_interpolated = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

        initial_values = vector_interpolated.copy()

        result = dist_weight_kernel(
            distances, dest_idx, src_index, vector_to_interp, vector_interpolated
        )

        assert result is True
        # All values should be greater than initial (accumulated)
        assert np.all(vector_interpolated > initial_values)

    def test_multiple_vector_components(self):
        """Test that all vector components are properly interpolated"""
        distances = np.array([[1.0], [3.0]])
        dest_idx = np.array([0, 1])
        src_index = 0

        # Different values for each component
        vector_to_interp = np.array([[100.0, 500.0, 1000.0]])
        vector_interpolated = np.zeros((2, 3))

        result = dist_weight_kernel(
            distances, dest_idx, src_index, vector_to_interp, vector_interpolated
        )

        assert result is True
        # Each component should maintain relative proportions
        for i in range(3):
            assert vector_interpolated[0, i] > vector_interpolated[1, i]

        # Check proportions are maintained
        ratio_x = vector_interpolated[0, 0] / vector_interpolated[1, 0]
        ratio_y = vector_interpolated[0, 1] / vector_interpolated[1, 1]
        ratio_z = vector_interpolated[0, 2] / vector_interpolated[1, 2]

        # Ratios should be approximately equal across components
        np.testing.assert_almost_equal(ratio_x, ratio_y, decimal=10)
        np.testing.assert_almost_equal(ratio_y, ratio_z, decimal=10)
