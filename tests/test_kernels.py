import numpy as np
import pytest
from sklearn.metrics.pairwise import euclidean_distances
from interpcore.kernels import (
    _dist_weight_kernel,
    _average_kernel,
    _closest_source_kernel,
    _FEM_interpolation_kernel,
    interpolate_block,
)
from interpcore.config import (
    InterpolationConfig,
    QUERY_TYPE,
    INTERPOLATED_LOAD_TYPE,
    INTERPOLATION_KERNEL,
)
from interpcore.errors import InterpolationError


class TestKernels:
    def test_dist_weigth_kernel(self):
        """Test that the distance weighted kernel produces expected results for a simple case"""
        # Simple case: 1 source distributing to 2 destinations
        src_point = np.array([[0.0, 0.0, 0.0]])
        dest_points = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

        dest_idx = np.array([0, 1])
        src_index = 0

        # Compute distances using euclidean_distances
        distances = euclidean_distances(dest_points, src_point)

        vector_to_interp = np.array([[10.0, 20.0, 30.0]])
        vector_interpolated = np.zeros((2, 3))

        result = _dist_weight_kernel(
            distances, dest_idx, src_index, vector_to_interp, vector_interpolated
        )

        assert result is True
        # Closer point should get more weight
        assert np.all(vector_interpolated[0] > vector_interpolated[1])
        # Total should sum to original value (weights normalized)
        total = np.sum(vector_interpolated, axis=0)
        np.testing.assert_array_almost_equal(total, vector_to_interp[0])

    def test_average_kernel(self):
        """Test that the average kernel produces expected results with scalar values"""
        # Simple case: 1 destination averaging 2 sources with scalar values
        neighbours_idx = np.array([0, 1])
        dest_index = 0

        # Use scalar values (1 component - like heat flux)
        vector_to_interp = np.array([[100.0], [200.0]])
        vector_interpolated = np.zeros((1, 1))

        result = _average_kernel(
            neighbours_idx, dest_index, vector_to_interp, vector_interpolated
        )

        assert result is True
        # Should be average: (100 + 200) / 2 = 150
        expected = 150.0
        np.testing.assert_almost_equal(vector_interpolated[0, 0], expected)

    def test_average_weighted_kernel(self):
        """Test that the weighted average kernel gives more weight to closer source points"""
        dest_point = np.array([[0.0, 0.0, 0.0]])
        # Source 0 is at distance 1, source 1 is at distance 2
        src_points = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

        neighbours_idx = np.array([0, 1])
        dest_index = 0

        distances = euclidean_distances(src_points, dest_point)  # shape (2, 1)

        vector_to_interp = np.array([[100.0], [200.0]])
        vector_interpolated = np.zeros((1, 1))

        result = _average_kernel(
            neighbours_idx, dest_index, vector_to_interp, vector_interpolated, distances=distances
        )

        assert result is True
        # weights: w0 = 1/1 = 2, w1 = 1/2 = 1  (after normalisation: 2/3, 1/3)
        # expected = 100 * 2/3 + 200 * 1/3 = 133.333...
        expected = 100.0 * (2.0 / 3.0) + 200.0 * (1.0 / 3.0)
        np.testing.assert_almost_equal(vector_interpolated[0, 0], expected)
        # Result must be closer to the nearer source (100) than the plain average (150)
        assert vector_interpolated[0, 0] < 150.0

    def test_closest_source_kernel(self):
        """Test that the closest source kernel produces expected results for a simple case"""
        # Simple case: 1 destination picking closest of 3 sources
        dest_point = np.array([[0.5, 0.0, 0.0]])
        src_points = np.array(
            [
                [0.0, 2.0, 0.0],  # Distance 2.06
                [0.0, 1.0, 0.0],  # Distance 1.12 (closest)
                [0.0, 3.0, 0.0],  # Distance 3.04
            ]
        )

        neighbours_idx = np.array([0, 1, 2])
        dest_index = 0

        # Compute distances using euclidean_distances
        distances = euclidean_distances(src_points, dest_point)

        vector_to_interp = np.array(
            [
                [10.0, 20.0, 30.0],
                [40.0, 50.0, 60.0],  # This should be selected (closest)
                [70.0, 80.0, 90.0],
            ]
        )
        vector_interpolated = np.zeros((1, 3))

        result = _closest_source_kernel(
            distances, neighbours_idx, dest_index, vector_to_interp, vector_interpolated
        )

        assert result is True
        # Should get value from index 1 (closest)
        expected = np.array([40.0, 50.0, 60.0])
        np.testing.assert_array_almost_equal(vector_interpolated[0], expected)

    def test_FEM_interpolation_kernel(self):
        """Test that the FEM interpolation kernel produces expected results for a simple case"""
        # Simple case: 1 source to 3 destinations forming a triangle
        src_coords = np.array([[0.0, 0.0, 0.0]])
        dest_coords = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        vector_to_interp = np.array([[10.0, 20.0, 30.0]])
        vector_interpolated = np.zeros((3, 3))

        dest_idx = np.array([0, 1, 2])
        src_idx = 0

        # Compute distances using euclidean_distances
        distances = euclidean_distances(dest_coords, src_coords[src_idx].reshape(1, -1))

        result = _FEM_interpolation_kernel(
            distances,
            dest_idx,
            src_idx,
            dest_coords,
            src_coords,
            vector_to_interp,
            vector_interpolated,
        )

        assert result is True
        # FEM should distribute forces to all destinations
        assert np.any(vector_interpolated != 0)

    def test_interpolate_block_no_neighbours(self):
        """Test that interpolate_block handles the case where no neighbors are found"""
        # Source point with no neighbors within max_distance (source-to-dest mode)
        src_coords = np.array([[100.0, 100.0, 100.0]])
        dest_coords = np.array([[0.0, 0.0, 0.0]])
        src_values = np.array([[50.0, 60.0, 70.0]])

        idx_query = np.array([np.array([0])], dtype=object)

        config = InterpolationConfig(
            kernel=INTERPOLATION_KERNEL.DISTANCE_WEIGHTED,
            max_distance=5.0,  # Too small to reach destination
            coincidence_tolerance=0.01,
            method=QUERY_TYPE.K,
            param=1,
            multithread=False,
            interpolated_load=INTERPOLATED_LOAD_TYPE.EM_FORCE,
        )

        interpolated, unmapped = interpolate_block(
            chunked_coords=src_coords,
            neighbours_coords=dest_coords,
            idx_query=idx_query,
            chunk_idx=slice(0, 1),
            src_values=src_values,
            config=config,
        )

        # Source too far away - should be unmapped
        np.testing.assert_array_almost_equal(unmapped[0], src_values[0])
        assert np.all(interpolated == 0)

    def test_interpolate_block_unmapped_source(self):
        """Test that interpolate_block correctly accumulates unmapped source values in source-to-destination mode"""
        # Two sources, one too far away
        src_coords = np.array([[0.0, 0.0, 0.0], [100.0, 100.0, 100.0]])
        dest_coords = np.array([[1.0, 0.0, 0.0]])
        src_values = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])

        idx_query = np.array([np.array([0]), np.array([0])], dtype=object)

        config = InterpolationConfig(
            kernel=INTERPOLATION_KERNEL.DISTANCE_WEIGHTED,
            max_distance=5.0,
            coincidence_tolerance=0.01,
            method=QUERY_TYPE.K,
            param=1,
            multithread=False,
            interpolated_load=INTERPOLATED_LOAD_TYPE.EM_FORCE,
        )

        interpolated, unmapped = interpolate_block(
            chunked_coords=src_coords,
            neighbours_coords=dest_coords,
            idx_query=idx_query,
            chunk_idx=slice(0, 2),
            src_values=src_values,
            config=config,
        )

        # First source should map, second should be unmapped
        assert np.any(interpolated > 0)
        # Unmapped should contain the second source value
        np.testing.assert_array_almost_equal(unmapped[0], src_values[1])

    def test_interpolate_block_unmapped_destination(self):
        """Test that interpolate_block raises an error when a destination point has no neighbors in destination-to-source mode"""
        # Destination point far from all sources (dest-to-source mode)
        src_coords = np.array([[0.0, 0.0, 0.0]])
        dest_coords = np.array([[100.0, 100.0, 100.0]])
        src_values = np.array([[10.0, 20.0, 30.0]])

        idx_query = np.array([np.array([0])], dtype=object)

        config = InterpolationConfig(
            kernel=INTERPOLATION_KERNEL.AVERAGE,  # Dest-to-source mode
            max_distance=5.0,  # Too small
            coincidence_tolerance=0.01,
            method=QUERY_TYPE.K,
            param=1,
            multithread=False,
            interpolated_load=INTERPOLATED_LOAD_TYPE.EM_FORCE,
        )

        # Should raise InterpolationError
        with pytest.raises(InterpolationError):
            interpolate_block(
                chunked_coords=dest_coords,
                neighbours_coords=src_coords,
                idx_query=idx_query,
                chunk_idx=slice(0, 1),
                src_values=src_values,
                config=config,
            )

    def test_interpolate_block_accept_no_neighbor(self):
        """Test that interpolate_block sets value to zero when accept_no_neighbor is True and destination has no neighbors"""
        # Destination point far from all sources (dest-to-source mode)
        src_coords = np.array([[0.0, 0.0, 0.0]])
        dest_coords = np.array([[100.0, 100.0, 100.0], [0.5, 0.0, 0.0]])
        src_values = np.array([[10.0, 20.0, 30.0]])

        idx_query = np.array([np.array([0]), np.array([0])], dtype=object)

        config = InterpolationConfig(
            kernel=INTERPOLATION_KERNEL.AVERAGE,  # Dest-to-source mode
            max_distance=5.0,  # Too small for first dest, ok for second
            coincidence_tolerance=0.01,
            method=QUERY_TYPE.K,
            param=1,
            multithread=False,
            interpolated_load=INTERPOLATED_LOAD_TYPE.EM_FORCE,
            accept_no_neighbor=True,  # Accept destinations with no neighbors
        )

        interpolated, unmapped = interpolate_block(
            chunked_coords=dest_coords,
            neighbours_coords=src_coords,
            idx_query=idx_query,
            chunk_idx=slice(0, 2),
            src_values=src_values,
            config=config,
        )

        # First destination (too far) should have zero values
        np.testing.assert_array_equal(interpolated[0], [0.0, 0.0, 0.0])
        # Second destination (close enough) should have interpolated values
        assert np.any(interpolated[1] > 0)
        # Nothing should be unmapped in dest-to-source mode
        assert np.all(unmapped == 0)

    def test_interpolate_block_coincident_node(self):
        """Test that coincident nodes are handled correctly in source-to-dest mode"""
        # Source point almost exactly coincides with a destination point
        src_coords = np.array([[1.0, 0.0, 0.0]])
        dest_coords = np.array([[0.0, 0.0, 0.0], [1.0001, 0.0, 0.0], [2.0, 0.0, 0.0]])
        src_values = np.array([[100.0, 200.0, 300.0]])

        idx_query = np.array([np.array([1])], dtype=object)

        config = InterpolationConfig(
            kernel=INTERPOLATION_KERNEL.DISTANCE_WEIGHTED,
            max_distance=5.0,
            coincidence_tolerance=0.01,  # 1cm tolerance - source within tolerance of dest[1]
            method=QUERY_TYPE.K,
            param=1,
            multithread=False,
            interpolated_load=INTERPOLATED_LOAD_TYPE.EM_FORCE,
        )

        interpolated, unmapped = interpolate_block(
            chunked_coords=src_coords,
            neighbours_coords=dest_coords,
            idx_query=idx_query,
            chunk_idx=slice(0, 1),
            src_values=src_values,
            config=config,
        )

        # Source coincides with dest[1], should get the exact source value
        np.testing.assert_array_almost_equal(interpolated[1], src_values[0])
        # Other destinations should remain zero
        assert interpolated[0, 0] == 0.0
        assert interpolated[2, 0] == 0.0
        # Nothing should be unmapped
        assert np.all(unmapped == 0)
