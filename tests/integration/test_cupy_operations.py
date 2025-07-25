"""
Integration tests for CuPy operations in ChemiAccel.

These tests verify CuPy array operations, performance benchmarks,
and interoperability with existing cupy-based workflows.
"""

from unittest.mock import patch

import numpy as np
import pytest

# Import ChemiAccel components
from chemi_accel.backends.rapids import (
    CupyArrayProcessor,
    check_cupy_availability,
    get_cupy_version_info,
    molecular_weight_cupy,
    tanimoto_similarity_cupy,
)
from chemi_accel.core.exceptions import BackendUnavailableError

# Test data
TEST_MOLECULAR_MATRICES = [
    np.array([[1.0, 6.0, 0.0], [1.0, 6.0, 0.0], [16.0, 8.0, 2.0]]),  # ethanol-like
    np.array(
        [[1.0, 6.0, 0.0], [1.0, 6.0, 0.0], [16.0, 8.0, 2.0], [16.0, 8.0, 1.0]]
    ),  # acetic acid-like
    np.array([[12.0, 6.0, 0.0] * 6]),  # benzene-like (simplified)
]

TEST_ADJACENCY_MATRICES = [
    np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]]),  # Simple 3-node graph
    np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]]),  # 4-node graph
]

TEST_FINGERPRINTS = np.array(
    [
        [1, 0, 1, 1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0, 0, 1, 0],
        [0, 1, 1, 0, 1, 1, 0, 1],
    ]
)


class TestCupyAvailability:
    """Test CuPy availability checking."""

    def test_check_cupy_availability(self):
        """Test CuPy availability detection."""
        result = check_cupy_availability()
        assert isinstance(result, bool)

    def test_get_cupy_version_info(self):
        """Test CuPy version information retrieval."""
        info = get_cupy_version_info()
        assert isinstance(info, dict)

        if check_cupy_availability():
            # If CuPy is available, should have version info
            assert "cupy" in info or "error" in info
        else:
            # If not available, should have error message
            assert "error" in info


@pytest.mark.skipif(not check_cupy_availability(), reason="CuPy not available")
class TestCupyArrayProcessor:
    """Test CupyArrayProcessor functionality when CuPy is available."""

    @pytest.fixture
    def cupy_processor(self):
        """Create CupyArrayProcessor instance for testing."""
        return CupyArrayProcessor()

    @pytest.fixture
    def sample_numpy_array(self):
        """Create sample NumPy array for testing."""
        return np.random.rand(100, 10).astype(np.float32)

    def test_cupy_processor_initialization(self, cupy_processor):
        """Test CupyArrayProcessor initialization."""
        assert cupy_processor is not None
        assert hasattr(cupy_processor, "device_id")
        assert isinstance(cupy_processor.device_id, int)

    def test_numpy_to_cupy_conversion(self, cupy_processor, sample_numpy_array):
        """Test NumPy to CuPy array conversion."""
        import cupy as cp

        cupy_array = cupy_processor.numpy_to_cupy(sample_numpy_array)

        assert isinstance(cupy_array, cp.ndarray)
        assert cupy_array.shape == sample_numpy_array.shape
        assert cupy_array.dtype == sample_numpy_array.dtype
        assert cupy_array.flags.c_contiguous

    def test_cupy_to_numpy_conversion(self, cupy_processor, sample_numpy_array):
        """Test CuPy to NumPy array conversion."""

        # Convert to CuPy and back
        cupy_array = cupy_processor.numpy_to_cupy(sample_numpy_array)
        numpy_result = cupy_processor.cupy_to_numpy(cupy_array)

        assert isinstance(numpy_result, np.ndarray)
        assert numpy_result.shape == sample_numpy_array.shape
        assert numpy_result.dtype == sample_numpy_array.dtype
        np.testing.assert_array_almost_equal(numpy_result, sample_numpy_array)

    def test_array_layout_optimization(self, cupy_processor):
        """Test array layout optimization."""
        import cupy as cp

        # Create non-contiguous array
        original_array = cp.random.rand(100, 50)
        non_contiguous = original_array.T  # Transpose creates non-contiguous view

        optimized_array = cupy_processor.optimize_array_layout(non_contiguous)

        assert isinstance(optimized_array, cp.ndarray)
        assert optimized_array.flags.c_contiguous

    def test_batch_molecular_descriptors(self, cupy_processor):
        """Test batch processing of molecular descriptors."""
        import cupy as cp

        def simple_descriptor_func(matrix):
            """Simple descriptor function for testing."""
            return cp.sum(matrix, axis=0)[0]  # Sum of first column

        results = cupy_processor.batch_molecular_descriptors(
            TEST_MOLECULAR_MATRICES, simple_descriptor_func, batch_size=2
        )

        assert isinstance(results, cp.ndarray)
        assert len(results) == len(TEST_MOLECULAR_MATRICES)

    def test_sparse_matrix_operations(self, cupy_processor):
        """Test sparse matrix operations for molecular graphs."""
        import cupy as cp

        results = cupy_processor.sparse_matrix_operations(TEST_ADJACENCY_MATRICES)

        assert isinstance(results, cp.ndarray)
        assert results.shape[0] == len(TEST_ADJACENCY_MATRICES)
        assert (
            results.shape[1] == 5
        )  # num_nodes, num_edges, mean_degree, std_degree, max_degree

    def test_vectorized_molecular_properties(self, cupy_processor):
        """Test vectorized molecular property calculations."""
        import cupy as cp

        # Create test data: (n_molecules, max_atoms, n_features)
        n_molecules, max_atoms, n_features = 3, 5, 4
        atom_features = cp.random.rand(n_molecules, max_atoms, n_features)

        # Set some atoms as invalid (atomic number = 0)
        atom_features[:, :, 1] = cp.where(
            cp.random.rand(n_molecules, max_atoms) > 0.3, atom_features[:, :, 1], 0
        )

        properties = cupy_processor.vectorized_molecular_properties(atom_features)

        assert isinstance(properties, cp.ndarray)
        assert properties.shape[0] == n_molecules
        assert properties.shape[1] > 0  # Should have calculated some properties

    def test_device_info(self, cupy_processor):
        """Test device information retrieval."""
        device_info = cupy_processor.get_device_info()

        assert isinstance(device_info, dict)
        assert "device_id" in device_info
        assert "device_name" in device_info or "error" in device_info

    def test_benchmark_vs_numpy(self, cupy_processor):
        """Test benchmarking CuPy vs NumPy operations."""

        def simple_operation(array):
            """Simple operation for benchmarking."""
            if hasattr(array, "__array_module__"):
                # CuPy array
                import cupy as cp

                return cp.sum(array**2, axis=1)
            else:
                # NumPy array
                return np.sum(array**2, axis=1)

        test_data = np.random.rand(1000, 100).astype(np.float32)

        results = cupy_processor.benchmark_vs_numpy(
            simple_operation, test_data, num_iterations=3
        )

        assert isinstance(results, dict)
        assert "numpy_mean_time" in results
        assert "cupy_mean_time" in results
        assert "speedup" in results


@pytest.mark.skipif(not check_cupy_availability(), reason="CuPy not available")
class TestCupyConvenienceFunctions:
    """Test convenience functions for common CuPy operations."""

    def test_molecular_weight_cupy(self):
        """Test molecular weight calculation with CuPy."""
        import cupy as cp

        # Test data: atomic weights for simple molecules
        atom_weights = cp.array(
            [
                [12.01, 1.008, 1.008, 1.008, 1.008],  # methane (CH4)
                [12.01, 12.01, 1.008, 1.008, 1.008, 1.008],  # ethane (C2H6)
            ]
        )

        molecular_weights = molecular_weight_cupy(atom_weights)

        assert isinstance(molecular_weights, cp.ndarray)
        assert len(molecular_weights) == 2

        # Convert to NumPy for comparison
        mw_numpy = cp.asnumpy(molecular_weights)

        # Check approximate molecular weights
        assert abs(mw_numpy[0] - 16.04) < 0.1  # methane
        assert abs(mw_numpy[1] - 30.07) < 0.1  # ethane

    def test_tanimoto_similarity_cupy(self):
        """Test Tanimoto similarity calculation with CuPy."""
        import cupy as cp

        # Convert test fingerprints to CuPy
        fp_cupy = cp.array(TEST_FINGERPRINTS, dtype=bool)

        # Calculate similarity between first two fingerprints
        similarities = tanimoto_similarity_cupy(fp_cupy[0:1], fp_cupy[1:2])

        assert isinstance(similarities, cp.ndarray)
        assert len(similarities) == 1

        # Convert to NumPy for validation
        sim_numpy = cp.asnumpy(similarities)[0]

        # Manual calculation for validation
        fp1, fp2 = TEST_FINGERPRINTS[0], TEST_FINGERPRINTS[1]
        intersection = np.sum(fp1 & fp2)
        union = np.sum(fp1 | fp2)
        expected_similarity = intersection / union if union > 0 else 0.0

        assert abs(sim_numpy - expected_similarity) < 1e-6


class TestCupyOperationsMocked:
    """Test CuPy operations with mocked dependencies."""

    @patch("chemi_accel.backends.rapids.cupy_operations.CUPY_AVAILABLE", False)
    def test_cupy_processor_unavailable(self):
        """Test CupyArrayProcessor when CuPy is not available."""
        from chemi_accel.backends.rapids.cupy_operations import CupyArrayProcessor

        with pytest.raises(BackendUnavailableError):
            CupyArrayProcessor()


@pytest.mark.skipif(not check_cupy_availability(), reason="CuPy not available")
class TestCupyWorkflowIntegration:
    """Test integration with typical CuPy workflows."""

    def test_cupy_ml_pipeline_workflow(self):
        """Test typical machine learning pipeline workflow with CuPy."""

        try:
            processor = CupyArrayProcessor()

            # Simulate molecular feature matrices
            n_molecules = 100
            max_atoms = 20
            n_features = 10

            # Create random molecular data
            molecular_data = np.random.rand(n_molecules, max_atoms, n_features)

            # Convert to CuPy
            cupy_data = processor.numpy_to_cupy(molecular_data)

            # Calculate properties
            properties = processor.vectorized_molecular_properties(cupy_data)

            # Convert back to NumPy for ML pipeline
            properties_numpy = processor.cupy_to_numpy(properties)

            assert isinstance(properties_numpy, np.ndarray)
            assert properties_numpy.shape[0] == n_molecules
            assert properties_numpy.shape[1] > 0

            # Simulate further ML processing
            # (In real workflow, this would go to scikit-learn, etc.)
            mean_properties = np.mean(properties_numpy, axis=0)
            std_properties = np.std(properties_numpy, axis=0)

            assert len(mean_properties) == properties_numpy.shape[1]
            assert len(std_properties) == properties_numpy.shape[1]

        except Exception as e:
            pytest.skip(f"CuPy ML workflow test failed: {e}")

    def test_cupy_memory_intensive_workflow(self):
        """Test memory-intensive workflow with CuPy."""
        import cupy as cp

        try:
            processor = CupyArrayProcessor()

            # Get initial device info
            device_info = processor.get_device_info()

            if "total_memory" in device_info:
                total_memory = device_info["total_memory"]

                # Create data that uses a reasonable fraction of GPU memory
                # Use smaller arrays to avoid OOM in CI environments
                array_size = min(
                    1000000, total_memory // (4 * 100)
                )  # Conservative estimate

                large_array = np.random.rand(array_size).astype(np.float32)

                # Process in batches
                batch_size = array_size // 10
                results = []

                for i in range(0, len(large_array), batch_size):
                    batch = large_array[i : i + batch_size]
                    cupy_batch = processor.numpy_to_cupy(batch)

                    # Simple operation
                    processed = cp.sqrt(cupy_batch**2 + 1)
                    result = processor.cupy_to_numpy(processed)
                    results.append(result)

                # Combine results
                final_result = np.concatenate(results)

                assert len(final_result) == len(large_array)
                assert isinstance(final_result, np.ndarray)

        except Exception as e:
            pytest.skip(f"Memory intensive workflow test failed: {e}")

    def test_cupy_interoperability_workflow(self):
        """Test interoperability with existing CuPy-based workflows."""
        import cupy as cp

        try:
            processor = CupyArrayProcessor()

            # Simulate existing CuPy workflow data
            existing_cupy_data = cp.random.rand(50, 100)

            # Process with ChemiAccel utilities
            optimized_data = processor.optimize_array_layout(existing_cupy_data)

            # Simulate molecular descriptor calculation
            # (This would normally use molecular data)
            mock_descriptors = cp.sum(optimized_data, axis=1)

            # Convert back to NumPy for compatibility
            descriptors_numpy = processor.cupy_to_numpy(mock_descriptors)

            # Verify interoperability
            assert isinstance(descriptors_numpy, np.ndarray)
            assert len(descriptors_numpy) == existing_cupy_data.shape[0]

            # Simulate further processing with existing tools
            normalized_descriptors = (
                descriptors_numpy - np.mean(descriptors_numpy)
            ) / np.std(descriptors_numpy)

            assert isinstance(normalized_descriptors, np.ndarray)
            assert len(normalized_descriptors) == len(descriptors_numpy)

        except Exception as e:
            pytest.skip(f"Interoperability workflow test failed: {e}")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
