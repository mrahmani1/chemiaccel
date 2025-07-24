"""
Unit tests for graph-based CUDA kernels for molecular descriptor calculations.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Import without mocking first to get the classes
from src.chemi_accel.backends.cuda.graph_kernels import (
    GraphKernelManager,
    GraphDescriptorCalculator,
    get_graph_kernel_manager
)
from src.chemi_accel.core.exceptions import GPUMemoryError


class TestGraphKernelManager:
    """Test GraphKernelManager class."""
    
    @pytest.fixture
    def mock_cupy_graph(self):
        """Mock cupy for graph kernel testing."""
        with patch('src.chemi_accel.backends.cuda.graph_kernels.CUDA_AVAILABLE', True), \
             patch('src.chemi_accel.backends.cuda.graph_kernels.cp') as mock_cp, \
             patch('src.chemi_accel.backends.cuda.graph_kernels.RawKernel') as mock_raw_kernel:
            
            # Mock cupy arrays
            mock_cp.zeros.return_value = Mock()
            mock_cp.float32 = np.float32
            mock_cp.int32 = np.int32
            mock_cp.ndarray = np.ndarray
            
            # Mock RawKernel
            mock_kernel = Mock()
            mock_raw_kernel.return_value = mock_kernel
            
            yield mock_cp, mock_raw_kernel, mock_kernel
    
    @pytest.fixture
    def mock_memory_manager(self):
        """Mock memory manager for testing."""
        return Mock()
    
    @pytest.fixture
    def graph_kernel_manager(self, mock_cupy_graph, mock_memory_manager):
        """Create graph kernel manager for testing."""
        mock_cp, mock_raw_kernel, mock_kernel = mock_cupy_graph
        return GraphKernelManager(mock_memory_manager)
    
    def test_graph_kernel_manager_initialization(self, graph_kernel_manager, mock_cupy_graph):
        """Test graph kernel manager initialization."""
        mock_cp, mock_raw_kernel, mock_kernel = mock_cupy_graph
        
        assert graph_kernel_manager.memory_manager is not None
        assert len(graph_kernel_manager._kernels) == 5  # 5 graph kernels compiled
        
        # Verify all expected kernels are compiled
        expected_kernels = [
            'wiener_index', 'zagreb_indices', 'substructure_matching',
            'tanimoto_similarity', 'randic_index'
        ]
        for kernel_name in expected_kernels:
            assert kernel_name in graph_kernel_manager._kernels
    
    def test_calculate_wiener_index(self, graph_kernel_manager, mock_cupy_graph):
        """Test Wiener index calculation."""
        mock_cp, mock_raw_kernel, mock_kernel = mock_cupy_graph
        
        # Mock input data
        csr_indices = Mock()
        csr_indptr = Mock()
        molecule_boundaries = Mock()
        molecule_csr_boundaries = Mock()
        molecule_boundaries.__len__ = Mock(return_value=2)  # 2 molecules
        
        # Mock result array
        mock_result = Mock()
        mock_cp.zeros.return_value = mock_result
        
        # Call method
        result = graph_kernel_manager.calculate_wiener_index(
            csr_indices, csr_indptr, molecule_boundaries, molecule_csr_boundaries
        )
        
        # Verify kernel was called
        mock_kernel.assert_called_once()
        assert result == mock_result
        
        # Verify zeros was called with correct parameters
        mock_cp.zeros.assert_called_with(2, dtype=mock_cp.float32)
    
    def test_calculate_zagreb_indices(self, graph_kernel_manager, mock_cupy_graph):
        """Test Zagreb indices calculation."""
        mock_cp, mock_raw_kernel, mock_kernel = mock_cupy_graph
        
        # Mock input data
        csr_indices = Mock()
        csr_indptr = Mock()
        molecule_boundaries = Mock()
        molecule_csr_boundaries = Mock()
        molecule_boundaries.__len__ = Mock(return_value=3)  # 3 molecules
        
        # Mock result arrays
        mock_zagreb1 = Mock()
        mock_zagreb2 = Mock()
        mock_cp.zeros.side_effect = [mock_zagreb1, mock_zagreb2]
        
        # Call method
        zagreb1, zagreb2 = graph_kernel_manager.calculate_zagreb_indices(
            csr_indices, csr_indptr, molecule_boundaries, molecule_csr_boundaries
        )
        
        # Verify kernel was called
        mock_kernel.assert_called_once()
        assert zagreb1 == mock_zagreb1
        assert zagreb2 == mock_zagreb2
        
        # Verify zeros was called twice
        assert mock_cp.zeros.call_count == 2
    
    def test_calculate_randic_index(self, graph_kernel_manager, mock_cupy_graph):
        """Test Randic index calculation."""
        mock_cp, mock_raw_kernel, mock_kernel = mock_cupy_graph
        
        # Mock input data
        csr_indices = Mock()
        csr_indptr = Mock()
        molecule_boundaries = Mock()
        molecule_csr_boundaries = Mock()
        molecule_boundaries.__len__ = Mock(return_value=1)  # 1 molecule
        
        # Mock result array
        mock_result = Mock()
        mock_cp.zeros.return_value = mock_result
        
        # Call method
        result = graph_kernel_manager.calculate_randic_index(
            csr_indices, csr_indptr, molecule_boundaries, molecule_csr_boundaries
        )
        
        # Verify kernel was called
        mock_kernel.assert_called_once()
        assert result == mock_result
    
    def test_substructure_matching(self, graph_kernel_manager, mock_cupy_graph):
        """Test substructure matching."""
        mock_cp, mock_raw_kernel, mock_kernel = mock_cupy_graph
        
        # Mock input data
        mol_csr_indices = Mock()
        mol_csr_indptr = Mock()
        mol_atom_types = Mock()
        molecule_boundaries = Mock()
        molecule_csr_boundaries = Mock()
        pattern_csr_indices = Mock()
        pattern_csr_indptr = Mock()
        pattern_atom_types = Mock()
        
        molecule_boundaries.__len__ = Mock(return_value=2)  # 2 molecules
        pattern_atom_types.__len__ = Mock(return_value=3)   # 3-atom pattern
        
        # Mock result array
        mock_result = Mock()
        mock_cp.zeros.return_value = mock_result
        
        # Call method
        result = graph_kernel_manager.substructure_matching(
            mol_csr_indices, mol_csr_indptr, mol_atom_types,
            molecule_boundaries, molecule_csr_boundaries,
            pattern_csr_indices, pattern_csr_indptr, pattern_atom_types
        )
        
        # Verify kernel was called
        mock_kernel.assert_called_once()
        assert result == mock_result
        
        # Verify zeros was called with int32 type
        mock_cp.zeros.assert_called_with(2, dtype=mock_cp.int32)
    
    def test_calculate_tanimoto_similarity(self, graph_kernel_manager, mock_cupy_graph):
        """Test Tanimoto similarity calculation."""
        mock_cp, mock_raw_kernel, mock_kernel = mock_cupy_graph
        
        # Mock input data
        fingerprint1 = Mock()
        fingerprint2 = Mock()
        molecule_boundaries = Mock()
        fingerprint_size = 1024
        
        molecule_boundaries.__len__ = Mock(return_value=5)  # 5 molecules
        
        # Mock result array
        mock_result = Mock()
        mock_cp.zeros.return_value = mock_result
        
        # Call method
        result = graph_kernel_manager.calculate_tanimoto_similarity(
            fingerprint1, fingerprint2, molecule_boundaries, fingerprint_size
        )
        
        # Verify kernel was called
        mock_kernel.assert_called_once()
        assert result == mock_result
    
    def test_kernel_compilation_error(self, mock_cupy_graph, mock_memory_manager):
        """Test handling of kernel compilation errors."""
        mock_cp, mock_raw_kernel, mock_kernel = mock_cupy_graph
        
        # Mock RawKernel to raise an exception
        mock_raw_kernel.side_effect = Exception("Graph kernel compilation failed")
        
        with pytest.raises(Exception, match="Graph kernel compilation failed"):
            GraphKernelManager(mock_memory_manager)


class TestGraphDescriptorCalculator:
    """Test GraphDescriptorCalculator class."""
    
    @pytest.fixture
    def mock_cupy_graph_calc(self):
        """Mock cupy for graph calculator testing."""
        with patch('src.chemi_accel.backends.cuda.graph_kernels.CUDA_AVAILABLE', True), \
             patch('src.chemi_accel.backends.cuda.graph_kernels.cp') as mock_cp, \
             patch('src.chemi_accel.backends.cuda.graph_kernels.RawKernel'):
            
            # Mock cupy arrays
            mock_cp.zeros.return_value = Mock()
            mock_cp.float32 = np.float32
            mock_cp.int32 = np.int32
            mock_cp.ndarray = np.ndarray
            
            yield mock_cp
    
    @pytest.fixture
    def mock_graph_kernel_manager(self):
        """Mock graph kernel manager for testing."""
        kernel_manager = Mock()
        
        # Mock kernel methods to return mock arrays
        kernel_manager.calculate_wiener_index.return_value = Mock()
        kernel_manager.calculate_zagreb_indices.return_value = (Mock(), Mock())
        kernel_manager.calculate_randic_index.return_value = Mock()
        kernel_manager.substructure_matching.return_value = Mock()
        kernel_manager.calculate_tanimoto_similarity.return_value = Mock()
        
        return kernel_manager
    
    @pytest.fixture
    def graph_descriptor_calculator(self, mock_cupy_graph_calc, mock_graph_kernel_manager):
        """Create graph descriptor calculator for testing."""
        with patch('src.chemi_accel.backends.cuda.graph_kernels.GraphKernelManager') as mock_gkm_class, \
             patch('src.chemi_accel.backends.cuda.graph_kernels.get_memory_manager') as mock_get_mm:
            mock_gkm_class.return_value = mock_graph_kernel_manager
            mock_get_mm.return_value = Mock()
            calc = GraphDescriptorCalculator()
            calc.kernel_manager = mock_graph_kernel_manager
            return calc
    
    def test_calculator_initialization(self, mock_cupy_graph_calc):
        """Test graph descriptor calculator initialization."""
        with patch('src.chemi_accel.backends.cuda.graph_kernels.GraphKernelManager'), \
             patch('src.chemi_accel.backends.cuda.graph_kernels.get_memory_manager') as mock_get_mm:
            mock_get_mm.return_value = Mock()
            calc = GraphDescriptorCalculator()
            assert calc.memory_manager is not None
            assert calc.kernel_manager is not None
    
    def test_calculate_topological_indices_complete(self, graph_descriptor_calculator, mock_graph_kernel_manager):
        """Test calculating topological indices with complete data."""
        # Mock complete graph data
        graph_data = {
            'csr_indices': Mock(),
            'csr_indptr': Mock(),
            'molecule_boundaries': Mock(),
            'molecule_csr_boundaries': Mock()
        }
        
        # Call method
        results = graph_descriptor_calculator.calculate_topological_indices(graph_data)
        
        # Verify all indices were calculated
        expected_indices = ['wiener_index', 'zagreb_m1', 'zagreb_m2', 'randic_index']
        
        for index in expected_indices:
            assert index in results
        
        # Verify kernel methods were called
        mock_graph_kernel_manager.calculate_wiener_index.assert_called_once()
        mock_graph_kernel_manager.calculate_zagreb_indices.assert_called_once()
        mock_graph_kernel_manager.calculate_randic_index.assert_called_once()
    
    def test_calculate_topological_indices_missing_data(self, graph_descriptor_calculator, mock_graph_kernel_manager):
        """Test calculating topological indices with missing data."""
        # Mock incomplete graph data
        graph_data = {
            'csr_indices': Mock(),
            # Missing other required keys
        }
        
        # Call method
        results = graph_descriptor_calculator.calculate_topological_indices(graph_data)
        
        # Should return empty results
        assert len(results) == 0
        
        # Verify no kernel methods were called
        mock_graph_kernel_manager.calculate_wiener_index.assert_not_called()
        mock_graph_kernel_manager.calculate_zagreb_indices.assert_not_called()
        mock_graph_kernel_manager.calculate_randic_index.assert_not_called()
    
    def test_calculate_molecular_similarity_tanimoto(self, graph_descriptor_calculator, mock_graph_kernel_manager):
        """Test molecular similarity calculation with Tanimoto coefficient."""
        # Mock input data
        fingerprints1 = Mock()
        fingerprints2 = Mock()
        molecule_boundaries = Mock()
        fingerprint_size = 1024
        
        # Call method
        result = graph_descriptor_calculator.calculate_molecular_similarity(
            fingerprints1, fingerprints2, molecule_boundaries, fingerprint_size, 'tanimoto'
        )
        
        # Verify kernel method was called
        mock_graph_kernel_manager.calculate_tanimoto_similarity.assert_called_once_with(
            fingerprints1, fingerprints2, molecule_boundaries, fingerprint_size
        )
        
        assert result is not None
    
    def test_calculate_molecular_similarity_unsupported(self, graph_descriptor_calculator, mock_graph_kernel_manager):
        """Test molecular similarity calculation with unsupported similarity type."""
        # Mock input data
        fingerprints1 = Mock()
        fingerprints2 = Mock()
        molecule_boundaries = Mock()
        fingerprint_size = 1024
        
        # Call method with unsupported similarity type
        with pytest.raises(ValueError, match="Unsupported similarity type"):
            graph_descriptor_calculator.calculate_molecular_similarity(
                fingerprints1, fingerprints2, molecule_boundaries, fingerprint_size, 'unsupported'
            )
    
    def test_perform_substructure_search_complete(self, graph_descriptor_calculator, mock_graph_kernel_manager):
        """Test substructure search with complete data."""
        # Mock complete molecular and pattern data
        molecular_data = {
            'csr_indices': Mock(),
            'csr_indptr': Mock(),
            'atom_types': Mock(),
            'molecule_boundaries': Mock(),
            'molecule_csr_boundaries': Mock()
        }
        
        pattern_data = {
            'csr_indices': Mock(),
            'csr_indptr': Mock(),
            'atom_types': Mock()
        }
        
        # Call method
        result = graph_descriptor_calculator.perform_substructure_search(molecular_data, pattern_data)
        
        # Verify kernel method was called
        mock_graph_kernel_manager.substructure_matching.assert_called_once()
        assert result is not None
    
    def test_perform_substructure_search_missing_molecular_data(self, graph_descriptor_calculator, mock_graph_kernel_manager):
        """Test substructure search with missing molecular data."""
        # Mock incomplete molecular data
        molecular_data = {
            'csr_indices': Mock(),
            # Missing other required keys
        }
        
        pattern_data = {
            'csr_indices': Mock(),
            'csr_indptr': Mock(),
            'atom_types': Mock()
        }
        
        # Call method - should raise ValueError
        with pytest.raises(ValueError, match="Missing required molecular data keys"):
            graph_descriptor_calculator.perform_substructure_search(molecular_data, pattern_data)
    
    def test_perform_substructure_search_missing_pattern_data(self, graph_descriptor_calculator, mock_graph_kernel_manager):
        """Test substructure search with missing pattern data."""
        # Mock complete molecular data
        molecular_data = {
            'csr_indices': Mock(),
            'csr_indptr': Mock(),
            'atom_types': Mock(),
            'molecule_boundaries': Mock(),
            'molecule_csr_boundaries': Mock()
        }
        
        # Mock incomplete pattern data
        pattern_data = {
            'csr_indices': Mock(),
            # Missing other required keys
        }
        
        # Call method - should raise ValueError
        with pytest.raises(ValueError, match="Missing required pattern data keys"):
            graph_descriptor_calculator.perform_substructure_search(molecular_data, pattern_data)


class TestGraphKernelIntegration:
    """Integration tests for graph kernel functionality."""
    
    @pytest.fixture
    def mock_cupy_integration(self):
        """Mock cupy for integration testing."""
        with patch('src.chemi_accel.backends.cuda.graph_kernels.CUDA_AVAILABLE', True), \
             patch('src.chemi_accel.backends.cuda.graph_kernels.cp') as mock_cp, \
             patch('src.chemi_accel.backends.cuda.graph_kernels.RawKernel') as mock_raw_kernel:
            
            # Mock cupy arrays and functions
            mock_cp.zeros.return_value = Mock()
            mock_cp.float32 = np.float32
            mock_cp.int32 = np.int32
            mock_cp.ndarray = np.ndarray
            
            # Mock RawKernel
            mock_kernel = Mock()
            mock_raw_kernel.return_value = mock_kernel
            
            yield mock_cp, mock_raw_kernel, mock_kernel
    
    def test_wiener_index_kernel_parameters(self, mock_cupy_integration):
        """Test Wiener index kernel is called with correct parameters."""
        mock_cp, mock_raw_kernel, mock_kernel = mock_cupy_integration
        
        # Create kernel manager
        with patch('src.chemi_accel.backends.cuda.graph_kernels.get_memory_manager'):
            gkm = GraphKernelManager()
        
        # Mock input data
        csr_indices = Mock()
        csr_indptr = Mock()
        molecule_boundaries = Mock()
        molecule_csr_boundaries = Mock()
        molecule_boundaries.__len__ = Mock(return_value=3)  # 3 molecules
        
        # Mock result
        mock_result = Mock()
        mock_cp.zeros.return_value = mock_result
        
        # Call method
        result = gkm.calculate_wiener_index(
            csr_indices, csr_indptr, molecule_boundaries, molecule_csr_boundaries, max_atoms=50
        )
        
        # Verify kernel call parameters
        mock_kernel.assert_called_once()
        call_args = mock_kernel.call_args
        
        # Check grid and block sizes
        grid_size, block_size = call_args[0][:2]
        assert block_size == (256,)  # Block size should be 256
        
        # Check kernel arguments
        kernel_args = call_args[0][2]
        assert len(kernel_args) == 7  # Should have 7 arguments
        assert kernel_args[0] == csr_indices
        assert kernel_args[1] == csr_indptr
        assert kernel_args[2] == molecule_boundaries
        assert kernel_args[3] == molecule_csr_boundaries
        assert kernel_args[4] == mock_result
        assert kernel_args[5] == 3  # num_molecules
        assert kernel_args[6] == 50  # max_atoms
        
        # Check shared memory allocation
        shared_mem = call_args[1]['shared_mem']
        expected_shared_mem = 50 * 50 * 4  # max_atoms * max_atoms * 4 bytes
        assert shared_mem == expected_shared_mem
    
    def test_zagreb_indices_kernel_parameters(self, mock_cupy_integration):
        """Test Zagreb indices kernel is called with correct parameters."""
        mock_cp, mock_raw_kernel, mock_kernel = mock_cupy_integration
        
        with patch('src.chemi_accel.backends.cuda.graph_kernels.get_memory_manager'):
            gkm = GraphKernelManager()
        
        # Mock input data
        csr_indices = Mock()
        csr_indptr = Mock()
        molecule_boundaries = Mock()
        molecule_csr_boundaries = Mock()
        molecule_boundaries.__len__ = Mock(return_value=2)  # 2 molecules
        
        # Mock results
        mock_zagreb1 = Mock()
        mock_zagreb2 = Mock()
        mock_cp.zeros.side_effect = [mock_zagreb1, mock_zagreb2]
        
        # Call method
        zagreb1, zagreb2 = gkm.calculate_zagreb_indices(
            csr_indices, csr_indptr, molecule_boundaries, molecule_csr_boundaries
        )
        
        # Verify kernel call parameters
        mock_kernel.assert_called_once()
        call_args = mock_kernel.call_args
        
        # Check kernel arguments
        kernel_args = call_args[0][2]
        assert len(kernel_args) == 7  # Should have 7 arguments
        assert kernel_args[0] == csr_indices
        assert kernel_args[1] == csr_indptr
        assert kernel_args[2] == molecule_boundaries
        assert kernel_args[3] == molecule_csr_boundaries
        assert kernel_args[4] == mock_zagreb1
        assert kernel_args[5] == mock_zagreb2
        assert kernel_args[6] == 2  # num_molecules


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_cuda_not_available_graph_kernel_manager(self):
        """Test behavior when CUDA is not available for graph kernel manager."""
        with patch('src.chemi_accel.backends.cuda.graph_kernels.CUDA_AVAILABLE', False):
            with pytest.raises(RuntimeError, match="CUDA not available"):
                GraphKernelManager()
    
    def test_cuda_not_available_graph_descriptor_calculator(self):
        """Test behavior when CUDA is not available for graph descriptor calculator."""
        with patch('src.chemi_accel.backends.cuda.graph_kernels.CUDA_AVAILABLE', False):
            with pytest.raises(RuntimeError, match="CUDA not available"):
                GraphDescriptorCalculator()


class TestGlobalFunctions:
    """Test global functions."""
    
    @patch('src.chemi_accel.backends.cuda.graph_kernels.CUDA_AVAILABLE', True)
    def test_get_graph_kernel_manager(self):
        """Test getting global graph kernel manager."""
        # Reset global instance
        import src.chemi_accel.backends.cuda.graph_kernels as graph_kernels_module
        graph_kernels_module._graph_kernel_manager = None
        
        with patch('src.chemi_accel.backends.cuda.graph_kernels.GraphKernelManager') as mock_gkm:
            mock_instance = Mock()
            mock_gkm.return_value = mock_instance
            
            manager1 = get_graph_kernel_manager()
            manager2 = get_graph_kernel_manager()
            
            assert manager1 is manager2  # Should be same instance
            mock_gkm.assert_called_once()  # Should only be created once


if __name__ == "__main__":
    pytest.main([__file__])