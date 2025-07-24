"""
Unit tests for CUDA kernels for molecular descriptor calculations.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Import without mocking first to get the classes
from src.chemi_accel.backends.cuda.kernels import (
    CUDAKernelManager,
    BasicDescriptorCalculator,
    get_kernel_manager
)
from src.chemi_accel.core.exceptions import GPUMemoryError


class TestCUDAKernelManager:
    """Test CUDAKernelManager class."""
    
    @pytest.fixture
    def mock_cupy_kernels(self):
        """Mock cupy for kernel testing."""
        with patch('src.chemi_accel.backends.cuda.kernels.CUDA_AVAILABLE', True), \
             patch('src.chemi_accel.backends.cuda.kernels.cp') as mock_cp, \
             patch('src.chemi_accel.backends.cuda.kernels.RawKernel') as mock_raw_kernel:
            
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
    def kernel_manager(self, mock_cupy_kernels, mock_memory_manager):
        """Create kernel manager for testing."""
        mock_cp, mock_raw_kernel, mock_kernel = mock_cupy_kernels
        return CUDAKernelManager(mock_memory_manager)
    
    def test_kernel_manager_initialization(self, kernel_manager, mock_cupy_kernels):
        """Test kernel manager initialization."""
        mock_cp, mock_raw_kernel, mock_kernel = mock_cupy_kernels
        
        assert kernel_manager.memory_manager is not None
        assert len(kernel_manager._kernels) == 6  # 6 kernels compiled
        
        # Verify all expected kernels are compiled
        expected_kernels = [
            'molecular_weight', 'atom_counts', 'bond_counts',
            'tpsa', 'rotatable_bonds', 'aromatic_rings'
        ]
        for kernel_name in expected_kernels:
            assert kernel_name in kernel_manager._kernels
    
    def test_calculate_molecular_weights(self, kernel_manager, mock_cupy_kernels):
        """Test molecular weight calculation."""
        mock_cp, mock_raw_kernel, mock_kernel = mock_cupy_kernels
        
        # Mock input data
        atom_weights = Mock()
        molecule_boundaries = Mock()
        molecule_boundaries.__len__ = Mock(return_value=3)  # 3 molecules
        
        # Mock result array
        mock_result = Mock()
        mock_cp.zeros.return_value = mock_result
        
        # Call method
        result = kernel_manager.calculate_molecular_weights(atom_weights, molecule_boundaries)
        
        # Verify kernel was called
        mock_kernel.assert_called_once()
        assert result == mock_result
        
        # Verify zeros was called with correct parameters
        mock_cp.zeros.assert_called_with(3, dtype=mock_cp.float32)
    
    def test_calculate_atom_counts(self, kernel_manager, mock_cupy_kernels):
        """Test atom count calculation."""
        mock_cp, mock_raw_kernel, mock_kernel = mock_cupy_kernels
        
        # Mock input data
        atomic_numbers = Mock()
        molecule_boundaries = Mock()
        molecule_boundaries.__len__ = Mock(return_value=2)  # 2 molecules
        
        # Mock result arrays
        mock_heavy = Mock()
        mock_total = Mock()
        mock_cp.zeros.side_effect = [mock_heavy, mock_total]
        
        # Call method
        heavy_counts, total_counts = kernel_manager.calculate_atom_counts(
            atomic_numbers, molecule_boundaries
        )
        
        # Verify kernel was called
        mock_kernel.assert_called_once()
        assert heavy_counts == mock_heavy
        assert total_counts == mock_total
        
        # Verify zeros was called twice
        assert mock_cp.zeros.call_count == 2
    
    def test_calculate_bond_counts(self, kernel_manager, mock_cupy_kernels):
        """Test bond count calculation."""
        mock_cp, mock_raw_kernel, mock_kernel = mock_cupy_kernels
        
        # Mock input data
        bond_types = Mock()
        molecule_bond_boundaries = Mock()
        molecule_bond_boundaries.__len__ = Mock(return_value=2)  # 2 molecules
        
        # Mock result arrays
        mock_results = [Mock() for _ in range(4)]  # 4 different bond count arrays
        mock_cp.zeros.side_effect = mock_results
        
        # Call method
        result = kernel_manager.calculate_bond_counts(bond_types, molecule_bond_boundaries)
        
        # Verify kernel was called
        mock_kernel.assert_called_once()
        
        # Verify result structure
        expected_keys = ['single_bonds', 'double_bonds', 'triple_bonds', 'total_bonds']
        assert all(key in result for key in expected_keys)
        assert len(result) == 4
    
    def test_calculate_tpsa(self, kernel_manager, mock_cupy_kernels):
        """Test TPSA calculation."""
        mock_cp, mock_raw_kernel, mock_kernel = mock_cupy_kernels
        
        # Mock input data
        atomic_numbers = Mock()
        hybridizations = Mock()
        hydrogen_counts = Mock()
        is_aromatic = Mock()
        molecule_boundaries = Mock()
        molecule_boundaries.__len__ = Mock(return_value=1)  # 1 molecule
        
        # Mock result array
        mock_result = Mock()
        mock_cp.zeros.return_value = mock_result
        
        # Call method
        result = kernel_manager.calculate_tpsa(
            atomic_numbers, hybridizations, hydrogen_counts,
            is_aromatic, molecule_boundaries
        )
        
        # Verify kernel was called
        mock_kernel.assert_called_once()
        assert result == mock_result
    
    def test_calculate_rotatable_bonds(self, kernel_manager, mock_cupy_kernels):
        """Test rotatable bonds calculation."""
        mock_cp, mock_raw_kernel, mock_kernel = mock_cupy_kernels
        
        # Mock input data
        bond_atom1 = Mock()
        bond_atom2 = Mock()
        bond_types = Mock()
        bond_in_ring = Mock()
        atomic_numbers = Mock()
        molecule_bond_boundaries = Mock()
        molecule_bond_boundaries.__len__ = Mock(return_value=2)  # 2 molecules
        
        # Mock result array
        mock_result = Mock()
        mock_cp.zeros.return_value = mock_result
        
        # Call method
        result = kernel_manager.calculate_rotatable_bonds(
            bond_atom1, bond_atom2, bond_types, bond_in_ring,
            atomic_numbers, molecule_bond_boundaries
        )
        
        # Verify kernel was called
        mock_kernel.assert_called_once()
        assert result == mock_result
    
    def test_calculate_aromatic_rings(self, kernel_manager, mock_cupy_kernels):
        """Test aromatic rings calculation."""
        mock_cp, mock_raw_kernel, mock_kernel = mock_cupy_kernels
        
        # Mock input data
        atom_is_aromatic = Mock()
        bond_atom1 = Mock()
        bond_atom2 = Mock()
        bond_is_aromatic = Mock()
        molecule_boundaries = Mock()
        molecule_bond_boundaries = Mock()
        molecule_boundaries.__len__ = Mock(return_value=1)  # 1 molecule
        
        # Mock result array
        mock_result = Mock()
        mock_cp.zeros.return_value = mock_result
        
        # Call method
        result = kernel_manager.calculate_aromatic_rings(
            atom_is_aromatic, bond_atom1, bond_atom2, bond_is_aromatic,
            molecule_boundaries, molecule_bond_boundaries
        )
        
        # Verify kernel was called
        mock_kernel.assert_called_once()
        assert result == mock_result
    
    def test_kernel_compilation_error(self, mock_cupy_kernels, mock_memory_manager):
        """Test handling of kernel compilation errors."""
        mock_cp, mock_raw_kernel, mock_kernel = mock_cupy_kernels
        
        # Mock RawKernel to raise an exception
        mock_raw_kernel.side_effect = Exception("Compilation failed")
        
        with pytest.raises(Exception, match="Compilation failed"):
            CUDAKernelManager(mock_memory_manager)


class TestBasicDescriptorCalculator:
    """Test BasicDescriptorCalculator class."""
    
    @pytest.fixture
    def mock_cupy_calculator(self):
        """Mock cupy for calculator testing."""
        with patch('src.chemi_accel.backends.cuda.kernels.CUDA_AVAILABLE', True), \
             patch('src.chemi_accel.backends.cuda.kernels.cp') as mock_cp, \
             patch('src.chemi_accel.backends.cuda.kernels.RawKernel'):
            
            # Mock cupy arrays
            mock_cp.zeros.return_value = Mock()
            mock_cp.float32 = np.float32
            mock_cp.int32 = np.int32
            mock_cp.ndarray = np.ndarray
            
            yield mock_cp
    
    @pytest.fixture
    def mock_kernel_manager(self):
        """Mock kernel manager for testing."""
        kernel_manager = Mock()
        
        # Mock kernel methods to return mock arrays
        kernel_manager.calculate_molecular_weights.return_value = Mock()
        kernel_manager.calculate_atom_counts.return_value = (Mock(), Mock())
        kernel_manager.calculate_bond_counts.return_value = {
            'single_bonds': Mock(),
            'double_bonds': Mock(),
            'triple_bonds': Mock(),
            'total_bonds': Mock()
        }
        kernel_manager.calculate_tpsa.return_value = Mock()
        kernel_manager.calculate_rotatable_bonds.return_value = Mock()
        kernel_manager.calculate_aromatic_rings.return_value = Mock()
        
        return kernel_manager
    
    @pytest.fixture
    def descriptor_calculator(self, mock_cupy_calculator, mock_kernel_manager):
        """Create descriptor calculator for testing."""
        with patch('src.chemi_accel.backends.cuda.kernels.CUDAKernelManager') as mock_km_class, \
             patch('src.chemi_accel.backends.cuda.kernels.get_memory_manager') as mock_get_mm:
            mock_km_class.return_value = mock_kernel_manager
            mock_get_mm.return_value = Mock()
            calc = BasicDescriptorCalculator()
            calc.kernel_manager = mock_kernel_manager
            return calc
    
    def test_calculator_initialization(self, mock_cupy_calculator):
        """Test descriptor calculator initialization."""
        with patch('src.chemi_accel.backends.cuda.kernels.CUDAKernelManager'), \
             patch('src.chemi_accel.backends.cuda.kernels.get_memory_manager') as mock_get_mm:
            mock_get_mm.return_value = Mock()
            calc = BasicDescriptorCalculator()
            assert calc.memory_manager is not None
            assert calc.kernel_manager is not None
    
    def test_calculate_basic_descriptors_complete(self, descriptor_calculator, mock_kernel_manager):
        """Test calculating all basic descriptors with complete data."""
        # Mock complete molecular data
        molecular_data = {
            'atom_weights': Mock(),
            'atomic_numbers': Mock(),
            'molecule_boundaries': Mock(),
            'bond_types': Mock(),
            'molecule_bond_boundaries': Mock(),
            'hybridizations': Mock(),
            'hydrogen_counts': Mock(),
            'is_aromatic': Mock(),
            'bond_atom1': Mock(),
            'bond_atom2': Mock(),
            'bond_in_ring': Mock(),
            'atom_is_aromatic': Mock(),
            'bond_is_aromatic': Mock()
        }
        
        # Call method
        results = descriptor_calculator.calculate_basic_descriptors(molecular_data)
        
        # Verify all descriptors were calculated
        expected_descriptors = [
            'molecular_weight', 'heavy_atom_count', 'total_atom_count',
            'single_bonds', 'double_bonds', 'triple_bonds', 'total_bonds',
            'tpsa', 'rotatable_bonds', 'aromatic_rings'
        ]
        
        for descriptor in expected_descriptors:
            assert descriptor in results
        
        # Verify kernel methods were called
        mock_kernel_manager.calculate_molecular_weights.assert_called_once()
        mock_kernel_manager.calculate_atom_counts.assert_called_once()
        mock_kernel_manager.calculate_bond_counts.assert_called_once()
        mock_kernel_manager.calculate_tpsa.assert_called_once()
        mock_kernel_manager.calculate_rotatable_bonds.assert_called_once()
        mock_kernel_manager.calculate_aromatic_rings.assert_called_once()
    
    def test_calculate_basic_descriptors_partial(self, descriptor_calculator, mock_kernel_manager):
        """Test calculating descriptors with partial data."""
        # Mock partial molecular data (only molecular weights)
        molecular_data = {
            'atom_weights': Mock(),
            'molecule_boundaries': Mock()
        }
        
        # Call method
        results = descriptor_calculator.calculate_basic_descriptors(molecular_data)
        
        # Verify only molecular weight was calculated
        assert 'molecular_weight' in results
        assert len(results) == 1
        
        # Verify only molecular weight kernel was called
        mock_kernel_manager.calculate_molecular_weights.assert_called_once()
        mock_kernel_manager.calculate_atom_counts.assert_not_called()
        mock_kernel_manager.calculate_bond_counts.assert_not_called()
        mock_kernel_manager.calculate_tpsa.assert_not_called()
        mock_kernel_manager.calculate_rotatable_bonds.assert_not_called()
        mock_kernel_manager.calculate_aromatic_rings.assert_not_called()
    
    def test_calculate_basic_descriptors_empty(self, descriptor_calculator, mock_kernel_manager):
        """Test calculating descriptors with empty data."""
        molecular_data = {}
        
        # Call method
        results = descriptor_calculator.calculate_basic_descriptors(molecular_data)
        
        # Verify no descriptors were calculated
        assert len(results) == 0
        
        # Verify no kernel methods were called
        mock_kernel_manager.calculate_molecular_weights.assert_not_called()
        mock_kernel_manager.calculate_atom_counts.assert_not_called()
        mock_kernel_manager.calculate_bond_counts.assert_not_called()
        mock_kernel_manager.calculate_tpsa.assert_not_called()
        mock_kernel_manager.calculate_rotatable_bonds.assert_not_called()
        mock_kernel_manager.calculate_aromatic_rings.assert_not_called()


class TestKernelIntegration:
    """Integration tests for kernel functionality."""
    
    @pytest.fixture
    def mock_cupy_integration(self):
        """Mock cupy for integration testing."""
        with patch('src.chemi_accel.backends.cuda.kernels.CUDA_AVAILABLE', True), \
             patch('src.chemi_accel.backends.cuda.kernels.cp') as mock_cp, \
             patch('src.chemi_accel.backends.cuda.kernels.RawKernel') as mock_raw_kernel:
            
            # Mock cupy arrays and functions
            mock_cp.zeros.return_value = Mock()
            mock_cp.float32 = np.float32
            mock_cp.int32 = np.int32
            mock_cp.ndarray = np.ndarray
            
            # Mock RawKernel
            mock_kernel = Mock()
            mock_raw_kernel.return_value = mock_kernel
            
            yield mock_cp, mock_raw_kernel, mock_kernel
    
    def test_molecular_weight_kernel_parameters(self, mock_cupy_integration):
        """Test molecular weight kernel is called with correct parameters."""
        mock_cp, mock_raw_kernel, mock_kernel = mock_cupy_integration
        
        # Create kernel manager
        with patch('src.chemi_accel.backends.cuda.kernels.get_memory_manager'):
            km = CUDAKernelManager()
        
        # Mock input data
        atom_weights = Mock()
        molecule_boundaries = Mock()
        molecule_boundaries.__len__ = Mock(return_value=5)  # 5 molecules
        
        # Mock result
        mock_result = Mock()
        mock_cp.zeros.return_value = mock_result
        
        # Call method
        result = km.calculate_molecular_weights(atom_weights, molecule_boundaries)
        
        # Verify kernel call parameters
        mock_kernel.assert_called_once()
        call_args = mock_kernel.call_args
        
        # Check grid and block sizes
        grid_size, block_size = call_args[0][:2]
        assert block_size == (256,)  # Block size should be 256
        
        # Check kernel arguments
        kernel_args = call_args[0][2]
        assert len(kernel_args) == 4  # Should have 4 arguments
        assert kernel_args[0] == atom_weights
        assert kernel_args[1] == molecule_boundaries
        assert kernel_args[2] == mock_result
        assert kernel_args[3] == 5  # num_molecules
    
    def test_kernel_grid_size_calculation(self, mock_cupy_integration):
        """Test grid size calculation for different molecule counts."""
        mock_cp, mock_raw_kernel, mock_kernel = mock_cupy_integration
        
        with patch('src.chemi_accel.backends.cuda.kernels.get_memory_manager'):
            km = CUDAKernelManager()
        
        test_cases = [
            (1, 1),      # 1 molecule -> 1 block
            (256, 1),    # 256 molecules -> 1 block
            (257, 2),    # 257 molecules -> 2 blocks
            (512, 2),    # 512 molecules -> 2 blocks
            (513, 3),    # 513 molecules -> 3 blocks
        ]
        
        for num_molecules, expected_blocks in test_cases:
            # Reset mock
            mock_kernel.reset_mock()
            mock_cp.zeros.reset_mock()
            
            # Mock input
            atom_weights = Mock()
            molecule_boundaries = Mock()
            molecule_boundaries.__len__ = Mock(return_value=num_molecules)
            
            # Call method
            km.calculate_molecular_weights(atom_weights, molecule_boundaries)
            
            # Check grid size
            call_args = mock_kernel.call_args
            grid_size = call_args[0][0]
            assert grid_size == (expected_blocks,), f"Failed for {num_molecules} molecules"


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_cuda_not_available_kernel_manager(self):
        """Test behavior when CUDA is not available for kernel manager."""
        with patch('src.chemi_accel.backends.cuda.kernels.CUDA_AVAILABLE', False):
            with pytest.raises(RuntimeError, match="CUDA not available"):
                CUDAKernelManager()
    
    def test_cuda_not_available_descriptor_calculator(self):
        """Test behavior when CUDA is not available for descriptor calculator."""
        with patch('src.chemi_accel.backends.cuda.kernels.CUDA_AVAILABLE', False):
            with pytest.raises(RuntimeError, match="CUDA not available"):
                BasicDescriptorCalculator()


class TestGlobalFunctions:
    """Test global functions."""
    
    @patch('src.chemi_accel.backends.cuda.kernels.CUDA_AVAILABLE', True)
    def test_get_kernel_manager(self):
        """Test getting global kernel manager."""
        # Reset global instance
        import src.chemi_accel.backends.cuda.kernels as kernels_module
        kernels_module._kernel_manager = None
        
        with patch('src.chemi_accel.backends.cuda.kernels.CUDAKernelManager') as mock_km:
            mock_instance = Mock()
            mock_km.return_value = mock_instance
            
            manager1 = get_kernel_manager()
            manager2 = get_kernel_manager()
            
            assert manager1 is manager2  # Should be same instance
            mock_km.assert_called_once()  # Should only be created once


if __name__ == "__main__":
    pytest.main([__file__])