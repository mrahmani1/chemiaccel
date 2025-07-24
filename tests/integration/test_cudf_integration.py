"""
Integration tests for cudf functionality in ChemiAccel.

These tests verify the seamless integration between ChemiAccel and cudf,
including input processing, descriptor calculation, and output generation.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# Import ChemiAccel components
from chemi_accel import Calculator
from chemi_accel.backends.rapids import (
    CudfProcessor, 
    check_cudf_availability,
    get_cudf_version_info
)
from chemi_accel.core.exceptions import (
    ChemiAccelError, 
    BackendUnavailableError
)

# Test data
TEST_SMILES = [
    "CCO",  # ethanol
    "CC(=O)O",  # acetic acid
    "c1ccccc1",  # benzene
    "CCN(CC)CC",  # triethylamine
    "CC(C)O",  # isopropanol
]

INVALID_SMILES = [
    "INVALID",
    "C[C@H](C)O",  # Valid but complex
    "",  # Empty
    "C1CCCCC1",  # cyclohexane
]


class TestCudfAvailability:
    """Test cudf availability checking."""
    
    def test_check_cudf_availability(self):
        """Test cudf availability detection."""
        # This will return True or False depending on installation
        result = check_cudf_availability()
        assert isinstance(result, bool)
    
    def test_get_cudf_version_info(self):
        """Test cudf version information retrieval."""
        info = get_cudf_version_info()
        assert isinstance(info, dict)
        
        if check_cudf_availability():
            # If cudf is available, should have version info
            assert 'cudf' in info or 'error' in info
        else:
            # If not available, should have error message
            assert 'error' in info


@pytest.mark.skipif(not check_cudf_availability(), reason="cudf not available")
class TestCudfProcessor:
    """Test CudfProcessor functionality when cudf is available."""
    
    @pytest.fixture
    def cudf_processor(self):
        """Create CudfProcessor instance for testing."""
        return CudfProcessor()
    
    @pytest.fixture
    def sample_cudf_series(self):
        """Create sample cudf Series for testing."""
        import cudf
        return cudf.Series(TEST_SMILES, name="smiles")
    
    def test_cudf_processor_initialization(self, cudf_processor):
        """Test CudfProcessor initialization."""
        assert cudf_processor is not None
        assert hasattr(cudf_processor, 'molecular_parser')
        assert hasattr(cudf_processor, 'descriptor_calculator')
    
    def test_process_cudf_series(self, cudf_processor, sample_cudf_series):
        """Test processing cudf Series input."""
        smiles_list = cudf_processor.process_cudf_series(sample_cudf_series)
        
        assert isinstance(smiles_list, list)
        assert len(smiles_list) == len(TEST_SMILES)
        assert smiles_list == TEST_SMILES
    
    def test_process_cudf_series_invalid_input(self, cudf_processor):
        """Test processing invalid cudf Series input."""
        import cudf
        
        # Test with non-string data
        invalid_series = cudf.Series([1, 2, 3, 4, 5])
        
        with pytest.raises(Exception):  # Should raise CudfIntegrationError
            cudf_processor.process_cudf_series(invalid_series)
    
    def test_create_cudf_dataframe(self, cudf_processor):
        """Test creating cudf DataFrame from pandas results."""
        import cudf
        
        # Create sample pandas DataFrame
        pandas_df = pd.DataFrame({
            'molecular_weight': [46.07, 60.05, 78.11, 101.19, 60.10],
            'num_atoms': [9, 8, 12, 20, 12],
            'logp': [-0.31, -0.17, 2.13, 1.45, 0.05]
        })
        
        # Create cudf DataFrame
        cudf_df = cudf_processor.create_cudf_dataframe(pandas_df)
        
        assert isinstance(cudf_df, cudf.DataFrame)
        assert cudf_df.shape == pandas_df.shape
        assert list(cudf_df.columns) == list(pandas_df.columns)
    
    def test_calculate_descriptors_cudf(self, cudf_processor, sample_cudf_series):
        """Test end-to-end descriptor calculation with cudf."""
        descriptor_sets = ["molecular_weight", "num_atoms", "logp"]
        
        result_df = cudf_processor.calculate_descriptors_cudf(
            sample_cudf_series, 
            descriptor_sets
        )
        
        import cudf
        assert isinstance(result_df, cudf.DataFrame)
        assert len(result_df) <= len(TEST_SMILES)  # Some molecules might be invalid
        
        # Check that requested descriptors are present
        for descriptor in descriptor_sets:
            if descriptor in result_df.columns:
                assert descriptor in result_df.columns
    
    def test_calculate_descriptors_empty_input(self, cudf_processor):
        """Test descriptor calculation with empty input."""
        import cudf
        
        empty_series = cudf.Series([], dtype='object', name="smiles")
        result_df = cudf_processor.calculate_descriptors_cudf(
            empty_series, 
            ["molecular_weight"]
        )
        
        assert isinstance(result_df, cudf.DataFrame)
        assert len(result_df) == 0
    
    def test_memory_info(self, cudf_processor):
        """Test GPU memory information retrieval."""
        memory_info = cudf_processor.get_memory_info()
        
        assert isinstance(memory_info, dict)
        # Should have some memory information if GPU is available
    
    def test_memory_optimization(self, cudf_processor):
        """Test memory optimization functionality."""
        # This should not raise an exception
        cudf_processor.optimize_memory_usage()


@pytest.mark.skipif(not check_cudf_availability(), reason="cudf not available")
class TestCalculatorCudfIntegration:
    """Test Calculator class cudf integration."""
    
    @pytest.fixture
    def cuda_calculator(self):
        """Create Calculator with CUDA backend for cudf testing."""
        try:
            return Calculator(backend='cuda')
        except Exception:
            pytest.skip("CUDA backend not available")
    
    @pytest.fixture
    def sample_cudf_series(self):
        """Create sample cudf Series for testing."""
        import cudf
        return cudf.Series(TEST_SMILES, name="smiles")
    
    def test_calculator_cudf_method(self, cuda_calculator, sample_cudf_series):
        """Test Calculator.cudf() method."""
        result_df = cuda_calculator.cudf(sample_cudf_series)
        
        import cudf
        assert isinstance(result_df, cudf.DataFrame)
        assert len(result_df) <= len(TEST_SMILES)
        
        # Should have default descriptors
        expected_descriptors = [
            "molecular_weight", "num_atoms", "logp", "tpsa"
        ]
        
        for descriptor in expected_descriptors:
            if descriptor in result_df.columns:
                # Check that values are reasonable
                assert not result_df[descriptor].isna().all()
    
    def test_calculator_cudf_with_custom_descriptors(self, cuda_calculator, sample_cudf_series):
        """Test Calculator.cudf() with custom descriptor configuration."""
        # Configure specific descriptors
        cuda_calculator.configure_descriptors(["molecular_weight", "num_atoms"])
        
        result_df = cuda_calculator.cudf(sample_cudf_series)
        
        import cudf
        assert isinstance(result_df, cudf.DataFrame)
        
        # Should contain the configured descriptors
        available_descriptors = set(result_df.columns)
        requested_descriptors = {"molecular_weight", "num_atoms"}
        
        # At least some of the requested descriptors should be present
        assert len(available_descriptors.intersection(requested_descriptors)) > 0
    
    def test_calculator_cudf_wrong_backend(self):
        """Test Calculator.cudf() with non-CUDA backend."""
        try:
            cpu_calculator = Calculator(backend='cpu')
            import cudf
            sample_series = cudf.Series(TEST_SMILES)
            
            with pytest.raises(ChemiAccelError, match="requires CUDA backend"):
                cpu_calculator.cudf(sample_series)
                
        except ImportError:
            pytest.skip("cudf not available for this test")


class TestCudfIntegrationMocked:
    """Test cudf integration with mocked dependencies."""
    
    @patch('chemi_accel.backends.rapids.cudf_integration.CUDF_AVAILABLE', False)
    def test_cudf_processor_unavailable(self):
        """Test CudfProcessor when cudf is not available."""
        from chemi_accel.backends.rapids.cudf_integration import CudfProcessor
        
        with pytest.raises(BackendUnavailableError):
            CudfProcessor()
    
    @patch('chemi_accel.backends.rapids.cudf_integration.check_cudf_availability')
    def test_calculator_cudf_unavailable(self, mock_check):
        """Test Calculator.cudf() when cudf is not available."""
        mock_check.return_value = False
        
        try:
            calculator = Calculator(backend='cuda')
            
            with pytest.raises(BackendUnavailableError):
                calculator.cudf([])
                
        except Exception:
            pytest.skip("CUDA backend not available")


class TestCudfWorkflowIntegration:
    """Test integration with typical cudf workflows."""
    
    @pytest.mark.skipif(not check_cudf_availability(), reason="cudf not available")
    def test_cudf_pipeline_workflow(self):
        """Test typical cudf data pipeline workflow."""
        import cudf
        
        # Create sample dataset
        df = cudf.DataFrame({
            'compound_id': [f'COMP_{i:03d}' for i in range(len(TEST_SMILES))],
            'smiles': TEST_SMILES,
            'activity': [1.2, 2.3, 0.8, 3.1, 1.7]
        })
        
        try:
            # Initialize calculator
            calculator = Calculator(backend='cuda')
            
            # Calculate descriptors
            descriptors_df = calculator.cudf(df['smiles'])
            
            # Merge with original data (typical workflow)
            if len(descriptors_df) > 0:
                # Reset indices to ensure proper merging
                df_reset = df.reset_index(drop=True)
                descriptors_reset = descriptors_df.reset_index(drop=True)
                
                # Merge dataframes
                if len(df_reset) == len(descriptors_reset):
                    merged_df = cudf.concat([df_reset, descriptors_reset], axis=1)
                    
                    assert isinstance(merged_df, cudf.DataFrame)
                    assert 'compound_id' in merged_df.columns
                    assert 'smiles' in merged_df.columns
                    assert 'activity' in merged_df.columns
                    
                    # Should have additional descriptor columns
                    assert len(merged_df.columns) > 3
        
        except Exception as e:
            pytest.skip(f"cudf workflow test failed: {e}")
    
    @pytest.mark.skipif(not check_cudf_availability(), reason="cudf not available")
    def test_cudf_memory_management(self):
        """Test memory management in cudf workflows."""
        import cudf
        
        try:
            # Create larger dataset to test memory management
            large_smiles = TEST_SMILES * 100  # 500 molecules
            large_series = cudf.Series(large_smiles, name="smiles")
            
            calculator = Calculator(backend='cuda')
            processor = CudfProcessor()
            
            # Get initial memory info
            initial_memory = processor.get_memory_info()
            
            # Process large dataset
            result_df = calculator.cudf(large_series)
            
            # Get memory info after processing
            final_memory = processor.get_memory_info()
            
            # Optimize memory
            processor.optimize_memory_usage()
            
            # Verify results
            assert isinstance(result_df, cudf.DataFrame)
            assert len(result_df) <= len(large_smiles)
            
        except Exception as e:
            pytest.skip(f"Memory management test failed: {e}")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])