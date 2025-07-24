#!/usr/bin/env python3
"""
RAPIDS Integration Demo for ChemiAccel.

This script demonstrates the RAPIDS integration capabilities of ChemiAccel,
including cudf DataFrame processing and CuPy array operations for molecular
descriptor calculations.

Requirements:
- ChemiAccel with RAPIDS support
- cudf and cupy (optional, will gracefully degrade if not available)
"""

import sys
import numpy as np
import pandas as pd
from typing import List

# Import ChemiAccel components
from chemi_accel import Calculator
from chemi_accel.backends.rapids import (
    check_cudf_availability,
    check_cupy_availability,
    get_cudf_version_info,
    get_cupy_version_info,
    CudfProcessor,
    CupyArrayProcessor
)

# Sample molecular data
SAMPLE_SMILES = [
    "CCO",                    # ethanol
    "CC(=O)O",               # acetic acid
    "c1ccccc1",              # benzene
    "CCN(CC)CC",             # triethylamine
    "CC(C)O",                # isopropanol
    "CCCCCCCCCCCCCCCCCC(=O)O",  # stearic acid
    "CC(C)(C)O",             # tert-butanol
    "c1ccc2ccccc2c1",        # naphthalene
    "CC(=O)Nc1ccc(O)cc1",    # acetaminophen
    "CC1=CC=C(C=C1)C(=O)O",  # p-toluic acid
]


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def check_rapids_availability():
    """Check and report RAPIDS component availability."""
    print_section("RAPIDS Availability Check")
    
    cudf_available = check_cudf_availability()
    cupy_available = check_cupy_availability()
    
    print(f"cudf available: {cudf_available}")
    print(f"cupy available: {cupy_available}")
    
    if cudf_available:
        cudf_info = get_cudf_version_info()
        print(f"cudf version info: {cudf_info}")
    
    if cupy_available:
        cupy_info = get_cupy_version_info()
        print(f"cupy version info: {cupy_info}")
    
    return cudf_available, cupy_available


def demo_basic_calculator():
    """Demonstrate basic Calculator functionality."""
    print_section("Basic Calculator Demo")
    
    try:
        # Create calculator with CPU backend (always available)
        calc = Calculator(backend='cpu')
        print(f"Calculator created: {calc}")
        
        # Calculate descriptors using pandas
        molecules_series = pd.Series(SAMPLE_SMILES, name="smiles")
        print(f"\nCalculating descriptors for {len(molecules_series)} molecules...")
        
        results_df = calc.pandas(molecules_series)
        print(f"Results shape: {results_df.shape}")
        print(f"Descriptors calculated: {list(results_df.columns)}")
        
        # Show sample results
        print("\nSample results:")
        print(results_df.head(3).to_string())
        
        return True
        
    except Exception as e:
        print(f"Basic calculator demo failed: {e}")
        return False


def demo_cudf_integration():
    """Demonstrate cudf integration if available."""
    print_section("cudf Integration Demo")
    
    if not check_cudf_availability():
        print("cudf not available - skipping cudf integration demo")
        print("To install cudf: pip install cudf-cu11")
        return False
    
    try:
        import cudf
        
        # Create cudf Series
        cudf_molecules = cudf.Series(SAMPLE_SMILES, name="smiles")
        print(f"Created cudf Series with {len(cudf_molecules)} molecules")
        print(f"cudf Series type: {type(cudf_molecules)}")
        
        # Test CudfProcessor directly
        processor = CudfProcessor()
        print("CudfProcessor initialized successfully")
        
        # Get memory info
        memory_info = processor.get_memory_info()
        print(f"GPU memory info: {memory_info}")
        
        # Calculate descriptors
        descriptor_sets = ["molecular_weight", "num_atoms", "logp"]
        print(f"\nCalculating descriptors: {descriptor_sets}")
        
        results_cudf = processor.calculate_descriptors_cudf(
            cudf_molecules, 
            descriptor_sets
        )
        
        print(f"Results type: {type(results_cudf)}")
        print(f"Results shape: {results_cudf.shape}")
        print(f"Results columns: {list(results_cudf.columns)}")
        
        # Show sample results
        print("\nSample cudf results:")
        print(results_cudf.head(3))
        
        # Test Calculator.cudf() method
        try:
            calc = Calculator(backend='cuda')
            calc_results = calc.cudf(cudf_molecules)
            print(f"\nCalculator.cudf() results shape: {calc_results.shape}")
            
        except Exception as e:
            print(f"Calculator.cudf() test failed (expected if no CUDA): {e}")
        
        return True
        
    except Exception as e:
        print(f"cudf integration demo failed: {e}")
        return False


def demo_cupy_operations():
    """Demonstrate CuPy array operations if available."""
    print_section("CuPy Operations Demo")
    
    if not check_cupy_availability():
        print("CuPy not available - skipping CuPy operations demo")
        print("To install CuPy: pip install cupy-cuda11x")
        return False
    
    try:
        import cupy as cp
        
        # Initialize CuPy processor
        processor = CupyArrayProcessor()
        print("CupyArrayProcessor initialized successfully")
        
        # Get device info
        device_info = processor.get_device_info()
        print(f"CUDA device info: {device_info}")
        
        # Test array conversions
        print("\nTesting array conversions...")
        numpy_array = np.random.rand(100, 10).astype(np.float32)
        print(f"Original NumPy array shape: {numpy_array.shape}")
        
        # Convert to CuPy
        cupy_array = processor.numpy_to_cupy(numpy_array)
        print(f"CuPy array shape: {cupy_array.shape}")
        print(f"CuPy array type: {type(cupy_array)}")
        
        # Convert back to NumPy
        numpy_result = processor.cupy_to_numpy(cupy_array)
        print(f"Converted back to NumPy: {numpy_result.shape}")
        
        # Verify arrays are equal
        arrays_equal = np.allclose(numpy_array, numpy_result)
        print(f"Arrays equal after round-trip: {arrays_equal}")
        
        # Test molecular weight calculation
        print("\nTesting molecular weight calculation...")
        from chemi_accel.backends.rapids import molecular_weight_cupy
        
        # Simulate atomic weights for molecules
        atom_weights = cp.array([
            [12.01, 1.008, 1.008, 1.008, 1.008, 0, 0],  # methane
            [12.01, 12.01, 1.008, 1.008, 1.008, 1.008, 0],  # ethane
            [12.01, 12.01, 16.0, 1.008, 1.008, 0, 0],  # ethanol (simplified)
        ])
        
        molecular_weights = molecular_weight_cupy(atom_weights)
        mw_numpy = cp.asnumpy(molecular_weights)
        print(f"Calculated molecular weights: {mw_numpy}")
        
        # Test Tanimoto similarity
        print("\nTesting Tanimoto similarity...")
        from chemi_accel.backends.rapids import tanimoto_similarity_cupy
        
        # Create sample fingerprints
        fp1 = cp.array([[1, 0, 1, 1, 0, 1, 0, 0]], dtype=bool)
        fp2 = cp.array([[1, 1, 0, 1, 0, 0, 1, 0]], dtype=bool)
        
        similarity = tanimoto_similarity_cupy(fp1, fp2)
        sim_numpy = cp.asnumpy(similarity)[0]
        print(f"Tanimoto similarity: {sim_numpy:.4f}")
        
        # Test performance benchmark
        print("\nRunning simple performance benchmark...")
        
        def simple_operation(array):
            if hasattr(array, '__array_module__'):
                # CuPy array
                return cp.sum(array ** 2, axis=1)
            else:
                # NumPy array
                return np.sum(array ** 2, axis=1)
        
        test_data = np.random.rand(1000, 100).astype(np.float32)
        
        benchmark_results = processor.benchmark_vs_numpy(
            simple_operation,
            test_data,
            num_iterations=3
        )
        
        print(f"Benchmark results: {benchmark_results}")
        
        return True
        
    except Exception as e:
        print(f"CuPy operations demo failed: {e}")
        return False


def demo_workflow_integration():
    """Demonstrate typical workflow integration."""
    print_section("Workflow Integration Demo")
    
    try:
        # Create a typical cheminformatics workflow
        print("Creating sample dataset...")
        
        # Create pandas DataFrame with molecular data
        df = pd.DataFrame({
            'compound_id': [f'COMP_{i:03d}' for i in range(len(SAMPLE_SMILES))],
            'smiles': SAMPLE_SMILES,
            'activity': np.random.rand(len(SAMPLE_SMILES)) * 10,  # Random activity values
            'source': ['synthetic'] * len(SAMPLE_SMILES)
        })
        
        print(f"Created dataset with {len(df)} compounds")
        print(df.head(3))
        
        # Calculate descriptors
        print("\nCalculating molecular descriptors...")
        calc = Calculator(backend='cpu')
        
        descriptors_df = calc.pandas(df['smiles'])
        print(f"Calculated {len(descriptors_df.columns)} descriptors")
        
        # Merge with original data
        print("\nMerging descriptors with original data...")
        
        # Reset indices to ensure proper merging
        df_reset = df.reset_index(drop=True)
        descriptors_reset = descriptors_df.reset_index(drop=True)
        
        # Merge dataframes
        if len(df_reset) == len(descriptors_reset):
            merged_df = pd.concat([df_reset, descriptors_reset], axis=1)
            print(f"Merged dataset shape: {merged_df.shape}")
            print(f"Total columns: {len(merged_df.columns)}")
            
            # Show sample of merged data
            print("\nSample merged data:")
            sample_cols = ['compound_id', 'smiles', 'activity', 'molecular_weight', 'logp']
            available_cols = [col for col in sample_cols if col in merged_df.columns]
            print(merged_df[available_cols].head(3).to_string())
            
            # Basic analysis
            print("\nBasic descriptor statistics:")
            numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                print(merged_df[numeric_cols].describe().round(3))
        
        return True
        
    except Exception as e:
        print(f"Workflow integration demo failed: {e}")
        return False


def main():
    """Main demo function."""
    print("ChemiAccel RAPIDS Integration Demo")
    print("=" * 60)
    
    # Check system capabilities
    cudf_available, cupy_available = check_rapids_availability()
    
    # Run demos
    demos_run = 0
    demos_passed = 0
    
    # Basic calculator (always available)
    demos_run += 1
    if demo_basic_calculator():
        demos_passed += 1
    
    # cudf integration (if available)
    if cudf_available:
        demos_run += 1
        if demo_cudf_integration():
            demos_passed += 1
    
    # CuPy operations (if available)
    if cupy_available:
        demos_run += 1
        if demo_cupy_operations():
            demos_passed += 1
    
    # Workflow integration
    demos_run += 1
    if demo_workflow_integration():
        demos_passed += 1
    
    # Summary
    print_section("Demo Summary")
    print(f"Demos run: {demos_run}")
    print(f"Demos passed: {demos_passed}")
    print(f"Success rate: {demos_passed/demos_run*100:.1f}%")
    
    if not cudf_available and not cupy_available:
        print("\nNote: RAPIDS components (cudf, CuPy) are not installed.")
        print("To install RAPIDS support:")
        print("  pip install cudf-cu11 cupy-cuda11x")
        print("  (Requires NVIDIA GPU with CUDA support)")
    
    return demos_passed == demos_run


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)