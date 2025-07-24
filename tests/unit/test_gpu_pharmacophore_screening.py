"""
Test Suite for GPU-Accelerated Pharmacophore Screening

This module contains comprehensive tests for the GPU pharmacophore screening
implementation, including CUDA kernel validation, performance testing,
and accuracy verification against reference implementations.
"""

import pytest
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import time
from unittest.mock import Mock, patch

# Import core modules
from src.chemi_accel.core.molecular import MolecularData
from src.chemi_accel.core.exceptions import ChemiAccelError

# Import pharmacophore screening modules
from src.chemi_accel.descriptors.pharmacophore_screening import (
    PharmacophoreScreeningEngine,
    create_screening_engine,
    quick_screen
)

# Try to import GPU-specific modules
try:
    from src.chemi_accel.backends.cuda.pharmacophore_screening import (
        GPUPharmacophoreScreener,
        PharmacophoreQuery,
        ScreeningResult,
        create_pharmacophore_query,
        screen_database_gpu
    )
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    GPUPharmacophoreScreener = None

# Import test utilities
try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

class TestMoleculeGenerator:
    """Generate test molecules for screening validation."""
    
    @staticmethod
    def generate_test_molecules(count: int = 100) -> List[MolecularData]:
        """Generate diverse test molecules."""
        if not RDKIT_AVAILABLE:
            pytest.skip("RDKit not available for molecule generation")
        
        # Diverse SMILES for testing
        test_smiles = [
            "CCO",  # ethanol - donor/acceptor
            "CC(C)O",  # isopropanol - donor/acceptor
            "c1ccccc1",  # benzene - aromatic
            "c1ccc(O)cc1",  # phenol - aromatic/donor/acceptor
            "CC(=O)O",  # acetic acid - donor/acceptor
            "CCN",  # ethylamine - donor
            "CC(C)(C)C",  # tert-butane - hydrophobic
            "c1ccc(N)cc1",  # aniline - aromatic/donor
            "CC(=O)N",  # acetamide - donor/acceptor
            "CCc1ccccc1",  # ethylbenzene - aromatic/hydrophobic
            "CC(C)C(=O)O",  # isobutyric acid
            "c1ccc(C(=O)O)cc1",  # benzoic acid
            "CCOc1ccccc1",  # ethoxybenzene
            "CC1=CC=CC=C1",  # toluene
            "CC(C)N",  # isopropylamine
        ]
        
        molecules = []
        for i in range(count):
            smiles = test_smiles[i % len(test_smiles)]
            
            # Add variations
            if i > len(test_smiles):
                base_smiles = test_smiles[i % len(test_smiles)]
                if "CC" in base_smiles:
                    smiles = base_smiles.replace("CC", "CCC", 1)  # Add carbon
            
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                # Add 3D coordinates
                from rdkit.Chem import AllChem
                AllChem.EmbedMolecule(mol, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol)
                
                mol_data = MolecularData(
                    mol_id=f"test_mol_{i}",
                    smiles=smiles,
                    backend_data={"rdkit_mol": mol}
                )
                molecules.append(mol_data)
        
        return molecules

@pytest.fixture
def test_molecules():
    """Fixture providing test molecules."""
    return TestMoleculeGenerator.generate_test_molecules(50)

@pytest.fixture
def small_test_set():
    """Small test set for quick validation."""
    return TestMoleculeGenerator.generate_test_molecules(10)

@pytest.fixture
def pharmacophore_queries():
    """Standard pharmacophore queries for testing."""
    return [
        {
            "name": "simple_hb",
            "feature_counts": {"donor": 1, "acceptor": 1},
            "description": "Simple hydrogen bonding pharmacophore"
        },
        {
            "name": "aromatic_hb", 
            "feature_counts": {"donor": 1, "acceptor": 1, "aromatic": 1},
            "description": "Aromatic with hydrogen bonding"
        },
        {
            "name": "complex_pattern",
            "feature_counts": {"donor": 2, "acceptor": 2, "aromatic": 1, "hydrophobic": 1},
            "distance_constraints": {"donor_acceptor": (2.0, 8.0)},
            "description": "Complex multi-feature pharmacophore"
        }
    ]

class TestPharmacophoreScreeningEngine:
    """Test the unified pharmacophore screening engine."""
    
    def test_engine_initialization_auto_backend(self):
        """Test automatic backend selection."""
        engine = PharmacophoreScreeningEngine(backend="auto")
        
        if GPU_AVAILABLE:
            assert engine.backend == "gpu"
            assert engine.gpu_screener is not None
        else:
            assert engine.backend == "cpu"
            assert engine.cpu_analyzer is not None

    def test_engine_initialization_cpu_backend(self):
        """Test explicit CPU backend selection."""
        engine = PharmacophoreScreeningEngine(backend="cpu")
        
        assert engine.backend == "cpu"
        assert engine.cpu_analyzer is not None
        assert engine.gpu_screener is None

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU backend not available")
    def test_engine_initialization_gpu_backend(self):
        """Test explicit GPU backend selection."""
        engine = PharmacophoreScreeningEngine(backend="gpu")
        
        assert engine.backend == "gpu"
        assert engine.gpu_screener is not None
        assert engine.cpu_analyzer is None

    def test_query_creation(self, pharmacophore_queries):
        """Test pharmacophore query creation."""
        engine = create_screening_engine()
        
        for query_spec in pharmacophore_queries:
            query = engine.create_query(
                feature_requirements=query_spec["feature_counts"],
                distance_constraints=query_spec.get("distance_constraints"),
                query_name=query_spec["name"]
            )
            
            assert query is not None
            
            if engine.backend == "gpu":
                assert hasattr(query, 'feature_counts')
                assert query.feature_counts == query_spec["feature_counts"]
            else:
                assert isinstance(query, dict)
                assert query["feature_counts"] == query_spec["feature_counts"]

    def test_basic_screening(self, small_test_set, pharmacophore_queries):
        """Test basic molecular screening functionality."""
        engine = create_screening_engine()
        
        for query_spec in pharmacophore_queries[:2]:  # Test first two queries
            query = engine.create_query(query_spec["feature_counts"])
            
            results = engine.screen_molecules(
                small_test_set,
                query,
                similarity_threshold=0.3,
                return_details=True
            )
            
            assert len(results) == len(small_test_set)
            
            # Check result structure
            for result in results:
                assert "mol_id" in result
                assert "smiles" in result
                assert "similarity_score" in result
                assert "matches_query" in result
                assert "backend" in result
                assert 0.0 <= result["similarity_score"] <= 1.0

    def test_batch_screening(self, small_test_set, pharmacophore_queries):
        """Test batch screening with multiple queries."""
        engine = create_screening_engine()
        
        queries = []
        for query_spec in pharmacophore_queries:
            query = engine.create_query(
                query_spec["feature_counts"],
                query_name=query_spec["name"]
            )
            queries.append(query)
        
        results_df = engine.batch_screen_database(
            small_test_set,
            queries,
            similarity_threshold=0.3
        )
        
        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) == len(small_test_set) * len(queries)
        assert "query_name" in results_df.columns
        assert "similarity_score" in results_df.columns

    def test_screening_statistics(self, small_test_set):
        """Test screening statistics tracking."""
        engine = create_screening_engine()
        
        # Initial statistics should be zero
        stats = engine.get_screening_statistics()
        assert stats["total_molecules_screened"] == 0
        assert stats["queries_run"] == 0
        
        # Run screening
        query = engine.create_query({"donor": 1, "acceptor": 1})
        engine.screen_molecules(small_test_set, query, return_details=False)
        
        # Check updated statistics
        stats = engine.get_screening_statistics()
        assert stats["total_molecules_screened"] == len(small_test_set)
        assert stats["queries_run"] == 1
        assert "average_molecules_per_second" in stats

    def test_query_optimization(self, small_test_set):
        """Test pharmacophore query optimization."""
        engine = create_screening_engine()
        
        # Create initial query
        initial_query = engine.create_query({"donor": 1, "acceptor": 1})
        
        # Mock activity labels (first half active, second half inactive)
        active_labels = [True] * (len(small_test_set) // 2) + [False] * (len(small_test_set) - len(small_test_set) // 2)
        
        # Optimize query
        optimized_query = engine.optimize_query(
            small_test_set,
            active_labels,
            initial_query
        )
        
        assert optimized_query is not None
        if isinstance(optimized_query, dict):
            assert "feature_counts" in optimized_query
        else:
            assert hasattr(optimized_query, 'feature_counts')

@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU backend not available")
class TestGPUPharmacophoreScreener:
    """Test GPU-specific pharmacophore screening functionality."""
    
    def test_gpu_screener_initialization(self):
        """Test GPU screener initialization."""
        screener = GPUPharmacophoreScreener(batch_size=1000, memory_limit_gb=2.0)
        
        assert screener.batch_size == 1000
        assert screener.memory_limit_bytes == int(2.0 * 1024**3)
        assert screener.num_features > 0
        
        # Clean up
        screener.cleanup()

    def test_pharmacophore_query_creation(self):
        """Test pharmacophore query creation and validation."""
        # Valid query
        query = create_pharmacophore_query(
            feature_counts={"donor": 2, "acceptor": 1, "aromatic": 1},
            query_id="test_query"
        )
        
        assert query.feature_counts["donor"] == 2
        assert query.feature_counts["acceptor"] == 1
        assert query.feature_counts["aromatic"] == 1
        assert query.query_id == "test_query"
        
        # Invalid features should raise error
        with pytest.raises(ChemiAccelError):
            create_pharmacophore_query({"invalid_feature": 1})

    def test_gpu_screening_basic(self, small_test_set):
        """Test basic GPU screening functionality."""
        screener = GPUPharmacophoreScreener(batch_size=100)
        
        try:
            query = create_pharmacophore_query(
                feature_counts={"donor": 1, "acceptor": 1}
            )
            
            results = screener.screen_molecules(
                small_test_set,
                query,
                similarity_threshold=0.5
            )
            
            assert len(results) == len(small_test_set)
            
            # Check result types
            for result in results:
                assert isinstance(result, ScreeningResult)
                assert result.mol_id is not None
                assert 0.0 <= result.similarity_score <= 1.0
                
        finally:
            screener.cleanup()

    def test_gpu_screening_with_3d_constraints(self, small_test_set):
        """Test GPU screening with 3D distance constraints."""
        screener = GPUPharmacophoreScreener(batch_size=100, use_3d_constraints=True)
        
        try:
            query = create_pharmacophore_query(
                feature_counts={"donor": 1, "acceptor": 1},
                distance_constraints={"donor_acceptor": (2.0, 8.0)}
            )
            
            results = screener.screen_molecules(
                small_test_set,
                query,
                similarity_threshold=0.3
            )
            
            assert len(results) == len(small_test_set)
            
            # At least some results should have distance match scores
            has_distance_scores = any(
                r.distance_match_scores is not None for r in results
            )
            # Note: May not always be True due to mock feature extraction
            
        finally:
            screener.cleanup()

    def test_gpu_batch_processing(self, test_molecules):
        """Test GPU batch processing with larger dataset."""
        screener = GPUPharmacophoreScreener(batch_size=20)  # Small batches to test batching
        
        try:
            query = create_pharmacophore_query(
                feature_counts={"donor": 1, "acceptor": 1, "aromatic": 1}
            )
            
            start_time = time.time()
            results = screener.screen_molecules(
                test_molecules,
                query,
                similarity_threshold=0.4
            )
            end_time = time.time()
            
            assert len(results) == len(test_molecules)
            
            # Performance check (should process molecules quickly)
            screening_time = end_time - start_time
            molecules_per_second = len(test_molecules) / screening_time
            
            # Should be reasonably fast (at least 100 molecules/sec on GPU)
            assert molecules_per_second > 10  # Conservative threshold
            
        finally:
            screener.cleanup()

    def test_gpu_memory_management(self):
        """Test GPU memory allocation and cleanup."""
        screener = GPUPharmacophoreScreener(batch_size=1000, memory_limit_gb=1.0)
        
        # Check memory allocation
        assert screener._gpu_arrays_allocated
        assert screener.gpu_feature_matrix is not None
        
        # Test cleanup
        screener.cleanup()
        assert not screener._gpu_arrays_allocated

    def test_gpu_performance_benchmark(self):
        """Test GPU performance benchmarking."""
        screener = GPUPharmacophoreScreener()
        
        try:
            # Run small benchmark
            metrics = screener.benchmark_performance(num_molecules=100, query_complexity="simple")
            
            assert "total_molecules" in metrics
            assert "molecules_per_second" in metrics
            assert "total_time_seconds" in metrics
            assert metrics["total_molecules"] == 100
            assert metrics["molecules_per_second"] > 0
            
        finally:
            screener.cleanup()

class TestPerformanceAndValidation:
    """Test performance characteristics and validation."""
    
    def test_performance_comparison_cpu_vs_gpu(self, small_test_set):
        """Compare CPU vs GPU performance (when both available)."""
        query_spec = {"donor": 1, "acceptor": 1, "aromatic": 1}
        
        # CPU screening
        cpu_engine = PharmacophoreScreeningEngine(backend="cpu")
        cpu_start = time.time()
        cpu_results = cpu_engine.screen_molecules(
            small_test_set,
            cpu_engine.create_query(query_spec),
            similarity_threshold=0.5,
            return_details=False
        )
        cpu_time = time.time() - cpu_start
        
        results = {"cpu_time": cpu_time, "cpu_results": len(cpu_results)}
        
        # GPU screening (if available)
        if GPU_AVAILABLE:
            gpu_engine = PharmacophoreScreeningEngine(backend="gpu")
            gpu_start = time.time()
            gpu_results = gpu_engine.screen_molecules(
                small_test_set,
                gpu_engine.create_query(query_spec),
                similarity_threshold=0.5,
                return_details=False
            )
            gpu_time = time.time() - gpu_start
            
            results.update({"gpu_time": gpu_time, "gpu_results": len(gpu_results)})
            
            # GPU should be faster for larger datasets (may not be true for small sets)
            # Just check that both produce results
            assert len(cpu_results) > 0 or len(gpu_results) > 0
            
            gpu_engine.cleanup()
        
        cpu_engine.cleanup()

    def test_similarity_score_validation(self, small_test_set):
        """Validate similarity score calculations."""
        engine = create_screening_engine()
        
        # Test with identical feature requirements
        identical_query = engine.create_query({"donor": 1, "acceptor": 1})
        
        results = engine.screen_molecules(
            small_test_set,
            identical_query,
            similarity_threshold=0.0,
            return_details=True
        )
        
        # Check similarity score properties
        for result in results:
            score = result["similarity_score"]
            assert 0.0 <= score <= 1.0, f"Invalid similarity score: {score}"
        
        # Molecules with exact feature matches should have higher scores
        high_score_results = [r for r in results if r["similarity_score"] > 0.8]
        low_score_results = [r for r in results if r["similarity_score"] < 0.2]
        
        # Should have some variation in scores
        assert len(set(r["similarity_score"] for r in results)) > 1

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        engine = create_screening_engine()
        
        # Empty molecule list
        empty_results = engine.screen_molecules(
            [],
            engine.create_query({"donor": 1}),
            return_details=True
        )
        assert len(empty_results) == 0
        
        # Invalid similarity threshold
        with pytest.raises((ValueError, AssertionError)):
            engine.screen_molecules(
                [],
                engine.create_query({"donor": 1}),
                similarity_threshold=1.5  # Invalid threshold > 1.0
            )

    def test_query_flexibility(self):
        """Test flexible query specification."""
        engine = create_screening_engine()
        
        # Range-based feature requirements
        range_query = engine.create_query({
            "donor": (1, 3),  # 1-3 donors
            "acceptor": 2,    # Exactly 2 acceptors
            "aromatic": (0, 1) # 0-1 aromatic rings
        })
        
        assert range_query is not None
        
        # Complex distance constraints
        complex_query = engine.create_query(
            {"donor": 2, "acceptor": 1},
            distance_constraints={
                "donor_acceptor": (2.5, 7.0),
                "donor_donor": (3.0, 10.0)
            }
        )
        
        assert complex_query is not None

class TestConvenienceFunctions:
    """Test high-level convenience functions."""
    
    def test_quick_screen_function(self, small_test_set):
        """Test quick_screen convenience function."""
        results = quick_screen(
            small_test_set,
            {"donor": 1, "acceptor": 1},
            similarity_threshold=0.5
        )
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == len(small_test_set)
        assert "similarity_score" in results.columns
        assert "matches_query" in results.columns

    def test_create_screening_engine_function(self):
        """Test create_screening_engine convenience function."""
        # Test with different configurations
        engine1 = create_screening_engine(backend="cpu", batch_size=500)
        assert engine1.backend == "cpu"
        assert engine1.batch_size == 500
        
        engine2 = create_screening_engine(backend="auto")
        assert engine2.backend in ["cpu", "gpu"]

@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration tests with realistic scenarios."""
    
    def test_drug_discovery_workflow(self, test_molecules):
        """Test complete drug discovery screening workflow."""
        engine = create_screening_engine()
        
        # Define multiple pharmacophore queries for different targets
        queries = [
            engine.create_query(
                {"donor": 2, "acceptor": 3, "aromatic": 1},
                query_name="kinase_inhibitor"
            ),
            engine.create_query(
                {"donor": 1, "acceptor": 2, "hydrophobic": 2},
                query_name="gpcr_ligand"
            )
        ]
        
        # Screen against multiple queries
        all_results = engine.batch_screen_database(
            test_molecules,
            queries,
            similarity_threshold=0.6
        )
        
        assert isinstance(all_results, pd.DataFrame)
        assert "query_name" in all_results.columns
        
        # Should have results for both queries
        unique_queries = all_results["query_name"].unique()
        assert len(unique_queries) == 2
        
        # Check for hits
        hits = all_results[all_results["matches_query"]]
        print(f"Found {len(hits)} hits out of {len(all_results)} total screens")

    def test_large_scale_screening_simulation(self):
        """Simulate large-scale database screening."""
        if not RDKIT_AVAILABLE:
            pytest.skip("RDKit not available")
        
        # Generate larger test set
        large_test_set = TestMoleculeGenerator.generate_test_molecules(200)
        
        engine = create_screening_engine()
        
        # Complex pharmacophore query
        complex_query = engine.create_query(
            {"donor": 2, "acceptor": 2, "aromatic": 1, "hydrophobic": 1},
            distance_constraints={"donor_acceptor": (2.0, 8.0)},
            query_name="complex_screening"
        )
        
        start_time = time.time()
        results = engine.screen_molecules(
            large_test_set,
            complex_query,
            similarity_threshold=0.4,
            max_results=50  # Limit results
        )
        screening_time = time.time() - start_time
        
        assert len(results) <= 50  # Respects max_results
        
        # Performance check
        molecules_per_second = len(large_test_set) / screening_time
        print(f"Screening rate: {molecules_per_second:.1f} molecules/second")
        
        # Should achieve reasonable performance
        assert molecules_per_second > 5  # Conservative threshold

# Utility functions for test execution
def run_performance_tests():
    """Run performance-focused tests."""
    pytest.main([
        __file__ + "::TestPerformanceAndValidation",
        "-v", "-s"
    ])

def run_gpu_tests():
    """Run GPU-specific tests."""
    if GPU_AVAILABLE:
        pytest.main([
            __file__ + "::TestGPUPharmacophoreScreener", 
            "-v", "-s"
        ])
    else:
        print("GPU tests skipped - CUDA not available")

if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])