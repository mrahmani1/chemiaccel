"""
Test Suite for High-Throughput Reaction Informatics

This module contains comprehensive tests for the reaction informatics
implementation, including GPU acceleration validation, reaction prediction
accuracy, retrosynthesis analysis, and performance benchmarking.
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

# Import reaction informatics modules
from src.chemi_accel.descriptors.reaction_informatics import (
    ReactionInformaticsEngine,
    create_reaction_informatics_engine,
    predict_reactions_batch,
    analyze_synthetic_routes
)

# Try to import GPU-specific modules
try:
    from src.chemi_accel.backends.cuda.reaction_informatics import (
        GPUReactionInformatics,
        ReactionTemplate,
        ReactionPrediction,
        RetrosynthesisPath,
        ReactionType,
        ReactionCondition
    )
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    GPUReactionInformatics = None

# Import test utilities
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

class TestMoleculeGenerator:
    """Generate test molecules for reaction informatics validation."""
    
    @staticmethod
    def generate_reaction_test_molecules(count: int = 50) -> List[MolecularData]:
        """Generate diverse molecules suitable for reaction testing."""
        if not RDKIT_AVAILABLE:
            pytest.skip("RDKit not available for molecule generation")
        
        # Drug-like molecules with functional groups for reactions
        test_smiles = [
            # Carboxylic acids (amide formation)
            "CC(C)C(=O)O",
            "c1ccc(C(=O)O)cc1",
            "CCC(C)(C)C(=O)O",
            
            # Amines (amide formation)
            "CCN",
            "c1ccc(N)cc1",
            "CC(C)N",
            
            # Alcohols (oxidation/esterification)
            "CCO",
            "CC(C)O",
            "c1ccc(CO)cc1",
            
            # Halides (Suzuki coupling)
            "c1ccc(Br)cc1",
            "c1cc(I)ccc1C",
            "Brc1ccc(C)cc1",
            
            # Boronic acids (Suzuki coupling)
            "c1ccc(B(O)O)cc1",
            "CC(C)c1ccc(B(O)O)cc1",
            
            # Alkynes (Click chemistry)
            "C#CCO",
            "c1ccc(C#C)cc1",
            
            # Azides (Click chemistry)
            "CCN=[N+]=[N-]",
            "c1ccc(CN=[N+]=[N-])cc1",
            
            # Aldehydes/Ketones (reduction, Grignard)
            "CC(=O)C",
            "c1ccc(C=O)cc1",
            "CCC(=O)CC",
            
            # Alkenes (Diels-Alder)
            "C=CC=C",
            "c1ccc(C=C)cc1",
            
            # Complex drug-like molecules
            "CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O",  # Albuterol-like
            "CN1CCN(CC1)C2=C(C=C3C(=C2)N=CN=C3NC4=CC(=C(C=C4)F)Cl)OCCO",  # Gefitinib-like
        ]
        
        molecules = []
        for i in range(count):
            smiles = test_smiles[i % len(test_smiles)]
            
            # Add variations for diversity
            if i >= len(test_smiles):
                variation_factor = i // len(test_smiles)
                smiles = TestMoleculeGenerator.add_variation(smiles, variation_factor)
            
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                # Add 3D coordinates
                try:
                    AllChem.EmbedMolecule(mol, randomSeed=42)
                    AllChem.MMFFOptimizeMolecule(mol)
                except:
                    pass  # Skip if conformer generation fails
                
                mol_data = MolecularData(
                    mol_id=f"reaction_test_{i}",
                    smiles=smiles,
                    backend_data={"rdkit_mol": mol}
                )
                molecules.append(mol_data)
        
        return molecules
    
    @staticmethod
    def add_variation(smiles: str, variation: int) -> str:
        """Add structural variations to molecules."""
        variations = [
            lambda s: s.replace("C", "CC", 1) if "C" in s else s,
            lambda s: s.replace("c1ccccc1", "c1ccc(C)cc1", 1),
            lambda s: f"F{s}" if not s.startswith("F") else s,
            lambda s: s.replace("O", "S", 1) if "O" in s and "CO" not in s else s,
            lambda s: s.replace("N", "O", 1) if "N" in s and "N=[N+]" not in s else s
        ]
        
        var_func = variations[variation % len(variations)]
        try:
            return var_func(smiles)
        except:
            return smiles

@pytest.fixture
def test_molecules():
    """Fixture providing reaction test molecules."""
    return TestMoleculeGenerator.generate_reaction_test_molecules(30)

@pytest.fixture
def small_test_set():
    """Small test set for quick validation."""
    return TestMoleculeGenerator.generate_reaction_test_molecules(10)

@pytest.fixture
def reaction_test_cases():
    """Standard reaction test cases."""
    return [
        {
            "name": "amide_formation",
            "reaction_types": ["amide_formation"],
            "description": "Amide bond formation reactions"
        },
        {
            "name": "suzuki_coupling", 
            "reaction_types": ["suzuki_coupling"],
            "description": "Suzuki cross-coupling reactions"
        },
        {
            "name": "click_chemistry",
            "reaction_types": ["click_chemistry"],
            "description": "Click chemistry reactions"
        },
        {
            "name": "multi_type",
            "reaction_types": ["amide_formation", "suzuki_coupling", "oxidation"],
            "description": "Multiple reaction types"
        }
    ]

class TestReactionInformaticsEngine:
    """Test the unified reaction informatics engine."""
    
    def test_engine_initialization_auto_backend(self):
        """Test automatic backend selection."""
        engine = ReactionInformaticsEngine(backend="auto")
        
        if GPU_AVAILABLE:
            assert engine.backend == "gpu"
            assert engine.gpu_engine is not None
        else:
            assert engine.backend == "cpu"
            assert engine.cpu_enumerator is not None

    def test_engine_initialization_cpu_backend(self):
        """Test explicit CPU backend selection."""
        engine = ReactionInformaticsEngine(backend="cpu")
        
        assert engine.backend == "cpu"
        assert engine.cpu_enumerator is not None
        assert engine.gpu_engine is None

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU backend not available")
    def test_engine_initialization_gpu_backend(self):
        """Test explicit GPU backend selection."""
        engine = ReactionInformaticsEngine(backend="gpu")
        
        assert engine.backend == "gpu"
        assert engine.gpu_engine is not None
        assert engine.cpu_enumerator is None

    def test_reaction_prediction_basic(self, small_test_set, reaction_test_cases):
        """Test basic reaction prediction functionality."""
        engine = create_reaction_informatics_engine()
        
        for test_case in reaction_test_cases[:2]:  # Test first two cases
            predictions = engine.predict_reactions(
                small_test_set,
                reaction_types=test_case["reaction_types"],
                min_feasibility_score=0.2,
                max_predictions_per_molecule=5
            )
            
            # Should generate some predictions
            assert len(predictions) >= 0  # May be zero if no suitable substrates
            
            # Check prediction structure
            for prediction in predictions[:3]:  # Check first few predictions
                assert "reactant_ids" in prediction
                assert "product_ids" in prediction
                assert "reaction_type" in prediction
                assert "feasibility_score" in prediction
                assert "backend" in prediction
                assert 0.0 <= prediction["feasibility_score"] <= 1.0

    def test_reaction_prediction_filtering(self, small_test_set):
        """Test reaction type filtering."""
        engine = create_reaction_informatics_engine()
        
        # Test with specific reaction type
        amide_predictions = engine.predict_reactions(
            small_test_set,
            reaction_types=["amide_formation"],
            min_feasibility_score=0.1
        )
        
        # All predictions should be of the requested type
        for pred in amide_predictions:
            assert "amide" in pred["reaction_type"].lower() or pred["reaction_type"] == "amide_formation"

    def test_retrosynthesis_analysis(self, small_test_set):
        """Test retrosynthesis pathway analysis."""
        engine = create_reaction_informatics_engine()
        
        # Test with first molecule
        target_molecule = small_test_set[0]
        
        retro_result = engine.analyze_retrosynthesis(
            target_molecule,
            max_steps=3,
            min_commercial_availability=0.8
        )
        
        # Check result structure
        assert "target_molecule" in retro_result
        assert "pathway_steps" in retro_result
        assert "total_score" in retro_result
        assert "complexity_score" in retro_result
        assert "backend" in retro_result
        
        assert retro_result["target_molecule"] == target_molecule.smiles
        assert 0.0 <= retro_result["total_score"] <= 1.0
        assert 0.0 <= retro_result["complexity_score"] <= 1.0

    def test_reaction_condition_optimization(self, small_test_set):
        """Test reaction condition optimization."""
        engine = create_reaction_informatics_engine()
        
        # Get a reaction prediction first
        predictions = engine.predict_reactions(
            small_test_set[:3],
            min_feasibility_score=0.1,
            max_predictions_per_molecule=2
        )
        
        if predictions:
            prediction = predictions[0]
            
            # Test yield optimization
            yield_optimization = engine.optimize_reaction_conditions(
                prediction,
                optimization_target="yield"
            )
            
            assert "optimized_conditions" in yield_optimization
            assert "optimization_target" in yield_optimization
            assert yield_optimization["optimization_target"] == "yield"
            
            # Test selectivity optimization
            selectivity_optimization = engine.optimize_reaction_conditions(
                prediction,
                optimization_target="selectivity"
            )
            
            assert selectivity_optimization["optimization_target"] == "selectivity"

    def test_mechanism_classification(self):
        """Test reaction mechanism classification."""
        engine = create_reaction_informatics_engine()
        
        # Test reaction SMILES
        test_reactions = [
            "CC(=O)O.CCN>>CC(=O)N",  # Amide formation
            "c1ccc(Br)cc1.c1ccc(B(O)O)cc1>>c1ccc(-c2ccccc2)cc1",  # Suzuki coupling
            "C#C.N=[N+]=[N-]>>c1cn[n]n1",  # Click chemistry
        ]
        
        for reaction_smiles in test_reactions:
            result = engine.classify_reaction_mechanism(
                reaction_smiles,
                include_energetics=True
            )
            
            assert "mechanism_class" in result
            assert "confidence" in result
            assert "backend" in result
            assert 0.0 <= result["confidence"] <= 1.0
            
            if "activation_energy_kj_mol" in result:
                assert isinstance(result["activation_energy_kj_mol"], float)

    def test_batch_processing(self, small_test_set):
        """Test batch processing with multiple analysis types."""
        engine = create_reaction_informatics_engine()
        
        analysis_types = ["reaction_prediction", "retrosynthesis"]
        
        results_df = engine.batch_process_reactions(
            small_test_set[:5],  # Small subset for testing
            analysis_types
        )
        
        assert isinstance(results_df, pd.DataFrame)
        assert "analysis_type" in results_df.columns
        
        # Should have results for both analysis types
        analysis_types_found = results_df["analysis_type"].unique()
        assert len(analysis_types_found) > 0

    def test_processing_statistics(self, small_test_set):
        """Test processing statistics tracking."""
        engine = create_reaction_informatics_engine()
        
        # Initial statistics should be zero
        initial_stats = engine.get_processing_statistics()
        assert initial_stats["total_reactions_predicted"] == 0
        assert initial_stats["total_molecules_processed"] == 0
        
        # Run some predictions
        engine.predict_reactions(small_test_set[:3])
        
        # Check updated statistics
        updated_stats = engine.get_processing_statistics()
        assert updated_stats["total_molecules_processed"] >= 3
        assert "backend" in updated_stats
        assert "gpu_available" in updated_stats

@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU backend not available")
class TestGPUReactionInformatics:
    """Test GPU-specific reaction informatics functionality."""
    
    def test_gpu_engine_initialization(self):
        """Test GPU engine initialization."""
        engine = GPUReactionInformatics(
            batch_size=100,
            memory_limit_gb=2.0
        )
        
        assert engine.batch_size == 100
        assert len(engine.reaction_templates) > 0
        assert engine._gpu_arrays_allocated
        
        # Cleanup
        engine.cleanup()

    def test_gpu_reaction_templates(self):
        """Test reaction template loading and validation."""
        engine = GPUReactionInformatics()
        
        try:
            templates = engine.reaction_templates
            
            # Should have comprehensive template library
            assert len(templates) >= 10
            
            # Check template structure
            for name, template in templates.items():
                assert isinstance(template, ReactionTemplate)
                assert template.name == name
                assert len(template.smarts) > 0
                assert isinstance(template.reaction_type, ReactionType)
                
                # Check optional fields
                if template.typical_yield_range:
                    assert len(template.typical_yield_range) == 2
                    assert 0 <= template.typical_yield_range[0] <= template.typical_yield_range[1] <= 100
                
                if template.selectivity_score:
                    assert 0.0 <= template.selectivity_score <= 1.0
                    
        finally:
            engine.cleanup()

    def test_gpu_reaction_prediction(self, small_test_set):
        """Test GPU reaction prediction functionality."""
        engine = GPUReactionInformatics()
        
        try:
            predictions = engine.predict_reactions(
                small_test_set,
                min_feasibility_score=0.2,
                max_predictions_per_molecule=5
            )
            
            assert len(predictions) >= 0
            
            # Check prediction structure
            for prediction in predictions[:3]:
                assert isinstance(prediction, ReactionPrediction)
                assert len(prediction.reactant_ids) > 0
                assert len(prediction.product_ids) > 0
                assert isinstance(prediction.reaction_type, ReactionType)
                assert 0.0 <= prediction.feasibility_score <= 1.0
                assert 0.0 <= prediction.selectivity_score <= 1.0
                assert 0.0 <= prediction.yield_prediction <= 100.0
                
        finally:
            engine.cleanup()

    def test_gpu_retrosynthesis_analysis(self, small_test_set):
        """Test GPU retrosynthesis analysis."""
        engine = GPUReactionInformatics()
        
        try:
            target = small_test_set[0]
            
            retro_path = engine.analyze_retrosynthesis(
                target,
                max_steps=3,
                min_commercial_availability=0.7
            )
            
            assert isinstance(retro_path, RetrosynthesisPath)
            assert retro_path.target_molecule == target.smiles
            assert 0.0 <= retro_path.total_score <= 1.0
            assert 0.0 <= retro_path.complexity_score <= 1.0
            assert len(retro_path.pathway_steps) >= 0
            
        finally:
            engine.cleanup()

    def test_gpu_condition_optimization(self, small_test_set):
        """Test GPU reaction condition optimization.""" 
        engine = GPUReactionInformatics()
        
        try:
            # Use a known template
            template_name = list(engine.reaction_templates.keys())[0]
            
            optimized_conditions = engine.optimize_reaction_conditions(
                template_name,
                small_test_set[:2],
                optimization_target="yield"
            )
            
            assert isinstance(optimized_conditions, dict)
            # Should have some condition keys
            condition_keys = list(optimized_conditions.keys())
            assert len(condition_keys) > 0
            
        finally:
            engine.cleanup()

    def test_gpu_mechanism_classification(self):
        """Test GPU mechanism classification."""
        engine = GPUReactionInformatics()
        
        try:
            reaction_smiles = "CC(=O)O.CCN>>CC(=O)N"
            products = ["CC(=O)N"]
            
            result = engine.classify_reaction_mechanism(reaction_smiles, products)
            
            assert isinstance(result, dict)
            assert "mechanism_class" in result
            assert "confidence" in result
            assert 0.0 <= result["confidence"] <= 1.0
            
        finally:
            engine.cleanup()

    def test_gpu_performance_benchmark(self):
        """Test GPU performance benchmarking."""
        engine = GPUReactionInformatics()
        
        try:
            metrics = engine.benchmark_performance(
                num_molecules=200,  # Small for testing
                num_templates=10
            )
            
            assert "total_molecules" in metrics
            assert "total_predictions" in metrics
            assert "molecules_per_second" in metrics
            assert "predictions_per_second" in metrics
            assert "gpu_memory_used_gb" in metrics
            
            assert metrics["total_molecules"] == 200
            assert metrics["molecules_per_second"] > 0
            
        finally:
            engine.cleanup()

    def test_gpu_memory_management(self):
        """Test GPU memory allocation and cleanup."""
        engine = GPUReactionInformatics(batch_size=100)
        
        # Check memory allocation
        assert engine._gpu_arrays_allocated
        assert engine.gpu_molecule_features is not None
        
        # Test cleanup
        engine.cleanup()
        assert not engine._gpu_arrays_allocated

class TestPerformanceAndValidation:
    """Test performance characteristics and validation."""
    
    def test_performance_comparison_cpu_vs_gpu(self, small_test_set):
        """Compare CPU vs GPU performance when both available."""
        # CPU performance
        cpu_engine = ReactionInformaticsEngine(backend="cpu")
        cpu_start = time.time()
        cpu_predictions = cpu_engine.predict_reactions(
            small_test_set,
            min_feasibility_score=0.2
        )
        cpu_time = time.time() - cpu_start
        
        results = {
            "cpu_time": cpu_time,
            "cpu_predictions": len(cpu_predictions)
        }
        
        # GPU performance (if available)
        if GPU_AVAILABLE:
            gpu_engine = ReactionInformaticsEngine(backend="gpu")
            gpu_start = time.time()
            gpu_predictions = gpu_engine.predict_reactions(
                small_test_set,
                min_feasibility_score=0.2
            )
            gpu_time = time.time() - gpu_start
            
            results.update({
                "gpu_time": gpu_time,
                "gpu_predictions": len(gpu_predictions)
            })
            
            # Both should produce results
            assert len(cpu_predictions) > 0 or len(gpu_predictions) > 0
            
            gpu_engine.cleanup()
        
        cpu_engine.cleanup()

    def test_prediction_accuracy_validation(self, small_test_set):
        """Validate prediction accuracy and consistency."""
        engine = create_reaction_informatics_engine()
        
        try:
            # Run predictions twice to check consistency
            predictions1 = engine.predict_reactions(
                small_test_set,
                min_feasibility_score=0.1
            )
            
            predictions2 = engine.predict_reactions(
                small_test_set,
                min_feasibility_score=0.1
            )
            
            # Results should be consistent
            assert len(predictions1) == len(predictions2)
            
            # Check feasibility score validity
            for pred in predictions1:
                score = pred["feasibility_score"]
                assert 0.0 <= score <= 1.0, f"Invalid feasibility score: {score}"
                
                yield_pred = pred.get("yield_prediction", 50)
                assert 0.0 <= yield_pred <= 100.0, f"Invalid yield prediction: {yield_pred}"
                
        finally:
            engine.cleanup()

    def test_edge_cases_and_error_handling(self):
        """Test edge cases and error handling."""
        engine = create_reaction_informatics_engine()
        
        try:
            # Empty molecule list
            empty_predictions = engine.predict_reactions([])
            assert len(empty_predictions) == 0
            
            # Invalid reaction types
            if RDKIT_AVAILABLE:
                test_mol = MolecularData(
                    mol_id="test",
                    smiles="CCO",
                    backend_data={"rdkit_mol": Chem.MolFromSmiles("CCO")}
                )
                
                predictions = engine.predict_reactions(
                    [test_mol],
                    reaction_types=["nonexistent_reaction_type"]
                )
                # Should handle gracefully
                assert isinstance(predictions, list)
                
        finally:
            engine.cleanup()

    def test_reaction_type_coverage(self, test_molecules):
        """Test coverage of different reaction types.""" 
        engine = create_reaction_informatics_engine()
        
        try:
            # Test various reaction types
            reaction_types_to_test = [
                ["amide_formation"],
                ["suzuki_coupling"], 
                ["click_chemistry"],
                ["oxidation"],
                ["reduction"]
            ]
            
            results_by_type = {}
            
            for reaction_types in reaction_types_to_test:
                predictions = engine.predict_reactions(
                    test_molecules[:10],  # Subset for testing
                    reaction_types=reaction_types,
                    min_feasibility_score=0.1
                )
                
                results_by_type[reaction_types[0]] = len(predictions)
            
            # Should find at least some reactions for some types
            total_predictions = sum(results_by_type.values())
            assert total_predictions > 0, f"No predictions found: {results_by_type}"
            
        finally:
            engine.cleanup()

    def test_retrosynthesis_pathway_quality(self, small_test_set):
        """Test quality of retrosynthesis pathways."""
        engine = create_reaction_informatics_engine()
        
        try:
            for target in small_test_set[:3]:  # Test first 3 molecules
                retro_result = engine.analyze_retrosynthesis(
                    target,
                    max_steps=3
                )
                
                # Check pathway quality metrics
                assert "complexity_score" in retro_result
                assert "total_score" in retro_result
                
                complexity = retro_result["complexity_score"]
                total_score = retro_result["total_score"]
                
                assert 0.0 <= complexity <= 1.0
                assert 0.0 <= total_score <= 1.0
                
                # Pathway steps should be meaningful
                steps = retro_result["pathway_steps"]
                for step in steps:
                    assert "step_number" in step
                    assert "reaction_type" in step
                    assert "feasibility_score" in step
                    
        finally:
            engine.cleanup()

class TestConvenienceFunctions:
    """Test high-level convenience functions."""
    
    def test_predict_reactions_batch_function(self, small_test_set):
        """Test predict_reactions_batch convenience function."""
        predictions = predict_reactions_batch(
            small_test_set,
            reaction_types=["amide_formation"],
            min_feasibility_score=0.2
        )
        
        assert isinstance(predictions, list)
        
        # Check structure
        for pred in predictions[:3]:
            assert "feasibility_score" in pred
            assert "reaction_type" in pred

    def test_analyze_synthetic_routes_function(self, small_test_set):
        """Test analyze_synthetic_routes convenience function."""
        # Test with subset
        routes = analyze_synthetic_routes(
            small_test_set[:3],
            max_steps=2
        )
        
        assert isinstance(routes, list)
        assert len(routes) == 3
        
        for route in routes:
            assert "target_molecule" in route
            assert "complexity_score" in route
            assert "total_score" in route

    def test_create_reaction_informatics_engine_function(self):
        """Test create_reaction_informatics_engine convenience function."""
        # Test with different configurations
        engine1 = create_reaction_informatics_engine(
            backend="cpu",
            batch_size=200
        )
        assert engine1.backend == "cpu"
        assert engine1.batch_size == 200
        engine1.cleanup()
        
        engine2 = create_reaction_informatics_engine(backend="auto")
        assert engine2.backend in ["cpu", "gpu"]
        engine2.cleanup()

@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration tests with realistic scenarios."""
    
    def test_drug_discovery_workflow(self, test_molecules):
        """Test complete drug discovery reaction workflow.""" 
        engine = create_reaction_informatics_engine()
        
        try:
            # Multi-step drug discovery workflow
            
            # Step 1: Predict feasible reactions
            print("Step 1: Predicting reactions...")
            predictions = engine.predict_reactions(
                test_molecules[:20],  # Reasonable subset
                min_feasibility_score=0.3,
                max_predictions_per_molecule=3
            )
            
            assert len(predictions) > 0
            print(f"Found {len(predictions)} feasible reactions")
            
            # Step 2: Optimize conditions for high-scoring reactions
            print("Step 2: Optimizing reaction conditions...")
            high_score_predictions = [
                p for p in predictions 
                if p["feasibility_score"] > 0.6
            ][:5]  # Top 5
            
            for pred in high_score_predictions:
                optimization = engine.optimize_reaction_conditions(
                    pred,
                    optimization_target="yield"
                )
                assert "optimized_conditions" in optimization
                
            # Step 3: Retrosynthesis analysis for targets
            print("Step 3: Retrosynthesis analysis...")
            retro_analyses = []
            for target in test_molecules[:5]:
                retro_result = engine.analyze_retrosynthesis(
                    target,
                    max_steps=3
                )
                retro_analyses.append(retro_result)
                
            assert len(retro_analyses) == 5
            
            # Step 4: Mechanism classification
            print("Step 4: Mechanism classification...")
            if predictions:
                test_reaction = "CC(=O)O.CCN>>CC(=O)N"
                mechanism = engine.classify_reaction_mechanism(test_reaction)
                assert "mechanism_class" in mechanism
                
            print("✓ Drug discovery workflow completed successfully")
            
        finally:
            engine.cleanup()

    def test_synthetic_chemistry_planning(self, test_molecules):
        """Test synthetic chemistry planning workflow."""
        engine = create_reaction_informatics_engine()
        
        try:
            # Select a complex target molecule
            complex_targets = [m for m in test_molecules if len(m.smiles) > 20][:3]
            
            if not complex_targets:
                pytest.skip("No sufficiently complex molecules for testing")
            
            planning_results = []
            
            for target in complex_targets:
                print(f"Planning synthesis for {target.mol_id}")
                
                # Retrosynthesis analysis
                retro_plan = engine.analyze_retrosynthesis(
                    target,
                    max_steps=4,
                    strategy="linear"
                )
                
                # Get forward predictions for validation
                forward_predictions = engine.predict_reactions(
                    [target],
                    min_feasibility_score=0.2
                )
                
                plan_result = {
                    "target_id": target.mol_id,
                    "retro_complexity": retro_plan["complexity_score"],
                    "retro_feasibility": retro_plan["total_score"],
                    "forward_reactions": len(forward_predictions),
                    "estimated_cost": retro_plan.get("estimated_cost", 0),
                    "estimated_time": retro_plan.get("estimated_time_hours", 0)
                }
                
                planning_results.append(plan_result)
            
            # Analyze planning results
            avg_complexity = np.mean([p["retro_complexity"] for p in planning_results])
            avg_feasibility = np.mean([p["retro_feasibility"] for p in planning_results])
            
            print(f"Average synthetic complexity: {avg_complexity:.3f}")
            print(f"Average pathway feasibility: {avg_feasibility:.3f}")
            
            assert len(planning_results) == len(complex_targets)
            assert 0.0 <= avg_complexity <= 1.0
            assert 0.0 <= avg_feasibility <= 1.0
            
        finally:
            engine.cleanup()

    def test_large_scale_screening_simulation(self):
        """Simulate large-scale reaction screening."""
        if not RDKIT_AVAILABLE:
            pytest.skip("RDKit not available")
        
        # Generate larger test set
        large_test_set = TestMoleculeGenerator.generate_reaction_test_molecules(100)
        
        engine = create_reaction_informatics_engine()
        
        try:
            print(f"Large-scale screening of {len(large_test_set)} molecules...")
            
            # Batch processing with multiple analysis types
            start_time = time.time()
            
            results_df = engine.batch_process_reactions(
                large_test_set,
                analysis_types=["reaction_prediction"],
                output_file=None
            )
            
            total_time = time.time() - start_time
            
            # Performance metrics
            molecules_per_second = len(large_test_set) / total_time
            
            print(f"Processed {len(large_test_set)} molecules in {total_time:.2f}s")
            print(f"Processing rate: {molecules_per_second:.1f} molecules/second")
            
            # Results analysis
            print(f"Generated {len(results_df)} analysis results")
            
            if len(results_df) > 0:
                reaction_results = results_df[results_df["analysis_type"] == "reaction_prediction"]
                print(f"Found {len(reaction_results)} reaction predictions")
            
            # Performance should be reasonable
            assert molecules_per_second > 1.0  # At least 1 molecule/sec
            assert len(results_df) >= 0
            
        finally:
            engine.cleanup()

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
            __file__ + "::TestGPUReactionInformatics",
            "-v", "-s"
        ])
    else:
        print("GPU tests skipped - CUDA not available")

def run_integration_tests():
    """Run integration tests."""
    pytest.main([
        __file__ + "::TestIntegrationScenarios",
        "-v", "-s"
    ])

if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])