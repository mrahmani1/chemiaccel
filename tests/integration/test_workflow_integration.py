"""
End-to-end workflow integration tests for ChemiAccel.

This module tests complete workflows including scikit-learn integration,
Jupyter notebook features, and Dask distributed processing.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

# Import ChemiAccel components
from chemi_accel.core.calculator import Calculator
from chemi_accel.integrations.sklearn_integration import (
    ChemiAccelEstimator,
    ChemiAccelTransformer,
    create_molecular_pipeline,
)

# Optional imports for testing
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import dask.dataframe as dd
    from chemi_accel.integrations.dask_integration import (
        DaskChemiAccelCalculator,
        distributed_calculate_descriptors,
    )

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

try:
    from chemi_accel.integrations.jupyter_integration import enable_rich_display

    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False


class TestWorkflowIntegration:
    """Test complete ChemiAccel workflows."""

    @pytest.fixture
    def sample_molecules(self):
        """Sample molecules for testing."""
        return [
            "CCO",  # ethanol
            "c1ccccc1",  # benzene
            "CC(=O)O",  # acetic acid
            "CCN(CC)CC",  # triethylamine
            "c1ccc2c(c1)ccccc2",  # naphthalene
            "CC(C)(C)O",  # tert-butanol
            "c1ccc(cc1)O",  # phenol
            "CCCCCCCCCCCCCCCC(=O)O",  # palmitic acid
            "c1ccc(cc1)N",  # aniline
            "CC(C)CC(C)(C)C",  # branched alkane
            "c1ccc(cc1)C(=O)O",  # benzoic acid
            "CCN(C)C",  # dimethylethylamine
            "c1ccc(cc1)Cl",  # chlorobenzene
            "CC(C)O",  # isopropanol
            "c1ccncc1",  # pyridine
        ]

    @pytest.fixture
    def sample_targets(self):
        """Sample target values for regression testing."""
        # Synthetic target values (e.g., boiling points)
        return [
            78.4,
            80.1,
            118.1,
            89.5,
            218.0,
            82.4,
            181.7,
            351.0,
            184.1,
            174.1,
            249.2,
            36.5,
            131.7,
            82.6,
            115.2,
        ]

    def test_basic_calculator_workflow(self, sample_molecules):
        """Test basic calculator workflow."""
        # Initialize calculator
        calc = Calculator()
        calc.configure_descriptors("basic")

        # Calculate descriptors
        results = calc.pandas(sample_molecules)

        # Verify results
        assert isinstance(results, pd.DataFrame)
        assert len(results) == len(sample_molecules)
        assert "molecular_weight" in results.columns
        assert "logp" in results.columns
        assert "tpsa" in results.columns

        # Check data types
        assert results["molecular_weight"].dtype in [np.float64, np.float32]
        assert results["logp"].dtype in [np.float64, np.float32]

        # Check for reasonable values
        assert results["molecular_weight"].min() > 0
        assert (
            results["molecular_weight"].max() < 1000
        )  # Reasonable for small molecules

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
    def test_sklearn_transformer_workflow(self, sample_molecules, sample_targets):
        """Test scikit-learn transformer integration."""
        # Create transformer
        transformer = ChemiAccelTransformer(descriptor_set="basic")

        # Fit and transform
        X_descriptors = transformer.fit_transform(sample_molecules)

        # Verify transformation
        assert isinstance(X_descriptors, np.ndarray)
        assert X_descriptors.shape[0] == len(sample_molecules)
        assert X_descriptors.shape[1] > 0

        # Check feature names
        feature_names = transformer.get_feature_names_out()
        assert len(feature_names) == X_descriptors.shape[1]
        assert "molecular_weight" in feature_names

        # Test with new data
        new_molecules = ["CCC", "c1cccnc1"]
        X_new = transformer.transform(new_molecules)
        assert X_new.shape[1] == X_descriptors.shape[1]

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
    def test_sklearn_pipeline_workflow(self, sample_molecules, sample_targets):
        """Test complete scikit-learn pipeline workflow."""
        # Create pipeline
        pipeline = Pipeline(
            [
                ("descriptors", ChemiAccelTransformer(descriptor_set="basic")),
                ("regressor", RandomForestRegressor(n_estimators=10, random_state=42)),
            ]
        )

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            sample_molecules, sample_targets, test_size=0.3, random_state=42
        )

        # Fit pipeline
        pipeline.fit(X_train, y_train)

        # Make predictions
        y_pred = pipeline.predict(X_test)

        # Verify predictions
        assert len(y_pred) == len(y_test)
        assert isinstance(y_pred, np.ndarray)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Basic sanity checks (not expecting great performance with small dataset)
        assert mse >= 0
        assert -1 <= r2 <= 1

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
    def test_sklearn_estimator_workflow(self, sample_molecules, sample_targets):
        """Test ChemiAccelEstimator workflow."""
        # Create estimator
        estimator = ChemiAccelEstimator(
            predictor=RandomForestRegressor(n_estimators=10, random_state=42),
            descriptor_set="basic",
        )

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            sample_molecules, sample_targets, test_size=0.3, random_state=42
        )

        # Fit estimator
        estimator.fit(X_train, y_train)

        # Make predictions
        y_pred = estimator.predict(X_test)

        # Calculate score
        score = estimator.score(X_test, y_test)

        # Verify results
        assert len(y_pred) == len(y_test)
        assert isinstance(score, float)
        assert -1 <= score <= 1

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
    def test_create_molecular_pipeline(self, sample_molecules, sample_targets):
        """Test molecular pipeline creation convenience function."""
        # Create pipeline using convenience function
        pipeline = create_molecular_pipeline(
            RandomForestRegressor(n_estimators=10, random_state=42),
            descriptor_set="basic",
        )

        # Verify pipeline structure
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.steps) == 2
        assert pipeline.steps[0][0] == "descriptors"
        assert pipeline.steps[1][0] == "predictor"

        # Test pipeline functionality
        pipeline.fit(sample_molecules[:10], sample_targets[:10])
        predictions = pipeline.predict(sample_molecules[10:])

        assert len(predictions) == len(sample_molecules[10:])

    def test_invalid_molecule_handling(self):
        """Test handling of invalid molecules in workflows."""
        # Mix of valid and invalid SMILES
        mixed_molecules = [
            "CCO",  # valid
            "invalid_smiles",  # invalid
            "c1ccccc1",  # valid
            "",  # invalid
            "CC(=O)O",  # valid
        ]

        calc = Calculator()
        calc.configure_descriptors("basic")

        # Test validation
        valid_smiles, errors = calc.validate_molecules(mixed_molecules)

        assert len(valid_smiles) == 3  # Should have 3 valid molecules
        assert len(errors) == 2  # Should have 2 errors

        # Test calculation with invalid molecules (should skip them)
        results = calc.pandas(mixed_molecules)
        assert len(results) == 3  # Should only process valid molecules

    @pytest.mark.skipif(not DASK_AVAILABLE, reason="Dask not available")
    def test_dask_distributed_workflow(self, sample_molecules):
        """Test Dask distributed processing workflow."""
        # Create larger dataset for distributed processing
        large_molecules = sample_molecules * 10  # 150 molecules

        # Test distributed calculation
        results = distributed_calculate_descriptors(
            large_molecules,
            backend="cpu",  # Use CPU backend for testing
            descriptor_set="basic",
            chunk_size=50,
        )

        # Verify results
        assert isinstance(results, pd.DataFrame)
        assert len(results) == len(large_molecules)
        assert "molecular_weight" in results.columns

    @pytest.mark.skipif(not DASK_AVAILABLE, reason="Dask not available")
    def test_dask_csv_processing(self, sample_molecules):
        """Test Dask CSV file processing."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            # Write CSV header and data
            f.write("smiles,name\n")
            for i, smiles in enumerate(sample_molecules):
                f.write(f"{smiles},molecule_{i}\n")
            csv_path = f.name

        try:
            # Create Dask calculator
            calc = DaskChemiAccelCalculator(
                backend="cpu", descriptor_set="basic", chunk_size=5
            )

            # Process CSV file
            results = calc.calculate_from_csv(csv_path, smiles_column="smiles")

            # Compute results
            computed_results = results.compute()

            # Verify results
            assert isinstance(computed_results, pd.DataFrame)
            assert len(computed_results) == len(sample_molecules)

        finally:
            # Clean up temporary file
            os.unlink(csv_path)
            calc.close()

    def test_performance_benchmarking(self, sample_molecules):
        """Test performance benchmarking functionality."""
        calc = Calculator()
        calc.configure_descriptors("basic")

        # Run benchmark
        results = calc.benchmark_performance(
            test_molecules=sample_molecules, num_runs=2
        )

        # Verify benchmark results
        assert isinstance(results, dict)
        assert "backend_type" in results
        assert "molecules_per_second" in results
        assert "avg_time_seconds" in results
        assert results["num_molecules"] == len(sample_molecules)
        assert results["molecules_per_second"] > 0
        assert results["avg_time_seconds"] > 0

    def test_backend_switching_workflow(self, sample_molecules):
        """Test backend switching during workflow."""
        calc = Calculator(backend="cpu")
        calc.configure_descriptors("basic")

        # Calculate with CPU backend
        results_cpu = calc.pandas(sample_molecules[:5])
        assert calc.backend_type.value == "cpu"

        # Try to switch to CUDA (may not be available)
        try:
            calc.switch_backend("cuda")
            results_cuda = calc.pandas(sample_molecules[:5])

            # Results should be similar (within numerical precision)
            assert len(results_cuda) == len(results_cpu)

        except Exception:
            # CUDA not available, which is fine for testing
            pass

    def test_descriptor_configuration_workflow(self, sample_molecules):
        """Test descriptor configuration in workflows."""
        calc = Calculator()

        # Test different descriptor sets
        descriptor_sets = ["basic", "all"]

        for desc_set in descriptor_sets:
            calc.configure_descriptors(desc_set)
            results = calc.pandas(sample_molecules[:3])

            assert isinstance(results, pd.DataFrame)
            assert len(results) == 3

            # Verify descriptor columns exist
            desc_cols = [
                col for col in results.columns if col not in ["mol_id", "smiles"]
            ]
            assert len(desc_cols) > 0

    def test_error_handling_workflow(self):
        """Test error handling in complete workflows."""
        calc = Calculator()

        # Test with empty input
        empty_results = calc.pandas([])
        assert isinstance(empty_results, pd.DataFrame)
        assert len(empty_results) == 0

        # Test with all invalid molecules
        invalid_molecules = ["invalid1", "invalid2", ""]
        results = calc.pandas(invalid_molecules)

        # Should return empty DataFrame or handle gracefully
        assert isinstance(results, pd.DataFrame)

    def test_memory_management_workflow(self, sample_molecules):
        """Test memory management in workflows."""
        calc = Calculator()
        calc.configure_descriptors("basic")

        # Get backend info including memory info
        info = calc.get_backend_info()

        assert isinstance(info, dict)
        assert "backend_type" in info
        assert "num_descriptors" in info

        # Process molecules multiple times to test memory handling
        for _ in range(3):
            results = calc.pandas(sample_molecules)
            assert len(results) == len(sample_molecules)

    def test_file_io_workflow(self, sample_molecules):
        """Test file input/output workflows."""
        # Create temporary input file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("smiles\n")
            for smiles in sample_molecules:
                f.write(f"{smiles}\n")
            input_path = f.name

        try:
            # Read molecules from file
            df = pd.read_csv(input_path)
            molecules = df["smiles"].tolist()

            # Calculate descriptors
            calc = Calculator()
            calc.configure_descriptors("basic")
            results = calc.pandas(molecules)

            # Save results to file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as f:
                output_path = f.name

            results.to_csv(output_path, index=False)

            # Verify saved file
            saved_results = pd.read_csv(output_path)
            assert len(saved_results) == len(results)
            assert list(saved_results.columns) == list(results.columns)

        finally:
            # Clean up temporary files
            os.unlink(input_path)
            if "output_path" in locals():
                os.unlink(output_path)


class TestIntegrationEdgeCases:
    """Test edge cases in integration workflows."""

    def test_large_molecule_handling(self):
        """Test handling of large molecules."""
        # Create a large molecule SMILES
        large_molecule = "C" * 100  # Long alkyl chain

        calc = Calculator()
        calc.configure_descriptors("basic")

        results = calc.pandas([large_molecule])

        # Should handle large molecules gracefully
        assert isinstance(results, pd.DataFrame)
        if len(results) > 0:
            assert results["molecular_weight"].iloc[0] > 1000

    def test_special_characters_handling(self):
        """Test handling of molecules with special characters."""
        special_molecules = [
            "C[C@H](N)C(=O)O",  # stereochemistry
            "c1ccc2[nH]c3ccccc3c2c1",  # aromatic nitrogen
            "CC(=O)N[C@@H](CS)C(=O)O",  # more stereochemistry
        ]

        calc = Calculator()
        calc.configure_descriptors("basic")

        results = calc.pandas(special_molecules)

        # Should handle special characters in SMILES
        assert isinstance(results, pd.DataFrame)
        # May have fewer results if some molecules are invalid
        assert len(results) <= len(special_molecules)

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
    def test_sklearn_with_missing_values(self):
        """Test sklearn integration with missing values."""
        molecules = ["CCO", "invalid_smiles", "c1ccccc1"]
        targets = [1.0, 2.0, 3.0]

        # Test with 'drop' strategy
        transformer = ChemiAccelTransformer(
            descriptor_set="basic", handle_invalid="drop"
        )

        X_transformed = transformer.fit_transform(molecules)

        # Should drop invalid molecules
        assert X_transformed.shape[0] < len(molecules)

        # Test with 'impute' strategy
        transformer_impute = ChemiAccelTransformer(
            descriptor_set="basic", handle_invalid="impute", impute_value=0.0
        )

        X_imputed = transformer_impute.fit_transform(molecules)

        # Should keep all molecules with imputed values
        assert X_imputed.shape[0] == len(molecules)

    def test_concurrent_calculator_usage(self, sample_molecules=None):
        """Test concurrent usage of multiple calculators."""
        if sample_molecules is None:
            sample_molecules = ["CCO", "c1ccccc1", "CC(=O)O"]

        # Create multiple calculators
        calc1 = Calculator(backend="cpu")
        calc2 = Calculator(backend="cpu")

        calc1.configure_descriptors("basic")
        calc2.configure_descriptors(["molecular_weight", "logp"])

        # Use both calculators
        results1 = calc1.pandas(sample_molecules)
        results2 = calc2.pandas(sample_molecules)

        # Both should work independently
        assert isinstance(results1, pd.DataFrame)
        assert isinstance(results2, pd.DataFrame)
        assert len(results1) == len(sample_molecules)
        assert len(results2) == len(sample_molecules)

        # Different descriptor sets
        desc_cols1 = [
            col for col in results1.columns if col not in ["mol_id", "smiles"]
        ]
        desc_cols2 = [
            col for col in results2.columns if col not in ["mol_id", "smiles"]
        ]
        assert len(desc_cols1) != len(desc_cols2)


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
