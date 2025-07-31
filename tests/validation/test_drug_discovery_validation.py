"""
Validation tests for drug discovery descriptors against medicinal chemistry datasets.

This module provides comprehensive validation of drug discovery descriptor calculations
against known drug-like compounds and established medicinal chemistry benchmarks.
"""

import numpy as np
import pandas as pd
import pytest

try:
    from rdkit import Chem

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from src.chemi_accel.descriptors.drug_discovery import DrugDiscoveryDescriptorCalculator


@pytest.fixture
def drug_discovery_calculator():
    """Fixture providing drug discovery descriptor calculator."""
    if not RDKIT_AVAILABLE:
        pytest.skip("RDKit not available")
    return DrugDiscoveryDescriptorCalculator()


@pytest.fixture
def drug_molecules():
    """Fixture providing drug molecules with known properties."""
    return {
        "aspirin": {
            "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "expected": {
                "qed_score": 0.7,  # Good drug-likeness
                "lipinski_violations": 0,  # No violations
                "synthetic_accessibility": 2.0,  # Easy to synthesize
                "lead_likeness": 0.8,  # Good lead-like properties
            },
        },
        "caffeine": {
            "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "expected": {
                "qed_score": 0.6,  # Moderate drug-likeness
                "cns_mpo_score": 4.0,  # Good CNS properties
                "bbb_mpo_score": 3.5,  # Good BBB penetration
                "lipinski_violations": 0,  # No violations
            },
        },
        "ibuprofen": {
            "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            "expected": {
                "qed_score": 0.8,  # High drug-likeness
                "lipinski_violations": 0,  # No violations
                "lead_likeness": 0.6,  # Moderate lead-likeness (higher MW)
                "synthetic_accessibility": 3.0,  # Moderate synthesis difficulty
            },
        },
        "morphine": {
            "smiles": "CN1CC[C@]23C4=C5C=CC(O)=C4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5",
            "expected": {
                "cns_mpo_score": 3.5,  # Good CNS properties
                "bbb_mpo_score": 3.0,  # Good BBB penetration
                "natural_product_likeness": 0.8,  # High NP-likeness
                "synthetic_accessibility": 8.0,  # Difficult to synthesize
            },
        },
    }


@pytest.fixture
def fragment_molecules():
    """Fixture providing fragment molecules."""
    return {
        "benzene": {
            "smiles": "C1=CC=CC=C1",
            "expected": {
                "fragment_likeness": 0.8,  # Good fragment
                "rule_of_three": 0,  # No violations
                "lead_likeness": 0.4,  # Poor lead (too small)
            },
        },
        "phenol": {
            "smiles": "C1=CC=C(C=C1)O",
            "expected": {
                "fragment_likeness": 0.9,  # Excellent fragment
                "rule_of_three": 0,  # No violations
                "pains_alerts": 0,  # Should be clean
            },
        },
        "aniline": {
            "smiles": "NC1=CC=CC=C1",
            "expected": {
                "fragment_likeness": 0.7,  # Good fragment
                "rule_of_three": 0,  # No violations
                "brenk_alerts": 1,  # May have alerts
            },
        },
    }


@pytest.fixture
def problematic_molecules():
    """Fixture providing molecules with known issues."""
    return {
        "quinone": {
            "smiles": "O=C1C=CC(=O)C=C1",
            "expected": {
                "pains_alerts": 1,  # Quinone is a PAINS
                "brenk_alerts": 1,  # Reactive quinone
            },
        },
        "large_molecule": {
            "smiles": "CC(C)(C)OC(=O)N[C@@H](CC1=CC=CC=C1)C(=O)N[C@@H](CC(=O)O)C(=O)N[C@@H](CC2=CC=C(C=C2)O)C(=O)O",
            "expected": {
                "lipinski_violations": 2,  # MW and HBA violations
                "veber_violations": 1,  # TPSA violation
                "lead_likeness": 0.2,  # Poor lead-likeness
            },
        },
    }


class TestDrugDiscoveryValidation:
    """Test class for drug discovery descriptor validation."""

    def test_qed_score_calculation(self, drug_discovery_calculator, drug_molecules):
        """Test QED score calculation accuracy."""
        smiles_list = [mol["smiles"] for mol in drug_molecules.values()]

        results = drug_discovery_calculator.calculate_descriptors(
            smiles_list, ["qed_score"]
        )

        assert len(results) == len(smiles_list)
        assert not results["qed_score"].isna().all()

        # QED scores should be between 0 and 1
        assert results["qed_score"].between(0, 1).all()

        # Known drugs should have reasonable QED scores
        for i, (drug_name, drug_data) in enumerate(drug_molecules.items()):
            if "qed_score" in drug_data["expected"]:
                calculated = results["qed_score"].iloc[i]
                expected = drug_data["expected"]["qed_score"]
                assert (
                    abs(calculated - expected) < 0.3
                ), f"QED score for {drug_name}: expected {expected}, got {calculated}"

    def test_cns_mpo_score(self, drug_discovery_calculator, drug_molecules):
        """Test CNS MPO score calculation."""
        smiles_list = [mol["smiles"] for mol in drug_molecules.values()]

        results = drug_discovery_calculator.calculate_descriptors(
            smiles_list, ["cns_mpo_score"]
        )

        assert len(results) == len(smiles_list)
        assert not results["cns_mpo_score"].isna().all()

        # CNS MPO scores should be between 0 and 6
        assert results["cns_mpo_score"].between(0, 6).all()

    def test_bbb_mpo_score(self, drug_discovery_calculator, drug_molecules):
        """Test BBB MPO score calculation."""
        smiles_list = [mol["smiles"] for mol in drug_molecules.values()]

        results = drug_discovery_calculator.calculate_descriptors(
            smiles_list, ["bbb_mpo_score"]
        )

        assert len(results) == len(smiles_list)
        assert not results["bbb_mpo_score"].isna().all()

        # BBB MPO scores should be between 0 and 4
        assert results["bbb_mpo_score"].between(0, 4).all()

    def test_synthetic_accessibility(self, drug_discovery_calculator, drug_molecules):
        """Test synthetic accessibility score calculation."""
        smiles_list = [mol["smiles"] for mol in drug_molecules.values()]

        results = drug_discovery_calculator.calculate_descriptors(
            smiles_list, ["synthetic_accessibility"]
        )

        assert len(results) == len(smiles_list)
        assert not results["synthetic_accessibility"].isna().all()

        # SA scores should be between 1 and 10
        assert results["synthetic_accessibility"].between(1, 10).all()

        # Simple molecules should have lower SA scores than complex ones
        aspirin_idx = 0  # First molecule
        morphine_idx = 3  # Fourth molecule (complex natural product)

        aspirin_sa = results["synthetic_accessibility"].iloc[aspirin_idx]
        morphine_sa = results["synthetic_accessibility"].iloc[morphine_idx]

        assert (
            aspirin_sa < morphine_sa
        ), "Aspirin should be easier to synthesize than morphine"

    def test_lead_likeness(self, drug_discovery_calculator, fragment_molecules):
        """Test lead-likeness assessment."""
        smiles_list = [mol["smiles"] for mol in fragment_molecules.values()]

        results = drug_discovery_calculator.calculate_descriptors(
            smiles_list, ["lead_likeness"]
        )

        assert len(results) == len(smiles_list)
        assert not results["lead_likeness"].isna().all()

        # Lead-likeness scores should be between 0 and 1
        assert results["lead_likeness"].between(0, 1).all()

    def test_fragment_likeness(self, drug_discovery_calculator, fragment_molecules):
        """Test fragment-likeness assessment."""
        smiles_list = [mol["smiles"] for mol in fragment_molecules.values()]

        results = drug_discovery_calculator.calculate_descriptors(
            smiles_list, ["fragment_likeness", "rule_of_three"]
        )

        assert len(results) == len(smiles_list)
        assert not results["fragment_likeness"].isna().all()
        assert not results["rule_of_three"].isna().all()

        # Fragment-likeness scores should be between 0 and 1
        assert results["fragment_likeness"].between(0, 1).all()

        # Rule of three violations should be non-negative integers
        assert results["rule_of_three"].ge(0).all()
        assert results["rule_of_three"].dtype in [np.int64, np.float64]

    def test_pains_alerts(self, drug_discovery_calculator, problematic_molecules):
        """Test PAINS alerts detection."""
        smiles_list = [mol["smiles"] for mol in problematic_molecules.values()]

        results = drug_discovery_calculator.calculate_descriptors(
            smiles_list, ["pains_alerts", "brenk_alerts", "nih_alerts"]
        )

        assert len(results) == len(smiles_list)

        # Alert counts should be non-negative integers
        for alert_type in ["pains_alerts", "brenk_alerts", "nih_alerts"]:
            assert results[alert_type].ge(0).all()

    def test_natural_product_likeness(self, drug_discovery_calculator, drug_molecules):
        """Test natural product-likeness scoring."""
        smiles_list = [mol["smiles"] for mol in drug_molecules.values()]

        results = drug_discovery_calculator.calculate_descriptors(
            smiles_list, ["natural_product_likeness"]
        )

        assert len(results) == len(smiles_list)
        assert not results["natural_product_likeness"].isna().all()

        # NP-likeness scores should be between 0 and 1
        assert results["natural_product_likeness"].between(0, 1).all()

        # Morphine (natural product) should have higher NP-likeness than synthetic drugs
        morphine_idx = 3
        aspirin_idx = 0

        morphine_np = results["natural_product_likeness"].iloc[morphine_idx]
        aspirin_np = results["natural_product_likeness"].iloc[aspirin_idx]

        assert (
            morphine_np > aspirin_np
        ), "Morphine should have higher NP-likeness than aspirin"

    def test_lipinski_violations(
        self, drug_discovery_calculator, problematic_molecules
    ):
        """Test Lipinski rule violations."""
        smiles_list = [mol["smiles"] for mol in problematic_molecules.values()]

        results = drug_discovery_calculator.calculate_descriptors(
            smiles_list,
            [
                "lipinski_violations",
                "veber_violations",
                "egan_violations",
                "muegge_violations",
            ],
        )

        assert len(results) == len(smiles_list)

        # Violation counts should be non-negative integers
        violation_types = [
            "lipinski_violations",
            "veber_violations",
            "egan_violations",
            "muegge_violations",
        ]
        for violation_type in violation_types:
            assert results[violation_type].ge(0).all()

        # Large molecule should have more violations than small ones
        large_mol_idx = 1  # Second molecule in problematic_molecules
        large_mol_violations = results["lipinski_violations"].iloc[large_mol_idx]
        assert (
            large_mol_violations >= 1
        ), "Large molecule should have Lipinski violations"

    def test_drug_likeness_consistency(self, drug_discovery_calculator):
        """Test consistency between different drug-likeness measures."""
        # Test with known drug-like and non-drug-like molecules
        drug_like = ["CC(=O)OC1=CC=CC=C1C(=O)O"]  # Aspirin
        non_drug_like = ["CCCCCCCCCCCCCCCCCCCC"]  # Long alkyl chain

        all_smiles = drug_like + non_drug_like

        results = drug_discovery_calculator.calculate_descriptors(
            all_smiles, ["qed_score", "lipinski_violations", "lead_likeness"]
        )

        # Drug-like molecule should score better
        drug_qed = results["qed_score"].iloc[0]
        non_drug_qed = results["qed_score"].iloc[1]

        drug_violations = results["lipinski_violations"].iloc[0]
        non_drug_violations = results["lipinski_violations"].iloc[1]

        assert drug_qed > non_drug_qed, "Drug-like molecule should have higher QED"
        assert (
            drug_violations <= non_drug_violations
        ), "Drug-like molecule should have fewer violations"

    def test_descriptor_categories(self, drug_discovery_calculator):
        """Test descriptor categorization."""
        categories = drug_discovery_calculator.get_categories()
        expected_categories = [
            "synthetic",
            "drug_likeness",
            "lead_likeness",
            "fragment_likeness",
            "structural_alerts",
            "natural_product",
        ]

        for category in expected_categories:
            assert category in categories

        # Test category filtering
        drug_likeness_descriptors = drug_discovery_calculator.list_descriptors(
            "drug_likeness"
        )
        assert "qed_score" in drug_likeness_descriptors
        assert "cns_mpo_score" in drug_likeness_descriptors

        alert_descriptors = drug_discovery_calculator.list_descriptors(
            "structural_alerts"
        )
        assert "pains_alerts" in alert_descriptors
        assert "brenk_alerts" in alert_descriptors

    def test_invalid_molecules(self, drug_discovery_calculator):
        """Test handling of invalid molecules."""
        invalid_smiles = ["INVALID", "C(C)(C", ""]

        results = drug_discovery_calculator.calculate_descriptors(
            invalid_smiles, ["qed_score", "lipinski_violations"]
        )

        assert len(results) == len(invalid_smiles)
        assert results["qed_score"].isna().all()
        assert results["lipinski_violations"].isna().all()

    def test_batch_processing(self, drug_discovery_calculator):
        """Test batch processing performance and consistency."""
        # Generate a batch of molecules
        test_smiles = [
            "CCO",  # Ethanol
            "CC(=O)O",  # Acetic acid
            "C1=CC=CC=C1",  # Benzene
            "CCN(CC)CC",  # Triethylamine
        ] * 25  # 100 molecules total

        results = drug_discovery_calculator.calculate_descriptors(
            test_smiles, ["qed_score", "lipinski_violations", "fragment_likeness"]
        )

        assert len(results) == len(test_smiles)

        # Check that identical molecules have identical results
        ethanol_results = results[results["smiles"] == "CCO"]
        for descriptor in ["qed_score", "lipinski_violations", "fragment_likeness"]:
            unique_values = ethanol_results[descriptor].nunique()
            assert unique_values == 1, f"Inconsistent results for {descriptor}"


class TestDrugDiscoveryPerformance:
    """Test class for drug discovery descriptor performance."""

    def test_calculation_speed(self, drug_discovery_calculator):
        """Test calculation speed for large batches."""
        import time

        # Generate test molecules
        test_smiles = ["CCO", "CC(=O)O", "C1=CC=CC=C1"] * 100

        start_time = time.time()
        results = drug_discovery_calculator.calculate_descriptors(
            test_smiles, ["qed_score", "lipinski_violations", "synthetic_accessibility"]
        )
        end_time = time.time()

        calculation_time = end_time - start_time
        molecules_per_second = len(test_smiles) / calculation_time

        # Should process at least 5 molecules per second (SA calculation is slower)
        assert (
            molecules_per_second > 5
        ), f"Too slow: {molecules_per_second} molecules/sec"
        assert len(results) == len(test_smiles)

    def test_memory_usage(self, drug_discovery_calculator):
        """Test memory usage for large batches."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process a large batch
        test_smiles = ["CCO"] * 1000
        results = drug_discovery_calculator.calculate_descriptors(
            test_smiles, ["qed_score", "lipinski_violations"]
        )

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (< 100 MB for 1000 molecules)
        assert memory_increase < 100, f"Excessive memory usage: {memory_increase} MB"
        assert len(results) == len(test_smiles)


@pytest.mark.integration
class TestDrugDiscoveryIntegration:
    """Integration tests for drug discovery descriptors."""

    def test_pandas_integration(self, drug_discovery_calculator):
        """Test integration with pandas DataFrames."""
        df = pd.DataFrame(
            {
                "compound_id": ["DRUG_001", "DRUG_002", "DRUG_003"],
                "smiles": ["CCO", "CC(=O)O", "C1=CC=CC=C1"],
                "target": ["kinase", "gpcr", "ion_channel"],
            }
        )

        results = drug_discovery_calculator.calculate_descriptors(
            df["smiles"], ["qed_score", "lipinski_violations"]
        )

        # Merge results back with original DataFrame
        merged = pd.merge(df, results, on="smiles")

        assert len(merged) == len(df)
        assert "compound_id" in merged.columns
        assert "target" in merged.columns
        assert "qed_score" in merged.columns
        assert "lipinski_violations" in merged.columns

    def test_filtering_workflow(self, drug_discovery_calculator):
        """Test typical drug discovery filtering workflow."""
        # Simulate a compound library
        compound_library = [
            "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin (drug-like)
            "CCCCCCCCCCCCCCCCCCCC",  # Long chain (non-drug-like)
            "C1=CC=CC=C1",  # Benzene (fragment-like)
            "O=C1C=CC(=O)C=C1",  # Quinone (PAINS)
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine (drug-like)
        ]

        results = drug_discovery_calculator.calculate_descriptors(
            compound_library,
            ["qed_score", "lipinski_violations", "pains_alerts", "fragment_likeness"],
        )

        # Apply typical filters
        drug_like = results[
            (results["qed_score"] > 0.5)
            & (results["lipinski_violations"] <= 1)
            & (results["pains_alerts"] == 0)
        ]

        fragment_like = results[
            (results["fragment_likeness"] > 0.6) & (results["pains_alerts"] == 0)
        ]

        # Should identify drug-like compounds
        assert len(drug_like) >= 2, "Should identify at least 2 drug-like compounds"

        # Should identify fragment-like compounds
        assert (
            len(fragment_like) >= 1
        ), "Should identify at least 1 fragment-like compound"

    def test_error_handling(self, drug_discovery_calculator):
        """Test error handling and recovery."""
        mixed_smiles = [
            "CCO",  # Valid
            "INVALID",  # Invalid
            "CC(=O)O",  # Valid
            "",  # Empty
            "C1=CC=CC=C1",  # Valid
        ]

        results = drug_discovery_calculator.calculate_descriptors(
            mixed_smiles, ["qed_score", "lipinski_violations"]
        )

        assert len(results) == len(mixed_smiles)

        # Valid molecules should have results
        valid_indices = [0, 2, 4]
        for idx in valid_indices:
            assert not pd.isna(results.iloc[idx]["qed_score"])
            assert not pd.isna(results.iloc[idx]["lipinski_violations"])

        # Invalid molecules should have NaN results
        invalid_indices = [1, 3]
        for idx in invalid_indices:
            assert pd.isna(results.iloc[idx]["qed_score"])
            assert pd.isna(results.iloc[idx]["lipinski_violations"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
