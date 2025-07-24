"""
Validation tests for ADMET descriptors against experimental databases.

This module provides comprehensive validation of ADMET descriptor calculations
against known experimental values and established benchmarks.
"""

import pytest
import numpy as np
import pandas as pd
from typing import List, Dict, Any

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from src.chemi_accel.descriptors.admet import ADMETDescriptorCalculator
from src.chemi_accel.core.exceptions import ChemiAccelError


@pytest.fixture
def admet_calculator():
    """Fixture providing ADMET descriptor calculator."""
    if not RDKIT_AVAILABLE:
        pytest.skip("RDKit not available")
    return ADMETDescriptorCalculator()


@pytest.fixture
def test_molecules():
    """Fixture providing test molecules with known ADMET properties."""
    return {
        # Drug molecules with known properties
        "aspirin": {
            "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "expected": {
                "hia_prediction": 0.8,  # High absorption
                "bbb_permeability": -1.2,  # Low BBB penetration
                "hepatotoxicity": 0.2,  # Low hepatotoxicity
            }
        },
        "caffeine": {
            "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "expected": {
                "hia_prediction": 0.9,  # High absorption
                "bbb_permeability": 0.1,  # Good BBB penetration
                "cns_mpo_score": 4.0,  # Moderate CNS score
            }
        },
        "ibuprofen": {
            "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            "expected": {
                "hia_prediction": 0.9,  # High absorption
                "plasma_protein_binding": 95.0,  # High protein binding
                "hepatotoxicity": 0.3,  # Moderate hepatotoxicity
            }
        },
        "morphine": {
            "smiles": "CN1CC[C@]23C4=C5C=CC(O)=C4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5",
            "expected": {
                "bbb_permeability": 0.5,  # Good BBB penetration
                "cns_mpo_score": 3.5,  # Good CNS score
                "cyp2d6_inhibition": 0.4,  # Moderate CYP2D6 inhibition
            }
        },
        "warfarin": {
            "smiles": "CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O",
            "expected": {
                "plasma_protein_binding": 98.0,  # Very high protein binding
                "cyp2c9_inhibition": 0.7,  # High CYP2C9 interaction
                "hepatotoxicity": 0.4,  # Moderate hepatotoxicity
            }
        }
    }


@pytest.fixture
def toxic_molecules():
    """Fixture providing molecules with known toxicity issues."""
    return {
        "benzene": {
            "smiles": "C1=CC=CC=C1",
            "expected": {
                "carcinogenicity": 0.8,  # Known carcinogen
                "ames_mutagenicity": 0.6,  # Mutagenic
            }
        },
        "chloroform": {
            "smiles": "C(Cl)(Cl)Cl",
            "expected": {
                "hepatotoxicity": 0.9,  # High hepatotoxicity
                "carcinogenicity": 0.7,  # Probable carcinogen
            }
        },
        "aniline": {
            "smiles": "NC1=CC=CC=C1",
            "expected": {
                "ames_mutagenicity": 0.5,  # Moderate mutagenicity
                "skin_sensitization": 0.6,  # Skin sensitizer
            }
        }
    }


class TestADMETValidation:
    """Test class for ADMET descriptor validation."""
    
    def test_absorption_descriptors(self, admet_calculator, test_molecules):
        """Test absorption descriptor accuracy."""
        smiles_list = [mol["smiles"] for mol in test_molecules.values()]
        
        results = admet_calculator.calculate_descriptors(
            smiles_list, 
            ["caco2_permeability", "hia_prediction", "pgp_substrate"]
        )
        
        assert len(results) == len(smiles_list)
        assert not results["caco2_permeability"].isna().all()
        assert not results["hia_prediction"].isna().all()
        assert not results["pgp_substrate"].isna().all()
        
        # Check value ranges
        assert results["hia_prediction"].between(0, 1).all()
        assert results["pgp_substrate"].between(0, 1).all()
        assert results["caco2_permeability"].between(-8, -4).all()
    
    def test_distribution_descriptors(self, admet_calculator, test_molecules):
        """Test distribution descriptor accuracy."""
        smiles_list = [mol["smiles"] for mol in test_molecules.values()]
        
        results = admet_calculator.calculate_descriptors(
            smiles_list,
            ["bbb_permeability", "plasma_protein_binding", "volume_distribution"]
        )
        
        assert len(results) == len(smiles_list)
        assert not results["bbb_permeability"].isna().all()
        assert not results["plasma_protein_binding"].isna().all()
        assert not results["volume_distribution"].isna().all()
        
        # Check value ranges
        assert results["plasma_protein_binding"].between(0, 100).all()
        assert results["volume_distribution"].gt(0).all()
        assert results["bbb_permeability"].between(-3, 2).all()
    
    def test_metabolism_descriptors(self, admet_calculator, test_molecules):
        """Test metabolism descriptor accuracy."""
        smiles_list = [mol["smiles"] for mol in test_molecules.values()]
        
        cyp_descriptors = [
            "cyp1a2_inhibition", "cyp2c9_inhibition", "cyp2c19_inhibition",
            "cyp2d6_inhibition", "cyp3a4_inhibition"
        ]
        
        results = admet_calculator.calculate_descriptors(smiles_list, cyp_descriptors)
        
        assert len(results) == len(smiles_list)
        
        # Check all CYP descriptors are in valid range [0, 1]
        for desc in cyp_descriptors:
            assert not results[desc].isna().all()
            assert results[desc].between(0, 1).all()
    
    def test_excretion_descriptors(self, admet_calculator, test_molecules):
        """Test excretion descriptor accuracy."""
        smiles_list = [mol["smiles"] for mol in test_molecules.values()]
        
        results = admet_calculator.calculate_descriptors(
            smiles_list,
            ["renal_clearance", "half_life"]
        )
        
        assert len(results) == len(smiles_list)
        assert not results["renal_clearance"].isna().all()
        assert not results["half_life"].isna().all()
        
        # Check value ranges
        assert results["renal_clearance"].between(0, 200).all()
        assert results["half_life"].between(0.1, 100).all()
    
    def test_toxicity_descriptors(self, admet_calculator, toxic_molecules):
        """Test toxicity descriptor accuracy."""
        smiles_list = [mol["smiles"] for mol in toxic_molecules.values()]
        
        toxicity_descriptors = [
            "herg_inhibition", "ames_mutagenicity", "hepatotoxicity",
            "skin_sensitization", "carcinogenicity"
        ]
        
        results = admet_calculator.calculate_descriptors(smiles_list, toxicity_descriptors)
        
        assert len(results) == len(smiles_list)
        
        # Check all toxicity descriptors are in valid range [0, 1]
        for desc in toxicity_descriptors:
            assert not results[desc].isna().all()
            assert results[desc].between(0, 1).all()
    
    def test_known_drug_properties(self, admet_calculator, test_molecules):
        """Test against known drug properties."""
        for drug_name, drug_data in test_molecules.items():
            smiles = drug_data["smiles"]
            expected = drug_data["expected"]
            
            results = admet_calculator.calculate_descriptors(
                [smiles], 
                list(expected.keys())
            )
            
            for descriptor, expected_value in expected.items():
                calculated_value = results[descriptor].iloc[0]
                
                # Allow for reasonable tolerance in predictions
                if descriptor in ["hia_prediction", "bbb_permeability"]:
                    tolerance = 0.3
                elif descriptor in ["plasma_protein_binding"]:
                    tolerance = 20.0  # 20% tolerance for protein binding
                elif descriptor in ["cns_mpo_score"]:
                    tolerance = 1.0
                else:
                    tolerance = 0.2
                
                assert abs(calculated_value - expected_value) <= tolerance, \
                    f"For {drug_name}, {descriptor}: expected {expected_value}, got {calculated_value}"
    
    def test_descriptor_consistency(self, admet_calculator):
        """Test descriptor consistency across similar molecules."""
        # Test with a series of similar molecules
        similar_molecules = [
            "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
            "CC(=O)OC1=CC=C(C=C1)C(=O)O",  # Para-substituted analog
            "CC(=O)OC1=CC=C(C(=O)O)C=C1",  # Meta-substituted analog
        ]
        
        results = admet_calculator.calculate_descriptors(
            similar_molecules,
            ["hia_prediction", "bbb_permeability", "hepatotoxicity"]
        )
        
        # Similar molecules should have similar properties
        for descriptor in ["hia_prediction", "bbb_permeability", "hepatotoxicity"]:
            values = results[descriptor].values
            std_dev = np.std(values)
            assert std_dev < 0.3, f"High variability in {descriptor} for similar molecules"
    
    def test_invalid_molecules(self, admet_calculator):
        """Test handling of invalid molecules."""
        invalid_smiles = ["INVALID", "C(C)(C", ""]
        
        results = admet_calculator.calculate_descriptors(
            invalid_smiles,
            ["hia_prediction", "hepatotoxicity"]
        )
        
        assert len(results) == len(invalid_smiles)
        assert results["hia_prediction"].isna().all()
        assert results["hepatotoxicity"].isna().all()
    
    def test_batch_processing(self, admet_calculator):
        """Test batch processing performance and consistency."""
        # Generate a larger batch of molecules
        test_smiles = [
            "CCO",  # Ethanol
            "CC(=O)O",  # Acetic acid
            "C1=CC=CC=C1",  # Benzene
            "CCN(CC)CC",  # Triethylamine
            "CC(C)O",  # Isopropanol
        ] * 20  # 100 molecules total
        
        results = admet_calculator.calculate_descriptors(
            test_smiles,
            ["hia_prediction", "bbb_permeability", "hepatotoxicity"]
        )
        
        assert len(results) == len(test_smiles)
        
        # Check that identical molecules have identical results
        ethanol_results = results[results["smiles"] == "CCO"]
        for descriptor in ["hia_prediction", "bbb_permeability", "hepatotoxicity"]:
            unique_values = ethanol_results[descriptor].nunique()
            assert unique_values == 1, f"Inconsistent results for {descriptor}"
    
    def test_descriptor_categories(self, admet_calculator):
        """Test descriptor categorization."""
        categories = admet_calculator.get_categories()
        expected_categories = ["absorption", "distribution", "metabolism", "excretion", "toxicity"]
        
        for category in expected_categories:
            assert category in categories
        
        # Test category filtering
        absorption_descriptors = admet_calculator.list_descriptors("absorption")
        assert "caco2_permeability" in absorption_descriptors
        assert "hia_prediction" in absorption_descriptors
        
        toxicity_descriptors = admet_calculator.list_descriptors("toxicity")
        assert "herg_inhibition" in toxicity_descriptors
        assert "ames_mutagenicity" in toxicity_descriptors
    
    def test_descriptor_info(self, admet_calculator):
        """Test descriptor information retrieval."""
        info = admet_calculator.get_descriptor_info("caco2_permeability")
        
        assert "description" in info
        assert "category" in info
        assert "units" in info
        assert info["category"] == "absorption"
        assert info["units"] == "log(cm/s)"


class TestADMETPerformance:
    """Test class for ADMET descriptor performance."""
    
    def test_calculation_speed(self, admet_calculator):
        """Test calculation speed for large batches."""
        import time
        
        # Generate test molecules
        test_smiles = ["CCO", "CC(=O)O", "C1=CC=CC=C1"] * 100
        
        start_time = time.time()
        results = admet_calculator.calculate_descriptors(
            test_smiles,
            ["hia_prediction", "bbb_permeability", "hepatotoxicity"]
        )
        end_time = time.time()
        
        calculation_time = end_time - start_time
        molecules_per_second = len(test_smiles) / calculation_time
        
        # Should process at least 10 molecules per second
        assert molecules_per_second > 10, f"Too slow: {molecules_per_second} molecules/sec"
        assert len(results) == len(test_smiles)
    
    def test_memory_usage(self, admet_calculator):
        """Test memory usage for large batches."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process a large batch
        test_smiles = ["CCO"] * 1000
        results = admet_calculator.calculate_descriptors(
            test_smiles,
            ["hia_prediction", "bbb_permeability"]
        )
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 100 MB for 1000 molecules)
        assert memory_increase < 100, f"Excessive memory usage: {memory_increase} MB"
        assert len(results) == len(test_smiles)


@pytest.mark.integration
class TestADMETIntegration:
    """Integration tests for ADMET descriptors."""
    
    def test_pandas_integration(self, admet_calculator):
        """Test integration with pandas DataFrames."""
        df = pd.DataFrame({
            "compound_id": ["COMP_001", "COMP_002", "COMP_003"],
            "smiles": ["CCO", "CC(=O)O", "C1=CC=CC=C1"]
        })
        
        results = admet_calculator.calculate_descriptors(
            df["smiles"],
            ["hia_prediction", "hepatotoxicity"]
        )
        
        # Merge results back with original DataFrame
        merged = pd.merge(df, results, on="smiles")
        
        assert len(merged) == len(df)
        assert "compound_id" in merged.columns
        assert "hia_prediction" in merged.columns
        assert "hepatotoxicity" in merged.columns
    
    def test_error_handling(self, admet_calculator):
        """Test error handling and recovery."""
        mixed_smiles = [
            "CCO",  # Valid
            "INVALID",  # Invalid
            "CC(=O)O",  # Valid
            "",  # Empty
            "C1=CC=CC=C1"  # Valid
        ]
        
        results = admet_calculator.calculate_descriptors(
            mixed_smiles,
            ["hia_prediction", "hepatotoxicity"]
        )
        
        assert len(results) == len(mixed_smiles)
        
        # Valid molecules should have results
        valid_indices = [0, 2, 4]
        for idx in valid_indices:
            assert not pd.isna(results.iloc[idx]["hia_prediction"])
            assert not pd.isna(results.iloc[idx]["hepatotoxicity"])
        
        # Invalid molecules should have NaN results
        invalid_indices = [1, 3]
        for idx in invalid_indices:
            assert pd.isna(results.iloc[idx]["hia_prediction"])
            assert pd.isna(results.iloc[idx]["hepatotoxicity"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])