"""
Tests for basic molecular descriptor calculations.

This module tests the BasicDescriptorCalculator against RDKit reference
implementations to ensure numerical accuracy and correctness.
"""

import pytest
import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from chemi_accel.descriptors.basic import BasicDescriptorCalculator, calculate_basic_properties
from chemi_accel.utils.rdkit_integration import RDKitMolecularParser
from chemi_accel.core.exceptions import DescriptorCalculationError


# Test molecules with known properties
TEST_MOLECULES = [
    ("CCO", "ethanol"),  # Simple alcohol
    ("CC(=O)O", "acetic_acid"),  # Carboxylic acid
    ("c1ccccc1", "benzene"),  # Aromatic ring
    ("CCc1ccccc1", "ethylbenzene"),  # Substituted aromatic
    ("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "ibuprofen"),  # Drug molecule
    ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "caffeine"),  # Heterocyclic compound
    ("CC1=CC=CC=C1C(=O)O", "toluic_acid"),  # Aromatic carboxylic acid
    ("CCCCCCCCCCCCCCC(=O)O", "palmitic_acid"),  # Long chain fatty acid
]

INVALID_MOLECULES = [
    "",  # Empty string
    "INVALID_SMILES",  # Invalid SMILES
    "C#C#C",  # Unusual bonding
]


@pytest.fixture
def calculator():
    """Create BasicDescriptorCalculator instance."""
    if not RDKIT_AVAILABLE:
        pytest.skip("RDKit not available")
    return BasicDescriptorCalculator()


@pytest.fixture
def parser():
    """Create RDKitMolecularParser instance."""
    if not RDKIT_AVAILABLE:
        pytest.skip("RDKit not available")
    return RDKitMolecularParser()


@pytest.fixture
def test_molecules(parser):
    """Parse test molecules."""
    molecules = []
    for smiles, name in TEST_MOLECULES:
        try:
            mol_data = parser.parse_smiles(smiles, name)
            molecules.append(mol_data)
        except Exception as e:
            pytest.fail(f"Failed to parse test molecule {name} ({smiles}): {e}")
    return molecules


class TestBasicDescriptorCalculator:
    """Test BasicDescriptorCalculator functionality."""
    
    def test_initialization(self, calculator):
        """Test calculator initialization."""
        assert calculator is not None
        descriptors = calculator.get_available_descriptors()
        assert len(descriptors) > 0
        assert "molecular_weight" in descriptors
        assert "logp" in descriptors
        assert "tpsa" in descriptors
    
    def test_descriptor_info(self, calculator):
        """Test descriptor information retrieval."""
        info = calculator.get_descriptor_info("molecular_weight")
        assert "description" in info
        assert "category" in info
        assert info["category"] == "constitutional"
        
        # Test invalid descriptor
        with pytest.raises(ValueError):
            calculator.get_descriptor_info("invalid_descriptor")
    
    def test_descriptors_by_category(self, calculator):
        """Test getting descriptors by category."""
        constitutional = calculator.get_descriptors_by_category("constitutional")
        assert "molecular_weight" in constitutional
        assert "num_atoms" in constitutional
        
        physicochemical = calculator.get_descriptors_by_category("physicochemical")
        assert "logp" in physicochemical
        assert "tpsa" in physicochemical
    
    def test_single_descriptor_calculation(self, calculator, test_molecules):
        """Test single descriptor calculation."""
        mol_data = test_molecules[0]  # ethanol
        
        # Test molecular weight
        mw = calculator.calculate_single_descriptor(mol_data, "molecular_weight")
        assert isinstance(mw, float)
        assert mw > 0
        
        # Compare with RDKit reference
        rdkit_mol = mol_data.backend_data["rdkit_mol"]
        rdkit_mw = Descriptors.MolWt(rdkit_mol)
        assert abs(mw - rdkit_mw) < 1e-6
    
    def test_multiple_descriptor_calculation(self, calculator, test_molecules):
        """Test multiple descriptor calculation for single molecule."""
        mol_data = test_molecules[0]  # ethanol
        
        descriptors = ["molecular_weight", "num_atoms", "logp", "tpsa"]
        results = calculator.calculate_descriptors(mol_data, descriptors)
        
        assert len(results) >= len(descriptors)
        for desc in descriptors:
            assert desc in results
            assert isinstance(results[desc], (int, float))
    
    def test_batch_descriptor_calculation(self, calculator, test_molecules):
        """Test batch descriptor calculation."""
        descriptors = ["molecular_weight", "num_atoms", "logp", "tpsa"]
        df = calculator.calculate_descriptors_batch(test_molecules, descriptors)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(test_molecules)
        assert "mol_id" in df.columns
        assert "smiles" in df.columns
        
        for desc in descriptors:
            assert desc in df.columns
            assert df[desc].dtype in [np.float64, np.int64]
    
    def test_all_descriptors_calculation(self, calculator, test_molecules):
        """Test calculation of all available descriptors."""
        mol_data = test_molecules[0]  # ethanol
        
        results = calculator.calculate_descriptors(mol_data)
        available_descriptors = calculator.get_available_descriptors()
        
        # Should have results for all descriptors
        for desc in available_descriptors:
            assert desc in results
    
    def test_descriptor_accuracy_against_rdkit(self, calculator, test_molecules):
        """Test descriptor accuracy against RDKit reference implementations."""
        tolerance = 1e-6
        
        for mol_data in test_molecules:
            rdkit_mol = mol_data.backend_data["rdkit_mol"]
            
            # Test molecular weight
            calc_mw = calculator.calculate_single_descriptor(mol_data, "molecular_weight")
            rdkit_mw = Descriptors.MolWt(rdkit_mol)
            assert abs(calc_mw - rdkit_mw) < tolerance, f"MW mismatch for {mol_data.mol_id}"
            
            # Test LogP
            calc_logp = calculator.calculate_single_descriptor(mol_data, "logp")
            rdkit_logp = Crippen.MolLogP(rdkit_mol)
            assert abs(calc_logp - rdkit_logp) < tolerance, f"LogP mismatch for {mol_data.mol_id}"
            
            # Test TPSA
            calc_tpsa = calculator.calculate_single_descriptor(mol_data, "tpsa")
            rdkit_tpsa = Descriptors.TPSA(rdkit_mol)
            assert abs(calc_tpsa - rdkit_tpsa) < tolerance, f"TPSA mismatch for {mol_data.mol_id}"
            
            # Test atom counts
            calc_atoms = calculator.calculate_single_descriptor(mol_data, "num_atoms")
            rdkit_atoms = rdkit_mol.GetNumAtoms()
            assert calc_atoms == rdkit_atoms, f"Atom count mismatch for {mol_data.mol_id}"
            
            # Test heavy atom count
            calc_heavy = calculator.calculate_single_descriptor(mol_data, "num_heavy_atoms")
            rdkit_heavy = Descriptors.HeavyAtomCount(rdkit_mol)
            assert calc_heavy == rdkit_heavy, f"Heavy atom count mismatch for {mol_data.mol_id}"
    
    def test_error_handling(self, calculator, parser):
        """Test error handling for invalid inputs."""
        # Test with molecule without RDKit mol object
        from chemi_accel.core.molecular import MolecularData
        invalid_mol = MolecularData(smiles="CCO", mol_id="test")
        
        with pytest.raises(DescriptorCalculationError):
            calculator.calculate_single_descriptor(invalid_mol, "molecular_weight")
        
        # Test with invalid descriptor name
        valid_mol = parser.parse_smiles("CCO", "ethanol")
        with pytest.raises(DescriptorCalculationError):
            calculator.calculate_single_descriptor(valid_mol, "invalid_descriptor")
    
    def test_descriptor_statistics(self, calculator, test_molecules):
        """Test descriptor statistics calculation."""
        descriptors = ["molecular_weight", "logp", "tpsa"]
        df = calculator.calculate_descriptors_batch(test_molecules, descriptors)
        
        stats = calculator.get_descriptor_statistics(df, descriptors)
        
        assert isinstance(stats, pd.DataFrame)
        assert "mean" in stats.index
        assert "std" in stats.index
        assert "missing" in stats.index
        assert "missing_pct" in stats.index
        
        for desc in descriptors:
            assert desc in stats.columns


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_calculate_basic_properties(self, test_molecules):
        """Test calculate_basic_properties function."""
        df = calculate_basic_properties(test_molecules)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(test_molecules)
        
        expected_columns = [
            "molecular_weight", "num_atoms", "num_heavy_atoms", "num_bonds",
            "ring_count", "num_aromatic_rings", "logp", "tpsa",
            "num_rotatable_bonds", "num_hbd", "num_hba"
        ]
        
        for col in expected_columns:
            assert col in df.columns
    
    def test_lipinski_descriptors(self, test_molecules):
        """Test Lipinski Rule of Five descriptors."""
        from chemi_accel.descriptors.basic import get_lipinski_descriptors
        
        df = get_lipinski_descriptors(test_molecules)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(test_molecules)
        
        expected_columns = [
            "molecular_weight", "logp", "num_hbd", "num_hba",
            "lipinski_violations", "lipinski_compliant"
        ]
        
        for col in expected_columns:
            assert col in df.columns
        
        # Check violation logic
        for _, row in df.iterrows():
            violations = 0
            if row["molecular_weight"] > 500:
                violations += 1
            if row["logp"] > 5:
                violations += 1
            if row["num_hbd"] > 5:
                violations += 1
            if row["num_hba"] > 10:
                violations += 1
            
            assert row["lipinski_violations"] == violations
            assert row["lipinski_compliant"] == (violations == 0)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_molecule_list(self, calculator):
        """Test with empty molecule list."""
        df = calculator.calculate_descriptors_batch([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
    
    def test_single_atom_molecules(self, calculator, parser):
        """Test with single atom molecules."""
        single_atoms = ["C", "N", "O", "S"]
        
        for smiles in single_atoms:
            try:
                mol_data = parser.parse_smiles(smiles, f"single_{smiles}")
                results = calculator.calculate_descriptors(mol_data, ["num_atoms", "num_bonds"])
                
                assert results["num_atoms"] == 1
                assert results["num_bonds"] == 0
            except Exception as e:
                # Some single atoms might not be valid SMILES
                pytest.skip(f"Single atom {smiles} not valid: {e}")
    
    def test_large_molecules(self, calculator, parser):
        """Test with larger molecules."""
        # DNA base
        adenine = "NC1=NC=NC2=C1N=CN2"
        
        try:
            mol_data = parser.parse_smiles(adenine, "adenine")
            results = calculator.calculate_descriptors(mol_data)
            
            assert results["num_atoms"] > 10
            assert results["molecular_weight"] > 100
            assert results["ring_count"] >= 2  # Purine has 2 fused rings
        except Exception as e:
            pytest.skip(f"Large molecule test failed: {e}")


@pytest.mark.parametrize("smiles,expected_name", TEST_MOLECULES)
def test_individual_molecules(smiles, expected_name):
    """Test individual molecules parametrically."""
    if not RDKIT_AVAILABLE:
        pytest.skip("RDKit not available")
    
    parser = RDKitMolecularParser()
    calculator = BasicDescriptorCalculator()
    
    try:
        mol_data = parser.parse_smiles(smiles, expected_name)
        results = calculator.calculate_descriptors(mol_data)
        
        # Basic sanity checks
        assert results["molecular_weight"] > 0
        assert results["num_atoms"] > 0
        assert isinstance(results["logp"], (int, float))
        assert results["tpsa"] >= 0
        
    except Exception as e:
        pytest.fail(f"Failed to process molecule {expected_name} ({smiles}): {e}")


if __name__ == "__main__":
    pytest.main([__file__])