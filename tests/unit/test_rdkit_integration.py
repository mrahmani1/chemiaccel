"""
Unit tests for RDKit integration functionality.

Tests SMILES parsing, validation, descriptor calculation,
and error handling with diverse molecular structures.
"""


import pandas as pd
import pytest
from chemi_accel.core.exceptions import ChemiAccelError, InvalidMoleculeError
from chemi_accel.core.molecular import MolecularData
from chemi_accel.utils.rdkit_integration import (
    RDKIT_AVAILABLE,
    RDKitDescriptorCalculator,
    RDKitMolecularParser,
    calculate_basic_descriptors,
    check_rdkit_availability,
    parse_smiles,
    parse_smiles_batch,
)

# Skip all tests if RDKit is not available
pytestmark = pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")


class TestRDKitAvailability:
    """Test RDKit availability checking."""

    def test_rdkit_availability_check(self):
        """Test RDKit availability check function."""
        # Should not raise error if RDKit is available
        check_rdkit_availability()


class TestRDKitMolecularParser:
    """Test RDKit molecular parser functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = RDKitMolecularParser()

        # Test molecules with known properties
        self.test_molecules = {
            "ethanol": "CCO",
            "benzene": "c1ccccc1",
            "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "water": "O",
        }

        # Invalid SMILES for testing error handling
        self.invalid_smiles = [
            "",  # Empty string
            "   ",  # Whitespace only
            "C[C",  # Incomplete brackets
            "C1CCC",  # Incomplete ring
            "X",  # Invalid atom
        ]

    def test_parser_initialization(self):
        """Test parser initialization with different options."""
        # Default initialization
        parser1 = RDKitMolecularParser()
        assert parser1.sanitize is True
        assert parser1.remove_hs is True
        assert parser1.standardize is False

        # Custom initialization
        parser2 = RDKitMolecularParser(sanitize=False, standardize=True)
        assert parser2.sanitize is False
        # Note: standardize might be disabled if not available in this RDKit version

    def test_parse_valid_smiles(self):
        """Test parsing of valid SMILES strings."""
        for mol_name, smiles in self.test_molecules.items():
            mol_data = self.parser.parse_smiles(smiles, mol_name)

            assert isinstance(mol_data, MolecularData)
            assert mol_data.smiles is not None
            assert mol_data.mol_id == mol_name
            assert mol_data.num_atoms > 0
            assert mol_data.num_bonds >= 0
            assert mol_data.molecular_weight > 0
            assert "rdkit_mol" in mol_data.backend_data
            assert mol_data.is_valid()

    def test_parse_invalid_smiles(self):
        """Test parsing of invalid SMILES strings."""
        for invalid_smiles in self.invalid_smiles:
            with pytest.raises(InvalidMoleculeError):
                self.parser.parse_smiles(invalid_smiles)

    def test_parse_smiles_batch_success(self):
        """Test batch parsing of valid SMILES."""
        smiles_list = list(self.test_molecules.values())
        mol_ids = list(self.test_molecules.keys())

        molecules = self.parser.parse_smiles_batch(smiles_list, mol_ids)

        assert len(molecules) == len(smiles_list)
        assert all(isinstance(mol, MolecularData) for mol in molecules)
        assert all(mol.is_valid() for mol in molecules)

        # Check that mol_ids are correctly assigned
        for mol, expected_id in zip(molecules, mol_ids):
            assert mol.mol_id == expected_id

    def test_parse_smiles_batch_with_errors(self):
        """Test batch parsing with some invalid SMILES."""
        mixed_smiles = ["CCO", "invalid", "CCC", "", "c1ccccc1"]

        # Test skip error handling (default)
        molecules = self.parser.parse_smiles_batch(mixed_smiles, error_handling="skip")
        assert len(molecules) == 3  # Only valid molecules

        # Test raise error handling
        with pytest.raises(ChemiAccelError):
            self.parser.parse_smiles_batch(mixed_smiles, error_handling="raise")

        # Test collect error handling
        molecules = self.parser.parse_smiles_batch(
            mixed_smiles, error_handling="collect"
        )
        assert len(molecules) == 3  # Only valid molecules
        if molecules:
            assert "parsing_errors" in molecules[0].backend_data

    def test_parse_smiles_batch_mol_ids_mismatch(self):
        """Test batch parsing with mismatched mol_ids length."""
        smiles_list = ["CCO", "CCC"]
        mol_ids = ["ethanol"]  # Wrong length

        with pytest.raises(ChemiAccelError):
            self.parser.parse_smiles_batch(smiles_list, mol_ids)

    def test_molecule_validation(self):
        """Test molecule validation functionality."""
        # Valid molecule
        mol_data = self.parser.parse_smiles("CCO", "ethanol")
        is_valid, issues = self.parser.validate_molecule(mol_data)
        assert is_valid
        assert len(issues) == 0

        # Test with molecule without RDKit object
        invalid_mol = MolecularData(smiles="CCO", mol_id="test")
        is_valid, issues = self.parser.validate_molecule(invalid_mol)
        assert not is_valid
        assert "No RDKit mol object found" in issues

    def test_standardization_option(self):
        """Test molecular standardization option."""
        parser_std = RDKitMolecularParser(standardize=True)

        # Test with a molecule that has tautomers
        tautomer_smiles = "CC(C)=O"  # Simple ketone
        mol_data = parser_std.parse_smiles(tautomer_smiles)

        assert mol_data.is_valid()
        assert "rdkit_mol" in mol_data.backend_data


class TestRDKitDescriptorCalculator:
    """Test RDKit descriptor calculator functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = RDKitDescriptorCalculator()
        self.parser = RDKitMolecularParser()

        # Parse test molecules
        self.test_molecules = [
            self.parser.parse_smiles("CCO", "ethanol"),
            self.parser.parse_smiles("c1ccccc1", "benzene"),
            self.parser.parse_smiles("O", "water"),
        ]

    def test_calculator_initialization(self):
        """Test descriptor calculator initialization."""
        assert len(self.calculator.descriptor_functions) > 0
        assert "molecular_weight" in self.calculator.descriptor_functions
        assert "logp" in self.calculator.descriptor_functions
        assert "tpsa" in self.calculator.descriptor_functions

    def test_get_available_descriptors(self):
        """Test getting list of available descriptors."""
        descriptors = self.calculator.get_available_descriptors()
        assert isinstance(descriptors, list)
        assert len(descriptors) > 0
        assert "molecular_weight" in descriptors
        assert "num_atoms" in descriptors

    def test_calculate_single_molecule_descriptors(self):
        """Test descriptor calculation for single molecule."""
        mol_data = self.test_molecules[0]  # ethanol
        descriptors = self.calculator.calculate_descriptors(mol_data)

        assert isinstance(descriptors, dict)
        assert len(descriptors) > 0

        # Check specific values for ethanol (CCO)
        assert descriptors["num_atoms"] == 3  # C, C, O
        assert descriptors["num_bonds"] == 2  # C-C, C-O
        assert descriptors["molecular_weight"] > 40  # Should be ~46
        assert descriptors["num_hbd"] == 1  # OH group
        assert descriptors["num_hba"] == 1  # O atom

    def test_calculate_specific_descriptors(self):
        """Test calculation of specific descriptors only."""
        mol_data = self.test_molecules[0]  # ethanol
        desc_names = ["molecular_weight", "num_atoms", "logp"]

        descriptors = self.calculator.calculate_descriptors(mol_data, desc_names)

        assert len(descriptors) == 3
        assert all(name in descriptors for name in desc_names)
        assert all(isinstance(val, float) for val in descriptors.values())

    def test_calculate_batch_descriptors(self):
        """Test batch descriptor calculation."""
        df = self.calculator.calculate_descriptors_batch(self.test_molecules)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(self.test_molecules)
        assert "mol_id" in df.columns
        assert "smiles" in df.columns
        assert "molecular_weight" in df.columns

        # Check that all values are numeric (except identifiers)
        numeric_cols = [col for col in df.columns if col not in ["mol_id", "smiles"]]
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(df[col])

    def test_calculate_batch_with_invalid_molecule(self):
        """Test batch calculation with invalid molecule."""
        # Create molecule without RDKit object
        invalid_mol = MolecularData(smiles="CCO", mol_id="invalid")
        molecules = self.test_molecules + [invalid_mol]

        df = self.calculator.calculate_descriptors_batch(molecules)

        assert len(df) == len(molecules)
        # Last row should have NaN values for descriptors
        last_row = df.iloc[-1]
        numeric_cols = [col for col in df.columns if col not in ["mol_id", "smiles"]]
        assert any(pd.isna(last_row[col]) for col in numeric_cols)

    def test_unknown_descriptor_handling(self):
        """Test handling of unknown descriptor names."""
        mol_data = self.test_molecules[0]
        desc_names = ["molecular_weight", "unknown_descriptor", "num_atoms"]

        # Should skip unknown descriptors with warning
        descriptors = self.calculator.calculate_descriptors(mol_data, desc_names)

        assert "molecular_weight" in descriptors
        assert "num_atoms" in descriptors
        assert "unknown_descriptor" not in descriptors


class TestConvenienceFunctions:
    """Test convenience functions for easy access."""

    def test_parse_smiles_function(self):
        """Test parse_smiles convenience function."""
        mol_data = parse_smiles("CCO", "ethanol")

        assert isinstance(mol_data, MolecularData)
        assert mol_data.smiles == "CCO"
        assert mol_data.mol_id == "ethanol"
        assert mol_data.is_valid()

    def test_parse_smiles_batch_function(self):
        """Test parse_smiles_batch convenience function."""
        smiles_list = ["CCO", "CCC", "c1ccccc1"]
        molecules = parse_smiles_batch(smiles_list)

        assert len(molecules) == 3
        assert all(isinstance(mol, MolecularData) for mol in molecules)
        assert all(mol.is_valid() for mol in molecules)

    def test_calculate_basic_descriptors_function(self):
        """Test calculate_basic_descriptors convenience function."""
        molecules = parse_smiles_batch(["CCO", "CCC", "c1ccccc1"])
        df = calculate_basic_descriptors(molecules)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "molecular_weight" in df.columns
        assert "smiles" in df.columns


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = RDKitMolecularParser()
        self.calculator = RDKitDescriptorCalculator()

    def test_empty_batch_processing(self):
        """Test processing of empty molecule batch."""
        df = self.calculator.calculate_descriptors_batch([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_large_molecule_processing(self):
        """Test processing of larger molecules."""
        # Larger molecule (cholesterol-like structure)
        large_smiles = "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C"

        mol_data = self.parser.parse_smiles(large_smiles, "large_molecule")
        assert mol_data.is_valid()
        assert mol_data.num_atoms > 20

        descriptors = self.calculator.calculate_descriptors(mol_data)
        assert descriptors["molecular_weight"] > 300

    def test_special_characters_in_mol_id(self):
        """Test handling of special characters in molecule IDs."""
        special_ids = ["mol-1", "mol_2", "mol.3", "mol@4", "mol#5"]
        smiles_list = ["CCO"] * len(special_ids)

        molecules = self.parser.parse_smiles_batch(smiles_list, special_ids)
        assert len(molecules) == len(special_ids)

        for mol, expected_id in zip(molecules, special_ids):
            assert mol.mol_id == expected_id

    def test_whitespace_handling(self):
        """Test handling of whitespace in SMILES."""
        smiles_with_whitespace = [
            "  CCO  ",  # Leading/trailing spaces
            "\tCCC\t",  # Tabs
            "\nO\n",  # Newlines
        ]

        for smiles in smiles_with_whitespace:
            mol_data = self.parser.parse_smiles(smiles)
            assert mol_data.is_valid()
            assert mol_data.smiles.strip() == smiles.strip()


class TestIntegrationWithCalculator:
    """Test integration with main Calculator class."""

    def test_calculator_with_rdkit_integration(self):
        """Test that Calculator class works with RDKit integration."""
        from chemi_accel import Calculator

        calc = Calculator(backend="cpu")

        # Test basic functionality
        smiles_list = ["CCO", "CCC", "c1ccccc1"]
        result = calc.pandas(smiles_list)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "smiles" in result.columns
        assert "molecular_weight" in result.columns

        # Check that we get real values, not zeros
        assert all(result["molecular_weight"] > 0)
        assert all(result["num_atoms"] > 0)


if __name__ == "__main__":
    pytest.main([__file__])
