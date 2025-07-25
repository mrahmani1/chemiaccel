"""
Tests for constitutional molecular descriptors.

This module tests the ConstitutionalDescriptorCalculator class and related
functions for calculating comprehensive constitutional descriptors including
atom counts, bond counts, complexity measures, and information content.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from src.chemi_accel.core.exceptions import ChemiAccelError, DescriptorCalculationError
from src.chemi_accel.core.molecular import MolecularData
from src.chemi_accel.descriptors.constitutional import (
    ConstitutionalDescriptorCalculator,
    calculate_constitutional_descriptors,
    get_constitutional_descriptor_info,
)


@pytest.fixture
def sample_molecules():
    """Create sample molecules for testing."""
    if not RDKIT_AVAILABLE:
        pytest.skip("RDKit not available")

    molecules = []
    smiles_list = [
        "CCO",  # Ethanol
        "c1ccccc1",  # Benzene
        "CC(=O)O",  # Acetic acid
        "CCN(CC)CC",  # Triethylamine
        "c1ccc2ccccc2c1",  # Naphthalene
        "CC(C)(C)O",  # tert-Butanol
        "CCCCCCCCCCCCCCCC(=O)O",  # Palmitic acid
        "C1CCC2CCCCC2C1",  # Decalin
    ]

    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mol_data = MolecularData(
                mol_id=f"mol_{i}", smiles=smiles, backend_data={"rdkit_mol": mol}
            )
            molecules.append(mol_data)

    return molecules


@pytest.fixture
def calculator():
    """Create constitutional descriptor calculator."""
    if not RDKIT_AVAILABLE:
        pytest.skip("RDKit not available")

    return ConstitutionalDescriptorCalculator(use_cuda=False)


class TestConstitutionalDescriptorCalculator:
    """Test ConstitutionalDescriptorCalculator class."""

    def test_initialization(self):
        """Test calculator initialization."""
        if not RDKIT_AVAILABLE:
            pytest.skip("RDKit not available")

        calc = ConstitutionalDescriptorCalculator(use_cuda=False)
        assert calc is not None
        assert len(calc.descriptor_registry) > 40  # Should have many descriptors
        assert not calc.use_cuda

    def test_initialization_without_rdkit(self):
        """Test initialization fails without RDKit."""
        with patch("src.chemi_accel.descriptors.constitutional.RDKIT_AVAILABLE", False):
            with pytest.raises(ChemiAccelError, match="RDKit is required"):
                ConstitutionalDescriptorCalculator()

    def test_get_available_descriptors(self, calculator):
        """Test getting available descriptor names."""
        descriptors = calculator.get_available_descriptors()
        assert isinstance(descriptors, list)
        assert len(descriptors) > 40

        # Check for key descriptors
        expected_descriptors = [
            "molecular_weight",
            "num_atoms",
            "num_heavy_atoms",
            "num_carbons",
            "num_nitrogens",
            "num_oxygens",
            "bertz_complexity",
            "information_content",
        ]
        for desc in expected_descriptors:
            assert desc in descriptors

    def test_get_descriptors_by_category(self, calculator):
        """Test getting descriptors by category."""
        # Test basic category
        basic_descriptors = calculator.get_descriptors_by_category("basic")
        assert isinstance(basic_descriptors, list)
        assert "molecular_weight" in basic_descriptors

        # Test element counts category
        element_descriptors = calculator.get_descriptors_by_category("element_counts")
        assert "num_carbons" in element_descriptors
        assert "num_nitrogens" in element_descriptors

        # Test complexity category
        complexity_descriptors = calculator.get_descriptors_by_category("complexity")
        assert "bertz_complexity" in complexity_descriptors
        assert "information_content" in complexity_descriptors

    def test_get_descriptor_categories(self, calculator):
        """Test getting all descriptor categories."""
        categories = calculator.get_descriptor_categories()
        assert isinstance(categories, list)

        expected_categories = [
            "basic",
            "atom_counts",
            "element_counts",
            "hybridization",
            "bond_counts",
            "rings",
            "complexity",
            "indices",
            "accessibility",
        ]
        for category in expected_categories:
            assert category in categories

    def test_calculate_single_descriptor_basic(self, calculator, sample_molecules):
        """Test calculating single basic descriptors."""
        mol_data = sample_molecules[0]  # Ethanol

        # Test molecular weight
        mw = calculator.calculate_single_descriptor(mol_data, "molecular_weight")
        assert isinstance(mw, float)
        assert 40 < mw < 50  # Ethanol MW ≈ 46

        # Test atom count
        num_atoms = calculator.calculate_single_descriptor(mol_data, "num_atoms")
        assert isinstance(num_atoms, int)
        assert num_atoms == 3  # C2H6O has 3 heavy atoms (RDKit removes H by default)

        # Test heavy atom count
        heavy_atoms = calculator.calculate_single_descriptor(
            mol_data, "num_heavy_atoms"
        )
        assert isinstance(heavy_atoms, int)
        assert heavy_atoms == 3  # C, C, O

    def test_calculate_single_descriptor_element_counts(
        self, calculator, sample_molecules
    ):
        """Test calculating element-specific counts."""
        mol_data = sample_molecules[0]  # Ethanol (CCO)

        # Test carbon count
        carbons = calculator.calculate_single_descriptor(mol_data, "num_carbons")
        assert carbons == 2

        # Test oxygen count
        oxygens = calculator.calculate_single_descriptor(mol_data, "num_oxygens")
        assert oxygens == 1

        # Test nitrogen count (should be 0 for ethanol)
        nitrogens = calculator.calculate_single_descriptor(mol_data, "num_nitrogens")
        assert nitrogens == 0

        # Test halogen count (should be 0 for ethanol)
        halogens = calculator.calculate_single_descriptor(mol_data, "num_halogens")
        assert halogens == 0

    def test_calculate_single_descriptor_complexity(self, calculator, sample_molecules):
        """Test calculating complexity descriptors."""
        mol_data = sample_molecules[1]  # Benzene

        # Test Bertz complexity
        bertz = calculator.calculate_single_descriptor(mol_data, "bertz_complexity")
        assert isinstance(bertz, float)
        assert bertz > 0

        # Test information content
        info_content = calculator.calculate_single_descriptor(
            mol_data, "information_content"
        )
        assert isinstance(info_content, float)
        assert info_content >= 0

        # Test molecular complexity
        mol_complexity = calculator.calculate_single_descriptor(
            mol_data, "molecular_complexity"
        )
        assert isinstance(mol_complexity, float)
        assert mol_complexity > 0

    def test_calculate_single_descriptor_rings(self, calculator, sample_molecules):
        """Test calculating ring-related descriptors."""
        benzene_data = sample_molecules[1]  # Benzene

        # Test ring count
        rings = calculator.calculate_single_descriptor(benzene_data, "ring_count")
        assert rings == 1

        # Test aromatic rings
        aromatic_rings = calculator.calculate_single_descriptor(
            benzene_data, "num_aromatic_rings"
        )
        assert aromatic_rings == 1

        # Test largest ring size
        largest_ring = calculator.calculate_single_descriptor(
            benzene_data, "largest_ring_size"
        )
        assert largest_ring == 6

    def test_calculate_single_descriptor_bonds(self, calculator, sample_molecules):
        """Test calculating bond-related descriptors."""
        mol_data = sample_molecules[0]  # Ethanol

        # Test total bonds
        total_bonds = calculator.calculate_single_descriptor(mol_data, "num_bonds")
        assert isinstance(total_bonds, int)
        assert (
            total_bonds == 2
        )  # Ethanol has 2 heavy atom bonds (RDKit removes H by default)

        # Test single bonds
        single_bonds = calculator.calculate_single_descriptor(
            mol_data, "num_single_bonds"
        )
        assert isinstance(single_bonds, int)
        assert single_bonds == 2  # All heavy atom bonds in ethanol are single

        # Test double bonds (should be 0 for ethanol)
        double_bonds = calculator.calculate_single_descriptor(
            mol_data, "num_double_bonds"
        )
        assert double_bonds == 0

    def test_calculate_single_descriptor_hybridization(
        self, calculator, sample_molecules
    ):
        """Test calculating hybridization-specific descriptors."""
        mol_data = sample_molecules[0]  # Ethanol

        # Test sp3 carbons
        sp3_carbons = calculator.calculate_single_descriptor(
            mol_data, "num_sp3_carbons"
        )
        assert sp3_carbons == 2  # Both carbons in ethanol are sp3

        # Test sp2 carbons (should be 0 for ethanol)
        sp2_carbons = calculator.calculate_single_descriptor(
            mol_data, "num_sp2_carbons"
        )
        assert sp2_carbons == 0

        # Test fraction CSP3
        frac_csp3 = calculator.calculate_single_descriptor(mol_data, "fraction_csp3")
        assert isinstance(frac_csp3, float)
        assert frac_csp3 == 1.0  # All carbons are sp3

    def test_calculate_single_descriptor_indices(self, calculator, sample_molecules):
        """Test calculating constitutional indices."""
        mol_data = sample_molecules[0]  # Ethanol

        # Test constitutional indices
        for i in range(1, 6):
            index_name = f"constitutional_index_{i}"
            index_value = calculator.calculate_single_descriptor(mol_data, index_name)
            assert isinstance(index_value, float)
            assert index_value >= 0

    def test_calculate_single_descriptor_accessibility(
        self, calculator, sample_molecules
    ):
        """Test calculating accessibility descriptors."""
        mol_data = sample_molecules[0]  # Ethanol (simple molecule)

        # Test synthetic accessibility
        sa_score = calculator.calculate_single_descriptor(
            mol_data, "synthetic_accessibility"
        )
        assert isinstance(sa_score, float)
        assert 1.0 <= sa_score <= 10.0
        assert sa_score < 3.0  # Ethanol should be easy to synthesize

        # Test natural product likeness
        np_score = calculator.calculate_single_descriptor(
            mol_data, "natural_product_likeness"
        )
        assert isinstance(np_score, float)
        assert 0.0 <= np_score <= 1.0

    def test_calculate_single_descriptor_invalid(self, calculator, sample_molecules):
        """Test calculating invalid descriptor raises error."""
        mol_data = sample_molecules[0]

        with pytest.raises(
            DescriptorCalculationError, match="Unknown constitutional descriptor"
        ):
            calculator.calculate_single_descriptor(mol_data, "invalid_descriptor")

    def test_calculate_single_descriptor_no_rdkit_mol(self, calculator):
        """Test calculating descriptor without RDKit mol raises error."""
        mol_data = MolecularData(
            mol_id="test", smiles="CCO", backend_data={}  # No rdkit_mol
        )

        with pytest.raises(
            DescriptorCalculationError, match="No RDKit mol object found"
        ):
            calculator.calculate_single_descriptor(mol_data, "molecular_weight")

    def test_calculate_descriptors_multiple(self, calculator, sample_molecules):
        """Test calculating multiple descriptors for single molecule."""
        mol_data = sample_molecules[0]  # Ethanol

        descriptor_names = [
            "molecular_weight",
            "num_atoms",
            "num_heavy_atoms",
            "num_carbons",
            "bertz_complexity",
        ]

        descriptors = calculator.calculate_descriptors(mol_data, descriptor_names)

        assert isinstance(descriptors, dict)
        assert len(descriptors) == len(descriptor_names)

        for desc_name in descriptor_names:
            assert desc_name in descriptors
            assert descriptors[desc_name] is not None

    def test_calculate_descriptors_all(self, calculator, sample_molecules):
        """Test calculating all descriptors for single molecule."""
        mol_data = sample_molecules[0]  # Ethanol

        descriptors = calculator.calculate_descriptors(mol_data)

        assert isinstance(descriptors, dict)
        assert len(descriptors) > 40  # Should have many descriptors

        # Check key descriptors are present
        key_descriptors = [
            "molecular_weight",
            "num_atoms",
            "num_heavy_atoms",
            "bertz_complexity",
            "information_content",
        ]
        for desc in key_descriptors:
            assert desc in descriptors

    def test_calculate_descriptors_batch(self, calculator, sample_molecules):
        """Test calculating descriptors for batch of molecules."""
        descriptor_names = [
            "molecular_weight",
            "num_atoms",
            "num_heavy_atoms",
            "num_carbons",
            "num_oxygens",
            "bertz_complexity",
        ]

        df = calculator.calculate_descriptors_batch(sample_molecules, descriptor_names)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_molecules)

        # Check columns
        expected_cols = ["mol_id", "smiles"] + descriptor_names
        for col in expected_cols:
            assert col in df.columns

        # Check data types and values
        assert df["molecular_weight"].dtype in [np.float64, np.float32]
        assert df["num_atoms"].dtype in [np.int64, np.int32]
        assert all(df["molecular_weight"] > 0)
        assert all(df["num_atoms"] > 0)

    def test_calculate_descriptors_batch_empty(self, calculator):
        """Test calculating descriptors for empty batch."""
        df = calculator.calculate_descriptors_batch([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_calculate_descriptors_batch_with_errors(self, calculator):
        """Test batch calculation with some invalid molecules."""
        # Create a mix of valid and invalid molecules
        molecules = []

        # Valid molecule
        mol = Chem.MolFromSmiles("CCO")
        mol_data = MolecularData(
            mol_id="valid", smiles="CCO", backend_data={"rdkit_mol": mol}
        )
        molecules.append(mol_data)

        # Invalid molecule (no RDKit mol)
        invalid_mol_data = MolecularData(
            mol_id="invalid", smiles="invalid", backend_data={}
        )
        molecules.append(invalid_mol_data)

        df = calculator.calculate_descriptors_batch(molecules, ["molecular_weight"])

        assert len(df) == 2
        assert not pd.isna(df.iloc[0]["molecular_weight"])  # Valid molecule
        assert pd.isna(df.iloc[1]["molecular_weight"])  # Invalid molecule

    def test_get_descriptor_statistics(self, calculator, sample_molecules):
        """Test calculating descriptor statistics."""
        descriptor_names = ["molecular_weight", "num_atoms", "bertz_complexity"]
        df = calculator.calculate_descriptors_batch(sample_molecules, descriptor_names)

        stats = calculator.get_descriptor_statistics(df, descriptor_names)

        assert isinstance(stats, pd.DataFrame)
        assert len(stats.columns) == len(descriptor_names)

        # Check statistics are present
        expected_stats = [
            "count",
            "mean",
            "std",
            "min",
            "25%",
            "50%",
            "75%",
            "max",
            "missing",
            "missing_pct",
        ]
        for stat in expected_stats:
            assert stat in stats.index


class TestConstitutionalDescriptorHelperFunctions:
    """Test helper functions for constitutional descriptors."""

    def test_element_counting_functions(self):
        """Test element counting helper functions."""
        if not RDKIT_AVAILABLE:
            pytest.skip("RDKit not available")

        calc = ConstitutionalDescriptorCalculator(use_cuda=False)
        mol = Chem.MolFromSmiles("CCO")  # Ethanol

        # Test carbon counting
        carbons = calc._count_atoms_by_element(mol, 6)
        assert carbons == 2

        # Test oxygen counting
        oxygens = calc._count_atoms_by_element(mol, 8)
        assert oxygens == 1

        # Test halogen counting
        halogens = calc._count_halogens(mol)
        assert halogens == 0

        # Test with halogenated molecule
        mol_hal = Chem.MolFromSmiles("CCF")  # Fluoroethane
        halogens_hal = calc._count_halogens(mol_hal)
        assert halogens_hal == 1

    def test_hybridization_counting_functions(self):
        """Test hybridization counting functions."""
        if not RDKIT_AVAILABLE:
            pytest.skip("RDKit not available")

        calc = ConstitutionalDescriptorCalculator(use_cuda=False)

        # Test sp3 carbons (ethanol)
        mol_sp3 = Chem.MolFromSmiles("CCO")
        sp3_count = calc._count_sp3_carbons(mol_sp3)
        assert sp3_count == 2

        # Test sp2 carbons (ethene)
        mol_sp2 = Chem.MolFromSmiles("C=C")
        sp2_count = calc._count_sp2_carbons(mol_sp2)
        assert sp2_count == 2

        # Test sp carbons (acetylene)
        mol_sp = Chem.MolFromSmiles("C#C")
        sp_count = calc._count_sp_carbons(mol_sp)
        assert sp_count == 2

    def test_bond_counting_functions(self):
        """Test bond counting functions."""
        if not RDKIT_AVAILABLE:
            pytest.skip("RDKit not available")

        calc = ConstitutionalDescriptorCalculator(use_cuda=False)

        # Test single bonds
        mol_single = Chem.MolFromSmiles("CCO")
        single_count = calc._count_bonds_by_type(mol_single, Chem.BondType.SINGLE)
        assert single_count == 2  # All heavy atom bonds in ethanol are single

        # Test double bonds
        mol_double = Chem.MolFromSmiles("C=C")
        double_count = calc._count_bonds_by_type(mol_double, Chem.BondType.DOUBLE)
        assert double_count == 1

        # Test triple bonds
        mol_triple = Chem.MolFromSmiles("C#C")
        triple_count = calc._count_bonds_by_type(mol_triple, Chem.BondType.TRIPLE)
        assert triple_count == 1

    def test_ring_size_functions(self):
        """Test ring size calculation functions."""
        if not RDKIT_AVAILABLE:
            pytest.skip("RDKit not available")

        calc = ConstitutionalDescriptorCalculator(use_cuda=False)

        # Test benzene (6-membered ring)
        mol_benzene = Chem.MolFromSmiles("c1ccccc1")
        largest = calc._calculate_largest_ring_size(mol_benzene)
        smallest = calc._calculate_smallest_ring_size(mol_benzene)
        assert largest == 6
        assert smallest == 6

        # Test molecule with no rings
        mol_no_ring = Chem.MolFromSmiles("CCO")
        largest_no_ring = calc._calculate_largest_ring_size(mol_no_ring)
        smallest_no_ring = calc._calculate_smallest_ring_size(mol_no_ring)
        assert largest_no_ring == 0
        assert smallest_no_ring == 0

    def test_complexity_functions(self):
        """Test complexity calculation functions."""
        if not RDKIT_AVAILABLE:
            pytest.skip("RDKit not available")

        calc = ConstitutionalDescriptorCalculator(use_cuda=False)

        # Test molecular complexity
        mol = Chem.MolFromSmiles("c1ccccc1")  # Benzene
        complexity = calc._calculate_molecular_complexity(mol)
        assert isinstance(complexity, float)
        assert complexity > 0

        # Test information content
        info_content = calc._calculate_information_content(mol)
        assert isinstance(info_content, float)
        assert info_content >= 0

        # Test bond information content
        bond_info = calc._calculate_bond_information_content(mol)
        assert isinstance(bond_info, float)
        assert bond_info >= 0

    def test_constitutional_indices(self):
        """Test constitutional index calculations."""
        if not RDKIT_AVAILABLE:
            pytest.skip("RDKit not available")

        calc = ConstitutionalDescriptorCalculator(use_cuda=False)
        mol = Chem.MolFromSmiles("CCO")  # Ethanol

        # Test all constitutional indices
        for i in range(1, 6):
            func_name = f"_calculate_constitutional_index_{i}"
            func = getattr(calc, func_name)
            index_value = func(mol)
            assert isinstance(index_value, float)
            assert index_value >= 0

    def test_accessibility_functions(self):
        """Test accessibility scoring functions."""
        if not RDKIT_AVAILABLE:
            pytest.skip("RDKit not available")

        calc = ConstitutionalDescriptorCalculator(use_cuda=False)

        # Test synthetic accessibility
        mol_simple = Chem.MolFromSmiles("CCO")  # Simple molecule
        sa_score = calc._calculate_synthetic_accessibility(mol_simple)
        assert isinstance(sa_score, float)
        assert 1.0 <= sa_score <= 10.0
        assert sa_score < 5.0  # Simple molecule should be easy

        # Test natural product likeness
        np_score = calc._calculate_natural_product_likeness(mol_simple)
        assert isinstance(np_score, float)
        assert 0.0 <= np_score <= 1.0


class TestConstitutionalDescriptorConvenienceFunctions:
    """Test convenience functions for constitutional descriptors."""

    def test_calculate_constitutional_descriptors_basic(self, sample_molecules):
        """Test convenience function with basic descriptor set."""
        df = calculate_constitutional_descriptors(sample_molecules, "basic")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_molecules)

        # Check expected columns
        expected_descriptors = [
            "molecular_weight",
            "num_atoms",
            "num_heavy_atoms",
            "num_carbons",
            "num_nitrogens",
            "num_oxygens",
            "num_bonds",
            "ring_count",
            "bertz_complexity",
            "information_content",
        ]
        for desc in expected_descriptors:
            assert desc in df.columns

    def test_calculate_constitutional_descriptors_all(self, sample_molecules):
        """Test convenience function with all descriptors."""
        df = calculate_constitutional_descriptors(sample_molecules, "all")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_molecules)
        assert len(df.columns) > 40  # Should have many descriptors

    def test_calculate_constitutional_descriptors_by_category(self, sample_molecules):
        """Test convenience function with category-based selection."""
        df = calculate_constitutional_descriptors(sample_molecules, "element_counts")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_molecules)

        # Check element count descriptors are present
        element_descriptors = ["num_carbons", "num_nitrogens", "num_oxygens"]
        for desc in element_descriptors:
            assert desc in df.columns

    def test_calculate_constitutional_descriptors_invalid_set(self, sample_molecules):
        """Test convenience function with invalid descriptor set."""
        with pytest.raises(ValueError, match="Unknown descriptor set"):
            calculate_constitutional_descriptors(sample_molecules, "invalid_set")

    def test_get_constitutional_descriptor_info(self):
        """Test getting descriptor information."""
        if not RDKIT_AVAILABLE:
            pytest.skip("RDKit not available")

        info = get_constitutional_descriptor_info()

        assert isinstance(info, dict)
        assert len(info) > 40  # Should have many descriptors

        # Check structure of info
        for name, desc_info in info.items():
            assert "description" in desc_info
            assert "category" in desc_info
            assert isinstance(desc_info["description"], str)
            assert isinstance(desc_info["category"], str)

        # Check specific descriptors
        assert "molecular_weight" in info
        assert "bertz_complexity" in info
        assert info["molecular_weight"]["category"] == "basic"
        assert info["bertz_complexity"]["category"] == "complexity"


@pytest.mark.integration
class TestConstitutionalDescriptorIntegration:
    """Integration tests for constitutional descriptors."""

    def test_large_molecule_batch(self):
        """Test calculating descriptors for large batch of molecules."""
        if not RDKIT_AVAILABLE:
            pytest.skip("RDKit not available")

        # Create larger batch of molecules
        smiles_list = [
            "CCO",
            "c1ccccc1",
            "CC(=O)O",
            "CCN(CC)CC",
            "c1ccc2ccccc2c1",
            "CC(C)(C)O",
            "CCCCCCCCCCCCCCCC(=O)O",
            "C1CCC2CCCCC2C1",
            "CC(C)CC(C)(C)C",
            "c1ccc(cc1)c2ccccc2",
            "CCCCCCCCCCO",
            "CC(=O)Nc1ccccc1",
            "c1ccc2c(c1)ccc3ccccc32",
            "CCCCCCCCCCCCCCCCCCCC(=O)O",
        ] * 10  # 140 molecules

        molecules = []
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mol_data = MolecularData(
                    mol_id=f"mol_{i}", smiles=smiles, backend_data={"rdkit_mol": mol}
                )
                molecules.append(mol_data)

        calc = ConstitutionalDescriptorCalculator(use_cuda=False)
        df = calc.calculate_descriptors_batch(
            molecules, ["molecular_weight", "bertz_complexity"]
        )

        assert len(df) == len(molecules)
        assert all(df["molecular_weight"] > 0)
        assert all(
            df["bertz_complexity"] >= 0
        )  # Some simple molecules may have 0 complexity

    def test_descriptor_correlation_analysis(self, sample_molecules):
        """Test correlation analysis between different descriptors."""
        calc = ConstitutionalDescriptorCalculator(use_cuda=False)

        descriptor_names = [
            "molecular_weight",
            "num_atoms",
            "num_heavy_atoms",
            "bertz_complexity",
            "information_content",
            "ring_count",
        ]

        df = calc.calculate_descriptors_batch(sample_molecules, descriptor_names)

        # Calculate correlations
        corr_matrix = df[descriptor_names].corr()

        # Check expected correlations
        assert (
            corr_matrix.loc["molecular_weight", "num_atoms"] > 0.5
        )  # Should be correlated
        assert (
            corr_matrix.loc["num_atoms", "num_heavy_atoms"] > 0.8
        )  # Should be highly correlated

        # Check matrix properties
        assert corr_matrix.shape == (len(descriptor_names), len(descriptor_names))
        assert all(np.diag(corr_matrix) == 1.0)  # Diagonal should be 1

    def test_descriptor_value_ranges(self, sample_molecules):
        """Test that descriptor values are within expected ranges."""
        calc = ConstitutionalDescriptorCalculator(use_cuda=False)

        df = calc.calculate_descriptors_batch(sample_molecules)

        # Check molecular weight range
        assert all(df["molecular_weight"] > 0)
        assert all(df["molecular_weight"] < 1000)  # Reasonable upper bound

        # Check atom counts
        assert all(df["num_atoms"] > 0)
        assert all(df["num_heavy_atoms"] > 0)
        assert all(df["num_heavy_atoms"] <= df["num_atoms"])

        # Check element counts are non-negative
        element_cols = [
            col for col in df.columns if col.startswith("num_") and "atoms" not in col
        ]
        for col in element_cols:
            if col in df.columns:
                assert all(df[col] >= 0)

        # Check fractions are between 0 and 1
        if "fraction_csp3" in df.columns:
            assert all(df["fraction_csp3"] >= 0)
            assert all(df["fraction_csp3"] <= 1)

        # Check synthetic accessibility is in valid range
        if "synthetic_accessibility" in df.columns:
            assert all(df["synthetic_accessibility"] >= 1)
            assert all(df["synthetic_accessibility"] <= 10)
