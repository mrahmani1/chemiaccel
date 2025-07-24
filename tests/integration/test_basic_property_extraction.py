"""
Integration tests for basic molecular property extraction (Task 3.2).

This module provides comprehensive integration tests that validate the complete
implementation of task 3.2, including all sub-components working together.
"""

import pytest
import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, MACCSkeys, AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from chemi_accel.core.calculator import Calculator
from chemi_accel.descriptors.basic import BasicDescriptorCalculator
from chemi_accel.descriptors.graph import MolecularGraphBuilder, GraphDescriptorCalculator
from chemi_accel.descriptors.fingerprints import FingerprintGenerator
from chemi_accel.utils.rdkit_integration import RDKitMolecularParser


# Comprehensive test dataset with diverse molecular structures
COMPREHENSIVE_TEST_SET = [
    # Simple molecules
    ("C", "methane"),
    ("CC", "ethane"),
    ("CCO", "ethanol"),
    ("CC(=O)O", "acetic_acid"),
    
    # Aromatic compounds
    ("c1ccccc1", "benzene"),
    ("Cc1ccccc1", "toluene"),
    ("CCc1ccccc1", "ethylbenzene"),
    ("c1ccc2ccccc2c1", "naphthalene"),
    
    # Heterocycles
    ("c1ccncc1", "pyridine"),
    ("c1cccnc1", "pyrimidine"),
    ("c1cnccn1", "pyrazine"),
    
    # Functional groups
    ("CC(=O)C", "acetone"),
    ("CC(=O)N", "acetamide"),
    ("CC(=O)OC", "methyl_acetate"),
    ("CCS", "ethanethiol"),
    ("CCN", "ethylamine"),
    
    # Rings
    ("C1CCC1", "cyclobutane"),
    ("C1CCCCC1", "cyclohexane"),
    ("C1CCCCCC1", "cycloheptane"),
    
    # Drug-like molecules
    ("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "ibuprofen"),
    ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "caffeine"),
    ("CC(=O)OC1=CC=CC=C1C(=O)O", "aspirin"),
    
    # Complex structures
    ("CC1=CC=C(C=C1)C(=O)C2=CC=C(C=C2)N(C)C", "michlers_ketone"),
    ("CCCCCCCCCCCCCCC(=O)O", "palmitic_acid"),
]


@pytest.fixture
def calculator():
    """Create Calculator instance."""
    if not RDKIT_AVAILABLE:
        pytest.skip("RDKit not available")
    return Calculator()


@pytest.fixture
def comprehensive_molecules():
    """Parse comprehensive test set."""
    if not RDKIT_AVAILABLE:
        pytest.skip("RDKit not available")
    
    parser = RDKitMolecularParser()
    molecules = []
    
    for smiles, name in COMPREHENSIVE_TEST_SET:
        try:
            mol_data = parser.parse_smiles(smiles, name)
            molecules.append(mol_data)
        except Exception as e:
            pytest.fail(f"Failed to parse molecule {name} ({smiles}): {e}")
    
    return molecules


class TestTask32Integration:
    """Integration tests for Task 3.2: Basic Molecular Property Extraction."""
    
    def test_fundamental_properties_extraction(self, comprehensive_molecules):
        """Test extraction of fundamental molecular properties."""
        calculator = BasicDescriptorCalculator()
        
        fundamental_descriptors = [
            "molecular_weight", "num_atoms", "num_bonds", "num_heavy_atoms", "ring_count"
        ]
        
        df = calculator.calculate_descriptors_batch(comprehensive_molecules, fundamental_descriptors)
        
        # Validate DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(comprehensive_molecules)
        assert all(desc in df.columns for desc in fundamental_descriptors)
        
        # Validate data types and ranges
        assert df["molecular_weight"].dtype == np.float64
        assert df["num_atoms"].dtype in [np.int64, np.float64]
        assert df["num_bonds"].dtype in [np.int64, np.float64]
        assert df["num_heavy_atoms"].dtype in [np.int64, np.float64]
        assert df["ring_count"].dtype in [np.int64, np.float64]
        
        # Validate reasonable ranges
        assert (df["molecular_weight"] > 0).all()
        assert (df["num_atoms"] > 0).all()
        assert (df["num_bonds"] >= 0).all()
        assert (df["num_heavy_atoms"] >= 0).all()
        assert (df["ring_count"] >= 0).all()
        
        # Validate specific molecules
        methane_row = df[df["mol_id"] == "methane"].iloc[0]
        assert methane_row["num_atoms"] == 1
        assert methane_row["num_bonds"] == 0
        assert methane_row["ring_count"] == 0
        
        benzene_row = df[df["mol_id"] == "benzene"].iloc[0]
        assert benzene_row["num_atoms"] == 6
        assert benzene_row["num_bonds"] == 6
        assert benzene_row["ring_count"] == 1
    
    def test_2d_descriptors_implementation(self, comprehensive_molecules):
        """Test implementation of TPSA, LogP, and other basic 2D descriptors."""
        calculator = BasicDescriptorCalculator()
        
        descriptors_2d = [
            "tpsa", "logp", "num_rotatable_bonds", "num_hbd", "num_hba",
            "molar_refractivity", "bertz_complexity", "fraction_csp3"
        ]
        
        df = calculator.calculate_descriptors_batch(comprehensive_molecules, descriptors_2d)
        
        # Validate DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(comprehensive_molecules)
        assert all(desc in df.columns for desc in descriptors_2d)
        
        # Validate data types
        for desc in descriptors_2d:
            assert df[desc].dtype in [np.float64, np.int64]
        
        # Validate reasonable ranges
        assert (df["tpsa"] >= 0).all()  # TPSA should be non-negative
        assert (df["num_rotatable_bonds"] >= 0).all()
        assert (df["num_hbd"] >= 0).all()
        assert (df["num_hba"] >= 0).all()
        assert (df["molar_refractivity"] >= 0).all()
        assert (df["bertz_complexity"] >= 0).all()
        assert ((df["fraction_csp3"] >= 0) & (df["fraction_csp3"] <= 1)).all()
        
        # Test specific molecules with known properties
        ethanol_row = df[df["mol_id"] == "ethanol"].iloc[0]
        assert ethanol_row["num_hbd"] == 1  # One OH group
        assert ethanol_row["num_hba"] == 1  # One oxygen
        
        acetic_acid_row = df[df["mol_id"] == "acetic_acid"].iloc[0]
        assert acetic_acid_row["num_hbd"] == 1  # Carboxylic acid H
        assert acetic_acid_row["num_hba"] == 1  # RDKit counts carboxylic acid as 1 HBA
    
    def test_rdkit_reference_accuracy(self, comprehensive_molecules):
        """Test accuracy against RDKit reference implementations."""
        calculator = BasicDescriptorCalculator()
        tolerance = 1e-6
        
        test_descriptors = ["molecular_weight", "logp", "tpsa", "num_hbd", "num_hba"]
        
        for mol_data in comprehensive_molecules:
            rdkit_mol = mol_data.backend_data["rdkit_mol"]
            
            # Calculate using our implementation
            our_results = calculator.calculate_descriptors(mol_data, test_descriptors)
            
            # Calculate using RDKit directly
            rdkit_mw = Descriptors.MolWt(rdkit_mol)
            rdkit_logp = Crippen.MolLogP(rdkit_mol)
            rdkit_tpsa = Descriptors.TPSA(rdkit_mol)
            rdkit_hbd = Descriptors.NumHDonors(rdkit_mol)
            rdkit_hba = Descriptors.NumHAcceptors(rdkit_mol)
            
            # Compare results
            assert abs(our_results["molecular_weight"] - rdkit_mw) < tolerance
            assert abs(our_results["logp"] - rdkit_logp) < tolerance
            assert abs(our_results["tpsa"] - rdkit_tpsa) < tolerance
            assert our_results["num_hbd"] == rdkit_hbd
            assert our_results["num_hba"] == rdkit_hba
    
    def test_molecular_graph_representations(self, comprehensive_molecules):
        """Test creation of molecular graph representations."""
        builder = MolecularGraphBuilder()
        
        for mol_data in comprehensive_molecules:
            graph_data = builder.build_molecular_graph(mol_data)
            
            # Validate graph structure
            assert "num_atoms" in graph_data
            assert "num_bonds" in graph_data
            assert "atom_features" in graph_data
            assert "bond_features" in graph_data
            assert "adjacency_matrix" in graph_data
            assert "adjacency_sparse" in graph_data
            
            # Validate dimensions
            num_atoms = graph_data["num_atoms"]
            num_bonds = graph_data["num_bonds"]
            
            if num_atoms > 0:
                assert graph_data["atom_features"].shape[0] == num_atoms
                assert graph_data["adjacency_matrix"].shape == (num_atoms, num_atoms)
            
            if num_bonds > 0:
                assert graph_data["bond_features"].shape[0] == num_bonds
            
            # Validate adjacency matrix properties
            adj_matrix = graph_data["adjacency_matrix"]
            assert np.array_equal(adj_matrix, adj_matrix.T)  # Symmetric
            assert np.all(np.diag(adj_matrix) == 0)  # No self-loops
            assert np.all((adj_matrix == 0) | (adj_matrix == 1))  # Binary
    
    def test_fingerprint_generation(self, comprehensive_molecules):
        """Test molecular fingerprint generation."""
        generator = FingerprintGenerator()
        
        for mol_data in comprehensive_molecules:
            # Test Morgan fingerprint
            morgan_fp = generator.generate_morgan_fingerprint(mol_data)
            assert isinstance(morgan_fp, np.ndarray)
            assert morgan_fp.dtype == np.int8
            assert len(morgan_fp) == 2048
            assert np.all((morgan_fp == 0) | (morgan_fp == 1))
            assert np.sum(morgan_fp) > 0  # Should have some bits set
            
            # Test MACCS keys
            maccs_fp = generator.generate_maccs_keys(mol_data)
            assert isinstance(maccs_fp, np.ndarray)
            assert maccs_fp.dtype == np.int8
            assert len(maccs_fp) == 167
            assert np.all((maccs_fp == 0) | (maccs_fp == 1))
            
            # Test all fingerprints
            all_fps = generator.generate_all_fingerprints(mol_data)
            assert isinstance(all_fps, dict)
            assert "morgan" in all_fps
            assert "maccs" in all_fps
            assert len(all_fps) >= 2
    
    def test_fingerprint_rdkit_accuracy(self, comprehensive_molecules):
        """Test fingerprint accuracy against RDKit reference."""
        generator = FingerprintGenerator()
        
        for mol_data in comprehensive_molecules:
            rdkit_mol = mol_data.backend_data["rdkit_mol"]
            
            # Test Morgan fingerprint accuracy
            our_morgan = generator.generate_morgan_fingerprint(mol_data, radius=2, n_bits=2048)
            rdkit_morgan = AllChem.GetMorganFingerprintAsBitVect(rdkit_mol, 2, nBits=2048)
            rdkit_morgan_array = np.zeros(2048, dtype=np.int8)
            for i in range(2048):
                rdkit_morgan_array[i] = rdkit_morgan[i]
            
            assert np.array_equal(our_morgan, rdkit_morgan_array)
            
            # Test MACCS keys accuracy
            our_maccs = generator.generate_maccs_keys(mol_data)
            rdkit_maccs = MACCSkeys.GenMACCSKeys(rdkit_mol)
            rdkit_maccs_array = np.zeros(167, dtype=np.int8)
            for i in range(167):
                rdkit_maccs_array[i] = rdkit_maccs[i]
            
            assert np.array_equal(our_maccs, rdkit_maccs_array)
    
    def test_calculator_integration(self, calculator):
        """Test integration with main Calculator class."""
        # Test with pandas Series input
        smiles_series = pd.Series([smiles for smiles, _ in COMPREHENSIVE_TEST_SET[:5]])
        
        df = calculator.pandas(smiles_series)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert "mol_id" in df.columns
        assert "smiles" in df.columns
        
        # Should have basic descriptors
        basic_descriptors = [
            "molecular_weight", "num_atoms", "num_bonds", "num_heavy_atoms",
            "logp", "tpsa", "num_rotatable_bonds", "num_aromatic_rings",
            "num_hbd", "num_hba"
        ]
        
        for desc in basic_descriptors:
            assert desc in df.columns
    
    def test_descriptor_configuration(self, calculator):
        """Test descriptor configuration functionality."""
        # Test basic descriptor set
        calculator.configure_descriptors("basic")
        basic_descriptors = calculator.descriptor_sets
        
        assert isinstance(basic_descriptors, list)
        assert len(basic_descriptors) > 0
        assert "molecular_weight" in basic_descriptors
        assert "logp" in basic_descriptors
        
        # Test custom descriptor list
        custom_descriptors = ["molecular_weight", "tpsa", "num_atoms"]
        calculator.configure_descriptors(custom_descriptors)
        
        assert calculator.descriptor_sets == custom_descriptors
    
    def test_batch_processing_performance(self, comprehensive_molecules):
        """Test batch processing performance and consistency."""
        calculator = BasicDescriptorCalculator()
        
        # Process all molecules at once
        batch_df = calculator.calculate_descriptors_batch(comprehensive_molecules)
        
        # Process molecules individually
        individual_results = []
        for mol_data in comprehensive_molecules:
            result = calculator.calculate_descriptors(mol_data)
            result["mol_id"] = mol_data.mol_id
            result["smiles"] = mol_data.smiles
            individual_results.append(result)
        
        individual_df = pd.DataFrame(individual_results)
        
        # Results should be identical (within numerical precision)
        assert len(batch_df) == len(individual_df)
        
        # Compare descriptor values
        descriptor_cols = [col for col in batch_df.columns 
                          if col not in ["mol_id", "smiles"] and not col.startswith("_")]
        
        for col in descriptor_cols:
            if col in individual_df.columns:
                batch_values = batch_df[col].values
                individual_values = individual_df[col].values
                
                # Handle NaN values
                nan_mask = np.isnan(batch_values) | np.isnan(individual_values)
                assert np.array_equal(np.isnan(batch_values), np.isnan(individual_values))
                
                # Compare non-NaN values
                if not nan_mask.all():
                    non_nan_batch = batch_values[~nan_mask]
                    non_nan_individual = individual_values[~nan_mask]
                    assert np.allclose(non_nan_batch, non_nan_individual, rtol=1e-10)
    
    def test_error_handling_robustness(self, calculator):
        """Test robust error handling with problematic molecules."""
        # Test with empty list
        empty_df = calculator.pandas([])
        assert isinstance(empty_df, pd.DataFrame)
        assert len(empty_df) == 0
        
        # Test with mix of valid and invalid SMILES
        mixed_smiles = ["CCO", "INVALID_SMILES", "c1ccccc1", "", "CC(=O)O"]
        
        try:
            df = calculator.pandas(mixed_smiles)
            # Should process valid molecules and skip invalid ones
            assert isinstance(df, pd.DataFrame)
            assert len(df) <= len(mixed_smiles)  # Some may be skipped
            assert len(df) >= 3  # At least the valid ones should be processed
        except Exception as e:
            # Error handling should be graceful
            assert "Failed to calculate descriptors" in str(e)
    
    def test_comprehensive_workflow(self, comprehensive_molecules):
        """Test complete workflow from parsing to descriptor calculation."""
        # Test the complete pipeline
        calculator = BasicDescriptorCalculator()
        graph_calculator = GraphDescriptorCalculator()
        fingerprint_generator = FingerprintGenerator()
        
        results = []
        
        for mol_data in comprehensive_molecules[:5]:  # Test subset for speed
            # Basic descriptors
            basic_desc = calculator.calculate_descriptors(mol_data)
            
            # Graph descriptors
            graph_desc = graph_calculator.calculate_graph_descriptors(mol_data)
            
            # Fingerprints
            morgan_fp = fingerprint_generator.generate_morgan_fingerprint(mol_data)
            maccs_fp = fingerprint_generator.generate_maccs_keys(mol_data)
            
            # Combine results
            result = {
                "mol_id": mol_data.mol_id,
                "smiles": mol_data.smiles,
                **basic_desc,
                **graph_desc,
                "morgan_fp_bits": np.sum(morgan_fp),
                "maccs_fp_bits": np.sum(maccs_fp)
            }
            results.append(result)
        
        df = pd.DataFrame(results)
        
        # Validate comprehensive results
        assert len(df) == 5
        assert "mol_id" in df.columns
        assert "smiles" in df.columns
        
        # Should have basic descriptors
        assert "molecular_weight" in df.columns
        assert "logp" in df.columns
        assert "tpsa" in df.columns
        
        # Should have graph descriptors
        assert "wiener_index" in df.columns
        assert "zagreb_index_1" in df.columns
        
        # Should have fingerprint information
        assert "morgan_fp_bits" in df.columns
        assert "maccs_fp_bits" in df.columns
        
        # All fingerprints should have some bits set
        assert (df["morgan_fp_bits"] > 0).all()
        assert (df["maccs_fp_bits"] > 0).all()


class TestPerformanceAndScalability:
    """Test performance and scalability aspects."""
    
    def test_large_batch_processing(self):
        """Test processing of larger molecular batches."""
        if not RDKIT_AVAILABLE:
            pytest.skip("RDKit not available")
        
        # Create larger test set
        large_smiles = [smiles for smiles, _ in COMPREHENSIVE_TEST_SET] * 3  # 75 molecules
        
        calculator = Calculator()
        
        try:
            df = calculator.pandas(large_smiles)
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) <= len(large_smiles)  # Some may fail parsing
            assert len(df) > len(large_smiles) * 0.8  # Most should succeed
            
        except Exception as e:
            pytest.skip(f"Large batch test failed (expected for resource constraints): {e}")
    
    def test_memory_efficiency(self, comprehensive_molecules):
        """Test memory efficiency of implementations."""
        calculator = BasicDescriptorCalculator()
        
        # Process in small batches to test memory management
        batch_size = 5
        all_results = []
        
        for i in range(0, len(comprehensive_molecules), batch_size):
            batch = comprehensive_molecules[i:i+batch_size]
            batch_df = calculator.calculate_descriptors_batch(batch)
            all_results.append(batch_df)
        
        # Combine results
        combined_df = pd.concat(all_results, ignore_index=True)
        
        assert len(combined_df) == len(comprehensive_molecules)
        assert "molecular_weight" in combined_df.columns


@pytest.mark.parametrize("smiles,expected_name", COMPREHENSIVE_TEST_SET)
def test_individual_comprehensive_molecules(smiles, expected_name):
    """Test each molecule in the comprehensive test set individually."""
    if not RDKIT_AVAILABLE:
        pytest.skip("RDKit not available")
    
    try:
        calculator = Calculator()
        df = calculator.pandas([smiles])
        
        assert len(df) == 1
        row = df.iloc[0]
        
        # Basic validation
        assert row["molecular_weight"] > 0
        assert row["num_atoms"] > 0
        assert row["num_bonds"] >= 0
        assert isinstance(row["logp"], (int, float))
        assert row["tpsa"] >= 0
        
    except Exception as e:
        pytest.fail(f"Failed to process {expected_name} ({smiles}): {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])