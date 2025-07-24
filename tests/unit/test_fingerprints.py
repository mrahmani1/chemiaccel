"""
Tests for molecular fingerprint generation.

This module tests the FingerprintGenerator and related functionality
to ensure correct fingerprint generation and similarity calculations.
"""

import pytest
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, MACCSkeys
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from chemi_accel.descriptors.fingerprints import (
    FingerprintGenerator,
    FingerprintSimilarity,
    generate_morgan_fingerprints,
    generate_maccs_fingerprints,
    generate_topological_fingerprints,
    generate_atom_pair_fingerprints,
    generate_topological_torsion_fingerprints,
    calculate_fingerprint_similarities
)
from chemi_accel.utils.rdkit_integration import RDKitMolecularParser
from chemi_accel.core.exceptions import DescriptorCalculationError


# Test molecules for fingerprint analysis
FINGERPRINT_TEST_MOLECULES = [
    ("CCO", "ethanol"),
    ("CC(=O)O", "acetic_acid"),
    ("c1ccccc1", "benzene"),
    ("CCc1ccccc1", "ethylbenzene"),
    ("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "ibuprofen"),
    ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "caffeine"),
    ("c1ccc2ccccc2c1", "naphthalene"),
    ("CC1=CC=CC=C1", "toluene"),
]


@pytest.fixture
def fingerprint_generator():
    """Create FingerprintGenerator instance."""
    if not RDKIT_AVAILABLE:
        pytest.skip("RDKit not available")
    return FingerprintGenerator()


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
    for smiles, name in FINGERPRINT_TEST_MOLECULES:
        try:
            mol_data = parser.parse_smiles(smiles, name)
            molecules.append(mol_data)
        except Exception as e:
            pytest.fail(f"Failed to parse test molecule {name} ({smiles}): {e}")
    return molecules


class TestFingerprintGenerator:
    """Test FingerprintGenerator functionality."""
    
    def test_initialization(self, fingerprint_generator):
        """Test fingerprint generator initialization."""
        assert fingerprint_generator is not None
    
    def test_morgan_fingerprint_generation(self, fingerprint_generator, test_molecules):
        """Test Morgan fingerprint generation."""
        mol_data = test_molecules[0]  # ethanol
        
        # Test default parameters
        fp = fingerprint_generator.generate_morgan_fingerprint(mol_data)
        
        assert isinstance(fp, np.ndarray)
        assert fp.dtype == np.int8
        assert len(fp) == 2048  # Default size
        assert np.all((fp == 0) | (fp == 1))  # Binary fingerprint
        
        # Test custom parameters
        fp_custom = fingerprint_generator.generate_morgan_fingerprint(
            mol_data, radius=3, n_bits=1024, use_chirality=True
        )
        
        assert len(fp_custom) == 1024
        assert isinstance(fp_custom, np.ndarray)
    
    def test_morgan_fingerprint_accuracy(self, fingerprint_generator, test_molecules):
        """Test Morgan fingerprint accuracy against RDKit reference."""
        for mol_data in test_molecules:
            rdkit_mol = mol_data.backend_data["rdkit_mol"]
            
            # Generate fingerprint using our implementation
            our_fp = fingerprint_generator.generate_morgan_fingerprint(mol_data, radius=2, n_bits=2048)
            
            # Generate reference fingerprint using RDKit directly
            rdkit_fp = AllChem.GetMorganFingerprintAsBitVect(rdkit_mol, 2, nBits=2048)
            rdkit_array = np.zeros((2048,), dtype=np.int8)
            for i in range(2048):
                rdkit_array[i] = rdkit_fp[i]
            
            # Should be identical
            assert np.array_equal(our_fp, rdkit_array), f"Morgan fingerprint mismatch for {mol_data.mol_id}"
    
    def test_morgan_counts(self, fingerprint_generator, test_molecules):
        """Test Morgan fingerprint with counts."""
        mol_data = test_molecules[0]  # ethanol
        
        counts = fingerprint_generator.generate_morgan_counts(mol_data)
        
        assert isinstance(counts, dict)
        assert len(counts) > 0
        
        # All values should be positive integers
        for hash_val, count in counts.items():
            assert isinstance(hash_val, int)
            assert isinstance(count, int)
            assert count > 0
    
    def test_maccs_keys_generation(self, fingerprint_generator, test_molecules):
        """Test MACCS keys generation."""
        mol_data = test_molecules[0]  # ethanol
        
        maccs = fingerprint_generator.generate_maccs_keys(mol_data)
        
        assert isinstance(maccs, np.ndarray)
        assert maccs.dtype == np.int8
        assert len(maccs) == 167  # MACCS keys are 167 bits
        assert np.all((maccs == 0) | (maccs == 1))  # Binary fingerprint
    
    def test_maccs_keys_accuracy(self, fingerprint_generator, test_molecules):
        """Test MACCS keys accuracy against RDKit reference."""
        for mol_data in test_molecules:
            rdkit_mol = mol_data.backend_data["rdkit_mol"]
            
            # Generate MACCS keys using our implementation
            our_maccs = fingerprint_generator.generate_maccs_keys(mol_data)
            
            # Generate reference MACCS keys using RDKit directly
            rdkit_maccs = MACCSkeys.GenMACCSKeys(rdkit_mol)
            rdkit_array = np.zeros((167,), dtype=np.int8)
            for i in range(167):
                rdkit_array[i] = rdkit_maccs[i]
            
            # Should be identical
            assert np.array_equal(our_maccs, rdkit_array), f"MACCS keys mismatch for {mol_data.mol_id}"
    
    def test_topological_fingerprint(self, fingerprint_generator, test_molecules):
        """Test topological fingerprint generation."""
        mol_data = test_molecules[0]  # ethanol
        
        fp = fingerprint_generator.generate_topological_fingerprint(mol_data)
        
        assert isinstance(fp, np.ndarray)
        assert fp.dtype == np.int8
        assert len(fp) == 2048  # Default size
        assert np.all((fp == 0) | (fp == 1))  # Binary fingerprint
        
        # Test custom size
        fp_custom = fingerprint_generator.generate_topological_fingerprint(mol_data, n_bits=1024)
        assert len(fp_custom) == 1024
    
    def test_atom_pair_fingerprint(self, fingerprint_generator, test_molecules):
        """Test atom pair fingerprint generation."""
        mol_data = test_molecules[0]  # ethanol
        
        fp = fingerprint_generator.generate_atom_pair_fingerprint(mol_data)
        
        assert isinstance(fp, np.ndarray)
        assert fp.dtype == np.int8
        assert len(fp) == 2048  # Default size
        assert np.all((fp == 0) | (fp == 1))  # Binary fingerprint
    
    def test_topological_torsion_fingerprint(self, fingerprint_generator, test_molecules):
        """Test topological torsion fingerprint generation."""
        mol_data = test_molecules[0]  # ethanol
        
        fp = fingerprint_generator.generate_topological_torsion_fingerprint(mol_data)
        
        assert isinstance(fp, np.ndarray)
        assert fp.dtype == np.int8
        assert len(fp) == 2048  # Default size
        assert np.all((fp == 0) | (fp == 1))  # Binary fingerprint
    
    def test_avalon_fingerprint(self, fingerprint_generator, test_molecules):
        """Test Avalon fingerprint generation (if available)."""
        mol_data = test_molecules[0]  # ethanol
        
        fp = fingerprint_generator.generate_avalon_fingerprint(mol_data)
        
        # Avalon might not be available
        if fp is not None:
            assert isinstance(fp, np.ndarray)
            assert fp.dtype == np.int8
            assert len(fp) == 512  # Default size
            assert np.all((fp == 0) | (fp == 1))  # Binary fingerprint
    
    def test_pharmacophore_fingerprint(self, fingerprint_generator, test_molecules):
        """Test pharmacophore fingerprint generation (if available)."""
        mol_data = test_molecules[0]  # ethanol
        
        fp = fingerprint_generator.generate_pharmacophore_fingerprint(mol_data)
        
        # Pharmacophore might not be available
        if fp is not None:
            assert isinstance(fp, np.ndarray)
            assert fp.dtype == np.int8
            assert np.all((fp == 0) | (fp == 1))  # Binary fingerprint
    
    def test_all_fingerprints_generation(self, fingerprint_generator, test_molecules):
        """Test generation of all fingerprint types."""
        mol_data = test_molecules[0]  # ethanol
        
        all_fps = fingerprint_generator.generate_all_fingerprints(mol_data)
        
        assert isinstance(all_fps, dict)
        assert len(all_fps) > 0
        
        # Should at least have Morgan and MACCS
        assert "morgan" in all_fps
        assert "maccs" in all_fps
        
        # Should also have additional fingerprint types
        expected_fps = ["morgan", "maccs", "topological", "atom_pair", "torsion"]
        for fp_name in expected_fps:
            if fp_name in all_fps:
                assert isinstance(all_fps[fp_name], np.ndarray)
                assert all_fps[fp_name].dtype == np.int8
                assert np.all((all_fps[fp_name] == 0) | (all_fps[fp_name] == 1))
        
        # Check fingerprint properties
        for fp_name, fp_array in all_fps.items():
            assert isinstance(fp_array, np.ndarray)
            assert fp_array.dtype == np.int8
            assert np.all((fp_array == 0) | (fp_array == 1))
    
    def test_error_handling(self, fingerprint_generator):
        """Test error handling for invalid inputs."""
        from chemi_accel.core.molecular import MolecularData
        
        # Test with molecule without RDKit mol object
        invalid_mol = MolecularData(smiles="CCO", mol_id="test")
        
        with pytest.raises(DescriptorCalculationError):
            fingerprint_generator.generate_morgan_fingerprint(invalid_mol)
        
        with pytest.raises(DescriptorCalculationError):
            fingerprint_generator.generate_maccs_keys(invalid_mol)


class TestFingerprintSimilarity:
    """Test fingerprint similarity calculations."""
    
    def test_tanimoto_similarity(self):
        """Test Tanimoto similarity calculation."""
        # Test identical fingerprints
        fp1 = np.array([1, 0, 1, 0, 1], dtype=np.int8)
        fp2 = np.array([1, 0, 1, 0, 1], dtype=np.int8)
        
        sim = FingerprintSimilarity.tanimoto_similarity(fp1, fp2)
        assert sim == 1.0
        
        # Test completely different fingerprints
        fp3 = np.array([0, 1, 0, 1, 0], dtype=np.int8)
        sim = FingerprintSimilarity.tanimoto_similarity(fp1, fp3)
        assert sim == 0.0
        
        # Test partial overlap
        fp4 = np.array([1, 0, 0, 0, 1], dtype=np.int8)
        sim = FingerprintSimilarity.tanimoto_similarity(fp1, fp4)
        expected = 2.0 / 3.0  # 2 bits in common, 3 bits in union
        assert abs(sim - expected) < 1e-10
    
    def test_dice_similarity(self):
        """Test Dice similarity calculation."""
        fp1 = np.array([1, 0, 1, 0, 1], dtype=np.int8)
        fp2 = np.array([1, 0, 1, 0, 1], dtype=np.int8)
        
        sim = FingerprintSimilarity.dice_similarity(fp1, fp2)
        assert sim == 1.0
        
        # Test partial overlap
        fp3 = np.array([1, 0, 0, 0, 1], dtype=np.int8)
        sim = FingerprintSimilarity.dice_similarity(fp1, fp3)
        expected = (2.0 * 2.0) / (3.0 + 2.0)  # 2*intersection / (sum1 + sum2)
        assert abs(sim - expected) < 1e-10
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        fp1 = np.array([1, 0, 1, 0, 1], dtype=np.int8)
        fp2 = np.array([1, 0, 1, 0, 1], dtype=np.int8)
        
        sim = FingerprintSimilarity.cosine_similarity(fp1, fp2)
        assert abs(sim - 1.0) < 1e-10
        
        # Test orthogonal fingerprints
        fp3 = np.array([0, 1, 0, 1, 0], dtype=np.int8)
        sim = FingerprintSimilarity.cosine_similarity(fp1, fp3)
        assert sim == 0.0
    
    def test_hamming_distance(self):
        """Test Hamming distance calculation."""
        fp1 = np.array([1, 0, 1, 0, 1], dtype=np.int8)
        fp2 = np.array([1, 0, 1, 0, 1], dtype=np.int8)
        
        dist = FingerprintSimilarity.hamming_distance(fp1, fp2)
        assert dist == 0
        
        # Test completely different fingerprints
        fp3 = np.array([0, 1, 0, 1, 0], dtype=np.int8)
        dist = FingerprintSimilarity.hamming_distance(fp1, fp3)
        assert dist == 5  # All bits are different
    
    def test_similarity_matrix(self):
        """Test similarity matrix calculation."""
        fp1 = np.array([1, 0, 1, 0, 1], dtype=np.int8)
        fp2 = np.array([1, 0, 1, 0, 1], dtype=np.int8)
        fp3 = np.array([0, 1, 0, 1, 0], dtype=np.int8)
        
        fingerprints = [fp1, fp2, fp3]
        
        sim_matrix = FingerprintSimilarity.calculate_similarity_matrix(fingerprints, "tanimoto")
        
        assert sim_matrix.shape == (3, 3)
        
        # Diagonal should be 1.0
        assert np.all(np.diag(sim_matrix) == 1.0)
        
        # Should be symmetric
        assert np.allclose(sim_matrix, sim_matrix.T)
        
        # fp1 and fp2 are identical
        assert sim_matrix[0, 1] == 1.0
        assert sim_matrix[1, 0] == 1.0
        
        # fp1 and fp3 are completely different
        assert sim_matrix[0, 2] == 0.0
        assert sim_matrix[2, 0] == 0.0
    
    def test_similarity_edge_cases(self):
        """Test similarity calculations with edge cases."""
        # Empty fingerprints
        fp_empty1 = np.array([0, 0, 0, 0, 0], dtype=np.int8)
        fp_empty2 = np.array([0, 0, 0, 0, 0], dtype=np.int8)
        
        sim = FingerprintSimilarity.tanimoto_similarity(fp_empty1, fp_empty2)
        assert sim == 1.0  # Both empty should be similar
        
        # One empty, one non-empty
        fp_nonempty = np.array([1, 0, 1, 0, 1], dtype=np.int8)
        sim = FingerprintSimilarity.tanimoto_similarity(fp_empty1, fp_nonempty)
        assert sim == 0.0
    
    def test_different_length_fingerprints(self):
        """Test error handling for different length fingerprints."""
        fp1 = np.array([1, 0, 1], dtype=np.int8)
        fp2 = np.array([1, 0, 1, 0, 1], dtype=np.int8)
        
        with pytest.raises(ValueError):
            FingerprintSimilarity.tanimoto_similarity(fp1, fp2)
        
        with pytest.raises(ValueError):
            FingerprintSimilarity.dice_similarity(fp1, fp2)
        
        with pytest.raises(ValueError):
            FingerprintSimilarity.cosine_similarity(fp1, fp2)
        
        with pytest.raises(ValueError):
            FingerprintSimilarity.hamming_distance(fp1, fp2)


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_generate_morgan_fingerprints(self, test_molecules):
        """Test generate_morgan_fingerprints function."""
        fingerprints = generate_morgan_fingerprints(test_molecules)
        
        assert len(fingerprints) == len(test_molecules)
        
        for fp in fingerprints:
            assert isinstance(fp, np.ndarray)
            assert fp.dtype == np.int8
            assert len(fp) == 2048  # Default size
            assert np.all((fp == 0) | (fp == 1))
    
    def test_generate_maccs_fingerprints(self, test_molecules):
        """Test generate_maccs_fingerprints function."""
        fingerprints = generate_maccs_fingerprints(test_molecules)
        
        assert len(fingerprints) == len(test_molecules)
        
        for fp in fingerprints:
            assert isinstance(fp, np.ndarray)
            assert fp.dtype == np.int8
            assert len(fp) == 167  # MACCS keys size
            assert np.all((fp == 0) | (fp == 1))
    
    def test_generate_topological_fingerprints(self, test_molecules):
        """Test generate_topological_fingerprints function."""
        fingerprints = generate_topological_fingerprints(test_molecules, use_gpu=False)
        
        assert len(fingerprints) == len(test_molecules)
        
        for fp in fingerprints:
            assert isinstance(fp, np.ndarray)
            assert fp.dtype == np.int8
            assert len(fp) == 2048  # Default size
            assert np.all((fp == 0) | (fp == 1))
    
    def test_generate_atom_pair_fingerprints(self, test_molecules):
        """Test generate_atom_pair_fingerprints function."""
        fingerprints = generate_atom_pair_fingerprints(test_molecules, use_gpu=False)
        
        assert len(fingerprints) == len(test_molecules)
        
        for fp in fingerprints:
            assert isinstance(fp, np.ndarray)
            assert fp.dtype == np.int8
            assert len(fp) == 2048  # Default size
            assert np.all((fp == 0) | (fp == 1))
    
    def test_generate_topological_torsion_fingerprints(self, test_molecules):
        """Test generate_topological_torsion_fingerprints function."""
        fingerprints = generate_topological_torsion_fingerprints(test_molecules, use_gpu=False)
        
        assert len(fingerprints) == len(test_molecules)
        
        for fp in fingerprints:
            assert isinstance(fp, np.ndarray)
            assert fp.dtype == np.int8
            assert len(fp) == 2048  # Default size
            assert np.all((fp == 0) | (fp == 1))
    
    def test_calculate_fingerprint_similarities(self, test_molecules):
        """Test calculate_fingerprint_similarities function."""
        fingerprints = generate_morgan_fingerprints(test_molecules, use_gpu=False)
        
        sim_matrix = calculate_fingerprint_similarities(fingerprints, "tanimoto")
        
        assert sim_matrix.shape == (len(test_molecules), len(test_molecules))
        assert np.all(np.diag(sim_matrix) == 1.0)  # Self-similarity
        assert np.allclose(sim_matrix, sim_matrix.T)  # Symmetric


class TestFingerprintProperties:
    """Test specific fingerprint properties and relationships."""
    
    def test_similar_molecules_high_similarity(self, fingerprint_generator, parser):
        """Test that similar molecules have high fingerprint similarity."""
        # Benzene and toluene should be similar
        benzene = parser.parse_smiles("c1ccccc1", "benzene")
        toluene = parser.parse_smiles("Cc1ccccc1", "toluene")
        
        fp_benzene = fingerprint_generator.generate_morgan_fingerprint(benzene)
        fp_toluene = fingerprint_generator.generate_morgan_fingerprint(toluene)
        
        similarity = FingerprintSimilarity.tanimoto_similarity(fp_benzene, fp_toluene)
        
        # Should be reasonably similar (> 0.2 for benzene/toluene)
        assert similarity > 0.2
    
    def test_different_molecules_low_similarity(self, fingerprint_generator, parser):
        """Test that very different molecules have low fingerprint similarity."""
        # Ethanol and caffeine should be quite different
        ethanol = parser.parse_smiles("CCO", "ethanol")
        caffeine = parser.parse_smiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "caffeine")
        
        fp_ethanol = fingerprint_generator.generate_morgan_fingerprint(ethanol)
        fp_caffeine = fingerprint_generator.generate_morgan_fingerprint(caffeine)
        
        similarity = FingerprintSimilarity.tanimoto_similarity(fp_ethanol, fp_caffeine)
        
        # Should be quite different (< 0.3)
        assert similarity < 0.3
    
    def test_fingerprint_consistency(self, fingerprint_generator, parser):
        """Test that fingerprints are consistent across multiple generations."""
        mol_data = parser.parse_smiles("CCO", "ethanol")
        
        # Generate same fingerprint multiple times
        fp1 = fingerprint_generator.generate_morgan_fingerprint(mol_data)
        fp2 = fingerprint_generator.generate_morgan_fingerprint(mol_data)
        fp3 = fingerprint_generator.generate_morgan_fingerprint(mol_data)
        
        # Should be identical
        assert np.array_equal(fp1, fp2)
        assert np.array_equal(fp2, fp3)
    
    def test_fingerprint_radius_effect(self, fingerprint_generator, parser):
        """Test effect of different radii on Morgan fingerprints."""
        mol_data = parser.parse_smiles("CCc1ccccc1", "ethylbenzene")
        
        fp_r1 = fingerprint_generator.generate_morgan_fingerprint(mol_data, radius=1)
        fp_r2 = fingerprint_generator.generate_morgan_fingerprint(mol_data, radius=2)
        fp_r3 = fingerprint_generator.generate_morgan_fingerprint(mol_data, radius=3)
        
        # Different radii should generally give different fingerprints
        # (though not guaranteed for all molecules)
        assert not np.array_equal(fp_r1, fp_r2) or not np.array_equal(fp_r2, fp_r3)
    
    def test_fingerprint_size_effect(self, fingerprint_generator, parser):
        """Test effect of different fingerprint sizes."""
        mol_data = parser.parse_smiles("CCc1ccccc1", "ethylbenzene")
        
        fp_1024 = fingerprint_generator.generate_morgan_fingerprint(mol_data, n_bits=1024)
        fp_2048 = fingerprint_generator.generate_morgan_fingerprint(mol_data, n_bits=2048)
        fp_4096 = fingerprint_generator.generate_morgan_fingerprint(mol_data, n_bits=4096)
        
        assert len(fp_1024) == 1024
        assert len(fp_2048) == 2048
        assert len(fp_4096) == 4096
        
        # All should be binary
        assert np.all((fp_1024 == 0) | (fp_1024 == 1))
        assert np.all((fp_2048 == 0) | (fp_2048 == 1))
        assert np.all((fp_4096 == 0) | (fp_4096 == 1))


@pytest.mark.parametrize("smiles,expected_name", FINGERPRINT_TEST_MOLECULES)
def test_individual_fingerprint_molecules(smiles, expected_name):
    """Test individual molecules parametrically for fingerprint generation."""
    if not RDKIT_AVAILABLE:
        pytest.skip("RDKit not available")
    
    parser = RDKitMolecularParser()
    generator = FingerprintGenerator()
    
    try:
        mol_data = parser.parse_smiles(smiles, expected_name)
        
        # Test Morgan fingerprint
        morgan_fp = generator.generate_morgan_fingerprint(mol_data)
        assert isinstance(morgan_fp, np.ndarray)
        assert len(morgan_fp) == 2048
        assert np.all((morgan_fp == 0) | (morgan_fp == 1))
        
        # Test MACCS keys
        maccs_fp = generator.generate_maccs_keys(mol_data)
        assert isinstance(maccs_fp, np.ndarray)
        assert len(maccs_fp) == 167
        assert np.all((maccs_fp == 0) | (maccs_fp == 1))
        
        # Test that fingerprints have some bits set (not all zeros)
        assert np.sum(morgan_fp) > 0, f"Morgan fingerprint all zeros for {expected_name}"
        assert np.sum(maccs_fp) > 0, f"MACCS keys all zeros for {expected_name}"
        
    except Exception as e:
        pytest.fail(f"Failed to process molecule {expected_name} ({smiles}): {e}")


if __name__ == "__main__":
    pytest.main([__file__])