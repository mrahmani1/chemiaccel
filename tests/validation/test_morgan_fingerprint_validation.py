"""
Validation tests for Morgan fingerprint implementation against RDKit reference.

This module provides comprehensive validation of the Morgan fingerprint implementation
by comparing results against RDKit's reference implementation across diverse molecular
structures and parameter combinations.
"""

import pytest
import numpy as np
from typing import List, Dict, Tuple
import time

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from chemi_accel.descriptors.fingerprints import FingerprintGenerator
from chemi_accel.core.molecular import MolecularData


# Comprehensive test dataset with diverse molecular structures
VALIDATION_SMILES = [
    # Simple molecules
    "C",  # Methane
    "CC",  # Ethane
    "CCO",  # Ethanol
    "CC(=O)O",  # Acetic acid
    
    # Aromatic compounds
    "c1ccccc1",  # Benzene
    "Cc1ccccc1",  # Toluene
    "CCc1ccccc1",  # Ethylbenzene
    "c1ccc2ccccc2c1",  # Naphthalene
    
    # Heterocycles
    "c1ccncc1",  # Pyridine
    "c1cccnc1",  # Pyrimidine
    "c1cnc2ccccc2n1",  # Quinazoline
    "c1coc2ccccc12",  # Benzofuran
    
    # Complex structures
    "CC(C)C(=O)Nc1ccc(cc1)C(=O)O",  # Ibuprofen-like
    "CC(C)(C)c1ccc(cc1)O",  # 4-tert-Butylphenol
    "CCN(CC)CC(=O)Nc1c(cccc1C)C",  # Lidocaine-like
    
    # Challenging cases
    "C1CCC2CCCCC2C1",  # Decalin
    "C1=CC=C2C(=C1)C=CC=C2",  # Naphthalene (alternative)
    "CC(C)(C)C1=CC=C(C=C1)C(C)(C)C",  # Di-tert-butylbenzene
    
    # Stereochemistry cases
    "C[C@H](O)C(=O)O",  # L-Lactic acid
    "C[C@@H](O)C(=O)O",  # D-Lactic acid
    
    # Functional groups
    "CC(=O)N",  # Acetamide
    "CC(=O)OC",  # Methyl acetate
    "CCS",  # Ethanethiol
    "CCN",  # Ethylamine
    "CC#N",  # Acetonitrile
]


@pytest.fixture
def validation_molecular_data():
    """Create molecular data objects for validation."""
    if not RDKIT_AVAILABLE:
        pytest.skip("RDKit not available")
    
    mol_data_list = []
    for i, smiles in enumerate(VALIDATION_SMILES):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mol_data = MolecularData(
                mol_id=f"val_mol_{i}",
                smiles=smiles,
                backend_data={"rdkit_mol": mol}
            )
            mol_data_list.append(mol_data)
    
    return mol_data_list


class TestMorganFingerprintValidation:
    """Comprehensive validation against RDKit reference implementation."""
    
    @pytest.mark.parametrize("radius", [0, 1, 2, 3, 4])
    @pytest.mark.parametrize("n_bits", [512, 1024, 2048, 4096])
    def test_morgan_fingerprint_exact_match(self, validation_molecular_data, radius, n_bits):
        """Test exact match with RDKit Morgan fingerprints across parameters."""
        generator = FingerprintGenerator(use_gpu=False)
        
        mismatches = []
        
        for mol_data in validation_molecular_data:
            rdkit_mol = mol_data.backend_data["rdkit_mol"]
            
            # Generate using our implementation
            our_fp = generator.generate_morgan_fingerprint(
                mol_data, radius=radius, n_bits=n_bits, force_cpu=True
            )
            
            # Generate using RDKit reference
            rdkit_fp = AllChem.GetMorganFingerprintAsBitVect(rdkit_mol, radius, nBits=n_bits)
            rdkit_fp_array = np.zeros(n_bits, dtype=np.int8)
            for i in range(n_bits):
                rdkit_fp_array[i] = rdkit_fp[i]
            
            # Check for exact match
            if not np.array_equal(our_fp, rdkit_fp_array):
                mismatches.append({
                    'smiles': mol_data.smiles,
                    'radius': radius,
                    'n_bits': n_bits,
                    'our_bits_set': np.sum(our_fp),
                    'rdkit_bits_set': np.sum(rdkit_fp_array),
                    'hamming_distance': np.sum(our_fp != rdkit_fp_array)
                })
        
        # Report any mismatches
        if mismatches:
            error_msg = f"Found {len(mismatches)} mismatches for radius={radius}, n_bits={n_bits}:\n"
            for mismatch in mismatches[:5]:  # Show first 5 mismatches
                error_msg += f"  {mismatch}\n"
            pytest.fail(error_msg)
    
    @pytest.mark.parametrize("radius", [1, 2, 3])
    def test_morgan_counts_exact_match(self, validation_molecular_data, radius):
        """Test exact match with RDKit Morgan counts."""
        generator = FingerprintGenerator(use_gpu=False)
        
        mismatches = []
        
        for mol_data in validation_molecular_data[:10]:  # Test subset for counts
            rdkit_mol = mol_data.backend_data["rdkit_mol"]
            
            # Generate using our implementation
            our_counts = generator.generate_morgan_counts(mol_data, radius=radius)
            
            # Generate using RDKit reference
            rdkit_fp = AllChem.GetMorganFingerprint(rdkit_mol, radius)
            rdkit_counts = rdkit_fp.GetNonzeroElements()
            
            # Check for exact match
            if set(our_counts.keys()) != set(rdkit_counts.keys()):
                mismatches.append({
                    'smiles': mol_data.smiles,
                    'radius': radius,
                    'our_keys': len(our_counts),
                    'rdkit_keys': len(rdkit_counts),
                    'key_diff': set(our_counts.keys()).symmetric_difference(set(rdkit_counts.keys()))
                })
            else:
                # Check counts for matching keys
                for key in our_counts.keys():
                    if our_counts[key] != rdkit_counts[key]:
                        mismatches.append({
                            'smiles': mol_data.smiles,
                            'radius': radius,
                            'key': key,
                            'our_count': our_counts[key],
                            'rdkit_count': rdkit_counts[key]
                        })
                        break
        
        if mismatches:
            error_msg = f"Found {len(mismatches)} count mismatches for radius={radius}:\n"
            for mismatch in mismatches[:5]:
                error_msg += f"  {mismatch}\n"
            pytest.fail(error_msg)
    
    def test_morgan_fingerprint_features_validation(self, validation_molecular_data):
        """Test Morgan fingerprints with features against RDKit."""
        generator = FingerprintGenerator(use_gpu=False)
        
        for mol_data in validation_molecular_data[:5]:  # Test subset
            rdkit_mol = mol_data.backend_data["rdkit_mol"]
            
            # Test feature-based fingerprints
            our_fp = generator.generate_morgan_fingerprint(
                mol_data, radius=2, n_bits=2048, use_features=True, force_cpu=True
            )
            
            rdkit_fp = AllChem.GetMorganFingerprintAsBitVect(
                rdkit_mol, 2, nBits=2048, useFeatures=True
            )
            rdkit_fp_array = np.zeros(2048, dtype=np.int8)
            for i in range(2048):
                rdkit_fp_array[i] = rdkit_fp[i]
            
            assert np.array_equal(our_fp, rdkit_fp_array), f"Feature fingerprint mismatch for {mol_data.smiles}"
    
    def test_morgan_fingerprint_chirality_validation(self, validation_molecular_data):
        """Test Morgan fingerprints with chirality against RDKit."""
        generator = FingerprintGenerator(use_gpu=False)
        
        # Focus on molecules with stereochemistry
        chiral_molecules = [mol for mol in validation_molecular_data if '@' in mol.smiles]
        
        for mol_data in chiral_molecules:
            rdkit_mol = mol_data.backend_data["rdkit_mol"]
            
            # Test chirality-aware fingerprints
            our_fp = generator.generate_morgan_fingerprint(
                mol_data, radius=2, n_bits=2048, use_chirality=True, force_cpu=True
            )
            
            rdkit_fp = AllChem.GetMorganFingerprintAsBitVect(
                rdkit_mol, 2, nBits=2048, useChirality=True
            )
            rdkit_fp_array = np.zeros(2048, dtype=np.int8)
            for i in range(2048):
                rdkit_fp_array[i] = rdkit_fp[i]
            
            assert np.array_equal(our_fp, rdkit_fp_array), f"Chirality fingerprint mismatch for {mol_data.smiles}"


class TestMorganFingerprintStatisticalValidation:
    """Statistical validation of Morgan fingerprint properties."""
    
    def test_fingerprint_bit_distribution(self, validation_molecular_data):
        """Test that fingerprint bit distributions are reasonable."""
        generator = FingerprintGenerator(use_gpu=False)
        
        all_fingerprints = []
        for mol_data in validation_molecular_data:
            fp = generator.generate_morgan_fingerprint(mol_data, radius=2, n_bits=2048)
            all_fingerprints.append(fp)
        
        # Convert to array for analysis
        fp_matrix = np.array(all_fingerprints)
        
        # Check bit density (should be reasonable, not too sparse or dense)
        bit_densities = np.mean(fp_matrix, axis=1)
        avg_density = np.mean(bit_densities)
        
        assert 0.01 < avg_density < 0.5, f"Unusual bit density: {avg_density}"
        
        # Check that different molecules have different fingerprints
        unique_fps = len(set(tuple(fp) for fp in all_fingerprints))
        assert unique_fps == len(all_fingerprints), "Some molecules have identical fingerprints"
    
    def test_fingerprint_similarity_properties(self, validation_molecular_data):
        """Test that fingerprint similarities have expected properties."""
        from chemi_accel.descriptors.fingerprints import FingerprintSimilarity
        
        generator = FingerprintGenerator(use_gpu=False)
        
        # Generate fingerprints for similar molecules
        benzene_idx = next(i for i, mol in enumerate(validation_molecular_data) if mol.smiles == "c1ccccc1")
        toluene_idx = next(i for i, mol in enumerate(validation_molecular_data) if mol.smiles == "Cc1ccccc1")
        
        benzene_fp = generator.generate_morgan_fingerprint(validation_molecular_data[benzene_idx])
        toluene_fp = generator.generate_morgan_fingerprint(validation_molecular_data[toluene_idx])
        
        # Benzene and toluene should have high similarity
        similarity = FingerprintSimilarity.tanimoto_similarity(benzene_fp, toluene_fp)
        assert similarity > 0.3, f"Benzene-toluene similarity too low: {similarity}"
        
        # Test with very different molecules
        methane_idx = next(i for i, mol in enumerate(validation_molecular_data) if mol.smiles == "C")
        methane_fp = generator.generate_morgan_fingerprint(validation_molecular_data[methane_idx])
        
        # Methane and benzene should have low similarity
        similarity_low = FingerprintSimilarity.tanimoto_similarity(methane_fp, benzene_fp)
        assert similarity_low < 0.2, f"Methane-benzene similarity too high: {similarity_low}"


class TestMorganFingerprintPerformance:
    """Performance validation of Morgan fingerprint generation."""
    
    def test_fingerprint_generation_performance(self, validation_molecular_data):
        """Test performance of fingerprint generation."""
        generator = FingerprintGenerator(use_gpu=False)
        
        # Time fingerprint generation
        start_time = time.time()
        
        fingerprints = []
        for mol_data in validation_molecular_data:
            fp = generator.generate_morgan_fingerprint(mol_data, radius=2, n_bits=2048)
            fingerprints.append(fp)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should be reasonably fast (less than 1 second for test set)
        molecules_per_second = len(validation_molecular_data) / total_time
        
        print(f"Generated {len(validation_molecular_data)} fingerprints in {total_time:.3f}s")
        print(f"Performance: {molecules_per_second:.1f} molecules/second")
        
        # Basic performance check - should process at least 10 molecules per second
        assert molecules_per_second > 10, f"Performance too slow: {molecules_per_second:.1f} mol/s"
    
    def test_batch_vs_individual_performance(self, validation_molecular_data):
        """Compare batch vs individual fingerprint generation performance."""
        from chemi_accel.descriptors.fingerprints import generate_morgan_fingerprints
        
        # Time individual generation
        generator = FingerprintGenerator(use_gpu=False)
        start_time = time.time()
        
        individual_fps = []
        for mol_data in validation_molecular_data:
            fp = generator.generate_morgan_fingerprint(mol_data, radius=2, n_bits=1024)
            individual_fps.append(fp)
        
        individual_time = time.time() - start_time
        
        # Time batch generation
        start_time = time.time()
        batch_fps = generate_morgan_fingerprints(
            validation_molecular_data, radius=2, n_bits=1024, use_gpu=False
        )
        batch_time = time.time() - start_time
        
        print(f"Individual: {individual_time:.3f}s, Batch: {batch_time:.3f}s")
        
        # Results should be identical
        assert len(individual_fps) == len(batch_fps)
        for i, (ind_fp, batch_fp) in enumerate(zip(individual_fps, batch_fps)):
            assert np.array_equal(ind_fp, batch_fp), f"Mismatch at index {i}"
        
        # Batch should not be significantly slower (within 50%)
        assert batch_time < individual_time * 1.5, "Batch processing significantly slower"


class TestMorganFingerprintRobustness:
    """Test robustness of Morgan fingerprint implementation."""
    
    def test_edge_case_molecules(self):
        """Test fingerprint generation for edge case molecules."""
        if not RDKIT_AVAILABLE:
            pytest.skip("RDKit not available")
        
        generator = FingerprintGenerator(use_gpu=False)
        
        edge_cases = [
            "C",  # Single carbon
            "[H]",  # Single hydrogen
            "CC",  # Two carbons
            "C=C",  # Double bond
            "C#C",  # Triple bond
            "[C-]",  # Charged carbon
            "[NH4+]",  # Charged nitrogen
        ]
        
        for smiles in edge_cases:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mol_data = MolecularData(
                    mol_id=f"edge_{smiles}",
                    smiles=smiles,
                    backend_data={"rdkit_mol": mol}
                )
                
                # Should not raise exceptions
                fp = generator.generate_morgan_fingerprint(mol_data)
                assert isinstance(fp, np.ndarray)
                assert fp.dtype == np.int8
                assert np.all((fp == 0) | (fp == 1))
    
    def test_large_molecule_handling(self):
        """Test handling of large molecules."""
        if not RDKIT_AVAILABLE:
            pytest.skip("RDKit not available")
        
        generator = FingerprintGenerator(use_gpu=False)
        
        # Create a large molecule (long alkyl chain)
        large_smiles = "C" + "C" * 100  # 101-carbon chain
        mol = Chem.MolFromSmiles(large_smiles)
        
        if mol is not None:
            mol_data = MolecularData(
                mol_id="large_molecule",
                smiles=large_smiles,
                backend_data={"rdkit_mol": mol}
            )
            
            # Should handle large molecules without issues
            fp = generator.generate_morgan_fingerprint(mol_data, radius=3, n_bits=4096)
            assert isinstance(fp, np.ndarray)
            assert fp.shape == (4096,)
            assert np.sum(fp) > 0  # Should have some bits set


if __name__ == "__main__":
    pytest.main([__file__, "-v"])