"""
Morgan Fingerprint Generation Demo.

This example demonstrates the GPU-accelerated Morgan fingerprint generation
capabilities of ChemiAccel, including both binary and count-based fingerprints,
similarity calculations, and performance comparisons.
"""

import numpy as np
import pandas as pd
import time
from typing import List, Dict

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Some comparisons will be skipped.")

from chemi_accel.descriptors.fingerprints import (
    FingerprintGenerator,
    generate_morgan_fingerprints,
    FingerprintSimilarity
)
from chemi_accel.core.molecular import MolecularData


def create_molecular_data(smiles_list: List[str]) -> List[MolecularData]:
    """Create MolecularData objects from SMILES strings."""
    if not RDKIT_AVAILABLE:
        raise RuntimeError("RDKit is required for this demo")
    
    mol_data_list = []
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mol_data = MolecularData(
                mol_id=f"mol_{i}",
                smiles=smiles,
                backend_data={"rdkit_mol": mol}
            )
            mol_data_list.append(mol_data)
        else:
            print(f"Warning: Could not parse SMILES: {smiles}")
    
    return mol_data_list


def demo_basic_morgan_fingerprints():
    """Demonstrate basic Morgan fingerprint generation."""
    print("=== Basic Morgan Fingerprint Generation ===")
    
    # Test molecules
    test_smiles = [
        "CCO",  # Ethanol
        "CC(=O)O",  # Acetic acid
        "c1ccccc1",  # Benzene
        "Cc1ccccc1",  # Toluene
        "CCc1ccccc1",  # Ethylbenzene
    ]
    
    mol_data_list = create_molecular_data(test_smiles)
    generator = FingerprintGenerator(use_gpu=False)  # Start with CPU
    
    print(f"Generated molecular data for {len(mol_data_list)} molecules")
    
    # Generate Morgan fingerprints with different parameters
    for radius in [1, 2, 3]:
        print(f"\nRadius {radius} fingerprints:")
        for mol_data in mol_data_list:
            fp = generator.generate_morgan_fingerprint(mol_data, radius=radius, n_bits=1024)
            bits_set = np.sum(fp)
            print(f"  {mol_data.smiles:15} -> {bits_set:3d} bits set")


def demo_morgan_counts():
    """Demonstrate Morgan fingerprint count generation."""
    print("\n=== Morgan Fingerprint Counts ===")
    
    test_smiles = ["c1ccccc1", "Cc1ccccc1", "CCc1ccccc1"]  # Benzene series
    mol_data_list = create_molecular_data(test_smiles)
    generator = FingerprintGenerator(use_gpu=False)
    
    for mol_data in mol_data_list:
        counts = generator.generate_morgan_counts(mol_data, radius=2)
        print(f"\n{mol_data.smiles}:")
        print(f"  Unique substructures: {len(counts)}")
        print(f"  Total count: {sum(counts.values())}")
        
        # Show top 5 most frequent substructures
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        print("  Top substructures:")
        for hash_val, count in sorted_counts[:5]:
            print(f"    Hash {hash_val:10d}: {count} occurrences")


def demo_fingerprint_similarity():
    """Demonstrate fingerprint similarity calculations."""
    print("\n=== Fingerprint Similarity Analysis ===")
    
    # Molecules with known structural relationships
    test_smiles = [
        "c1ccccc1",      # Benzene
        "Cc1ccccc1",     # Toluene (similar to benzene)
        "CCc1ccccc1",    # Ethylbenzene (similar to toluene)
        "c1ccc2ccccc2c1", # Naphthalene (aromatic, but different)
        "CCO",           # Ethanol (very different)
    ]
    
    mol_names = ["Benzene", "Toluene", "Ethylbenzene", "Naphthalene", "Ethanol"]
    mol_data_list = create_molecular_data(test_smiles)
    generator = FingerprintGenerator(use_gpu=False)
    
    # Generate fingerprints
    fingerprints = []
    for mol_data in mol_data_list:
        fp = generator.generate_morgan_fingerprint(mol_data, radius=2, n_bits=2048)
        fingerprints.append(fp)
    
    # Calculate similarity matrix
    similarity_matrix = FingerprintSimilarity.calculate_similarity_matrix(
        fingerprints, method="tanimoto"
    )
    
    print("Tanimoto Similarity Matrix:")
    print("           ", end="")
    for name in mol_names:
        print(f"{name[:8]:>8}", end="")
    print()
    
    for i, name in enumerate(mol_names):
        print(f"{name[:10]:10}", end="")
        for j in range(len(mol_names)):
            print(f"{similarity_matrix[i, j]:8.3f}", end="")
        print()
    
    # Highlight interesting similarities
    print("\nInteresting similarities:")
    print(f"Benzene vs Toluene: {similarity_matrix[0, 1]:.3f}")
    print(f"Toluene vs Ethylbenzene: {similarity_matrix[1, 2]:.3f}")
    print(f"Benzene vs Naphthalene: {similarity_matrix[0, 3]:.3f}")
    print(f"Benzene vs Ethanol: {similarity_matrix[0, 4]:.3f}")


def demo_batch_processing():
    """Demonstrate batch fingerprint generation."""
    print("\n=== Batch Processing Demo ===")
    
    # Generate a larger set of molecules
    test_smiles = [
        "CCO", "CC(=O)O", "c1ccccc1", "Cc1ccccc1", "CCc1ccccc1",
        "c1ccncc1", "CC(C)C", "C1CCCCC1", "CCN(CC)CC", "CC(=O)N",
        "CC(=O)OC", "CCS", "CCN", "CC#N", "c1coc2ccccc12",
        "CC(C)(C)c1ccc(cc1)O", "CCN(CC)CC(=O)Nc1c(cccc1C)C",
        "C1CCC2CCCCC2C1", "C1=CC=C2C(=C1)C=CC=C2", "CC(C)(C)C1=CC=C(C=C1)C(C)(C)C"
    ]
    
    mol_data_list = create_molecular_data(test_smiles)
    print(f"Processing {len(mol_data_list)} molecules")
    
    # Time individual processing
    start_time = time.time()
    generator = FingerprintGenerator(use_gpu=False)
    individual_fps = []
    for mol_data in mol_data_list:
        fp = generator.generate_morgan_fingerprint(mol_data, radius=2, n_bits=1024)
        individual_fps.append(fp)
    individual_time = time.time() - start_time
    
    # Time batch processing
    start_time = time.time()
    batch_fps = generate_morgan_fingerprints(
        mol_data_list, radius=2, n_bits=1024, use_gpu=False
    )
    batch_time = time.time() - start_time
    
    print(f"Individual processing: {individual_time:.3f}s ({len(mol_data_list)/individual_time:.1f} mol/s)")
    print(f"Batch processing: {batch_time:.3f}s ({len(mol_data_list)/batch_time:.1f} mol/s)")
    
    # Verify results are identical
    matches = sum(1 for fp1, fp2 in zip(individual_fps, batch_fps) if np.array_equal(fp1, fp2))
    print(f"Result consistency: {matches}/{len(mol_data_list)} matches")


def demo_parameter_effects():
    """Demonstrate effects of different Morgan fingerprint parameters."""
    print("\n=== Parameter Effects Demo ===")
    
    # Use a moderately complex molecule
    smiles = "CC(=O)Nc1ccc(cc1)C(=O)O"  # Para-aminobenzoic acid derivative
    mol_data_list = create_molecular_data([smiles])
    mol_data = mol_data_list[0]
    generator = FingerprintGenerator(use_gpu=False)
    
    print(f"Test molecule: {smiles}")
    
    # Effect of radius
    print("\nEffect of radius:")
    for radius in range(0, 5):
        fp = generator.generate_morgan_fingerprint(mol_data, radius=radius, n_bits=2048)
        bits_set = np.sum(fp)
        print(f"  Radius {radius}: {bits_set:3d} bits set")
    
    # Effect of fingerprint size
    print("\nEffect of fingerprint size:")
    for n_bits in [512, 1024, 2048, 4096]:
        fp = generator.generate_morgan_fingerprint(mol_data, radius=2, n_bits=n_bits)
        bits_set = np.sum(fp)
        density = bits_set / n_bits
        print(f"  {n_bits:4d} bits: {bits_set:3d} set ({density:.3f} density)")
    
    # Effect of features and chirality (if applicable)
    print("\nEffect of features and chirality:")
    fp_normal = generator.generate_morgan_fingerprint(mol_data, radius=2, n_bits=2048)
    fp_features = generator.generate_morgan_fingerprint(mol_data, radius=2, n_bits=2048, use_features=True)
    fp_chirality = generator.generate_morgan_fingerprint(mol_data, radius=2, n_bits=2048, use_chirality=True)
    
    print(f"  Normal:    {np.sum(fp_normal):3d} bits set")
    print(f"  Features:  {np.sum(fp_features):3d} bits set")
    print(f"  Chirality: {np.sum(fp_chirality):3d} bits set")


def demo_rdkit_comparison():
    """Compare ChemiAccel results with RDKit reference."""
    if not RDKIT_AVAILABLE:
        print("\n=== RDKit Comparison (Skipped - RDKit not available) ===")
        return
    
    print("\n=== RDKit Comparison ===")
    
    test_smiles = ["CCO", "c1ccccc1", "CC(=O)O", "Cc1ccccc1"]
    mol_data_list = create_molecular_data(test_smiles)
    generator = FingerprintGenerator(use_gpu=False)
    
    print("Comparing with RDKit reference implementation:")
    
    for mol_data in mol_data_list:
        rdkit_mol = mol_data.backend_data["rdkit_mol"]
        
        # Our implementation
        our_fp = generator.generate_morgan_fingerprint(mol_data, radius=2, n_bits=1024)
        
        # RDKit reference
        rdkit_fp = AllChem.GetMorganFingerprintAsBitVect(rdkit_mol, 2, nBits=1024)
        rdkit_fp_array = np.zeros(1024, dtype=np.int8)
        for i in range(1024):
            rdkit_fp_array[i] = rdkit_fp[i]
        
        # Compare
        match = np.array_equal(our_fp, rdkit_fp_array)
        our_bits = np.sum(our_fp)
        rdkit_bits = np.sum(rdkit_fp_array)
        
        print(f"  {mol_data.smiles:15} -> Match: {match}, Ours: {our_bits:3d}, RDKit: {rdkit_bits:3d}")


def demo_gpu_acceleration():
    """Demonstrate GPU acceleration (if available)."""
    print("\n=== GPU Acceleration Demo ===")
    
    try:
        gpu_generator = FingerprintGenerator(use_gpu=True)
        print("GPU acceleration available!")
        
        # This would demonstrate GPU vs CPU performance
        # For now, just show that GPU generator can be created
        print("GPU fingerprint generator initialized successfully")
        
    except Exception as e:
        print(f"GPU acceleration not available: {e}")
        print("Falling back to CPU implementation")


def main():
    """Run all Morgan fingerprint demos."""
    print("ChemiAccel Morgan Fingerprint Demo")
    print("=" * 50)
    
    if not RDKIT_AVAILABLE:
        print("Error: RDKit is required for this demo")
        return
    
    try:
        demo_basic_morgan_fingerprints()
        demo_morgan_counts()
        demo_fingerprint_similarity()
        demo_batch_processing()
        demo_parameter_effects()
        demo_rdkit_comparison()
        demo_gpu_acceleration()
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()