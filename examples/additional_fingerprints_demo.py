#!/usr/bin/env python3
"""
Demonstration of additional fingerprint types in ChemiAccel.

This script demonstrates the GPU-accelerated generation of MACCS keys,
topological fingerprints, atom pair fingerprints, and topological torsion
fingerprints using the ChemiAccel library.
"""

import numpy as np
import time
from typing import List

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("RDKit not available - some comparisons will be skipped")

from chemi_accel.descriptors.fingerprints import (
    FingerprintGenerator,
    generate_maccs_fingerprints,
    generate_topological_fingerprints,
    generate_atom_pair_fingerprints,
    generate_topological_torsion_fingerprints,
    calculate_fingerprint_similarities
)
from chemi_accel.core.molecular import MolecularData


def create_test_molecules() -> List[MolecularData]:
    """Create a set of test molecules for fingerprint demonstration."""
    test_smiles = [
        ("CCO", "ethanol"),
        ("CC(=O)O", "acetic_acid"),
        ("c1ccccc1", "benzene"),
        ("Cc1ccccc1", "toluene"),
        ("CCc1ccccc1", "ethylbenzene"),
        ("c1ccc2ccccc2c1", "naphthalene"),
        ("CC(C)C(=O)Nc1ccc(cc1)C(=O)O", "ibuprofen"),
        ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "caffeine"),
        ("c1ccncc1", "pyridine"),
        ("CC(=O)N", "acetamide"),
    ]
    
    molecules = []
    for smiles, name in test_smiles:
        if RDKIT_AVAILABLE:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mol_data = MolecularData(
                    mol_id=name,
                    smiles=smiles,
                    backend_data={"rdkit_mol": mol}
                )
                molecules.append(mol_data)
        else:
            # Create minimal molecular data without RDKit
            mol_data = MolecularData(mol_id=name, smiles=smiles)
            molecules.append(mol_data)
    
    return molecules


def demonstrate_individual_fingerprints():
    """Demonstrate individual fingerprint generation."""
    print("=== Individual Fingerprint Generation ===")
    
    molecules = create_test_molecules()
    if not molecules:
        print("No molecules available for demonstration")
        return
    
    generator = FingerprintGenerator(use_gpu=False)  # Use CPU for demonstration
    
    # Test with ethanol
    ethanol = molecules[0]
    print(f"\nGenerating fingerprints for {ethanol.mol_id} ({ethanol.smiles}):")
    
    try:
        # MACCS keys
        maccs = generator.generate_maccs_keys(ethanol)
        print(f"  MACCS keys: {len(maccs)} bits, {np.sum(maccs)} bits set")
        
        # Topological fingerprint
        topo_fp = generator.generate_topological_fingerprint(ethanol, n_bits=2048)
        print(f"  Topological: {len(topo_fp)} bits, {np.sum(topo_fp)} bits set")
        
        # Atom pair fingerprint
        ap_fp = generator.generate_atom_pair_fingerprint(ethanol, n_bits=2048)
        print(f"  Atom pair: {len(ap_fp)} bits, {np.sum(ap_fp)} bits set")
        
        # Topological torsion fingerprint
        tt_fp = generator.generate_topological_torsion_fingerprint(ethanol, n_bits=2048)
        print(f"  Torsion: {len(tt_fp)} bits, {np.sum(tt_fp)} bits set")
        
    except Exception as e:
        print(f"  Error generating fingerprints: {e}")


def demonstrate_batch_fingerprints():
    """Demonstrate batch fingerprint generation."""
    print("\n=== Batch Fingerprint Generation ===")
    
    molecules = create_test_molecules()
    if not molecules:
        print("No molecules available for demonstration")
        return
    
    print(f"\nGenerating fingerprints for {len(molecules)} molecules:")
    
    try:
        # Time MACCS keys generation
        start_time = time.time()
        maccs_fps = generate_maccs_fingerprints(molecules, use_gpu=False)
        maccs_time = time.time() - start_time
        
        print(f"  MACCS keys: {len(maccs_fps)} fingerprints in {maccs_time:.3f}s")
        print(f"    Average bits set: {np.mean([np.sum(fp) for fp in maccs_fps]):.1f}")
        
        # Time topological fingerprints
        start_time = time.time()
        topo_fps = generate_topological_fingerprints(molecules, n_bits=2048, use_gpu=False)
        topo_time = time.time() - start_time
        
        print(f"  Topological: {len(topo_fps)} fingerprints in {topo_time:.3f}s")
        print(f"    Average bits set: {np.mean([np.sum(fp) for fp in topo_fps]):.1f}")
        
        # Time atom pair fingerprints
        start_time = time.time()
        ap_fps = generate_atom_pair_fingerprints(molecules, n_bits=2048, use_gpu=False)
        ap_time = time.time() - start_time
        
        print(f"  Atom pair: {len(ap_fps)} fingerprints in {ap_time:.3f}s")
        print(f"    Average bits set: {np.mean([np.sum(fp) for fp in ap_fps]):.1f}")
        
        # Time topological torsion fingerprints
        start_time = time.time()
        tt_fps = generate_topological_torsion_fingerprints(molecules, n_bits=2048, use_gpu=False)
        tt_time = time.time() - start_time
        
        print(f"  Torsion: {len(tt_fps)} fingerprints in {tt_time:.3f}s")
        print(f"    Average bits set: {np.mean([np.sum(fp) for fp in tt_fps]):.1f}")
        
    except Exception as e:
        print(f"  Error in batch generation: {e}")


def demonstrate_fingerprint_similarities():
    """Demonstrate fingerprint similarity calculations."""
    print("\n=== Fingerprint Similarity Analysis ===")
    
    molecules = create_test_molecules()
    if len(molecules) < 3:
        print("Need at least 3 molecules for similarity demonstration")
        return
    
    try:
        # Generate MACCS keys for similarity comparison
        maccs_fps = generate_maccs_fingerprints(molecules[:5], use_gpu=False)  # Use first 5 molecules
        
        # Calculate similarity matrix
        sim_matrix = calculate_fingerprint_similarities(maccs_fps, method="tanimoto")
        
        print(f"\nTanimoto similarity matrix for MACCS keys ({len(maccs_fps)} molecules):")
        print("Molecules:", [mol.mol_id for mol in molecules[:5]])
        
        # Print similarity matrix with molecule names
        for i, mol_i in enumerate(molecules[:5]):
            row_str = f"{mol_i.mol_id:12s}: "
            for j in range(len(maccs_fps)):
                row_str += f"{sim_matrix[i, j]:.3f} "
            print(row_str)
        
        # Find most similar pair
        max_sim = 0.0
        max_pair = (0, 0)
        for i in range(len(maccs_fps)):
            for j in range(i + 1, len(maccs_fps)):
                if sim_matrix[i, j] > max_sim:
                    max_sim = sim_matrix[i, j]
                    max_pair = (i, j)
        
        if max_sim > 0:
            mol_i = molecules[max_pair[0]]
            mol_j = molecules[max_pair[1]]
            print(f"\nMost similar pair: {mol_i.mol_id} - {mol_j.mol_id} (similarity: {max_sim:.3f})")
        
    except Exception as e:
        print(f"  Error in similarity analysis: {e}")


def demonstrate_fingerprint_properties():
    """Demonstrate fingerprint properties and characteristics."""
    print("\n=== Fingerprint Properties Analysis ===")
    
    molecules = create_test_molecules()
    if not molecules:
        print("No molecules available for demonstration")
        return
    
    generator = FingerprintGenerator(use_gpu=False)
    
    # Analyze fingerprint properties for different molecule types
    test_cases = [
        ("Simple molecule", molecules[0]),  # ethanol
        ("Aromatic molecule", molecules[2] if len(molecules) > 2 else molecules[0]),  # benzene
        ("Complex molecule", molecules[6] if len(molecules) > 6 else molecules[0]),  # ibuprofen
    ]
    
    for case_name, molecule in test_cases:
        print(f"\n{case_name}: {molecule.mol_id} ({molecule.smiles})")
        
        try:
            # Generate all fingerprint types
            maccs = generator.generate_maccs_keys(molecule)
            topo_fp = generator.generate_topological_fingerprint(molecule, n_bits=1024)
            ap_fp = generator.generate_atom_pair_fingerprint(molecule, n_bits=1024)
            tt_fp = generator.generate_topological_torsion_fingerprint(molecule, n_bits=1024)
            
            # Calculate bit densities
            maccs_density = np.sum(maccs) / len(maccs)
            topo_density = np.sum(topo_fp) / len(topo_fp)
            ap_density = np.sum(ap_fp) / len(ap_fp)
            tt_density = np.sum(tt_fp) / len(tt_fp)
            
            print(f"  MACCS keys bit density: {maccs_density:.3f} ({np.sum(maccs)}/167)")
            print(f"  Topological bit density: {topo_density:.3f} ({np.sum(topo_fp)}/1024)")
            print(f"  Atom pair bit density: {ap_density:.3f} ({np.sum(ap_fp)}/1024)")
            print(f"  Torsion bit density: {tt_density:.3f} ({np.sum(tt_fp)}/1024)")
            
        except Exception as e:
            print(f"  Error analyzing {molecule.mol_id}: {e}")


def main():
    """Main demonstration function."""
    print("ChemiAccel Additional Fingerprints Demonstration")
    print("=" * 50)
    
    if not RDKIT_AVAILABLE:
        print("Warning: RDKit not available. Some features may not work properly.")
        print()
    
    try:
        demonstrate_individual_fingerprints()
        demonstrate_batch_fingerprints()
        demonstrate_fingerprint_similarities()
        demonstrate_fingerprint_properties()
        
        print("\n=== Summary ===")
        print("Successfully demonstrated:")
        print("✓ MACCS keys generation")
        print("✓ Topological fingerprint generation")
        print("✓ Atom pair fingerprint generation")
        print("✓ Topological torsion fingerprint generation")
        print("✓ Batch processing capabilities")
        print("✓ Similarity calculations")
        print("✓ Fingerprint property analysis")
        
    except Exception as e:
        print(f"\nError in demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()