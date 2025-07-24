"""
3D Descriptor Calculation Demo for ChemiAccel.

This example demonstrates the GPU-accelerated 3D descriptor calculation
capabilities including conformer generation, radius of gyration, principal
moments of inertia, asphericity, and shape-based descriptors.
"""

import pandas as pd
import numpy as np
import time

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("RDKit not available. Please install RDKit to run this demo.")
    exit(1)

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.chemi_accel.core.molecular import MolecularData
from src.chemi_accel.descriptors.descriptors_3d_integrated import Integrated3DDescriptorCalculator
from src.chemi_accel.descriptors.conformer_generation import GPUConformerGenerator, validate_conformer_quality


def create_test_molecules():
    """Create a diverse set of test molecules."""
    test_molecules = [
        ("CC", "Ethane"),
        ("CCO", "Ethanol"),
        ("c1ccccc1", "Benzene"),
        ("CCC(C)C", "Isopentane"),
        ("C1CCCCC1", "Cyclohexane"),
        ("CC(C)(C)C", "Neopentane"),
        ("CCCCCCCC", "Octane"),
        ("c1ccc2ccccc2c1", "Naphthalene"),
        ("CC(=O)O", "Acetic acid"),
        ("CCN(CC)CC", "Triethylamine")
    ]
    
    molecules = []
    
    for i, (smiles, name) in enumerate(test_molecules):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Failed to parse SMILES: {smiles}")
                continue
                
            mol = Chem.AddHs(mol)
            
            mol_data = MolecularData(
                smiles=smiles,
                mol_id=f"mol_{i}_{name.lower().replace(' ', '_')}",
                num_atoms=mol.GetNumAtoms(),
                num_bonds=mol.GetNumBonds(),
                molecular_weight=Chem.Descriptors.MolWt(mol),
                backend_data={"rdkit_mol": mol}
            )
            
            molecules.append((mol_data, name))
            
        except Exception as e:
            print(f"Error processing {name} ({smiles}): {e}")
            continue
    
    return molecules


def demo_conformer_generation():
    """Demonstrate conformer generation capabilities."""
    print("=" * 60)
    print("CONFORMER GENERATION DEMO")
    print("=" * 60)
    
    # Create test molecules
    molecules, names = zip(*create_test_molecules()[:5])  # Use first 5 molecules
    
    # Initialize conformer generator
    generator = GPUConformerGenerator(
        num_conformers=10,
        use_gpu=False,  # Use CPU for compatibility
        energy_minimize=True,
        force_field="UFF",
        prune_rms_thresh=0.5
    )
    
    print(f"Generating conformers for {len(molecules)} molecules...")
    
    results = []
    for mol_data, name in zip(molecules, names):
        start_time = time.time()
        
        try:
            result = generator.generate_conformers_parallel(mol_data)
            generation_time = time.time() - start_time
            
            rdkit_mol = result.backend_data["rdkit_mol"]
            num_conformers = rdkit_mol.GetNumConformers()
            
            # Validate conformer quality
            quality_metrics = validate_conformer_quality(result)
            
            results.append({
                "Molecule": name,
                "SMILES": mol_data.smiles,
                "Atoms": mol_data.num_atoms,
                "Conformers": num_conformers,
                "Generation Time (s)": f"{generation_time:.3f}",
                "RMSD Mean": f"{quality_metrics.get('rmsd_mean', 0):.3f}",
                "Diversity Score": f"{quality_metrics.get('diversity_score', 0):.3f}"
            })
            
        except Exception as e:
            print(f"Error generating conformers for {name}: {e}")
            results.append({
                "Molecule": name,
                "SMILES": mol_data.smiles,
                "Atoms": mol_data.num_atoms,
                "Conformers": 0,
                "Generation Time (s)": "Failed",
                "RMSD Mean": "N/A",
                "Diversity Score": "N/A"
            })
    
    # Display results
    df = pd.DataFrame(results)
    print("\nConformer Generation Results:")
    print(df.to_string(index=False))
    
    return molecules


def demo_3d_descriptors():
    """Demonstrate 3D descriptor calculations."""
    print("\n" + "=" * 60)
    print("3D DESCRIPTOR CALCULATION DEMO")
    print("=" * 60)
    
    # Create test molecules
    molecules, names = zip(*create_test_molecules())
    
    # Initialize 3D descriptor calculator
    calculator = Integrated3DDescriptorCalculator(
        generate_conformers=True,
        num_conformers=5,
        use_gpu=False,  # Use CPU for compatibility
        energy_minimize=True
    )
    
    print(f"Calculating 3D descriptors for {len(molecules)} molecules...")
    
    # Calculate descriptors
    start_time = time.time()
    results_df = calculator.calculate_3d_descriptors_batch(list(molecules))
    total_time = time.time() - start_time
    
    print(f"Total calculation time: {total_time:.2f} seconds")
    print(f"Average time per molecule: {total_time/len(molecules):.3f} seconds")
    
    # Display key descriptors
    key_descriptors = [
        "mol_id", "smiles", "radius_of_gyration", "asphericity", 
        "eccentricity", "pmi1", "pmi2", "pmi3", "molecular_volume"
    ]
    
    display_df = results_df[key_descriptors].copy()
    
    # Round numerical values for display
    numerical_cols = ["radius_of_gyration", "asphericity", "eccentricity", 
                     "pmi1", "pmi2", "pmi3", "molecular_volume"]
    
    for col in numerical_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(4)
    
    print("\n3D Descriptor Results:")
    print(display_df.to_string(index=False))
    
    return results_df


def demo_descriptor_analysis():
    """Demonstrate analysis of 3D descriptors."""
    print("\n" + "=" * 60)
    print("3D DESCRIPTOR ANALYSIS")
    print("=" * 60)
    
    # Get descriptor results
    molecules, names = zip(*create_test_molecules())
    
    calculator = Integrated3DDescriptorCalculator(
        generate_conformers=True,
        num_conformers=3,
        use_gpu=False
    )
    
    results_df = calculator.calculate_3d_descriptors_batch(list(molecules))
    
    # Analyze descriptor distributions
    numerical_descriptors = [
        "radius_of_gyration", "asphericity", "eccentricity",
        "pmi1", "pmi2", "pmi3", "molecular_volume", "geometric_diameter"
    ]
    
    print("Descriptor Statistics:")
    print("-" * 40)
    
    for desc in numerical_descriptors:
        if desc in results_df.columns:
            values = results_df[desc].dropna()
            if len(values) > 0:
                print(f"{desc:20s}: mean={values.mean():.4f}, std={values.std():.4f}, "
                      f"min={values.min():.4f}, max={values.max():.4f}")
    
    # Identify molecular shape categories
    print("\nMolecular Shape Analysis:")
    print("-" * 40)
    
    for idx, row in results_df.iterrows():
        mol_id = row.get("mol_id", f"mol_{idx}")
        asphericity = row.get("asphericity", 0)
        eccentricity = row.get("eccentricity", 0)
        
        # Classify molecular shape
        if asphericity < 0.1 and eccentricity < 0.1:
            shape = "Spherical"
        elif asphericity > 0.7 or eccentricity > 0.8:
            shape = "Linear"
        elif asphericity < 0.3:
            shape = "Oblate"
        else:
            shape = "Prolate"
        
        print(f"{mol_id:25s}: {shape:10s} (A={asphericity:.3f}, E={eccentricity:.3f})")


def demo_performance_comparison():
    """Demonstrate performance comparison between CPU and GPU (if available)."""
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    
    # Create larger test set
    molecules, names = zip(*create_test_molecules())
    test_molecules = list(molecules) * 3  # Replicate for performance testing
    
    print(f"Performance test with {len(test_molecules)} molecules...")
    
    # CPU calculation
    print("\nCPU Calculation:")
    cpu_calculator = Integrated3DDescriptorCalculator(
        generate_conformers=True,
        num_conformers=3,
        use_gpu=False
    )
    
    start_time = time.time()
    cpu_results = cpu_calculator.calculate_3d_descriptors_batch(test_molecules)
    cpu_time = time.time() - start_time
    
    print(f"CPU Time: {cpu_time:.2f} seconds")
    print(f"CPU Rate: {len(test_molecules)/cpu_time:.1f} molecules/second")
    
    # GPU calculation (if available)
    try:
        import cupy as cp
        print("\nGPU Calculation:")
        gpu_calculator = Integrated3DDescriptorCalculator(
            generate_conformers=True,
            num_conformers=3,
            use_gpu=True
        )
        
        start_time = time.time()
        gpu_results = gpu_calculator.calculate_3d_descriptors_batch(test_molecules)
        gpu_time = time.time() - start_time
        
        print(f"GPU Time: {gpu_time:.2f} seconds")
        print(f"GPU Rate: {len(test_molecules)/gpu_time:.1f} molecules/second")
        print(f"Speedup: {cpu_time/gpu_time:.1f}x")
        
    except ImportError:
        print("\nGPU calculation not available (CuPy not installed)")
    except Exception as e:
        print(f"\nGPU calculation failed: {e}")


def main():
    """Run all 3D descriptor demos."""
    print("ChemiAccel 3D Descriptor Calculation Demo")
    print("=" * 60)
    
    if not RDKIT_AVAILABLE:
        print("RDKit is required for this demo. Please install RDKit.")
        return
    
    try:
        # Run demos
        demo_conformer_generation()
        demo_3d_descriptors()
        demo_descriptor_analysis()
        demo_performance_comparison()
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()