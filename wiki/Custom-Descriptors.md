# Custom Descriptors

ChemiAccel provides a flexible framework for implementing custom molecular descriptors. This guide covers the creation, registration, and optimization of user-defined descriptors for specialized applications.

## Overview

Custom descriptor capabilities include:
- **Flexible API**: Simple base classes for descriptor implementation
- **Multiple Backend Support**: Custom descriptors work across CPU, CUDA, and JAX backends
- **Performance Optimization**: Vectorization and batch processing support
- **Caching Integration**: Automatic caching for expensive custom calculations
- **Validation Framework**: Built-in testing and validation tools
- **Integration Support**: Seamless integration with existing ChemiAccel workflows

## Quick Start

### Basic Custom Descriptor

```python
from chemi_accel.descriptors.custom import CustomDescriptor
from chemi_accel import Calculator
import numpy as np
from rdkit import Chem

class AromaticAtomCount(CustomDescriptor):
    """Count the number of aromatic atoms in a molecule."""
    
    name = "aromatic_atom_count"
    description = "Number of aromatic atoms in the molecule"
    return_type = int
    
    def calculate(self, mol):
        """Calculate descriptor for a single molecule."""
        if mol is None:
            return 0
        
        aromatic_count = 0
        for atom in mol.GetAtoms():
            if atom.GetIsAromatic():
                aromatic_count += 1
        
        return aromatic_count

# Register and use custom descriptor
calc = Calculator()
custom_descriptor = AromaticAtomCount()
calc.register_custom_descriptor(custom_descriptor)

molecules = ["c1ccccc1", "CCO", "c1ccccc1c2ccccc2"]
results = calc.pandas(molecules, descriptors=["aromatic_atom_count"])
print(results)
```

## Advanced Custom Descriptors

### Multi-Value Descriptors

```python
class MolecularComplexityProfile(CustomDescriptor):
    """Calculate a comprehensive molecular complexity profile."""
    
    name = "complexity_profile"
    description = "Multi-dimensional molecular complexity assessment"
    return_type = dict
    output_columns = [
        "structural_complexity",
        "electronic_complexity", 
        "conformational_complexity",
        "synthetic_complexity"
    ]
    
    def calculate(self, mol):
        """Calculate complexity profile for a molecule."""
        if mol is None:
            return {col: np.nan for col in self.output_columns}
        
        try:
            # Structural complexity based on graph properties
            structural = self.calculate_structural_complexity(mol)
            
            # Electronic complexity based on atomic properties
            electronic = self.calculate_electronic_complexity(mol)
            
            # Conformational complexity based on rotatable bonds
            conformational = self.calculate_conformational_complexity(mol)
            
            # Synthetic complexity estimation
            synthetic = self.calculate_synthetic_complexity(mol)
            
            return {
                "structural_complexity": structural,
                "electronic_complexity": electronic,
                "conformational_complexity": conformational,
                "synthetic_complexity": synthetic
            }
            
        except Exception as e:
            # Return NaN values if calculation fails
            return {col: np.nan for col in self.output_columns}
    
    def calculate_structural_complexity(self, mol):
        """Calculate structural complexity score."""
        
        # Graph-based complexity metrics
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()
        num_rings = mol.GetRingInfo().NumRings()
        
        # Branching complexity
        branching_points = sum(1 for atom in mol.GetAtoms() if atom.GetDegree() > 2)
        
        # Heteratom complexity
        heteroatoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() != 6)
        
        # Normalize and combine metrics
        complexity = (
            num_atoms * 0.1 +
            num_bonds * 0.15 +
            num_rings * 0.3 +
            branching_points * 0.25 +
            heteroatoms * 0.2
        )
        
        return min(complexity / 10.0, 1.0)  # Scale to 0-1
    
    def calculate_electronic_complexity(self, mol):
        """Calculate electronic complexity based on atomic properties."""
        
        from rdkit.Chem import Descriptors, Crippen
        
        # Electronegativity variance
        electronegativities = []
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            # Simplified electronegativity scale
            eneg_map = {1: 2.2, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 
                       15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66}
            electronegativities.append(eneg_map.get(atomic_num, 2.0))
        
        if len(electronegativities) > 1:
            eneg_variance = np.var(electronegativities)
        else:
            eneg_variance = 0
        
        # Partial charge distribution (simplified)
        try:
            formal_charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
            charge_variance = np.var(formal_charges) if formal_charges else 0
        except:
            charge_variance = 0
        
        # Combine electronic descriptors
        complexity = min((eneg_variance * 2 + charge_variance) / 5.0, 1.0)
        
        return complexity
    
    def calculate_conformational_complexity(self, mol):
        """Calculate conformational complexity."""
        
        from rdkit.Chem import Descriptors
        
        # Rotatable bonds
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        
        # Ring flexibility
        ring_info = mol.GetRingInfo()
        flexible_rings = 0
        
        for ring in ring_info.AtomRings():
            if len(ring) > 6:  # Large rings are more flexible
                flexible_rings += 1
        
        # Normalize
        complexity = min((rotatable_bonds * 0.1 + flexible_rings * 0.2), 1.0)
        
        return complexity
    
    def calculate_synthetic_complexity(self, mol):
        """Estimate synthetic complexity."""
        
        # Simple heuristic-based SA score
        from rdkit.Chem import Descriptors
        
        # Factors that increase synthetic complexity
        complexity_factors = []
        
        # Ring systems
        num_rings = mol.GetRingInfo().NumRings()
        complexity_factors.append(num_rings * 0.1)
        
        # Heteroatoms
        heteroatoms = sum(1 for atom in mol.GetAtoms() 
                         if atom.GetAtomicNum() not in [1, 6])
        complexity_factors.append(heteroatoms * 0.05)
        
        # Functional groups (simplified detection)
        smiles = Chem.MolToSmiles(mol)
        functional_groups = ['C=O', 'C#N', 'S(=O)', 'N=O', '[nH]']
        fg_count = sum(smiles.count(fg) for fg in functional_groups)
        complexity_factors.append(fg_count * 0.15)
        
        # Combine factors
        total_complexity = sum(complexity_factors)
        
        return min(total_complexity, 1.0)

# Usage example
complexity_descriptor = MolecularComplexityProfile()
calc.register_custom_descriptor(complexity_descriptor)

molecules = [
    "CCO",                    # Simple alcohol
    "c1ccccc1",              # Benzene
    "CC(C)(C)c1ccc(O)cc1",   # tert-Butylphenol
    "CN1CCC[C@H]1c2cccnc2"   # Complex alkaloid
]

results = calc.pandas(molecules, descriptors=["complexity_profile"])
print(results)
```

### Performance-Optimized Descriptors

```python
class OptimizedDescriptor(CustomDescriptor):
    """High-performance custom descriptor with batch processing."""
    
    name = "optimized_descriptor"
    description = "Performance-optimized custom descriptor"
    return_type = float
    
    # Enable batch processing for better performance
    supports_batch_calculation = True
    
    def __init__(self):
        super().__init__()
        self.lookup_table = self._build_lookup_table()
        self.calculation_cache = {}
    
    def _build_lookup_table(self):
        """Build lookup table for common substructures."""
        
        from rdkit import Chem
        
        # Common functional groups and their contributions
        functional_groups = {
            'hydroxyl': ('[OH]', 1.5),
            'carbonyl': ('C=O', 2.0),
            'amino': ('[NH2]', 1.8),
            'carboxyl': ('C(=O)O', 3.0),
            'phenyl': ('c1ccccc1', 2.5),
            'methyl': ('C', 0.5)
        }
        
        lookup_table = {}
        for name, (smarts, weight) in functional_groups.items():
            pattern = Chem.MolFromSmarts(smarts)
            lookup_table[name] = (pattern, weight)
        
        return lookup_table
    
    def calculate(self, mol):
        """Calculate descriptor for single molecule (fallback)."""
        return self.batch_calculate([mol])[0]
    
    def batch_calculate(self, mols):
        """Optimized batch calculation."""
        
        results = []
        
        # Pre-allocate result array
        batch_size = len(mols)
        batch_results = np.zeros(batch_size, dtype=np.float32)
        
        # Vectorized calculations where possible
        for i, mol in enumerate(mols):
            if mol is None:
                batch_results[i] = np.nan
                continue
            
            # Check cache first
            mol_key = Chem.MolToSmiles(mol)
            if mol_key in self.calculation_cache:
                batch_results[i] = self.calculation_cache[mol_key]
                continue
            
            # Fast calculation using lookup table
            descriptor_value = 0.0
            
            for name, (pattern, weight) in self.lookup_table.items():
                matches = mol.GetSubstructMatches(pattern)
                descriptor_value += len(matches) * weight
            
            # Normalize by molecular size
            descriptor_value /= mol.GetNumAtoms()
            
            batch_results[i] = descriptor_value
            
            # Cache result
            self.calculation_cache[mol_key] = descriptor_value
        
        return batch_results.tolist()
    
    def clear_cache(self):
        """Clear calculation cache."""
        self.calculation_cache.clear()

# CUDA-Accelerated Custom Descriptor
class CUDAAcceleratedDescriptor(CustomDescriptor):
    """Custom descriptor with CUDA acceleration."""
    
    name = "cuda_accelerated_descriptor"
    description = "CUDA-accelerated custom molecular descriptor"
    return_type = float
    supports_cuda = True
    
    def __init__(self):
        super().__init__()
        self.cuda_available = self._check_cuda_availability()
        if self.cuda_available:
            self.cuda_kernel = self._compile_cuda_kernel()
    
    def _check_cuda_availability(self):
        """Check if CUDA is available."""
        try:
            import cupy as cp
            return cp.cuda.is_available()
        except ImportError:
            return False
    
    def _compile_cuda_kernel(self):
        """Compile CUDA kernel for descriptor calculation."""
        
        if not self.cuda_available:
            return None
        
        import cupy as cp
        
        cuda_code = '''
        extern "C" __global__
        void calculate_descriptor_kernel(
            const float* molecular_data,
            float* results,
            int num_molecules,
            int features_per_molecule
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (idx < num_molecules) {
                float descriptor_value = 0.0f;
                
                // Custom calculation logic
                for (int i = 0; i < features_per_molecule; i++) {
                    float feature = molecular_data[idx * features_per_molecule + i];
                    descriptor_value += feature * feature;  // Example calculation
                }
                
                results[idx] = sqrtf(descriptor_value / features_per_molecule);
            }
        }
        '''
        
        return cp.RawKernel(cuda_code, 'calculate_descriptor_kernel')
    
    def calculate(self, mol):
        """Calculate descriptor for single molecule."""
        if self.cuda_available:
            return self.cuda_calculate([mol])[0]
        else:
            return self.cpu_calculate(mol)
    
    def cuda_calculate(self, mols):
        """CUDA-accelerated batch calculation."""
        
        import cupy as cp
        
        # Convert molecules to feature vectors
        feature_vectors = []
        for mol in mols:
            if mol is None:
                features = np.zeros(10, dtype=np.float32)
            else:
                features = self.mol_to_features(mol)
            feature_vectors.append(features)
        
        # Transfer to GPU
        molecular_data = cp.array(np.array(feature_vectors), dtype=cp.float32)
        results = cp.zeros(len(mols), dtype=cp.float32)
        
        # Launch CUDA kernel
        threads_per_block = 256
        blocks_per_grid = (len(mols) + threads_per_block - 1) // threads_per_block
        
        self.cuda_kernel(
            (blocks_per_grid,), (threads_per_block,),
            (molecular_data, results, len(mols), 10)
        )
        
        # Transfer results back to CPU
        return results.get().tolist()
    
    def cpu_calculate(self, mol):
        """CPU fallback calculation."""
        if mol is None:
            return np.nan
        
        features = self.mol_to_features(mol)
        return np.linalg.norm(features) / len(features)
    
    def mol_to_features(self, mol):
        """Convert molecule to feature vector."""
        
        features = np.zeros(10, dtype=np.float32)
        
        if mol is None:
            return features
        
        # Example feature extraction
        features[0] = mol.GetNumAtoms()
        features[1] = mol.GetNumBonds()
        features[2] = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        features[3] = mol.GetRingInfo().NumRings()
        
        # Add more features as needed
        atom_types = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        features[4] = np.mean(atom_types) if atom_types else 0
        features[5] = np.std(atom_types) if len(atom_types) > 1 else 0
        
        # Normalize features
        features = features / np.linalg.norm(features) if np.linalg.norm(features) > 0 else features
        
        return features

# Example usage
optimized_desc = OptimizedDescriptor()
cuda_desc = CUDAAcceleratedDescriptor()

calc = Calculator(backend="cuda")
calc.register_custom_descriptor(optimized_desc)
calc.register_custom_descriptor(cuda_desc)

# Test with larger dataset
large_dataset = ["CCO", "c1ccccc1", "CC(=O)O"] * 1000

results = calc.pandas(
    large_dataset, 
    descriptors=["optimized_descriptor", "cuda_accelerated_descriptor"]
)

print(f"Calculated descriptors for {len(results)} molecules")
print(results.head())
```

## Specialized Domain Descriptors

### Polymer-Specific Descriptors

```python
class PolymerDescriptor(CustomDescriptor):
    """Specialized descriptors for polymer analysis."""
    
    name = "polymer_properties"
    description = "Polymer-specific molecular properties"
    return_type = dict
    output_columns = [
        "repeat_unit_mw",
        "tacticity_factor",
        "chain_flexibility",
        "glass_transition_predictor"
    ]
    
    def calculate(self, mol):
        """Calculate polymer-specific properties."""
        
        if mol is None:
            return {col: np.nan for col in self.output_columns}
        
        try:
            return {
                "repeat_unit_mw": self.calculate_repeat_unit_mw(mol),
                "tacticity_factor": self.calculate_tacticity_factor(mol),
                "chain_flexibility": self.calculate_chain_flexibility(mol),
                "glass_transition_predictor": self.predict_glass_transition(mol)
            }
        except Exception:
            return {col: np.nan for col in self.output_columns}
    
    def calculate_repeat_unit_mw(self, mol):
        """Calculate molecular weight of repeat unit."""
        from rdkit.Chem import Descriptors
        return Descriptors.MolWt(mol)
    
    def calculate_tacticity_factor(self, mol):
        """Estimate tacticity influence factor."""
        
        # Count stereocenters
        stereocenters = 0
        for atom in mol.GetAtoms():
            if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
                stereocenters += 1
        
        # Normalize by chain length (approximated by heavy atoms)
        heavy_atoms = mol.GetNumHeavyAtoms()
        
        return stereocenters / heavy_atoms if heavy_atoms > 0 else 0
    
    def calculate_chain_flexibility(self, mol):
        """Calculate chain flexibility factor."""
        
        from rdkit.Chem import Descriptors
        
        # Rotatable bonds indicate flexibility
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        
        # Ring systems reduce flexibility
        rings = mol.GetRingInfo().NumRings()
        
        # Aromatic content reduces flexibility
        aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        
        total_atoms = mol.GetNumAtoms()
        
        flexibility = (rotatable_bonds - rings * 0.5 - aromatic_atoms * 0.1) / total_atoms
        
        return max(0, flexibility)
    
    def predict_glass_transition(self, mol):
        """Predict glass transition temperature contribution."""
        
        # Simplified group contribution method
        contributions = {
            'CH2': -20,      # Methylene groups lower Tg
            'CH3': -5,       # Methyl groups lower Tg  
            'aromatic': 100, # Aromatic rings raise Tg
            'OH': 50,        # Hydrogen bonding raises Tg
            'C=O': 30,       # Polar groups raise Tg
            'CF2': 40        # Fluorine raises Tg
        }
        
        tg_contribution = 0
        
        # Count group contributions
        smiles = Chem.MolToSmiles(mol)
        
        # Simplified pattern matching
        if 'CC' in smiles:
            tg_contribution += contributions['CH2'] * smiles.count('CC')
        
        aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        tg_contribution += contributions['aromatic'] * (aromatic_atoms / 6)  # Per benzene ring
        
        if 'O' in smiles:
            tg_contribution += contributions['OH'] * smiles.count('O')
        
        return tg_contribution

# Drug Discovery Descriptors
class DrugLikenessDescriptor(CustomDescriptor):
    """Comprehensive drug-likeness assessment."""
    
    name = "drug_likeness_profile"
    description = "Multi-faceted drug-likeness evaluation"
    return_type = dict
    output_columns = [
        "lipinski_score",
        "bioavailability_score",
        "synthetic_accessibility",
        "toxicity_alerts",
        "metabolic_stability"
    ]
    
    def calculate(self, mol):
        """Calculate drug-likeness profile."""
        
        if mol is None:
            return {col: np.nan for col in self.output_columns}
        
        try:
            return {
                "lipinski_score": self.calculate_lipinski_score(mol),
                "bioavailability_score": self.calculate_bioavailability_score(mol),
                "synthetic_accessibility": self.calculate_sa_score(mol),
                "toxicity_alerts": self.count_toxicity_alerts(mol),
                "metabolic_stability": self.predict_metabolic_stability(mol)
            }
        except Exception:
            return {col: np.nan for col in self.output_columns}
    
    def calculate_lipinski_score(self, mol):
        """Calculate Lipinski Rule of Five compliance score."""
        
        from rdkit.Chem import Descriptors, Crippen
        
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        
        violations = 0
        if mw > 500: violations += 1
        if logp > 5: violations += 1
        if hbd > 5: violations += 1
        if hba > 10: violations += 1
        
        return (4 - violations) / 4.0  # 0-1 score
    
    def calculate_bioavailability_score(self, mol):
        """Estimate oral bioavailability score."""
        
        from rdkit.Chem import Descriptors
        
        # Additional bioavailability factors
        tpsa = Descriptors.TPSA(mol)
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        
        score = 1.0
        
        # TPSA penalty
        if tpsa > 140:
            score *= 0.5
        elif tpsa > 90:
            score *= 0.8
        
        # Rotatable bonds penalty
        if rotatable_bonds > 10:
            score *= 0.6
        elif rotatable_bonds > 7:
            score *= 0.8
        
        return score
    
    def calculate_sa_score(self, mol):
        """Calculate synthetic accessibility score."""
        
        # Simplified SA score based on molecular complexity
        rings = mol.GetRingInfo().NumRings()
        stereocenters = sum(1 for atom in mol.GetAtoms() 
                          if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED)
        
        # Heteroatoms add complexity
        heteroatoms = sum(1 for atom in mol.GetAtoms() 
                         if atom.GetAtomicNum() not in [1, 6])
        
        complexity = rings * 0.3 + stereocenters * 0.5 + heteroatoms * 0.2
        
        # Convert to 1-10 scale (lower is easier to synthesize)
        sa_score = min(1 + complexity, 10)
        
        return 11 - sa_score  # Invert so higher is better
    
    def count_toxicity_alerts(self, mol):
        """Count potential toxicity alerts."""
        
        # Common toxicophores (simplified)
        toxicophores = [
            'N=O',      # Nitro groups
            '[N+](=O)[O-]',  # Nitro groups
            'S(=O)(=O)',     # Sulfonyl
            'C#N',           # Nitriles
            '[As]',          # Arsenic
            '[Hg]',          # Mercury
            '[Pb]'           # Lead
        ]
        
        smiles = Chem.MolToSmiles(mol)
        alerts = sum(1 for tox in toxicophores if tox in smiles)
        
        return alerts
    
    def predict_metabolic_stability(self, mol):
        """Predict metabolic stability."""
        
        # Factors affecting metabolic stability
        aromatic_rings = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic()) / 6
        aliphatic_ch = sum(1 for atom in mol.GetAtoms() 
                          if atom.GetAtomicNum() == 6 and not atom.GetIsAromatic())
        
        # Aromatic rings are generally more stable
        # Aliphatic carbons are more susceptible to metabolism
        stability_score = aromatic_rings * 0.3 - aliphatic_ch * 0.1
        
        return max(0, min(1, 0.5 + stability_score))  # 0-1 score

# Register specialized descriptors
polymer_desc = PolymerDescriptor()
drug_desc = DrugLikenessDescriptor()

calc.register_custom_descriptor(polymer_desc)
calc.register_custom_descriptor(drug_desc)

# Test on specialized molecules
polymer_molecules = [
    "C=CC(=O)OC",           # Methyl acrylate (polymer precursor)
    "c1ccc(C=C)cc1",        # Styrene
    "C=CC(C)(C)C(=O)OC"     # Methyl methacrylate
]

drug_molecules = [
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",  # Ibuprofen
    "CC(=O)Oc1ccccc1C(=O)O",        # Aspirin
    "CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12"  # Chloroquine
]

polymer_results = calc.pandas(polymer_molecules, descriptors=["polymer_properties"])
drug_results = calc.pandas(drug_molecules, descriptors=["drug_likeness_profile"])

print("Polymer analysis results:")
print(polymer_results)
print("\nDrug-likeness analysis results:")
print(drug_results)
```

## Testing and Validation Framework

### Descriptor Testing Suite

```python
import unittest
from chemi_accel.descriptors.testing import DescriptorTestSuite

class CustomDescriptorTests(unittest.TestCase):
    """Test suite for custom descriptors."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_molecules = [
            "CCO",              # Ethanol
            "c1ccccc1",         # Benzene
            "CC(=O)O",          # Acetic acid
            "invalid_smiles",   # Invalid SMILES for error testing
            None                # None molecule
        ]
        
        self.descriptor = AromaticAtomCount()
        self.test_suite = DescriptorTestSuite()
    
    def test_valid_molecules(self):
        """Test descriptor on valid molecules."""
        
        valid_molecules = self.test_molecules[:3]
        expected_results = [0, 6, 0]  # Expected aromatic atom counts
        
        for mol_smiles, expected in zip(valid_molecules, expected_results):
            mol = Chem.MolFromSmiles(mol_smiles)
            result = self.descriptor.calculate(mol)
            
            self.assertEqual(result, expected, 
                           f"Failed for {mol_smiles}: expected {expected}, got {result}")
    
    def test_invalid_molecules(self):
        """Test descriptor on invalid molecules."""
        
        # Test with None
        result = self.descriptor.calculate(None)
        self.assertEqual(result, 0)
        
        # Test with invalid SMILES
        invalid_mol = Chem.MolFromSmiles("invalid_smiles")
        result = self.descriptor.calculate(invalid_mol)
        self.assertEqual(result, 0)
    
    def test_batch_processing(self):
        """Test batch processing if supported."""
        
        if hasattr(self.descriptor, 'batch_calculate'):
            mols = [Chem.MolFromSmiles(smi) if smi and smi != "invalid_smiles" else None 
                   for smi in self.test_molecules]
            
            batch_results = self.descriptor.batch_calculate(mols)
            
            # Compare with individual calculations
            individual_results = [self.descriptor.calculate(mol) for mol in mols]
            
            self.assertEqual(len(batch_results), len(individual_results))
            
            for batch_result, individual_result in zip(batch_results, individual_results):
                if np.isnan(batch_result) and np.isnan(individual_result):
                    continue
                self.assertAlmostEqual(batch_result, individual_result, places=5)
    
    def test_performance(self):
        """Test performance with large dataset."""
        
        import time
        
        # Generate large test dataset
        large_dataset = ["CCO", "c1ccccc1", "CC(=O)O"] * 1000
        mols = [Chem.MolFromSmiles(smi) for smi in large_dataset]
        
        start_time = time.time()
        
        if hasattr(self.descriptor, 'batch_calculate'):
            results = self.descriptor.batch_calculate(mols)
        else:
            results = [self.descriptor.calculate(mol) for mol in mols]
        
        end_time = time.time()
        
        throughput = len(mols) / (end_time - start_time)
        
        print(f"Descriptor throughput: {throughput:.1f} molecules/second")
        
        # Performance should be reasonable (>100 molecules/second for simple descriptors)
        self.assertGreater(throughput, 100, 
                          f"Performance too slow: {throughput:.1f} mol/s")
    
    def test_reproducibility(self):
        """Test that results are reproducible."""
        
        mol = Chem.MolFromSmiles("c1ccccc1")
        
        # Calculate same molecule multiple times
        results = [self.descriptor.calculate(mol) for _ in range(10)]
        
        # All results should be identical
        self.assertTrue(all(r == results[0] for r in results),
                       "Results are not reproducible")
    
    def test_data_types(self):
        """Test correct data types are returned."""
        
        mol = Chem.MolFromSmiles("CCO")
        result = self.descriptor.calculate(mol)
        
        if self.descriptor.return_type == int:
            self.assertIsInstance(result, int)
        elif self.descriptor.return_type == float:
            self.assertIsInstance(result, (int, float))
        elif self.descriptor.return_type == dict:
            self.assertIsInstance(result, dict)
            
            # Check that all expected columns are present
            if hasattr(self.descriptor, 'output_columns'):
                for col in self.descriptor.output_columns:
                    self.assertIn(col, result)

def run_comprehensive_descriptor_tests(descriptor_class):
    """Run comprehensive tests on a custom descriptor."""
    
    print(f"Testing {descriptor_class.__name__}...")
    
    # Create test instance
    descriptor = descriptor_class()
    
    # Run standard test suite
    test_suite = unittest.TestSuite()
    
    # Add test methods
    test_methods = [
        'test_valid_molecules',
        'test_invalid_molecules', 
        'test_batch_processing',
        'test_performance',
        'test_reproducibility',
        'test_data_types'
    ]
    
    for test_method in test_methods:
        test_suite.addTest(CustomDescriptorTests(test_method))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    if result.wasSuccessful():
        print(f"✓ All tests passed for {descriptor_class.__name__}")
        return True
    else:
        print(f"✗ {len(result.failures)} test(s) failed for {descriptor_class.__name__}")
        return False

# Example: Test custom descriptors
test_descriptors = [
    AromaticAtomCount,
    MolecularComplexityProfile,
    PolymerDescriptor,
    DrugLikenessDescriptor
]

for descriptor_class in test_descriptors:
    success = run_comprehensive_descriptor_tests(descriptor_class)
    print()
```

## Integration and Deployment

### Descriptor Package Creation

```python
# Create installable descriptor package
def create_descriptor_package(descriptor_classes, package_name):
    """Create installable package for custom descriptors."""
    
    package_structure = f"""
{package_name}/
├── __init__.py
├── descriptors.py
├── tests/
│   ├── __init__.py
│   └── test_descriptors.py
├── setup.py
└── README.md
"""
    
    # Generate __init__.py
    init_content = f'''"""
{package_name} - Custom ChemiAccel Descriptors
"""

from .descriptors import {", ".join([cls.__name__ for cls in descriptor_classes])}

__version__ = "1.0.0"
__all__ = {[cls.__name__ for cls in descriptor_classes]}
'''
    
    # Generate setup.py
    setup_content = f'''
from setuptools import setup, find_packages

setup(
    name="{package_name}",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Custom descriptors for ChemiAccel",
    packages=find_packages(),
    install_requires=[
        "chemi-accel>=1.0.0",
        "rdkit-pypi>=2022.3.1",
        "numpy>=1.21.0"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
'''
    
    print(f"Package structure for {package_name}:")
    print(package_structure)
    print("Generated files: __init__.py, setup.py")
    
    return {
        'init_content': init_content,
        'setup_content': setup_content,
        'package_structure': package_structure
    }

# Example: Create package
custom_descriptors = [
    AromaticAtomCount,
    MolecularComplexityProfile,
    PolymerDescriptor,
    DrugLikenessDescriptor
]

package_info = create_descriptor_package(custom_descriptors, "my_custom_descriptors")
```

### Production Deployment

```python
class ProductionDescriptorManager:
    """Manager for production deployment of custom descriptors."""
    
    def __init__(self):
        self.registered_descriptors = {}
        self.performance_cache = {}
        self.validation_results = {}
    
    def register_descriptor(self, descriptor_class, validation_required=True):
        """Register descriptor with production validation."""
        
        descriptor = descriptor_class()
        
        if validation_required:
            validation_result = self.validate_descriptor(descriptor)
            
            if not validation_result['is_valid']:
                raise ValueError(f"Descriptor validation failed: {validation_result['errors']}")
            
            self.validation_results[descriptor.name] = validation_result
        
        self.registered_descriptors[descriptor.name] = descriptor
        
        print(f"✓ Registered descriptor: {descriptor.name}")
    
    def validate_descriptor(self, descriptor):
        """Comprehensive descriptor validation."""
        
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'performance_metrics': {}
        }
        
        # Check required attributes
        required_attrs = ['name', 'description', 'return_type']
        for attr in required_attrs:
            if not hasattr(descriptor, attr):
                validation_result['errors'].append(f"Missing required attribute: {attr}")
        
        # Performance validation
        try:
            test_molecules = ["CCO", "c1ccccc1", "CC(=O)O"] * 100
            mols = [Chem.MolFromSmiles(smi) for smi in test_molecules]
            
            import time
            start_time = time.time()
            
            if hasattr(descriptor, 'batch_calculate'):
                results = descriptor.batch_calculate(mols)
            else:
                results = [descriptor.calculate(mol) for mol in mols]
            
            end_time = time.time()
            
            throughput = len(mols) / (end_time - start_time)
            validation_result['performance_metrics']['throughput'] = throughput
            
            if throughput < 10:  # Less than 10 molecules/second
                validation_result['warnings'].append("Low performance: consider optimization")
            
        except Exception as e:
            validation_result['errors'].append(f"Performance test failed: {e}")
        
        # Result validation
        try:
            test_mol = Chem.MolFromSmiles("CCO")
            result = descriptor.calculate(test_mol)
            
            # Check result type
            expected_type = descriptor.return_type
            if expected_type == dict:
                if not isinstance(result, dict):
                    validation_result['errors'].append("Expected dict return type")
            elif expected_type in [int, float]:
                if not isinstance(result, (int, float)) or np.isnan(result):
                    validation_result['errors'].append("Invalid numeric result")
        
        except Exception as e:
            validation_result['errors'].append(f"Result validation failed: {e}")
        
        validation_result['is_valid'] = len(validation_result['errors']) == 0
        
        return validation_result
    
    def get_descriptor_info(self, descriptor_name):
        """Get comprehensive descriptor information."""
        
        if descriptor_name not in self.registered_descriptors:
            return None
        
        descriptor = self.registered_descriptors[descriptor_name]
        validation = self.validation_results.get(descriptor_name, {})
        
        info = {
            'name': descriptor.name,
            'description': descriptor.description,
            'return_type': descriptor.return_type.__name__,
            'supports_batch': hasattr(descriptor, 'batch_calculate'),
            'supports_cuda': getattr(descriptor, 'supports_cuda', False),
            'validation_passed': validation.get('is_valid', False),
            'performance_metrics': validation.get('performance_metrics', {}),
            'output_columns': getattr(descriptor, 'output_columns', None)
        }
        
        return info
    
    def list_descriptors(self):
        """List all registered descriptors."""
        
        descriptor_list = []
        
        for name in self.registered_descriptors:
            info = self.get_descriptor_info(name)
            descriptor_list.append(info)
        
        return descriptor_list

# Production deployment example
production_manager = ProductionDescriptorManager()

# Register descriptors with validation
descriptors_to_deploy = [
    AromaticAtomCount,
    MolecularComplexityProfile,
    OptimizedDescriptor
]

for desc_class in descriptors_to_deploy:
    try:
        production_manager.register_descriptor(desc_class, validation_required=True)
    except ValueError as e:
        print(f"Failed to register {desc_class.__name__}: {e}")

# List deployed descriptors
deployed_descriptors = production_manager.list_descriptors()
print("\nDeployed descriptors:")
for desc_info in deployed_descriptors:
    print(f"  {desc_info['name']}: {desc_info['description']}")
    print(f"    Performance: {desc_info['performance_metrics'].get('throughput', 'N/A')} mol/s")
```

## Best Practices

### 1. Performance Optimization
```python
# Use vectorized operations where possible
def optimized_calculation(mols):
    # Pre-allocate arrays
    results = np.zeros(len(mols))
    
    # Batch similar operations
    for i, mol in enumerate(mols):
        results[i] = fast_calculation(mol)
    
    return results
```

### 2. Error Handling
```python
def robust_descriptor_calculate(self, mol):
    try:
        if mol is None:
            return self.default_value
        
        result = self.core_calculation(mol)
        
        # Validate result
        if not self.is_valid_result(result):
            return self.default_value
        
        return result
    
    except Exception as e:
        self.log_error(f"Calculation failed: {e}")
        return self.default_value
```

### 3. Testing Strategy
```python
# Comprehensive testing approach
def test_descriptor_thoroughly():
    test_cases = [
        ("simple_molecules", ["CCO", "CH4"]),
        ("complex_molecules", ["complex_drug_molecules"]),
        ("edge_cases", [None, "invalid_smiles"]),
        ("large_dataset", ["CCO"] * 10000)
    ]
    
    for test_name, molecules in test_cases:
        run_test_suite(test_name, molecules)
```

---

*For integration with existing ChemiAccel workflows, see the [Basic Usage](Basic-Usage) and [API Reference](API-Reference) guides.*
