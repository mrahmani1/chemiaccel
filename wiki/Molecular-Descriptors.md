# Molecular Descriptors Guide

This comprehensive guide covers all molecular descriptors available in ChemiAccel, their scientific meaning, computational methods, and practical applications.

## Overview

ChemiAccel provides over 200 molecular descriptors organized into categories based on their computational origin and chemical meaning. All descriptors are GPU-accelerated and can be calculated in batch mode for high-throughput applications.

## Descriptor Categories

### Constitutional Descriptors

Basic molecular constitution and connectivity descriptors.

```python
from chemi_accel import Calculator

calc = Calculator()
molecules = ["CCO", "c1ccccc1", "CC(=O)O"]

constitutional = calc.pandas(
    molecules,
    descriptors=[
        "molecular_weight",      # Molecular weight
        "num_atoms",            # Number of atoms
        "num_heavy_atoms",      # Number of non-hydrogen atoms
        "num_bonds",            # Number of bonds
        "num_single_bonds",     # Number of single bonds
        "num_double_bonds",     # Number of double bonds
        "num_triple_bonds",     # Number of triple bonds
        "num_aromatic_bonds",   # Number of aromatic bonds
        "num_rings",            # Number of rings
        "num_aromatic_rings",   # Number of aromatic rings
        "num_saturated_rings",  # Number of saturated rings
        "num_aliphatic_rings",  # Number of aliphatic rings
    ]
)
```

#### Applications
- **Drug discovery**: Molecular weight filtering (Rule of Five)
- **QSAR modeling**: Basic structural features
- **Chemical space analysis**: Molecular complexity assessment

### Physicochemical Descriptors

Properties related to molecular interactions and solubility.

```python
physicochemical = calc.pandas(
    molecules,
    descriptors=[
        "logp",                 # Octanol-water partition coefficient
        "slogp",               # Crippen LogP
        "mlogp",               # Moriguchi LogP
        "tpsa",                # Topological polar surface area
        "asa",                 # Accessible surface area
        "polar_surface_area",  # Polar surface area
        "molecular_volume",    # van der Waals volume
        "density",             # Molecular density
        "polarizability",      # Molecular polarizability
        "refractivity",        # Molar refractivity
        "dipole_moment",       # Dipole moment
    ]
)
```

#### Scientific Background
- **LogP**: Measures lipophilicity, crucial for membrane permeability
- **TPSA**: Correlates with drug absorption and blood-brain barrier penetration
- **Polarizability**: Related to London dispersion forces and molecular interactions

### Electronic Descriptors

Quantum mechanical and electronic properties.

```python
electronic = calc.pandas(
    molecules,
    descriptors=[
        "homo_energy",         # HOMO energy level
        "lumo_energy",         # LUMO energy level
        "homo_lumo_gap",       # HOMO-LUMO gap
        "ionization_potential", # Ionization potential
        "electron_affinity",   # Electron affinity
        "electronegativity",   # Electronegativity
        "hardness",            # Chemical hardness
        "softness",            # Chemical softness
        "electrophilicity",    # Electrophilicity index
        "nucleophilicity",     # Nucleophilicity index
    ]
)
```

#### Computational Methods
- **Semi-empirical**: Fast approximations (AM1, PM3, PM6)
- **DFT**: Density functional theory calculations
- **Machine learning**: Trained predictors for rapid estimation

### Topological Descriptors

Graph-based descriptors from molecular topology.

```python
topological = calc.pandas(
    molecules,
    descriptors=[
        "wiener_index",        # Wiener index
        "zagreb1",             # First Zagreb index
        "zagreb2",             # Second Zagreb index
        "randic_index",        # Randić connectivity index
        "kappa1",              # First kappa shape index
        "kappa2",              # Second kappa shape index
        "kappa3",              # Third kappa shape index
        "chi0",                # Chi-0 connectivity index
        "chi1",                # Chi-1 connectivity index
        "chi2",                # Chi-2 connectivity index
        "balaban_j",           # Balaban J index
        "petitjean_number",    # Petitjean shape index
    ]
)
```

#### Mathematical Foundation
These descriptors are derived from graph theory applied to molecular structure:
- **Connectivity indices**: Based on vertex degrees
- **Shape indices**: Encode molecular shape information
- **Distance-based indices**: Use shortest path distances

### Geometric Descriptors (3D)

Three-dimensional shape and spatial descriptors.

```python
geometric_3d = calc.pandas(
    molecules,
    descriptors=[
        "radius_of_gyration",   # Radius of gyration
        "asphericity",         # Asphericity parameter
        "eccentricity",        # Eccentricity
        "inertia_moment_x",    # Principal moment of inertia X
        "inertia_moment_y",    # Principal moment of inertia Y
        "inertia_moment_z",    # Principal moment of inertia Z
        "surface_area",        # Molecular surface area
        "volume",              # Molecular volume
        "ovality",             # Ovality descriptor
        "sphericity",          # Sphericity index
    ],
    generate_conformers=True  # Required for 3D descriptors
)
```

#### Conformer Generation
```python
# Configure conformer generation
calc = Calculator(
    conformer_config={
        "num_conformers": 50,
        "energy_window": 10.0,  # kcal/mol
        "rms_threshold": 0.5,   # Å
        "optimize": True,
        "force_field": "MMFF94"
    }
)
```

### Fingerprint Descriptors

Binary or count-based molecular fingerprints.

```python
fingerprints = calc.pandas(
    molecules,
    descriptors=[
        "morgan_fingerprint_1024", # Morgan fingerprint (1024 bits)
        "morgan_fingerprint_2048", # Morgan fingerprint (2048 bits)
        "maccs_keys",             # MACCS structural keys
        "atom_pair_fingerprint",  # Atom pair fingerprint
        "topological_torsion",    # Topological torsion fingerprint
        "avalon_fingerprint",     # Avalon fingerprint
        "rdkit_fingerprint",      # RDKit structural fingerprint
        "pattern_fingerprint",    # Substructure patterns
    ]
)
```

#### Fingerprint Types

**Morgan Fingerprints (ECFP)**
```python
# Different radius and bit sizes
morgan_configs = [
    ("morgan_fingerprint_1024_r2", {"radius": 2, "n_bits": 1024}),
    ("morgan_fingerprint_2048_r3", {"radius": 3, "n_bits": 2048}),
    ("morgan_fingerprint_4096_r4", {"radius": 4, "n_bits": 4096}),
]

for desc_name, config in morgan_configs:
    fp_results = calc.pandas(molecules, descriptors=[desc_name])
```

**MACCS Keys**
```python
# 166-bit structural keys
maccs_results = calc.pandas(molecules, descriptors=["maccs_keys"])
print(f"MACCS keys shape: {maccs_results.shape}")  # Should be (n_molecules, 166)
```

### Pharmacophore Descriptors

Descriptors relevant to drug design and pharmacology.

```python
pharmacophore = calc.pandas(
    molecules,
    descriptors=[
        "num_hbd",             # Hydrogen bond donors
        "num_hba",             # Hydrogen bond acceptors
        "num_rotatable_bonds", # Rotatable bonds
        "num_rigid_bonds",     # Rigid bonds
        "formal_charge",       # Formal charge
        "num_radical_electrons", # Radical electrons
        "num_valence_electrons", # Valence electrons
        "num_heteroatoms",     # Heteroatoms
        "fraction_csp3",       # Fraction of C sp3
        "num_saturated_carbocycles", # Saturated carbocycles
        "num_saturated_heterocycles", # Saturated heterocycles
        "num_aliphatic_carbocycles",  # Aliphatic carbocycles
        "num_aliphatic_heterocycles", # Aliphatic heterocycles
    ]
)
```

#### Drug-likeness Filters
```python
def apply_lipinski_rule_of_five(df):
    """Apply Lipinski's Rule of Five filter."""
    return df[
        (df["molecular_weight"] <= 500) &
        (df["logp"] <= 5) &
        (df["num_hbd"] <= 5) &
        (df["num_hba"] <= 10)
    ]

def apply_veber_rules(df):
    """Apply Veber rules for oral bioavailability."""
    return df[
        (df["num_rotatable_bonds"] <= 10) &
        (df["tpsa"] <= 140)
    ]

# Apply filters
drug_like = apply_lipinski_rule_of_five(pharmacophore)
oral_bioavailable = apply_veber_rules(drug_like)
```

## Advanced Descriptor Categories

### Quantum Chemical Descriptors

High-level quantum mechanical properties.

```python
quantum = calc.pandas(
    molecules,
    descriptors=[
        "atomic_charges",      # Partial atomic charges
        "bond_orders",         # Bond orders
        "fukui_indices",       # Fukui indices
        "local_softness",      # Local softness
        "dual_descriptor",     # Dual descriptor
        "molecular_electrostatic_potential", # MEP
        "electron_density",    # Electron density at nuclei
        "laplacian_electron_density", # Laplacian of electron density
    ],
    quantum_method="PM6"  # Quantum method: PM6, AM1, B3LYP, etc.
)
```

### Spectroscopic Descriptors

Descriptors related to spectroscopic properties.

```python
spectroscopic = calc.pandas(
    molecules,
    descriptors=[
        "ir_spectrum",         # Infrared spectrum features
        "nmr_spectrum",        # NMR spectrum features  
        "uv_vis_spectrum",     # UV-Vis spectrum features
        "vibrational_frequencies", # Vibrational frequencies
        "raman_spectrum",      # Raman spectrum features
    ]
)
```

### Biological Activity Descriptors

Descriptors related to biological activity and ADMET properties.

```python
biological = calc.pandas(
    molecules,
    descriptors=[
        "bioavailability_score", # Oral bioavailability
        "bbb_permeability",     # Blood-brain barrier permeability
        "pgp_substrate",        # P-glycoprotein substrate
        "cyp_inhibition",       # Cytochrome P450 inhibition
        "herg_blocking",        # hERG channel blocking
        "plasma_protein_binding", # Plasma protein binding
        "volume_distribution",  # Volume of distribution
        "clearance",           # Hepatic clearance
        "half_life",           # Elimination half-life
        "toxicity_score",      # General toxicity score
    ]
)
```

## Custom Descriptor Calculation

### Define Custom Descriptors

```python
from chemi_accel.descriptors.custom import CustomDescriptor
from rdkit import Chem
import numpy as np

class AromaticRatio(CustomDescriptor):
    """Custom descriptor: ratio of aromatic atoms to total atoms."""
    
    name = "aromatic_ratio"
    
    def calculate(self, mol):
        if mol is None:
            return np.nan
        
        num_aromatic = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        num_total = mol.GetNumAtoms()
        
        return num_aromatic / num_total if num_total > 0 else 0.0

# Register custom descriptor
calc.register_custom_descriptor(AromaticRatio())

# Use custom descriptor
custom_results = calc.pandas(molecules, descriptors=["aromatic_ratio"])
```

### Batch Custom Descriptors

```python
class CustomDescriptorSet:
    """Set of custom descriptors."""
    
    @staticmethod
    def nitrogen_ratio(mol):
        """Ratio of nitrogen atoms to total atoms."""
        if mol is None:
            return np.nan
        
        num_nitrogen = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7)
        num_total = mol.GetNumAtoms()
        return num_nitrogen / num_total if num_total > 0 else 0.0
    
    @staticmethod
    def oxygen_ratio(mol):
        """Ratio of oxygen atoms to total atoms."""
        if mol is None:
            return np.nan
        
        num_oxygen = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8)
        num_total = mol.GetNumAtoms()
        return num_oxygen / num_total if num_total > 0 else 0.0

# Register multiple custom descriptors
custom_set = CustomDescriptorSet()
calc.register_custom_descriptors({
    "nitrogen_ratio": custom_set.nitrogen_ratio,
    "oxygen_ratio": custom_set.oxygen_ratio
})
```

## Performance Optimization

### Backend-Specific Optimizations

```python
# CUDA backend optimization
cuda_calc = Calculator(
    backend="cuda",
    batch_size=5000,      # Larger batches for GPU
    memory_pool_mb=8192,  # Pre-allocate GPU memory
    streams=4             # Multiple CUDA streams
)

# JAX backend optimization
jax_calc = Calculator(
    backend="jax",
    jit_compile=True,     # JIT compilation
    batch_size=2000,
    precision="float32"   # Use 32-bit precision
)

# CPU backend optimization
cpu_calc = Calculator(
    backend="cpu",
    num_threads=16,       # Use all CPU cores
    batch_size=1000,
    enable_mkl=True       # Intel MKL acceleration
)
```

### Descriptor-Specific Optimizations

```python
# Fast descriptors only (< 1ms per molecule)
fast_descriptors = [
    "molecular_weight", "num_atoms", "num_bonds", "num_rings",
    "logp", "tpsa", "num_hbd", "num_hba"
]

# Medium speed descriptors (1-10ms per molecule)
medium_descriptors = [
    "morgan_fingerprint_2048", "maccs_keys", "topological_descriptors"
]

# Slow descriptors (> 10ms per molecule)
slow_descriptors = [
    "3d_descriptors", "quantum_descriptors", "spectroscopic_descriptors"
]

# Calculate in order of increasing computational cost
results_fast = calc.pandas(molecules, descriptors=fast_descriptors)
results_medium = calc.pandas(molecules, descriptors=medium_descriptors)
results_slow = calc.pandas(molecules, descriptors=slow_descriptors)

# Combine results
all_results = pd.concat([results_fast, results_medium, results_slow], axis=1)
```

## Descriptor Selection and Analysis

### Statistical Analysis

```python
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.feature_selection import SelectKBest, f_regression

# Calculate comprehensive descriptors
all_descriptors = calc.pandas(molecules, descriptors="all")

# Remove constant descriptors
constant_mask = all_descriptors.var() > 1e-8
filtered_descriptors = all_descriptors.loc[:, constant_mask]

# Correlation analysis
correlation_matrix = filtered_descriptors.corr()

# Find highly correlated pairs
def find_high_correlations(corr_matrix, threshold=0.95):
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))
    return high_corr_pairs

high_corr = find_high_correlations(correlation_matrix)
print(f"Found {len(high_corr)} highly correlated descriptor pairs")
```

### Feature Selection

```python
# Assume we have target values
target_values = load_target_values()  # Your experimental data

# Select best descriptors for prediction
selector = SelectKBest(score_func=f_regression, k=50)
X_selected = selector.fit_transform(filtered_descriptors, target_values)

# Get selected descriptor names
selected_mask = selector.get_support()
selected_descriptors = filtered_descriptors.columns[selected_mask].tolist()

print(f"Selected descriptors: {selected_descriptors}")
```

### Descriptor Importance Analysis

```python
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Train random forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(filtered_descriptors, target_values)

# Get feature importances
importances = rf_model.feature_importances_
descriptor_names = filtered_descriptors.columns

# Sort by importance
importance_df = pd.DataFrame({
    'descriptor': descriptor_names,
    'importance': importances
}).sort_values('importance', ascending=False)

# Plot top 20 most important descriptors
plt.figure(figsize=(12, 8))
plt.barh(range(20), importance_df['importance'].head(20))
plt.yticks(range(20), importance_df['descriptor'].head(20))
plt.xlabel('Feature Importance')
plt.title('Top 20 Most Important Molecular Descriptors')
plt.tight_layout()
plt.show()
```

## Validation and Quality Control

### Descriptor Validation

```python
def validate_descriptors(df, molecules):
    """Validate calculated descriptors for quality."""
    validation_report = {
        'total_molecules': len(molecules),
        'successful_calculations': len(df),
        'missing_values': df.isnull().sum().sum(),
        'infinite_values': np.isinf(df.select_dtypes(include=[np.number])).sum().sum(),
        'constant_descriptors': (df.var() == 0).sum(),
        'descriptor_ranges': {}
    }
    
    # Check descriptor ranges
    for col in df.select_dtypes(include=[np.number]).columns:
        validation_report['descriptor_ranges'][col] = {
            'min': df[col].min(),
            'max': df[col].max(),
            'mean': df[col].mean(),
            'std': df[col].std()
        }
    
    return validation_report

# Validate results
validation = validate_descriptors(all_descriptors, molecules)
print(f"Validation report: {validation}")
```

### Reference Data Comparison

```python
def compare_with_reference(calculated_df, reference_df, tolerance=0.01):
    """Compare calculated descriptors with reference values."""
    common_descriptors = set(calculated_df.columns) & set(reference_df.columns)
    
    comparison_results = {}
    for descriptor in common_descriptors:
        calc_values = calculated_df[descriptor]
        ref_values = reference_df[descriptor]
        
        # Calculate metrics
        mae = np.mean(np.abs(calc_values - ref_values))
        mse = np.mean((calc_values - ref_values) ** 2)
        correlation, _ = pearsonr(calc_values, ref_values)
        
        # Check if within tolerance
        within_tolerance = np.mean(np.abs(calc_values - ref_values) <= tolerance)
        
        comparison_results[descriptor] = {
            'mae': mae,
            'mse': mse,
            'correlation': correlation,
            'within_tolerance': within_tolerance
        }
    
    return comparison_results

# Compare with reference data (if available)
# reference_data = load_reference_data()
# comparison = compare_with_reference(all_descriptors, reference_data)
```

## Best Practices

### 1. Choose Appropriate Descriptors

```python
# For drug discovery
drug_descriptors = [
    "molecular_weight", "logp", "tpsa", "num_hbd", "num_hba",
    "num_rotatable_bonds", "morgan_fingerprint_2048"
]

# For QSAR modeling
qsar_descriptors = [
    "constitutional", "topological", "physicochemical", 
    "electronic", "geometric_3d"
]

# For similarity searching
similarity_descriptors = [
    "morgan_fingerprint_2048", "maccs_keys", "atom_pair_fingerprint"
]
```

### 2. Handle Missing Values

```python
# Strategy 1: Remove molecules with missing descriptors
clean_data = all_descriptors.dropna()

# Strategy 2: Impute missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
imputed_data = pd.DataFrame(
    imputer.fit_transform(all_descriptors),
    columns=all_descriptors.columns
)

# Strategy 3: Use descriptors with minimal missing values
missing_counts = all_descriptors.isnull().sum()
reliable_descriptors = missing_counts[missing_counts < len(all_descriptors) * 0.05].index
reliable_data = all_descriptors[reliable_descriptors]
```

### 3. Scale Descriptors Appropriately

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Standardization (mean=0, std=1)
scaler = StandardScaler()
standardized_data = pd.DataFrame(
    scaler.fit_transform(all_descriptors),
    columns=all_descriptors.columns
)

# Min-max scaling (0-1 range)
minmax_scaler = MinMaxScaler()
normalized_data = pd.DataFrame(
    minmax_scaler.fit_transform(all_descriptors),
    columns=all_descriptors.columns
)

# Robust scaling (less sensitive to outliers)
robust_scaler = RobustScaler()
robust_data = pd.DataFrame(
    robust_scaler.fit_transform(all_descriptors),
    columns=all_descriptors.columns
)
```

---

*For specific applications, see the domain-specific guides: [Drug Discovery](ADMET-Prediction), [Polymer Science](Polymer-Science), and [Materials Science](Materials-Science).*
