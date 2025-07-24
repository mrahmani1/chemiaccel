# 3D Geometrical and Shape Descriptors

ChemiAccel provides comprehensive 3D molecular descriptors for analyzing molecular shape, electrostatic properties, and geometric characteristics.

## Overview

The 3D descriptor module includes:

- **Conformer Generation**: Generate optimized 3D conformers using distance geometry and force field optimization
- **Shape Descriptors**: Calculate molecular volume, surface area, geometric diameter, and shape anisotropy
- **Electrostatic Descriptors**: Compute partial charges, dipole moments, and electrostatic interactions
- **Comprehensive Analysis**: Combined 3D analysis with batch processing capabilities

## Key Features

### Conformer Generation
- Multiple conformer generation with pruning
- Force field optimization (UFF, MMFF)
- Automatic fallback for difficult molecules
- Batch processing support

### Shape Descriptors
- **Radius of Gyration**: Measure of molecular size distribution
- **Asphericity**: Deviation from spherical shape (0=sphere, 1=linear)
- **Eccentricity**: Shape elongation measure
- **Molecular Volume**: Van der Waals volume calculation
- **Surface Area**: Molecular surface area estimation
- **Geometric Diameter**: Maximum inter-atomic distance
- **Compactness**: Volume-to-surface-area ratio

### Electrostatic Descriptors
- **Partial Charges**: Gasteiger or MMFF charge models
- **Dipole Moment**: Molecular polarity measure
- **Charge Distribution**: Statistics of charge distribution
- **Electrostatic Energy**: Coulombic interaction energies

## Usage Examples

### Basic 3D Descriptor Calculation

```python
from chemi_accel.descriptors.geometrical_3d import calculate_3d_descriptors
from chemi_accel.utils.rdkit_integration import parse_smiles_batch

# Parse molecules
molecules = parse_smiles_batch(["CCO", "c1ccccc1", "CC(C)C"])

# Calculate comprehensive 3D descriptors
results = calculate_3d_descriptors(
    molecules,
    include_shape=True,
    include_electrostatic=True,
    generate_conformers=True,
    num_conformers=5
)

print(results[["smiles", "molecular_volume", "dipole_moment", "asphericity"]])
```

### Conformer Generation

```python
from chemi_accel.descriptors.geometrical_3d import ConformerGenerator

# Initialize conformer generator
generator = ConformerGenerator(
    num_conformers=10,
    optimize_conformers=True,
    force_field="UFF"
)

# Generate conformers
molecules_with_conformers = generator.generate_conformers_batch(molecules)

# Check conformer counts
for mol in molecules_with_conformers:
    rdkit_mol = mol.backend_data.get("rdkit_mol")
    if rdkit_mol:
        print(f"{mol.smiles}: {rdkit_mol.GetNumConformers()} conformers")
```

### Shape Analysis

```python
from chemi_accel.descriptors.geometrical_3d import Shape3DDescriptors

calculator = Shape3DDescriptors()

# Calculate specific shape descriptors
shape_results = calculator.calculate_shape_descriptors_batch(
    molecules_with_conformers,
    descriptor_names=["molecular_volume", "asphericity", "geometric_diameter"]
)

print(shape_results)
```

### Electrostatic Analysis

```python
from chemi_accel.descriptors.geometrical_3d import ElectrostaticDescriptors

calculator = ElectrostaticDescriptors(charge_model="gasteiger")

# Calculate electrostatic properties
for mol in molecules_with_conformers:
    if mol.backend_data.get("rdkit_mol").GetNumConformers() > 0:
        descriptors = calculator.calculate_electrostatic_descriptors(mol)
        print(f"{mol.smiles}: dipole={descriptors['dipole_moment']:.2f}D")
```

## Performance Characteristics

### Throughput
- **Conformer Generation**: ~10-50 molecules/second (depends on complexity)
- **Shape Descriptors**: ~30-100 molecules/second
- **Electrostatic Descriptors**: ~100-500 molecules/second

### Scalability
- Batch processing optimized for large datasets
- Memory-efficient conformer handling
- Parallel processing support for independent calculations

## Descriptor Interpretation

### Shape Descriptors
- **Asphericity < 0.1**: Nearly spherical molecules
- **Asphericity > 0.5**: Highly elongated molecules
- **Radius of Gyration**: Larger values indicate more extended conformations
- **Compactness**: Higher values suggest more compact, globular shapes

### Electrostatic Properties
- **Dipole Moment < 1D**: Nonpolar molecules
- **Dipole Moment 1-3D**: Moderately polar molecules
- **Dipole Moment > 3D**: Highly polar molecules
- **Total Charge**: Should be near zero for neutral molecules

## Validation and Accuracy

The 3D descriptors are validated against:
- RDKit's built-in 3D descriptor implementations
- Literature values for known molecules
- Cross-validation with other cheminformatics tools

Key validation metrics:
- Numerical accuracy within 1% of reference implementations
- Charge conservation for electrostatic descriptors
- Reasonable descriptor ranges for diverse molecular sets

## Integration with ChemiAccel

3D descriptors integrate seamlessly with other ChemiAccel modules:
- Use with fingerprint generation for enhanced molecular representations
- Combine with 2D descriptors for comprehensive molecular profiling
- Integration with CUDA/JAX backends for GPU acceleration (future)

## Limitations and Considerations

- Conformer generation can be time-consuming for large, flexible molecules
- Surface area calculations use approximations for speed
- Some descriptors require high-quality 3D coordinates
- Memory usage scales with number of conformers and molecular size

## Future Enhancements

Planned improvements include:
- GPU-accelerated conformer generation
- Advanced surface area calculations using FreeSASA
- Integration with quantum chemistry descriptors
- Enhanced polymer-specific 3D descriptors