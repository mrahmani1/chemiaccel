# Basic Usage Guide

This guide introduces the fundamental concepts and workflows for using ChemiAccel to calculate molecular descriptors and perform computational chemistry tasks.

## Getting Started

### Your First Calculation

```python
from chemi_accel import Calculator

# Create a calculator instance
calc = Calculator()

# Define some molecules (SMILES format)
molecules = [
    "CCO",           # Ethanol
    "CC(=O)O",       # Acetic acid
    "c1ccccc1",      # Benzene
    "CCN(CC)CC"      # Triethylamine
]

# Calculate descriptors
results = calc.pandas(molecules)
print(results.head())
```

This basic example will output a pandas DataFrame containing molecular descriptors for each input molecule.

## Core Concepts

### Calculator Class

The `Calculator` class is the main entry point for ChemiAccel:

```python
from chemi_accel import Calculator

# Default configuration (auto-selects best backend)
calc = Calculator()

# Specify backend explicitly
calc = Calculator(backend="cuda")  # GPU acceleration
calc = Calculator(backend="cpu")   # CPU only
calc = Calculator(backend="jax")   # JAX/TPU support
```

### Input Formats

ChemiAccel accepts molecules in multiple formats:

```python
# SMILES strings
smiles_list = ["CCO", "CC(=O)O", "c1ccccc1"]

# From pandas DataFrame
import pandas as pd
df = pd.DataFrame({"smiles": smiles_list, "name": ["ethanol", "acetic_acid", "benzene"]})

# From CSV file
df = pd.read_csv("molecules.csv")

# From SDF file (requires RDKit)
from rdkit import Chem
mols = [mol for mol in Chem.SDMolSupplier("molecules.sdf")]
```

### Output Formats

ChemiAccel provides multiple output formats:

```python
# Pandas DataFrame (default)
results_df = calc.pandas(molecules)

# NumPy arrays (for performance)
results_arrays = calc.numpy(molecules)

# Dictionary format
results_dict = calc.dict(molecules)

# JAX arrays (for JAX backend)
results_jax = calc.jax(molecules)

# GPU DataFrame (RAPIDS cuDF)
if calc.backend == "cuda":
    results_cudf = calc.cudf(molecules)
```

## Descriptor Categories

### Basic Molecular Descriptors

```python
from chemi_accel import Calculator

calc = Calculator()
molecules = ["CCO", "c1ccccc1", "CC(=O)O"]

# Calculate basic descriptors
basic_results = calc.pandas(
    molecules,
    descriptors=[
        "molecular_weight",
        "logp",
        "tpsa",
        "num_atoms",
        "num_bonds",
        "num_rings"
    ]
)

print(basic_results)
```

### Fingerprint Descriptors

```python
# Morgan fingerprints
fingerprint_results = calc.pandas(
    molecules,
    descriptors=[
        "morgan_fingerprint_2048",
        "morgan_fingerprint_1024",
        "maccs_keys",
        "atom_pair_fingerprint"
    ]
)

print(f"Fingerprint shape: {fingerprint_results.shape}")
```

### 3D Descriptors (requires conformer generation)

```python
# 3D molecular descriptors
threeD_results = calc.pandas(
    molecules,
    descriptors=[
        "radius_of_gyration",
        "asphericity",
        "eccentricity",
        "principal_moments",
        "surface_area",
        "volume"
    ],
    generate_conformers=True  # Required for 3D descriptors
)
```

## Working with Different Data Sizes

### Small Datasets (< 1,000 molecules)

```python
# Direct calculation
molecules = ["CCO", "c1ccccc1", "CC(=O)O"] * 100  # 300 molecules
results = calc.pandas(molecules)
```

### Medium Datasets (1,000 - 100,000 molecules)

```python
# Use batching for better memory management
molecules = load_molecules_from_file("medium_dataset.csv")  # 10,000 molecules

results = calc.pandas(
    molecules, 
    batch_size=1000,  # Process in batches of 1000
    show_progress=True  # Show progress bar
)
```

### Large Datasets (> 100,000 molecules)

```python
# Use chunked processing with caching
def process_large_dataset(filename, chunk_size=10000):
    results = []
    
    for chunk in pd.read_csv(filename, chunksize=chunk_size):
        chunk_results = calc.pandas(
            chunk["smiles"].tolist(),
            enable_cache=True,  # Cache intermediate results
            batch_size=1000
        )
        results.append(chunk_results)
    
    return pd.concat(results, ignore_index=True)

# Process 1 million molecules
large_results = process_large_dataset("large_dataset.csv")
```

## Performance Optimization

### Backend Selection

```python
# Check available backends
print(f"Available backends: {calc.available_backends}")

# Performance comparison
import time

backends = ["cpu", "cuda"] if "cuda" in calc.available_backends else ["cpu"]
molecules = ["CCO"] * 1000

for backend in backends:
    calc_backend = Calculator(backend=backend)
    
    start_time = time.time()
    results = calc_backend.pandas(molecules)
    end_time = time.time()
    
    print(f"{backend} backend: {end_time - start_time:.2f} seconds")
```

### Memory Management

```python
# Configure memory usage
calc = Calculator(
    max_memory_gb=16,    # Limit memory usage
    batch_size=500,      # Smaller batches for less memory
    enable_cache=False   # Disable caching if memory is tight
)
```

### Parallel Processing

```python
# Increase parallelism
calc = Calculator(
    num_threads=8,       # Use 8 CPU threads
    backend="cuda",      # GPU acceleration
    batch_size=2000      # Larger batches for GPU
)
```

## Error Handling

### Basic Error Handling

```python
from chemi_accel.core.exceptions import ChemiAccelError, InvalidMoleculeError

try:
    molecules = ["CCO", "INVALID_SMILES", "c1ccccc1"]
    results = calc.pandas(molecules)
except InvalidMoleculeError as e:
    print(f"Invalid molecule detected: {e}")
    # Handle invalid molecules
    valid_molecules = [mol for mol in molecules if Chem.MolFromSmiles(mol) is not None]
    results = calc.pandas(valid_molecules)
except ChemiAccelError as e:
    print(f"ChemiAccel error: {e}")
```

### Robust Processing

```python
def robust_calculation(molecules, max_retries=3):
    """Calculate descriptors with error handling and retries."""
    for attempt in range(max_retries):
        try:
            return calc.pandas(molecules)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            # Reduce batch size on retry
            calc.batch_size = max(100, calc.batch_size // 2)
    
# Use robust calculation
results = robust_calculation(molecules)
```

## Working with Results

### Exploring Results

```python
# Calculate descriptors
results = calc.pandas(molecules)

# Basic statistics
print(f"Dataset shape: {results.shape}")
print(f"Descriptor columns: {results.columns.tolist()}")
print("\nBasic statistics:")
print(results.describe())

# Check for missing values
print(f"\nMissing values: {results.isnull().sum().sum()}")
```

### Filtering and Selection

```python
# Filter by molecular weight
heavy_molecules = results[results["molecular_weight"] > 300]

# Select drug-like molecules (Lipinski's Rule of Five)
drug_like = results[
    (results["molecular_weight"] <= 500) &
    (results["logp"] <= 5) &
    (results["num_hbd"] <= 5) &
    (results["num_hba"] <= 10)
]

print(f"Drug-like molecules: {len(drug_like)}")
```

### Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Molecular weight distribution
plt.figure(figsize=(10, 6))
plt.hist(results["molecular_weight"], bins=50, alpha=0.7)
plt.xlabel("Molecular Weight")
plt.ylabel("Frequency")
plt.title("Molecular Weight Distribution")
plt.show()

# LogP vs TPSA scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(results["logp"], results["tpsa"], alpha=0.6)
plt.xlabel("LogP")
plt.ylabel("TPSA")
plt.title("LogP vs TPSA")
plt.show()
```

## Saving and Loading Results

### Save Results

```python
# Save to CSV
results.to_csv("molecular_descriptors.csv", index=False)

# Save to Excel
results.to_excel("molecular_descriptors.xlsx", index=False)

# Save to HDF5 (more efficient for large datasets)
results.to_hdf("molecular_descriptors.h5", key="descriptors", mode="w")

# Save to Parquet (best compression and performance)
results.to_parquet("molecular_descriptors.parquet")
```

### Load Cached Results

```python
# Load previously calculated results
cached_results = pd.read_parquet("molecular_descriptors.parquet")

# Combine with new calculations
new_molecules = ["CCCC", "CCCCC"]
new_results = calc.pandas(new_molecules)

combined_results = pd.concat([cached_results, new_results], ignore_index=True)
```

## Integration with Machine Learning

### Preparing Data for ML

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Prepare features and targets
X = results.drop(["smiles"], axis=1)  # Remove non-numeric columns
y = load_target_values()  # Your target property

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Simple ML Pipeline

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")
```

## Common Workflows

### Drug Discovery Pipeline

```python
def drug_discovery_pipeline(molecules):
    """Complete drug discovery descriptor pipeline."""
    
    # Calculate comprehensive descriptors
    descriptors = calc.pandas(
        molecules,
        descriptors=[
            "molecular_weight", "logp", "tpsa", "num_hbd", "num_hba",
            "num_rotatable_bonds", "num_aromatic_rings", "slogp",
            "morgan_fingerprint_2048", "maccs_keys"
        ]
    )
    
    # Apply drug-likeness filters
    drug_like = descriptors[
        (descriptors["molecular_weight"] <= 500) &
        (descriptors["logp"] <= 5) &
        (descriptors["num_hbd"] <= 5) &
        (descriptors["num_hba"] <= 10) &
        (descriptors["tpsa"] <= 140)
    ]
    
    return drug_like

# Use the pipeline
drug_candidates = drug_discovery_pipeline(molecules)
print(f"Drug-like candidates: {len(drug_candidates)}")
```

### Materials Science Workflow

```python
def materials_analysis_pipeline(molecules):
    """Materials-focused descriptor calculation."""
    
    descriptors = calc.pandas(
        molecules,
        descriptors=[
            "molecular_weight", "logp", "polarizability",
            "num_atoms", "num_bonds", "density",
            "dipole_moment", "homo_lumo_gap"
        ],
        generate_conformers=True  # Needed for some 3D properties
    )
    
    return descriptors

# Analyze materials
materials_data = materials_analysis_pipeline(molecules)
```

## Best Practices

### 1. Start Small
```python
# Test with a small subset first
test_molecules = molecules[:10]
test_results = calc.pandas(test_molecules)
# Verify results before scaling up
```

### 2. Use Appropriate Backends
```python
# Small datasets: CPU is often sufficient
if len(molecules) < 1000:
    calc = Calculator(backend="cpu")
else:
    calc = Calculator(backend="cuda")  # Use GPU for larger datasets
```

### 3. Enable Caching for Development
```python
# Enable caching during development
calc = Calculator(enable_cache=True)
# Disable for production to save disk space
```

### 4. Monitor Memory Usage
```python
import psutil

def monitor_memory():
    memory = psutil.virtual_memory()
    print(f"Memory usage: {memory.percent}%")

monitor_memory()
results = calc.pandas(molecules)
monitor_memory()
```

### 5. Validate Input Data
```python
from rdkit import Chem

def validate_smiles(smiles_list):
    """Validate SMILES strings before processing."""
    valid_smiles = []
    invalid_count = 0
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_smiles.append(smiles)
        else:
            invalid_count += 1
    
    print(f"Valid: {len(valid_smiles)}, Invalid: {invalid_count}")
    return valid_smiles

# Use validation
validated_molecules = validate_smiles(molecules)
results = calc.pandas(validated_molecules)
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `batch_size` or use `max_memory_gb` parameter
2. **Slow Performance**: Try different backends or increase `num_threads`
3. **Invalid Molecules**: Use input validation and error handling
4. **GPU Not Found**: Check CUDA installation and drivers

### Getting Help

```python
# Check system configuration
from chemi_accel.core.system import system_info
system_info()

# Check backend availability
calc = Calculator()
print(f"Available backends: {calc.available_backends}")

# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

*For more advanced usage patterns, see the [Molecular Descriptors](Molecular-Descriptors) and [Backend Selection](Backend-Selection) guides.*
