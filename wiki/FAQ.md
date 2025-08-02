# Frequently Asked Questions (FAQ)

This FAQ covers common questions about installation, usage, performance, and troubleshooting for ChemiAccel.

## General Questions

### What is ChemiAccel?

ChemiAccel is a next-generation computational chemistry platform designed for GPU-accelerated molecular descriptor calculations and AI-driven discovery. It provides 100-1000x speedups over traditional CPU-based tools like RDKit and Mordred through massively parallel processing on modern hardware.

### How is ChemiAccel different from RDKit or Mordred?

**Performance**: ChemiAccel is designed from the ground up for parallel processing with GPU acceleration, providing orders of magnitude better performance.

**Scale**: While RDKit and Mordred work well for thousands of molecules, ChemiAccel can efficiently process millions of molecules.

**AI Integration**: Native integration with machine learning frameworks and AI-driven molecular design tools.

**Cloud Native**: Built-in support for cloud platforms (AWS, GCP, Azure) and distributed computing.

**Domain Expertise**: Specialized modules for polymer science, materials science, and drug discovery.

### What types of molecules can ChemiAccel process?

ChemiAccel can process:
- Small organic molecules (drugs, natural products)
- Large macromolecules (polymers, proteins)
- Inorganic compounds and materials
- Crystal structures and periodic systems
- Any molecule representable as SMILES, SDF, or other standard formats

### Is ChemiAccel free to use?

ChemiAccel follows a dual-licensing model:
- **Academic/Research Use**: Free for academic research and educational purposes
- **Commercial Use**: Commercial licenses available for industrial applications
- **Open Source Components**: Core algorithms are open source under MIT license

## Installation and Setup

### Why can't I install ChemiAccel?

**Common solutions:**

1. **Python Version**: Ensure you're using Python 3.8 or higher
   ```bash
   python --version  # Should be 3.8+
   ```

2. **Virtual Environment**: Use a clean virtual environment
   ```bash
   python -m venv chemiaccel-env
   source chemiaccel-env/bin/activate  # Linux/Mac
   # or
   chemiaccel-env\Scripts\activate  # Windows
   ```

3. **Update pip**: Ensure you have the latest pip
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

4. **System Dependencies**: Install system dependencies
   ```bash
   # Ubuntu/Debian
   sudo apt install build-essential python3-dev
   
   # CentOS/RHEL
   sudo yum groupinstall "Development Tools"
   ```

### How do I know if GPU acceleration is working?

```python
from chemi_accel import Calculator

calc = Calculator()
print(f"Available backends: {calc.available_backends}")

# Test CUDA backend
if "cuda" in calc.available_backends:
    cuda_calc = Calculator(backend="cuda")
    print("CUDA acceleration is available!")
else:
    print("CUDA not available - check GPU drivers and CUDA installation")
```

### Can I use ChemiAccel without a GPU?

Yes! ChemiAccel works on CPU-only systems. While GPU acceleration provides significant speedups, the CPU backend is highly optimized and still faster than many traditional tools.

```python
# Force CPU backend
calc = Calculator(backend="cpu")
```

### How much memory do I need?

**Minimum**: 8 GB RAM for small datasets (< 10K molecules)
**Recommended**: 16-32 GB RAM for typical workloads
**Large Scale**: 64+ GB RAM for datasets > 1M molecules

**GPU Memory**:
- 4 GB: Small to medium datasets
- 8-16 GB: Large datasets and complex descriptors
- 24+ GB: Very large datasets or multi-GPU setups

## Usage Questions

### What descriptors are available?

ChemiAccel provides 200+ molecular descriptors across categories:

- **Constitutional**: Molecular weight, atom counts, bond counts
- **Physicochemical**: LogP, TPSA, solubility parameters
- **Topological**: Connectivity indices, shape descriptors
- **Electronic**: HOMO/LUMO energies, partial charges
- **3D Geometric**: Surface area, volume, shape descriptors
- **Fingerprints**: Morgan, MACCS, atom pairs, topological torsions

See the [Molecular Descriptors](Molecular-Descriptors) guide for complete details.

### How do I calculate specific descriptors?

```python
from chemi_accel import Calculator

calc = Calculator()
molecules = ["CCO", "c1ccccc1", "CC(=O)O"]

# Specific descriptors
results = calc.pandas(molecules, descriptors=["molecular_weight", "logp", "tpsa"])

# Category of descriptors
results = calc.pandas(molecules, descriptors="physicochemical")

# All available descriptors
results = calc.pandas(molecules, descriptors="all")
```

### How do I handle invalid SMILES?

```python
from chemi_accel import Calculator
from chemi_accel.core.exceptions import InvalidMoleculeError
from rdkit import Chem

def validate_and_process(smiles_list):
    """Validate SMILES and process only valid ones."""
    valid_smiles = []
    invalid_smiles = []
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_smiles.append(smiles)
        else:
            invalid_smiles.append(smiles)
    
    print(f"Valid: {len(valid_smiles)}, Invalid: {len(invalid_smiles)}")
    
    if valid_smiles:
        calc = Calculator()
        results = calc.pandas(valid_smiles)
        return results, invalid_smiles
    else:
        return None, invalid_smiles

# Usage
molecules = ["CCO", "INVALID_SMILES", "c1ccccc1"]
results, invalid = validate_and_process(molecules)
```

### How do I process very large datasets?

For datasets > 100K molecules:

```python
import pandas as pd
from chemi_accel import Calculator

def process_large_dataset(csv_file, chunk_size=10000):
    """Process large dataset in chunks."""
    calc = Calculator(backend="cuda")  # Use GPU for speed
    results = []
    
    for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
        smiles_chunk = chunk["smiles"].tolist()
        
        chunk_results = calc.pandas(
            smiles_chunk,
            descriptors="default",
            batch_size=2000,  # Large GPU batches
            show_progress=True
        )
        
        results.append(chunk_results)
        print(f"Processed chunk of {len(chunk_results)} molecules")
    
    return pd.concat(results, ignore_index=True)

# Process 1M molecule dataset
results = process_large_dataset("large_dataset.csv")
```

### Can I add custom descriptors?

Yes! ChemiAccel supports custom descriptors:

```python
from chemi_accel.descriptors.custom import CustomDescriptor
from chemi_accel import Calculator
import numpy as np

class MyCustomDescriptor(CustomDescriptor):
    name = "my_descriptor"
    description = "My custom molecular descriptor"
    
    def calculate(self, mol):
        if mol is None:
            return np.nan
        # Your calculation logic here
        return mol.GetNumAtoms() / mol.GetNumBonds() if mol.GetNumBonds() > 0 else 0

# Register and use
calc = Calculator()
calc.register_custom_descriptor(MyCustomDescriptor())
results = calc.pandas(molecules, descriptors=["my_descriptor"])
```

## Performance Questions

### Why is ChemiAccel slower than expected?

**Common causes and solutions:**

1. **Wrong Backend**: Ensure you're using the appropriate backend
   ```python
   # Check what backend is being used
   calc = Calculator()
   print(f"Current backend: {calc.backend}")
   
   # Force GPU backend if available
   calc = Calculator(backend="cuda")
   ```

2. **Small Batch Size**: Increase batch size for GPU backends
   ```python
   calc = Calculator(backend="cuda")
   results = calc.pandas(molecules, batch_size=5000)  # Larger batches
   ```

3. **Memory Limits**: Remove memory constraints
   ```python
   calc = Calculator(max_memory_gb=None)  # Remove memory limit
   ```

4. **Cold Start**: First run includes JIT compilation overhead
   ```python
   # Warm-up run
   calc.pandas(molecules[:10])
   
   # Timed run
   import time
   start = time.time()
   results = calc.pandas(molecules)
   print(f"Processing time: {time.time() - start:.2f}s")
   ```

### How do I choose the optimal batch size?

Batch size depends on your hardware and dataset:

```python
def find_optimal_batch_size(molecules, max_batch_size=10000):
    """Find optimal batch size through testing."""
    calc = Calculator(backend="cuda")
    test_molecules = molecules[:1000]  # Use subset for testing
    
    best_batch_size = 100
    best_throughput = 0
    
    for batch_size in [100, 500, 1000, 2000, 5000, max_batch_size]:
        try:
            import time
            start = time.time()
            calc.pandas(test_molecules, batch_size=batch_size)
            end = time.time()
            
            throughput = len(test_molecules) / (end - start)
            print(f"Batch size {batch_size}: {throughput:.1f} mol/s")
            
            if throughput > best_throughput:
                best_throughput = throughput
                best_batch_size = batch_size
                
        except Exception as e:
            print(f"Batch size {batch_size} failed: {e}")
            break
    
    return best_batch_size

optimal_batch = find_optimal_batch_size(molecules)
print(f"Optimal batch size: {optimal_batch}")
```

### How many molecules can I process at once?

This depends on your system resources:

| System Configuration | Recommended Limit | Processing Time* |
|---------------------|-------------------|------------------|
| 8 GB RAM, CPU only | 10K molecules | 10-30 minutes |
| 16 GB RAM, RTX 3060 | 50K molecules | 5-15 minutes |
| 32 GB RAM, RTX 3080 | 200K molecules | 10-30 minutes |
| 64 GB RAM, A100 | 1M+ molecules | 30-120 minutes |

*For comprehensive descriptor calculation (200+ descriptors)

## Backend and Hardware Questions

### Which backend should I use?

**Decision flowchart:**

1. **Do you have an NVIDIA GPU?**
   - Yes → Use `"cuda"` backend
   - No → Go to step 2

2. **Do you have access to TPUs?**
   - Yes → Use `"jax"` backend
   - No → Go to step 3

3. **Is your dataset > 100K molecules?**
   - Yes → Consider cloud computing with GPU instances
   - No → Use `"cpu"` backend

```python
# Automatic backend selection
calc = Calculator()  # Chooses best available backend

# Manual backend selection
calc = Calculator(backend="cuda")   # Force CUDA
calc = Calculator(backend="cpu")    # Force CPU
calc = Calculator(backend="jax")    # Force JAX
```

### My GPU isn't being detected. What should I do?

**Troubleshooting steps:**

1. **Check NVIDIA drivers:**
   ```bash
   nvidia-smi  # Should show GPU information
   ```

2. **Check CUDA installation:**
   ```bash
   nvcc --version  # Should show CUDA version
   ```

3. **Check CuPy installation:**
   ```python
   import cupy as cp
   print(f"CUDA available: {cp.cuda.is_available()}")
   print(f"GPU count: {cp.cuda.runtime.getDeviceCount()}")
   ```

4. **Reinstall CUDA dependencies:**
   ```bash
   pip uninstall cupy-cuda11x  # or appropriate version
   pip install cupy-cuda11x    # match your CUDA version
   ```

### Can I use multiple GPUs?

Yes! ChemiAccel supports multi-GPU processing:

```python
from chemi_accel import Calculator
from chemi_accel.backends.cuda import CudaConfig

# Multi-GPU configuration
cuda_config = CudaConfig(
    device_ids=[0, 1, 2, 3],     # Use 4 GPUs
    data_parallel=True,          # Enable data parallelism
    batch_size=10000             # Large batches for multi-GPU
)

calc = Calculator(backend="cuda", config=cuda_config)
results = calc.pandas(molecules)
```

### How do I optimize memory usage?

**Memory optimization strategies:**

1. **Reduce batch size:**
   ```python
   calc = Calculator(batch_size=500)  # Smaller batches
   ```

2. **Process in chunks:**
   ```python
   def process_in_chunks(molecules, chunk_size=5000):
       results = []
       for i in range(0, len(molecules), chunk_size):
           chunk = molecules[i:i+chunk_size]
           chunk_results = calc.pandas(chunk)
           results.append(chunk_results)
       return pd.concat(results, ignore_index=True)
   ```

3. **Use streaming mode:**
   ```python
   calc = Calculator(enable_streaming=True)
   ```

4. **Limit descriptor selection:**
   ```python
   # Calculate only essential descriptors
   essential_descriptors = ["molecular_weight", "logp", "tpsa"]
   results = calc.pandas(molecules, descriptors=essential_descriptors)
   ```

## Cloud Computing Questions

### How do I use ChemiAccel in the cloud?

ChemiAccel integrates with major cloud platforms:

**AWS SageMaker:**
```python
from chemi_accel.cloud.aws import SageMakerJob

job = SageMakerJob()
job_name = job.submit_job({
    "instance_type": "ml.g4dn.xlarge",
    "input_data": "s3://my-bucket/molecules.csv",
    "output_data": "s3://my-bucket/results/"
})
```

**Google Cloud Vertex AI:**
```python
from chemi_accel.cloud.gcp import VertexAIJob

job = VertexAIJob()
job_name = job.submit_training_job({
    "machine_type": "n1-standard-4",
    "accelerator": "NVIDIA_TESLA_T4"
})
```

**Azure ML:**
```python
from chemi_accel.cloud.azure import AzureMLJob

job = AzureMLJob()
job_name = job.submit_experiment({
    "vm_size": "Standard_NC6s_v3",
    "node_count": 2
})
```

### What cloud instance types should I use?

**AWS Recommendations:**
- **Development**: `g4dn.xlarge` (1 GPU, 4 vCPU, 16 GB)
- **Production**: `g4dn.2xlarge` (1 GPU, 8 vCPU, 32 GB)
- **Large Scale**: `p3.8xlarge` (4 V100 GPUs, 32 vCPU, 244 GB)

**GCP Recommendations:**
- **Development**: `n1-standard-4` + 1 T4 GPU
- **Production**: `n1-standard-8` + 1 V100 GPU
- **Large Scale**: `n1-standard-32` + 4 V100 GPUs

**Azure Recommendations:**
- **Development**: `Standard_NC6s_v3` (1 V100, 6 vCPU, 112 GB)
- **Production**: `Standard_NC12s_v3` (2 V100, 12 vCPU, 224 GB)

### How much does cloud computing cost?

Approximate costs (USD per hour):

| Platform | Instance Type | GPUs | Cost/hour | Molecules/hour* |
|----------|---------------|------|-----------|----------------|
| AWS | g4dn.xlarge | 1 T4 | $0.526 | 50K-100K |
| AWS | p3.2xlarge | 1 V100 | $3.06 | 200K-500K |
| GCP | n1-standard-4 + T4 | 1 T4 | $0.60 | 50K-100K |
| Azure | NC6s_v3 | 1 V100 | $3.06 | 200K-500K |

*Estimated throughput for comprehensive descriptor calculation

## Error Messages and Troubleshooting

### "CUDA out of memory" error

**Solutions:**

1. **Reduce batch size:**
   ```python
   calc = Calculator(backend="cuda", batch_size=1000)  # Smaller batches
   ```

2. **Enable memory growth:**
   ```python
   from chemi_accel.backends.cuda import CudaConfig
   config = CudaConfig(memory_growth=True)
   calc = Calculator(backend="cuda", config=config)
   ```

3. **Clear GPU memory:**
   ```python
   import cupy as cp
   cp.get_default_memory_pool().free_all_blocks()
   ```

4. **Use CPU backend as fallback:**
   ```python
   try:
       calc = Calculator(backend="cuda")
       results = calc.pandas(molecules)
   except Exception:
       calc = Calculator(backend="cpu")
       results = calc.pandas(molecules)
   ```

### "ImportError: No module named 'cupy'" error

**Solution:**
```bash
# Install CuPy for your CUDA version
pip install cupy-cuda11x  # For CUDA 11.x
pip install cupy-cuda12x  # For CUDA 12.x

# Or install without CUDA support
pip install chemi-accel  # CPU-only installation
```

### "RuntimeError: Backend not available" error

**Solution:**
```python
from chemi_accel import Calculator

# Check available backends
calc = Calculator()
print(f"Available backends: {calc.available_backends}")

# Use an available backend
if "cuda" in calc.available_backends:
    calc = Calculator(backend="cuda")
elif "jax" in calc.available_backends:
    calc = Calculator(backend="jax")
else:
    calc = Calculator(backend="cpu")
```

### "ValueError: Invalid SMILES" error

**Solution:**
```python
from rdkit import Chem

def clean_smiles_list(smiles_list):
    """Remove invalid SMILES from list."""
    valid_smiles = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_smiles.append(smiles)
        else:
            print(f"Invalid SMILES: {smiles}")
    return valid_smiles

# Clean your data
clean_molecules = clean_smiles_list(molecules)
results = calc.pandas(clean_molecules)
```

## Integration Questions

### How do I integrate ChemiAccel with scikit-learn?

```python
from chemi_accel import Calculator
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Calculate descriptors
calc = Calculator()
X = calc.pandas(molecules, descriptors="default")
y = target_values  # Your experimental data

# Prepare for scikit-learn
X_numeric = X.select_dtypes(include=[np.number])
X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestRegressor()
model.fit(X_train_scaled, y_train)

# Make predictions
predictions = model.predict(X_test_scaled)
```

### Can I use ChemiAccel with Jupyter notebooks?

Yes! ChemiAccel works perfectly in Jupyter notebooks:

```python
# Enable progress bars in Jupyter
from tqdm.notebook import tqdm

# Calculate with progress bar
results = calc.pandas(molecules, show_progress=True)

# Interactive visualization
from chemi_accel.visualization import MultiObjectiveVisualizer
viz = MultiObjectiveVisualizer()
fig = viz.plot_property_distributions(results)
fig.show()
```

### How do I save and load results?

```python
# Save results
results.to_csv("descriptors.csv", index=False)
results.to_parquet("descriptors.parquet")  # More efficient
results.to_hdf("descriptors.h5", key="descriptors")

# Load results
import pandas as pd
results = pd.read_csv("descriptors.csv")
results = pd.read_parquet("descriptors.parquet")
results = pd.read_hdf("descriptors.h5", key="descriptors")
```

## License and Commercial Use

### Can I use ChemiAccel commercially?

ChemiAccel offers different licensing options:

- **Academic License**: Free for academic research and education
- **Commercial License**: Required for commercial use and distribution
- **Enterprise License**: Includes support and additional features

Contact licensing@chemiaccel.com for commercial licensing information.

### How do I cite ChemiAccel in publications?

```bibtex
@software{chemiaccel2024,
  title={ChemiAccel: GPU-Accelerated Molecular Descriptor Calculations},
  author={ChemiAccel Development Team},
  year={2024},
  url={https://github.com/chemiaccel/chemiaccel},
  version={1.0.0}
}
```

### Is my data secure when using ChemiAccel?

Yes! ChemiAccel includes several security features:

- **Local Processing**: All calculations can be performed locally
- **Encryption**: Optional data encryption for cloud operations
- **Privacy**: No data is transmitted unless explicitly configured
- **Compliance**: Supports GDPR and other data protection regulations

## Getting Help

### Where can I get additional help?

1. **Documentation**: Comprehensive guides in this wiki
2. **Examples**: See the [Examples](Examples) gallery
3. **GitHub Issues**: Report bugs and request features
4. **Community Forum**: Ask questions and share knowledge
5. **Email Support**: support@chemiaccel.com (commercial users)

### How do I report a bug?

1. Check if the issue is already reported on [GitHub Issues](https://github.com/chemiaccel/chemiaccel/issues)
2. Create a minimal example that reproduces the problem
3. Include system information:
   ```python
   from chemi_accel.core.system import system_info
   system_info()
   ```
4. Submit a detailed bug report with:
   - ChemiAccel version
   - Operating system
   - Hardware configuration
   - Error messages and stack traces
   - Minimal reproducible example

### How do I request a new feature?

1. Check existing [feature requests](https://github.com/chemiaccel/chemiaccel/issues?q=is%3Aissue+is%3Aopen+label%3Aenhancement)
2. Open a new issue with the "enhancement" label
3. Describe the feature in detail:
   - Use case and motivation
   - Proposed API or interface
   - Expected behavior
   - Any relevant references or examples

---

*Can't find your question? Check the [Community Forum](https://github.com/chemiaccel/community/discussions) or [create an issue](https://github.com/chemiaccel/chemiaccel/issues/new).*
