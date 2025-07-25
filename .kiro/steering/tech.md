# Technology Stack

## Core Technologies
- **Python 3.8+**: Primary language with type hints and modern features (supports 3.9, 3.10, 3.11)
- **CUDA**: GPU acceleration for molecular calculations with custom C++/CUDA kernels
- **JAX**: TPU/XLA compilation, automatic differentiation, and JIT compilation
- **RAPIDS**: cudf, cupy integration for GPU-native DataFrames and processing
- **RDKit**: Molecular parsing, cheminformatics backbone, and reference implementations
- **Cloud Platforms**: Native integration with AWS, GCP, and Azure for serverless computing
- **HPC Systems**: MPI, SLURM, and distributed computing support

## Key Dependencies

### Core Backend Dependencies
- **CUDA Backend**: CUDA Toolkit 11.x, CuPy >=10.0.0, cuDF >=22.04, cuGraph >=22.04, Numba >=0.56.0
- **JAX Backend**: JAX[cuda11_pip] >=0.4.0, JAXlib >=0.4.0, Optax >=0.1.4 (optimization)
- **Data Processing**: pandas >=1.3.0, numpy >=1.20.0, scipy >=1.7.0, dask (distributed computing)

### Chemistry and Materials
- **Cheminformatics**: RDKit >=2022.03.1, mordred (validation), tqdm >=4.60.0
- **3D/Conformers**: OpenMM (conformer generation), OpenEye (optional), rdkit-pypi
- **Polymer Science**: polyinfo, pysimm (polymer utilities), polymer property databases
- **Materials Science**: pymatgen (materials analysis), ASE (atomic simulation environment)
- **Quantum Chemistry**: ANI models, GFN2-xTB surrogates, custom QML implementations

### AI and Machine Learning
- **Deep Learning**: PyTorch >=1.9.0, TensorFlow >=2.6.0 (optional), transformers
- **Graph Neural Networks**: PyTorch Geometric, DGL (Deep Graph Library)
- **Optimization**: scikit-optimize, optuna, ax-platform (Bayesian optimization)
- **Uncertainty**: scikit-learn, GPyTorch (Gaussian processes), conformal prediction libraries

### Cloud and Distributed Computing
- **Cloud Platforms**: boto3 (AWS), google-cloud (GCP), azure-sdk (Azure)
- **Batch Processing**: AWS Batch, Google Cloud Batch, Azure Batch APIs
- **ML Platforms**: sagemaker-python-sdk, google-cloud-aiplatform, azureml-sdk
- **Distributed**: Ray >=1.9.0, Dask >=2021.0.0, MPI4Py (HPC environments)

### Development and Testing
- **Testing**: pytest >=7.0.0, pytest-cov >=3.0.0, pytest-xdist >=2.5.0, hypothesis >=6.40.0
- **Performance**: pytest-benchmark >=3.4.0, memory-profiler, line-profiler
- **Code Quality**: black >=22.0.0, isort >=5.10.0, mypy >=0.950, ruff >=0.0.260
- **Build**: setuptools/pyproject.toml, conda packaging, Docker, pre-commit >=2.17.0

## Development Commands

### Environment Setup
```bash
# Install with CUDA support
pip install chemi-accel[cuda]

# Install with JAX/TPU support  
pip install chemi-accel[jax]

# Install all features including cloud and AI modules
pip install chemi-accel[all]

# Install development dependencies
pip install -e .[dev,test]

# Install cloud-specific dependencies
pip install chemi-accel[cloud]

# Install federated learning capabilities
pip install chemi-accel[federated]
```

### Testing
```bash
# Run full test suite
pytest tests/

# Run backend-specific tests
pytest tests/ -m cuda      # CUDA backend tests
pytest tests/ -m jax       # JAX backend tests
pytest tests/ -m "not cuda and not jax"  # CPU-only tests

# Run domain-specific tests
pytest tests/unit/polymer/     # Polymer-specific tests
pytest tests/unit/materials/   # Materials science tests
pytest tests/integration/cloud/  # Cloud integration tests

# Run performance and scaling tests
pytest tests/ -m slow      # Include slow/performance tests
pytest tests/ -m benchmark # Benchmark tests only

# Run comprehensive validation
python tests/validation/run_comprehensive_validation.py
python tests/validation/run_comprehensive_validation.py --include-slow

# Run specific validation categories
pytest tests/validation/test_reference_validation.py
pytest tests/validation/test_polymer_validation.py  
pytest tests/validation/test_materials_validation.py

# Coverage reporting
pytest tests/ --cov=src/chemi_accel --cov-report=html
```

### Build and Distribution
```bash
# Build package
python -m build

# Run type checking
mypy src/

# Code formatting and linting
black src/ tests/ examples/ benchmarks/
isort src/ tests/ examples/ benchmarks/
ruff src/ tests/ examples/ benchmarks/

# Generate documentation
sphinx-build docs/ docs/_build/

# Run performance benchmarks vs RDKit/Mordred
python benchmarks/run_benchmarks.py --performance

# Run scaling benchmarks across dataset sizes
python benchmarks/run_benchmarks.py --scaling

# Run all benchmarks with HTML report generation
python benchmarks/run_benchmarks.py --all --generate-report

# Pre-commit hooks (if configured)
pre-commit run --all-files
```

## Performance Considerations

### GPU Optimization
- Memory management critical for GPU workflows with unified memory and memory pools
- Batch size optimization based on available GPU memory and workload characteristics
- Coalesced memory access patterns for optimal GPU bandwidth utilization
- Shared memory caching for frequently accessed molecular data
- CUDA graphs for reduced kernel launch overhead

### JAX/TPU Optimization
- JIT compilation warmup for JAX backend with graph caching
- Batch size tuning for TPU matrix multiplication units (128x128 tiles)
- Mixed precision (bfloat16) for improved TPU performance
- Pipeline parallelism to overlap computation and data transfer

### Distributed Computing
- Intelligent auto-scaling based on cloud queue length and workload
- MPI communication optimization with non-blocking operations
- Load balancing across heterogeneous cluster nodes
- Efficient data serialization and cloud storage streaming

### AI Module Optimization
- Model quantization and mixed precision for generative models
- Gradient accumulation for large batch training
- Efficient acquisition function computation for active learning
- Uncertainty quantification optimization with ensemble methods

### Data Pipeline
- Zero-copy operations preferred for large datasets
- Efficient handling of variable-length polymer chains and conformer ensembles
- Parallel conformer generation for oligomer flexibility
- Memory-efficient storage of repeating polymer units and crystal structures
- Streaming support for billion-molecule datasets
