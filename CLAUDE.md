# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ChemiAccel is a next-generation computational science platform designed to accelerate chemical and materials discovery through massively parallel, GPU-accelerated molecular descriptor calculations. It's positioned as more than just a library - it's an integrated ecosystem that leverages modern hardware (GPUs, TPUs) to eliminate computational bottlenecks across the entire discovery cycle, targeting 100-1000x speedup over traditional CPU-based tools like RDKit and Mordred.

### Core Mission & Differentiators

- **GPU-accelerated molecular descriptor calculations** with 100-1000x speedups
- **Multi-backend support**: CUDA for NVIDIA GPUs, JAX/TPU for diverse accelerators, RAPIDS integration
- **First-class polymer support**: Specialized modules for monomers, dimers, oligomers, copolymers
- **Cloud-native architecture**: Serverless batch processing, ML platform integration
- **AI-driven discovery**: Active learning, generative chemistry, uncertainty quantification
- **Production-ready**: Robust error handling, comprehensive testing, enterprise deployment

### Target Users

- Computational chemists and cheminformatics researchers
- ML engineers working on molecular property prediction
- Pharmaceutical and drug discovery teams
- Polymer scientists and materials researchers
- Cloud-native data science teams

## Architecture Overview

ChemiAccel employs a sophisticated multi-layered architecture:

### Core Architecture (`src/chemi_accel/core/`)
- **calculator.py**: Main Calculator class with multi-backend support
- **backend.py**: Backend selection and management system
- **molecular.py**: MolecularData structures and representations
- **exceptions.py**: Comprehensive error handling framework

### Multi-Backend System
- **CUDA Backend** (`src/chemi_accel/backends/cuda/`): Custom C++/CUDA kernels, RAPIDS integration, optimized memory management
- **JAX Backend** (`src/chemi_accel/backends/jax/`): TPU support, JIT compilation, automatic differentiation
- **RAPIDS Backend** (`src/chemi_accel/backends/rapids/`): GPU-native DataFrames with cuDF/cuPy

### Descriptor Engine (`src/chemi_accel/descriptors/`)
- **basic.py**: Fundamental molecular descriptors (MW, LogP, TPSA)
- **fingerprints.py**: Morgan fingerprints, MACCS keys, topological fingerprints
- **geometrical_3d.py**: 3D shape descriptors, conformer generation, electrostatic properties
- **constitutional.py**: Constitutional and topological descriptors
- **advanced.py**: Complex descriptors for specialized applications

### Specialized Domain Modules

#### Polymer Science (`src/chemi_accel/polymer/`)
- **monomer.py**: Monomer analysis and reactivity prediction
- **copolymer.py**: Copolymer composition analysis and property prediction
- **predictive/**: ML models for polymer property prediction
- **thermoset.py**: Thermoset polymer analysis
- **conformers.py**: Polymer-specific conformational analysis

#### Materials Science (`src/chemi_accel/materials/`)
- **crystal_parser.py**: CIF and POSCAR file parsing with periodic boundary conditions
- **crystal_descriptors.py**: GPU-accelerated geometric and topological descriptors
- **periodic_descriptors.py**: Symmetry analysis, distribution functions, structure factors
- **materials_calculator.py**: Unified interface with pymatgen/ASE integration

#### AI & Machine Learning (`src/chemi_accel/ai/`)
- **inverse_design.py**: Inverse molecular design algorithms
- **multi_objective_optimization.py**: Multi-objective optimization for molecular design
- **active_learning.py**: Active learning strategies for property prediction
- **generative_chemistry.py**: Generative models for novel molecule discovery
- **uncertainty_quantification.py**: Uncertainty estimation for predictions

### Cloud & Distribution (`src/chemi_accel/cloud/`)
- **ml_platforms.py**: Native integration with SageMaker, Vertex AI, Azure ML
- **distributed_computing.py**: Multi-node distributed processing
- **hpc_integration.py**: HPC cluster integration with SLURM/PBS
- **batch_processor.py**: Serverless batch processing workflows

## Development Commands

### Installation and Setup

```bash
# Install with CUDA support
pip install -e .[cuda]

# Install with JAX/TPU support  
pip install -e .[jax]

# Install all dependencies including dev tools
pip install -e .[all]

# Install core dependencies only
pip install -e .
```

### Code Quality and Development

```bash
# Format code with black (line length: 88)
black src/ tests/ examples/ benchmarks/

# Sort imports with isort
isort src/ tests/ examples/ benchmarks/

# Lint with ruff (comprehensive rule set)
ruff src/ tests/ examples/ benchmarks/

# Type checking with mypy (strict mode)
mypy src/

# Pre-commit hooks (if configured)
pre-commit run --all-files
```

### Testing Strategy

```bash
# Run all tests
pytest tests/

# Run unit tests only
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run validation tests (accuracy verification)
pytest tests/validation/

# Backend-specific testing
pytest tests/ -m cuda      # CUDA backend tests
pytest tests/ -m jax       # JAX backend tests
pytest tests/ -m "not cuda and not jax"  # CPU-only tests

# Performance and scaling tests
pytest tests/ -m slow      # Include slow/performance tests
pytest tests/ -m benchmark # Benchmark tests only

# Specific domain testing
pytest tests/unit/polymer/     # Polymer-specific tests
pytest tests/unit/materials/   # Materials science tests
pytest tests/integration/cloud/  # Cloud integration tests

# Coverage reporting
pytest tests/ --cov=src/chemi_accel --cov-report=html
```

### Comprehensive Validation

```bash
# Run full validation suite
python tests/validation/run_comprehensive_validation.py

# Include slow stress tests
python tests/validation/run_comprehensive_validation.py --include-slow

# Save validation results
python tests/validation/run_comprehensive_validation.py --save-results --output-dir validation_results/

# Specific validation categories
python tests/validation/test_reference_validation.py
python tests/validation/test_polymer_validation.py  
python tests/validation/test_materials_validation.py
```

### Benchmarking and Performance

```bash
# Run performance benchmarks vs RDKit/Mordred
python benchmarks/run_benchmarks.py --performance

# Run scaling benchmarks across dataset sizes
python benchmarks/run_benchmarks.py --scaling

# Run all benchmarks with HTML report generation
python benchmarks/run_benchmarks.py --all --generate-report

# Custom configuration
python benchmarks/run_benchmarks.py --performance \
  --dataset-sizes 100 1000 10000 50000 \
  --descriptors molecular_weight logp tpsa morgan_fingerprint \
  --iterations 10
```

## API Usage Patterns

### Multi-Backend Calculator

```python
from chemi_accel import Calculator

# Auto backend selection
calc = Calculator()
results = calc.pandas(smiles_list)

# Specific backends
cuda_calc = Calculator(backend="cuda")     # NVIDIA GPU
jax_calc = Calculator(backend="jax")       # TPU/other accelerators
rapids_calc = Calculator(backend="rapids") # GPU-native DataFrames
cpu_calc = Calculator(backend="cpu")       # CPU fallback

# Output format selection
pandas_df = calc.pandas(smiles_list)       # pandas DataFrame
cudf_df = calc.cudf(smiles_series)         # GPU DataFrame
jax_arrays = calc.jax(smiles_list)         # JAX arrays
```

### Polymer Analysis

```python
from chemi_accel.polymer import CopolymerAnalyzer, MonomerPredictor

# Copolymer analysis
analyzer = CopolymerAnalyzer()
composition_data = analyzer.analyze_copolymer_composition(
    monomers=['styrene', 'acrylonitrile'],
    ratios=[0.7, 0.3]
)

# Monomer property prediction
predictor = MonomerPredictor()
reactivity = predictor.predict_reactivity('vinyl_acetate')
```

### Materials Science

```python
from chemi_accel.materials import MaterialsCalculator

# Crystal structure analysis
calc = MaterialsCalculator(use_gpu=True)
df = calc.from_cif_file("structure.cif")
df = calc.from_poscar_file("POSCAR")

# Batch processing
df = calc.batch_process_directory("cif_files/", "*.cif")
```

### Cloud ML Platform Integration

```python
from chemi_accel.cloud.ml_platforms import submit_chemi_accel_job, MLJobConfig

# Submit to SageMaker
job_config = MLJobConfig(
    instance_type="ml.g4dn.xlarge",
    input_data_uri="s3://bucket/compounds.csv",
    output_data_uri="s3://bucket/results/"
)

job_name = submit_chemi_accel_job(
    platform='sagemaker',
    job_config=job_config,
    descriptor_config={'descriptors': ['molecular_weight', 'logp', 'tpsa']}
)
```

## Key Dependencies and Requirements

### Core Dependencies
- **Python**: >=3.8 (supports 3.9, 3.10, 3.11)
- **NumPy**: >=1.20.0 (numerical computing)
- **Pandas**: >=1.3.0 (data manipulation)
- **RDKit**: >=2022.03.1 (cheminformatics backbone)
- **SciPy**: >=1.7.0 (scientific computing)
- **tqdm**: >=4.60.0 (progress bars)

### Backend-Specific Requirements

#### CUDA Backend
- CUDA Toolkit 11.x
- **CuPy**: >=10.0.0 (GPU arrays)
- **cuDF**: >=22.04 (GPU DataFrames)
- **cuGraph**: >=22.04 (GPU graph processing)
- **Numba**: >=0.56.0 (JIT compilation)

#### JAX Backend
- **JAX[cuda11_pip]**: >=0.4.0 (array programming)
- **JAXlib**: >=0.4.0 (XLA compilation)
- **Optax**: >=0.1.4 (optimization)

#### Development Tools
- **Black**: >=22.0.0 (code formatting)
- **isort**: >=5.10.0 (import sorting)
- **mypy**: >=0.950 (type checking)
- **ruff**: >=0.0.260 (fast linting)
- **pre-commit**: >=2.17.0 (git hooks)

#### Testing Framework
- **pytest**: >=7.0.0 (test runner)
- **pytest-cov**: >=3.0.0 (coverage)
- **pytest-xdist**: >=2.5.0 (parallel testing)
- **hypothesis**: >=6.40.0 (property-based testing)
- **pytest-benchmark**: >=3.4.0 (performance testing)

## Project Conventions and Standards

### Code Style
- **Classes**: PascalCase (e.g., `MolecularDescriptor`, `CudaBackend`)
- **Functions/Variables**: snake_case (e.g., `calculate_descriptors`, `mol_data`)
- **Constants**: UPPER_CASE (e.g., `DEFAULT_BATCH_SIZE`, `MAX_CONFORMERS`)
- **Line Length**: 88 characters (Black standard)
- **Type Hints**: Comprehensive type annotations required
- **Docstrings**: Google-style docstrings for all public APIs

### Error Handling
- Comprehensive exception hierarchy in `core.exceptions`
- Graceful degradation when optional dependencies unavailable
- Detailed error messages with context and suggested solutions
- Proper logging with structured information

### Performance Considerations
- Memory-efficient batch processing
- Zero-copy operations where possible
- Automatic GPU memory management
- JIT compilation optimization
- Streaming support for large datasets

## Testing Strategy and Markers

### Test Organization
- **Unit Tests** (`tests/unit/`): Component-level testing with mocks
- **Integration Tests** (`tests/integration/`): End-to-end workflow testing
- **Validation Tests** (`tests/validation/`): Accuracy and robustness validation
- **Benchmark Tests** (`benchmarks/`): Performance comparison and regression

### Test Markers
Use pytest markers to control test execution:
- **`cuda`**: Requires CUDA-capable hardware
- **`jax`**: Requires JAX/TPU backend
- **`slow`**: Long-running tests (>30 seconds)
- **`integration`**: Integration tests requiring external services
- **`benchmark`**: Performance benchmarking tests

### Current Development Status

Based on the comprehensive implementation plan:

#### ✅ Completed Infrastructure
- Core project foundation and configuration systems
- Backend selection and molecular data structures
- Comprehensive testing and validation frameworks
- CUDA and JAX backend implementations
- Materials science module with crystal structure support
- Cloud ML platform integrations (SageMaker, Vertex AI, Azure ML)
- Polymer analysis modules and predictive models
- AI-driven discovery tools and optimization algorithms

#### 🚧 Active Development Areas
- Advanced molecular analysis and quantum descriptors
- Enhanced 3D conformer generation and optimization
- Expanded generative chemistry capabilities
- Production deployment and monitoring systems
- Performance optimization and scaling improvements

#### 🔜 Planned Features
- Quantum chemistry descriptor integration
- Advanced reaction informatics
- Federated learning capabilities
- Enhanced cloud-native workflows
- Real-time streaming processing

## Integration Points

### External Library Integration
- **pymatgen/ASE**: Materials science workflow integration
- **scikit-learn**: ML pipeline compatibility
- **Dask**: Distributed computing for large-scale processing
- **RAPIDS**: GPU-accelerated data science stack

### Cloud Platform Integration
- **AWS**: SageMaker, S3, Lambda, Batch
- **GCP**: Vertex AI, Cloud Storage, Cloud Functions
- **Azure**: ML Studio, Blob Storage, Functions

### HPC Integration
- **SLURM**: Job scheduling and resource management
- **MPI**: Multi-node parallel processing
- **Container platforms**: Docker, Singularity, Kubernetes

This represents a comprehensive computational chemistry platform that extends far beyond traditional descriptor libraries, incorporating cutting-edge AI, cloud computing, and specialized domain expertise for polymer and materials science applications.
