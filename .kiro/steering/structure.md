# Project Structure

## Package Organization
```
chemi-accel/
├── src/chemi_accel/           # Main package source
│   ├── __init__.py           # Public API exports
│   ├── core/                 # Core abstractions and interfaces
│   │   ├── calculator.py     # Main Calculator class
│   │   ├── backend.py        # Backend selection logic
│   │   ├── molecular.py      # Molecular data structures
│   │   └── exceptions.py     # Comprehensive error handling framework
│   ├── backends/             # Backend implementations
│   │   ├── cuda/            # CUDA kernels and utilities
│   │   │   ├── kernels.py           # Core CUDA kernels
│   │   │   ├── memory.py            # GPU memory management
│   │   │   ├── streaming.py         # Large dataset streaming
│   │   │   ├── pharmacophore_screening.py  # 3D pharmacophore CUDA kernels
│   │   │   ├── reaction_informatics.py     # Reaction enumeration kernels
│   │   │   └── solvation_gist.py           # GIST analysis kernels
│   │   ├── jax/             # JAX/TPU implementations
│   │   │   ├── molecular.py         # JAX molecular representations
│   │   │   ├── descriptors.py       # JAX descriptor algorithms
│   │   │   ├── vectorization.py     # jax.vmap implementations
│   │   │   └── batch_processing.py  # Efficient TPU batching
│   │   └── rapids/          # RAPIDS integration
│   │       ├── cudf_integration.py  # cuDF DataFrame processing
│   │       └── cupy_operations.py   # cuPy array operations
│   ├── descriptors/         # Descriptor calculation modules
│   │   ├── basic.py         # Simple molecular properties
│   │   ├── fingerprints.py  # Molecular fingerprints
│   │   ├── advanced.py      # 3D and complex descriptors
│   │   ├── constitutional.py        # Constitutional descriptors (150+)
│   │   ├── topological.py          # Topological descriptors
│   │   ├── geometrical_3d.py       # Advanced 3D descriptors
│   │   ├── admet.py                # ADMET and drug discovery descriptors
│   │   ├── gnn_features.py         # Graph neural network features
│   │   ├── conformer_generation.py # GPU conformer generation
│   │   ├── pharmacophore_screening.py # 3D pharmacophore modeling
│   │   ├── reaction_informatics.py    # Reaction enumeration and analysis
│   │   ├── solvation_thermodynamics.py # GIST solvation analysis
│   │   └── polymer.py       # Polymer-specific descriptors
│   ├── ai/                  # AI-driven discovery modules
│   │   ├── active_learning.py       # Active learning and acquisition functions
│   │   ├── inverse_design.py        # Gradient-based molecular optimization
│   │   ├── generative_chemistry.py  # Multiple generative architectures
│   │   ├── multi_objective_optimization.py # Pareto optimization
│   │   └── uncertainty_quantification.py   # UQ methods
│   ├── polymer/             # Polymer processing modules
│   │   ├── monomer.py       # Monomer analysis and reactivity
│   │   ├── oligomer.py      # Dimer/oligomer processing
│   │   ├── copolymer.py     # Copolymer property prediction
│   │   ├── thermoset.py     # Thermoset curing simulation
│   │   ├── conformers.py    # Polymer chain conformational sampling
│   │   ├── repeating.py     # Repeating unit identification
│   │   ├── inverse_design.py        # Inverse polymer design
│   │   └── predictive/      # ML models for polymer properties
│   │       ├── descriptors.py       # Polymer-specific descriptors
│   │       ├── models/              # Pre-trained ML models
│   │       ├── model_training.py    # Training infrastructure
│   │       └── property_engine.py   # Polymer property prediction
│   ├── materials/           # Materials science modules
│   │   ├── crystal_parser.py        # CIF and POSCAR parsing
│   │   ├── crystal_descriptors.py   # Crystal structure descriptors
│   │   ├── periodic_descriptors.py  # Periodic system analysis
│   │   ├── materials_calculator.py  # Unified materials interface
│   │   └── validation.py           # Materials validation framework
│   ├── analysis/            # Advanced molecular analysis
│   │   ├── trajectory_reader.py     # MD trajectory processing
│   │   ├── dynamic_networks.py      # Dynamic network analysis
│   │   ├── dynamic_plif.py         # Dynamic interaction fingerprints
│   │   └── timeseries_properties.py # Time-series analysis
│   ├── cloud/               # Cloud-native and distributed computing
│   │   ├── batch_processor.py       # Serverless batch processing
│   │   ├── ml_platforms.py         # ML platform integration
│   │   ├── distributed_computing.py # Multi-node processing
│   │   ├── hpc_integration.py      # HPC cluster integration
│   │   ├── storage.py              # Cloud storage streaming
│   │   └── containers.py           # Container deployment
│   ├── data_curation/       # Data curation and quality control  
│   │   ├── gpu_molecular_cleaning.py    # GPU-accelerated cleaning
│   │   ├── parallel_validation.py       # Parallel validation
│   │   ├── tautomer_standardization.py  # Tautomer handling
│   │   ├── inchikey_deduplication.py    # Deduplication
│   │   └── pains_substructure_filter.py # PAINS filtering
│   ├── federated/           # Federated learning infrastructure
│   │   ├── core.py                 # Core federated functionality
│   │   ├── gradients.py            # Gradient computation
│   │   └── orchestrators.py        # FL orchestration
│   ├── inference/           # High-throughput inference server
│   │   └── server.py               # REST API server implementation
│   ├── cli/                 # Command-line interfaces
│   │   └── ml_platforms.py         # ML platform CLI tools
│   ├── integrations/        # External framework integrations
│   │   ├── sklearn_integration.py  # scikit-learn compatibility
│   │   ├── dask_integration.py     # Dask distributed computing
│   │   ├── hpc_integration.py      # HPC workflow integration
│   │   └── jupyter_integration.py  # Jupyter notebook support
│   └── utils/               # Shared utilities
│       ├── memory.py        # Memory management
│       ├── parsing.py       # SMILES/molecular parsing
│       ├── validation.py    # Input validation
│       ├── geometry.py      # 3D geometry utilities
│       ├── config.py        # Configuration management
│       ├── logging.py       # Structured logging
│       └── distributed.py   # Distributed computing utilities
├── tests/                   # Test suite
│   ├── unit/               # Unit tests by module
│   ├── integration/        # Cross-module integration tests
│   ├── validation/         # Reference implementation validation
│   └── benchmarks/         # Performance benchmarks
├── docs/                   # Documentation
├── examples/               # Usage examples and tutorials
└── benchmarks/            # Standalone benchmark scripts
```

## Coding Conventions

### Module Structure
- Each backend in separate module with consistent interface
- Descriptor calculations grouped by complexity/type
- Clear separation between GPU kernels and Python wrappers

### Naming Patterns
- Classes: PascalCase (Calculator, MolecularGraph)
- Functions/methods: snake_case (calculate_descriptors, parse_smiles)
- Constants: UPPER_SNAKE_CASE (DEFAULT_BATCH_SIZE)
- Private members: leading underscore (_internal_method)

### Import Organization
- Standard library imports first
- Third-party imports second
- Local imports last
- Backend-specific imports isolated in respective modules

### Error Handling
- Custom exception hierarchy in core module
- Graceful degradation when GPU unavailable
- Clear error messages with suggested solutions
- Automatic fallback between backends when possible
