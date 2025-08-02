# API Reference

Complete API documentation for ChemiAccel, covering all classes, methods, and functions available in the library.

## Core API

### Calculator Class

The main entry point for all ChemiAccel functionality.

```python
class Calculator:
    """Main calculator class for molecular descriptor computation."""
    
    def __init__(self, backend: str = "auto", config: Optional[ChemiAccelConfig] = None):
        """Initialize calculator with specified backend.
        
        Args:
            backend: Computational backend ("auto", "cpu", "cuda", "jax", "rapids")
            config: Configuration object for advanced settings
        """
```

#### Methods

##### pandas()
```python
def pandas(self, 
           molecules: Union[List[str], pd.Series, np.ndarray],
           descriptors: Union[str, List[str]] = "default",
           batch_size: Optional[int] = None,
           show_progress: bool = False,
           **kwargs) -> pd.DataFrame:
    """Calculate descriptors and return as pandas DataFrame.
    
    Args:
        molecules: SMILES strings or molecular representations
        descriptors: Descriptor names or categories to calculate
        batch_size: Number of molecules to process per batch
        show_progress: Show progress bar for large datasets
        **kwargs: Additional backend-specific options
        
    Returns:
        DataFrame with molecules as rows and descriptors as columns
        
    Raises:
        InvalidMoleculeError: If invalid SMILES detected
        BackendError: If backend computation fails
        MemoryError: If insufficient memory available
    """
```

##### numpy()
```python
def numpy(self,
          molecules: Union[List[str], pd.Series, np.ndarray],
          descriptors: Union[str, List[str]] = "default",
          batch_size: Optional[int] = None,
          **kwargs) -> Dict[str, np.ndarray]:
    """Calculate descriptors and return as numpy arrays.
    
    Args:
        molecules: SMILES strings or molecular representations
        descriptors: Descriptor names or categories to calculate
        batch_size: Number of molecules to process per batch
        **kwargs: Additional backend-specific options
        
    Returns:
        Dictionary mapping descriptor names to numpy arrays
    """
```

##### cudf()
```python
def cudf(self,
         molecules: Union[List[str], cudf.Series],
         descriptors: Union[str, List[str]] = "default",
         batch_size: Optional[int] = None,
         **kwargs) -> cudf.DataFrame:
    """Calculate descriptors and return as cuDF DataFrame (GPU).
    
    Args:
        molecules: SMILES strings or molecular representations
        descriptors: Descriptor names or categories to calculate
        batch_size: Number of molecules to process per batch
        **kwargs: Additional backend-specific options
        
    Returns:
        cuDF DataFrame with results on GPU
        
    Note:
        Requires RAPIDS backend and NVIDIA GPU
    """
```

##### jax()
```python
def jax(self,
        molecules: Union[List[str], jax.Array],
        descriptors: Union[str, List[str]] = "default",
        batch_size: Optional[int] = None,
        **kwargs) -> Dict[str, jax.Array]:
    """Calculate descriptors and return as JAX arrays.
    
    Args:
        molecules: SMILES strings or molecular representations
        descriptors: Descriptor names or categories to calculate
        batch_size: Number of molecules to process per batch
        **kwargs: Additional backend-specific options
        
    Returns:
        Dictionary mapping descriptor names to JAX arrays
    """
```

#### Properties

```python
@property
def backend(self) -> str:
    """Current computational backend."""

@property
def available_backends(self) -> List[str]:
    """List of available backends on current system."""

@property
def available_descriptors(self) -> List[str]:
    """List of all available descriptor names."""

@property
def descriptor_categories(self) -> Dict[str, List[str]]:
    """Dictionary mapping categories to descriptor lists."""
```

## Configuration API

### ChemiAccelConfig

```python
@dataclass
class ChemiAccelConfig:
    """Main configuration class for ChemiAccel."""
    
    backend: str = "auto"
    max_memory_gb: Optional[int] = None
    num_threads: Optional[int] = None
    batch_size: Optional[int] = None
    enable_cache: bool = True
    cache_size_mb: int = 2048
    backend_config: Optional[BackendConfig] = None
    performance_config: Optional[PerformanceConfig] = None
    logging_config: Optional[LoggingConfig] = None
```

#### Methods

```python
def validate(self) -> ValidationResult:
    """Validate configuration parameters."""

def to_dict(self) -> Dict[str, Any]:
    """Convert configuration to dictionary."""

@classmethod
def from_dict(cls, config_dict: Dict[str, Any]) -> 'ChemiAccelConfig':
    """Create configuration from dictionary."""

@classmethod
def from_yaml(cls, yaml_path: str) -> 'ChemiAccelConfig':
    """Load configuration from YAML file."""
```

### Backend-Specific Configurations

#### CudaConfig
```python
@dataclass
class CudaConfig:
    """CUDA backend configuration."""
    
    device_ids: List[int] = field(default_factory=lambda: [0])
    memory_pool_mb: int = 8192
    streams: int = 4
    batch_size: int = 5000
    enable_peer_access: bool = True
    memory_growth: bool = True
    synchronize_streams: bool = False
```

#### JaxConfig
```python
@dataclass
class JaxConfig:
    """JAX backend configuration."""
    
    platform: str = "gpu"  # "gpu", "tpu", "cpu"
    preallocate: bool = False
    memory_fraction: float = 0.75
    enable_x64: bool = False
    jit_cache_size: int = 1000
    device_count: int = 1
```

#### CpuConfig
```python
@dataclass
class CpuConfig:
    """CPU backend configuration."""
    
    num_threads: int = -1  # Use all available cores
    batch_size: int = 1000
    enable_mkl: bool = True
    memory_mapping: bool = True
    numa_policy: str = "interleave"
```

## Descriptor API

### Descriptor Categories

```python
# Constitutional descriptors
CONSTITUTIONAL_DESCRIPTORS = [
    "molecular_weight",
    "num_atoms", 
    "num_heavy_atoms",
    "num_bonds",
    "num_rings",
    # ... more descriptors
]

# Physicochemical descriptors  
PHYSICOCHEMICAL_DESCRIPTORS = [
    "logp",
    "slogp", 
    "mlogp",
    "tpsa",
    "asa",
    # ... more descriptors
]

# Fingerprint descriptors
FINGERPRINT_DESCRIPTORS = [
    "morgan_fingerprint_1024",
    "morgan_fingerprint_2048", 
    "maccs_keys",
    "atom_pair_fingerprint",
    # ... more descriptors
]
```

### Custom Descriptor API

```python
class CustomDescriptor:
    """Base class for custom descriptors."""
    
    name: str
    description: str = ""
    return_type: type = float
    
    def calculate(self, mol: Chem.Mol) -> Union[float, int, np.ndarray]:
        """Calculate descriptor for a single molecule.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Calculated descriptor value(s)
        """
        raise NotImplementedError
    
    def batch_calculate(self, mols: List[Chem.Mol]) -> np.ndarray:
        """Calculate descriptor for multiple molecules.
        
        Args:
            mols: List of RDKit molecule objects
            
        Returns:
            Array of calculated values
        """
        return np.array([self.calculate(mol) for mol in mols])
```

#### Example Custom Descriptor

```python
class AromaticRatio(CustomDescriptor):
    """Ratio of aromatic atoms to total atoms."""
    
    name = "aromatic_ratio"
    description = "Fraction of atoms that are aromatic"
    return_type = float
    
    def calculate(self, mol: Chem.Mol) -> float:
        if mol is None:
            return np.nan
        
        num_aromatic = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        num_total = mol.GetNumAtoms()
        
        return num_aromatic / num_total if num_total > 0 else 0.0
```

## Polymer Science API

### CopolymerAnalyzer

```python
class CopolymerAnalyzer:
    """Analyzer for copolymer properties and composition."""
    
    def __init__(self, config: Optional[PolymerConfig] = None):
        """Initialize copolymer analyzer."""
```

#### Methods

```python
def analyze_copolymer_composition(self,
                                monomers: List[str],
                                ratios: List[float]) -> Dict[str, Any]:
    """Analyze copolymer composition and predict properties.
    
    Args:
        monomers: List of monomer SMILES
        ratios: Molar ratios of each monomer
        
    Returns:
        Dictionary with composition analysis and property predictions
    """

def predict_glass_transition_temperature(self,
                                       monomer_a: str,
                                       monomer_b: str,
                                       ratio_a: float,
                                       ratio_b: float) -> float:
    """Predict glass transition temperature using Fox equation."""

def predict_mechanical_properties(self,
                                monomer_a: str,
                                monomer_b: str,
                                ratio_a: float,
                                ratio_b: float) -> Dict[str, float]:
    """Predict mechanical properties of copolymer."""
```

### MonomerPredictor

```python
class MonomerPredictor:
    """Predictor for monomer reactivity and polymerization behavior."""
    
    def predict_reactivity(self, monomer_smiles: str) -> Dict[str, float]:
        """Predict monomer reactivity parameters."""
    
    def predict_radical_polymerization_rate(self, monomer_smiles: str) -> float:
        """Predict radical polymerization rate constant."""
```

## Materials Science API

### MaterialsCalculator

```python
class MaterialsCalculator:
    """Calculator for crystal structures and materials properties."""
    
    def __init__(self, use_gpu: bool = True, config: Optional[MaterialsConfig] = None):
        """Initialize materials calculator."""
```

#### Methods

```python
def from_cif_file(self, cif_path: str) -> pd.DataFrame:
    """Load crystal structure from CIF file and calculate descriptors."""

def from_poscar_file(self, poscar_path: str) -> pd.DataFrame:
    """Load crystal structure from POSCAR file and calculate descriptors."""

def calculate_structure_descriptors(self, structure) -> Dict[str, Any]:
    """Calculate structural descriptors for crystal structure."""

def calculate_electronic_properties(self, 
                                   structure,
                                   method: str = "DFT-PBE") -> Dict[str, float]:
    """Calculate electronic properties using specified method."""

def calculate_mechanical_properties(self, structure) -> Dict[str, float]:
    """Calculate mechanical properties (bulk modulus, shear modulus, etc.)."""
```

### CrystalStructureAnalyzer

```python
class CrystalStructureAnalyzer:
    """Analyzer for crystal structure properties."""
    
    def analyze_symmetry(self, structure) -> Dict[str, Any]:
        """Analyze crystal symmetry and space group."""
    
    def calculate_coordination_numbers(self, structure) -> np.ndarray:
        """Calculate coordination numbers for all atoms."""
    
    def identify_structure_type(self, structure) -> str:
        """Identify crystal structure type (perovskite, spinel, etc.)."""
```

## AI and Machine Learning API

### MultiObjectiveOptimizer

```python
class MultiObjectiveOptimizer:
    """Multi-objective optimization for molecular design."""
    
    def optimize(self,
                evaluate_function: Callable,
                initial_population: List[str],
                objectives: List[str],
                population_size: int = 50,
                num_generations: int = 100,
                **kwargs) -> OptimizationResult:
        """Run multi-objective optimization."""
```

#### OptimizationResult

```python
@dataclass
class OptimizationResult:
    """Result of multi-objective optimization."""
    
    pareto_front: List[Dict[str, Any]]
    objectives: np.ndarray
    parameters: np.ndarray
    hypervolume: float
    convergence_history: List[float]
    generation_data: List[Dict[str, Any]]
```

### MolecularGAN

```python
class MolecularGAN:
    """Generative Adversarial Network for molecular generation."""
    
    def train(self,
              training_molecules: List[str],
              training_descriptors: pd.DataFrame,
              epochs: int = 100,
              batch_size: int = 64,
              **kwargs) -> Dict[str, List[float]]:
        """Train the molecular GAN."""
    
    def generate(self,
                 num_molecules: int,
                 temperature: float = 0.8,
                 filter_valid: bool = True) -> List[str]:
        """Generate new molecules."""
```

### PropertyPredictor

```python
class PropertyPredictor:
    """Neural network-based property predictor."""
    
    def train(self,
              model: NeuralNetworkModel,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              **kwargs) -> Dict[str, List[float]]:
        """Train property prediction model."""
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
    
    def get_feature_importance(self,
                              X: np.ndarray,
                              y: np.ndarray,
                              method: str = "permutation") -> np.ndarray:
        """Calculate feature importance scores."""
```

## Cloud Computing API

### AWS Integration

#### SageMakerJob

```python
class SageMakerJob:
    """Integration with AWS SageMaker for distributed computing."""
    
    def submit_job(self, config: Dict[str, Any]) -> str:
        """Submit job to SageMaker."""
    
    def get_job_status(self, job_name: str) -> str:
        """Get current job status."""
    
    def get_job_logs(self, job_name: str) -> str:
        """Retrieve job logs."""
```

#### S3DataManager

```python
class S3DataManager:
    """Manager for S3 data operations."""
    
    def upload_dataframe(self, dataframe: pd.DataFrame, key: str) -> str:
        """Upload DataFrame to S3."""
    
    def download_dataframe(self, s3_uri: str) -> pd.DataFrame:
        """Download DataFrame from S3."""
```

### Google Cloud Integration

#### VertexAIJob

```python
class VertexAIJob:
    """Integration with Google Cloud Vertex AI."""
    
    def submit_training_job(self, config: Dict[str, Any]) -> str:
        """Submit training job to Vertex AI."""
    
    def submit_batch_prediction_job(self, config: Dict[str, Any]) -> str:
        """Submit batch prediction job."""
```

### Azure Integration

#### AzureMLJob

```python
class AzureMLJob:
    """Integration with Azure Machine Learning."""
    
    def submit_experiment(self, config: Dict[str, Any]) -> str:
        """Submit experiment to Azure ML."""
    
    def deploy_model(self, model_config: Dict[str, Any]) -> str:
        """Deploy model as web service."""
```

## Visualization API

### MultiObjectiveVisualizer

```python
class MultiObjectiveVisualizer:
    """Visualizer for multi-objective optimization results."""
    
    def plot_pareto_front_2d(self,
                           objectives: np.ndarray,
                           pareto_mask: np.ndarray,
                           objective_names: List[str],
                           **kwargs) -> go.Figure:
        """Create 2D Pareto front plot."""
    
    def plot_pareto_front_3d(self,
                           objectives: np.ndarray,
                           pareto_mask: np.ndarray,
                           objective_names: List[str],
                           **kwargs) -> go.Figure:
        """Create 3D Pareto front plot."""
    
    def create_interactive_dashboard(self,
                                   result: OptimizationResult,
                                   **kwargs) -> Dict[str, go.Figure]:
        """Create comprehensive optimization dashboard."""
```

### MolecularGenerationVisualizer

```python
class MolecularGenerationVisualizer:
    """Visualizer for molecular generation results."""
    
    def create_molecule_gallery(self,
                              molecules: List[GeneratedMolecule],
                              sort_by: str = "generation_score",
                              n_display: int = 20) -> go.Figure:
        """Create interactive molecule gallery."""
    
    def plot_property_distributions(self,
                                  molecules: List[GeneratedMolecule],
                                  properties: List[str]) -> go.Figure:
        """Plot property distributions."""
```

### PolymerPropertyDashboard

```python
class PolymerPropertyDashboard:
    """Dashboard for polymer property analysis."""
    
    def create_property_overview_dashboard(self,
                                         polymers: List[PolymerData]) -> go.Figure:
        """Create property overview dashboard."""
    
    def create_composition_property_plot(self,
                                       copolymers: List[CopolymerData],
                                       property_name: str) -> go.Figure:
        """Create composition-property relationship plot."""
```

## Error Handling API

### Exception Classes

```python
class ChemiAccelError(Exception):
    """Base exception for ChemiAccel errors."""
    pass

class InvalidMoleculeError(ChemiAccelError):
    """Raised when invalid molecule is encountered."""
    pass

class BackendError(ChemiAccelError):
    """Raised when backend computation fails."""
    pass

class ConfigurationError(ChemiAccelError):
    """Raised when configuration is invalid."""
    pass

class MemoryError(ChemiAccelError):
    """Raised when insufficient memory is available."""
    pass

class CloudServiceError(ChemiAccelError):
    """Raised when cloud service operation fails."""
    pass
```

## Utility Functions

### Data Processing

```python
def validate_smiles(smiles: str) -> bool:
    """Validate SMILES string."""

def canonicalize_smiles(smiles: str) -> str:
    """Canonicalize SMILES string."""

def remove_duplicates(molecules: List[str]) -> List[str]:
    """Remove duplicate molecules from list."""

def filter_by_molecular_weight(molecules: List[str], 
                             min_weight: float = 0,
                             max_weight: float = float('inf')) -> List[str]:
    """Filter molecules by molecular weight range."""
```

### System Information

```python
def system_info() -> Dict[str, Any]:
    """Get system information for diagnostics."""

def check_gpu_availability() -> Dict[str, bool]:
    """Check availability of different GPU backends."""

def benchmark_backends(molecules: List[str], 
                      descriptors: List[str]) -> Dict[str, float]:
    """Benchmark performance of available backends."""
```

### File I/O

```python
def load_molecules_from_csv(filepath: str, 
                           smiles_column: str = "smiles") -> List[str]:
    """Load molecules from CSV file."""

def load_molecules_from_sdf(filepath: str) -> List[str]:
    """Load molecules from SDF file."""

def save_results_to_csv(results: pd.DataFrame, filepath: str) -> None:
    """Save results to CSV file."""

def save_results_to_parquet(results: pd.DataFrame, filepath: str) -> None:
    """Save results to Parquet file."""
```

## Constants and Enumerations

### Descriptor Categories

```python
from enum import Enum

class DescriptorCategory(Enum):
    """Enumeration of descriptor categories."""
    CONSTITUTIONAL = "constitutional"
    PHYSICOCHEMICAL = "physicochemical"
    TOPOLOGICAL = "topological"
    ELECTRONIC = "electronic"
    GEOMETRIC_3D = "geometric_3d"
    FINGERPRINTS = "fingerprints"
    PHARMACOPHORE = "pharmacophore"
    ALL = "all"
```

### Backend Types

```python
class BackendType(Enum):
    """Enumeration of backend types."""
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    JAX = "jax"
    RAPIDS = "rapids"
```

### Physical Constants

```python
# Physical constants used in calculations
AVOGADRO_NUMBER = 6.02214076e23
PLANCK_CONSTANT = 6.62607015e-34
BOLTZMANN_CONSTANT = 1.380649e-23
GAS_CONSTANT = 8.314462618
```

## Version Information

```python
def get_version() -> str:
    """Get ChemiAccel version."""

def get_build_info() -> Dict[str, str]:
    """Get build information."""

def check_dependencies() -> Dict[str, bool]:
    """Check status of optional dependencies."""
```

---

*For detailed examples of API usage, see the [Examples](Examples) gallery and domain-specific guides.*
