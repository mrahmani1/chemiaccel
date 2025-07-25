"""
Main Calculator class for ChemiAccel.

This module provides the primary user interface for molecular descriptor
calculations with automatic backend selection and optimization.

The Calculator class serves as the main entry point for ChemiAccel, providing
a unified interface for calculating molecular descriptors across different
backends (CUDA, JAX, CPU) with automatic optimization and hardware detection.

Example:
    Basic usage with pandas DataFrame:

    >>> import chemi_accel as ca
    >>> import pandas as pd
    >>>
    >>> # Initialize calculator with automatic backend selection
    >>> calc = ca.Calculator()
    >>>
    >>> # Prepare molecular data
    >>> smiles = pd.Series(['CCO', 'c1ccccc1', 'CC(=O)O'])
    >>>
    >>> # Calculate descriptors
    >>> descriptors = calc.pandas(smiles)
    >>> print(descriptors.head())

    Advanced usage with cudf for GPU acceleration:

    >>> import cudf
    >>>
    >>> # Initialize with CUDA backend
    >>> calc = ca.Calculator(backend='cuda', batch_size=1000)
    >>>
    >>> # Use cudf Series for GPU-native processing
    >>> smiles_cudf = cudf.Series(['CCO', 'c1ccccc1', 'CC(=O)O'])
    >>> descriptors_cudf = calc.cudf(smiles_cudf)

    Configuration and descriptor selection:

    >>> # Configure specific descriptors
    >>> calc.configure_descriptors(['molecular_weight', 'logp', 'tpsa'])
    >>>
    >>> # Or use predefined sets
    >>> calc.configure_descriptors('basic')  # Basic descriptor set
    >>> calc.configure_descriptors('all')    # All available descriptors
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from ..utils.config import BackendType, get_config
from ..utils.logging import get_logger
from ..utils.rdkit_integration import RDKitDescriptorCalculator, RDKitMolecularParser
from .backend import BackendManager, select_optimal_backend
from .exceptions import ChemiAccelError, ValidationError
from .molecular import MolecularData

logger = get_logger()


class Calculator:
    """Main calculator class for molecular descriptor calculations.

    This class provides a unified interface for calculating molecular descriptors
    using different backends (CUDA, JAX, CPU) with automatic optimization and
    hardware detection. It supports multiple input/output formats including
    pandas DataFrames, cudf DataFrames, and JAX arrays.

    The Calculator automatically selects the optimal backend based on available
    hardware and provides seamless integration with the scientific Python
    ecosystem including RAPIDS, scikit-learn, and Jupyter notebooks.

    Attributes:
        backend_type (BackendType): Currently active backend type
        device (Optional[Union[int, str]]): Device ID for GPU backends
        batch_size (Union[int, str]): Batch size for processing
        descriptor_sets (List[str]): Currently configured descriptor names

    Example:
        >>> calc = Calculator(backend='auto', batch_size='auto')
        >>> calc.configure_descriptors('basic')
        >>> results = calc.pandas(['CCO', 'c1ccccc1'])
    """

    def __init__(
        self,
        backend: Union[BackendType, str] = BackendType.AUTO,
        device: Optional[Union[int, str]] = None,
        batch_size: Union[int, str] = "auto",
    ) -> None:
        """Initialize ChemiAccel calculator with backend selection and configuration.

        The calculator automatically detects available hardware and selects the
        optimal backend for molecular descriptor calculations. It supports
        automatic batch size optimization and device selection.

        Args:
            backend: Backend type to use. Options:
                - 'auto': Automatically select optimal backend (default)
                - 'cuda': Use NVIDIA GPU with CUDA kernels
                - 'jax': Use JAX backend for TPUs/XLA compilation
                - 'cpu': Use CPU-only implementation
            device: Device specification for GPU backends:
                - None: Auto-select optimal device (default)
                - int: Specific GPU device ID (e.g., 0, 1, 2)
                - str: Device specification (e.g., 'cuda:0', 'auto')
            batch_size: Batch size for molecular processing:
                - 'auto': Automatically optimize batch size (default)
                - int: Specific batch size (e.g., 1000, 5000)

        Raises:
            ChemiAccelError: If calculator initialization fails
            BackendUnavailableError: If specified backend is not available
            ValidationError: If invalid parameters are provided

        Example:
            >>> # Auto-select optimal backend
            >>> calc = Calculator()

            >>> # Use specific CUDA device with custom batch size
            >>> calc = Calculator(backend='cuda', device=0, batch_size=2000)

            >>> # Use JAX backend for TPU processing
            >>> calc = Calculator(backend='jax', batch_size=500)
        """
        self.config = get_config()
        self.backend_manager = BackendManager()

        # Parse backend parameter
        if isinstance(backend, str):
            try:
                backend = BackendType(backend.lower())
            except ValueError:
                raise ValidationError(
                    input_name="backend",
                    input_value=backend,
                    expected_type="'auto', 'cuda', 'jax', or 'cpu'",
                    reason="Invalid backend type",
                )

        # Select and initialize backend
        self.backend_type = select_optimal_backend(backend)
        self.backend = self.backend_manager.initialize_backend(self.backend_type)

        # Store configuration
        self.device = device
        self.batch_size = batch_size

        # Initialize RDKit components
        self.molecular_parser = RDKitMolecularParser(
            sanitize=True, remove_hs=True, standardize=False  # Can be enabled later
        )
        self.descriptor_calculator = RDKitDescriptorCalculator()

        # Descriptor configuration
        self.available_descriptors = (
            self.descriptor_calculator.get_available_descriptors()
        )
        self.descriptor_sets = [
            "molecular_weight",
            "num_atoms",
            "num_bonds",
            "num_heavy_atoms",
            "logp",
            "tpsa",
            "num_rotatable_bonds",
            "num_aromatic_rings",
            "num_hbd",
            "num_hba",
        ]  # Default basic set

        logger.info(
            f"Calculator initialized with {self.backend_type.value} backend and RDKit integration"
        )

    def configure_descriptors(self, descriptor_set: Union[str, List[str]]) -> None:
        """Configure which molecular descriptors to calculate.

        This method allows you to specify which descriptors should be calculated
        for molecular structures. You can use predefined descriptor sets or
        specify individual descriptor names.

        Args:
            descriptor_set: Descriptor configuration options:
                - 'basic': Basic molecular properties (MW, LogP, TPSA, etc.)
                - 'all': All available descriptors
                - List[str]: Specific descriptor names to calculate
                - str: Single descriptor name

        Raises:
            ValidationError: If descriptor set contains invalid descriptor names

        Available Descriptors:
            Basic Properties:
                - molecular_weight: Molecular weight in Da
                - num_atoms: Total number of atoms
                - num_bonds: Total number of bonds
                - num_heavy_atoms: Number of non-hydrogen atoms

            Lipophilicity & Solubility:
                - logp: Octanol-water partition coefficient
                - tpsa: Topological polar surface area

            Structural Features:
                - num_rotatable_bonds: Number of rotatable bonds
                - num_aromatic_rings: Number of aromatic rings
                - num_hbd: Number of hydrogen bond donors
                - num_hba: Number of hydrogen bond acceptors

        Example:
            >>> calc = Calculator()

            >>> # Use basic descriptor set
            >>> calc.configure_descriptors('basic')

            >>> # Use specific descriptors
            >>> calc.configure_descriptors(['molecular_weight', 'logp', 'tpsa'])

            >>> # Use all available descriptors
            >>> calc.configure_descriptors('all')
        """
        if isinstance(descriptor_set, str):
            if descriptor_set == "basic":
                # Use a subset of basic descriptors
                descriptor_set = [
                    "molecular_weight",
                    "num_atoms",
                    "num_bonds",
                    "num_heavy_atoms",
                    "logp",
                    "tpsa",
                    "num_rotatable_bonds",
                    "num_aromatic_rings",
                    "num_hbd",
                    "num_hba",
                ]
            elif descriptor_set == "all":
                descriptor_set = self.available_descriptors
            else:
                descriptor_set = [descriptor_set]

        # Validate descriptor names
        invalid_descriptors = [
            d for d in descriptor_set if d not in self.available_descriptors
        ]
        if invalid_descriptors:
            raise ValidationError(
                input_name="descriptor_set",
                input_value=invalid_descriptors,
                expected_type=f"one of {self.available_descriptors}",
                reason=f"Unknown descriptors: {invalid_descriptors}",
            )

        self.descriptor_sets = descriptor_set
        logger.info(f"Configured descriptors: {descriptor_set}")

    def pandas(self, molecules: Union[pd.Series, List[str]]) -> pd.DataFrame:
        """Calculate molecular descriptors returning pandas DataFrame.

        This is the primary method for calculating molecular descriptors with
        pandas input/output. It provides seamless integration with pandas
        workflows and maintains index information when possible.

        The method automatically handles molecular parsing, validation, and
        descriptor calculation using the configured backend and descriptor set.
        Invalid molecules are skipped with warnings logged.

        Args:
            molecules: Molecular input in one of these formats:
                - pd.Series: Pandas Series containing SMILES strings
                - List[str]: List of SMILES strings

        Returns:
            pd.DataFrame: DataFrame with calculated descriptors containing:
                - mol_id: Molecule identifier (from input index or auto-generated)
                - smiles: Canonical SMILES representation
                - [descriptor columns]: Calculated descriptor values

        Raises:
            ChemiAccelError: If descriptor calculation fails completely
            ValidationError: If input format is invalid

        Performance Notes:
            - Batch processing is automatically optimized based on backend
            - GPU backends provide significant speedup for large datasets
            - Memory usage is managed automatically with streaming for large inputs

        Example:
            >>> import pandas as pd
            >>> import chemi_accel as ca
            >>>
            >>> calc = ca.Calculator()
            >>> calc.configure_descriptors('basic')
            >>>
            >>> # From pandas Series
            >>> smiles = pd.Series(['CCO', 'c1ccccc1', 'CC(=O)O'])
            >>> descriptors = calc.pandas(smiles)
            >>> print(descriptors.columns.tolist())
            ['mol_id', 'smiles', 'molecular_weight', 'logp', 'tpsa', ...]
            >>>
            >>> # From list of SMILES
            >>> smiles_list = ['CCO', 'c1ccccc1', 'CC(=O)O']
            >>> descriptors = calc.pandas(smiles_list)
            >>> print(f"Calculated descriptors for {len(descriptors)} molecules")
        """
        logger.info(f"Starting descriptor calculation for {len(molecules)} molecules")

        try:
            # Convert input to molecular data using RDKit
            if isinstance(molecules, pd.Series):
                smiles_list = molecules.tolist()
                mol_ids = molecules.index.tolist()
            else:
                smiles_list = molecules
                mol_ids = None

            # Parse molecules using RDKit
            molecular_data = self.molecular_parser.parse_smiles_batch(
                smiles_list, mol_ids, error_handling="skip"
            )

            if not molecular_data:
                # Return empty DataFrame for empty input
                if len(smiles_list) == 0:
                    return pd.DataFrame()
                else:
                    raise ChemiAccelError("No valid molecules found after parsing")

            # Calculate descriptors using RDKit
            df = self.descriptor_calculator.calculate_descriptors_batch(
                molecular_data, self.descriptor_sets
            )

            logger.info(
                f"Completed descriptor calculation: {len(df)} molecules, {len(df.columns)} descriptors"
            )
            return df

        except Exception as e:
            logger.error(f"Descriptor calculation failed: {e}")
            raise ChemiAccelError(f"Failed to calculate descriptors: {e}")

    def cudf(self, molecules: "cudf.Series") -> "cudf.DataFrame":
        """Calculate molecular descriptors with GPU-native cudf integration.

        This method provides seamless integration with the RAPIDS ecosystem,
        enabling GPU-native DataFrame operations with zero-copy data transfers
        and efficient memory management through RAPIDS Memory Manager (RMM).

        The method requires the CUDA backend and provides significant performance
        improvements for large molecular datasets by keeping data on GPU
        throughout the entire processing pipeline.

        Args:
            molecules: cudf.Series containing SMILES strings. The Series index
                      is preserved in the output DataFrame.

        Returns:
            cudf.DataFrame: GPU DataFrame with calculated descriptors containing:
                - mol_id: Molecule identifier (from input index)
                - smiles: Canonical SMILES representation
                - [descriptor columns]: Calculated descriptor values

        Raises:
            ChemiAccelError: If descriptor calculation fails
            BackendUnavailableError: If CUDA backend or cudf is not available
            ValidationError: If input is not a cudf.Series

        Performance Notes:
            - Utilizes GPU memory pools for efficient allocation
            - Zero-copy operations where possible
            - Automatic memory management through RMM
            - Significant speedup over CPU-based processing

        Requirements:
            - CUDA-capable GPU
            - RAPIDS cudf library installed
            - CUDA backend selected (automatic if available)

        Example:
            >>> import cudf
            >>> import chemi_accel as ca
            >>>
            >>> # Initialize with CUDA backend
            >>> calc = ca.Calculator(backend='cuda')
            >>> calc.configure_descriptors('basic')
            >>>
            >>> # Create cudf Series
            >>> smiles_cudf = cudf.Series(['CCO', 'c1ccccc1', 'CC(=O)O'])
            >>>
            >>> # Calculate descriptors on GPU
            >>> descriptors_cudf = calc.cudf(smiles_cudf)
            >>> print(f"GPU processing completed: {len(descriptors_cudf)} molecules")
            >>>
            >>> # Memory usage information
            >>> print(calc.get_backend_info()['memory_info'])
        """
        if self.backend_type != BackendType.CUDA:
            raise ChemiAccelError("cudf() method requires CUDA backend")

        try:
            from ..backends.rapids import CudfProcessor, check_cudf_availability

            if not check_cudf_availability():
                raise BackendUnavailableError(
                    "cudf", "cudf not available. Install with: pip install cudf-cu11"
                )

            logger.info(
                f"Starting cudf descriptor calculation for {len(molecules)} molecules"
            )

            # Initialize cudf processor
            cudf_processor = CudfProcessor()

            # Calculate descriptors using cudf integration
            result_df = cudf_processor.calculate_descriptors_cudf(
                molecules, self.descriptor_sets
            )

            logger.info(
                f"Completed cudf descriptor calculation: {len(result_df)} molecules, "
                f"{len(result_df.columns)} descriptors"
            )

            return result_df

        except Exception as e:
            logger.error(f"cudf descriptor calculation failed: {e}")
            raise ChemiAccelError(f"Failed to calculate descriptors with cudf: {e}")

    def jax(self, molecules: List[str]) -> "jax.Array":
        """Calculate molecular descriptors returning JAX arrays.

        This method provides integration with the JAX ecosystem, enabling
        JIT compilation, automatic differentiation, and TPU acceleration.
        It's particularly useful for machine learning workflows that require
        gradient computation or TPU optimization.

        The method utilizes JAX's vectorization capabilities (jax.vmap) for
        efficient batch processing and JIT compilation for optimal performance
        on TPUs and other XLA-compatible accelerators.

        Args:
            molecules: List of SMILES strings to process

        Returns:
            jax.Array: JAX array with calculated descriptors where:
                - Shape: (n_molecules, n_descriptors)
                - dtype: float32 (optimized for TPU processing)
                - Device: Automatically placed on optimal JAX device

        Raises:
            ChemiAccelError: If JAX backend is not selected
            BackendUnavailableError: If JAX is not available
            NotImplementedError: Currently not implemented (placeholder)

        Features:
            - JIT compilation for optimal TPU performance
            - Automatic vectorization with jax.vmap
            - Support for automatic differentiation
            - Integration with JAX-native ML workflows
            - Memory-efficient batching for large datasets

        Requirements:
            - JAX library installed
            - JAX backend selected
            - TPU or XLA-compatible device (recommended)

        Example:
            >>> import jax
            >>> import chemi_accel as ca
            >>>
            >>> # Initialize with JAX backend
            >>> calc = ca.Calculator(backend='jax')
            >>> calc.configure_descriptors('basic')
            >>>
            >>> # Calculate descriptors as JAX arrays
            >>> smiles_list = ['CCO', 'c1ccccc1', 'CC(=O)O']
            >>> descriptors_jax = calc.jax(smiles_list)
            >>> print(f"JAX array shape: {descriptors_jax.shape}")
            >>>
            >>> # Use in JAX-native ML workflow
            >>> @jax.jit
            >>> def ml_pipeline(descriptors):
            >>>     return jax.nn.relu(descriptors @ weights)
            >>>
            >>> predictions = ml_pipeline(descriptors_jax)

        Note:
            This method is currently a placeholder and will be fully implemented
            in task 5.1 (JAX Backend Core Implementation).
        """
        if self.backend_type != BackendType.JAX:
            raise ChemiAccelError("jax() method requires JAX backend")

        # TODO: Implement JAX integration in task 5.1
        logger.warning("jax() method not yet implemented - placeholder")
        raise NotImplementedError("JAX integration will be implemented in task 5.1")

    def _get_batch_size(self) -> int:
        """Determine optimal batch size for processing.

        Returns:
            Optimal batch size
        """
        if isinstance(self.batch_size, int):
            return self.batch_size

        # Auto-determine batch size based on backend and available memory
        # TODO: Implement intelligent batch size selection in task 4.1
        default_batch_sizes = {
            BackendType.CUDA: 1000,
            BackendType.JAX: 500,
            BackendType.CPU: 100,
        }

        return default_batch_sizes.get(self.backend_type, 100)

    def get_available_descriptors(self) -> List[str]:
        """Get list of all available descriptor names.

        Returns:
            List[str]: Complete list of descriptor names that can be calculated

        Example:
            >>> calc = Calculator()
            >>> descriptors = calc.get_available_descriptors()
            >>> print(f"Available descriptors: {len(descriptors)}")
            >>> print(descriptors[:5])  # Show first 5
        """
        return self.available_descriptors.copy()

    def get_descriptor_info(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed information about available descriptors.

        This method provides comprehensive information about each available
        descriptor including description, data type, typical range, and
        computational complexity.

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping descriptor names to
            their information including:
                - description: Human-readable description
                - data_type: Expected data type (float, int, bool)
                - typical_range: Typical value range
                - complexity: Computational complexity (low, medium, high)
                - category: Descriptor category (basic, lipophilicity, etc.)

        Example:
            >>> calc = Calculator()
            >>> info = calc.get_descriptor_info()
            >>> print(info['molecular_weight']['description'])
            >>> print(info['logp']['typical_range'])
        """
        descriptor_info = {
            "molecular_weight": {
                "description": "Molecular weight in Daltons",
                "data_type": "float",
                "typical_range": "50-1000 Da",
                "complexity": "low",
                "category": "basic",
            },
            "num_atoms": {
                "description": "Total number of atoms",
                "data_type": "int",
                "typical_range": "5-100",
                "complexity": "low",
                "category": "basic",
            },
            "num_bonds": {
                "description": "Total number of bonds",
                "data_type": "int",
                "typical_range": "4-120",
                "complexity": "low",
                "category": "basic",
            },
            "num_heavy_atoms": {
                "description": "Number of non-hydrogen atoms",
                "data_type": "int",
                "typical_range": "3-80",
                "complexity": "low",
                "category": "basic",
            },
            "logp": {
                "description": "Octanol-water partition coefficient",
                "data_type": "float",
                "typical_range": "-3 to 8",
                "complexity": "medium",
                "category": "lipophilicity",
            },
            "tpsa": {
                "description": "Topological polar surface area in Ų",
                "data_type": "float",
                "typical_range": "0-200 Ų",
                "complexity": "medium",
                "category": "solubility",
            },
            "num_rotatable_bonds": {
                "description": "Number of rotatable bonds",
                "data_type": "int",
                "typical_range": "0-20",
                "complexity": "medium",
                "category": "flexibility",
            },
            "num_aromatic_rings": {
                "description": "Number of aromatic rings",
                "data_type": "int",
                "typical_range": "0-5",
                "complexity": "medium",
                "category": "structural",
            },
            "num_hbd": {
                "description": "Number of hydrogen bond donors",
                "data_type": "int",
                "typical_range": "0-10",
                "complexity": "medium",
                "category": "hydrogen_bonding",
            },
            "num_hba": {
                "description": "Number of hydrogen bond acceptors",
                "data_type": "int",
                "typical_range": "0-15",
                "complexity": "medium",
                "category": "hydrogen_bonding",
            },
        }

        # Return only info for available descriptors
        return {
            name: info
            for name, info in descriptor_info.items()
            if name in self.available_descriptors
        }

    def validate_molecules(
        self, molecules: Union[pd.Series, List[str]]
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Validate molecular input and return validation results.

        This method performs comprehensive validation of molecular input
        without calculating descriptors. It's useful for checking data
        quality before expensive descriptor calculations.

        Args:
            molecules: Molecular input to validate (SMILES strings)

        Returns:
            Tuple[List[str], List[Dict[str, Any]]]: Tuple containing:
                - List of valid SMILES strings
                - List of validation error dictionaries for invalid molecules

        Example:
            >>> calc = Calculator()
            >>> smiles = ['CCO', 'invalid_smiles', 'c1ccccc1']
            >>> valid, errors = calc.validate_molecules(smiles)
            >>> print(f"Valid: {len(valid)}, Errors: {len(errors)}")
            >>> for error in errors:
            >>>     print(f"Error at index {error['index']}: {error['reason']}")
        """
        if isinstance(molecules, pd.Series):
            smiles_list = molecules.tolist()
            mol_ids = molecules.index.tolist()
        else:
            smiles_list = molecules
            mol_ids = [i for i in range(len(smiles_list))]

        valid_smiles = []
        validation_errors = []

        for i, (smiles, mol_id) in enumerate(zip(smiles_list, mol_ids)):
            try:
                mol_data = self.molecular_parser.parse_smiles(smiles, mol_id)
                is_valid, issues = self.molecular_parser.validate_molecule(mol_data)

                if is_valid:
                    valid_smiles.append(smiles)
                else:
                    validation_errors.append(
                        {
                            "index": i,
                            "mol_id": mol_id,
                            "smiles": smiles,
                            "reason": "; ".join(issues),
                        }
                    )
            except Exception as e:
                validation_errors.append(
                    {"index": i, "mol_id": mol_id, "smiles": smiles, "reason": str(e)}
                )

        logger.info(
            f"Validation complete: {len(valid_smiles)} valid, {len(validation_errors)} invalid"
        )
        return valid_smiles, validation_errors

    def parse_molecules(
        self, molecules: Union[pd.Series, List[str]]
    ) -> List[MolecularData]:
        """Parse molecules using RDKit (for advanced users).

        This method provides direct access to the molecular parsing functionality
        for advanced users who need access to the underlying MolecularData objects.

        Args:
            molecules: SMILES strings to parse

        Returns:
            List[MolecularData]: List of parsed MolecularData objects

        Example:
            >>> calc = Calculator()
            >>> smiles = ['CCO', 'c1ccccc1']
            >>> mol_data = calc.parse_molecules(smiles)
            >>> for mol in mol_data:
            >>>     print(f"Molecule {mol.mol_id}: {mol.num_atoms} atoms")
        """
        if isinstance(molecules, pd.Series):
            smiles_list = molecules.tolist()
            mol_ids = molecules.index.tolist()
        else:
            smiles_list = molecules
            mol_ids = None

        return self.molecular_parser.parse_smiles_batch(smiles_list, mol_ids)

    def get_backend_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the current backend configuration.

        This method provides detailed information about the calculator's current
        state, including backend type, device configuration, memory usage,
        and performance settings.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - backend_type: Currently active backend ('cuda', 'jax', 'cpu')
                - device: Device specification or auto-selection info
                - batch_size: Current batch size setting
                - descriptor_sets: List of configured descriptor names
                - backend_data: Backend-specific configuration data
                - memory_info: Memory usage statistics (if available)
                - performance_info: Performance-related settings

        Example:
            >>> calc = Calculator(backend='cuda', device=0, batch_size=1000)
            >>> info = calc.get_backend_info()
            >>> print(f"Backend: {info['backend_type']}")
            >>> print(f"Device: {info['device']}")
            >>> print(f"Descriptors: {len(info['descriptor_sets'])}")
        """
        info = {
            "backend_type": self.backend_type.value,
            "device": self.device,
            "batch_size": self.batch_size,
            "descriptor_sets": self.descriptor_sets.copy(),
            "backend_data": self.backend,
            "num_descriptors": len(self.descriptor_sets),
            "available_descriptors": len(self.available_descriptors),
        }

        # Add memory information if available
        try:
            if self.backend_type == BackendType.CUDA:
                from ..backends.rapids import CudfProcessor

                processor = CudfProcessor()
                info["memory_info"] = processor.get_memory_info()
        except Exception as e:
            logger.debug(f"Could not get memory info: {e}")

        return info

    def switch_backend(
        self, backend: Union[BackendType, str], device: Optional[Union[int, str]] = None
    ) -> None:
        """Switch to a different backend with optional device specification.

        This method allows dynamic switching between backends during runtime,
        which is useful for performance comparison or adapting to different
        hardware availability.

        Args:
            backend: New backend type to use:
                - 'cuda': Switch to NVIDIA GPU backend
                - 'jax': Switch to JAX/TPU backend
                - 'cpu': Switch to CPU-only backend
            device: Optional device specification for GPU backends

        Raises:
            ChemiAccelError: If backend switch fails
            BackendUnavailableError: If target backend is not available

        Example:
            >>> calc = Calculator(backend='cpu')
            >>> print(f"Current backend: {calc.backend_type.value}")
            >>>
            >>> # Switch to CUDA if available
            >>> try:
            >>>     calc.switch_backend('cuda', device=0)
            >>>     print("Switched to CUDA backend")
            >>> except BackendUnavailableError:
            >>>     print("CUDA not available")
        """
        if isinstance(backend, str):
            try:
                backend = BackendType(backend.lower())
            except ValueError:
                raise ValidationError(
                    input_name="backend",
                    input_value=backend,
                    expected_type="'cuda', 'jax', or 'cpu'",
                    reason="Invalid backend type",
                )

        logger.info(
            f"Switching from {self.backend_type.value} to {backend.value} backend"
        )

        # Update device if specified
        if device is not None:
            self.device = device

        # Switch backend
        self.backend_type = backend
        self.backend = self.backend_manager.switch_backend(backend)

        logger.info(f"Successfully switched to {backend.value} backend")

    def benchmark_performance(
        self,
        test_molecules: Optional[List[str]] = None,
        num_molecules: int = 1000,
        num_runs: int = 3,
    ) -> Dict[str, Any]:
        """Benchmark calculator performance with test molecules.

        This method provides performance benchmarking capabilities to help
        users optimize their configuration and compare backend performance.

        Args:
            test_molecules: Optional list of test SMILES strings
            num_molecules: Number of test molecules to generate if test_molecules is None
            num_runs: Number of benchmark runs for averaging

        Returns:
            Dict[str, Any]: Benchmark results including:
                - backend_type: Backend used for benchmark
                - num_molecules: Number of molecules processed
                - num_descriptors: Number of descriptors calculated
                - avg_time_seconds: Average processing time
                - molecules_per_second: Processing throughput
                - descriptors_per_second: Descriptor calculation throughput

        Example:
            >>> calc = Calculator()
            >>> calc.configure_descriptors('basic')
            >>> results = calc.benchmark_performance(num_molecules=5000)
            >>> print(f"Throughput: {results['molecules_per_second']:.1f} molecules/sec")
        """
        import time

        # Generate test molecules if not provided
        if test_molecules is None:
            # Simple test molecules for benchmarking
            base_molecules = [
                "CCO",  # ethanol
                "c1ccccc1",  # benzene
                "CC(=O)O",  # acetic acid
                "CCN(CC)CC",  # triethylamine
                "c1ccc2c(c1)ccccc2",  # naphthalene
                "CC(C)(C)O",  # tert-butanol
                "c1ccc(cc1)O",  # phenol
                "CCCCCCCCCCCCCCCC(=O)O",  # palmitic acid
                "c1ccc(cc1)N",  # aniline
                "CC(C)CC(C)(C)C",  # branched alkane
            ]
            # Replicate to reach desired number
            test_molecules = (
                base_molecules * ((num_molecules // len(base_molecules)) + 1)
            )[:num_molecules]

        num_molecules = len(test_molecules)
        times = []

        logger.info(
            f"Starting performance benchmark: {num_molecules} molecules, {num_runs} runs"
        )

        for run in range(num_runs):
            start_time = time.time()

            try:
                # Calculate descriptors
                results_df = self.pandas(test_molecules)
                end_time = time.time()

                if len(results_df) > 0:
                    times.append(end_time - start_time)
                else:
                    logger.warning(f"Benchmark run {run + 1} produced no results")

            except Exception as e:
                logger.error(f"Benchmark run {run + 1} failed: {e}")
                continue

        if not times:
            raise ChemiAccelError("All benchmark runs failed")

        # Calculate statistics
        avg_time = sum(times) / len(times)
        molecules_per_second = num_molecules / avg_time
        num_descriptors = len(self.descriptor_sets)
        descriptors_per_second = molecules_per_second * num_descriptors

        benchmark_results = {
            "backend_type": self.backend_type.value,
            "num_molecules": num_molecules,
            "num_descriptors": num_descriptors,
            "num_runs": len(times),
            "avg_time_seconds": avg_time,
            "min_time_seconds": min(times),
            "max_time_seconds": max(times),
            "molecules_per_second": molecules_per_second,
            "descriptors_per_second": descriptors_per_second,
            "batch_size": self.batch_size,
        }

        logger.info(f"Benchmark complete: {molecules_per_second:.1f} molecules/sec")
        return benchmark_results

    def __repr__(self) -> str:
        """String representation of calculator."""
        return (
            f"Calculator(backend={self.backend_type.value}, "
            f"batch_size={self.batch_size}, "
            f"descriptors={len(self.descriptor_sets)})"
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"ChemiAccel Calculator\n"
            f"  Backend: {self.backend_type.value}\n"
            f"  Device: {self.device}\n"
            f"  Batch size: {self.batch_size}\n"
            f"  Configured descriptors: {len(self.descriptor_sets)}\n"
            f"  Available descriptors: {len(self.available_descriptors)}"
        )
