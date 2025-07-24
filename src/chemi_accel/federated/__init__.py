"""
ChemiAccel Federated Learning and Privacy-Preserving Computing - Task 21.1

This package provides comprehensive federated learning infrastructure for 
collaborative molecular computation while preserving privacy and security.
It enables secure multi-party computation across pharmaceutical companies,
research institutions, and other organizations without exposing sensitive
molecular data or proprietary datasets.

Key Components:
- FederatedCalculator: Secure local processing with privacy preservation
- Orchestrator integrations: PySyft, Flower, and native ChemiAccel coordination
- Privacy-preserving gradient computation without molecular structure exposure
- Differential privacy, encryption, and data anonymization capabilities
- Comprehensive privacy auditing and budget tracking
- Secure aggregation protocols for collaborative learning

Scientific Applications:
- Multi-institutional drug discovery collaborations
- Privacy-preserving chemical database analysis  
- Federated molecular property prediction models
- Secure consortium learning for proprietary datasets
- Collaborative reaction prediction across organizations

Example Usage:
    ```python
    from chemi_accel.federated import (
        FederatedCalculator, FederatedConfig, PrivacyConfig,
        create_federated_coordinator
    )
    
    # Configure privacy settings
    privacy_config = PrivacyConfig(
        epsilon=1.0,  # Privacy budget
        enable_differential_privacy=True,
        enable_encryption=True,
        enable_data_anonymization=True
    )
    
    # Create federated calculator
    federated_config = FederatedConfig(privacy_config=privacy_config)
    calculator = FederatedCalculator(federated_config)
    
    # Load local molecular data
    smiles_data = ['CCO', 'c1ccccc1', 'CC(=O)O']
    calculator.load_local_data(smiles_data, anonymize=True)
    
    # Compute descriptors with privacy preservation
    results = calculator.compute_local_descriptors(add_privacy_noise=True)
    
    # Create federated coordinator
    coordinator = create_federated_coordinator(federated_config)
    await coordinator.start_federation()
    
    # Run federated training
    round_results = await coordinator.run_federated_training(num_rounds=10)
    ```

Security Notice:
This implementation provides research-grade privacy preservation mechanisms.
For production deployment in sensitive environments, additional security
audits, formal privacy analysis, and compliance validation may be required.
"""

# Core federated learning components
from .core import (
    FederatedCalculator,
    FederatedConfig,
    PrivacyConfig,
    PrivacyPreservingNoise,
    SecureEncryption,
    DataAnonymizer,
    PrivacyAuditor,
    FederatedError,
    PrivacyError,
    EncryptionError
)

# Orchestrator integrations
from .orchestrators import (
    FederatedOrchestrator,
    PySyftOrchestrator,
    FlowerOrchestrator,
    ChemiAccelOrchestrator,
    FederatedCoordinator,
    ClientInfo,
    RoundResults,
    OrchestrationError,
    create_pysyft_orchestrator,
    create_flower_orchestrator,
    create_chemi_accel_orchestrator,
    create_federated_coordinator
)

# Privacy-preserving gradient computation
from .gradients import (
    PrivacyPreservingGradientComputer,
    GradientComputationConfig,
    SecureGradientAggregator,
    MolecularEmbedding,
    GradientPrivacyError,
    create_privacy_preserving_gradient_computer,
    create_secure_gradient_aggregator
)

# Version and metadata
__version__ = "1.0.0"
__author__ = "ChemiAccel Team"
__email__ = "info@chemi-accel.com"

# Package-level convenience functions
def create_secure_federation(privacy_epsilon: float = 1.0,
                           enable_encryption: bool = True,
                           enable_anonymization: bool = True,
                           orchestrator_type: str = "chemi_accel") -> tuple:
    """Create a complete secure federation setup with privacy preservation.
    
    Args:
        privacy_epsilon: Privacy budget (lower = more private)
        enable_encryption: Enable homomorphic encryption
        enable_anonymization: Enable data anonymization
        orchestrator_type: Type of orchestrator ('chemi_accel', 'pysyft', 'flower')
        
    Returns:
        Tuple of (FederatedCalculator, FederatedCoordinator)
    """
    # Create privacy configuration
    privacy_config = PrivacyConfig(
        epsilon=privacy_epsilon,
        enable_differential_privacy=True,
        enable_encryption=enable_encryption,
        enable_data_anonymization=enable_anonymization,
        enable_privacy_auditing=True
    )
    
    # Create federated configuration
    federated_config = FederatedConfig(privacy_config=privacy_config)
    
    # Create calculator
    calculator = FederatedCalculator(federated_config)
    
    # Create coordinator
    coordinator = create_federated_coordinator(
        federated_config, 
        orchestrator_types=[orchestrator_type]
    )
    
    return calculator, coordinator


def validate_privacy_setup(calculator: FederatedCalculator) -> dict:
    """Validate that privacy mechanisms are properly configured.
    
    Args:
        calculator: FederatedCalculator instance to validate
        
    Returns:
        Dictionary with validation results
    """
    return calculator.validate_privacy_preservation()


def get_privacy_report(calculator: FederatedCalculator) -> dict:
    """Get comprehensive privacy usage report.
    
    Args:
        calculator: FederatedCalculator instance
        
    Returns:
        Privacy usage report
    """
    return calculator.get_privacy_report()


# Export all public components
__all__ = [
    # Core components
    "FederatedCalculator",
    "FederatedConfig", 
    "PrivacyConfig",
    "PrivacyPreservingNoise",
    "SecureEncryption",
    "DataAnonymizer",
    "PrivacyAuditor",
    
    # Orchestrators
    "FederatedOrchestrator",
    "PySyftOrchestrator",
    "FlowerOrchestrator", 
    "ChemiAccelOrchestrator",
    "FederatedCoordinator",
    "ClientInfo",
    "RoundResults",
    
    # Gradient computation
    "PrivacyPreservingGradientComputer",
    "GradientComputationConfig",
    "SecureGradientAggregator",
    "MolecularEmbedding",
    
    # Exceptions
    "FederatedError",
    "PrivacyError",
    "EncryptionError",
    "OrchestrationError",
    "GradientPrivacyError",
    
    # Factory functions
    "create_pysyft_orchestrator",
    "create_flower_orchestrator",
    "create_chemi_accel_orchestrator",
    "create_federated_coordinator",
    "create_privacy_preserving_gradient_computer",
    "create_secure_gradient_aggregator",
    
    # Convenience functions
    "create_secure_federation",
    "validate_privacy_setup",
    "get_privacy_report"
]