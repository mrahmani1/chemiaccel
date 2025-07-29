"""
GPU-Accelerated Predictive Spectroscopy Models.

This module implements sophisticated machine learning models for predicting various types
of molecular spectra from 3D structures, leveraging GPU acceleration and quantum machine
learning techniques for high-throughput spectral predictions.
"""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Optional imports with fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.cuda.amp import GradScaler, autocast

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. GPU acceleration will be limited.")

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    warnings.warn("CuPy not available. GPU acceleration will be limited.")

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn("RDKit not available. Molecular feature extraction will be limited.")

try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, vmap

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    warnings.warn("JAX not available. Some QML features will be unavailable.")


# =============================================================================
# Enumerations
# =============================================================================


class SpectroscopyType(Enum):
    """Types of spectroscopy techniques."""

    NMR_1H = "1h_nmr"
    NMR_13C = "13c_nmr"
    NMR_2D = "2d_nmr"
    IR = "infrared"
    MS = "mass_spectrometry"
    UV_VIS = "uv_visible"
    RAMAN = "raman"
    FLUORESCENCE = "fluorescence"


class ModelArchitecture(Enum):
    """Neural network architectures for spectral prediction."""

    TRANSFORMER = "transformer"
    CONVOLUTIONAL = "convolutional"
    GRAPH_NEURAL = "graph_neural"
    RECURRENT = "recurrent"
    ATTENTION = "attention"
    HYBRID = "hybrid"


class QuantumMethod(Enum):
    """Quantum chemistry methods for feature generation."""

    DFT_B3LYP = "dft_b3lyp"
    DFT_PBE = "dft_pbe"
    SEMI_EMPIRICAL = "semi_empirical"
    TIGHT_BINDING = "tight_binding"
    ML_SURROGATE = "ml_surrogate"
    APPROXIMATE_DFT = "approximate_dft"


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class SpectralPrediction:
    """Container for spectral prediction results."""

    frequencies: np.ndarray
    intensities: np.ndarray
    spectroscopy_type: SpectroscopyType
    confidence_intervals: Optional[np.ndarray] = None
    peak_assignments: Optional[Dict[float, str]] = None
    quantum_features: Optional[Dict[str, float]] = None
    computation_time: float = 0.0
    model_version: str = "1.0"

    def __post_init__(self):
        """Validate prediction data."""
        if len(self.frequencies) != len(self.intensities):
            raise ValueError("Frequencies and intensities must have same length")

        if self.confidence_intervals is not None:
            if self.confidence_intervals.shape[0] != len(self.frequencies):
                raise ValueError("Confidence intervals must match frequency array")


@dataclass
class ModelConfiguration:
    """Configuration for spectral prediction models."""

    architecture: ModelArchitecture
    spectroscopy_type: SpectroscopyType
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    dropout_rate: float = 0.1
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100
    use_gpu: bool = True
    quantum_method: QuantumMethod = QuantumMethod.ML_SURROGATE
    frequency_range: Tuple[float, float] = (0.0, 4000.0)
    frequency_resolution: float = 1.0

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.dropout_rate < 0 or self.dropout_rate > 1:
            raise ValueError("Dropout rate must be between 0 and 1")
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")


@dataclass
class TrainingData:
    """Container for model training data."""

    molecular_features: np.ndarray
    spectra: np.ndarray
    frequencies: np.ndarray
    molecule_ids: Optional[List[str]] = None
    weights: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate training data."""
        if len(self.molecular_features) != len(self.spectra):
            raise ValueError("Features and spectra must have same number of samples")

        if self.weights is not None and len(self.weights) != len(self.spectra):
            raise ValueError("Weights must match number of samples")


@dataclass
class PredictionResult:
    """Container for batch prediction results."""

    predictions: List[SpectralPrediction]
    batch_computation_time: float
    gpu_memory_used: Optional[float] = None
    model_performance_metrics: Dict[str, float] = field(default_factory=dict)

    def get_average_confidence(self) -> float:
        """Calculate average confidence across predictions."""
        confidences = []
        for pred in self.predictions:
            if pred.confidence_intervals is not None:
                # Use the inverse of confidence interval width as confidence measure
                avg_width = np.mean(
                    pred.confidence_intervals[:, 1] - pred.confidence_intervals[:, 0]
                )
                confidences.append(1.0 / (1.0 + avg_width))

        return np.mean(confidences) if confidences else 0.0


# =============================================================================
# Base Classes
# =============================================================================


class BaseSpectralPredictor(ABC):
    """Abstract base class for spectral predictors."""

    def __init__(self, config: ModelConfiguration):
        self.config = config
        self.model = None
        self.is_trained = False
        self.feature_scaler = None
        self.spectrum_scaler = None

    @abstractmethod
    def train(self, training_data: TrainingData) -> Dict[str, float]:
        """Train the spectral prediction model."""
        pass

    @abstractmethod
    def predict(self, molecular_features: np.ndarray) -> List[SpectralPrediction]:
        """Predict spectra for given molecular features."""
        pass

    @abstractmethod
    def save_model(self, filepath: str) -> None:
        """Save trained model to file."""
        pass

    @abstractmethod
    def load_model(self, filepath: str) -> None:
        """Load trained model from file."""
        pass


# =============================================================================
# Neural Network Architectures
# =============================================================================

if TORCH_AVAILABLE:

    class SpectralTransformer(nn.Module):
        """Transformer architecture for spectral prediction."""

        def __init__(self, input_dim: int, output_dim: int, config: ModelConfiguration):
            super().__init__()
            self.config = config

            # Input projection
            self.input_projection = nn.Linear(input_dim, config.hidden_dims[0])

            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_dims[0],
                nhead=8,
                dim_feedforward=config.hidden_dims[0] * 2,
                dropout=config.dropout_rate,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

            # Output layers
            self.output_layers = nn.ModuleList()
            prev_dim = config.hidden_dims[0]
            for dim in config.hidden_dims[1:]:
                self.output_layers.append(nn.Linear(prev_dim, dim))
                self.output_layers.append(nn.ReLU())
                self.output_layers.append(nn.Dropout(config.dropout_rate))
                prev_dim = dim

            self.final_layer = nn.Linear(prev_dim, output_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Input projection
            x = self.input_projection(x)

            # Add sequence dimension for transformer
            if x.dim() == 2:
                x = x.unsqueeze(1)

            # Transformer encoding
            x = self.transformer(x)

            # Remove sequence dimension
            x = x.squeeze(1)

            # Output layers
            for layer in self.output_layers:
                x = layer(x)

            x = self.final_layer(x)
            return x

    class ConvolutionalSpectralNet(nn.Module):
        """1D convolutional network for spectral prediction."""

        def __init__(self, input_dim: int, output_dim: int, config: ModelConfiguration):
            super().__init__()
            self.config = config

            # Initial feature extraction
            self.feature_layers = nn.Sequential(
                nn.Linear(input_dim, config.hidden_dims[0]),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate),
            )

            # Convolutional layers
            self.conv_layers = nn.ModuleList()
            in_channels = 1
            for i, out_channels in enumerate([32, 64, 128]):
                self.conv_layers.append(
                    nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
                )
                self.conv_layers.append(nn.ReLU())
                self.conv_layers.append(nn.BatchNorm1d(out_channels))
                if i < 2:  # Don't pool after last conv layer
                    self.conv_layers.append(nn.MaxPool1d(2))
                in_channels = out_channels

            # Calculate flattened size after convolutions
            conv_output_size = (
                config.hidden_dims[0] // 4 * 128
            )  # Approximate after pooling

            # Final layers
            self.final_layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(conv_output_size, config.hidden_dims[1]),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate),
                nn.Linear(config.hidden_dims[1], output_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Feature extraction
            x = self.feature_layers(x)

            # Reshape for convolution (batch, channels, sequence)
            x = x.unsqueeze(1)

            # Convolutional layers
            for layer in self.conv_layers:
                x = layer(x)

            # Final prediction
            x = self.final_layers(x)
            return x

    class GraphSpectralNet(nn.Module):
        """Graph neural network for molecular spectral prediction."""

        def __init__(
            self,
            node_dim: int,
            edge_dim: int,
            output_dim: int,
            config: ModelConfiguration,
        ):
            super().__init__()
            self.config = config

            # Graph convolution layers
            self.node_embedding = nn.Linear(node_dim, config.hidden_dims[0])
            self.edge_embedding = nn.Linear(edge_dim, config.hidden_dims[0])

            # Message passing layers
            self.message_layers = nn.ModuleList()
            for _ in range(3):
                self.message_layers.append(
                    nn.Sequential(
                        nn.Linear(config.hidden_dims[0] * 2, config.hidden_dims[0]),
                        nn.ReLU(),
                        nn.Dropout(config.dropout_rate),
                    )
                )

            # Global pooling and prediction
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            self.predictor = nn.Sequential(
                nn.Linear(config.hidden_dims[0], config.hidden_dims[1]),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate),
                nn.Linear(config.hidden_dims[1], output_dim),
            )

        def forward(
            self,
            node_features: torch.Tensor,
            edge_features: torch.Tensor,
            edge_indices: torch.Tensor,
        ) -> torch.Tensor:
            # Embed features
            node_emb = self.node_embedding(node_features)
            edge_emb = self.edge_embedding(edge_features)

            # Message passing
            for layer in self.message_layers:
                # Gather node features for edges
                src_features = node_emb[edge_indices[0]]
                dst_features = node_emb[edge_indices[1]]

                # Compute messages
                messages = layer(torch.cat([src_features, dst_features], dim=-1))

                # Aggregate messages (simple mean aggregation)
                node_emb = torch.zeros_like(node_emb)
                node_emb.index_add_(0, edge_indices[1], messages)

                # Normalize by degree
                degree = torch.bincount(edge_indices[1], minlength=node_emb.size(0))
                node_emb = node_emb / (degree.unsqueeze(-1) + 1e-8)

            # Global pooling
            graph_emb = torch.mean(node_emb, dim=0, keepdim=True)

            # Predict spectrum
            spectrum = self.predictor(graph_emb)
            return spectrum


# =============================================================================
# Main Prediction Classes
# =============================================================================


class GPUSpectroscopyPredictor(BaseSpectralPredictor):
    """Main GPU-accelerated spectroscopy prediction engine."""

    def __init__(self, config: ModelConfiguration):
        super().__init__(config)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and config.use_gpu else "cpu"
        )
        self.scaler = None

        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for GPUSpectroscopyPredictor")

    def _create_model(self, input_dim: int, output_dim: int) -> nn.Module:
        """Create neural network model based on configuration."""
        if self.config.architecture == ModelArchitecture.TRANSFORMER:
            return SpectralTransformer(input_dim, output_dim, self.config)
        elif self.config.architecture == ModelArchitecture.CONVOLUTIONAL:
            return ConvolutionalSpectralNet(input_dim, output_dim, self.config)
        elif self.config.architecture == ModelArchitecture.GRAPH_NEURAL:
            # For graph networks, assume input includes node/edge features
            return GraphSpectralNet(
                input_dim // 2, input_dim // 2, output_dim, self.config
            )
        else:
            # Default to fully connected
            layers = []
            prev_dim = input_dim

            for dim in self.config.hidden_dims:
                layers.extend(
                    [
                        nn.Linear(prev_dim, dim),
                        nn.ReLU(),
                        nn.Dropout(self.config.dropout_rate),
                    ]
                )
                prev_dim = dim

            layers.append(nn.Linear(prev_dim, output_dim))
            return nn.Sequential(*layers)

    def train(self, training_data: TrainingData) -> Dict[str, float]:
        """Train the spectral prediction model."""
        # Prepare data
        X = torch.tensor(training_data.molecular_features, dtype=torch.float32)
        y = torch.tensor(training_data.spectra, dtype=torch.float32)

        input_dim = X.shape[1]
        output_dim = y.shape[1]

        # Create model
        self.model = self._create_model(input_dim, output_dim).to(self.device)

        # Setup training
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )
        criterion = nn.MSELoss()
        scaler = GradScaler() if self.device.type == "cuda" else None

        # Training loop
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True
        )

        losses = []
        self.model.train()

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0

            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()

                if scaler is not None:
                    with autocast():
                        predictions = self.model(batch_x)
                        loss = criterion(predictions, batch_y)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    predictions = self.model(batch_x)
                    loss = criterion(predictions, batch_y)
                    loss.backward()
                    optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")

        self.is_trained = True

        return {
            "final_loss": losses[-1],
            "min_loss": min(losses),
            "epochs_trained": len(losses),
        }

    def predict(self, molecular_features: np.ndarray) -> List[SpectralPrediction]:
        """Predict spectra for given molecular features."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        self.model.eval()
        predictions = []

        # Generate frequency grid
        freq_start, freq_end = self.config.frequency_range
        frequencies = np.arange(freq_start, freq_end, self.config.frequency_resolution)

        with torch.no_grad():
            X = torch.tensor(molecular_features, dtype=torch.float32).to(self.device)

            # Batch prediction
            if self.config.use_gpu and torch.cuda.is_available():
                with autocast():
                    pred_spectra = self.model(X)
            else:
                pred_spectra = self.model(X)

            pred_spectra = pred_spectra.cpu().numpy()

            # Convert to SpectralPrediction objects
            for i, spectrum in enumerate(pred_spectra):
                prediction = SpectralPrediction(
                    frequencies=frequencies,
                    intensities=spectrum,
                    spectroscopy_type=self.config.spectroscopy_type,
                    model_version="1.0",
                )
                predictions.append(prediction)

        return predictions

    def save_model(self, filepath: str) -> None:
        """Save trained model to file."""
        if self.model is None:
            raise RuntimeError("No model to save")

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.config,
                "is_trained": self.is_trained,
            },
            filepath,
        )

    def load_model(self, filepath: str) -> None:
        """Load trained model from file."""
        checkpoint = torch.load(filepath, map_location=self.device)

        # Recreate model with saved config
        self.config = checkpoint["config"]
        input_dim = 256  # Default, should be stored in checkpoint
        output_dim = len(
            np.arange(
                self.config.frequency_range[0],
                self.config.frequency_range[1],
                self.config.frequency_resolution,
            )
        )

        self.model = self._create_model(input_dim, output_dim).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.is_trained = checkpoint["is_trained"]


class QMLSurrogateModel(BaseSpectralPredictor):
    """Quantum Machine Learning surrogate for fast spectral prediction."""

    def __init__(self, config: ModelConfiguration):
        super().__init__(config)
        self.quantum_features = {}
        self.classical_model = None

        if not JAX_AVAILABLE:
            warnings.warn("JAX not available. QML features will be limited.")

    def _extract_quantum_features(self, molecular_features: np.ndarray) -> np.ndarray:
        """Extract quantum-inspired features for molecules."""
        # Placeholder for quantum feature extraction
        # In practice, this would use quantum chemistry calculations

        batch_size, feature_dim = molecular_features.shape

        # Simulate quantum features (HOMO-LUMO gap, dipole moment, etc.)
        quantum_feats = np.random.random((batch_size, 50))  # Placeholder

        # Combine with classical features
        combined_features = np.concatenate([molecular_features, quantum_feats], axis=1)

        return combined_features

    def train(self, training_data: TrainingData) -> Dict[str, float]:
        """Train the QML surrogate model."""
        # Extract quantum features
        enhanced_features = self._extract_quantum_features(
            training_data.molecular_features
        )

        # Create enhanced training data
        enhanced_training = TrainingData(
            molecular_features=enhanced_features,
            spectra=training_data.spectra,
            frequencies=training_data.frequencies,
            molecule_ids=training_data.molecule_ids,
            weights=training_data.weights,
            metadata=training_data.metadata,
        )

        # Use classical model for now (would be replaced with quantum circuits)
        classical_config = ModelConfiguration(
            architecture=ModelArchitecture.TRANSFORMER,
            spectroscopy_type=self.config.spectroscopy_type,
            use_gpu=self.config.use_gpu,
        )

        self.classical_model = GPUSpectroscopyPredictor(classical_config)
        return self.classical_model.train(enhanced_training)

    def predict(self, molecular_features: np.ndarray) -> List[SpectralPrediction]:
        """Predict spectra using QML surrogate."""
        if self.classical_model is None:
            raise RuntimeError("Model must be trained before prediction")

        # Extract quantum features
        enhanced_features = self._extract_quantum_features(molecular_features)

        # Use classical model for prediction
        return self.classical_model.predict(enhanced_features)

    def save_model(self, filepath: str) -> None:
        """Save QML surrogate model."""
        if self.classical_model is not None:
            self.classical_model.save_model(filepath)

    def load_model(self, filepath: str) -> None:
        """Load QML surrogate model."""
        classical_config = ModelConfiguration(
            architecture=ModelArchitecture.TRANSFORMER,
            spectroscopy_type=self.config.spectroscopy_type,
            use_gpu=self.config.use_gpu,
        )
        self.classical_model = GPUSpectroscopyPredictor(classical_config)
        self.classical_model.load_model(filepath)


# =============================================================================
# Specialized Predictors
# =============================================================================


class NMRPredictor(GPUSpectroscopyPredictor):
    """Specialized predictor for NMR spectroscopy."""

    def __init__(self, nmr_type: str = "1H", use_gpu: bool = True):
        spectroscopy_type = (
            SpectroscopyType.NMR_1H if nmr_type == "1H" else SpectroscopyType.NMR_13C
        )

        config = ModelConfiguration(
            architecture=ModelArchitecture.TRANSFORMER,
            spectroscopy_type=spectroscopy_type,
            frequency_range=(0.0, 12.0) if nmr_type == "1H" else (0.0, 220.0),
            frequency_resolution=0.01,
            use_gpu=use_gpu,
        )
        super().__init__(config)


class IRPredictor(GPUSpectroscopyPredictor):
    """Specialized predictor for IR spectroscopy."""

    def __init__(self, use_gpu: bool = True):
        config = ModelConfiguration(
            architecture=ModelArchitecture.CONVOLUTIONAL,
            spectroscopy_type=SpectroscopyType.IR,
            frequency_range=(400.0, 4000.0),
            frequency_resolution=1.0,
            use_gpu=use_gpu,
        )
        super().__init__(config)


class MSPredictor(GPUSpectroscopyPredictor):
    """Specialized predictor for mass spectrometry."""

    def __init__(self, use_gpu: bool = True):
        config = ModelConfiguration(
            architecture=ModelArchitecture.GRAPH_NEURAL,
            spectroscopy_type=SpectroscopyType.MS,
            frequency_range=(0.0, 1000.0),  # m/z range
            frequency_resolution=0.1,
            use_gpu=use_gpu,
        )
        super().__init__(config)


class UVVisPredictor(GPUSpectroscopyPredictor):
    """Specialized predictor for UV-Vis spectroscopy."""

    def __init__(self, use_gpu: bool = True):
        config = ModelConfiguration(
            architecture=ModelArchitecture.TRANSFORMER,
            spectroscopy_type=SpectroscopyType.UV_VIS,
            frequency_range=(200.0, 800.0),  # wavelength in nm
            frequency_resolution=0.5,
            use_gpu=use_gpu,
        )
        super().__init__(config)


# =============================================================================
# Utility Functions
# =============================================================================


def predict_spectrum_batch(
    predictor: BaseSpectralPredictor,
    molecular_features: np.ndarray,
    batch_size: int = 32,
) -> PredictionResult:
    """Perform batch spectral prediction with performance monitoring."""
    import time

    start_time = time.time()

    # Split into batches
    n_samples = len(molecular_features)
    predictions = []

    for i in range(0, n_samples, batch_size):
        batch_features = molecular_features[i : i + batch_size]
        batch_predictions = predictor.predict(batch_features)
        predictions.extend(batch_predictions)

    computation_time = time.time() - start_time

    # Monitor GPU memory if available
    gpu_memory = None
    if TORCH_AVAILABLE and torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**2  # MB

    return PredictionResult(
        predictions=predictions,
        batch_computation_time=computation_time,
        gpu_memory_used=gpu_memory,
        model_performance_metrics={
            "samples_per_second": n_samples / computation_time,
            "average_prediction_time": computation_time / n_samples,
        },
    )


def train_spectral_model(
    model_type: str,
    training_data: TrainingData,
    config: Optional[ModelConfiguration] = None,
) -> BaseSpectralPredictor:
    """Factory function to train spectral models."""
    if config is None:
        config = ModelConfiguration(
            architecture=ModelArchitecture.TRANSFORMER,
            spectroscopy_type=SpectroscopyType.IR,
        )

    if model_type.lower() == "nmr":
        predictor = NMRPredictor(use_gpu=config.use_gpu)
    elif model_type.lower() == "ir":
        predictor = IRPredictor(use_gpu=config.use_gpu)
    elif model_type.lower() == "ms":
        predictor = MSPredictor(use_gpu=config.use_gpu)
    elif model_type.lower() == "uv-vis":
        predictor = UVVisPredictor(use_gpu=config.use_gpu)
    elif model_type.lower() == "qml":
        predictor = QMLSurrogateModel(config)
    else:
        predictor = GPUSpectroscopyPredictor(config)

    # Train the model
    training_metrics = predictor.train(training_data)
    print(f"Training completed: {training_metrics}")

    return predictor


def evaluate_model_accuracy(
    predictor: BaseSpectralPredictor, test_data: TrainingData
) -> Dict[str, float]:
    """Evaluate model accuracy on test data."""
    predictions = predictor.predict(test_data.molecular_features)

    # Calculate metrics
    mse_values = []
    mae_values = []

    for i, prediction in enumerate(predictions):
        true_spectrum = test_data.spectra[i]
        pred_spectrum = prediction.intensities

        # Ensure same length
        min_len = min(len(true_spectrum), len(pred_spectrum))
        true_spectrum = true_spectrum[:min_len]
        pred_spectrum = pred_spectrum[:min_len]

        mse = np.mean((true_spectrum - pred_spectrum) ** 2)
        mae = np.mean(np.abs(true_spectrum - pred_spectrum))

        mse_values.append(mse)
        mae_values.append(mae)

    return {
        "mean_squared_error": np.mean(mse_values),
        "mean_absolute_error": np.mean(mae_values),
        "r_squared": 1 - (np.var(mse_values) / np.var(test_data.spectra.flatten())),
        "samples_evaluated": len(predictions),
    }


# =============================================================================
# Exception Classes
# =============================================================================


class PredictionError(Exception):
    """Exception raised during spectral prediction."""

    pass


class ModelError(Exception):
    """Exception raised during model operations."""

    pass


class TrainingError(Exception):
    """Exception raised during model training."""

    pass
