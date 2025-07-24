"""
Uncertainty Quantification module for ChemiAccel.

Provides multiple methods for estimating prediction uncertainty:
- Deep Ensembles: Train multiple models and use prediction variance
- Monte Carlo Dropout: Use dropout at inference time for uncertainty sampling
- Conformal Prediction: Mathematically rigorous prediction intervals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Some UQ methods will be disabled.")

try:
    import jax
    import jax.numpy as jnp
    from jax import random, vmap, jit
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    warnings.warn("JAX not available. JAX-based UQ methods will be disabled.")

from ..core.exceptions import ChemiAccelError


class UncertaintyQuantificationError(ChemiAccelError):
    """Raised when uncertainty quantification operations fail."""
    pass


@dataclass
class UncertaintyResult:
    """Results from uncertainty quantification."""
    predictions: np.ndarray
    uncertainties: np.ndarray
    confidence_intervals: Optional[np.ndarray] = None
    prediction_intervals: Optional[np.ndarray] = None
    method: str = ""
    calibration_score: Optional[float] = None


class BaseUQMethod(ABC):
    """Base class for uncertainty quantification methods."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the uncertainty quantification method."""
        pass
    
    @abstractmethod
    def predict_with_uncertainty(self, X: np.ndarray) -> UncertaintyResult:
        """Make predictions with uncertainty estimates."""
        pass
    
    @abstractmethod
    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> None:
        """Calibrate uncertainty estimates on validation data."""
        pass


class DeepEnsemble(BaseUQMethod):
    """
    Deep Ensemble uncertainty quantification.
    
    Trains multiple neural networks and uses the variance of their predictions
    as a measure of epistemic uncertainty.
    """
    
    def __init__(self, 
                 n_models: int = 5,
                 model_factory: Optional[Callable] = None,
                 device: str = 'auto'):
        """
        Initialize Deep Ensemble.
        
        Args:
            n_models: Number of models in ensemble
            model_factory: Function that returns a new model instance
            device: Device to use ('cpu', 'cuda', 'auto')
        """
        if not TORCH_AVAILABLE:
            raise UncertaintyQuantificationError("PyTorch required for Deep Ensemble")
            
        self.n_models = n_models
        self.model_factory = model_factory or self._default_model_factory
        self.device = self._get_device(device)
        self.models = []
        self.is_fitted = False
    
    def _get_device(self, device: str) -> str:
        """Get appropriate device."""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def _default_model_factory(self) -> nn.Module:
        """Default neural network architecture."""
        return nn.Sequential(
            nn.Linear(100, 256),  # Will be adjusted based on input size
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            epochs: int = 100, lr: float = 0.001, batch_size: int = 32) -> None:
        """
        Fit ensemble of models.
        
        Args:
            X: Input features
            y: Target values
            epochs: Training epochs per model
            lr: Learning rate
            batch_size: Batch size for training
        """
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y.reshape(-1, 1)).to(self.device)
        
        # Adjust first layer size based on input
        input_size = X.shape[1]
        
        self.models = []
        for i in range(self.n_models):
            # Create model with correct input size
            model = self._create_model_with_input_size(input_size)
            model = model.to(self.device)
            
            # Train model with different random initialization
            torch.manual_seed(i * 42)  # Different seed for each model
            self._train_single_model(model, X_tensor, y_tensor, epochs, lr, batch_size)
            
            self.models.append(model)
        
        self.is_fitted = True
    
    def _create_model_with_input_size(self, input_size: int) -> nn.Module:
        """Create model with correct input size."""
        return nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def _train_single_model(self, model: nn.Module, X: torch.Tensor, y: torch.Tensor,
                           epochs: int, lr: float, batch_size: int) -> None:
        """Train a single model in the ensemble."""
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
    
    def predict_with_uncertainty(self, X: np.ndarray) -> UncertaintyResult:
        """Make predictions with uncertainty from ensemble variance."""
        if not self.is_fitted:
            raise UncertaintyQuantificationError("Model must be fitted before prediction")
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(X_tensor).cpu().numpy().flatten()
                predictions.append(pred)
        
        predictions = np.array(predictions)  # Shape: (n_models, n_samples)
        
        # Calculate ensemble statistics
        mean_pred = np.mean(predictions, axis=0)
        uncertainty = np.std(predictions, axis=0)
        
        # Calculate confidence intervals (assuming normal distribution)
        confidence_intervals = np.stack([
            mean_pred - 1.96 * uncertainty,  # Lower bound (95% CI)
            mean_pred + 1.96 * uncertainty   # Upper bound (95% CI)
        ], axis=1)
        
        return UncertaintyResult(
            predictions=mean_pred,
            uncertainties=uncertainty,
            confidence_intervals=confidence_intervals,
            method="deep_ensemble"
        )
    
    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> None:
        """Calibrate uncertainty estimates using validation data."""
        result = self.predict_with_uncertainty(X_cal)
        
        # Calculate calibration score (coverage probability)
        lower_bounds = result.confidence_intervals[:, 0]
        upper_bounds = result.confidence_intervals[:, 1]
        coverage = np.mean((y_cal >= lower_bounds) & (y_cal <= upper_bounds))
        
        print(f"Calibration coverage: {coverage:.3f} (target: 0.95)")


class MonteCarloDropout(BaseUQMethod):
    """
    Monte Carlo Dropout uncertainty quantification.
    
    Uses dropout at inference time to sample from the model's posterior distribution.
    """
    
    def __init__(self, 
                 model: Optional[nn.Module] = None,
                 n_samples: int = 100,
                 dropout_rate: float = 0.1,
                 device: str = 'auto'):
        """
        Initialize MC Dropout.
        
        Args:
            model: Neural network model with dropout layers
            n_samples: Number of forward passes for uncertainty estimation
            dropout_rate: Dropout rate for uncertainty sampling
            device: Device to use
        """
        if not TORCH_AVAILABLE:
            raise UncertaintyQuantificationError("PyTorch required for MC Dropout")
            
        self.model = model
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate
        self.device = self._get_device(device)
        self.is_fitted = False
    
    def _get_device(self, device: str) -> str:
        """Get appropriate device."""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def _create_default_model(self, input_size: int) -> nn.Module:
        """Create default model with dropout layers."""
        return nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(64, 1)
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            epochs: int = 100, lr: float = 0.001, batch_size: int = 32) -> None:
        """Fit model with dropout layers."""
        if self.model is None:
            self.model = self._create_default_model(X.shape[1])
        
        self.model = self.model.to(self.device)
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y.reshape(-1, 1)).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        self.is_fitted = True
    
    def predict_with_uncertainty(self, X: np.ndarray) -> UncertaintyResult:
        """Make predictions with MC Dropout uncertainty."""
        if not self.is_fitted:
            raise UncertaintyQuantificationError("Model must be fitted before prediction")
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Enable dropout during inference
        self.model.train()
        
        predictions = []
        for _ in range(self.n_samples):
            with torch.no_grad():
                pred = self.model(X_tensor).cpu().numpy().flatten()
                predictions.append(pred)
        
        predictions = np.array(predictions)  # Shape: (n_samples, n_data)
        
        # Calculate statistics
        mean_pred = np.mean(predictions, axis=0)
        uncertainty = np.std(predictions, axis=0)
        
        # Calculate confidence intervals
        confidence_intervals = np.stack([
            mean_pred - 1.96 * uncertainty,
            mean_pred + 1.96 * uncertainty
        ], axis=1)
        
        return UncertaintyResult(
            predictions=mean_pred,
            uncertainties=uncertainty,
            confidence_intervals=confidence_intervals,
            method="mc_dropout"
        )
    
    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> None:
        """Calibrate MC Dropout uncertainty estimates."""
        result = self.predict_with_uncertainty(X_cal)
        
        lower_bounds = result.confidence_intervals[:, 0]
        upper_bounds = result.confidence_intervals[:, 1]
        coverage = np.mean((y_cal >= lower_bounds) & (y_cal <= upper_bounds))
        
        print(f"MC Dropout calibration coverage: {coverage:.3f} (target: 0.95)")


class ConformalPrediction(BaseUQMethod):
    """
    Conformal Prediction for uncertainty quantification.
    
    Provides mathematically rigorous prediction intervals with guaranteed coverage.
    """
    
    def __init__(self, 
                 base_model: Any,
                 confidence_level: float = 0.95):
        """
        Initialize Conformal Prediction.
        
        Args:
            base_model: Base prediction model (sklearn-compatible)
            confidence_level: Desired confidence level (e.g., 0.95 for 95%)
        """
        self.base_model = base_model
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.conformity_scores = None
        self.quantile = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit base model for conformal prediction."""
        self.base_model.fit(X, y)
        self.is_fitted = True
    
    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> None:
        """Calibrate conformal prediction using calibration set."""
        if not self.is_fitted:
            raise UncertaintyQuantificationError("Model must be fitted before calibration")
        
        # Get predictions on calibration set
        y_pred_cal = self.base_model.predict(X_cal)
        
        # Calculate conformity scores (absolute residuals)
        self.conformity_scores = np.abs(y_cal - y_pred_cal)
        
        # Calculate quantile for prediction intervals
        n = len(self.conformity_scores)
        self.quantile = np.quantile(
            self.conformity_scores, 
            np.ceil((n + 1) * (1 - self.alpha)) / n
        )
    
    def predict_with_uncertainty(self, X: np.ndarray) -> UncertaintyResult:
        """Make predictions with conformal prediction intervals."""
        if not self.is_fitted:
            raise UncertaintyQuantificationError("Model must be fitted before prediction")
        
        if self.quantile is None:
            raise UncertaintyQuantificationError("Model must be calibrated before prediction")
        
        # Get point predictions
        predictions = self.base_model.predict(X)
        
        # Create prediction intervals
        prediction_intervals = np.stack([
            predictions - self.quantile,  # Lower bound
            predictions + self.quantile   # Upper bound
        ], axis=1)
        
        # Uncertainty is half the interval width
        uncertainties = np.full(len(predictions), self.quantile)
        
        return UncertaintyResult(
            predictions=predictions,
            uncertainties=uncertainties,
            prediction_intervals=prediction_intervals,
            confidence_intervals=prediction_intervals,
            method="conformal_prediction"
        )


class UncertaintyQuantification:
    """
    Main uncertainty quantification interface for ChemiAccel.
    
    Provides unified access to multiple uncertainty quantification methods.
    """
    
    def __init__(self, method: str = 'deep_ensemble', **kwargs):
        """
        Initialize uncertainty quantification.
        
        Args:
            method: UQ method ('deep_ensemble', 'mc_dropout', 'conformal')
            **kwargs: Method-specific parameters
        """
        self.method = method
        self.kwargs = kwargs
        self.uq_model = None
        self._initialize_method()
    
    def _initialize_method(self) -> None:
        """Initialize the specified UQ method."""
        if self.method == 'deep_ensemble':
            self.uq_model = DeepEnsemble(**self.kwargs)
        elif self.method == 'mc_dropout':
            self.uq_model = MonteCarloDropout(**self.kwargs)
        elif self.method == 'conformal':
            self.uq_model = ConformalPrediction(**self.kwargs)
        else:
            raise UncertaintyQuantificationError(f"Unknown UQ method: {self.method}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, **fit_kwargs) -> None:
        """Fit the uncertainty quantification method."""
        self.uq_model.fit(X, y, **fit_kwargs)
    
    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> None:
        """Calibrate uncertainty estimates."""
        self.uq_model.calibrate(X_cal, y_cal)
    
    def predict_with_uncertainty(self, X: np.ndarray) -> UncertaintyResult:
        """Make predictions with uncertainty estimates."""
        return self.uq_model.predict_with_uncertainty(X)
    
    def evaluate_calibration(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate calibration quality on test set."""
        result = self.predict_with_uncertainty(X_test)
        
        # Calculate coverage probability
        if result.confidence_intervals is not None:
            lower_bounds = result.confidence_intervals[:, 0]
            upper_bounds = result.confidence_intervals[:, 1]
            coverage = np.mean((y_test >= lower_bounds) & (y_test <= upper_bounds))
        else:
            coverage = None
        
        # Calculate mean absolute error
        mae = np.mean(np.abs(y_test - result.predictions))
        
        # Calculate root mean squared error
        rmse = np.sqrt(np.mean((y_test - result.predictions) ** 2))
        
        # Calculate uncertainty quality metrics
        if result.uncertainties is not None:
            # Correlation between uncertainty and absolute error
            abs_errors = np.abs(y_test - result.predictions)
            uncertainty_correlation = np.corrcoef(result.uncertainties, abs_errors)[0, 1]
        else:
            uncertainty_correlation = None
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'coverage_probability': coverage,
            'uncertainty_correlation': uncertainty_correlation,
            'method': self.method
        }
        
        return {k: v for k, v in metrics.items() if v is not None}


# GPU-accelerated uncertainty estimation utilities
class GPUAcceleratedUQ:
    """GPU-accelerated uncertainty estimation algorithms."""
    
    @staticmethod
    def ensemble_prediction_gpu(models: List[Any], X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        GPU-accelerated ensemble prediction.
        
        Args:
            models: List of trained models
            X: Input features
            
        Returns:
            Tuple of (predictions, uncertainties)
        """
        if not TORCH_AVAILABLE:
            raise UncertaintyQuantificationError("PyTorch required for GPU acceleration")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        X_tensor = torch.FloatTensor(X).to(device)
        
        predictions = []
        for model in models:
            model = model.to(device)
            model.eval()
            with torch.no_grad():
                pred = model(X_tensor).cpu().numpy().flatten()
                predictions.append(pred)
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        uncertainty = np.std(predictions, axis=0)
        
        return mean_pred, uncertainty
    
    @staticmethod
    def batch_mc_dropout_gpu(model: nn.Module, X: np.ndarray, 
                           n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        GPU-accelerated batch Monte Carlo Dropout.
        
        Args:
            model: Model with dropout layers
            X: Input features
            n_samples: Number of MC samples
            
        Returns:
            Tuple of (predictions, uncertainties)
        """
        if not TORCH_AVAILABLE:
            raise UncertaintyQuantificationError("PyTorch required for GPU acceleration")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        X_tensor = torch.FloatTensor(X).to(device)
        
        model.train()  # Enable dropout
        
        # Batch processing for efficiency
        batch_size = min(1000, len(X))
        all_predictions = []
        
        for i in range(0, len(X), batch_size):
            batch_X = X_tensor[i:i+batch_size]
            batch_predictions = []
            
            for _ in range(n_samples):
                with torch.no_grad():
                    pred = model(batch_X).cpu().numpy().flatten()
                    batch_predictions.append(pred)
            
            batch_predictions = np.array(batch_predictions)
            all_predictions.append(batch_predictions)
        
        # Combine all batches
        predictions = np.concatenate(all_predictions, axis=1)
        mean_pred = np.mean(predictions, axis=0)
        uncertainty = np.std(predictions, axis=0)
        
        return mean_pred, uncertainty


# JAX-based uncertainty quantification for TPU acceleration
if JAX_AVAILABLE:
    class JAXUncertaintyQuantification:
        """JAX-based uncertainty quantification for TPU acceleration."""
        
        def __init__(self, model_fn: Callable, n_ensemble: int = 5):
            """
            Initialize JAX UQ.
            
            Args:
                model_fn: JAX model function
                n_ensemble: Number of models in ensemble
            """
            self.model_fn = model_fn
            self.n_ensemble = n_ensemble
            self.params_ensemble = []
        
        def fit_ensemble(self, X: jnp.ndarray, y: jnp.ndarray, 
                        key: jax.random.PRNGKey) -> None:
            """Fit ensemble of JAX models."""
            keys = jax.random.split(key, self.n_ensemble)
            
            for i, subkey in enumerate(keys):
                # Initialize and train model with different random seed
                params = self._train_single_model(X, y, subkey)
                self.params_ensemble.append(params)
        
        def _train_single_model(self, X: jnp.ndarray, y: jnp.ndarray, 
                               key: jax.random.PRNGKey) -> Dict:
            """Train a single model in the ensemble."""
            # Placeholder for JAX model training
            # In practice, this would use optax for optimization
            pass
        
        @jit
        def predict_ensemble(self, X: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """Make ensemble predictions with JAX JIT compilation."""
            predictions = []
            
            for params in self.params_ensemble:
                pred = self.model_fn(params, X)
                predictions.append(pred)
            
            predictions = jnp.stack(predictions)
            mean_pred = jnp.mean(predictions, axis=0)
            uncertainty = jnp.std(predictions, axis=0)
            
            return mean_pred, uncertainty