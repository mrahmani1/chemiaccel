"""
GPU-Accelerated Bayesian Optimization Framework for ChemiAccel

This module provides a comprehensive Bayesian optimization framework leveraging ChemiAccel's
GPU capabilities for rapid surrogate model training and acquisition function evaluation
in high-dimensional experimental spaces.

Key Features:
- GPU-accelerated Gaussian Process surrogate models
- Parallel acquisition function evaluation
- Multi-objective optimization support
- High-dimensional parameter space handling
- Adaptive experimental design
- Real-time optimization monitoring
"""

import asyncio
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.optimize
import scipy.stats

try:
    import gpytorch
    import torch
    import torch.nn as nn
    import torch.optim as optim

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn(
        "PyTorch/GPyTorch not available. GPU Bayesian optimization will be disabled."
    )

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    warnings.warn("CuPy not available. GPU acceleration will be limited.")

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, random, vmap

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    warnings.warn("JAX not available. JAX-based optimization will be disabled.")

from ..core.exceptions import ChemiAccelError
from .multi_objective_optimization import (
    MPOptimizer,
    MultiObjectiveResult,
    ObjectiveConfig,
)


class BayesianOptimizationError(ChemiAccelError):
    """Raised when Bayesian optimization operations fail."""

    pass


class AcquisitionFunction(Enum):
    """Types of acquisition functions."""

    EXPECTED_IMPROVEMENT = "expected_improvement"
    PROBABILITY_OF_IMPROVEMENT = "probability_of_improvement"
    UPPER_CONFIDENCE_BOUND = "upper_confidence_bound"
    THOMPSON_SAMPLING = "thompson_sampling"
    ENTROPY_SEARCH = "entropy_search"
    KNOWLEDGE_GRADIENT = "knowledge_gradient"
    PARALLEL_EI = "parallel_expected_improvement"
    MULTI_OBJECTIVE_EI = "multi_objective_expected_improvement"


class SurrogateModelType(Enum):
    """Types of surrogate models."""

    GAUSSIAN_PROCESS = "gaussian_process"
    SPARSE_GP = "sparse_gaussian_process"
    DEEP_GP = "deep_gaussian_process"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"


class OptimizationStrategy(Enum):
    """Optimization strategies for acquisition function."""

    RANDOM_SEARCH = "random_search"
    GRADIENT_BASED = "gradient_based"
    EVOLUTIONARY = "evolutionary"
    MULTI_START = "multi_start"
    BAYESIAN_SEARCH = "bayesian_search"


@dataclass
class ParameterSpace:
    """Definition of parameter search space."""

    name: str
    bounds: Tuple[float, float]
    parameter_type: str = "continuous"  # 'continuous', 'discrete', 'categorical'
    log_scale: bool = False
    discrete_values: Optional[List[Any]] = None
    categorical_values: Optional[List[str]] = None

    def sample(self, n_samples: int, random_state: Optional[int] = None) -> np.ndarray:
        """Sample from parameter space."""
        if random_state is not None:
            np.random.seed(random_state)

        if self.parameter_type == "continuous":
            if self.log_scale:
                log_bounds = (np.log10(self.bounds[0]), np.log10(self.bounds[1]))
                samples = np.random.uniform(log_bounds[0], log_bounds[1], n_samples)
                return 10**samples
            else:
                return np.random.uniform(self.bounds[0], self.bounds[1], n_samples)

        elif self.parameter_type == "discrete":
            if self.discrete_values is not None:
                return np.random.choice(self.discrete_values, n_samples)
            else:
                return np.random.randint(
                    int(self.bounds[0]), int(self.bounds[1]) + 1, n_samples
                )

        elif self.parameter_type == "categorical":
            if self.categorical_values is not None:
                return np.random.choice(self.categorical_values, n_samples)
            else:
                raise BayesianOptimizationError(
                    "Categorical values must be specified for categorical parameters"
                )

        else:
            raise BayesianOptimizationError(
                f"Unknown parameter type: {self.parameter_type}"
            )

    def normalize(self, values: np.ndarray) -> np.ndarray:
        """Normalize values to [0, 1] range."""
        if self.parameter_type == "continuous":
            if self.log_scale:
                log_bounds = (np.log10(self.bounds[0]), np.log10(self.bounds[1]))
                log_values = np.log10(np.clip(values, self.bounds[0], self.bounds[1]))
                return (log_values - log_bounds[0]) / (log_bounds[1] - log_bounds[0])
            else:
                return (values - self.bounds[0]) / (self.bounds[1] - self.bounds[0])
        else:
            # For discrete/categorical, return as-is (handled separately)
            return values

    def denormalize(self, normalized_values: np.ndarray) -> np.ndarray:
        """Denormalize values from [0, 1] range."""
        if self.parameter_type == "continuous":
            if self.log_scale:
                log_bounds = (np.log10(self.bounds[0]), np.log10(self.bounds[1]))
                log_values = (
                    normalized_values * (log_bounds[1] - log_bounds[0]) + log_bounds[0]
                )
                return 10**log_values
            else:
                return (
                    normalized_values * (self.bounds[1] - self.bounds[0])
                    + self.bounds[0]
                )
        else:
            return normalized_values


@dataclass
class BayesianOptimizationConfig:
    """Configuration for Bayesian optimization."""

    # Model configuration
    surrogate_model: SurrogateModelType = SurrogateModelType.GAUSSIAN_PROCESS
    acquisition_function: AcquisitionFunction = AcquisitionFunction.EXPECTED_IMPROVEMENT
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.MULTI_START

    # GPU/compute configuration
    use_gpu: bool = True
    device: str = "auto"  # "auto", "cpu", "cuda", "cuda:0", etc.
    parallel_workers: int = 4

    # Optimization parameters
    n_initial_points: int = 10
    max_iterations: int = 100
    batch_size: int = 1  # Number of points to evaluate in parallel
    convergence_tolerance: float = 1e-6

    # Acquisition function parameters
    exploration_weight: float = 2.0  # For UCB
    xi: float = 0.01  # For EI/PI
    n_thompson_samples: int = 10

    # Model training parameters
    model_training_iterations: int = 1000
    learning_rate: float = 0.1
    noise_variance: float = 1e-4
    signal_variance: float = 1.0
    lengthscale_prior: Tuple[float, float] = (0.1, 10.0)

    # Multi-objective parameters
    scalarization_weights: Optional[np.ndarray] = None
    reference_point: Optional[np.ndarray] = None

    # Experimental design
    latin_hypercube_initial: bool = True
    adaptive_batch_size: bool = False
    max_batch_size: int = 10

    # Performance monitoring
    save_history: bool = True
    log_level: str = "INFO"


@dataclass
class OptimizationResult:
    """Results from Bayesian optimization."""

    best_parameters: Dict[str, float]
    best_objective: float
    best_objectives: Optional[np.ndarray] = None  # For multi-objective

    # Optimization history
    parameter_history: List[Dict[str, float]] = field(default_factory=list)
    objective_history: List[float] = field(default_factory=list)
    acquisition_history: List[float] = field(default_factory=list)

    # Model performance
    model_performance: Dict[str, float] = field(default_factory=dict)
    convergence_metrics: Dict[str, Any] = field(default_factory=dict)

    # Timing information
    total_time: float = 0.0
    model_training_time: float = 0.0
    acquisition_optimization_time: float = 0.0
    objective_evaluation_time: float = 0.0

    # Multi-objective results
    pareto_front: Optional[MultiObjectiveResult] = None

    def get_summary(self) -> Dict[str, Any]:
        """Get optimization summary."""
        return {
            "best_objective": self.best_objective,
            "n_iterations": len(self.objective_history),
            "total_time": self.total_time,
            "average_iteration_time": self.total_time
            / max(len(self.objective_history), 1),
            "convergence_achieved": self.convergence_metrics.get("converged", False),
            "model_performance": self.model_performance,
        }


class SurrogateModel(ABC):
    """Abstract base class for surrogate models."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the surrogate model to data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mean and variance at query points."""
        pass

    @abstractmethod
    def sample(self, X: np.ndarray, n_samples: int = 1) -> np.ndarray:
        """Sample from the posterior distribution."""
        pass

    @abstractmethod
    def get_model_parameters(self) -> Dict[str, Any]:
        """Get model hyperparameters."""
        pass


class GPyTorchGaussianProcess(SurrogateModel):
    """GPU-accelerated Gaussian Process using GPyTorch."""

    def __init__(self, config: BayesianOptimizationConfig):
        if not TORCH_AVAILABLE:
            raise BayesianOptimizationError(
                "PyTorch/GPyTorch required for GP surrogate model"
            )

        self.config = config
        self.device = self._get_device()
        self.model = None
        self.likelihood = None
        self.X_train = None
        self.y_train = None

    def _get_device(self) -> torch.device:
        """Get appropriate device for computation."""
        if self.config.device == "auto":
            if torch.cuda.is_available() and self.config.use_gpu:
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        else:
            return torch.device(self.config.device)

    def _create_model(self, input_dim: int):
        """Create GP model and likelihood."""

        class ExactGPModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)
                )

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.model = ExactGPModel(self.X_train, self.y_train, self.likelihood).to(
            self.device
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the Gaussian Process model."""
        # Convert to tensors
        self.X_train = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.y_train = torch.tensor(
            y.flatten(), dtype=torch.float32, device=self.device
        )

        # Create model
        self._create_model(X.shape[1])

        # Set to training mode
        self.model.train()
        self.likelihood.train()

        # Optimize hyperparameters
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(self.config.model_training_iterations):
            optimizer.zero_grad()
            output = self.model(self.X_train)
            loss = -mll(output, self.y_train)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(
                    f"Iteration {i}/{self.config.model_training_iterations}, Loss: {loss.item():.4f}"
                )

        # Set to evaluation mode
        self.model.eval()
        self.likelihood.eval()

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mean and variance."""
        if self.model is None:
            raise BayesianOptimizationError("Model not fitted yet")

        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = self.likelihood(self.model(X_tensor))
            mean = posterior.mean.cpu().numpy()
            variance = posterior.variance.cpu().numpy()

        return mean, variance

    def sample(self, X: np.ndarray, n_samples: int = 1) -> np.ndarray:
        """Sample from the posterior."""
        if self.model is None:
            raise BayesianOptimizationError("Model not fitted yet")

        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            posterior = self.likelihood(self.model(X_tensor))
            samples = posterior.rsample(sample_shape=torch.Size([n_samples]))

        return samples.cpu().numpy().T

    def get_model_parameters(self) -> Dict[str, Any]:
        """Get model hyperparameters."""
        if self.model is None:
            return {}

        return {
            "noise_variance": self.likelihood.noise.item(),
            "signal_variance": self.model.covar_module.outputscale.item(),
            "lengthscales": self.model.covar_module.base_kernel.lengthscale.detach()
            .cpu()
            .numpy(),
            "mean": self.model.mean_module.constant.item(),
        }


class JAXGaussianProcess(SurrogateModel):
    """JAX-based Gaussian Process implementation."""

    def __init__(self, config: BayesianOptimizationConfig):
        if not JAX_AVAILABLE:
            raise BayesianOptimizationError("JAX required for JAX GP implementation")

        self.config = config
        self.key = random.PRNGKey(42)
        self.params = None
        self.X_train = None
        self.y_train = None

    def _rbf_kernel(
        self,
        X1: jnp.ndarray,
        X2: jnp.ndarray,
        lengthscale: jnp.ndarray,
        signal_variance: float,
    ) -> jnp.ndarray:
        """RBF kernel function."""
        X1_scaled = X1 / lengthscale
        X2_scaled = X2 / lengthscale

        # Compute squared distances
        sq_dists = jnp.sum((X1_scaled[:, None, :] - X2_scaled[None, :, :]) ** 2, axis=2)

        return signal_variance * jnp.exp(-0.5 * sq_dists)

    def _fit_gp(self, X: jnp.ndarray, y: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """Fit GP hyperparameters using JAX optimization."""

        def neg_log_likelihood(params):
            lengthscale = jnp.exp(params["log_lengthscale"])
            signal_variance = jnp.exp(params["log_signal_variance"])
            noise_variance = jnp.exp(params["log_noise_variance"])

            K = self._rbf_kernel(X, X, lengthscale, signal_variance)
            K_noise = K + noise_variance * jnp.eye(K.shape[0])

            try:
                L = jnp.linalg.cholesky(K_noise)
                alpha = jnp.linalg.solve(L, y)

                log_likelihood = (
                    -0.5 * jnp.sum(alpha**2)
                    - jnp.sum(jnp.log(jnp.diag(L)))
                    - 0.5 * len(y) * jnp.log(2 * jnp.pi)
                )
                return -log_likelihood
            except:
                return 1e6  # Return large value if Cholesky fails

        # Initialize parameters
        initial_params = {
            "log_lengthscale": jnp.log(jnp.ones(X.shape[1])),
            "log_signal_variance": jnp.log(1.0),
            "log_noise_variance": jnp.log(0.1),
        }

        # Optimize using JAX
        optimizer = jax.example_libraries.optimizers.adam(learning_rate=0.01)
        opt_state = optimizer.init(initial_params)

        @jit
        def update(opt_state, params):
            loss, grads = jax.value_and_grad(neg_log_likelihood)(params)
            return optimizer.update(0, grads, opt_state), loss

        params = initial_params
        for i in range(self.config.model_training_iterations):
            opt_state, loss = update(opt_state, params)
            params = optimizer.get_params(opt_state)

            if i % 100 == 0:
                print(f"JAX GP Iteration {i}, Loss: {loss:.4f}")

        return params

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the JAX Gaussian Process."""
        self.X_train = jnp.array(X)
        self.y_train = jnp.array(y.flatten())

        # Optimize hyperparameters
        self.params = self._fit_gp(self.X_train, self.y_train)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mean and variance."""
        if self.params is None:
            raise BayesianOptimizationError("Model not fitted yet")

        X_test = jnp.array(X)

        # Extract hyperparameters
        lengthscale = jnp.exp(self.params["log_lengthscale"])
        signal_variance = jnp.exp(self.params["log_signal_variance"])
        noise_variance = jnp.exp(self.params["log_noise_variance"])

        # Compute kernel matrices
        K_train = self._rbf_kernel(
            self.X_train, self.X_train, lengthscale, signal_variance
        )
        K_train_noise = K_train + noise_variance * jnp.eye(K_train.shape[0])
        K_test = self._rbf_kernel(X_test, self.X_train, lengthscale, signal_variance)
        K_test_test = self._rbf_kernel(X_test, X_test, lengthscale, signal_variance)

        # Compute posterior mean and variance
        L = jnp.linalg.cholesky(K_train_noise)
        alpha = jnp.linalg.solve(L, self.y_train)

        mean = jnp.dot(K_test, jnp.linalg.solve(K_train_noise, self.y_train))
        v = jnp.linalg.solve(L, K_test.T)
        variance = jnp.diag(K_test_test) - jnp.sum(v**2, axis=0)

        return np.array(mean), np.array(variance)

    def sample(self, X: np.ndarray, n_samples: int = 1) -> np.ndarray:
        """Sample from the posterior."""
        mean, variance = self.predict(X)

        # Generate samples
        samples = []
        for _ in range(n_samples):
            self.key, subkey = random.split(self.key)
            sample = random.normal(subkey, shape=mean.shape) * np.sqrt(variance) + mean
            samples.append(sample)

        return np.array(samples).T

    def get_model_parameters(self) -> Dict[str, Any]:
        """Get model hyperparameters."""
        if self.params is None:
            return {}

        return {
            "lengthscales": np.array(jnp.exp(self.params["log_lengthscale"])),
            "signal_variance": float(jnp.exp(self.params["log_signal_variance"])),
            "noise_variance": float(jnp.exp(self.params["log_noise_variance"])),
        }


class AcquisitionFunctionOptimizer:
    """Optimizes acquisition functions for candidate point selection."""

    def __init__(
        self, config: BayesianOptimizationConfig, parameter_spaces: List[ParameterSpace]
    ):
        self.config = config
        self.parameter_spaces = parameter_spaces
        self.bounds = np.array([space.bounds for space in parameter_spaces])

    def optimize(
        self, acquisition_func: Callable, n_candidates: int = 1000
    ) -> Tuple[np.ndarray, float]:
        """Optimize acquisition function to find next evaluation point."""

        if self.config.optimization_strategy == OptimizationStrategy.RANDOM_SEARCH:
            return self._random_search(acquisition_func, n_candidates)
        elif self.config.optimization_strategy == OptimizationStrategy.GRADIENT_BASED:
            return self._gradient_based_optimization(acquisition_func)
        elif self.config.optimization_strategy == OptimizationStrategy.MULTI_START:
            return self._multi_start_optimization(acquisition_func)
        else:
            raise BayesianOptimizationError(
                f"Unknown optimization strategy: {self.config.optimization_strategy}"
            )

    def _random_search(
        self, acquisition_func: Callable, n_candidates: int
    ) -> Tuple[np.ndarray, float]:
        """Random search over candidate points."""
        # Generate random candidates
        candidates = np.random.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            size=(n_candidates, len(self.parameter_spaces)),
        )

        # Evaluate acquisition function
        acq_values = acquisition_func(candidates)

        # Find best candidate
        best_idx = np.argmax(acq_values)
        return candidates[best_idx], acq_values[best_idx]

    def _gradient_based_optimization(
        self, acquisition_func: Callable
    ) -> Tuple[np.ndarray, float]:
        """Gradient-based optimization of acquisition function."""

        def objective(x):
            return -acquisition_func(x.reshape(1, -1))[
                0
            ]  # Minimize negative acquisition

        # Random starting point
        x0 = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])

        # Optimize
        result = scipy.optimize.minimize(
            objective, x0, bounds=self.bounds, method="L-BFGS-B"
        )

        return result.x, -result.fun

    def _multi_start_optimization(
        self, acquisition_func: Callable, n_starts: int = 10
    ) -> Tuple[np.ndarray, float]:
        """Multi-start gradient-based optimization."""
        best_x = None
        best_value = -np.inf

        for _ in range(n_starts):
            x, value = self._gradient_based_optimization(acquisition_func)
            if value > best_value:
                best_value = value
                best_x = x

        return best_x, best_value


class AcquisitionFunctions:
    """Collection of acquisition functions for Bayesian optimization."""

    @staticmethod
    def expected_improvement(
        mean: np.ndarray, variance: np.ndarray, best_value: float, xi: float = 0.01
    ) -> np.ndarray:
        """Expected Improvement acquisition function."""
        std = np.sqrt(variance)
        z = (mean - best_value - xi) / (std + 1e-9)

        ei = (mean - best_value - xi) * scipy.stats.norm.cdf(
            z
        ) + std * scipy.stats.norm.pdf(z)
        return np.maximum(ei, 0.0)

    @staticmethod
    def probability_of_improvement(
        mean: np.ndarray, variance: np.ndarray, best_value: float, xi: float = 0.01
    ) -> np.ndarray:
        """Probability of Improvement acquisition function."""
        std = np.sqrt(variance)
        z = (mean - best_value - xi) / (std + 1e-9)
        return scipy.stats.norm.cdf(z)

    @staticmethod
    def upper_confidence_bound(
        mean: np.ndarray, variance: np.ndarray, exploration_weight: float = 2.0
    ) -> np.ndarray:
        """Upper Confidence Bound acquisition function."""
        std = np.sqrt(variance)
        return mean + exploration_weight * std

    @staticmethod
    def thompson_sampling(
        surrogate_model: SurrogateModel, X_candidates: np.ndarray, n_samples: int = 1
    ) -> np.ndarray:
        """Thompson Sampling acquisition function."""
        samples = surrogate_model.sample(X_candidates, n_samples)
        return np.max(samples, axis=1)


class BayesianOptimizer:
    """Main Bayesian optimization engine with GPU acceleration."""

    def __init__(
        self, parameter_spaces: List[ParameterSpace], config: BayesianOptimizationConfig
    ):
        self.parameter_spaces = parameter_spaces
        self.config = config

        # Initialize components
        self.surrogate_model = self._create_surrogate_model()
        self.acquisition_optimizer = AcquisitionFunctionOptimizer(
            config, parameter_spaces
        )

        # Optimization state
        self.X_evaluated = []
        self.y_evaluated = []
        self.iteration = 0
        self.best_x = None
        self.best_y = -np.inf

        # Multi-objective state
        self.multi_objective_optimizer = None
        self.is_multi_objective = False

        # History tracking
        self.optimization_history = []

    def _create_surrogate_model(self) -> SurrogateModel:
        """Create appropriate surrogate model based on configuration."""
        if self.config.surrogate_model == SurrogateModelType.GAUSSIAN_PROCESS:
            if TORCH_AVAILABLE and self.config.use_gpu:
                return GPyTorchGaussianProcess(self.config)
            elif JAX_AVAILABLE:
                return JAXGaussianProcess(self.config)
            else:
                raise BayesianOptimizationError(
                    "No suitable GP implementation available"
                )
        else:
            raise BayesianOptimizationError(
                f"Unsupported surrogate model: {self.config.surrogate_model}"
            )

    def _generate_initial_points(self) -> np.ndarray:
        """Generate initial evaluation points using Latin Hypercube Sampling."""
        if self.config.latin_hypercube_initial:
            from scipy.stats import qmc

            sampler = qmc.LatinHypercube(d=len(self.parameter_spaces))
            unit_samples = sampler.random(n=self.config.n_initial_points)

            # Transform to parameter space bounds
            initial_points = np.zeros_like(unit_samples)
            for i, space in enumerate(self.parameter_spaces):
                initial_points[:, i] = space.denormalize(unit_samples[:, i])

            return initial_points
        else:
            # Random sampling
            initial_points = []
            for _ in range(self.config.n_initial_points):
                point = []
                for space in self.parameter_spaces:
                    point.append(space.sample(1)[0])
                initial_points.append(point)
            return np.array(initial_points)

    def _create_acquisition_function(self) -> Callable:
        """Create acquisition function based on configuration."""

        def acquisition_func(X_candidates: np.ndarray) -> np.ndarray:
            if len(X_candidates.shape) == 1:
                X_candidates = X_candidates.reshape(1, -1)

            mean, variance = self.surrogate_model.predict(X_candidates)

            if (
                self.config.acquisition_function
                == AcquisitionFunction.EXPECTED_IMPROVEMENT
            ):
                return AcquisitionFunctions.expected_improvement(
                    mean, variance, self.best_y, self.config.xi
                )
            elif (
                self.config.acquisition_function
                == AcquisitionFunction.PROBABILITY_OF_IMPROVEMENT
            ):
                return AcquisitionFunctions.probability_of_improvement(
                    mean, variance, self.best_y, self.config.xi
                )
            elif (
                self.config.acquisition_function
                == AcquisitionFunction.UPPER_CONFIDENCE_BOUND
            ):
                return AcquisitionFunctions.upper_confidence_bound(
                    mean, variance, self.config.exploration_weight
                )
            elif (
                self.config.acquisition_function
                == AcquisitionFunction.THOMPSON_SAMPLING
            ):
                return AcquisitionFunctions.thompson_sampling(
                    self.surrogate_model, X_candidates, self.config.n_thompson_samples
                )
            else:
                raise BayesianOptimizationError(
                    f"Unsupported acquisition function: {self.config.acquisition_function}"
                )

        return acquisition_func

    async def optimize(
        self,
        objective_function: Callable,
        multi_objective_configs: Optional[List[ObjectiveConfig]] = None,
    ) -> OptimizationResult:
        """Run Bayesian optimization."""
        start_time = time.time()

        # Check if multi-objective
        self.is_multi_objective = multi_objective_configs is not None
        if self.is_multi_objective:
            self.multi_objective_optimizer = MPOptimizer(
                objectives=multi_objective_configs, use_gpu=self.config.use_gpu
            )

        try:
            # Generate initial points
            initial_points = self._generate_initial_points()

            # Evaluate initial points
            for point in initial_points:
                y = await self._evaluate_objective(objective_function, point)
                self.X_evaluated.append(point)
                self.y_evaluated.append(y)

                if not self.is_multi_objective and y > self.best_y:
                    self.best_y = y
                    self.best_x = point.copy()

            # Main optimization loop
            for iteration in range(self.config.max_iterations):
                self.iteration = iteration

                # Fit surrogate model
                model_start = time.time()
                X_array = np.array(self.X_evaluated)

                if self.is_multi_objective:
                    # For multi-objective, optimize scalarized objective
                    y_array = self._scalarize_objectives(np.array(self.y_evaluated))
                else:
                    y_array = np.array(self.y_evaluated)

                self.surrogate_model.fit(X_array, y_array)
                model_time = time.time() - model_start

                # Optimize acquisition function
                acq_start = time.time()
                acquisition_func = self._create_acquisition_function()

                # Get candidate points (batch if specified)
                candidate_points = []
                acquisition_values = []

                for _ in range(self.config.batch_size):
                    x_next, acq_value = self.acquisition_optimizer.optimize(
                        acquisition_func
                    )
                    candidate_points.append(x_next)
                    acquisition_values.append(acq_value)

                acq_time = time.time() - acq_start

                # Evaluate new points
                eval_start = time.time()
                for i, x_next in enumerate(candidate_points):
                    y_next = await self._evaluate_objective(objective_function, x_next)

                    self.X_evaluated.append(x_next)
                    self.y_evaluated.append(y_next)

                    if not self.is_multi_objective and y_next > self.best_y:
                        self.best_y = y_next
                        self.best_x = x_next.copy()

                eval_time = time.time() - eval_start

                # Record iteration
                self.optimization_history.append(
                    {
                        "iteration": iteration,
                        "best_objective": self.best_y
                        if not self.is_multi_objective
                        else np.max(y_array),
                        "acquisition_values": acquisition_values,
                        "model_training_time": model_time,
                        "acquisition_time": acq_time,
                        "evaluation_time": eval_time,
                    }
                )

                # Check convergence
                if self._check_convergence():
                    print(f"Converged after {iteration + 1} iterations")
                    break

                print(
                    f"Iteration {iteration + 1}/{self.config.max_iterations}, "
                    f"Best: {self.best_y:.4f}"
                    if not self.is_multi_objective
                    else f"Latest eval: {y_array[-1]:.4f}"
                )

            # Handle multi-objective results
            pareto_front = None
            if self.is_multi_objective:
                # Create property dictionary for multi-objective optimization
                properties = {}
                for i, obj_config in enumerate(multi_objective_configs):
                    properties[obj_config.name] = np.array(
                        [y[i] for y in self.y_evaluated]
                    )

                # Get molecule identifiers
                molecules = [f"point_{i}" for i in range(len(self.X_evaluated))]

                # Run multi-objective optimization
                pareto_front = self.multi_objective_optimizer.optimize(
                    molecules, properties
                )

                # Update best results based on Pareto front
                if len(pareto_front.pareto_front.pareto_indices) > 0:
                    best_idx = pareto_front.pareto_front.pareto_indices[
                        0
                    ]  # First Pareto point
                    self.best_x = self.X_evaluated[best_idx]
                    self.best_y = self.y_evaluated[best_idx]

            total_time = time.time() - start_time

            # Create result
            result = OptimizationResult(
                best_parameters={
                    space.name: val
                    for space, val in zip(self.parameter_spaces, self.best_x)
                },
                best_objective=self.best_y if not self.is_multi_objective else None,
                best_objectives=self.best_y if self.is_multi_objective else None,
                parameter_history=[
                    {space.name: val for space, val in zip(self.parameter_spaces, x)}
                    for x in self.X_evaluated
                ],
                objective_history=self.y_evaluated,
                acquisition_history=[
                    h["acquisition_values"] for h in self.optimization_history
                ],
                model_performance=self.surrogate_model.get_model_parameters(),
                total_time=total_time,
                model_training_time=sum(
                    h["model_training_time"] for h in self.optimization_history
                ),
                acquisition_optimization_time=sum(
                    h["acquisition_time"] for h in self.optimization_history
                ),
                objective_evaluation_time=sum(
                    h["evaluation_time"] for h in self.optimization_history
                ),
                pareto_front=pareto_front,
            )

            return result

        except Exception as e:
            raise BayesianOptimizationError(f"Optimization failed: {e}")

    async def _evaluate_objective(
        self, objective_function: Callable, x: np.ndarray
    ) -> Union[float, np.ndarray]:
        """Evaluate objective function at point x."""
        if asyncio.iscoroutinefunction(objective_function):
            return await objective_function(x)
        else:
            return objective_function(x)

    def _scalarize_objectives(self, y_multi: np.ndarray) -> np.ndarray:
        """Scalarize multi-objective values for surrogate model training."""
        if self.config.scalarization_weights is not None:
            weights = self.config.scalarization_weights
        else:
            weights = np.ones(y_multi.shape[1]) / y_multi.shape[1]

        return np.sum(y_multi * weights, axis=1)

    def _check_convergence(self) -> bool:
        """Check if optimization has converged."""
        if len(self.optimization_history) < 5:
            return False

        # Check if best objective has improved significantly in last few iterations
        recent_bests = [h["best_objective"] for h in self.optimization_history[-5:]]
        improvement = max(recent_bests) - min(recent_bests)

        return improvement < self.config.convergence_tolerance


# Utility functions for common optimization scenarios


def optimize_molecular_properties(
    parameter_spaces: List[ParameterSpace],
    objective_function: Callable,
    config: Optional[BayesianOptimizationConfig] = None,
) -> OptimizationResult:
    """Optimize molecular properties using Bayesian optimization."""

    if config is None:
        config = BayesianOptimizationConfig()

    optimizer = BayesianOptimizer(parameter_spaces, config)

    # Run optimization
    import asyncio

    return asyncio.run(optimizer.optimize(objective_function))


def optimize_process_conditions(
    temperature_range: Tuple[float, float],
    pressure_range: Tuple[float, float],
    flow_rate_range: Tuple[float, float],
    objective_function: Callable,
    config: Optional[BayesianOptimizationConfig] = None,
) -> OptimizationResult:
    """Optimize chemical process conditions."""

    parameter_spaces = [
        ParameterSpace("temperature", temperature_range),
        ParameterSpace("pressure", pressure_range),
        ParameterSpace("flow_rate", flow_rate_range),
    ]

    return optimize_molecular_properties(parameter_spaces, objective_function, config)


def multi_objective_drug_discovery_optimization(
    parameter_spaces: List[ParameterSpace],
    objective_function: Callable,
    config: Optional[BayesianOptimizationConfig] = None,
) -> OptimizationResult:
    """Multi-objective optimization for drug discovery."""

    from .multi_objective_optimization import create_drug_discovery_objectives

    if config is None:
        config = BayesianOptimizationConfig(
            acquisition_function=AcquisitionFunction.MULTI_OBJECTIVE_EI,
            max_iterations=50,
        )

    objectives = create_drug_discovery_objectives()
    optimizer = BayesianOptimizer(parameter_spaces, config)

    import asyncio

    return asyncio.run(optimizer.optimize(objective_function, objectives))


def create_example_parameter_spaces() -> List[ParameterSpace]:
    """Create example parameter spaces for testing."""
    return [
        ParameterSpace("temperature", (273.15, 373.15), "continuous"),
        ParameterSpace("pressure", (1e5, 10e5), "continuous", log_scale=True),
        ParameterSpace("catalyst_loading", (0.1, 10.0), "continuous"),
        ParameterSpace("reaction_time", (0.5, 24.0), "continuous"),
        ParameterSpace(
            "solvent_type",
            (0, 3),
            "discrete",
            discrete_values=["water", "ethanol", "acetone", "dmf"],
        ),
    ]


# Example objective functions for testing and demonstration


def branin_function(x: np.ndarray) -> float:
    """Branin function for testing (2D)."""
    x1, x2 = x[0], x[1]
    a, b, c, r, s, t = 1, 5.1 / (4 * np.pi**2), 5 / np.pi, 6, 10, 1 / (8 * np.pi)

    term1 = a * (x2 - b * x1**2 + c * x1 - r) ** 2
    term2 = s * (1 - t) * np.cos(x1)

    return -(term1 + term2 + s)  # Negative for maximization


def synthetic_chemical_objective(x: np.ndarray) -> np.ndarray:
    """Synthetic multi-objective chemical optimization function."""
    temp, pressure, catalyst = x[0], x[1], x[2]

    # Yield objective (maximize)
    yield_obj = (
        0.8
        - 0.001 * (temp - 350) ** 2
        - 0.0001 * (pressure - 2e5) ** 2
        + 0.1 * catalyst
    )

    # Selectivity objective (maximize)
    selectivity_obj = (
        0.9 - 0.0005 * (temp - 300) ** 2 + 0.05 * catalyst - 0.00001 * pressure
    )

    # Cost objective (minimize -> negate for maximization)
    cost_obj = -(0.01 * temp + 0.000001 * pressure + 0.5 * catalyst**2)

    return np.array([yield_obj, selectivity_obj, cost_obj])
