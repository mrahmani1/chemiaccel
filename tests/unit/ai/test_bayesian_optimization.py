"""
Comprehensive tests for the GPU-Accelerated Bayesian Optimization Framework.

This test suite covers all components of the Bayesian optimization system including:
- Surrogate model implementations (GPyTorch, JAX)
- Acquisition function optimization
- Parameter space handling
- Multi-objective optimization integration
- GPU acceleration features
"""

import asyncio
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

# Import the Bayesian optimization components
from src.chemi_accel.ai.bayesian_optimization import (
    CUPY_AVAILABLE,
    JAX_AVAILABLE,
    TORCH_AVAILABLE,
    AcquisitionFunction,
    AcquisitionFunctionOptimizer,
    AcquisitionFunctions,
    BayesianOptimizationConfig,
    BayesianOptimizationError,
    BayesianOptimizer,
    GPyTorchGaussianProcess,
    JAXGaussianProcess,
    OptimizationResult,
    OptimizationStrategy,
    ParameterSpace,
    SurrogateModelType,
    branin_function,
    create_example_parameter_spaces,
    optimize_molecular_properties,
    optimize_process_conditions,
    synthetic_chemical_objective,
)
from src.chemi_accel.ai.multi_objective_optimization import ObjectiveConfig


class TestParameterSpace:
    """Test parameter space functionality."""

    def test_continuous_parameter_space(self):
        """Test continuous parameter space sampling and normalization."""
        space = ParameterSpace("temperature", (273.15, 373.15), "continuous")

        # Test sampling
        samples = space.sample(100, random_state=42)
        assert len(samples) == 100
        assert np.all(samples >= 273.15)
        assert np.all(samples <= 373.15)

        # Test normalization
        test_values = np.array([273.15, 323.15, 373.15])
        normalized = space.normalize(test_values)
        expected = np.array([0.0, 0.5, 1.0])
        np.testing.assert_allclose(normalized, expected)

        # Test denormalization
        denormalized = space.denormalize(normalized)
        np.testing.assert_allclose(denormalized, test_values)

    def test_log_scale_parameter_space(self):
        """Test log-scale parameter space."""
        space = ParameterSpace("pressure", (1e5, 10e5), "continuous", log_scale=True)

        # Test sampling
        samples = space.sample(100, random_state=42)
        assert len(samples) == 100
        assert np.all(samples >= 1e5)
        assert np.all(samples <= 10e5)

        # Test normalization with log scale
        test_values = np.array([1e5, np.sqrt(1e5 * 10e5), 10e5])
        normalized = space.normalize(test_values)
        expected = np.array([0.0, 0.5, 1.0])
        np.testing.assert_allclose(normalized, expected, rtol=1e-6)

    def test_discrete_parameter_space(self):
        """Test discrete parameter space."""
        space = ParameterSpace("catalyst_amount", (1, 5), "discrete")

        # Test sampling
        samples = space.sample(50, random_state=42)
        assert len(samples) == 50
        assert np.all(np.isin(samples, [1, 2, 3, 4, 5]))

        # Test with discrete values
        discrete_space = ParameterSpace(
            "solvent",
            (0, 2),
            "discrete",
            discrete_values=["water", "ethanol", "acetone"],
        )
        samples = discrete_space.sample(30, random_state=42)
        assert len(samples) == 30
        assert np.all(np.isin(samples, ["water", "ethanol", "acetone"]))

    def test_categorical_parameter_space(self):
        """Test categorical parameter space."""
        space = ParameterSpace(
            "catalyst_type",
            (0, 2),
            "categorical",
            categorical_values=["Pd", "Pt", "Ru"],
        )

        samples = space.sample(40, random_state=42)
        assert len(samples) == 40
        assert np.all(np.isin(samples, ["Pd", "Pt", "Ru"]))

    def test_invalid_parameter_space(self):
        """Test error handling for invalid parameter spaces."""
        with pytest.raises(BayesianOptimizationError):
            space = ParameterSpace("invalid", (0, 1), "invalid_type")
            space.sample(10)

        with pytest.raises(BayesianOptimizationError):
            space = ParameterSpace("cat", (0, 1), "categorical")
            space.sample(10)  # No categorical values specified


class TestBayesianOptimizationConfig:
    """Test configuration object."""

    def test_default_config(self):
        """Test default configuration."""
        config = BayesianOptimizationConfig()

        assert config.surrogate_model == SurrogateModelType.GAUSSIAN_PROCESS
        assert config.acquisition_function == AcquisitionFunction.EXPECTED_IMPROVEMENT
        assert config.optimization_strategy == OptimizationStrategy.MULTI_START
        assert config.use_gpu is True
        assert config.n_initial_points == 10
        assert config.max_iterations == 100

    def test_custom_config(self):
        """Test custom configuration."""
        config = BayesianOptimizationConfig(
            surrogate_model=SurrogateModelType.SPARSE_GP,
            acquisition_function=AcquisitionFunction.UPPER_CONFIDENCE_BOUND,
            optimization_strategy=OptimizationStrategy.RANDOM_SEARCH,
            use_gpu=False,
            n_initial_points=20,
            max_iterations=50,
            exploration_weight=3.0,
        )

        assert config.surrogate_model == SurrogateModelType.SPARSE_GP
        assert config.acquisition_function == AcquisitionFunction.UPPER_CONFIDENCE_BOUND
        assert config.optimization_strategy == OptimizationStrategy.RANDOM_SEARCH
        assert config.use_gpu is False
        assert config.n_initial_points == 20
        assert config.max_iterations == 50
        assert config.exploration_weight == 3.0


class TestAcquisitionFunctions:
    """Test acquisition function implementations."""

    def test_expected_improvement(self):
        """Test Expected Improvement acquisition function."""
        mean = np.array([0.5, 0.8, 0.2])
        variance = np.array([0.1, 0.05, 0.2])
        best_value = 0.6

        ei = AcquisitionFunctions.expected_improvement(mean, variance, best_value)

        assert len(ei) == 3
        assert np.all(ei >= 0)
        assert ei[1] > ei[0]  # Higher mean should give higher EI when above best
        assert ei[2] < ei[0]  # Lower mean should give lower EI

    def test_probability_of_improvement(self):
        """Test Probability of Improvement acquisition function."""
        mean = np.array([0.5, 0.8, 0.2])
        variance = np.array([0.1, 0.05, 0.2])
        best_value = 0.6

        pi = AcquisitionFunctions.probability_of_improvement(mean, variance, best_value)

        assert len(pi) == 3
        assert np.all(pi >= 0)
        assert np.all(pi <= 1)
        assert pi[1] > pi[0]  # Higher mean should give higher PI when above best

    def test_upper_confidence_bound(self):
        """Test Upper Confidence Bound acquisition function."""
        mean = np.array([0.5, 0.8, 0.2])
        variance = np.array([0.1, 0.05, 0.2])
        exploration_weight = 2.0

        ucb = AcquisitionFunctions.upper_confidence_bound(
            mean, variance, exploration_weight
        )

        assert len(ucb) == 3
        assert np.all(ucb >= mean)  # UCB should be at least the mean

        # Higher variance should contribute more to UCB
        assert (ucb[2] - mean[2]) > (ucb[1] - mean[1])  # variance[2] > variance[1]


class TestSurrogateModels:
    """Test surrogate model implementations."""

    def create_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create test data for surrogate model fitting."""
        np.random.seed(42)
        X = np.random.uniform(-1, 1, (20, 2))
        y = (X[:, 0] ** 2 + X[:, 1] ** 2 + 0.1 * np.random.randn(20)).reshape(-1, 1)
        return X, y

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_jax_gaussian_process(self):
        """Test JAX Gaussian Process implementation."""
        config = BayesianOptimizationConfig(
            use_gpu=False, model_training_iterations=100
        )
        model = JAXGaussianProcess(config)

        X_train, y_train = self.create_test_data()

        # Test fitting
        model.fit(X_train, y_train)
        assert model.params is not None

        # Test prediction
        X_test = np.random.uniform(-1, 1, (5, 2))
        mean, variance = model.predict(X_test)

        assert len(mean) == 5
        assert len(variance) == 5
        assert np.all(variance > 0)

        # Test sampling
        samples = model.sample(X_test, n_samples=3)
        assert samples.shape == (5, 3)

        # Test parameter extraction
        params = model.get_model_parameters()
        assert "lengthscales" in params
        assert "signal_variance" in params
        assert "noise_variance" in params

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch/GPyTorch not available")
    def test_gpytorch_gaussian_process(self):
        """Test GPyTorch Gaussian Process implementation."""
        config = BayesianOptimizationConfig(
            use_gpu=False, device="cpu", model_training_iterations=50
        )
        model = GPyTorchGaussianProcess(config)

        X_train, y_train = self.create_test_data()

        # Test fitting
        model.fit(X_train, y_train)
        assert model.model is not None
        assert model.likelihood is not None

        # Test prediction
        X_test = np.random.uniform(-1, 1, (5, 2))
        mean, variance = model.predict(X_test)

        assert len(mean) == 5
        assert len(variance) == 5
        assert np.all(variance > 0)

        # Test sampling
        samples = model.sample(X_test, n_samples=3)
        assert samples.shape == (5, 3)

        # Test parameter extraction
        params = model.get_model_parameters()
        assert "noise_variance" in params
        assert "signal_variance" in params
        assert "lengthscales" in params


class TestAcquisitionFunctionOptimizer:
    """Test acquisition function optimization."""

    def create_test_optimizer(self) -> AcquisitionFunctionOptimizer:
        """Create test acquisition function optimizer."""
        parameter_spaces = [
            ParameterSpace("x1", (-5, 5), "continuous"),
            ParameterSpace("x2", (-5, 5), "continuous"),
        ]
        config = BayesianOptimizationConfig()
        return AcquisitionFunctionOptimizer(config, parameter_spaces)

    def test_random_search_optimization(self):
        """Test random search optimization."""
        optimizer = self.create_test_optimizer()
        optimizer.config.optimization_strategy = OptimizationStrategy.RANDOM_SEARCH

        # Simple quadratic acquisition function (maximize at origin)
        def acquisition_func(X):
            return -(X[:, 0] ** 2 + X[:, 1] ** 2)

        best_x, best_value = optimizer.optimize(acquisition_func, n_candidates=1000)

        assert len(best_x) == 2
        assert abs(best_x[0]) < 1.0  # Should be close to origin
        assert abs(best_x[1]) < 1.0
        assert best_value >= -2.0

    def test_gradient_based_optimization(self):
        """Test gradient-based optimization."""
        optimizer = self.create_test_optimizer()
        optimizer.config.optimization_strategy = OptimizationStrategy.GRADIENT_BASED

        # Simple quadratic acquisition function
        def acquisition_func(X):
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            return -(X[:, 0] ** 2 + X[:, 1] ** 2)

        best_x, best_value = optimizer.optimize(acquisition_func)

        assert len(best_x) == 2
        assert abs(best_x[0]) < 0.5  # Should be very close to origin
        assert abs(best_x[1]) < 0.5

    def test_multi_start_optimization(self):
        """Test multi-start optimization."""
        optimizer = self.create_test_optimizer()
        optimizer.config.optimization_strategy = OptimizationStrategy.MULTI_START

        # Function with multiple local optima
        def acquisition_func(X):
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            return -(X[:, 0] ** 2 + X[:, 1] ** 2) + 0.5 * np.sin(5 * X[:, 0]) * np.sin(
                5 * X[:, 1]
            )

        best_x, best_value = optimizer.optimize(acquisition_func)

        assert len(best_x) == 2
        assert best_value > -1.0  # Should find a reasonable optimum


class TestBayesianOptimizer:
    """Test main Bayesian optimizer."""

    def create_test_optimizer(self) -> BayesianOptimizer:
        """Create test Bayesian optimizer."""
        parameter_spaces = [
            ParameterSpace("x1", (-5, 5), "continuous"),
            ParameterSpace("x2", (-5, 5), "continuous"),
        ]
        config = BayesianOptimizationConfig(
            n_initial_points=5,
            max_iterations=3,
            model_training_iterations=20,
            use_gpu=False,
            device="cpu",
        )
        return BayesianOptimizer(parameter_spaces, config)

    def test_initial_point_generation(self):
        """Test initial point generation."""
        optimizer = self.create_test_optimizer()

        # Test Latin Hypercube Sampling
        optimizer.config.latin_hypercube_initial = True
        initial_points = optimizer._generate_initial_points()

        assert initial_points.shape == (5, 2)
        assert np.all(initial_points >= -5)
        assert np.all(initial_points <= 5)

        # Test random sampling
        optimizer.config.latin_hypercube_initial = False
        initial_points = optimizer._generate_initial_points()

        assert initial_points.shape == (5, 2)
        assert np.all(initial_points >= -5)
        assert np.all(initial_points <= 5)

    @pytest.mark.asyncio
    async def test_simple_optimization(self):
        """Test simple optimization with quadratic function."""
        optimizer = self.create_test_optimizer()

        # Simple quadratic objective (maximize at origin)
        async def objective_function(x):
            await asyncio.sleep(0.001)  # Simulate computation
            return -(x[0] ** 2 + x[1] ** 2)

        result = await optimizer.optimize(objective_function)

        assert isinstance(result, OptimizationResult)
        assert len(result.parameter_history) == 8  # 5 initial + 3 iterations
        assert len(result.objective_history) == 8
        assert result.total_time > 0

        # Check that optimization found a reasonable solution
        best_params = result.best_parameters
        assert abs(best_params["x1"]) < 2.0
        assert abs(best_params["x2"]) < 2.0
        assert result.best_objective > -10.0

    @pytest.mark.asyncio
    async def test_branin_optimization(self):
        """Test optimization on Branin function."""
        parameter_spaces = [
            ParameterSpace("x1", (-5, 10), "continuous"),
            ParameterSpace("x2", (0, 15), "continuous"),
        ]
        config = BayesianOptimizationConfig(
            n_initial_points=8,
            max_iterations=5,
            model_training_iterations=50,
            use_gpu=False,
            device="cpu",
            acquisition_function=AcquisitionFunction.EXPECTED_IMPROVEMENT,
        )
        optimizer = BayesianOptimizer(parameter_spaces, config)

        async def objective_function(x):
            return branin_function(x)

        result = await optimizer.optimize(objective_function)

        assert isinstance(result, OptimizationResult)
        assert result.best_objective > -10.0  # Branin minimum is around -0.4

        # Check convergence
        improvement = result.objective_history[-1] - result.objective_history[0]
        assert improvement >= 0  # Should improve or stay the same

    @pytest.mark.asyncio
    async def test_multi_objective_optimization(self):
        """Test multi-objective optimization."""
        parameter_spaces = [
            ParameterSpace("temperature", (300, 400), "continuous"),
            ParameterSpace("pressure", (1e5, 5e5), "continuous"),
            ParameterSpace("catalyst", (0.1, 2.0), "continuous"),
        ]

        config = BayesianOptimizationConfig(
            n_initial_points=6,
            max_iterations=3,
            model_training_iterations=20,
            use_gpu=False,
            device="cpu",
            scalarization_weights=np.array([0.4, 0.4, 0.2]),
        )

        optimizer = BayesianOptimizer(parameter_spaces, config)

        # Multi-objective configuration
        objectives = [
            ObjectiveConfig("yield", minimize=False, weight=1.0),
            ObjectiveConfig("selectivity", minimize=False, weight=1.0),
            ObjectiveConfig("cost", minimize=True, weight=0.5),
        ]

        async def objective_function(x):
            return synthetic_chemical_objective(x)

        result = await optimizer.optimize(objective_function, objectives)

        assert isinstance(result, OptimizationResult)
        assert result.pareto_front is not None
        assert len(result.pareto_front.molecules) > 0
        assert len(result.pareto_front.pareto_front.pareto_indices) > 0

    def test_convergence_checking(self):
        """Test convergence checking mechanism."""
        optimizer = self.create_test_optimizer()

        # Simulate optimization history with no improvement
        optimizer.optimization_history = [
            {"iteration": 0, "best_objective": 1.0},
            {"iteration": 1, "best_objective": 1.001},
            {"iteration": 2, "best_objective": 1.0005},
            {"iteration": 3, "best_objective": 1.0002},
            {"iteration": 4, "best_objective": 1.0001},
        ]

        # Should converge due to small improvements
        assert optimizer._check_convergence()

        # Simulate optimization history with significant improvement
        optimizer.optimization_history = [
            {"iteration": 0, "best_objective": 1.0},
            {"iteration": 1, "best_objective": 1.5},
            {"iteration": 2, "best_objective": 2.0},
            {"iteration": 3, "best_objective": 2.5},
            {"iteration": 4, "best_objective": 3.0},
        ]

        # Should not converge due to significant improvements
        assert not optimizer._check_convergence()


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_example_parameter_spaces(self):
        """Test example parameter space creation."""
        spaces = create_example_parameter_spaces()

        assert len(spaces) == 5
        assert spaces[0].name == "temperature"
        assert spaces[1].name == "pressure"
        assert spaces[1].log_scale is True
        assert spaces[4].parameter_type == "discrete"

    @pytest.mark.asyncio
    async def test_optimize_molecular_properties(self):
        """Test molecular properties optimization wrapper."""
        parameter_spaces = [
            ParameterSpace("descriptor1", (-1, 1), "continuous"),
            ParameterSpace("descriptor2", (-1, 1), "continuous"),
        ]

        def objective_function(x):
            return -(x[0] ** 2 + x[1] ** 2)

        config = BayesianOptimizationConfig(
            n_initial_points=3, max_iterations=2, use_gpu=False, device="cpu"
        )

        result = optimize_molecular_properties(
            parameter_spaces, objective_function, config
        )

        assert isinstance(result, OptimizationResult)
        assert len(result.parameter_history) == 5  # 3 initial + 2 iterations

    @pytest.mark.asyncio
    async def test_optimize_process_conditions(self):
        """Test process conditions optimization wrapper."""

        def objective_function(x):
            temp, pressure, flow_rate = x
            # Simple objective favoring moderate conditions
            return (
                -(
                    abs(temp - 350) ** 2
                    + abs(pressure - 2e5) ** 2
                    + abs(flow_rate - 5) ** 2
                )
                / 1e8
            )

        config = BayesianOptimizationConfig(
            n_initial_points=4, max_iterations=2, use_gpu=False, device="cpu"
        )

        result = optimize_process_conditions(
            temperature_range=(300, 400),
            pressure_range=(1e5, 3e5),
            flow_rate_range=(1, 10),
            objective_function=objective_function,
            config=config,
        )

        assert isinstance(result, OptimizationResult)
        assert "temperature" in result.best_parameters
        assert "pressure" in result.best_parameters
        assert "flow_rate" in result.best_parameters

    def test_branin_function(self):
        """Test Branin test function."""
        # Test at known optimum locations
        optima = [
            np.array([-np.pi, 12.275]),
            np.array([np.pi, 2.275]),
            np.array([9.42478, 2.475]),
        ]

        for optimum in optima:
            value = branin_function(optimum)
            assert value > -1.0  # Should be close to global minimum (~-0.4)
            assert value < 0.5

    def test_synthetic_chemical_objective(self):
        """Test synthetic multi-objective chemical function."""
        # Test with reasonable chemical conditions
        x = np.array([350, 2e5, 1.0])  # temp, pressure, catalyst
        objectives = synthetic_chemical_objective(x)

        assert len(objectives) == 3
        assert np.all(np.isfinite(objectives))

        # Test that all objectives are reasonable
        assert -1.0 < objectives[0] < 1.0  # yield
        assert -1.0 < objectives[1] < 1.0  # selectivity
        assert objectives[2] < 0  # cost (negative for maximization)


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_surrogate_model(self):
        """Test error handling for invalid surrogate model."""
        spaces = [ParameterSpace("x", (-1, 1), "continuous")]
        config = BayesianOptimizationConfig(surrogate_model=SurrogateModelType.ENSEMBLE)

        with pytest.raises(BayesianOptimizationError):
            BayesianOptimizer(spaces, config)

    def test_invalid_acquisition_function(self):
        """Test error handling for invalid acquisition function."""
        spaces = [ParameterSpace("x", (-1, 1), "continuous")]
        config = BayesianOptimizationConfig(
            acquisition_function=AcquisitionFunction.ENTROPY_SEARCH,
            use_gpu=False,
            device="cpu",
        )
        optimizer = BayesianOptimizer(spaces, config)

        with pytest.raises(BayesianOptimizationError):
            optimizer._create_acquisition_function()

    def test_missing_dependencies(self):
        """Test graceful handling of missing dependencies."""
        spaces = [ParameterSpace("x", (-1, 1), "continuous")]

        # Test when PyTorch is not available
        if not TORCH_AVAILABLE:
            config = BayesianOptimizationConfig(use_gpu=True)
            with pytest.raises(BayesianOptimizationError):
                optimizer = BayesianOptimizer(spaces, config)

        # Test when JAX is not available
        if not JAX_AVAILABLE and not TORCH_AVAILABLE:
            config = BayesianOptimizationConfig(use_gpu=False)
            with pytest.raises(BayesianOptimizationError):
                optimizer = BayesianOptimizer(spaces, config)


class TestPerformanceAndScaling:
    """Test performance and scaling characteristics."""

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_gpu_vs_cpu_performance(self):
        """Test GPU vs CPU performance comparison."""
        X_train = np.random.uniform(-1, 1, (50, 3))
        y_train = np.random.randn(50, 1)

        # CPU version
        config_cpu = BayesianOptimizationConfig(
            use_gpu=False, device="cpu", model_training_iterations=10
        )
        model_cpu = GPyTorchGaussianProcess(config_cpu)

        start_time = time.time()
        model_cpu.fit(X_train, y_train)
        cpu_time = time.time() - start_time

        # GPU version (if available)
        if TORCH_AVAILABLE and torch.cuda.is_available():
            config_gpu = BayesianOptimizationConfig(
                use_gpu=True, device="cuda", model_training_iterations=10
            )
            model_gpu = GPyTorchGaussianProcess(config_gpu)

            start_time = time.time()
            model_gpu.fit(X_train, y_train)
            gpu_time = time.time() - start_time

            print(f"CPU time: {cpu_time:.4f}s, GPU time: {gpu_time:.4f}s")

        assert cpu_time > 0  # Basic sanity check

    def test_high_dimensional_optimization(self):
        """Test optimization in high-dimensional spaces."""
        # Create 10D parameter space
        parameter_spaces = [
            ParameterSpace(f"x{i}", (-2, 2), "continuous") for i in range(10)
        ]

        config = BayesianOptimizationConfig(
            n_initial_points=20,
            max_iterations=5,
            model_training_iterations=20,
            use_gpu=False,
            device="cpu",
        )

        optimizer = BayesianOptimizer(parameter_spaces, config)

        # High-dimensional quadratic function
        def objective_function(x):
            return -np.sum(x**2)

        # Test that it can handle high-dimensional spaces
        initial_points = optimizer._generate_initial_points()
        assert initial_points.shape == (20, 10)

        # Test acquisition function creation
        optimizer.X_evaluated = [initial_points[0]]
        optimizer.y_evaluated = [objective_function(initial_points[0])]
        optimizer.best_y = optimizer.y_evaluated[0]

        X_array = np.array(optimizer.X_evaluated)
        y_array = np.array(optimizer.y_evaluated)

        # Should be able to fit model in high dimensions
        if JAX_AVAILABLE:
            jax_model = JAXGaussianProcess(config)
            jax_model.fit(X_array, y_array)

            # Test prediction
            X_test = np.random.uniform(-2, 2, (5, 10))
            mean, variance = jax_model.predict(X_test)
            assert len(mean) == 5
            assert len(variance) == 5


if __name__ == "__main__":
    # Run specific test categories

    print("Testing Parameter Space...")
    test_param = TestParameterSpace()
    test_param.test_continuous_parameter_space()
    test_param.test_log_scale_parameter_space()
    test_param.test_discrete_parameter_space()
    test_param.test_categorical_parameter_space()
    print("✓ Parameter Space tests passed")

    print("\nTesting Acquisition Functions...")
    test_acq = TestAcquisitionFunctions()
    test_acq.test_expected_improvement()
    test_acq.test_probability_of_improvement()
    test_acq.test_upper_confidence_bound()
    print("✓ Acquisition Function tests passed")

    print("\nTesting Utility Functions...")
    test_util = TestUtilityFunctions()
    test_util.test_create_example_parameter_spaces()
    test_util.test_branin_function()
    test_util.test_synthetic_chemical_objective()
    print("✓ Utility Function tests passed")

    print("\nAll manual tests completed successfully!")
    print("Run 'pytest test_bayesian_optimization.py -v' for full test suite")
