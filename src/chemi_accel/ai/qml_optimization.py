"""
QML Property-Based Molecular Optimization Algorithms for ChemiAccel.

This module provides sophisticated optimization algorithms specifically designed
for QML-guided molecular design, including gradient-based optimization,
evolutionary algorithms, and hybrid approaches with constraint handling.
"""

import threading
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import optax
    from jax import jit, vmap
    from jax import random as jax_random

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..core.exceptions import ChemiAccelError
from ..qml.base import QuantumProperty
from .qml_gradients import QMLGradientComputer
from .qml_inverse_design import QMLOptimizationResult


class OptimizationError(ChemiAccelError):
    """Raised when optimization algorithms fail."""

    pass


class OptimizationMethod(Enum):
    """Available optimization methods."""

    GRADIENT_DESCENT = "gradient_descent"
    ADAM = "adam"
    LBFGS = "lbfgs"
    GENETIC_ALGORITHM = "genetic_algorithm"
    PARTICLE_SWARM = "particle_swarm"
    SIMULATED_ANNEALING = "simulated_annealing"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    EVOLUTIONARY_STRATEGY = "evolutionary_strategy"
    HYBRID = "hybrid"


@dataclass
class OptimizationParams:
    """Parameters for optimization algorithms."""

    method: OptimizationMethod = OptimizationMethod.ADAM
    max_iterations: int = 1000
    learning_rate: float = 0.01
    tolerance: float = 1e-6
    population_size: int = 50
    elite_ratio: float = 0.2
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    temperature_init: float = 100.0
    temperature_decay: float = 0.95
    patience: int = 50
    adaptive_lr: bool = True
    gradient_clipping: float = 1.0
    regularization: float = 0.001
    constraint_penalty: float = 10.0


@dataclass
class OptimizationState:
    """Current state of optimization process."""

    iteration: int = 0
    best_loss: float = float("inf")
    best_solution: Optional[jnp.ndarray] = None
    loss_history: List[float] = field(default_factory=list)
    gradient_norms: List[float] = field(default_factory=list)
    constraint_violations: List[float] = field(default_factory=list)
    population: Optional[List[jnp.ndarray]] = None
    convergence_criteria: Dict[str, bool] = field(default_factory=dict)
    elapsed_time: float = 0.0


class BaseOptimizer(ABC):
    """Abstract base class for molecular optimization algorithms."""

    def __init__(
        self,
        params: OptimizationParams,
        gradient_computer: Optional[QMLGradientComputer] = None,
    ):
        """
        Initialize optimizer.

        Args:
            params: Optimization parameters
            gradient_computer: Optional gradient computer for gradient-based methods
        """
        self.params = params
        self.gradient_computer = gradient_computer
        self.state = OptimizationState()

        # Callbacks
        self.iteration_callbacks = []
        self.convergence_callbacks = []

        # Thread safety
        self._lock = threading.Lock()

    @abstractmethod
    def optimize(
        self,
        initial_solution: jnp.ndarray,
        objective_function: Callable[[jnp.ndarray], float],
        constraint_function: Optional[Callable[[jnp.ndarray], float]] = None,
    ) -> Tuple[jnp.ndarray, OptimizationState]:
        """
        Optimize the objective function.

        Args:
            initial_solution: Starting point for optimization
            objective_function: Function to minimize
            constraint_function: Optional constraint function

        Returns:
            Tuple of (optimal_solution, optimization_state)
        """
        pass

    def register_callback(self, callback_type: str, callback: Callable):
        """Register optimization callback."""
        if callback_type == "iteration":
            self.iteration_callbacks.append(callback)
        elif callback_type == "convergence":
            self.convergence_callbacks.append(callback)

    def _check_convergence(self, current_loss: float) -> bool:
        """Check if optimization has converged."""
        if len(self.state.loss_history) < 2:
            return False

        # Loss improvement convergence
        loss_improvement = abs(self.state.loss_history[-2] - current_loss)
        loss_converged = loss_improvement < self.params.tolerance

        # Gradient norm convergence (if available)
        grad_converged = True
        if self.state.gradient_norms:
            grad_converged = self.state.gradient_norms[-1] < self.params.tolerance

        # Patience-based convergence
        patience_converged = False
        if len(self.state.loss_history) >= self.params.patience:
            recent_losses = self.state.loss_history[-self.params.patience :]
            best_recent = min(recent_losses)
            patience_converged = abs(current_loss - best_recent) < self.params.tolerance

        self.state.convergence_criteria = {
            "loss_converged": loss_converged,
            "gradient_converged": grad_converged,
            "patience_converged": patience_converged,
        }

        return loss_converged and grad_converged

    def _call_iteration_callbacks(self):
        """Call iteration callbacks."""
        for callback in self.iteration_callbacks:
            try:
                callback(self.state)
            except Exception as e:
                warnings.warn(f"Iteration callback failed: {e}")

    def _call_convergence_callbacks(self):
        """Call convergence callbacks."""
        for callback in self.convergence_callbacks:
            try:
                callback(self.state)
            except Exception as e:
                warnings.warn(f"Convergence callback failed: {e}")


class GradientBasedOptimizer(BaseOptimizer):
    """Gradient-based optimization algorithms."""

    def __init__(
        self, params: OptimizationParams, gradient_computer: QMLGradientComputer
    ):
        """Initialize gradient-based optimizer."""
        super().__init__(params, gradient_computer)

        if not JAX_AVAILABLE:
            raise OptimizationError("JAX required for gradient-based optimization")

        # Initialize JAX optimizer
        self._setup_jax_optimizer()

    def _setup_jax_optimizer(self):
        """Setup JAX optimizer based on method."""
        if self.params.method == OptimizationMethod.ADAM:
            self.jax_optimizer = optax.adam(
                learning_rate=self.params.learning_rate, b1=0.9, b2=0.999
            )
        elif self.params.method == OptimizationMethod.GRADIENT_DESCENT:
            self.jax_optimizer = optax.sgd(learning_rate=self.params.learning_rate)
        elif self.params.method == OptimizationMethod.LBFGS:
            # L-BFGS implementation using optax
            self.jax_optimizer = optax.lbfgs()
        else:
            self.jax_optimizer = optax.adam(learning_rate=self.params.learning_rate)

    def optimize(
        self,
        initial_solution: jnp.ndarray,
        objective_function: Callable[[jnp.ndarray], float],
        constraint_function: Optional[Callable[[jnp.ndarray], float]] = None,
    ) -> Tuple[jnp.ndarray, OptimizationState]:
        """Optimize using gradient-based methods."""
        start_time = time.time()

        # Initialize optimization state
        current_solution = initial_solution
        opt_state = self.jax_optimizer.init(current_solution)

        # Create combined objective (with constraints)
        def combined_objective(x):
            obj_loss = objective_function(x)
            constraint_loss = 0.0

            if constraint_function is not None:
                constraint_violation = constraint_function(x)
                constraint_loss = (
                    self.params.constraint_penalty
                    * jnp.maximum(0, constraint_violation) ** 2
                )

            # Regularization
            reg_loss = self.params.regularization * jnp.sum(x**2)

            return obj_loss + constraint_loss + reg_loss

        # JIT compile the gradient function
        grad_fn = jit(jax.grad(combined_objective))
        loss_and_grad_fn = jit(jax.value_and_grad(combined_objective))

        # Optimization loop
        for iteration in range(self.params.max_iterations):
            self.state.iteration = iteration

            # Compute loss and gradients
            current_loss, gradients = loss_and_grad_fn(current_solution)
            gradient_norm = float(jnp.linalg.norm(gradients))

            # Apply gradient clipping
            if self.params.gradient_clipping > 0:
                if gradient_norm > self.params.gradient_clipping:
                    gradients = gradients * (
                        self.params.gradient_clipping / gradient_norm
                    )
                    gradient_norm = self.params.gradient_clipping

            # Update solution
            updates, opt_state = self.jax_optimizer.update(gradients, opt_state)
            current_solution = optax.apply_updates(current_solution, updates)

            # Update state
            self.state.loss_history.append(float(current_loss))
            self.state.gradient_norms.append(gradient_norm)

            if constraint_function is not None:
                constraint_violation = float(constraint_function(current_solution))
                self.state.constraint_violations.append(constraint_violation)

            # Update best solution
            if current_loss < self.state.best_loss:
                self.state.best_loss = float(current_loss)
                self.state.best_solution = current_solution.copy()

            # Call iteration callbacks
            self._call_iteration_callbacks()

            # Check convergence
            if self._check_convergence(float(current_loss)):
                self._call_convergence_callbacks()
                break

            # Adaptive learning rate
            if self.params.adaptive_lr and iteration % 50 == 0 and iteration > 0:
                if len(self.state.loss_history) >= 2:
                    recent_improvement = (
                        self.state.loss_history[-50] - self.state.loss_history[-1]
                    ) / max(abs(self.state.loss_history[-50]), 1e-10)

                    if recent_improvement < 0.01:  # Less than 1% improvement
                        # Reduce learning rate
                        new_lr = self.params.learning_rate * 0.8
                        self.params.learning_rate = max(new_lr, 1e-8)
                        self._setup_jax_optimizer()
                        opt_state = self.jax_optimizer.init(current_solution)

        self.state.elapsed_time = time.time() - start_time
        return self.state.best_solution or current_solution, self.state


class EvolutionaryOptimizer(BaseOptimizer):
    """Evolutionary optimization algorithms."""

    def __init__(self, params: OptimizationParams):
        """Initialize evolutionary optimizer."""
        super().__init__(params)
        self.random_key = jax_random.PRNGKey(42) if JAX_AVAILABLE else None

    def optimize(
        self,
        initial_solution: jnp.ndarray,
        objective_function: Callable[[jnp.ndarray], float],
        constraint_function: Optional[Callable[[jnp.ndarray], float]] = None,
    ) -> Tuple[jnp.ndarray, OptimizationState]:
        """Optimize using evolutionary algorithms."""
        start_time = time.time()

        if self.params.method == OptimizationMethod.GENETIC_ALGORITHM:
            result = self._genetic_algorithm(
                initial_solution, objective_function, constraint_function
            )
        elif self.params.method == OptimizationMethod.PARTICLE_SWARM:
            result = self._particle_swarm_optimization(
                initial_solution, objective_function, constraint_function
            )
        elif self.params.method == OptimizationMethod.EVOLUTIONARY_STRATEGY:
            result = self._evolutionary_strategy(
                initial_solution, objective_function, constraint_function
            )
        else:
            raise OptimizationError(
                f"Unknown evolutionary method: {self.params.method}"
            )

        self.state.elapsed_time = time.time() - start_time
        return result, self.state

    def _genetic_algorithm(
        self,
        initial_solution: jnp.ndarray,
        objective_function: Callable[[jnp.ndarray], float],
        constraint_function: Optional[Callable[[jnp.ndarray], float]],
    ) -> jnp.ndarray:
        """Genetic algorithm implementation."""
        # Initialize population
        population = self._initialize_population(initial_solution)

        for generation in range(self.params.max_iterations):
            self.state.iteration = generation

            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                fitness = self._evaluate_fitness(
                    individual, objective_function, constraint_function
                )
                fitness_scores.append(fitness)

            fitness_scores = np.array(fitness_scores)

            # Update best solution
            best_idx = np.argmin(fitness_scores)
            best_fitness = fitness_scores[best_idx]

            if best_fitness < self.state.best_loss:
                self.state.best_loss = best_fitness
                self.state.best_solution = population[best_idx].copy()

            self.state.loss_history.append(best_fitness)

            # Selection
            elite_count = int(self.params.elite_ratio * self.params.population_size)
            elite_indices = np.argsort(fitness_scores)[:elite_count]
            new_population = [population[i] for i in elite_indices]

            # Crossover and mutation
            while len(new_population) < self.params.population_size:
                # Tournament selection for parents
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)

                # Crossover
                if np.random.random() < self.params.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                # Mutation
                if np.random.random() < self.params.mutation_rate:
                    child1 = self._mutate(child1)
                if np.random.random() < self.params.mutation_rate:
                    child2 = self._mutate(child2)

                new_population.extend([child1, child2])

            population = new_population[: self.params.population_size]
            self.state.population = population

            # Call iteration callbacks
            self._call_iteration_callbacks()

            # Check convergence
            if self._check_convergence(best_fitness):
                self._call_convergence_callbacks()
                break

        return self.state.best_solution

    def _particle_swarm_optimization(
        self,
        initial_solution: jnp.ndarray,
        objective_function: Callable[[jnp.ndarray], float],
        constraint_function: Optional[Callable[[jnp.ndarray], float]],
    ) -> jnp.ndarray:
        """Particle Swarm Optimization implementation."""
        # Initialize particles
        particles = self._initialize_population(initial_solution)
        velocities = [
            np.random.randn(*initial_solution.shape) * 0.1
            for _ in range(self.params.population_size)
        ]
        personal_best = particles.copy()
        personal_best_fitness = [float("inf")] * self.params.population_size

        # PSO parameters
        w = 0.7  # Inertia weight
        c1 = 1.5  # Cognitive coefficient
        c2 = 1.5  # Social coefficient

        for iteration in range(self.params.max_iterations):
            self.state.iteration = iteration

            # Evaluate particles
            for i, particle in enumerate(particles):
                fitness = self._evaluate_fitness(
                    particle, objective_function, constraint_function
                )

                # Update personal best
                if fitness < personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness
                    personal_best[i] = particle.copy()

                # Update global best
                if fitness < self.state.best_loss:
                    self.state.best_loss = fitness
                    self.state.best_solution = particle.copy()

            self.state.loss_history.append(self.state.best_loss)

            # Update velocities and positions
            for i in range(self.params.population_size):
                r1, r2 = np.random.random(), np.random.random()

                # Update velocity
                velocities[i] = (
                    w * velocities[i]
                    + c1 * r1 * (personal_best[i] - particles[i])
                    + c2 * r2 * (self.state.best_solution - particles[i])
                )

                # Update position
                particles[i] = particles[i] + velocities[i]

                # Apply bounds
                particles[i] = np.clip(particles[i], -5.0, 5.0)

            self.state.population = particles

            # Call iteration callbacks
            self._call_iteration_callbacks()

            # Check convergence
            if self._check_convergence(self.state.best_loss):
                self._call_convergence_callbacks()
                break

        return self.state.best_solution

    def _evolutionary_strategy(
        self,
        initial_solution: jnp.ndarray,
        objective_function: Callable[[jnp.ndarray], float],
        constraint_function: Optional[Callable[[jnp.ndarray], float]],
    ) -> jnp.ndarray:
        """Evolution Strategy (ES) implementation."""
        # Initialize population
        population = self._initialize_population(initial_solution)
        sigma = 0.1  # Step size

        for generation in range(self.params.max_iterations):
            self.state.iteration = generation

            # Generate offspring
            offspring = []
            for parent in population:
                for _ in range(2):  # 2 offspring per parent
                    noise = np.random.normal(0, sigma, parent.shape)
                    child = parent + noise
                    offspring.append(child)

            # Evaluate fitness
            all_individuals = population + offspring
            fitness_scores = []

            for individual in all_individuals:
                fitness = self._evaluate_fitness(
                    individual, objective_function, constraint_function
                )
                fitness_scores.append(fitness)

            fitness_scores = np.array(fitness_scores)

            # Selection (keep best mu individuals)
            mu = self.params.population_size
            best_indices = np.argsort(fitness_scores)[:mu]
            population = [all_individuals[i] for i in best_indices]

            # Update best solution
            best_fitness = fitness_scores[best_indices[0]]
            if best_fitness < self.state.best_loss:
                self.state.best_loss = best_fitness
                self.state.best_solution = all_individuals[best_indices[0]].copy()

            self.state.loss_history.append(best_fitness)
            self.state.population = population

            # Adapt step size (1/5 success rule)
            if generation > 0 and generation % 10 == 0:
                recent_improvements = sum(
                    1
                    for i in range(
                        max(0, len(self.state.loss_history) - 10),
                        len(self.state.loss_history) - 1,
                    )
                    if self.state.loss_history[i] > self.state.loss_history[i + 1]
                )
                success_rate = recent_improvements / 10

                if success_rate > 1 / 5:
                    sigma *= 1.2
                elif success_rate < 1 / 5:
                    sigma *= 0.8

            # Call iteration callbacks
            self._call_iteration_callbacks()

            # Check convergence
            if self._check_convergence(best_fitness):
                self._call_convergence_callbacks()
                break

        return self.state.best_solution

    def _initialize_population(
        self, initial_solution: jnp.ndarray
    ) -> List[jnp.ndarray]:
        """Initialize population around initial solution."""
        population = []

        # Add initial solution
        population.append(initial_solution)

        # Generate random variations
        for _ in range(self.params.population_size - 1):
            variation = initial_solution + np.random.normal(
                0, 0.5, initial_solution.shape
            )
            population.append(jnp.array(variation))

        return population

    def _evaluate_fitness(
        self,
        individual: jnp.ndarray,
        objective_function: Callable[[jnp.ndarray], float],
        constraint_function: Optional[Callable[[jnp.ndarray], float]],
    ) -> float:
        """Evaluate fitness of an individual."""
        try:
            obj_loss = objective_function(individual)

            # Add constraint penalty
            if constraint_function is not None:
                constraint_violation = constraint_function(individual)
                constraint_penalty = (
                    self.params.constraint_penalty * max(0, constraint_violation) ** 2
                )
                obj_loss += constraint_penalty

            return float(obj_loss)
        except Exception:
            # Return high penalty for invalid individuals
            return 1e10

    def _tournament_selection(
        self, population: List[jnp.ndarray], fitness_scores: np.ndarray
    ) -> jnp.ndarray:
        """Tournament selection for genetic algorithm."""
        tournament_size = min(3, len(population))
        tournament_indices = np.random.choice(
            len(population), tournament_size, replace=False
        )
        tournament_fitness = fitness_scores[tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return population[winner_idx]

    def _crossover(
        self, parent1: jnp.ndarray, parent2: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Crossover operation for genetic algorithm."""
        # Uniform crossover
        mask = np.random.random(parent1.shape) < 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return jnp.array(child1), jnp.array(child2)

    def _mutate(self, individual: jnp.ndarray) -> jnp.ndarray:
        """Mutation operation for genetic algorithm."""
        mutation_strength = 0.1
        noise = np.random.normal(0, mutation_strength, individual.shape)
        return individual + noise


class SimulatedAnnealingOptimizer(BaseOptimizer):
    """Simulated Annealing optimization algorithm."""

    def optimize(
        self,
        initial_solution: jnp.ndarray,
        objective_function: Callable[[jnp.ndarray], float],
        constraint_function: Optional[Callable[[jnp.ndarray], float]] = None,
    ) -> Tuple[jnp.ndarray, OptimizationState]:
        """Optimize using simulated annealing."""
        start_time = time.time()

        current_solution = initial_solution
        current_loss = self._evaluate_with_constraints(
            current_solution, objective_function, constraint_function
        )

        self.state.best_loss = current_loss
        self.state.best_solution = current_solution.copy()

        temperature = self.params.temperature_init

        for iteration in range(self.params.max_iterations):
            self.state.iteration = iteration

            # Generate neighbor solution
            step_size = temperature / self.params.temperature_init * 0.5
            neighbor = current_solution + np.random.normal(
                0, step_size, current_solution.shape
            )
            neighbor_loss = self._evaluate_with_constraints(
                neighbor, objective_function, constraint_function
            )

            # Accept or reject
            delta = neighbor_loss - current_loss
            if delta < 0 or np.random.random() < np.exp(-delta / temperature):
                current_solution = neighbor
                current_loss = neighbor_loss

                # Update best solution
                if current_loss < self.state.best_loss:
                    self.state.best_loss = current_loss
                    self.state.best_solution = current_solution.copy()

            # Cool down
            temperature *= self.params.temperature_decay

            # Update state
            self.state.loss_history.append(current_loss)

            # Call iteration callbacks
            self._call_iteration_callbacks()

            # Check convergence
            if self._check_convergence(current_loss):
                self._call_convergence_callbacks()
                break

        self.state.elapsed_time = time.time() - start_time
        return self.state.best_solution, self.state

    def _evaluate_with_constraints(
        self,
        solution: jnp.ndarray,
        objective_function: Callable[[jnp.ndarray], float],
        constraint_function: Optional[Callable[[jnp.ndarray], float]],
    ) -> float:
        """Evaluate solution with constraints."""
        try:
            obj_loss = objective_function(solution)

            if constraint_function is not None:
                constraint_violation = constraint_function(solution)
                constraint_penalty = (
                    self.params.constraint_penalty * max(0, constraint_violation) ** 2
                )
                obj_loss += constraint_penalty

            return float(obj_loss)
        except Exception:
            return 1e10  # High penalty for invalid solutions


class HybridOptimizer(BaseOptimizer):
    """Hybrid optimization combining multiple methods."""

    def __init__(
        self,
        params: OptimizationParams,
        gradient_computer: Optional[QMLGradientComputer] = None,
    ):
        """Initialize hybrid optimizer."""
        super().__init__(params, gradient_computer)

        # Initialize sub-optimizers
        self.gradient_optimizer = None
        if gradient_computer:
            grad_params = OptimizationParams(
                method=OptimizationMethod.ADAM,
                max_iterations=params.max_iterations // 2,
                learning_rate=params.learning_rate,
            )
            self.gradient_optimizer = GradientBasedOptimizer(
                grad_params, gradient_computer
            )

        evo_params = OptimizationParams(
            method=OptimizationMethod.GENETIC_ALGORITHM,
            max_iterations=params.max_iterations // 4,
            population_size=params.population_size,
        )
        self.evolutionary_optimizer = EvolutionaryOptimizer(evo_params)

        sa_params = OptimizationParams(
            method=OptimizationMethod.SIMULATED_ANNEALING,
            max_iterations=params.max_iterations // 4,
            temperature_init=params.temperature_init,
        )
        self.sa_optimizer = SimulatedAnnealingOptimizer(sa_params)

    def optimize(
        self,
        initial_solution: jnp.ndarray,
        objective_function: Callable[[jnp.ndarray], float],
        constraint_function: Optional[Callable[[jnp.ndarray], float]] = None,
    ) -> Tuple[jnp.ndarray, OptimizationState]:
        """Optimize using hybrid approach."""
        start_time = time.time()

        current_best = initial_solution
        current_best_loss = float("inf")

        # Phase 1: Evolutionary exploration
        print("Phase 1: Evolutionary exploration...")
        evo_solution, evo_state = self.evolutionary_optimizer.optimize(
            initial_solution, objective_function, constraint_function
        )

        if evo_state.best_loss < current_best_loss:
            current_best = evo_solution
            current_best_loss = evo_state.best_loss

        # Phase 2: Simulated annealing refinement
        print("Phase 2: Simulated annealing refinement...")
        sa_solution, sa_state = self.sa_optimizer.optimize(
            current_best, objective_function, constraint_function
        )

        if sa_state.best_loss < current_best_loss:
            current_best = sa_solution
            current_best_loss = sa_state.best_loss

        # Phase 3: Gradient-based fine-tuning (if available)
        if self.gradient_optimizer:
            print("Phase 3: Gradient-based fine-tuning...")
            grad_solution, grad_state = self.gradient_optimizer.optimize(
                current_best, objective_function, constraint_function
            )

            if grad_state.best_loss < current_best_loss:
                current_best = grad_solution
                current_best_loss = grad_state.best_loss

        # Combine states
        self.state.best_solution = current_best
        self.state.best_loss = current_best_loss
        self.state.elapsed_time = time.time() - start_time

        # Combine loss histories
        all_losses = []
        all_losses.extend(evo_state.loss_history)
        all_losses.extend(sa_state.loss_history)
        if self.gradient_optimizer and hasattr(grad_state, "loss_history"):
            all_losses.extend(grad_state.loss_history)

        self.state.loss_history = all_losses

        return current_best, self.state


class QMLMolecularOptimizer:
    """High-level interface for QML-guided molecular optimization."""

    def __init__(
        self,
        qml_model,
        gradient_computer: QMLGradientComputer,
        optimization_params: OptimizationParams,
    ):
        """
        Initialize QML molecular optimizer.

        Args:
            qml_model: QML model for property prediction
            gradient_computer: Gradient computation system
            optimization_params: Optimization parameters
        """
        self.qml_model = qml_model
        self.gradient_computer = gradient_computer
        self.params = optimization_params

        # Create optimizer based on method
        if optimization_params.method in [
            OptimizationMethod.ADAM,
            OptimizationMethod.GRADIENT_DESCENT,
            OptimizationMethod.LBFGS,
        ]:
            self.optimizer = GradientBasedOptimizer(
                optimization_params, gradient_computer
            )
        elif optimization_params.method in [
            OptimizationMethod.GENETIC_ALGORITHM,
            OptimizationMethod.PARTICLE_SWARM,
            OptimizationMethod.EVOLUTIONARY_STRATEGY,
        ]:
            self.optimizer = EvolutionaryOptimizer(optimization_params)
        elif optimization_params.method == OptimizationMethod.SIMULATED_ANNEALING:
            self.optimizer = SimulatedAnnealingOptimizer(optimization_params)
        elif optimization_params.method == OptimizationMethod.HYBRID:
            self.optimizer = HybridOptimizer(optimization_params, gradient_computer)
        else:
            raise OptimizationError(
                f"Unknown optimization method: {optimization_params.method}"
            )

    def optimize_towards_targets(
        self,
        initial_molecules: List[str],
        target_properties: Dict[QuantumProperty, float],
        molecular_representation,
        constraint_function: Optional[Callable] = None,
    ) -> QMLOptimizationResult:
        """
        Optimize molecules towards target QML properties.

        Args:
            initial_molecules: Starting molecules
            target_properties: Target property values
            molecular_representation: Molecular representation system
            constraint_function: Optional constraint function

        Returns:
            QMLOptimizationResult with optimization details
        """
        # Encode initial molecules
        initial_representation = molecular_representation.encode(initial_molecules)

        # Create objective function
        def objective_function(representation):
            # Decode to molecules
            molecules = molecular_representation.decode(representation)

            # Predict properties
            properties = list(target_properties.keys())
            (
                predictions,
                _,
            ) = self.gradient_computer.qml_property_predictor.predict_properties(
                molecules, properties
            )

            # Compute loss
            total_loss = 0.0
            for prop, target_val in target_properties.items():
                if prop in predictions:
                    pred_vals = predictions[prop]
                    prop_loss = jnp.mean((pred_vals - target_val) ** 2)
                    total_loss += prop_loss

            return total_loss

        # Optimize
        optimal_representation, opt_state = self.optimizer.optimize(
            initial_representation, objective_function, constraint_function
        )

        # Decode final molecules
        final_molecules = molecular_representation.decode(optimal_representation)

        # Get final predictions
        properties = list(target_properties.keys())
        (
            final_predictions,
            final_uncertainties,
        ) = self.gradient_computer.qml_property_predictor.predict_with_uncertainty(
            final_molecules, properties
        )

        # Create result
        result = QMLOptimizationResult(
            optimized_molecules=final_molecules,
            optimization_history=[],  # Would need to be populated from callbacks
            final_properties={
                f"qml_{prop.name.lower()}": vals
                for prop, vals in final_predictions.items()
            },
            constraint_violations={},
            qml_predictions=[
                {prop: final_predictions[prop][i] for prop in final_predictions}
                for i in range(len(final_molecules))
            ],
            qml_uncertainties=[
                {prop: final_uncertainties[prop][i] for prop in final_uncertainties}
                for i in range(len(final_molecules))
            ]
            if final_uncertainties
            else None,
            gradient_norms=opt_state.gradient_norms,
            property_evolution={},
        )

        return result


# Utility functions for optimization
def create_optimization_params(
    method: OptimizationMethod = OptimizationMethod.ADAM,
    max_iterations: int = 1000,
    learning_rate: float = 0.01,
    population_size: int = 50,
) -> OptimizationParams:
    """Create standard optimization parameters."""
    return OptimizationParams(
        method=method,
        max_iterations=max_iterations,
        learning_rate=learning_rate,
        population_size=population_size,
        tolerance=1e-6,
        patience=50,
        adaptive_lr=True,
    )


def create_hybrid_optimization_params(max_iterations: int = 1000) -> OptimizationParams:
    """Create parameters for hybrid optimization."""
    return OptimizationParams(
        method=OptimizationMethod.HYBRID,
        max_iterations=max_iterations,
        learning_rate=0.01,
        population_size=30,
        elite_ratio=0.2,
        mutation_rate=0.1,
        temperature_init=100.0,
        patience=100,
    )
