"""
Closed-Loop Optimization System

This module implements comprehensive closed-loop optimization workflows that integrate
ChemiAccel's AI decision-making capabilities (Active Learning, Inverse Design) with
robotic platforms for autonomous experimental optimization.
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import deque

# Optional ML imports
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy.optimize import minimize, differential_evolution
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from .robotic_control import (
    RobotManager, ExperimentPlan, ExperimentalAction, RobotFeedback,
    OptimizationObjective, OptimizationState, ExperimentStatus
)
from .adaptive_learning import AdaptiveLearningManager, TrainingData
from .data_ingestion import ExperimentalDataPoint


logger = logging.getLogger(__name__)


class ClosedLoopStrategy(Enum):
    """Strategy for closed-loop optimization."""
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    GENETIC_ALGORITHM = "genetic_algorithm"
    ACTIVE_LEARNING = "active_learning"
    MULTI_OBJECTIVE = "multi_objective"
    HYBRID = "hybrid"


class ConvergenceCriteria(Enum):
    """Criteria for optimization convergence."""
    MAX_ITERATIONS = "max_iterations"
    OBJECTIVE_THRESHOLD = "objective_threshold"
    IMPROVEMENT_THRESHOLD = "improvement_threshold"
    TIME_LIMIT = "time_limit"
    UNCERTAINTY_THRESHOLD = "uncertainty_threshold"


@dataclass
class OptimizationConfig:
    """Configuration for closed-loop optimization."""
    strategy: ClosedLoopStrategy
    objective: OptimizationObjective
    parameter_space: Dict[str, Tuple[float, float]]  # parameter: (min, max)
    convergence_criteria: Dict[ConvergenceCriteria, float]
    batch_size: int = 1
    initial_samples: int = 5
    exploration_weight: float = 0.1
    acquisition_function: str = "expected_improvement"
    constraint_functions: List[Callable[[Dict[str, Any]], bool]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Result from a completed experiment."""
    experiment_id: str
    parameters: Dict[str, Any]
    objective_values: Dict[str, float]
    success: bool
    execution_time: float
    robot_feedback: List[RobotFeedback]
    experimental_data: List[ExperimentalDataPoint]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ActiveLearningQuery:
    """Query for active learning system."""
    query_id: str
    parameters: Dict[str, Any]
    uncertainty_score: float
    expected_information_gain: float
    diversity_score: float
    feasibility_score: float
    combined_score: float
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AcquisitionFunction(ABC):
    """Abstract base class for acquisition functions."""
    
    @abstractmethod
    def evaluate(
        self, 
        candidate_points: np.ndarray, 
        model: Any,
        current_best: float,
        exploration_weight: float = 0.1
    ) -> np.ndarray:
        """Evaluate acquisition function at candidate points."""
        pass


class ExpectedImprovement(AcquisitionFunction):
    """Expected Improvement acquisition function."""
    
    def evaluate(
        self, 
        candidate_points: np.ndarray, 
        model: Any,
        current_best: float,
        exploration_weight: float = 0.1
    ) -> np.ndarray:
        """Calculate Expected Improvement."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("Expected Improvement requires scikit-learn")
        
        try:
            mu, sigma = model.predict(candidate_points, return_std=True)
            
            # Avoid division by zero
            sigma = np.maximum(sigma, 1e-9)
            
            # Calculate improvement
            improvement = mu - current_best
            
            # Calculate Z-score
            z = improvement / sigma
            
            # Expected improvement
            from scipy.stats import norm
            ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
            
            return ei
            
        except Exception as e:
            logger.error(f"Error calculating Expected Improvement: {e}")
            return np.zeros(len(candidate_points))


class UpperConfidenceBound(AcquisitionFunction):
    """Upper Confidence Bound acquisition function."""
    
    def evaluate(
        self, 
        candidate_points: np.ndarray, 
        model: Any,
        current_best: float,
        exploration_weight: float = 0.1
    ) -> np.ndarray:
        """Calculate Upper Confidence Bound."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("UCB requires scikit-learn")
        
        try:
            mu, sigma = model.predict(candidate_points, return_std=True)
            
            # UCB = mean + exploration_weight * std
            ucb = mu + exploration_weight * sigma
            
            return ucb
            
        except Exception as e:
            logger.error(f"Error calculating UCB: {e}")
            return np.zeros(len(candidate_points))


class ActiveLearningAcquisition(AcquisitionFunction):
    """Active learning-based acquisition function."""
    
    def __init__(self, adaptive_learning_manager: AdaptiveLearningManager):
        self.adaptive_manager = adaptive_learning_manager
    
    def evaluate(
        self, 
        candidate_points: np.ndarray, 
        model: Any,
        current_best: float,
        exploration_weight: float = 0.1
    ) -> np.ndarray:
        """Calculate active learning-based acquisition."""
        try:
            # Get model predictions and uncertainties
            mu, sigma = model.predict(candidate_points, return_std=True)
            
            # Calculate information gain (uncertainty-based)
            information_gain = sigma
            
            # Calculate diversity score (distance from existing points)
            diversity_scores = self._calculate_diversity(candidate_points)
            
            # Calculate feasibility scores
            feasibility_scores = self._calculate_feasibility(candidate_points)
            
            # Combine scores
            combined_scores = (
                0.4 * information_gain +
                0.3 * diversity_scores +
                0.2 * feasibility_scores +
                0.1 * (mu - current_best)  # Slight bias toward improvement
            )
            
            return combined_scores
            
        except Exception as e:
            logger.error(f"Error in active learning acquisition: {e}")
            return np.zeros(len(candidate_points))
    
    def _calculate_diversity(self, candidate_points: np.ndarray) -> np.ndarray:
        """Calculate diversity scores based on distance from existing data."""
        try:
            # Get training data from adaptive learning manager
            training_data = self.adaptive_manager.get_training_data()
            
            if training_data is None or len(training_data) == 0:
                return np.ones(len(candidate_points))
            
            # Calculate minimum distances to existing points
            distances = []
            for point in candidate_points:
                min_dist = float('inf')
                for existing_point in training_data:
                    dist = np.linalg.norm(point - existing_point)
                    min_dist = min(min_dist, dist)
                distances.append(min_dist)
            
            # Normalize distances
            distances = np.array(distances)
            if np.max(distances) > 0:
                distances = distances / np.max(distances)
            
            return distances
            
        except Exception as e:
            logger.warning(f"Error calculating diversity scores: {e}")
            return np.ones(len(candidate_points))
    
    def _calculate_feasibility(self, candidate_points: np.ndarray) -> np.ndarray:
        """Calculate feasibility scores for candidate points."""
        # Simple feasibility based on parameter bounds
        # In practice, this would include more sophisticated constraints
        return np.ones(len(candidate_points))


class ClosedLoopOptimizer:
    """
    Main closed-loop optimization system that coordinates AI decision-making
    with robotic experiment execution.
    """
    
    def __init__(
        self,
        robot_manager: RobotManager,
        adaptive_learning_manager: AdaptiveLearningManager,
        config: OptimizationConfig
    ):
        self.robot_manager = robot_manager
        self.adaptive_manager = adaptive_learning_manager
        self.config = config
        
        # Optimization state
        self.optimization_state = OptimizationState(
            iteration=0,
            current_parameters={},
            objective_values=[],
            best_parameters={},
            best_objective=float('-inf') if self._is_maximization_objective() else float('inf'),
            convergence_criteria=config.convergence_criteria,
            total_experiments=0,
            successful_experiments=0
        )
        
        # Experiment tracking
        self.experiment_results: List[ExperimentResult] = []
        self.active_experiments: Dict[str, ExperimentPlan] = {}
        self.experiment_queue: deque = deque()
        
        # Models and acquisition functions
        self.surrogate_model: Optional[Any] = None
        self.acquisition_functions = {
            "expected_improvement": ExpectedImprovement(),
            "upper_confidence_bound": UpperConfidenceBound(),
            "active_learning": ActiveLearningAcquisition(adaptive_learning_manager)
        }
        
        # Threading and async support
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._optimization_task: Optional[asyncio.Task] = None
        self._is_running = False
        self._lock = threading.Lock()
        
        # Callbacks
        self.experiment_callbacks: List[Callable[[ExperimentResult], None]] = []
        self.optimization_callbacks: List[Callable[[OptimizationState], None]] = []
        
        # Setup robot feedback handling
        self.robot_manager.add_feedback_callback(self._handle_robot_feedback)
    
    def _is_maximization_objective(self) -> bool:
        """Check if objective should be maximized."""
        maximization_objectives = {
            OptimizationObjective.MAXIMIZE_YIELD,
            OptimizationObjective.MAXIMIZE_PURITY
        }
        return self.config.objective in maximization_objectives
    
    async def start_optimization(self) -> None:
        """Start the closed-loop optimization process."""
        if self._is_running:
            logger.warning("Optimization already running")
            return
        
        self._is_running = True
        logger.info(f"Starting closed-loop optimization with strategy: {self.config.strategy}")
        
        # Initialize with random samples if no prior data
        if len(self.experiment_results) < self.config.initial_samples:
            await self._generate_initial_samples()
        
        # Start main optimization loop
        self._optimization_task = asyncio.create_task(self._optimization_loop())
    
    async def stop_optimization(self) -> None:
        """Stop the optimization process."""
        self._is_running = False
        
        if self._optimization_task:
            self._optimization_task.cancel()
            try:
                await self._optimization_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped closed-loop optimization")
    
    async def _optimization_loop(self) -> None:
        """Main optimization loop."""
        while self._is_running and not self._check_convergence():
            try:
                # Update surrogate model with latest results
                await self._update_surrogate_model()
                
                # Generate new experimental candidates
                candidates = await self._generate_candidates()
                
                if candidates:
                    # Create and execute experiment plans
                    plans = await self._create_experiment_plans(candidates)
                    
                    for plan in plans:
                        await self.robot_manager.execute_experiment_plan(plan)
                        self.active_experiments[plan.plan_id] = plan
                    
                    # Wait for experiments to complete
                    await self._wait_for_experiment_completion()
                
                self.optimization_state.iteration += 1
                
                # Notify optimization callbacks
                for callback in self.optimization_callbacks:
                    try:
                        callback(self.optimization_state)
                    except Exception as e:
                        logger.error(f"Error in optimization callback: {e}")
                
                # Brief pause before next iteration
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(5.0)  # Wait longer on error
    
    async def _generate_initial_samples(self) -> None:
        """Generate initial random samples for optimization."""
        logger.info(f"Generating {self.config.initial_samples} initial samples")
        
        initial_candidates = []
        for _ in range(self.config.initial_samples):
            candidate = {}
            for param_name, (min_val, max_val) in self.config.parameter_space.items():
                candidate[param_name] = np.random.uniform(min_val, max_val)
            initial_candidates.append(candidate)
        
        # Create and execute experiment plans
        plans = await self._create_experiment_plans(initial_candidates)
        for plan in plans:
            await self.robot_manager.execute_experiment_plan(plan)
            self.active_experiments[plan.plan_id] = plan
        
        # Wait for completion
        await self._wait_for_experiment_completion()
    
    async def _update_surrogate_model(self) -> None:
        """Update the surrogate model with latest experimental results."""
        if not SKLEARN_AVAILABLE or len(self.experiment_results) < 2:
            return
        
        try:
            # Prepare training data
            X = []
            y = []
            
            for result in self.experiment_results:
                if result.success:
                    # Convert parameters to feature vector
                    features = []
                    for param_name in sorted(self.config.parameter_space.keys()):
                        features.append(result.parameters.get(param_name, 0.0))
                    X.append(features)
                    
                    # Get objective value
                    objective_value = self._extract_objective_value(result)
                    y.append(objective_value)
            
            if len(X) >= 2:
                X = np.array(X)
                y = np.array(y)
                
                # Create and train Gaussian Process model
                kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
                self.surrogate_model = GaussianProcessRegressor(
                    kernel=kernel,
                    n_restarts_optimizer=10,
                    alpha=1e-6,
                    normalize_y=True
                )
                
                self.surrogate_model.fit(X, y)
                logger.debug(f"Updated surrogate model with {len(X)} data points")
                
        except Exception as e:
            logger.error(f"Error updating surrogate model: {e}")
    
    def _extract_objective_value(self, result: ExperimentResult) -> float:
        """Extract objective value from experiment result."""
        if self.config.objective == OptimizationObjective.MAXIMIZE_YIELD:
            return result.objective_values.get('yield', 0.0)
        elif self.config.objective == OptimizationObjective.MINIMIZE_TIME:
            return -result.execution_time  # Minimize time = maximize negative time
        elif self.config.objective == OptimizationObjective.MAXIMIZE_PURITY:
            return result.objective_values.get('purity', 0.0)
        elif self.config.objective == OptimizationObjective.MINIMIZE_COST:
            return -result.objective_values.get('cost', 0.0)
        else:
            # Multi-objective: return weighted sum
            weights = self.config.metadata.get('objective_weights', {})
            total = 0.0
            for obj_name, value in result.objective_values.items():
                weight = weights.get(obj_name, 1.0)
                total += weight * value
            return total
    
    async def _generate_candidates(self) -> List[Dict[str, Any]]:
        """Generate candidate experiments using the selected strategy."""
        if self.config.strategy == ClosedLoopStrategy.BAYESIAN_OPTIMIZATION:
            return await self._bayesian_optimization_candidates()
        elif self.config.strategy == ClosedLoopStrategy.ACTIVE_LEARNING:
            return await self._active_learning_candidates()
        elif self.config.strategy == ClosedLoopStrategy.GENETIC_ALGORITHM:
            return await self._genetic_algorithm_candidates()
        else:
            # Random sampling as fallback
            return await self._random_candidates()
    
    async def _bayesian_optimization_candidates(self) -> List[Dict[str, Any]]:
        """Generate candidates using Bayesian optimization."""
        if self.surrogate_model is None:
            return await self._random_candidates()
        
        try:
            # Generate candidate points
            num_candidates = 1000
            candidate_points = []
            
            for _ in range(num_candidates):
                point = []
                for param_name in sorted(self.config.parameter_space.keys()):
                    min_val, max_val = self.config.parameter_space[param_name]
                    point.append(np.random.uniform(min_val, max_val))
                candidate_points.append(point)
            
            candidate_points = np.array(candidate_points)
            
            # Evaluate acquisition function
            acquisition_func = self.acquisition_functions[self.config.acquisition_function]
            current_best = self.optimization_state.best_objective
            
            acquisition_values = acquisition_func.evaluate(
                candidate_points,
                self.surrogate_model,
                current_best,
                self.config.exploration_weight
            )
            
            # Select best candidates
            best_indices = np.argsort(acquisition_values)[-self.config.batch_size:]
            
            candidates = []
            for idx in best_indices:
                candidate = {}
                point = candidate_points[idx]
                for i, param_name in enumerate(sorted(self.config.parameter_space.keys())):
                    candidate[param_name] = point[i]
                candidates.append(candidate)
            
            logger.debug(f"Generated {len(candidates)} Bayesian optimization candidates")
            return candidates
            
        except Exception as e:
            logger.error(f"Error in Bayesian optimization: {e}")
            return await self._random_candidates()
    
    async def _active_learning_candidates(self) -> List[Dict[str, Any]]:
        """Generate candidates using active learning strategy."""
        try:
            # Create active learning queries
            queries = []
            
            # Generate diverse candidate points
            for _ in range(self.config.batch_size * 10):  # Generate more than needed
                candidate = {}
                for param_name, (min_val, max_val) in self.config.parameter_space.items():
                    candidate[param_name] = np.random.uniform(min_val, max_val)
                
                # Calculate query scores
                uncertainty_score = self._calculate_uncertainty(candidate)
                information_gain = self._calculate_information_gain(candidate)
                diversity_score = self._calculate_diversity_score(candidate)
                feasibility_score = self._calculate_feasibility_score(candidate)
                
                combined_score = (
                    0.3 * uncertainty_score +
                    0.3 * information_gain +
                    0.2 * diversity_score +
                    0.2 * feasibility_score
                )
                
                query = ActiveLearningQuery(
                    query_id=str(uuid.uuid4()),
                    parameters=candidate,
                    uncertainty_score=uncertainty_score,
                    expected_information_gain=information_gain,
                    diversity_score=diversity_score,
                    feasibility_score=feasibility_score,
                    combined_score=combined_score
                )
                queries.append(query)
            
            # Select top queries
            queries.sort(key=lambda q: q.combined_score, reverse=True)
            selected_queries = queries[:self.config.batch_size]
            
            candidates = [query.parameters for query in selected_queries]
            logger.debug(f"Generated {len(candidates)} active learning candidates")
            return candidates
            
        except Exception as e:
            logger.error(f"Error in active learning candidates: {e}")
            return await self._random_candidates()
    
    async def _genetic_algorithm_candidates(self) -> List[Dict[str, Any]]:
        """Generate candidates using genetic algorithm."""
        if not SCIPY_AVAILABLE:
            logger.warning("Genetic algorithm requires scipy, falling back to random")
            return await self._random_candidates()
        
        try:
            # Define objective function for genetic algorithm
            def objective_function(x):
                if self.surrogate_model is None:
                    return np.random.random()  # Random if no model
                
                # Convert to proper format
                x_reshaped = x.reshape(1, -1)
                prediction, _ = self.surrogate_model.predict(x_reshaped, return_std=True)
                
                # Return negative for minimization (genetic algorithm minimizes)
                if self._is_maximization_objective():
                    return -prediction[0]
                else:
                    return prediction[0]
            
            # Define bounds
            bounds = []
            param_names = sorted(self.config.parameter_space.keys())
            for param_name in param_names:
                min_val, max_val = self.config.parameter_space[param_name]
                bounds.append((min_val, max_val))
            
            # Run genetic algorithm
            result = differential_evolution(
                objective_function,
                bounds,
                maxiter=50,
                popsize=15,
                seed=None
            )
            
            # Create candidate from result
            candidate = {}
            for i, param_name in enumerate(param_names):
                candidate[param_name] = result.x[i]
            
            candidates = [candidate]
            
            # Generate additional candidates if batch_size > 1
            for _ in range(self.config.batch_size - 1):
                candidate = {}
                for param_name, (min_val, max_val) in self.config.parameter_space.items():
                    candidate[param_name] = np.random.uniform(min_val, max_val)
                candidates.append(candidate)
            
            logger.debug(f"Generated {len(candidates)} genetic algorithm candidates")
            return candidates
            
        except Exception as e:
            logger.error(f"Error in genetic algorithm: {e}")
            return await self._random_candidates()
    
    async def _random_candidates(self) -> List[Dict[str, Any]]:
        """Generate random candidates as fallback."""
        candidates = []
        for _ in range(self.config.batch_size):
            candidate = {}
            for param_name, (min_val, max_val) in self.config.parameter_space.items():
                candidate[param_name] = np.random.uniform(min_val, max_val)
            candidates.append(candidate)
        
        logger.debug(f"Generated {len(candidates)} random candidates")
        return candidates
    
    def _calculate_uncertainty(self, parameters: Dict[str, Any]) -> float:
        """Calculate uncertainty score for parameter set."""
        if self.surrogate_model is None:
            return 1.0
        
        try:
            # Convert parameters to feature vector
            features = []
            for param_name in sorted(self.config.parameter_space.keys()):
                features.append(parameters.get(param_name, 0.0))
            
            X = np.array([features])
            _, std = self.surrogate_model.predict(X, return_std=True)
            return std[0]
            
        except Exception:
            return 1.0
    
    def _calculate_information_gain(self, parameters: Dict[str, Any]) -> float:
        """Calculate expected information gain."""
        # Simplified information gain calculation
        uncertainty = self._calculate_uncertainty(parameters)
        return uncertainty
    
    def _calculate_diversity_score(self, parameters: Dict[str, Any]) -> float:
        """Calculate diversity score compared to existing experiments."""
        if not self.experiment_results:
            return 1.0
        
        # Calculate minimum distance to existing experiments
        min_distance = float('inf')
        
        for result in self.experiment_results:
            distance = 0.0
            for param_name, value in parameters.items():
                if param_name in result.parameters:
                    existing_value = result.parameters[param_name]
                    # Normalized distance
                    min_val, max_val = self.config.parameter_space[param_name]
                    norm_distance = abs(value - existing_value) / (max_val - min_val)
                    distance += norm_distance ** 2
            
            distance = np.sqrt(distance)
            min_distance = min(min_distance, distance)
        
        # Normalize to 0-1 range
        return min(min_distance, 1.0)
    
    def _calculate_feasibility_score(self, parameters: Dict[str, Any]) -> float:
        """Calculate feasibility score for parameter set."""
        # Check constraint functions
        for constraint_func in self.config.constraint_functions:
            try:
                if not constraint_func(parameters):
                    return 0.0
            except Exception:
                return 0.0
        
        return 1.0
    
    async def _create_experiment_plans(self, candidates: List[Dict[str, Any]]) -> List[ExperimentPlan]:
        """Create experiment plans for candidate parameters."""
        plans = []
        
        for i, candidate in enumerate(candidates):
            plan_id = f"opt_plan_{self.optimization_state.iteration}_{i}"
            experiment_id = f"opt_exp_{self.optimization_state.iteration}_{i}"
            
            # Create experimental actions based on optimization objective
            actions = self._create_actions_for_candidate(candidate, experiment_id)
            
            plan = ExperimentPlan(
                plan_id=plan_id,
                experiment_id=experiment_id,
                actions=actions,
                optimization_objective=self.config.objective,
                target_parameters=candidate,
                metadata={
                    "optimization_iteration": self.optimization_state.iteration,
                    "strategy": self.config.strategy.value,
                    "candidate_index": i
                }
            )
            
            plans.append(plan)
        
        return plans
    
    def _create_actions_for_candidate(self, candidate: Dict[str, Any], experiment_id: str) -> List[ExperimentalAction]:
        """Create experimental actions for a candidate parameter set."""
        actions = []
        
        # Find suitable robots for different capabilities
        synthesis_robots = self.robot_manager.find_robots_with_capability("chemical_synthesis")
        analysis_robots = self.robot_manager.find_robots_with_capability("analysis")
        
        # Create synthesis action
        if synthesis_robots:
            synthesis_action = ExperimentalAction(
                action_id=f"{experiment_id}_synthesis",
                robot_id=synthesis_robots[0],
                capability_id="chemical_synthesis",
                parameters={
                    "reaction_parameters": candidate,
                    "experiment_id": experiment_id
                },
                priority=3,
                expected_duration=7200.0
            )
            actions.append(synthesis_action)
        
        # Create analysis action (depends on synthesis)
        if analysis_robots:
            analysis_action = ExperimentalAction(
                action_id=f"{experiment_id}_analysis",
                robot_id=analysis_robots[0],
                capability_id="analysis",
                parameters={
                    "analysis_type": "comprehensive",
                    "experiment_id": experiment_id
                },
                priority=2,
                dependencies=[f"{experiment_id}_synthesis"] if synthesis_robots else [],
                expected_duration=600.0
            )
            actions.append(analysis_action)
        
        return actions
    
    async def _wait_for_experiment_completion(self) -> None:
        """Wait for all active experiments to complete."""
        max_wait_time = 24 * 3600  # 24 hours max
        start_time = datetime.now(timezone.utc)
        
        while self.active_experiments and (datetime.now(timezone.utc) - start_time).total_seconds() < max_wait_time:
            await asyncio.sleep(10.0)  # Check every 10 seconds
            
            # Remove completed experiments
            completed_plans = []
            for plan_id, plan in self.active_experiments.items():
                # Check if all actions are complete (simplified check)
                all_complete = True
                for action in plan.actions:
                    # In a real implementation, would check action status
                    pass
                
                if all_complete:
                    completed_plans.append(plan_id)
            
            for plan_id in completed_plans:
                del self.active_experiments[plan_id]
    
    def _handle_robot_feedback(self, feedback: RobotFeedback) -> None:
        """Handle feedback from robots during experiments."""
        logger.debug(f"Received robot feedback: {feedback.action_id} - {feedback.status}")
        
        # Check if this completes an experiment
        if feedback.status == ExperimentStatus.COMPLETED:
            self._process_completed_experiment(feedback)
        elif feedback.status == ExperimentStatus.FAILED:
            self._process_failed_experiment(feedback)
    
    def _process_completed_experiment(self, feedback: RobotFeedback) -> None:
        """Process a completed experiment and extract results."""
        try:
            # Find the corresponding experiment plan
            experiment_plan = None
            for plan in self.active_experiments.values():
                if any(action.action_id == feedback.action_id for action in plan.actions):
                    experiment_plan = plan
                    break
            
            if experiment_plan is None:
                logger.warning(f"Could not find experiment plan for action {feedback.action_id}")
                return
            
            # Extract experimental data from feedback
            experimental_data = []
            if 'experimental_data' in feedback.data:
                for data_item in feedback.data['experimental_data']:
                    exp_data = ExperimentalDataPoint(
                        data_id=data_item.get('data_id', str(uuid.uuid4())),
                        experiment_id=experiment_plan.experiment_id,
                        instrument_id=feedback.robot_id,
                        timestamp=feedback.timestamp,
                        data_type=data_item.get('data_type', 'unknown'),
                        raw_data=data_item.get('raw_data', {}),
                        metadata=data_item.get('metadata', {})
                    )
                    experimental_data.append(exp_data)
            
            # Extract objective values
            objective_values = feedback.data.get('objective_values', {})
            
            # Create experiment result
            result = ExperimentResult(
                experiment_id=experiment_plan.experiment_id,
                parameters=experiment_plan.target_parameters,
                objective_values=objective_values,
                success=True,
                execution_time=(feedback.timestamp - experiment_plan.created_at).total_seconds(),
                robot_feedback=[feedback],
                experimental_data=experimental_data,
                metadata=experiment_plan.metadata
            )
            
            # Add to results
            with self._lock:
                self.experiment_results.append(result)
                self.optimization_state.total_experiments += 1
                self.optimization_state.successful_experiments += 1
                
                # Update best result if needed
                objective_value = self._extract_objective_value(result)
                self.optimization_state.objective_values.append(objective_value)
                
                if self._is_better_objective(objective_value, self.optimization_state.best_objective):
                    self.optimization_state.best_objective = objective_value
                    self.optimization_state.best_parameters = result.parameters.copy()
            
            # Notify callbacks
            for callback in self.experiment_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Error in experiment callback: {e}")
            
            # Update adaptive learning model
            if self.adaptive_manager:
                try:
                    # Convert result to training data
                    features = np.array([list(result.parameters.values())])
                    targets = np.array([objective_value])
                    
                    training_data = TrainingData(
                        features=features,
                        targets=targets,
                        experiment_ids=[result.experiment_id],
                        metadata={"optimization_result": True}
                    )
                    
                    asyncio.create_task(self.adaptive_manager.add_new_data(training_data))
                    
                except Exception as e:
                    logger.error(f"Error updating adaptive learning: {e}")
            
            logger.info(f"Processed completed experiment {experiment_plan.experiment_id} with objective value {objective_value}")
            
        except Exception as e:
            logger.error(f"Error processing completed experiment: {e}")
    
    def _process_failed_experiment(self, feedback: RobotFeedback) -> None:
        """Process a failed experiment."""
        logger.warning(f"Experiment failed: {feedback.action_id} - {feedback.error_message}")
        
        with self._lock:
            self.optimization_state.total_experiments += 1
    
    def _is_better_objective(self, new_value: float, current_best: float) -> bool:
        """Check if new objective value is better than current best."""
        if self._is_maximization_objective():
            return new_value > current_best
        else:
            return new_value < current_best
    
    def _check_convergence(self) -> bool:
        """Check if optimization has converged."""
        criteria = self.config.convergence_criteria
        
        # Check maximum iterations
        if ConvergenceCriteria.MAX_ITERATIONS in criteria:
            max_iter = criteria[ConvergenceCriteria.MAX_ITERATIONS]
            if self.optimization_state.iteration >= max_iter:
                logger.info(f"Optimization converged: reached maximum iterations ({max_iter})")
                return True
        
        # Check objective threshold
        if ConvergenceCriteria.OBJECTIVE_THRESHOLD in criteria:
            threshold = criteria[ConvergenceCriteria.OBJECTIVE_THRESHOLD]
            if self._is_maximization_objective():
                if self.optimization_state.best_objective >= threshold:
                    logger.info(f"Optimization converged: objective threshold reached ({threshold})")
                    return True
            else:
                if self.optimization_state.best_objective <= threshold:
                    logger.info(f"Optimization converged: objective threshold reached ({threshold})")
                    return True
        
        # Check improvement threshold
        if ConvergenceCriteria.IMPROVEMENT_THRESHOLD in criteria and len(self.optimization_state.objective_values) >= 5:
            improvement_threshold = criteria[ConvergenceCriteria.IMPROVEMENT_THRESHOLD]
            recent_values = self.optimization_state.objective_values[-5:]
            max_recent = max(recent_values)
            min_recent = min(recent_values)
            
            if abs(max_recent - min_recent) < improvement_threshold:
                logger.info(f"Optimization converged: improvement threshold reached ({improvement_threshold})")
                return True
        
        return False
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get comprehensive optimization summary."""
        with self._lock:
            return {
                "optimization_state": {
                    "iteration": self.optimization_state.iteration,
                    "total_experiments": self.optimization_state.total_experiments,
                    "successful_experiments": self.optimization_state.successful_experiments,
                    "success_rate": self.optimization_state.successful_experiments / max(1, self.optimization_state.total_experiments),
                    "best_objective": self.optimization_state.best_objective,
                    "best_parameters": self.optimization_state.best_parameters,
                    "is_converged": self._check_convergence()
                },
                "config": {
                    "strategy": self.config.strategy.value,
                    "objective": self.config.objective.value,
                    "parameter_space": self.config.parameter_space,
                    "batch_size": self.config.batch_size
                },
                "results_summary": {
                    "total_results": len(self.experiment_results),
                    "successful_results": len([r for r in self.experiment_results if r.success]),
                    "average_execution_time": np.mean([r.execution_time for r in self.experiment_results]) if self.experiment_results else 0,
                    "objective_history": self.optimization_state.objective_values[-10:]  # Last 10 values
                },
                "active_experiments": len(self.active_experiments)
            }
    
    def add_experiment_callback(self, callback: Callable[[ExperimentResult], None]) -> None:
        """Add callback for experiment completion."""
        self.experiment_callbacks.append(callback)
    
    def add_optimization_callback(self, callback: Callable[[OptimizationState], None]) -> None:
        """Add callback for optimization state updates."""
        self.optimization_callbacks.append(callback)


# Factory functions

def create_bayesian_optimizer(
    robot_manager: RobotManager,
    adaptive_learning_manager: AdaptiveLearningManager,
    parameter_space: Dict[str, Tuple[float, float]],
    objective: OptimizationObjective = OptimizationObjective.MAXIMIZE_YIELD
) -> ClosedLoopOptimizer:
    """Create a Bayesian optimization system."""
    config = OptimizationConfig(
        strategy=ClosedLoopStrategy.BAYESIAN_OPTIMIZATION,
        objective=objective,
        parameter_space=parameter_space,
        convergence_criteria={
            ConvergenceCriteria.MAX_ITERATIONS: 50,
            ConvergenceCriteria.IMPROVEMENT_THRESHOLD: 0.01
        },
        batch_size=2,
        initial_samples=10,
        exploration_weight=0.1,
        acquisition_function="expected_improvement"
    )
    
    return ClosedLoopOptimizer(robot_manager, adaptive_learning_manager, config)


def create_active_learning_optimizer(
    robot_manager: RobotManager,
    adaptive_learning_manager: AdaptiveLearningManager,
    parameter_space: Dict[str, Tuple[float, float]],
    objective: OptimizationObjective = OptimizationObjective.MAXIMIZE_YIELD
) -> ClosedLoopOptimizer:
    """Create an active learning optimization system."""
    config = OptimizationConfig(
        strategy=ClosedLoopStrategy.ACTIVE_LEARNING,
        objective=objective,
        parameter_space=parameter_space,
        convergence_criteria={
            ConvergenceCriteria.MAX_ITERATIONS: 30,
            ConvergenceCriteria.UNCERTAINTY_THRESHOLD: 0.05
        },
        batch_size=3,
        initial_samples=8,
        exploration_weight=0.2,
        acquisition_function="active_learning"
    )
    
    return ClosedLoopOptimizer(robot_manager, adaptive_learning_manager, config)


def create_multi_objective_optimizer(
    robot_manager: RobotManager,
    adaptive_learning_manager: AdaptiveLearningManager,
    parameter_space: Dict[str, Tuple[float, float]],
    objective_weights: Dict[str, float]
) -> ClosedLoopOptimizer:
    """Create a multi-objective optimization system."""
    config = OptimizationConfig(
        strategy=ClosedLoopStrategy.MULTI_OBJECTIVE,
        objective=OptimizationObjective.MULTI_OBJECTIVE,
        parameter_space=parameter_space,
        convergence_criteria={
            ConvergenceCriteria.MAX_ITERATIONS: 40,
            ConvergenceCriteria.IMPROVEMENT_THRESHOLD: 0.02
        },
        batch_size=2,
        initial_samples=12,
        exploration_weight=0.15,
        acquisition_function="expected_improvement",
        metadata={"objective_weights": objective_weights}
    )
    
    return ClosedLoopOptimizer(robot_manager, adaptive_learning_manager, config)