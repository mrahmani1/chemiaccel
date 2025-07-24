"""
Acquisition Functions for Active Learning in ChemiAccel.

This module provides various acquisition functions for selecting the most informative
molecules during active learning campaigns.
"""

import numpy as np
from typing import Optional, Any
from abc import ABC, abstractmethod
from scipy.stats import norm
from scipy.spatial.distance import cdist
import warnings

from ..core.exceptions import ChemiAccelError


class AcquisitionFunctionError(ChemiAccelError):
    """Raised when acquisition function operations fail."""
    pass


class BaseAcquisitionFunction(ABC):
    """Base class for acquisition functions."""
    
    @abstractmethod
    def calculate_scores(self, 
                        predictions: np.ndarray,
                        uncertainties: np.ndarray,
                        features: np.ndarray,
                        training_features: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate acquisition scores for candidate molecules.
        
        Args:
            predictions: Predicted property values
            uncertainties: Prediction uncertainties
            features: Feature vectors for candidates
            training_features: Feature vectors for training set (optional)
            
        Returns:
            Array of acquisition scores (higher is better)
        """
        pass


class UncertaintyAcquisition(BaseAcquisitionFunction):
    """
    Uncertainty-based acquisition function.
    
    Selects molecules with highest prediction uncertainty to reduce model uncertainty.
    """
    
    def __init__(self, temperature: float = 1.0):
        """
        Initialize uncertainty acquisition.
        
        Args:
            temperature: Temperature parameter for score scaling
        """
        self.temperature = temperature
    
    def calculate_scores(self, 
                        predictions: np.ndarray,
                        uncertainties: np.ndarray,
                        features: np.ndarray,
                        training_features: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate uncertainty-based acquisition scores."""
        # Higher uncertainty = higher score
        scores = uncertainties / self.temperature
        return scores


class ExpectedImprovement(BaseAcquisitionFunction):
    """
    Expected Improvement acquisition function.
    
    Balances exploration (uncertainty) with exploitation (predicted improvement).
    """
    
    def __init__(self, xi: float = 0.01, maximize: bool = True):
        """
        Initialize Expected Improvement.
        
        Args:
            xi: Exploration parameter (higher = more exploration)
            maximize: Whether to maximize or minimize the objective
        """
        self.xi = xi
        self.maximize = maximize
        self.best_observed = None
    
    def set_best_observed(self, best_value: float):
        """Set the best observed value for EI calculation."""
        self.best_observed = best_value
    
    def calculate_scores(self, 
                        predictions: np.ndarray,
                        uncertainties: np.ndarray,
                        features: np.ndarray,
                        training_features: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate Expected Improvement scores."""
        if self.best_observed is None:
            # If no best value set, use maximum prediction as proxy
            self.best_observed = np.max(predictions) if self.maximize else np.min(predictions)
        
        # Avoid division by zero
        uncertainties = np.maximum(uncertainties, 1e-8)
        
        if self.maximize:
            improvement = predictions - self.best_observed - self.xi
        else:
            improvement = self.best_observed - predictions - self.xi
        
        # Calculate Z-score
        z = improvement / uncertainties
        
        # Expected Improvement formula
        ei = improvement * norm.cdf(z) + uncertainties * norm.pdf(z)
        
        # Ensure non-negative
        ei = np.maximum(ei, 0)
        
        return ei


class UpperConfidenceBound(BaseAcquisitionFunction):
    """
    Upper Confidence Bound acquisition function.
    
    Balances exploitation and exploration using confidence bounds.
    """
    
    def __init__(self, beta: float = 2.0, maximize: bool = True):
        """
        Initialize Upper Confidence Bound.
        
        Args:
            beta: Exploration parameter (higher = more exploration)
            maximize: Whether to maximize or minimize the objective
        """
        self.beta = beta
        self.maximize = maximize
    
    def calculate_scores(self, 
                        predictions: np.ndarray,
                        uncertainties: np.ndarray,
                        features: np.ndarray,
                        training_features: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate Upper Confidence Bound scores."""
        if self.maximize:
            # Upper confidence bound for maximization
            ucb = predictions + self.beta * uncertainties
        else:
            # Lower confidence bound for minimization
            ucb = predictions - self.beta * uncertainties
        
        return ucb


class DiversityAcquisition(BaseAcquisitionFunction):
    """
    Diversity-based acquisition function.
    
    Selects molecules that are most different from the training set.
    """
    
    def __init__(self, distance_metric: str = 'euclidean', temperature: float = 1.0):
        """
        Initialize diversity acquisition.
        
        Args:
            distance_metric: Distance metric for diversity calculation
            temperature: Temperature parameter for score scaling
        """
        self.distance_metric = distance_metric
        self.temperature = temperature
    
    def calculate_scores(self, 
                        predictions: np.ndarray,
                        uncertainties: np.ndarray,
                        features: np.ndarray,
                        training_features: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate diversity-based acquisition scores."""
        if training_features is None or len(training_features) == 0:
            # If no training data, return uniform scores
            return np.ones(len(features))
        
        # Calculate distances to training set
        distances = cdist(features, training_features, metric=self.distance_metric)
        
        # Use minimum distance to training set as diversity measure
        min_distances = np.min(distances, axis=1)
        
        # Scale by temperature
        diversity_scores = min_distances / self.temperature
        
        return diversity_scores


class GreedyAcquisition(BaseAcquisitionFunction):
    """
    Greedy acquisition function.
    
    Selects molecules with best predicted properties (pure exploitation).
    """
    
    def __init__(self, maximize: bool = True):
        """
        Initialize greedy acquisition.
        
        Args:
            maximize: Whether to maximize or minimize the objective
        """
        self.maximize = maximize
    
    def calculate_scores(self, 
                        predictions: np.ndarray,
                        uncertainties: np.ndarray,
                        features: np.ndarray,
                        training_features: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate greedy acquisition scores."""
        if self.maximize:
            return predictions
        else:
            return -predictions


class ThompsonSampling(BaseAcquisitionFunction):
    """
    Thompson Sampling acquisition function.
    
    Samples from the posterior distribution to balance exploration and exploitation.
    """
    
    def __init__(self, n_samples: int = 1, random_state: Optional[int] = None):
        """
        Initialize Thompson Sampling.
        
        Args:
            n_samples: Number of samples to draw from posterior
            random_state: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
    
    def calculate_scores(self, 
                        predictions: np.ndarray,
                        uncertainties: np.ndarray,
                        features: np.ndarray,
                        training_features: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate Thompson Sampling scores."""
        # Sample from posterior (assuming Gaussian)
        samples = []
        for _ in range(self.n_samples):
            sample = np.random.normal(predictions, uncertainties)
            samples.append(sample)
        
        # Use maximum sampled value as score
        samples = np.array(samples)
        thompson_scores = np.max(samples, axis=0)
        
        return thompson_scores


class ProbabilityOfImprovement(BaseAcquisitionFunction):
    """
    Probability of Improvement acquisition function.
    
    Calculates the probability that a candidate will improve upon the best observed value.
    """
    
    def __init__(self, xi: float = 0.01, maximize: bool = True):
        """
        Initialize Probability of Improvement.
        
        Args:
            xi: Exploration parameter
            maximize: Whether to maximize or minimize the objective
        """
        self.xi = xi
        self.maximize = maximize
        self.best_observed = None
    
    def set_best_observed(self, best_value: float):
        """Set the best observed value for PI calculation."""
        self.best_observed = best_value
    
    def calculate_scores(self, 
                        predictions: np.ndarray,
                        uncertainties: np.ndarray,
                        features: np.ndarray,
                        training_features: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate Probability of Improvement scores."""
        if self.best_observed is None:
            # If no best value set, use maximum prediction as proxy
            self.best_observed = np.max(predictions) if self.maximize else np.min(predictions)
        
        # Avoid division by zero
        uncertainties = np.maximum(uncertainties, 1e-8)
        
        if self.maximize:
            improvement = predictions - self.best_observed - self.xi
        else:
            improvement = self.best_observed - predictions - self.xi
        
        # Calculate Z-score
        z = improvement / uncertainties
        
        # Probability of improvement
        pi = norm.cdf(z)
        
        return pi


class EntropySearch(BaseAcquisitionFunction):
    """
    Entropy Search acquisition function.
    
    Selects points that maximally reduce the entropy of the posterior distribution.
    """
    
    def __init__(self, n_samples: int = 100, random_state: Optional[int] = None):
        """
        Initialize Entropy Search.
        
        Args:
            n_samples: Number of samples for entropy estimation
            random_state: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
    
    def calculate_scores(self, 
                        predictions: np.ndarray,
                        uncertainties: np.ndarray,
                        features: np.ndarray,
                        training_features: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate Entropy Search scores."""
        # Simplified entropy calculation
        # In practice, this would involve more sophisticated entropy estimation
        
        # Use uncertainty as proxy for entropy reduction potential
        entropy_scores = uncertainties * np.log(uncertainties + 1e-8)
        
        # Higher entropy = higher score (more information gain potential)
        return -entropy_scores  # Negative because we want to maximize information gain


class MultiObjectiveAcquisition(BaseAcquisitionFunction):
    """
    Multi-objective acquisition function.
    
    Combines multiple acquisition strategies with specified weights.
    """
    
    def __init__(self, 
                 acquisition_functions: list,
                 weights: Optional[list] = None):
        """
        Initialize multi-objective acquisition.
        
        Args:
            acquisition_functions: List of acquisition function instances
            weights: Weights for combining functions (default: equal weights)
        """
        self.acquisition_functions = acquisition_functions
        
        if weights is None:
            self.weights = [1.0 / len(acquisition_functions)] * len(acquisition_functions)
        else:
            if len(weights) != len(acquisition_functions):
                raise AcquisitionFunctionError("Number of weights must match number of acquisition functions")
            self.weights = weights
    
    def calculate_scores(self, 
                        predictions: np.ndarray,
                        uncertainties: np.ndarray,
                        features: np.ndarray,
                        training_features: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate multi-objective acquisition scores."""
        combined_scores = np.zeros(len(predictions))
        
        for acq_func, weight in zip(self.acquisition_functions, self.weights):
            scores = acq_func.calculate_scores(
                predictions, uncertainties, features, training_features
            )
            
            # Normalize scores to [0, 1]
            if np.max(scores) > np.min(scores):
                normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
            else:
                normalized_scores = np.ones_like(scores)
            
            combined_scores += weight * normalized_scores
        
        return combined_scores


# Utility functions for acquisition functions
def create_balanced_acquisition(uncertainty_weight: float = 0.4,
                              diversity_weight: float = 0.3,
                              exploitation_weight: float = 0.3) -> MultiObjectiveAcquisition:
    """
    Create a balanced multi-objective acquisition function.
    
    Args:
        uncertainty_weight: Weight for uncertainty-based exploration
        diversity_weight: Weight for diversity-based exploration
        exploitation_weight: Weight for greedy exploitation
        
    Returns:
        MultiObjectiveAcquisition instance
    """
    acquisition_functions = [
        UncertaintyAcquisition(),
        DiversityAcquisition(),
        GreedyAcquisition()
    ]
    
    weights = [uncertainty_weight, diversity_weight, exploitation_weight]
    
    return MultiObjectiveAcquisition(acquisition_functions, weights)


def create_exploration_acquisition(beta: float = 2.0) -> MultiObjectiveAcquisition:
    """
    Create an exploration-focused acquisition function.
    
    Args:
        beta: Exploration parameter
        
    Returns:
        MultiObjectiveAcquisition instance focused on exploration
    """
    acquisition_functions = [
        UncertaintyAcquisition(temperature=0.5),
        DiversityAcquisition(temperature=0.5),
        UpperConfidenceBound(beta=beta)
    ]
    
    weights = [0.4, 0.4, 0.2]
    
    return MultiObjectiveAcquisition(acquisition_functions, weights)


def create_exploitation_acquisition() -> MultiObjectiveAcquisition:
    """
    Create an exploitation-focused acquisition function.
    
    Returns:
        MultiObjectiveAcquisition instance focused on exploitation
    """
    acquisition_functions = [
        GreedyAcquisition(),
        ExpectedImprovement(xi=0.001),  # Low exploration
        UpperConfidenceBound(beta=0.5)  # Low exploration
    ]
    
    weights = [0.5, 0.3, 0.2]
    
    return MultiObjectiveAcquisition(acquisition_functions, weights)


def adaptive_acquisition_strategy(iteration: int, 
                                max_iterations: int,
                                exploration_decay: float = 0.8) -> BaseAcquisitionFunction:
    """
    Create an adaptive acquisition strategy that changes over iterations.
    
    Args:
        iteration: Current iteration number
        max_iterations: Total number of iterations
        exploration_decay: Rate of exploration decay
        
    Returns:
        Acquisition function appropriate for current iteration
    """
    # Calculate exploration factor (starts high, decays over time)
    progress = iteration / max_iterations
    exploration_factor = exploration_decay ** progress
    
    if progress < 0.3:
        # Early iterations: focus on exploration
        return create_exploration_acquisition(beta=2.0 * exploration_factor)
    elif progress < 0.7:
        # Middle iterations: balanced approach
        uncertainty_weight = 0.3 * exploration_factor + 0.1
        diversity_weight = 0.3 * exploration_factor + 0.1
        exploitation_weight = 1.0 - uncertainty_weight - diversity_weight
        
        return create_balanced_acquisition(
            uncertainty_weight, diversity_weight, exploitation_weight
        )
    else:
        # Late iterations: focus on exploitation
        return create_exploitation_acquisition()


# Advanced acquisition functions for specific use cases
class BatchAcquisition(BaseAcquisitionFunction):
    """
    Batch acquisition function for selecting multiple diverse candidates simultaneously.
    """
    
    def __init__(self, 
                 base_acquisition: BaseAcquisitionFunction,
                 diversity_weight: float = 0.5):
        """
        Initialize batch acquisition.
        
        Args:
            base_acquisition: Base acquisition function
            diversity_weight: Weight for diversity among selected candidates
        """
        self.base_acquisition = base_acquisition
        self.diversity_weight = diversity_weight
    
    def select_batch(self, 
                    predictions: np.ndarray,
                    uncertainties: np.ndarray,
                    features: np.ndarray,
                    batch_size: int,
                    training_features: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Select a diverse batch of candidates.
        
        Args:
            predictions: Predicted property values
            uncertainties: Prediction uncertainties
            features: Feature vectors for candidates
            batch_size: Number of candidates to select
            training_features: Training set features
            
        Returns:
            Indices of selected candidates
        """
        # Calculate base acquisition scores
        base_scores = self.base_acquisition.calculate_scores(
            predictions, uncertainties, features, training_features
        )
        
        selected_indices = []
        remaining_indices = list(range(len(features)))
        
        for _ in range(batch_size):
            if not remaining_indices:
                break
            
            # Calculate diversity penalty for remaining candidates
            diversity_penalties = np.zeros(len(remaining_indices))
            
            if selected_indices:
                selected_features = features[selected_indices]
                for i, idx in enumerate(remaining_indices):
                    # Calculate minimum distance to already selected candidates
                    distances = cdist([features[idx]], selected_features, metric='euclidean')
                    min_distance = np.min(distances)
                    diversity_penalties[i] = -self.diversity_weight / (min_distance + 1e-8)
            
            # Combine base scores with diversity penalties
            remaining_base_scores = base_scores[remaining_indices]
            combined_scores = remaining_base_scores + diversity_penalties
            
            # Select best candidate
            best_idx = np.argmax(combined_scores)
            selected_idx = remaining_indices[best_idx]
            
            selected_indices.append(selected_idx)
            remaining_indices.remove(selected_idx)
        
        return np.array(selected_indices)
    
    def calculate_scores(self, 
                        predictions: np.ndarray,
                        uncertainties: np.ndarray,
                        features: np.ndarray,
                        training_features: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate acquisition scores (delegates to base acquisition)."""
        return self.base_acquisition.calculate_scores(
            predictions, uncertainties, features, training_features
        )