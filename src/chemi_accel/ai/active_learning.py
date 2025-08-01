"""
Active Learning module for ChemiAccel.

This module provides intelligent molecule selection for experimental validation
using various acquisition functions and uncertainty quantification methods.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn(
        "scikit-learn not available. Some active learning features will be limited."
    )

from ..core.calculator import Calculator
from ..core.exceptions import ChemiAccelError
from .acquisition_functions import (
    DiversityAcquisition,
    ExpectedImprovement,
    GreedyAcquisition,
    UncertaintyAcquisition,
    UpperConfidenceBound,
)
from .uncertainty_quantification import UncertaintyQuantification

try:
    from .kg_integration import (
        AIContextualPrediction,
        AIKGIntegrationManager,
        AITaskType,
        KGContextType,
        KGUpdateSuggestion,
    )

    KG_INTEGRATION_AVAILABLE = True
except ImportError:
    KG_INTEGRATION_AVAILABLE = False
    warnings.warn("KG Integration not available. Contextual features will be limited.")


class ActiveLearningError(ChemiAccelError):
    """Raised when active learning operations fail."""

    pass


@dataclass
class ActiveLearningResult:
    """Results from active learning iteration."""

    selected_molecules: List[str]
    acquisition_scores: np.ndarray
    predicted_properties: np.ndarray
    uncertainty_estimates: np.ndarray
    diversity_scores: np.ndarray
    iteration: int
    model_performance: Dict[str, float]


class KGContextualActiveLearning:
    """
    Knowledge Graph-enhanced Active Learning for ChemiAccel.

    This class implements intelligent molecule selection for experimental validation
    using active learning strategies enhanced with contextual information from the
    IC-KG for more intelligent compound selection and explanations.
    """

    def __init__(
        self,
        calculator: Calculator,
        kg_integration_manager: Optional["AIKGIntegrationManager"] = None,
        acquisition_function: str = "uncertainty",
        uncertainty_method: str = "deep_ensemble",
        use_kg_guidance: bool = True,
        kg_weight: float = 0.3,
        **kwargs,
    ):
        """
        Initialize KG-contextual active learning.

        Args:
            calculator: ChemiAccel calculator for feature extraction
            kg_integration_manager: Manager for KG integration
            acquisition_function: Acquisition function type
            uncertainty_method: Uncertainty quantification method
            use_kg_guidance: Whether to use KG guidance for selection
            kg_weight: Weight for KG-based scoring (0-1)
            **kwargs: Additional parameters for acquisition function and UQ
        """
        self.calculator = calculator
        self.kg_integration_manager = kg_integration_manager
        self.acquisition_function_name = acquisition_function
        self.uncertainty_method = uncertainty_method
        self.use_kg_guidance = use_kg_guidance
        self.kg_weight = kg_weight

        # Initialize uncertainty quantification
        self.uq = UncertaintyQuantification(method=uncertainty_method, **kwargs)

        # Initialize acquisition function
        self.acquisition_function = self._create_acquisition_function(
            acquisition_function, **kwargs
        )

        # Training data
        self.X_train = None
        self.y_train = None
        self.training_molecules = []

        # Search space
        self.search_space = None
        self.search_space_features = None

        # Active learning history
        self.iteration_history = []
        self.current_iteration = 0

        # Model performance tracking
        self.performance_history = []

        # KG-specific tracking
        self.kg_contexts = {}
        self.kg_update_suggestions = []
        self.contextual_predictions = []

    def _create_acquisition_function(self, function_name: str, **kwargs):
        """Create acquisition function instance."""
        if function_name == "uncertainty":
            return UncertaintyAcquisition(**kwargs)
        elif function_name == "expected_improvement":
            return ExpectedImprovement(**kwargs)
        elif function_name == "upper_confidence_bound":
            return UpperConfidenceBound(**kwargs)
        elif function_name == "diversity":
            return DiversityAcquisition(**kwargs)
        elif function_name == "greedy":
            return GreedyAcquisition(**kwargs)
        else:
            raise ActiveLearningError(f"Unknown acquisition function: {function_name}")

    def initialize_search_space(self, search_space: pd.Series) -> None:
        """
        Initialize the search space for active learning.

        Args:
            search_space: Series of SMILES strings representing the search space
        """
        self.search_space = search_space.copy()

        # Calculate features for search space using ChemiAccel
        print(
            f"Calculating features for {len(search_space)} molecules in search space..."
        )
        try:
            features_df = self.calculator.pandas(search_space)
            self.search_space_features = features_df.values
            print(
                f"Calculated {self.search_space_features.shape[1]} features per molecule"
            )
        except Exception as e:
            raise ActiveLearningError(
                f"Failed to calculate features for search space: {e}"
            )

    def add_training_data(self, molecules: List[str], properties: np.ndarray) -> None:
        """
        Add training data to the active learning loop.

        Args:
            molecules: List of SMILES strings
            properties: Array of property values
        """
        if len(molecules) != len(properties):
            raise ActiveLearningError(
                "Number of molecules must match number of property values"
            )

        # Calculate features for training molecules
        mol_series = pd.Series(molecules)
        try:
            features_df = self.calculator.pandas(mol_series)
            X_new = features_df.values
        except Exception as e:
            raise ActiveLearningError(
                f"Failed to calculate features for training molecules: {e}"
            )

        # Add to training set
        if self.X_train is None:
            self.X_train = X_new
            self.y_train = properties.copy()
            self.training_molecules = molecules.copy()
        else:
            self.X_train = np.vstack([self.X_train, X_new])
            self.y_train = np.concatenate([self.y_train, properties])
            self.training_molecules.extend(molecules)

        print(f"Training set now contains {len(self.training_molecules)} molecules")

    def fit_model(self, validation_split: float = 0.2) -> Dict[str, float]:
        """
        Fit the uncertainty quantification model on current training data.

        Args:
            validation_split: Fraction of data to use for validation

        Returns:
            Dictionary of model performance metrics
        """
        if self.X_train is None or len(self.X_train) == 0:
            raise ActiveLearningError(
                "No training data available. Add training data first."
            )

        # Split data for validation
        if len(self.X_train) > 10 and validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                self.X_train, self.y_train, test_size=validation_split, random_state=42
            )
        else:
            X_train, X_val = self.X_train, self.X_train
            y_train, y_val = self.y_train, self.y_train

        # Fit uncertainty quantification model
        print("Fitting uncertainty quantification model...")
        self.uq.fit(X_train, y_train)

        # Calibrate if validation data is available
        if len(X_val) > 0 and not np.array_equal(X_train, X_val):
            self.uq.calibrate(X_val, y_val)

        # Evaluate model performance
        performance = self.uq.evaluate_calibration(X_val, y_val)
        self.performance_history.append(performance)

        print(
            f"Model performance: MAE={performance.get('mae', 'N/A'):.3f}, "
            f"RMSE={performance.get('rmse', 'N/A'):.3f}"
        )

        return performance

    async def select_kg_contextual_candidates(
        self,
        n_candidates: int = 50,
        exclude_training: bool = True,
        kg_context_types: Optional[List[KGContextType]] = None,
    ) -> ActiveLearningResult:
        """
        Select most informative molecules using KG-contextual active learning.

        Args:
            n_candidates: Number of candidates to select
            exclude_training: Whether to exclude molecules already in training set
            kg_context_types: Types of KG context to consider

        Returns:
            ActiveLearningResult with selected molecules and scores
        """
        # Set default KG context types
        if kg_context_types is None:
            kg_context_types = [
                KGContextType.MOLECULAR_SIMILARITY,
                KGContextType.STRUCTURE_ACTIVITY,
                KGContextType.PROPERTY_CORRELATION,
                KGContextType.DRUG_TARGET,
            ]
        if self.search_space is None:
            raise ActiveLearningError(
                "Search space not initialized. Call initialize_search_space first."
            )

        if self.search_space_features is None:
            raise ActiveLearningError("Search space features not calculated.")

        # Get available molecules (exclude training set if requested)
        available_mask = np.ones(len(self.search_space), dtype=bool)

        if exclude_training and self.training_molecules:
            training_set = set(self.training_molecules)
            for i, mol in enumerate(self.search_space):
                if mol in training_set:
                    available_mask[i] = False

        available_indices = np.where(available_mask)[0]
        if len(available_indices) == 0:
            raise ActiveLearningError("No available molecules in search space")

        available_features = self.search_space_features[available_indices]
        available_molecules = self.search_space.iloc[available_indices].tolist()

        logging.info(
            f"Evaluating {len(available_molecules)} available molecules with KG context..."
        )

        # Step 1: Get KG context for available molecules
        kg_contexts = {}
        if self.use_kg_guidance and self.kg_integration_manager:
            # Sample a subset for KG context retrieval (to manage computational cost)
            sample_size = min(500, len(available_molecules))
            sample_indices = np.random.choice(
                len(available_molecules), sample_size, replace=False
            )
            sample_molecules = [available_molecules[i] for i in sample_indices]

            logging.info(
                f"Retrieving KG context for {len(sample_molecules)} molecules..."
            )
            kg_contexts = await self.kg_integration_manager.context_provider.get_molecular_context(
                sample_molecules, kg_context_types
            )

            # Expand contexts to full available set (using empty context for non-sampled)
            full_kg_contexts = {}
            for i, mol in enumerate(available_molecules):
                if mol in kg_contexts:
                    full_kg_contexts[mol] = kg_contexts[mol]
                else:
                    # Empty context for molecules not sampled
                    from .kg_integration import KGContext

                    full_kg_contexts[mol] = KGContext(
                        context_type=KGContextType.MOLECULAR_SIMILARITY,
                        entities=[],
                        relationships=[],
                        confidence_scores={"default": 0.0},
                    )
            kg_contexts = full_kg_contexts

        # Step 2: Get predictions with uncertainty
        uq_result = self.uq.predict_with_uncertainty(available_features)

        # Step 3: Calculate base acquisition scores
        base_acquisition_scores = self.acquisition_function.calculate_scores(
            predictions=uq_result.predictions,
            uncertainties=uq_result.uncertainties,
            features=available_features,
            training_features=self.X_train if self.X_train is not None else None,
        )

        # Step 4: Calculate KG-enhanced scores
        kg_scores = self._calculate_kg_scores(available_molecules, kg_contexts)

        # Step 5: Calculate diversity scores
        diversity_scores = self._calculate_diversity_scores(
            available_features, available_molecules
        )

        # Step 6: Combine scores with weights
        combined_scores = self._combine_acquisition_scores(
            base_acquisition_scores, kg_scores, diversity_scores
        )

        # Step 7: Select top candidates using multi-objective selection
        if len(combined_scores) < n_candidates:
            n_candidates = len(combined_scores)

        selected_indices = self._select_diverse_candidates(
            combined_scores, available_features, n_candidates
        )

        # Step 8: Extract selected molecules and their scores
        selected_molecules = [available_molecules[i] for i in selected_indices]
        selected_acquisition_scores = combined_scores[selected_indices]
        selected_predictions = uq_result.predictions[selected_indices]
        selected_uncertainties = uq_result.uncertainties[selected_indices]
        selected_diversity_scores = diversity_scores[selected_indices]
        selected_kg_scores = kg_scores[selected_indices]

        # Step 9: Generate contextual predictions if KG integration available
        contextual_predictions = []
        if self.kg_integration_manager:
            for i, mol in enumerate(selected_molecules):
                prediction = {
                    "selected_molecules": [mol],
                    "acquisition_score": float(selected_acquisition_scores[i]),
                    "predicted_property": float(selected_predictions[i]),
                    "uncertainty": float(selected_uncertainties[i]),
                    "kg_score": float(selected_kg_scores[i]),
                }

                contextual_pred = (
                    await self.kg_integration_manager.enrich_ai_prediction(
                        prediction=prediction,
                        molecules=[mol],
                        task_type=AITaskType.ACTIVE_LEARNING,
                        context_types=kg_context_types,
                    )
                )
                contextual_predictions.append(contextual_pred)

            # Generate KG update suggestions
            self.kg_update_suggestions.extend(
                self.kg_integration_manager.feedback_manager.suggest_kg_updates(
                    contextual_predictions, AITaskType.ACTIVE_LEARNING
                )
            )

        # Step 10: Create enhanced result
        result = ActiveLearningResult(
            selected_molecules=selected_molecules,
            acquisition_scores=selected_acquisition_scores,
            predicted_properties=selected_predictions,
            uncertainty_estimates=selected_uncertainties,
            diversity_scores=selected_diversity_scores,
            iteration=self.current_iteration,
            model_performance=(
                self.performance_history[-1] if self.performance_history else {}
            ),
        )

        # Add KG-specific metadata
        if hasattr(result, "metadata"):
            result.metadata = {}
        else:
            result.metadata = {}

        result.metadata.update(
            {
                "kg_contexts_retrieved": len(
                    [c for c in kg_contexts.values() if len(c.entities) > 0]
                ),
                "kg_scores": selected_kg_scores.tolist(),
                "contextual_predictions": len(contextual_predictions),
                "kg_integration_enabled": self.kg_integration_manager is not None,
            }
        )

        # Store contexts and predictions
        self.kg_contexts.update(kg_contexts)
        self.contextual_predictions.extend(contextual_predictions)

        # Store in history
        self.iteration_history.append(result)
        self.current_iteration += 1

        logging.info(
            f"Selected {len(selected_molecules)} candidates for iteration {self.current_iteration}. "
            f"Average KG score: {np.mean(selected_kg_scores):.3f}"
        )

        return result

    def _calculate_diversity_scores(
        self, features: np.ndarray, molecules: pd.Series
    ) -> np.ndarray:
        """Calculate diversity scores for molecules."""
        if self.X_train is None or len(self.X_train) == 0:
            # If no training data, return uniform diversity
            return np.ones(len(features))

        # Calculate minimum distance to training set
        diversity_scores = []

        for feature_vec in features:
            # Calculate distances to all training molecules
            distances = np.linalg.norm(self.X_train - feature_vec, axis=1)
            min_distance = np.min(distances)
            diversity_scores.append(min_distance)

        return np.array(diversity_scores)

    def update_model(self, new_data: pd.DataFrame) -> Dict[str, float]:
        """
        Update predictive model with new experimental results.

        Args:
            new_data: DataFrame with 'smiles' and 'property' columns

        Returns:
            Updated model performance metrics
        """
        if "smiles" not in new_data.columns or "property" not in new_data.columns:
            raise ActiveLearningError(
                "new_data must contain 'smiles' and 'property' columns"
            )

        molecules = new_data["smiles"].tolist()
        properties = new_data["property"].values

        # Add new training data
        self.add_training_data(molecules, properties)

        # Refit model
        performance = self.fit_model()

        return performance

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of active learning optimization progress."""
        if not self.iteration_history:
            return {"message": "No iterations completed yet"}

        # Extract metrics over iterations
        iterations = [result.iteration for result in self.iteration_history]
        avg_uncertainties = [
            np.mean(result.uncertainty_estimates) for result in self.iteration_history
        ]
        avg_acquisition_scores = [
            np.mean(result.acquisition_scores) for result in self.iteration_history
        ]

        # Model performance over time
        model_performances = []
        for perf in self.performance_history:
            model_performances.append(perf.get("rmse", np.nan))

        summary = {
            "total_iterations": len(self.iteration_history),
            "total_training_molecules": len(self.training_molecules),
            "search_space_size": (
                len(self.search_space) if self.search_space is not None else 0
            ),
            "iterations": iterations,
            "average_uncertainties": avg_uncertainties,
            "average_acquisition_scores": avg_acquisition_scores,
            "model_rmse_history": model_performances,
            "current_model_performance": (
                self.performance_history[-1] if self.performance_history else {}
            ),
            "acquisition_function": self.acquisition_function_name,
            "uncertainty_method": self.uncertainty_method,
        }

        return summary

    def export_results(self, filepath: str) -> None:
        """Export active learning results to file."""
        summary = self.get_optimization_summary()

        # Create detailed results DataFrame
        all_results = []
        for result in self.iteration_history:
            for i, mol in enumerate(result.selected_molecules):
                all_results.append(
                    {
                        "iteration": result.iteration,
                        "smiles": mol,
                        "acquisition_score": result.acquisition_scores[i],
                        "predicted_property": result.predicted_properties[i],
                        "uncertainty": result.uncertainty_estimates[i],
                        "diversity_score": result.diversity_scores[i],
                    }
                )

        results_df = pd.DataFrame(all_results)

        # Save to file
        if filepath.endswith(".csv"):
            results_df.to_csv(filepath, index=False)
        elif filepath.endswith(".json"):
            import json

            with open(filepath, "w") as f:
                json.dump(
                    {
                        "summary": summary,
                        "detailed_results": results_df.to_dict("records"),
                    },
                    f,
                    indent=2,
                )
        else:
            raise ActiveLearningError("Unsupported file format. Use .csv or .json")

        print(f"Active learning results exported to {filepath}")

    def _calculate_kg_scores(
        self, molecules: List[str], kg_contexts: Dict[str, Any]
    ) -> np.ndarray:
        """Calculate KG-based scores for molecules."""
        kg_scores = []

        for mol in molecules:
            context = kg_contexts.get(mol)
            if not context or len(context.entities) == 0:
                kg_scores.append(0.0)
                continue

            score = 0.0

            # Score based on similarity to known actives
            similar_molecules = [e for e in context.entities if "similarity_score" in e]
            if similar_molecules:
                max_similarity = max(e["similarity_score"] for e in similar_molecules)
                score += max_similarity * 0.3

            # Score based on known activities (higher uncertainty = higher score)
            activities = [e for e in context.entities if "activity_type" in e]
            if activities:
                # Molecules with known activities might be less informative
                avg_confidence = np.mean([e.get("confidence", 0.0) for e in activities])
                score += (
                    1.0 - avg_confidence
                ) * 0.2  # Inverse confidence for exploration

            # Score based on target interactions (unexplored targets = higher score)
            target_rels = [r for r in context.relationships if r["type"] == "TARGETS"]
            if target_rels:
                # Favor molecules targeting less-studied proteins
                avg_confidence = np.mean(
                    [r.get("confidence", 0.0) for r in target_rels]
                )
                score += (1.0 - avg_confidence) * 0.3

            # Score based on structural novelty
            if (
                not similar_molecules
                or max(e["similarity_score"] for e in similar_molecules) < 0.6
            ):
                score += 0.2  # Bonus for novel structures

            kg_scores.append(min(1.0, score))

        return np.array(kg_scores)

    def _combine_acquisition_scores(
        self,
        base_scores: np.ndarray,
        kg_scores: np.ndarray,
        diversity_scores: np.ndarray,
    ) -> np.ndarray:
        """Combine different types of acquisition scores."""
        # Normalize all scores to [0, 1]
        base_scores_norm = (base_scores - np.min(base_scores)) / (
            np.max(base_scores) - np.min(base_scores) + 1e-8
        )
        kg_scores_norm = (kg_scores - np.min(kg_scores)) / (
            np.max(kg_scores) - np.min(kg_scores) + 1e-8
        )
        diversity_scores_norm = (diversity_scores - np.min(diversity_scores)) / (
            np.max(diversity_scores) - np.min(diversity_scores) + 1e-8
        )

        # Weighted combination
        base_weight = 1.0 - self.kg_weight - 0.2  # Reserve 0.2 for diversity
        kg_weight = self.kg_weight if self.use_kg_guidance else 0.0
        diversity_weight = 0.2

        # Adjust weights if KG guidance is disabled
        if not self.use_kg_guidance:
            base_weight += kg_weight
            kg_weight = 0.0

        combined_scores = (
            base_weight * base_scores_norm
            + kg_weight * kg_scores_norm
            + diversity_weight * diversity_scores_norm
        )

        return combined_scores

    def _select_diverse_candidates(
        self, scores: np.ndarray, features: np.ndarray, n_candidates: int
    ) -> np.ndarray:
        """Select diverse candidates using greedy diversification."""
        selected_indices = []
        remaining_indices = list(range(len(scores)))

        # Select the best candidate first
        best_idx = np.argmax(scores)
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

        # Greedily select remaining candidates based on score and diversity
        while len(selected_indices) < n_candidates and remaining_indices:
            best_candidate = None
            best_combined_score = -1

            for idx in remaining_indices:
                # Calculate diversity penalty
                min_distance = float("inf")
                for selected_idx in selected_indices:
                    distance = np.linalg.norm(features[idx] - features[selected_idx])
                    min_distance = min(min_distance, distance)

                # Normalize distance and combine with acquisition score
                diversity_bonus = min_distance / (np.linalg.norm(features[idx]) + 1e-8)
                combined_score = scores[idx] + 0.1 * diversity_bonus

                if combined_score > best_combined_score:
                    best_combined_score = combined_score
                    best_candidate = idx

            if best_candidate is not None:
                selected_indices.append(best_candidate)
                remaining_indices.remove(best_candidate)

        return np.array(selected_indices)

    async def update_knowledge_graph(self) -> Dict[str, Any]:
        """Update knowledge graph based on active learning results."""
        if not self.kg_integration_manager or not self.kg_update_suggestions:
            return {"message": "No KG updates available"}

        # Apply accumulated KG update suggestions
        update_results = await self.kg_integration_manager.feedback_manager.validate_and_apply_updates(
            self.kg_update_suggestions
        )

        # Clear processed suggestions
        self.kg_update_suggestions = []

        return update_results

    def get_kg_enhanced_summary(self) -> Dict[str, Any]:
        """Get summary including KG-enhanced metrics."""
        base_summary = self.get_optimization_summary()

        if not self.kg_integration_manager:
            return base_summary

        # Add KG-specific metrics
        kg_metrics = {
            "kg_contexts_collected": len(self.kg_contexts),
            "contextual_predictions_made": len(self.contextual_predictions),
            "kg_update_suggestions": len(self.kg_update_suggestions),
            "avg_kg_confidence": (
                np.mean(
                    [
                        np.mean(list(context.confidence_scores.values()))
                        for context in self.kg_contexts.values()
                        if context.confidence_scores
                    ]
                )
                if self.kg_contexts
                else 0.0
            ),
            "kg_guidance_enabled": self.use_kg_guidance,
            "kg_weight": self.kg_weight,
        }

        base_summary.update(kg_metrics)
        return base_summary


# Keep the original ActiveLearningLoop class for backward compatibility
class ActiveLearningLoop(KGContextualActiveLearning):
    """
    Backward-compatible Active Learning class.

    This class maintains the original API while optionally supporting KG integration.
    """

    def __init__(
        self,
        calculator: Calculator,
        acquisition_function: str = "uncertainty",
        uncertainty_method: str = "deep_ensemble",
        **kwargs,
    ):
        """Initialize active learning (backward compatible)."""
        super().__init__(
            calculator=calculator,
            kg_integration_manager=None,  # No KG integration by default
            acquisition_function=acquisition_function,
            uncertainty_method=uncertainty_method,
            use_kg_guidance=False,
            **kwargs,
        )

    def select_candidates(
        self, n_candidates: int = 50, exclude_training: bool = True
    ) -> ActiveLearningResult:
        """Original select_candidates method (synchronous, no KG)."""
        # Use asyncio to run the async method synchronously
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.select_kg_contextual_candidates(
                n_candidates=n_candidates,
                exclude_training=exclude_training,
                kg_context_types=[],  # No KG context for backward compatibility
            )
        )


# Utility functions for active learning
def create_balanced_acquisition_function(
    uncertainty_weight: float = 0.5,
    diversity_weight: float = 0.3,
    exploitation_weight: float = 0.2,
):
    """
    Create a balanced acquisition function combining multiple strategies.

    Args:
        uncertainty_weight: Weight for uncertainty-based selection
        diversity_weight: Weight for diversity-based selection
        exploitation_weight: Weight for exploitation (greedy) selection

    Returns:
        Balanced acquisition function
    """

    class BalancedAcquisition:
        def __init__(self):
            self.uncertainty_acq = UncertaintyAcquisition()
            self.diversity_acq = DiversityAcquisition()
            self.greedy_acq = GreedyAcquisition()

        def calculate_scores(
            self, predictions, uncertainties, features, training_features=None
        ):
            # Calculate individual scores
            uncertainty_scores = self.uncertainty_acq.calculate_scores(
                predictions, uncertainties, features, training_features
            )
            diversity_scores = self.diversity_acq.calculate_scores(
                predictions, uncertainties, features, training_features
            )
            greedy_scores = self.greedy_acq.calculate_scores(
                predictions, uncertainties, features, training_features
            )

            # Normalize scores to [0, 1]
            uncertainty_scores = (uncertainty_scores - np.min(uncertainty_scores)) / (
                np.max(uncertainty_scores) - np.min(uncertainty_scores) + 1e-8
            )
            diversity_scores = (diversity_scores - np.min(diversity_scores)) / (
                np.max(diversity_scores) - np.min(diversity_scores) + 1e-8
            )
            greedy_scores = (greedy_scores - np.min(greedy_scores)) / (
                np.max(greedy_scores) - np.min(greedy_scores) + 1e-8
            )

            # Combine with weights
            combined_scores = (
                uncertainty_weight * uncertainty_scores
                + diversity_weight * diversity_scores
                + exploitation_weight * greedy_scores
            )

            return combined_scores

    return BalancedAcquisition()


def run_active_learning_campaign(
    calculator: Calculator,
    search_space: pd.Series,
    initial_molecules: List[str],
    initial_properties: np.ndarray,
    n_iterations: int = 10,
    n_candidates_per_iteration: int = 20,
    acquisition_function: str = "uncertainty",
    uncertainty_method: str = "deep_ensemble",
) -> ActiveLearningLoop:
    """
    Run a complete active learning campaign.

    Args:
        calculator: ChemiAccel calculator
        search_space: Series of SMILES strings for search space
        initial_molecules: Initial training molecules
        initial_properties: Initial property values
        n_iterations: Number of active learning iterations
        n_candidates_per_iteration: Candidates to select per iteration
        acquisition_function: Acquisition function to use
        uncertainty_method: Uncertainty quantification method

    Returns:
        Configured ActiveLearningLoop with completed campaign
    """
    # Initialize active learning loop
    al_loop = ActiveLearningLoop(
        calculator=calculator,
        acquisition_function=acquisition_function,
        uncertainty_method=uncertainty_method,
    )

    # Set up search space and initial training data
    al_loop.initialize_search_space(search_space)
    al_loop.add_training_data(initial_molecules, initial_properties)

    # Run iterations
    for iteration in range(n_iterations):
        print(f"\n=== Active Learning Iteration {iteration + 1}/{n_iterations} ===")

        # Fit model
        performance = al_loop.fit_model()

        # Select candidates
        result = al_loop.select_candidates(n_candidates=n_candidates_per_iteration)

        print(
            f"Selected molecules with average uncertainty: {np.mean(result.uncertainty_estimates):.3f}"
        )
        print("Top 3 selected molecules:")
        for i in range(min(3, len(result.selected_molecules))):
            print(
                f"  {i+1}. {result.selected_molecules[i]} "
                f"(score: {result.acquisition_scores[i]:.3f}, "
                f"uncertainty: {result.uncertainty_estimates[i]:.3f})"
            )

        # In a real campaign, you would:
        # 1. Synthesize/test the selected molecules
        # 2. Measure their properties
        # 3. Add results to training set with al_loop.update_model()

        # For demonstration, we'll simulate this step
        # (In practice, this would be replaced with actual experimental data)
        simulated_properties = np.random.normal(
            result.predicted_properties, result.uncertainty_estimates * 0.5
        )

        # Update model with simulated results
        new_data = pd.DataFrame(
            {"smiles": result.selected_molecules, "property": simulated_properties}
        )
        al_loop.update_model(new_data)

    return al_loop
