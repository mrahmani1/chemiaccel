"""
Adaptive Model Retraining and Fine-tuning System

This module provides comprehensive functionality for automatic model retraining,
incremental learning, and strategies to prevent catastrophic forgetting in
autonomous laboratory environments.
"""

import asyncio
import copy
import hashlib
import logging
import threading
import warnings
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class RetrainingStrategy(Enum):
    """Available retraining strategies."""

    FULL_RETRAIN = "full_retrain"
    INCREMENTAL = "incremental"
    TRANSFER_LEARNING = "transfer_learning"
    CONTINUAL_LEARNING = "continual_learning"
    ENSEMBLE_UPDATE = "ensemble_update"


class DataBalancingStrategy(Enum):
    """Data balancing strategies to prevent catastrophic forgetting."""

    RESERVOIR_SAMPLING = "reservoir_sampling"
    IMPORTANCE_WEIGHTED = "importance_weighted"
    GRADUAL_FORGETTING = "gradual_forgetting"
    EXPERIENCE_REPLAY = "experience_replay"
    ELASTIC_WEIGHT_CONSOLIDATION = "elastic_weight_consolidation"


@dataclass
class ModelSnapshot:
    """Represents a snapshot of a model at a specific point in time."""

    model_id: str
    timestamp: datetime
    model_state: Any
    performance_metrics: Dict[str, float]
    training_data_hash: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingData:
    """Container for training data with metadata."""

    features: np.ndarray
    targets: np.ndarray
    sample_weights: Optional[np.ndarray] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    experiment_ids: Optional[List[str]] = None
    data_quality_scores: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate data consistency after initialization."""
        if len(self.features) != len(self.targets):
            raise ValueError("Features and targets must have the same length")

        if self.sample_weights is not None and len(self.sample_weights) != len(
            self.features
        ):
            raise ValueError("Sample weights must have the same length as features")

        if self.data_quality_scores is not None and len(
            self.data_quality_scores
        ) != len(self.features):
            raise ValueError(
                "Data quality scores must have the same length as features"
            )


@dataclass
class RetrainingConfig:
    """Configuration for adaptive retraining."""

    strategy: RetrainingStrategy = RetrainingStrategy.INCREMENTAL
    balancing_strategy: DataBalancingStrategy = DataBalancingStrategy.RESERVOIR_SAMPLING
    min_new_samples: int = 100
    performance_threshold: float = 0.95
    retrain_interval_hours: int = 24
    max_memory_samples: int = 10000
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    catastrophic_forgetting_threshold: float = 0.1
    enable_performance_monitoring: bool = True
    save_model_snapshots: bool = True
    max_snapshots: int = 10


class PerformanceMonitor:
    """Monitors model performance and triggers retraining when necessary."""

    def __init__(self, config: RetrainingConfig):
        self.config = config
        self.performance_history: List[Dict[str, Any]] = []
        self.baseline_performance: Optional[Dict[str, float]] = None
        self.monitoring_active = False
        self.last_evaluation: Optional[datetime] = None

    def update_baseline(self, metrics: Dict[str, float]):
        """Update baseline performance metrics."""
        self.baseline_performance = metrics.copy()
        logger.info(f"Updated baseline performance: {metrics}")

    def evaluate_model(self, model: Any, test_data: TrainingData) -> Dict[str, float]:
        """Evaluate model performance on test data."""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available, using mock metrics")
            return {"mse": 0.1, "mae": 0.05, "r2": 0.9}

        try:
            predictions = model.predict(test_data.features)

            metrics = {
                "mse": mean_squared_error(test_data.targets, predictions),
                "mae": mean_absolute_error(test_data.targets, predictions),
                "r2": r2_score(test_data.targets, predictions),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "n_samples": len(test_data.features),
            }

            # Add custom quality metrics if available
            if test_data.data_quality_scores is not None:
                metrics["avg_data_quality"] = float(
                    np.mean(test_data.data_quality_scores)
                )

            self.performance_history.append(metrics)
            self.last_evaluation = datetime.now(timezone.utc)

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {"error": str(e)}

    def check_performance_degradation(self, current_metrics: Dict[str, float]) -> bool:
        """Check if performance has degraded significantly."""
        if not self.baseline_performance:
            return False

        # Check for catastrophic forgetting
        for metric in ["r2", "mse", "mae"]:
            if metric in current_metrics and metric in self.baseline_performance:
                if metric == "r2":
                    # Higher is better for R²
                    degradation = (
                        self.baseline_performance[metric] - current_metrics[metric]
                    )
                else:
                    # Lower is better for MSE and MAE
                    degradation = (
                        current_metrics[metric] - self.baseline_performance[metric]
                    )

                if degradation > self.config.catastrophic_forgetting_threshold:
                    logger.warning(
                        f"Performance degradation detected in {metric}: {degradation}"
                    )
                    return True

        return False

    def should_retrain(
        self, new_data_count: int, current_metrics: Optional[Dict[str, float]] = None
    ) -> bool:
        """Determine if model should be retrained."""
        reasons = []

        # Check minimum new samples
        if new_data_count >= self.config.min_new_samples:
            reasons.append(f"Sufficient new samples: {new_data_count}")

        # Check time-based trigger
        if self.last_evaluation:
            time_since_last = datetime.now(timezone.utc) - self.last_evaluation
            if (
                time_since_last.total_seconds() / 3600
                >= self.config.retrain_interval_hours
            ):
                reasons.append(f"Time threshold reached: {time_since_last}")

        # Check performance degradation
        if current_metrics and self.check_performance_degradation(current_metrics):
            reasons.append("Performance degradation detected")

        should_retrain = len(reasons) > 0
        if should_retrain:
            logger.info(f"Retraining triggered. Reasons: {'; '.join(reasons)}")

        return should_retrain

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance monitoring."""
        return {
            "monitoring_active": self.monitoring_active,
            "baseline_performance": self.baseline_performance,
            "last_evaluation": self.last_evaluation.isoformat()
            if self.last_evaluation
            else None,
            "evaluation_count": len(self.performance_history),
            "recent_performance": self.performance_history[-5:]
            if self.performance_history
            else [],
        }


class DataBalancer:
    """Manages data balancing to prevent catastrophic forgetting."""

    def __init__(
        self, strategy: DataBalancingStrategy, max_memory_samples: int = 10000
    ):
        self.strategy = strategy
        self.max_memory_samples = max_memory_samples
        self.memory_buffer: deque = deque(maxlen=max_memory_samples)
        self.importance_weights: Dict[str, float] = {}
        self.data_hash_to_importance: Dict[str, float] = {}

    def add_to_memory(self, data: TrainingData, importance_score: float = 1.0):
        """Add data to memory buffer with importance scoring."""
        data_hash = self._hash_data(data.features, data.targets)

        memory_item = {
            "data": data,
            "importance": importance_score,
            "timestamp": datetime.now(timezone.utc),
            "hash": data_hash,
        }

        self.memory_buffer.append(memory_item)
        self.data_hash_to_importance[data_hash] = importance_score

    def balance_data(
        self, new_data: TrainingData, historical_data: Optional[TrainingData] = None
    ) -> TrainingData:
        """Balance new data with historical data to prevent catastrophic forgetting."""
        if self.strategy == DataBalancingStrategy.RESERVOIR_SAMPLING:
            return self._reservoir_sampling(new_data, historical_data)
        elif self.strategy == DataBalancingStrategy.IMPORTANCE_WEIGHTED:
            return self._importance_weighted_balance(new_data, historical_data)
        elif self.strategy == DataBalancingStrategy.EXPERIENCE_REPLAY:
            return self._experience_replay_balance(new_data, historical_data)
        elif self.strategy == DataBalancingStrategy.GRADUAL_FORGETTING:
            return self._gradual_forgetting_balance(new_data, historical_data)
        else:
            logger.warning(
                f"Unknown balancing strategy: {self.strategy}, using reservoir sampling"
            )
            return self._reservoir_sampling(new_data, historical_data)

    def _reservoir_sampling(
        self, new_data: TrainingData, historical_data: Optional[TrainingData]
    ) -> TrainingData:
        """Implement reservoir sampling for data balancing."""
        if not self.memory_buffer:
            return new_data

        # Calculate how many historical samples to include
        n_new = len(new_data.features)
        n_historical = min(len(self.memory_buffer), self.max_memory_samples - n_new)

        if n_historical <= 0:
            return new_data

        # Sample from memory buffer
        memory_indices = np.random.choice(
            len(self.memory_buffer), n_historical, replace=False
        )
        memory_items = [self.memory_buffer[i] for i in memory_indices]

        # Combine data
        historical_features = np.vstack(
            [item["data"].features for item in memory_items]
        )
        historical_targets = np.concatenate(
            [item["data"].targets for item in memory_items]
        )

        combined_features = np.vstack([new_data.features, historical_features])
        combined_targets = np.concatenate([new_data.targets, historical_targets])

        # Create sample weights favoring new data slightly
        new_weights = np.ones(n_new) * 1.1
        # Create weights for historical data - each sample in a memory item gets the item's importance
        historical_weights = []
        for item in memory_items:
            n_samples_in_item = len(item["data"].features)
            historical_weights.extend([item["importance"]] * n_samples_in_item)
        historical_weights = np.array(historical_weights)
        combined_weights = np.concatenate([new_weights, historical_weights])

        return TrainingData(
            features=combined_features,
            targets=combined_targets,
            sample_weights=combined_weights,
            timestamp=datetime.now(timezone.utc),
            metadata={
                "balancing_strategy": "reservoir_sampling",
                "n_new": n_new,
                "n_historical": n_historical,
            },
        )

    def _importance_weighted_balance(
        self, new_data: TrainingData, historical_data: Optional[TrainingData]
    ) -> TrainingData:
        """Balance data using importance weighting."""
        if not self.memory_buffer:
            return new_data

        # Sort memory by importance and recency
        sorted_memory = sorted(
            self.memory_buffer,
            key=lambda x: (x["importance"], x["timestamp"]),
            reverse=True,
        )

        # Select top samples by importance
        n_new = len(new_data.features)
        n_historical = min(len(sorted_memory), self.max_memory_samples - n_new)

        if n_historical <= 0:
            return new_data

        selected_items = sorted_memory[:n_historical]

        # Combine data
        historical_features = np.vstack(
            [item["data"].features for item in selected_items]
        )
        historical_targets = np.concatenate(
            [item["data"].targets for item in selected_items]
        )

        combined_features = np.vstack([new_data.features, historical_features])
        combined_targets = np.concatenate([new_data.targets, historical_targets])

        # Use importance scores as weights
        new_weights = np.ones(n_new) * 1.2  # Slightly favor new data
        # Create weights for historical data - each sample in a memory item gets the item's importance
        historical_weights = []
        for item in selected_items:
            n_samples_in_item = len(item["data"].features)
            historical_weights.extend([item["importance"]] * n_samples_in_item)
        historical_weights = np.array(historical_weights)
        combined_weights = np.concatenate([new_weights, historical_weights])

        return TrainingData(
            features=combined_features,
            targets=combined_targets,
            sample_weights=combined_weights,
            timestamp=datetime.now(timezone.utc),
            metadata={
                "balancing_strategy": "importance_weighted",
                "n_new": n_new,
                "n_historical": n_historical,
            },
        )

    def _experience_replay_balance(
        self, new_data: TrainingData, historical_data: Optional[TrainingData]
    ) -> TrainingData:
        """Implement experience replay for continual learning."""
        if not self.memory_buffer:
            return new_data

        # Select diverse samples from memory
        n_new = len(new_data.features)
        n_replay = min(len(self.memory_buffer) // 2, self.max_memory_samples - n_new)

        if n_replay <= 0:
            return new_data

        # Use stratified sampling based on target distribution
        memory_items = list(self.memory_buffer)
        if len(memory_items) > n_replay:
            # Simple random sampling for now (can be enhanced with diversity measures)
            selected_indices = np.random.choice(
                len(memory_items), n_replay, replace=False
            )
            selected_items = [memory_items[i] for i in selected_indices]
        else:
            selected_items = memory_items

        # Combine data
        replay_features = np.vstack([item["data"].features for item in selected_items])
        replay_targets = np.concatenate(
            [item["data"].targets for item in selected_items]
        )

        combined_features = np.vstack([new_data.features, replay_features])
        combined_targets = np.concatenate([new_data.targets, replay_targets])

        # Equal weighting for experience replay
        combined_weights = np.ones(len(combined_features))

        return TrainingData(
            features=combined_features,
            targets=combined_targets,
            sample_weights=combined_weights,
            timestamp=datetime.now(timezone.utc),
            metadata={
                "balancing_strategy": "experience_replay",
                "n_new": n_new,
                "n_replay": len(selected_items),
            },
        )

    def _gradual_forgetting_balance(
        self, new_data: TrainingData, historical_data: Optional[TrainingData]
    ) -> TrainingData:
        """Implement gradual forgetting with time-decay weighting."""
        if not self.memory_buffer:
            return new_data

        current_time = datetime.now(timezone.utc)
        n_new = len(new_data.features)

        # Calculate time-decay weights for historical data
        time_weighted_items = []
        for item in self.memory_buffer:
            time_diff = current_time - item["timestamp"]
            decay_factor = np.exp(
                -time_diff.total_seconds() / (24 * 3600)
            )  # Daily decay
            weight = item["importance"] * decay_factor
            time_weighted_items.append((item, weight))

        # Sort by decayed importance
        time_weighted_items.sort(key=lambda x: x[1], reverse=True)

        # Select samples
        n_historical = min(len(time_weighted_items), self.max_memory_samples - n_new)
        if n_historical <= 0:
            return new_data

        selected_items = time_weighted_items[:n_historical]

        # Combine data
        historical_features = np.vstack(
            [item[0]["data"].features for item in selected_items]
        )
        historical_targets = np.concatenate(
            [item[0]["data"].targets for item in selected_items]
        )

        combined_features = np.vstack([new_data.features, historical_features])
        combined_targets = np.concatenate([new_data.targets, historical_targets])

        # Use decayed weights
        new_weights = np.ones(n_new) * 1.0
        # Create weights for historical data - each sample in a memory item gets the item's decayed weight
        historical_weights = []
        for item in selected_items:
            n_samples_in_item = len(item[0]["data"].features)
            historical_weights.extend([item[1]] * n_samples_in_item)
        historical_weights = np.array(historical_weights)
        combined_weights = np.concatenate([new_weights, historical_weights])

        return TrainingData(
            features=combined_features,
            targets=combined_targets,
            sample_weights=combined_weights,
            timestamp=datetime.now(timezone.utc),
            metadata={
                "balancing_strategy": "gradual_forgetting",
                "n_new": n_new,
                "n_historical": n_historical,
            },
        )

    def _hash_data(self, features: np.ndarray, targets: np.ndarray) -> str:
        """Create hash of data for deduplication."""
        combined = np.concatenate([features.flatten(), targets.flatten()])
        return hashlib.md5(combined.tobytes()).hexdigest()

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about memory buffer."""
        if not self.memory_buffer:
            return {"size": 0, "strategy": self.strategy.value}

        importances = [item["importance"] for item in self.memory_buffer]
        timestamps = [item["timestamp"] for item in self.memory_buffer]

        return {
            "size": len(self.memory_buffer),
            "strategy": self.strategy.value,
            "max_size": self.max_memory_samples,
            "avg_importance": float(np.mean(importances)),
            "oldest_sample": min(timestamps).isoformat(),
            "newest_sample": max(timestamps).isoformat(),
        }


class ModelRetrainer:
    """Handles model retraining with various strategies."""

    def __init__(self, config: RetrainingConfig):
        self.config = config
        self.current_model: Optional[Any] = None
        self.model_snapshots: List[ModelSnapshot] = []
        self.scaler: Optional[Any] = None
        self.training_history: List[Dict[str, Any]] = []
        self.retrain_lock = threading.Lock()

    def set_current_model(self, model: Any, scaler: Optional[Any] = None):
        """Set the current model for retraining."""
        self.current_model = model
        self.scaler = scaler

        # Create initial snapshot
        if self.config.save_model_snapshots:
            snapshot = self._create_model_snapshot("initial")
            self.model_snapshots.append(snapshot)

    async def retrain_model(
        self,
        training_data: TrainingData,
        validation_data: Optional[TrainingData] = None,
    ) -> Tuple[Any, Dict[str, float]]:
        """Retrain the model with new data."""
        if not self.current_model:
            raise ValueError("No current model set for retraining")

        with self.retrain_lock:
            logger.info(
                f"Starting model retraining with strategy: {self.config.strategy}"
            )

            if self.config.strategy == RetrainingStrategy.FULL_RETRAIN:
                return await self._full_retrain(training_data, validation_data)
            elif self.config.strategy == RetrainingStrategy.INCREMENTAL:
                return await self._incremental_retrain(training_data, validation_data)
            elif self.config.strategy == RetrainingStrategy.TRANSFER_LEARNING:
                return await self._transfer_learning_retrain(
                    training_data, validation_data
                )
            elif self.config.strategy == RetrainingStrategy.ENSEMBLE_UPDATE:
                return await self._ensemble_update(training_data, validation_data)
            else:
                logger.warning(
                    f"Unknown retraining strategy: {self.config.strategy}, using incremental"
                )
                return await self._incremental_retrain(training_data, validation_data)

    async def _full_retrain(
        self, training_data: TrainingData, validation_data: Optional[TrainingData]
    ) -> Tuple[Any, Dict[str, float]]:
        """Perform full model retraining."""
        logger.info("Performing full model retraining")

        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available, using mock retraining")
            return self.current_model, {"mse": 0.1, "r2": 0.9}

        try:
            # Create new model instance
            if hasattr(self.current_model, "get_params"):
                params = self.current_model.get_params()
                new_model = type(self.current_model)(**params)
            else:
                new_model = RandomForestRegressor(n_estimators=100, random_state=42)

            # Prepare data
            X_train = training_data.features
            y_train = training_data.targets
            sample_weight = training_data.sample_weights

            # Scale features if scaler is available
            if self.scaler is not None:
                self.scaler.fit(X_train)
                X_train = self.scaler.transform(X_train)

            # Train model
            new_model.fit(X_train, y_train, sample_weight=sample_weight)

            # Evaluate
            metrics = {}
            if validation_data is not None:
                X_val = validation_data.features
                if self.scaler is not None:
                    X_val = self.scaler.transform(X_val)

                y_pred = new_model.predict(X_val)
                metrics = {
                    "mse": mean_squared_error(validation_data.targets, y_pred),
                    "mae": mean_absolute_error(validation_data.targets, y_pred),
                    "r2": r2_score(validation_data.targets, y_pred),
                }

            # Update current model
            self.current_model = new_model

            # Save snapshot
            if self.config.save_model_snapshots:
                snapshot = self._create_model_snapshot("full_retrain", metrics)
                self.model_snapshots.append(snapshot)
                self._cleanup_snapshots()

            # Record training history
            self.training_history.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "strategy": "full_retrain",
                    "n_samples": len(training_data.features),
                    "metrics": metrics,
                }
            )

            logger.info(f"Full retraining completed. Metrics: {metrics}")
            return new_model, metrics

        except Exception as e:
            logger.error(f"Error in full retraining: {e}")
            raise

    async def _incremental_retrain(
        self, training_data: TrainingData, validation_data: Optional[TrainingData]
    ) -> Tuple[Any, Dict[str, float]]:
        """Perform incremental model retraining."""
        logger.info("Performing incremental model retraining")

        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available, using mock retraining")
            return self.current_model, {"mse": 0.1, "r2": 0.9}

        try:
            # For sklearn models that don't support incremental learning,
            # we simulate it by retraining with combined data
            if hasattr(self.current_model, "partial_fit"):
                # True incremental learning
                X_train = training_data.features
                y_train = training_data.targets

                if self.scaler is not None:
                    X_train = self.scaler.transform(X_train)

                self.current_model.partial_fit(X_train, y_train)
            else:
                # Simulate incremental learning with full retrain on combined data
                # This is where the DataBalancer comes in to manage historical data
                return await self._full_retrain(training_data, validation_data)

            # Evaluate
            metrics = {}
            if validation_data is not None:
                X_val = validation_data.features
                if self.scaler is not None:
                    X_val = self.scaler.transform(X_val)

                y_pred = self.current_model.predict(X_val)
                metrics = {
                    "mse": mean_squared_error(validation_data.targets, y_pred),
                    "mae": mean_absolute_error(validation_data.targets, y_pred),
                    "r2": r2_score(validation_data.targets, y_pred),
                }

            # Save snapshot
            if self.config.save_model_snapshots:
                snapshot = self._create_model_snapshot("incremental", metrics)
                self.model_snapshots.append(snapshot)
                self._cleanup_snapshots()

            # Record training history
            self.training_history.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "strategy": "incremental",
                    "n_samples": len(training_data.features),
                    "metrics": metrics,
                }
            )

            logger.info(f"Incremental retraining completed. Metrics: {metrics}")
            return self.current_model, metrics

        except Exception as e:
            logger.error(f"Error in incremental retraining: {e}")
            raise

    async def _transfer_learning_retrain(
        self, training_data: TrainingData, validation_data: Optional[TrainingData]
    ) -> Tuple[Any, Dict[str, float]]:
        """Perform transfer learning-based retraining."""
        logger.info("Performing transfer learning retraining")

        # For traditional ML models, transfer learning is similar to fine-tuning
        # with careful weight initialization and learning rate adjustment
        return await self._incremental_retrain(training_data, validation_data)

    async def _ensemble_update(
        self, training_data: TrainingData, validation_data: Optional[TrainingData]
    ) -> Tuple[Any, Dict[str, float]]:
        """Update model using ensemble approach."""
        logger.info("Performing ensemble-based model update")

        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available, using mock retraining")
            return self.current_model, {"mse": 0.1, "r2": 0.9}

        try:
            # Train a new model on new data
            new_model = RandomForestRegressor(n_estimators=50, random_state=42)

            X_train = training_data.features
            y_train = training_data.targets

            if self.scaler is not None:
                X_train = self.scaler.transform(X_train)

            new_model.fit(X_train, y_train, sample_weight=training_data.sample_weights)

            # Create ensemble wrapper (simplified version)
            # In practice, this would be a more sophisticated ensemble
            class SimpleEnsemble:
                def __init__(
                    self, old_model, new_model, old_weight=0.7, new_weight=0.3
                ):
                    self.old_model = old_model
                    self.new_model = new_model
                    self.old_weight = old_weight
                    self.new_weight = new_weight

                def predict(self, X):
                    old_pred = self.old_model.predict(X)
                    new_pred = self.new_model.predict(X)
                    return self.old_weight * old_pred + self.new_weight * new_pred

                def get_params(self, deep=True):
                    return {
                        "old_weight": self.old_weight,
                        "new_weight": self.new_weight,
                    }

            ensemble_model = SimpleEnsemble(self.current_model, new_model)

            # Evaluate ensemble
            metrics = {}
            if validation_data is not None:
                X_val = validation_data.features
                if self.scaler is not None:
                    X_val = self.scaler.transform(X_val)

                y_pred = ensemble_model.predict(X_val)
                metrics = {
                    "mse": mean_squared_error(validation_data.targets, y_pred),
                    "mae": mean_absolute_error(validation_data.targets, y_pred),
                    "r2": r2_score(validation_data.targets, y_pred),
                }

            # Update current model
            self.current_model = ensemble_model

            # Save snapshot
            if self.config.save_model_snapshots:
                snapshot = self._create_model_snapshot("ensemble_update", metrics)
                self.model_snapshots.append(snapshot)
                self._cleanup_snapshots()

            # Record training history
            self.training_history.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "strategy": "ensemble_update",
                    "n_samples": len(training_data.features),
                    "metrics": metrics,
                }
            )

            logger.info(f"Ensemble update completed. Metrics: {metrics}")
            return ensemble_model, metrics

        except Exception as e:
            logger.error(f"Error in ensemble update: {e}")
            raise

    def _create_model_snapshot(
        self, snapshot_type: str, metrics: Optional[Dict[str, float]] = None
    ) -> ModelSnapshot:
        """Create a snapshot of the current model."""
        model_state = copy.deepcopy(self.current_model) if self.current_model else None

        return ModelSnapshot(
            model_id=f"{snapshot_type}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(timezone.utc),
            model_state=model_state,
            performance_metrics=metrics or {},
            training_data_hash="",  # Would compute actual hash in production
            metadata={"snapshot_type": snapshot_type},
        )

    def _cleanup_snapshots(self):
        """Clean up old snapshots to maintain maximum count."""
        if len(self.model_snapshots) > self.config.max_snapshots:
            # Remove oldest snapshots
            self.model_snapshots.sort(key=lambda x: x.timestamp)
            self.model_snapshots = self.model_snapshots[-self.config.max_snapshots :]

    def rollback_to_snapshot(self, snapshot_id: str) -> bool:
        """Rollback to a previous model snapshot."""
        for snapshot in self.model_snapshots:
            if snapshot.model_id == snapshot_id:
                self.current_model = copy.deepcopy(snapshot.model_state)
                logger.info(f"Rolled back to snapshot: {snapshot_id}")
                return True

        logger.error(f"Snapshot not found: {snapshot_id}")
        return False

    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get training history."""
        return self.training_history.copy()

    def get_model_snapshots_info(self) -> List[Dict[str, Any]]:
        """Get information about model snapshots."""
        return [
            {
                "model_id": snapshot.model_id,
                "timestamp": snapshot.timestamp.isoformat(),
                "performance_metrics": snapshot.performance_metrics,
                "metadata": snapshot.metadata,
            }
            for snapshot in self.model_snapshots
        ]


class AdaptiveLearningManager:
    """Main manager for adaptive model retraining and fine-tuning."""

    def __init__(self, config: RetrainingConfig):
        self.config = config
        self.performance_monitor = PerformanceMonitor(config)
        self.data_balancer = DataBalancer(
            config.balancing_strategy, config.max_memory_samples
        )
        self.model_retrainer = ModelRetrainer(config)
        self.new_data_buffer: List[TrainingData] = []
        self.is_running = False
        self.background_task: Optional[asyncio.Task] = None
        self.retraining_in_progress = False

    def initialize_model(
        self,
        model: Any,
        scaler: Optional[Any] = None,
        initial_data: Optional[TrainingData] = None,
    ):
        """Initialize the adaptive learning system with a base model."""
        self.model_retrainer.set_current_model(model, scaler)

        if initial_data is not None:
            # Add initial data to memory buffer
            self.data_balancer.add_to_memory(initial_data, importance_score=1.0)

            # Set baseline performance
            if self.config.enable_performance_monitoring:
                metrics = self.performance_monitor.evaluate_model(model, initial_data)
                self.performance_monitor.update_baseline(metrics)

    async def add_new_data(self, data: TrainingData, importance_score: float = 1.0):
        """Add new training data and trigger retraining if necessary."""
        self.new_data_buffer.append(data)
        self.data_balancer.add_to_memory(data, importance_score)

        logger.info(f"Added new data batch with {len(data.features)} samples")

        # Check if retraining should be triggered
        total_new_samples = sum(len(d.features) for d in self.new_data_buffer)

        if self.performance_monitor.should_retrain(total_new_samples):
            await self.trigger_retraining()

    async def trigger_retraining(self):
        """Trigger model retraining process."""
        if self.retraining_in_progress:
            logger.warning("Retraining already in progress, skipping trigger")
            return

        if not self.new_data_buffer:
            logger.warning("No new data available for retraining")
            return

        self.retraining_in_progress = True

        try:
            logger.info("Starting adaptive retraining process")

            # Combine new data
            combined_features = np.vstack([d.features for d in self.new_data_buffer])
            combined_targets = np.concatenate([d.targets for d in self.new_data_buffer])
            combined_weights = None

            if all(d.sample_weights is not None for d in self.new_data_buffer):
                combined_weights = np.concatenate(
                    [d.sample_weights for d in self.new_data_buffer]
                )

            new_data = TrainingData(
                features=combined_features,
                targets=combined_targets,
                sample_weights=combined_weights,
                timestamp=datetime.now(timezone.utc),
            )

            # Balance with historical data
            balanced_data = self.data_balancer.balance_data(new_data)

            # Split for validation
            train_features, val_features, train_targets, val_targets = train_test_split(
                balanced_data.features,
                balanced_data.targets,
                test_size=self.config.validation_split,
                random_state=42,
            )

            train_data = TrainingData(features=train_features, targets=train_targets)
            val_data = TrainingData(features=val_features, targets=val_targets)

            # Perform retraining
            updated_model, metrics = await self.model_retrainer.retrain_model(
                train_data, val_data
            )

            # Update performance monitoring
            if self.config.enable_performance_monitoring:
                self.performance_monitor.evaluate_model(updated_model, val_data)

            # Clear new data buffer
            self.new_data_buffer.clear()

            logger.info(
                f"Adaptive retraining completed successfully. Metrics: {metrics}"
            )

        except Exception as e:
            logger.error(f"Error during adaptive retraining: {e}")
            raise
        finally:
            self.retraining_in_progress = False

    async def start_background_monitoring(self):
        """Start background monitoring and periodic retraining."""
        if self.is_running:
            logger.warning("Background monitoring already running")
            return

        self.is_running = True
        self.background_task = asyncio.create_task(self._background_monitoring_loop())
        logger.info("Started background adaptive learning monitoring")

    async def stop_background_monitoring(self):
        """Stop background monitoring."""
        if not self.is_running:
            return

        self.is_running = False
        if self.background_task:
            self.background_task.cancel()
            try:
                await self.background_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped background adaptive learning monitoring")

    async def _background_monitoring_loop(self):
        """Background loop for monitoring and periodic retraining."""
        while self.is_running:
            try:
                # Check if periodic retraining is needed
                total_new_samples = sum(len(d.features) for d in self.new_data_buffer)

                if self.performance_monitor.should_retrain(total_new_samples):
                    await self.trigger_retraining()

                # Wait for next check
                await asyncio.sleep(3600)  # Check every hour

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in background monitoring loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the adaptive learning system."""
        return {
            "is_running": self.is_running,
            "retraining_in_progress": self.retraining_in_progress,
            "config": {
                "strategy": self.config.strategy.value,
                "balancing_strategy": self.config.balancing_strategy.value,
                "min_new_samples": self.config.min_new_samples,
                "performance_threshold": self.config.performance_threshold,
            },
            "new_data_buffer_size": len(self.new_data_buffer),
            "total_new_samples": sum(len(d.features) for d in self.new_data_buffer),
            "performance_monitoring": self.performance_monitor.get_performance_summary(),
            "memory_stats": self.data_balancer.get_memory_stats(),
            "model_snapshots": len(self.model_retrainer.model_snapshots),
            "training_history_count": len(self.model_retrainer.training_history),
        }

    def get_current_model(self) -> Optional[Any]:
        """Get the current model."""
        return self.model_retrainer.current_model

    async def evaluate_current_model(self, test_data: TrainingData) -> Dict[str, float]:
        """Evaluate the current model on test data."""
        if not self.model_retrainer.current_model:
            raise ValueError("No current model available for evaluation")

        return self.performance_monitor.evaluate_model(
            self.model_retrainer.current_model, test_data
        )


# Factory Functions


def create_default_adaptive_learning_manager(
    strategy: RetrainingStrategy = RetrainingStrategy.INCREMENTAL,
    balancing_strategy: DataBalancingStrategy = DataBalancingStrategy.RESERVOIR_SAMPLING,
) -> AdaptiveLearningManager:
    """Create adaptive learning manager with default configuration."""
    config = RetrainingConfig(
        strategy=strategy,
        balancing_strategy=balancing_strategy,
        min_new_samples=50,
        performance_threshold=0.95,
        retrain_interval_hours=12,
        max_memory_samples=5000,
        enable_performance_monitoring=True,
        save_model_snapshots=True,
    )

    return AdaptiveLearningManager(config)


def create_continual_learning_manager() -> AdaptiveLearningManager:
    """Create adaptive learning manager optimized for continual learning."""
    config = RetrainingConfig(
        strategy=RetrainingStrategy.INCREMENTAL,
        balancing_strategy=DataBalancingStrategy.EXPERIENCE_REPLAY,
        min_new_samples=25,
        performance_threshold=0.92,
        retrain_interval_hours=6,
        max_memory_samples=10000,
        catastrophic_forgetting_threshold=0.05,
        enable_performance_monitoring=True,
        save_model_snapshots=True,
        max_snapshots=20,
    )

    return AdaptiveLearningManager(config)


def create_production_adaptive_manager() -> AdaptiveLearningManager:
    """Create adaptive learning manager for production environments."""
    config = RetrainingConfig(
        strategy=RetrainingStrategy.ENSEMBLE_UPDATE,
        balancing_strategy=DataBalancingStrategy.IMPORTANCE_WEIGHTED,
        min_new_samples=100,
        performance_threshold=0.98,
        retrain_interval_hours=24,
        max_memory_samples=20000,
        catastrophic_forgetting_threshold=0.02,
        enable_performance_monitoring=True,
        save_model_snapshots=True,
        max_snapshots=50,
        early_stopping_patience=20,
    )

    return AdaptiveLearningManager(config)
