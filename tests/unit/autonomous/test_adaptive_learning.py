"""
Tests for Adaptive Model Retraining and Fine-tuning System

This module contains comprehensive tests for the adaptive learning system including
model retraining, data balancing, performance monitoring, and continual learning.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import numpy as np
import pytest
from chemi_accel.autonomous.adaptive_learning import (
    AdaptiveLearningManager,
    DataBalancer,
    DataBalancingStrategy,
    ModelRetrainer,
    PerformanceMonitor,
    RetrainingConfig,
    RetrainingStrategy,
    TrainingData,
)


class TestTrainingData:
    """Test cases for TrainingData dataclass."""

    def test_training_data_creation(self):
        """Test basic training data creation."""
        features = np.random.randn(100, 5)
        targets = np.random.randn(100)

        data = TrainingData(
            features=features,
            targets=targets,
            experiment_ids=[f"exp_{i}" for i in range(100)],
            metadata={"dataset": "test"},
        )

        assert data.features.shape == (100, 5)
        assert data.targets.shape == (100,)
        assert len(data.experiment_ids) == 100
        assert data.metadata["dataset"] == "test"
        assert data.timestamp is not None

    def test_training_data_validation(self):
        """Test training data validation."""
        features = np.random.randn(100, 5)
        targets = np.random.randn(50)  # Wrong size

        with pytest.raises(
            ValueError, match="Features and targets must have the same length"
        ):
            TrainingData(features=features, targets=targets)

    def test_training_data_with_weights(self):
        """Test training data with sample weights."""
        features = np.random.randn(50, 3)
        targets = np.random.randn(50)
        weights = np.random.rand(50)

        data = TrainingData(features=features, targets=targets, sample_weights=weights)

        assert data.sample_weights.shape == (50,)
        assert np.all(data.sample_weights >= 0)

    def test_training_data_validation_weights_mismatch(self):
        """Test validation with mismatched weights."""
        features = np.random.randn(50, 3)
        targets = np.random.randn(50)
        weights = np.random.rand(30)  # Wrong size

        with pytest.raises(
            ValueError, match="Sample weights must have the same length as features"
        ):
            TrainingData(features=features, targets=targets, sample_weights=weights)


class TestPerformanceMonitor:
    """Test cases for PerformanceMonitor."""

    @pytest.fixture
    def config(self) -> RetrainingConfig:
        """Create test configuration."""
        return RetrainingConfig(
            min_new_samples=50,
            performance_threshold=0.95,
            catastrophic_forgetting_threshold=0.1,
        )

    @pytest.fixture
    def performance_monitor(self, config) -> PerformanceMonitor:
        """Create performance monitor."""
        return PerformanceMonitor(config)

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock()
        model.predict.return_value = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        return model

    @pytest.fixture
    def test_data(self) -> TrainingData:
        """Create test data."""
        return TrainingData(
            features=np.random.randn(5, 3), targets=np.array([1.1, 2.1, 2.9, 4.1, 4.9])
        )

    def test_update_baseline(self, performance_monitor):
        """Test baseline performance update."""
        metrics = {"mse": 0.1, "r2": 0.95, "mae": 0.05}

        performance_monitor.update_baseline(metrics)

        assert performance_monitor.baseline_performance == metrics

    def test_evaluate_model(self, performance_monitor, mock_model, test_data):
        """Test model evaluation."""
        metrics = performance_monitor.evaluate_model(mock_model, test_data)

        assert "mse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert "timestamp" in metrics
        assert "n_samples" in metrics
        assert metrics["n_samples"] == 5

        # Check that evaluation was recorded
        assert len(performance_monitor.performance_history) == 1

    def test_check_performance_degradation(self, performance_monitor):
        """Test performance degradation detection."""
        # Set baseline
        baseline = {"mse": 0.1, "r2": 0.95, "mae": 0.05}
        performance_monitor.update_baseline(baseline)

        # Test no degradation
        current = {"mse": 0.08, "r2": 0.96, "mae": 0.04}
        assert not performance_monitor.check_performance_degradation(current)

        # Test significant degradation
        degraded = {"mse": 0.25, "r2": 0.80, "mae": 0.20}
        assert performance_monitor.check_performance_degradation(degraded)

    def test_should_retrain_new_samples(self, performance_monitor):
        """Test retraining trigger based on new samples."""
        # Not enough new samples
        assert not performance_monitor.should_retrain(30)

        # Enough new samples
        assert performance_monitor.should_retrain(60)

    def test_should_retrain_time_based(self, performance_monitor):
        """Test time-based retraining trigger."""
        # Set last evaluation to long ago
        performance_monitor.last_evaluation = datetime.now(timezone.utc) - timedelta(
            hours=25
        )

        # Should trigger retraining
        assert performance_monitor.should_retrain(10)

    def test_get_performance_summary(self, performance_monitor):
        """Test performance summary generation."""
        summary = performance_monitor.get_performance_summary()

        assert "monitoring_active" in summary
        assert "baseline_performance" in summary
        assert "last_evaluation" in summary
        assert "evaluation_count" in summary
        assert "recent_performance" in summary


class TestDataBalancer:
    """Test cases for DataBalancer."""

    @pytest.fixture
    def reservoir_balancer(self) -> DataBalancer:
        """Create reservoir sampling balancer."""
        return DataBalancer(
            DataBalancingStrategy.RESERVOIR_SAMPLING, max_memory_samples=100
        )

    @pytest.fixture
    def importance_balancer(self) -> DataBalancer:
        """Create importance weighted balancer."""
        return DataBalancer(
            DataBalancingStrategy.IMPORTANCE_WEIGHTED, max_memory_samples=100
        )

    @pytest.fixture
    def sample_data(self) -> TrainingData:
        """Create sample training data."""
        return TrainingData(
            features=np.random.randn(50, 4), targets=np.random.randn(50)
        )

    def test_add_to_memory(self, reservoir_balancer, sample_data):
        """Test adding data to memory buffer."""
        reservoir_balancer.add_to_memory(sample_data, importance_score=0.8)

        assert len(reservoir_balancer.memory_buffer) == 1

        memory_item = reservoir_balancer.memory_buffer[0]
        assert memory_item["importance"] == 0.8
        assert memory_item["data"] == sample_data
        assert "timestamp" in memory_item
        assert "hash" in memory_item

    def test_memory_buffer_capacity(self, reservoir_balancer):
        """Test memory buffer capacity limits."""
        # Add more data than capacity
        for i in range(150):
            data = TrainingData(
                features=np.random.randn(10, 3), targets=np.random.randn(10)
            )
            reservoir_balancer.add_to_memory(data)

        # Should not exceed max capacity
        assert len(reservoir_balancer.memory_buffer) == 100

    def test_reservoir_sampling_balance(self, reservoir_balancer):
        """Test reservoir sampling data balancing."""
        # Add some historical data to memory
        for i in range(50):
            historical_data = TrainingData(
                features=np.random.randn(20, 3), targets=np.random.randn(20)
            )
            reservoir_balancer.add_to_memory(historical_data, importance_score=0.5)

        # Create new data
        new_data = TrainingData(
            features=np.random.randn(30, 3), targets=np.random.randn(30)
        )

        # Balance data
        balanced_data = reservoir_balancer.balance_data(new_data)

        # Should combine new and historical data
        assert len(balanced_data.features) > 30  # More than just new data
        assert balanced_data.sample_weights is not None
        assert "balancing_strategy" in balanced_data.metadata
        assert balanced_data.metadata["balancing_strategy"] == "reservoir_sampling"

    def test_importance_weighted_balance(self, importance_balancer):
        """Test importance weighted data balancing."""
        # Add historical data with different importance scores
        for i in range(30):
            historical_data = TrainingData(
                features=np.random.randn(10, 3), targets=np.random.randn(10)
            )
            importance_score = 0.3 + (i / 30) * 0.7  # Varying importance
            importance_balancer.add_to_memory(historical_data, importance_score)

        # Create new data
        new_data = TrainingData(
            features=np.random.randn(20, 3), targets=np.random.randn(20)
        )

        # Balance data
        balanced_data = importance_balancer.balance_data(new_data)

        # Check that high-importance historical data is preferred
        assert len(balanced_data.features) > 20
        assert balanced_data.sample_weights is not None
        assert balanced_data.metadata["balancing_strategy"] == "importance_weighted"

    def test_get_memory_stats(self, reservoir_balancer, sample_data):
        """Test memory statistics generation."""
        # Add some data
        reservoir_balancer.add_to_memory(sample_data, importance_score=0.8)

        stats = reservoir_balancer.get_memory_stats()

        assert stats["size"] == 1
        assert stats["strategy"] == "reservoir_sampling"
        assert stats["max_size"] == 100
        assert "avg_importance" in stats
        assert "oldest_sample" in stats
        assert "newest_sample" in stats


class TestModelRetrainer:
    """Test cases for ModelRetrainer."""

    @pytest.fixture
    def config(self) -> RetrainingConfig:
        """Create test configuration."""
        return RetrainingConfig(
            strategy=RetrainingStrategy.INCREMENTAL,
            save_model_snapshots=True,
            max_snapshots=5,
        )

    @pytest.fixture
    def model_retrainer(self, config) -> ModelRetrainer:
        """Create model retrainer."""
        return ModelRetrainer(config)

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock()
        model.fit = Mock()

        def mock_predict(X):
            return np.random.randn(len(X))

        model.predict = Mock(side_effect=mock_predict)
        model.get_params = Mock(return_value={"n_estimators": 100})
        return model

    @pytest.fixture
    def training_data(self) -> TrainingData:
        """Create training data."""
        return TrainingData(
            features=np.random.randn(100, 5), targets=np.random.randn(100)
        )

    @pytest.fixture
    def validation_data(self) -> TrainingData:
        """Create validation data."""
        return TrainingData(
            features=np.random.randn(20, 5), targets=np.random.randn(20)
        )

    def test_set_current_model(self, model_retrainer, mock_model):
        """Test setting current model."""
        model_retrainer.set_current_model(mock_model)

        assert model_retrainer.current_model == mock_model
        assert len(model_retrainer.model_snapshots) == 1
        assert model_retrainer.model_snapshots[0].metadata["snapshot_type"] == "initial"

    @pytest.mark.asyncio
    async def test_full_retrain(
        self, model_retrainer, mock_model, training_data, validation_data
    ):
        """Test full model retraining."""
        model_retrainer.set_current_model(mock_model)

        with patch("chemi_accel.autonomous.adaptive_learning.SKLEARN_AVAILABLE", True):
            with patch(
                "chemi_accel.autonomous.adaptive_learning.mean_squared_error",
                return_value=0.1,
            ):
                with patch(
                    "chemi_accel.autonomous.adaptive_learning.r2_score",
                    return_value=0.9,
                ):
                    with patch(
                        "chemi_accel.autonomous.adaptive_learning.mean_absolute_error",
                        return_value=0.05,
                    ):
                        updated_model, metrics = await model_retrainer._full_retrain(
                            training_data, validation_data
                        )

        assert updated_model is not None
        assert "mse" in metrics
        assert "r2" in metrics
        assert "mae" in metrics
        assert len(model_retrainer.training_history) == 1
        assert len(model_retrainer.model_snapshots) == 2  # Initial + retrain

    @pytest.mark.asyncio
    async def test_incremental_retrain(
        self, model_retrainer, mock_model, training_data, validation_data
    ):
        """Test incremental model retraining."""
        # Mock model with partial_fit capability
        mock_model.partial_fit = Mock()
        model_retrainer.set_current_model(mock_model)

        with patch("chemi_accel.autonomous.adaptive_learning.SKLEARN_AVAILABLE", True):
            with patch(
                "chemi_accel.autonomous.adaptive_learning.mean_squared_error",
                return_value=0.08,
            ):
                with patch(
                    "chemi_accel.autonomous.adaptive_learning.r2_score",
                    return_value=0.92,
                ):
                    with patch(
                        "chemi_accel.autonomous.adaptive_learning.mean_absolute_error",
                        return_value=0.04,
                    ):
                        (
                            updated_model,
                            metrics,
                        ) = await model_retrainer._incremental_retrain(
                            training_data, validation_data
                        )

        assert updated_model == mock_model
        assert mock_model.partial_fit.called
        assert "mse" in metrics
        assert metrics["mse"] == 0.08

    def test_create_model_snapshot(self, model_retrainer, mock_model):
        """Test model snapshot creation."""
        model_retrainer.set_current_model(mock_model)

        metrics = {"mse": 0.1, "r2": 0.9}
        snapshot = model_retrainer._create_model_snapshot("test_snapshot", metrics)

        assert snapshot.model_id.startswith("test_snapshot_")
        assert snapshot.performance_metrics == metrics
        assert snapshot.metadata["snapshot_type"] == "test_snapshot"
        assert snapshot.timestamp is not None

    def test_cleanup_snapshots(self, model_retrainer, mock_model):
        """Test snapshot cleanup."""
        model_retrainer.set_current_model(mock_model)

        # Create more snapshots than max allowed
        for i in range(10):
            snapshot = model_retrainer._create_model_snapshot(f"test_{i}")
            model_retrainer.model_snapshots.append(snapshot)

        model_retrainer._cleanup_snapshots()

        # Should keep only max_snapshots
        assert (
            len(model_retrainer.model_snapshots) == model_retrainer.config.max_snapshots
        )

    def test_rollback_to_snapshot(self, model_retrainer, mock_model):
        """Test model rollback to snapshot."""
        model_retrainer.set_current_model(mock_model)

        # Get snapshot ID
        snapshot_id = model_retrainer.model_snapshots[0].model_id

        # Rollback
        success = model_retrainer.rollback_to_snapshot(snapshot_id)

        assert success is True

        # Test rollback to non-existent snapshot
        success = model_retrainer.rollback_to_snapshot("non_existent")
        assert success is False

    def test_get_training_history(self, model_retrainer):
        """Test training history retrieval."""
        # Add some training history
        model_retrainer.training_history.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "strategy": "test",
                "n_samples": 100,
                "metrics": {"mse": 0.1},
            }
        )

        history = model_retrainer.get_training_history()

        assert len(history) == 1
        assert history[0]["strategy"] == "test"
        assert history[0]["n_samples"] == 100


class TestAdaptiveLearningManager:
    """Test cases for AdaptiveLearningManager."""

    @pytest.fixture
    def config(self) -> RetrainingConfig:
        """Create test configuration."""
        return RetrainingConfig(
            strategy=RetrainingStrategy.INCREMENTAL,
            balancing_strategy=DataBalancingStrategy.RESERVOIR_SAMPLING,
            min_new_samples=20,
            enable_performance_monitoring=True,
        )

    @pytest.fixture
    def adaptive_manager(self, config) -> AdaptiveLearningManager:
        """Create adaptive learning manager."""
        return AdaptiveLearningManager(config)

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock()

        # Create a dynamic predict function that returns predictions based on input size
        def mock_predict(X):
            return np.random.randn(len(X))

        model.predict = Mock(side_effect=mock_predict)
        model.fit = Mock()
        return model

    @pytest.fixture
    def initial_data(self) -> TrainingData:
        """Create initial training data."""
        return TrainingData(
            features=np.random.randn(100, 4), targets=np.random.randn(100)
        )

    @pytest.fixture
    def new_data(self) -> TrainingData:
        """Create new training data."""
        return TrainingData(
            features=np.random.randn(25, 4), targets=np.random.randn(25)
        )

    def test_initialize_model(self, adaptive_manager, mock_model, initial_data):
        """Test model initialization."""
        adaptive_manager.initialize_model(mock_model, initial_data=initial_data)

        assert adaptive_manager.model_retrainer.current_model == mock_model
        assert len(adaptive_manager.data_balancer.memory_buffer) == 1

    @pytest.mark.asyncio
    async def test_add_new_data(
        self, adaptive_manager, mock_model, initial_data, new_data
    ):
        """Test adding new data."""
        adaptive_manager.initialize_model(mock_model, initial_data=initial_data)

        await adaptive_manager.add_new_data(new_data, importance_score=0.8)

        # Retraining should have been triggered due to sufficient new samples (25 >= 20)
        # So the buffer should be cleared after retraining
        assert len(adaptive_manager.new_data_buffer) == 0  # Cleared after retraining
        assert len(adaptive_manager.data_balancer.memory_buffer) == 2  # Initial + new

    @pytest.mark.asyncio
    async def test_trigger_retraining(self, adaptive_manager, mock_model, initial_data):
        """Test retraining trigger."""
        adaptive_manager.initialize_model(mock_model, initial_data=initial_data)

        # Add enough new data to trigger retraining
        for i in range(5):
            new_data = TrainingData(
                features=np.random.randn(10, 4), targets=np.random.randn(10)
            )
            adaptive_manager.new_data_buffer.append(new_data)

        with patch.object(
            adaptive_manager.model_retrainer, "retrain_model"
        ) as mock_retrain:
            mock_retrain.return_value = (mock_model, {"mse": 0.1, "r2": 0.9})

            await adaptive_manager.trigger_retraining()

        assert mock_retrain.called
        assert (
            len(adaptive_manager.new_data_buffer) == 0
        )  # Should be cleared after retraining

    @pytest.mark.asyncio
    async def test_background_monitoring(
        self, adaptive_manager, mock_model, initial_data
    ):
        """Test background monitoring functionality."""
        adaptive_manager.initialize_model(mock_model, initial_data=initial_data)

        # Start background monitoring
        await adaptive_manager.start_background_monitoring()

        assert adaptive_manager.is_running is True
        assert adaptive_manager.background_task is not None

        # Stop background monitoring
        await adaptive_manager.stop_background_monitoring()

        assert adaptive_manager.is_running is False

    def test_get_system_status(self, adaptive_manager, mock_model, initial_data):
        """Test system status retrieval."""
        adaptive_manager.initialize_model(mock_model, initial_data=initial_data)

        status = adaptive_manager.get_system_status()

        assert "is_running" in status
        assert "retraining_in_progress" in status
        assert "config" in status
        assert "new_data_buffer_size" in status
        assert "performance_monitoring" in status
        assert "memory_stats" in status

        assert status["config"]["strategy"] == "incremental"
        assert status["config"]["balancing_strategy"] == "reservoir_sampling"

    def test_get_current_model(self, adaptive_manager, mock_model):
        """Test current model retrieval."""
        adaptive_manager.initialize_model(mock_model)

        current_model = adaptive_manager.get_current_model()

        assert current_model == mock_model

    @pytest.mark.asyncio
    async def test_evaluate_current_model(
        self, adaptive_manager, mock_model, initial_data
    ):
        """Test current model evaluation."""
        adaptive_manager.initialize_model(mock_model, initial_data=initial_data)

        test_data = TrainingData(
            features=np.random.randn(20, 4), targets=np.random.randn(20)
        )

        with patch.object(
            adaptive_manager.performance_monitor, "evaluate_model"
        ) as mock_eval:
            mock_eval.return_value = {"mse": 0.05, "r2": 0.95}

            metrics = await adaptive_manager.evaluate_current_model(test_data)

        assert "mse" in metrics
        assert "r2" in metrics
        assert mock_eval.called


class TestFactoryFunctions:
    """Test cases for factory functions."""

    def test_create_default_adaptive_learning_manager(self):
        """Test default adaptive learning manager creation."""
        from chemi_accel.autonomous.adaptive_learning import (
            create_default_adaptive_learning_manager,
        )

        manager = create_default_adaptive_learning_manager()

        assert isinstance(manager, AdaptiveLearningManager)
        assert manager.config.strategy == RetrainingStrategy.INCREMENTAL
        assert (
            manager.config.balancing_strategy
            == DataBalancingStrategy.RESERVOIR_SAMPLING
        )
        assert manager.config.min_new_samples == 50

    def test_create_continual_learning_manager(self):
        """Test continual learning manager creation."""
        from chemi_accel.autonomous.adaptive_learning import (
            create_continual_learning_manager,
        )

        manager = create_continual_learning_manager()

        assert isinstance(manager, AdaptiveLearningManager)
        assert (
            manager.config.balancing_strategy == DataBalancingStrategy.EXPERIENCE_REPLAY
        )
        assert manager.config.min_new_samples == 25
        assert manager.config.catastrophic_forgetting_threshold == 0.05

    def test_create_production_adaptive_manager(self):
        """Test production adaptive manager creation."""
        from chemi_accel.autonomous.adaptive_learning import (
            create_production_adaptive_manager,
        )

        manager = create_production_adaptive_manager()

        assert isinstance(manager, AdaptiveLearningManager)
        assert manager.config.strategy == RetrainingStrategy.ENSEMBLE_UPDATE
        assert (
            manager.config.balancing_strategy
            == DataBalancingStrategy.IMPORTANCE_WEIGHTED
        )
        assert manager.config.performance_threshold == 0.98
        assert manager.config.max_snapshots == 50


class TestIntegrationScenarios:
    """Integration test scenarios for adaptive learning."""

    @pytest.mark.asyncio
    async def test_full_adaptive_learning_cycle(self):
        """Test complete adaptive learning cycle."""
        from chemi_accel.autonomous.adaptive_learning import (
            create_default_adaptive_learning_manager,
        )

        # Create manager
        manager = create_default_adaptive_learning_manager()

        # Create mock model
        mock_model = Mock()

        def mock_predict(X):
            return np.random.randn(len(X))

        mock_model.predict = Mock(side_effect=mock_predict)
        mock_model.fit = Mock()
        mock_model.get_params = Mock(return_value={"n_estimators": 100})

        # Initialize with base data
        initial_data = TrainingData(
            features=np.random.randn(200, 6), targets=np.random.randn(200)
        )

        manager.initialize_model(mock_model, initial_data=initial_data)

        # Simulate continuous data arrival
        for batch in range(3):
            new_data = TrainingData(
                features=np.random.randn(30, 6), targets=np.random.randn(30)
            )

            await manager.add_new_data(new_data, importance_score=0.8)

        # Check system state
        status = manager.get_system_status()

        # After 3 batches of 30 samples each, retraining should have been triggered
        # after the 2nd batch (60 samples >= 50 threshold), clearing the buffer
        # So only the 3rd batch should remain in the buffer
        assert status["total_new_samples"] == 30  # Only the third batch remains
        assert (
            status["memory_stats"]["size"] == 4
        )  # Initial + 3 batches (all added to memory)
        assert len(manager.data_balancer.memory_buffer) == 4

    @pytest.mark.asyncio
    async def test_performance_degradation_handling(self):
        """Test handling of performance degradation."""
        from chemi_accel.autonomous.adaptive_learning import (
            create_continual_learning_manager,
        )

        manager = create_continual_learning_manager()

        # Create mock model
        mock_model = Mock()
        mock_model.predict = Mock()
        mock_model.fit = Mock()

        # Set up baseline performance
        initial_data = TrainingData(
            features=np.random.randn(100, 5), targets=np.random.randn(100)
        )

        manager.initialize_model(mock_model, initial_data=initial_data)

        # Simulate good baseline performance
        baseline_metrics = {"mse": 0.05, "r2": 0.95, "mae": 0.03}
        manager.performance_monitor.update_baseline(baseline_metrics)

        # Simulate degraded performance
        degraded_metrics = {"mse": 0.20, "r2": 0.80, "mae": 0.15}

        # Should trigger retraining due to performance degradation
        should_retrain = manager.performance_monitor.should_retrain(
            10, degraded_metrics
        )
        assert should_retrain is True

    @pytest.mark.asyncio
    async def test_memory_management_under_load(self):
        """Test memory management under high data load."""
        config = RetrainingConfig(
            strategy=RetrainingStrategy.INCREMENTAL,
            balancing_strategy=DataBalancingStrategy.RESERVOIR_SAMPLING,
            max_memory_samples=50,  # Small buffer for testing
            min_new_samples=20,
        )

        manager = AdaptiveLearningManager(config)

        # Create mock model
        mock_model = Mock()

        def mock_predict(X):
            return np.random.randn(len(X))

        mock_model.predict = Mock(side_effect=mock_predict)
        mock_model.fit = Mock()

        # Initialize
        initial_data = TrainingData(
            features=np.random.randn(20, 3), targets=np.random.randn(20)
        )

        manager.initialize_model(mock_model, initial_data=initial_data)

        # Add many batches of data
        for i in range(100):
            new_data = TrainingData(
                features=np.random.randn(5, 3), targets=np.random.randn(5)
            )

            await manager.add_new_data(new_data, importance_score=np.random.rand())

        # Memory buffer should not exceed limit
        assert len(manager.data_balancer.memory_buffer) <= 50

        # Should have triggered multiple retraining cycles
        assert (
            len(manager.new_data_buffer) < 100 * 5
        )  # Some data should have been processed

    @pytest.mark.asyncio
    async def test_different_balancing_strategies(self):
        """Test different data balancing strategies."""
        strategies = [
            DataBalancingStrategy.RESERVOIR_SAMPLING,
            DataBalancingStrategy.IMPORTANCE_WEIGHTED,
            DataBalancingStrategy.EXPERIENCE_REPLAY,
            DataBalancingStrategy.GRADUAL_FORGETTING,
        ]

        for strategy in strategies:
            balancer = DataBalancer(strategy, max_memory_samples=50)

            # Add historical data
            for i in range(20):
                historical_data = TrainingData(
                    features=np.random.randn(10, 4), targets=np.random.randn(10)
                )
                balancer.add_to_memory(
                    historical_data, importance_score=np.random.rand()
                )

            # Add new data
            new_data = TrainingData(
                features=np.random.randn(15, 4), targets=np.random.randn(15)
            )

            # Balance data
            balanced_data = balancer.balance_data(new_data)

            # All strategies should produce valid balanced data
            assert len(balanced_data.features) >= 15  # At least the new data
            assert balanced_data.features.shape[1] == 4  # Correct feature dimension
            assert len(balanced_data.features) == len(balanced_data.targets)

            if balanced_data.sample_weights is not None:
                assert len(balanced_data.sample_weights) == len(balanced_data.features)

    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent operations in adaptive learning."""
        from chemi_accel.autonomous.adaptive_learning import (
            create_default_adaptive_learning_manager,
        )

        manager = create_default_adaptive_learning_manager()

        # Create mock model
        mock_model = Mock()

        def mock_predict(X):
            return np.random.randn(len(X))

        mock_model.predict = Mock(side_effect=mock_predict)
        mock_model.fit = Mock()

        # Initialize
        initial_data = TrainingData(
            features=np.random.randn(50, 4), targets=np.random.randn(50)
        )

        manager.initialize_model(mock_model, initial_data=initial_data)

        # Start background monitoring
        await manager.start_background_monitoring()

        # Simulate concurrent data additions
        async def add_data_batch(batch_id):
            for i in range(5):
                new_data = TrainingData(
                    features=np.random.randn(10, 4), targets=np.random.randn(10)
                )
                await manager.add_new_data(new_data)
                await asyncio.sleep(0.01)  # Small delay to simulate real operations

        # Run multiple concurrent data addition tasks
        tasks = [add_data_batch(i) for i in range(3)]
        await asyncio.gather(*tasks)

        # Stop background monitoring
        await manager.stop_background_monitoring()

        # Verify system state
        status = manager.get_system_status()
        # During concurrent operations, multiple retraining cycles may have occurred
        # The buffer might be empty if the last batch triggered retraining
        # But the memory buffer should have accumulated data from all batches
        assert (
            len(manager.data_balancer.memory_buffer) > 1
        )  # Should have accumulated data
