"""
Tests for QML memory management system.
"""

import time
from unittest.mock import Mock, patch

import numpy as np
from chemi_accel.qml.memory_manager import (
    BatchingStrategy,
    BatchMetrics,
    MemoryAwareProcessor,
    MemoryMonitor,
    MemoryProfile,
    MemoryStrategy,
    SmartBatcher,
)
from chemi_accel.qml.on_the_fly import GeometryInput


class TestMemoryProfile:
    """Test MemoryProfile dataclass."""

    def test_initialization(self):
        """Test memory profile initialization."""
        profile = MemoryProfile(
            system_total=16.0,
            system_available=8.0,
            gpu_total=8.0,
            gpu_available=6.0,
            process_memory=2.0,
            peak_memory=4.0,
        )

        assert profile.system_total == 16.0
        assert profile.system_available == 8.0
        assert profile.gpu_total == 8.0
        assert profile.gpu_available == 6.0
        assert profile.process_memory == 2.0
        assert profile.peak_memory == 4.0
        assert isinstance(profile.timestamp, float)


class TestBatchMetrics:
    """Test BatchMetrics dataclass."""

    def test_initialization(self):
        """Test batch metrics initialization."""
        metrics = BatchMetrics(
            batch_size=32,
            memory_usage=1.5,
            processing_time=2.0,
            throughput=16.0,
            molecules_per_gb=10.67,
            success_rate=0.95,
            avg_molecular_size=25.0,
        )

        assert metrics.batch_size == 32
        assert metrics.memory_usage == 1.5
        assert metrics.processing_time == 2.0
        assert metrics.throughput == 16.0
        assert metrics.molecules_per_gb == 10.67
        assert metrics.success_rate == 0.95
        assert metrics.avg_molecular_size == 25.0


class TestMemoryMonitor:
    """Test MemoryMonitor class."""

    def test_initialization(self):
        """Test monitor initialization."""
        monitor = MemoryMonitor(
            monitoring_interval=0.5, warning_threshold=0.8, critical_threshold=0.9
        )

        assert monitor.monitoring_interval == 0.5
        assert monitor.warning_threshold == 0.8
        assert monitor.critical_threshold == 0.9
        assert monitor._monitoring == False

    @patch("chemi_accel.qml.memory_manager.psutil")
    def test_get_current_profile(self, mock_psutil):
        """Test current profile retrieval."""
        monitor = MemoryMonitor()

        # Mock system memory
        mock_vmem = Mock()
        mock_vmem.total = 16 * 1024**3  # 16 GB
        mock_vmem.available = 8 * 1024**3  # 8 GB
        mock_psutil.virtual_memory.return_value = mock_vmem

        # Mock process memory
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 2 * 1024**3  # 2 GB
        mock_psutil.Process.return_value = mock_process

        with patch("chemi_accel.qml.memory_manager.TORCH_AVAILABLE", False):
            profile = monitor.get_current_profile()

            assert profile.system_total == 16.0
            assert profile.system_available == 8.0
            assert profile.process_memory == 2.0
            assert profile.gpu_total == 0.0
            assert profile.gpu_available == 0.0

    @patch("chemi_accel.qml.memory_manager.torch")
    @patch("chemi_accel.qml.memory_manager.psutil")
    def test_get_current_profile_with_gpu(self, mock_psutil, mock_torch):
        """Test current profile with GPU information."""
        monitor = MemoryMonitor()

        # Mock system memory
        mock_vmem = Mock()
        mock_vmem.total = 16 * 1024**3
        mock_vmem.available = 8 * 1024**3
        mock_psutil.virtual_memory.return_value = mock_vmem

        # Mock process memory
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 2 * 1024**3
        mock_psutil.Process.return_value = mock_process

        # Mock GPU
        mock_torch.cuda.is_available.return_value = True
        mock_props = Mock()
        mock_props.total_memory = 8 * 1024**3  # 8 GB
        mock_torch.cuda.get_device_properties.return_value = mock_props
        mock_torch.cuda.memory_allocated.return_value = 2 * 1024**3  # 2 GB used

        with patch("chemi_accel.qml.memory_manager.TORCH_AVAILABLE", True):
            profile = monitor.get_current_profile()

            assert profile.gpu_total == 8.0
            assert profile.gpu_available == 6.0

    def test_callback_registration(self):
        """Test callback registration."""
        monitor = MemoryMonitor()

        callback_called = []

        def test_callback(profile):
            callback_called.append(profile)

        monitor.register_callback("warning", test_callback)

        assert "warning" in monitor._callbacks
        assert test_callback in monitor._callbacks["warning"]

    def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring."""
        monitor = MemoryMonitor(monitoring_interval=0.1)

        assert monitor._monitoring == False
        assert monitor._monitor_thread is None

        monitor.start_monitoring()
        assert monitor._monitoring == True
        assert monitor._monitor_thread is not None

        # Let it run briefly
        time.sleep(0.2)

        monitor.stop_monitoring()
        assert monitor._monitoring == False


class TestSmartBatcher:
    """Test SmartBatcher class."""

    def test_initialization(self):
        """Test batcher initialization."""
        batcher = SmartBatcher(
            initial_batch_size=16,
            min_batch_size=4,
            max_batch_size=128,
            strategy=BatchingStrategy.DYNAMIC,
        )

        assert batcher.current_batch_size == 16
        assert batcher.min_batch_size == 4
        assert batcher.max_batch_size == 128
        assert batcher.strategy == BatchingStrategy.DYNAMIC

    def test_fixed_batch_size(self):
        """Test fixed batching strategy."""
        batcher = SmartBatcher(initial_batch_size=32, strategy=BatchingStrategy.FIXED)

        molecules = ["CCO"] * 50
        memory_profile = MemoryProfile(
            system_total=16.0,
            system_available=8.0,
            gpu_total=0.0,
            gpu_available=0.0,
            process_memory=2.0,
            peak_memory=2.0,
        )

        batch_size = batcher.calculate_optimal_batch_size(molecules, memory_profile)
        assert batch_size == min(32, len(molecules))

    def test_dynamic_batch_size(self):
        """Test dynamic batching strategy."""
        batcher = SmartBatcher(
            initial_batch_size=32,
            strategy=BatchingStrategy.DYNAMIC,
            memory_fraction=0.5,
        )

        molecules = ["CCO"] * 100
        memory_profile = MemoryProfile(
            system_total=16.0,
            system_available=8.0,
            gpu_total=0.0,
            gpu_available=0.0,
            process_memory=2.0,
            peak_memory=2.0,
        )

        batch_size = batcher.calculate_optimal_batch_size(molecules, memory_profile)

        # Should be reasonable size based on available memory
        assert batcher.min_batch_size <= batch_size <= batcher.max_batch_size
        assert batch_size <= len(molecules)

    def test_molecular_size_batch_size(self):
        """Test molecular size-based batching."""
        batcher = SmartBatcher(strategy=BatchingStrategy.MOLECULAR_SIZE)

        # Mock molecules with different complexities
        simple_molecules = ["CC"] * 20  # Simple molecules
        complex_molecules = ["CC(C)CC(C(C)C)CC(C)C"] * 20  # Complex molecules

        memory_profile = MemoryProfile(
            system_total=16.0,
            system_available=8.0,
            gpu_total=0.0,
            gpu_available=0.0,
            process_memory=2.0,
            peak_memory=2.0,
        )

        simple_batch_size = batcher.calculate_optimal_batch_size(
            simple_molecules, memory_profile
        )
        complex_batch_size = batcher.calculate_optimal_batch_size(
            complex_molecules, memory_profile
        )

        # Should handle complex molecules with smaller batches
        assert complex_batch_size <= simple_batch_size

    def test_estimate_molecule_complexity(self):
        """Test molecular complexity estimation."""
        batcher = SmartBatcher()

        # Test with SMILES strings
        simple_molecules = ["CC", "CCO"]
        complex_molecules = ["CC(C)CC(C(C)C)CC(C)C", "CCCCCCCCCCCCCCCC"]

        simple_complexity = batcher._estimate_molecule_complexity(simple_molecules)
        complex_complexity = batcher._estimate_molecule_complexity(complex_molecules)

        assert complex_complexity > simple_complexity

        # Test with GeometryInput
        geom_molecules = [
            GeometryInput(
                coordinates=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                atomic_numbers=np.array([6, 1]),
            ),
            GeometryInput(
                coordinates=np.array([[0.0, 0.0, 0.0]]), atomic_numbers=np.array([6])
            ),
        ]

        geom_complexity = batcher._estimate_molecule_complexity(geom_molecules)
        assert geom_complexity > 0

    def test_performance_update(self):
        """Test performance metrics update."""
        batcher = SmartBatcher(strategy=BatchingStrategy.GRADIENT)

        # Add some performance history
        metrics1 = BatchMetrics(
            batch_size=32,
            memory_usage=1.0,
            processing_time=2.0,
            throughput=16.0,
            molecules_per_gb=32.0,
            success_rate=1.0,
            avg_molecular_size=20.0,
        )

        initial_batch_size = batcher.current_batch_size
        batcher.update_performance(metrics1)

        # Should have updated history
        assert len(batcher._batch_history) == 1

        # Test with poor performance (should reduce batch size)
        metrics2 = BatchMetrics(
            batch_size=32,
            memory_usage=1.0,
            processing_time=2.0,
            throughput=8.0,  # Lower throughput
            molecules_per_gb=32.0,
            success_rate=0.5,  # Poor success rate
            avg_molecular_size=20.0,
        )

        batcher.update_performance(metrics2)

        # Should reduce batch size due to poor success rate
        assert batcher.current_batch_size <= initial_batch_size


class TestMemoryAwareProcessor:
    """Test MemoryAwareProcessor class."""

    def test_initialization(self):
        """Test processor initialization."""
        processor = MemoryAwareProcessor(
            strategy=MemoryStrategy.BALANCED, cache_size_gb=2.0, gc_threshold=0.7
        )

        assert processor.strategy == MemoryStrategy.BALANCED
        assert processor.cache_size_gb == 2.0
        assert processor.gc_threshold == 0.7
        assert isinstance(processor.monitor, MemoryMonitor)
        assert isinstance(processor.batcher, SmartBatcher)

    def test_memory_managed_processing_context(self):
        """Test memory managed processing context manager."""
        processor = MemoryAwareProcessor()

        with processor.memory_managed_processing(estimated_memory_gb=1.0) as ctx:
            assert ctx is processor

        # Context should complete without errors

    def test_conservative_processing(self):
        """Test conservative memory processing strategy."""
        processor = MemoryAwareProcessor(strategy=MemoryStrategy.CONSERVATIVE)

        def mock_processing_func(data, **kwargs):
            return [f"processed_{item}" for item in data]

        data = ["item1", "item2", "item3"]
        results = processor._conservative_processing(mock_processing_func, data)

        assert len(results) == 3
        assert results[0] == "processed_item1"

    def test_balanced_processing(self):
        """Test balanced memory processing strategy."""
        processor = MemoryAwareProcessor(strategy=MemoryStrategy.BALANCED)

        def mock_processing_func(data, **kwargs):
            return [f"processed_{item}" for item in data]

        data = ["item1", "item2", "item3", "item4", "item5"]
        results = processor._balanced_processing(mock_processing_func, data)

        assert len(results) == 5
        assert all("processed_" in result for result in results)

    def test_aggressive_processing(self):
        """Test aggressive memory processing strategy."""
        processor = MemoryAwareProcessor(strategy=MemoryStrategy.AGGRESSIVE)

        def mock_processing_func(data, **kwargs):
            return [f"processed_{item}" for item in data]

        data = ["item1", "item2", "item3"]
        results = processor._aggressive_processing(mock_processing_func, data)

        assert len(results) == 3
        assert all("processed_" in result for result in results)

    def test_adaptive_processing(self):
        """Test adaptive processing based on memory conditions."""
        processor = MemoryAwareProcessor(strategy=MemoryStrategy.ADAPTIVE)

        def mock_processing_func(data, **kwargs):
            return [f"processed_{item}" for item in data]

        # Mock different memory profiles
        high_memory_profile = MemoryProfile(
            system_total=16.0,
            system_available=2.0,  # Low available memory
            gpu_total=0.0,
            gpu_available=0.0,
            process_memory=12.0,
            peak_memory=12.0,
        )

        low_memory_profile = MemoryProfile(
            system_total=16.0,
            system_available=12.0,  # High available memory
            gpu_total=0.0,
            gpu_available=0.0,
            process_memory=2.0,
            peak_memory=2.0,
        )

        data = ["item1", "item2", "item3"]

        # Test with high memory usage (should use conservative)
        results1 = processor._adaptive_processing(
            mock_processing_func, data, high_memory_profile
        )
        assert len(results1) == 3

        # Test with low memory usage (should use aggressive)
        results2 = processor._adaptive_processing(
            mock_processing_func, data, low_memory_profile
        )
        assert len(results2) == 3

    def test_process_with_memory_awareness(self):
        """Test main processing function with memory awareness."""
        processor = MemoryAwareProcessor()

        def mock_processing_func(data, **kwargs):
            return [f"processed_{item}" for item in data]

        data = ["item1", "item2", "item3"]
        results = processor.process_with_memory_awareness(
            mock_processing_func, data, test_param="test"
        )

        assert len(results) == 3
        assert all("processed_" in result for result in results)

    def test_proactive_cleanup(self):
        """Test proactive memory cleanup."""
        processor = MemoryAwareProcessor()

        # Add some cache data
        processor._memory_cache["test"] = "data"
        processor._cache_size_bytes = (
            processor.cache_size_gb * 1024**3 + 1
        )  # Over limit

        processor._proactive_cleanup()

        # Cache should be cleared
        assert len(processor._memory_cache) == 0
        assert processor._cache_size_bytes == 0

    def test_emergency_cleanup(self):
        """Test emergency memory cleanup."""
        processor = MemoryAwareProcessor()

        # Add cache data
        processor._memory_cache["test1"] = "data1"
        processor._memory_cache["test2"] = "data2"
        processor._cache_size_bytes = 1000

        initial_batch_size = processor.batcher.current_batch_size

        # Trigger emergency cleanup
        mock_profile = MemoryProfile(
            system_total=16.0,
            system_available=0.2,  # Very low
            gpu_total=0.0,
            gpu_available=0.0,
            process_memory=15.0,
            peak_memory=15.0,
        )

        processor._emergency_cleanup(mock_profile)

        # Should clear cache and reduce batch size
        assert len(processor._memory_cache) == 0
        assert processor._cache_size_bytes == 0
        assert processor.batcher.current_batch_size <= initial_batch_size

    def test_memory_statistics(self):
        """Test memory statistics retrieval."""
        processor = MemoryAwareProcessor()

        stats = processor.get_memory_statistics()

        assert "current_profile" in stats
        assert "strategy" in stats
        assert "cache_size_gb" in stats
        assert "emergency_mode" in stats
        assert "batch_size" in stats
        assert isinstance(stats["current_profile"], MemoryProfile)

    def test_force_garbage_collection(self):
        """Test forced garbage collection."""
        processor = MemoryAwareProcessor()

        initial_time = processor._last_gc_time
        processor._force_garbage_collection()

        # Should update last GC time
        assert processor._last_gc_time > initial_time

    def test_clear_memory_cache(self):
        """Test memory cache clearing."""
        processor = MemoryAwareProcessor()

        # Add cache data
        processor._memory_cache["test1"] = "data1"
        processor._memory_cache["test2"] = "data2"
        processor._cache_size_bytes = 1000

        processor._clear_memory_cache()

        assert len(processor._memory_cache) == 0
        assert processor._cache_size_bytes == 0
