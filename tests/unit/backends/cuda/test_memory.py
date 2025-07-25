"""
Unit tests for CUDA memory management system.
"""

import threading
import time
from unittest.mock import Mock, patch

import pytest

# Import without mocking first to get the classes
from src.chemi_accel.backends.cuda.memory import (
    BatchSizeOptimizer,
    CUDAMemoryManager,
    CUDAMemoryPool,
    MemoryInfo,
    cuda_memory_context,
    get_memory_manager,
)
from src.chemi_accel.core.exceptions import GPUMemoryError


class TestMemoryInfo:
    """Test MemoryInfo dataclass."""

    def test_memory_info_creation(self):
        """Test MemoryInfo creation and properties."""
        info = MemoryInfo(total=1000, free=600, used=400)

        assert info.total == 1000
        assert info.free == 600
        assert info.used == 400
        assert info.utilization == 40.0

    def test_memory_info_zero_total(self):
        """Test MemoryInfo with zero total memory."""
        info = MemoryInfo(total=0, free=0, used=0)
        assert info.utilization == 0.0


class TestCUDAMemoryPool:
    """Test CUDAMemoryPool class."""

    @pytest.fixture
    def mock_cupy(self):
        """Mock cupy for testing."""
        with patch("src.chemi_accel.backends.cuda.memory.CUDA_AVAILABLE", True), patch(
            "src.chemi_accel.backends.cuda.memory.cp"
        ) as mock_cp:
            # Mock device
            mock_device = Mock()
            mock_device.__enter__ = Mock(return_value=mock_device)
            mock_device.__exit__ = Mock(return_value=None)
            mock_cp.cuda.Device.return_value = mock_device

            # Mock memory info - return (free, total)
            mock_cp.cuda.runtime.memGetInfo.return_value = (
                800 * 1024**3,
                1000 * 1024**3,
            )

            # Mock memory allocation - create unique pointers
            def create_mock_ptr(size):
                mock_ptr = Mock()
                mock_ptr.size = size
                return mock_ptr

            mock_cp.cuda.alloc.side_effect = create_mock_ptr

            # Mock memory pool cleanup
            mock_pool = Mock()
            mock_pool.free_all_blocks = Mock()
            mock_cp.cuda.memory.get_default_memory_pool.return_value = mock_pool

            yield mock_cp

    @pytest.fixture
    def memory_pool(self, mock_cupy):
        """Create memory pool for testing."""
        with patch("src.chemi_accel.backends.cuda.memory.CUDA_AVAILABLE", True):
            return CUDAMemoryPool(device_id=0, initial_size=1024 * 1024)

    def test_memory_pool_initialization(self, memory_pool, mock_cupy):
        """Test memory pool initialization."""
        assert memory_pool.device_id == 0
        assert memory_pool._pool_size == 1024 * 1024
        assert len(memory_pool._allocated_blocks) == 0
        assert len(memory_pool._free_blocks) == 0

    def test_get_memory_info(self, memory_pool, mock_cupy):
        """Test getting memory information."""
        info = memory_pool.get_memory_info()

        assert isinstance(info, MemoryInfo)
        assert info.total == 1000 * 1024**3
        assert info.free == 800 * 1024**3
        assert info.used == 200 * 1024**3

    def test_allocate_memory(self, memory_pool, mock_cupy):
        """Test memory allocation."""
        ptr = memory_pool.allocate(1024)

        assert ptr is not None
        assert len(memory_pool._allocated_blocks) == 1
        mock_cupy.cuda.alloc.assert_called_once()

    def test_allocate_with_alignment(self, memory_pool, mock_cupy):
        """Test memory allocation with alignment."""
        ptr = memory_pool.allocate(1000, alignment=512)

        # Should allocate aligned size (1024 bytes for 1000 with 512 alignment)
        mock_cupy.cuda.alloc.assert_called_with(1024)

    def test_deallocate_memory(self, memory_pool, mock_cupy):
        """Test memory deallocation."""
        ptr = memory_pool.allocate(1024)
        memory_pool.deallocate(ptr)

        assert len(memory_pool._allocated_blocks) == 0
        assert 1024 in memory_pool._free_blocks
        assert len(memory_pool._free_blocks[1024]) == 1

    def test_memory_reuse(self, memory_pool, mock_cupy):
        """Test memory block reuse."""
        # Allocate and deallocate
        ptr1 = memory_pool.allocate(1024)
        memory_pool.deallocate(ptr1)

        # Allocate again - should reuse
        ptr2 = memory_pool.allocate(1024)

        assert ptr2 == ptr1  # Should be the same pointer
        assert len(memory_pool._free_blocks[1024]) == 0  # Should be empty now

    def test_out_of_memory_error(self, memory_pool, mock_cupy):
        """Test out of memory error handling."""

        # Mock out of memory error - need to create a proper exception
        class MockOutOfMemoryError(Exception):
            pass

        # Set the class name to match what the code expects
        MockOutOfMemoryError.__name__ = "OutOfMemoryError"
        mock_cupy.cuda.memory.OutOfMemoryError = MockOutOfMemoryError
        mock_cupy.cuda.alloc.side_effect = MockOutOfMemoryError("Out of memory")

        with pytest.raises(GPUMemoryError):
            memory_pool.allocate(1024)

    def test_clear_pool(self, memory_pool, mock_cupy):
        """Test clearing memory pool."""
        # Allocate some memory
        ptr1 = memory_pool.allocate(1024)
        ptr2 = memory_pool.allocate(2048)
        memory_pool.deallocate(ptr1)

        # Clear pool
        memory_pool.clear()

        assert len(memory_pool._allocated_blocks) == 0
        assert len(memory_pool._free_blocks) == 0
        assert memory_pool._total_allocated == 0

    def test_get_stats(self, memory_pool, mock_cupy):
        """Test getting pool statistics."""
        # Allocate and deallocate some memory
        ptr1 = memory_pool.allocate(1024)
        ptr2 = memory_pool.allocate(2048)
        memory_pool.deallocate(ptr1)

        stats = memory_pool.get_stats()

        assert stats["allocated_blocks"] == 1  # ptr2 still allocated
        assert stats["free_blocks"] == 1  # ptr1 in free pool
        assert stats["device_id"] == 0
        assert "total_allocated" in stats
        assert "free_size" in stats

    def test_thread_safety(self, memory_pool, mock_cupy):
        """Test thread safety of memory pool."""
        results = []
        errors = []

        def allocate_worker():
            try:
                for _ in range(10):
                    ptr = memory_pool.allocate(1024)
                    results.append(ptr)
                    time.sleep(0.001)  # Small delay
                    memory_pool.deallocate(ptr)
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = [threading.Thread(target=allocate_worker) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(errors) == 0  # No errors should occur
        assert len(results) == 50  # 5 threads * 10 allocations each


class TestBatchSizeOptimizer:
    """Test BatchSizeOptimizer class."""

    @pytest.fixture
    def mock_memory_pool(self):
        """Mock memory pool for testing."""
        pool = Mock()
        pool.get_memory_info.return_value = MemoryInfo(
            total=1000 * 1024**3, free=800 * 1024**3, used=200 * 1024**3
        )
        return pool

    @pytest.fixture
    def optimizer(self, mock_memory_pool):
        """Create optimizer for testing."""
        return BatchSizeOptimizer(mock_memory_pool, safety_factor=0.8)

    def test_optimizer_initialization(self, optimizer, mock_memory_pool):
        """Test optimizer initialization."""
        assert optimizer.memory_pool == mock_memory_pool
        assert optimizer.safety_factor == 0.8
        assert len(optimizer._cache) == 0

    def test_estimate_optimal_batch_size(self, optimizer):
        """Test batch size estimation."""
        # 1 MB per item, should fit many in 800GB * 0.8
        batch_size = optimizer.estimate_optimal_batch_size(1024 * 1024)

        expected = int((800 * 1024**3 * 0.8) // (1024 * 1024))
        assert batch_size == min(expected, 10000)  # Capped at max_batch_size

    def test_estimate_with_max_constraint(self, optimizer):
        """Test batch size estimation with max constraint."""
        # Very small memory per item
        batch_size = optimizer.estimate_optimal_batch_size(1024, max_batch_size=100)

        assert batch_size == 100  # Should be capped at max_batch_size

    def test_estimate_minimum_batch_size(self, optimizer):
        """Test minimum batch size constraint."""
        # Very large memory per item
        batch_size = optimizer.estimate_optimal_batch_size(1000 * 1024**3)

        assert batch_size == 1  # Should be at least 1

    def test_caching(self, optimizer):
        """Test result caching."""
        # First call
        batch_size1 = optimizer.estimate_optimal_batch_size(1024 * 1024)

        # Second call with same parameters
        batch_size2 = optimizer.estimate_optimal_batch_size(1024 * 1024)

        assert batch_size1 == batch_size2
        assert len(optimizer._cache) == 1

    def test_clear_cache(self, optimizer):
        """Test cache clearing."""
        optimizer.estimate_optimal_batch_size(1024 * 1024)
        assert len(optimizer._cache) == 1

        optimizer.clear_cache()
        assert len(optimizer._cache) == 0


class TestCUDAMemoryManager:
    """Test CUDAMemoryManager class."""

    @pytest.fixture
    def mock_cupy(self):
        """Mock cupy for testing."""
        with patch("src.chemi_accel.backends.cuda.memory.cp") as mock_cp:
            mock_device = Mock()
            mock_cp.cuda.Device.return_value = mock_device
            mock_cp.cuda.runtime.memGetInfo.return_value = (
                800 * 1024**3,
                1000 * 1024**3,
            )
            yield mock_cp

    @pytest.fixture
    def memory_manager(self, mock_cupy):
        """Create memory manager for testing."""
        with patch("src.chemi_accel.backends.cuda.memory.CUDA_AVAILABLE", True):
            # Reset singleton
            CUDAMemoryManager._instance = None
            return CUDAMemoryManager()

    def test_singleton_pattern(self, memory_manager, mock_cupy):
        """Test singleton pattern."""
        manager2 = CUDAMemoryManager()
        assert manager2 is memory_manager

    def test_device_initialization(self, memory_manager, mock_cupy):
        """Test device initialization."""
        assert 0 in memory_manager._pools
        assert 0 in memory_manager._optimizers
        assert memory_manager._current_device == 0

    def test_set_device(self, memory_manager, mock_cupy):
        """Test setting device."""
        memory_manager.set_device(1)

        assert memory_manager._current_device == 1
        assert 1 in memory_manager._pools
        assert 1 in memory_manager._optimizers

    def test_get_pool(self, memory_manager, mock_cupy):
        """Test getting memory pool."""
        pool = memory_manager.get_pool()
        assert pool is memory_manager._pools[0]

        pool1 = memory_manager.get_pool(1)
        assert pool1 is memory_manager._pools[1]

    def test_get_optimizer(self, memory_manager, mock_cupy):
        """Test getting batch size optimizer."""
        optimizer = memory_manager.get_optimizer()
        assert optimizer is memory_manager._optimizers[0]

        optimizer1 = memory_manager.get_optimizer(1)
        assert optimizer1 is memory_manager._optimizers[1]

    def test_device_context(self, memory_manager, mock_cupy):
        """Test device context manager."""
        original_device = memory_manager._current_device

        with memory_manager.device_context(1):
            assert memory_manager._current_device == 1

        assert memory_manager._current_device == original_device

    def test_allocate_deallocate(self, memory_manager, mock_cupy):
        """Test memory allocation and deallocation."""
        # Mock allocation
        mock_ptr = Mock()
        mock_ptr.size = 1024
        mock_cupy.cuda.alloc.return_value = mock_ptr

        ptr = memory_manager.allocate(1024)
        assert ptr is not None

        memory_manager.deallocate(ptr)
        # Should not raise any errors

    def test_get_memory_info(self, memory_manager, mock_cupy):
        """Test getting memory information."""
        info = memory_manager.get_memory_info()

        assert isinstance(info, MemoryInfo)
        assert info.total == 1000 * 1024**3
        assert info.free == 800 * 1024**3

    def test_optimize_batch_size(self, memory_manager, mock_cupy):
        """Test batch size optimization."""
        batch_size = memory_manager.optimize_batch_size(1024 * 1024)
        assert isinstance(batch_size, int)
        assert batch_size > 0

    def test_clear_all(self, memory_manager, mock_cupy):
        """Test clearing all pools."""
        # Should not raise any errors
        memory_manager.clear_all()

    def test_get_stats(self, memory_manager, mock_cupy):
        """Test getting statistics."""
        stats = memory_manager.get_stats()

        assert isinstance(stats, dict)
        assert 0 in stats  # Default device should be present


class TestGlobalFunctions:
    """Test global functions."""

    @patch("src.chemi_accel.backends.cuda.memory.CUDA_AVAILABLE", True)
    def test_get_memory_manager(self):
        """Test getting global memory manager."""
        # Reset global instance
        import src.chemi_accel.backends.cuda.memory as memory_module

        memory_module._memory_manager = None

        with patch("src.chemi_accel.backends.cuda.memory.cp"):
            manager1 = get_memory_manager()
            manager2 = get_memory_manager()

            assert manager1 is manager2  # Should be same instance

    @patch("src.chemi_accel.backends.cuda.memory.CUDA_AVAILABLE", True)
    def test_cuda_memory_context(self):
        """Test CUDA memory context manager."""
        with patch("src.chemi_accel.backends.cuda.memory.cp"):
            with cuda_memory_context() as manager:
                assert isinstance(manager, CUDAMemoryManager)

            with cuda_memory_context(device_id=1) as manager:
                assert isinstance(manager, CUDAMemoryManager)


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_cuda_not_available(self):
        """Test behavior when CUDA is not available."""
        with patch("src.chemi_accel.backends.cuda.memory.CUDA_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="CUDA not available"):
                CUDAMemoryPool(device_id=0)

            with pytest.raises(RuntimeError, match="CUDA not available"):
                CUDAMemoryManager()


if __name__ == "__main__":
    pytest.main([__file__])
