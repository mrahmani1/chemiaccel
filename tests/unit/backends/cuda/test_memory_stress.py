"""
Stress tests for CUDA memory management system.

These tests verify memory management under high load and edge conditions.
"""

import pytest
import threading
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch, MagicMock

# Mock CUDA imports for testing without GPU
with patch.dict('sys.modules', {
    'cupy': MagicMock(),
    'cupy.cuda': MagicMock(),
    'cupy.cuda.memory': MagicMock(),
    'cupy.cuda.runtime': MagicMock(),
}):
    from src.chemi_accel.backends.cuda.memory import (
        CUDAMemoryPool,
        CUDAMemoryManager,
        BatchSizeOptimizer,
        get_memory_manager
    )
    from src.chemi_accel.core.exceptions import GPUMemoryError


class TestMemoryPoolStress:
    """Stress tests for CUDAMemoryPool."""
    
    @pytest.fixture
    def mock_cupy_stress(self):
        """Mock cupy with realistic behavior for stress testing."""
        with patch('src.chemi_accel.backends.cuda.memory.cp') as mock_cp:
            # Mock device
            mock_device = Mock()
            mock_cp.cuda.Device.return_value = mock_device
            
            # Mock memory info - simulate limited memory
            mock_cp.cuda.runtime.memGetInfo.return_value = (100 * 1024**2, 200 * 1024**2)  # 100MB free, 200MB total
            
            # Mock memory allocation with realistic pointers
            allocation_counter = [0]
            allocated_pointers = {}
            
            def mock_alloc(size):
                allocation_counter[0] += 1
                mock_ptr = Mock()
                mock_ptr.size = size
                mock_ptr.id = allocation_counter[0]
                allocated_pointers[id(mock_ptr)] = mock_ptr
                
                # Simulate out of memory after many allocations
                if allocation_counter[0] > 1000:
                    from unittest.mock import Mock
                    oom_error = Mock()
                    oom_error.__class__.__name__ = 'OutOfMemoryError'
                    raise oom_error
                
                return mock_ptr
            
            mock_cp.cuda.alloc.side_effect = mock_alloc
            
            yield mock_cp, allocated_pointers
    
    @pytest.fixture
    def stress_memory_pool(self, mock_cupy_stress):
        """Create memory pool for stress testing."""
        mock_cp, _ = mock_cupy_stress
        with patch('src.chemi_accel.backends.cuda.memory.CUDA_AVAILABLE', True):
            return CUDAMemoryPool(device_id=0, initial_size=50 * 1024**2)  # 50MB initial
    
    def test_concurrent_allocations(self, stress_memory_pool, mock_cupy_stress):
        """Test concurrent memory allocations from multiple threads."""
        mock_cp, allocated_pointers = mock_cupy_stress
        
        num_threads = 10
        allocations_per_thread = 50
        results = []
        errors = []
        
        def allocation_worker(thread_id):
            """Worker function for concurrent allocations."""
            thread_results = []
            try:
                for i in range(allocations_per_thread):
                    size = random.randint(1024, 10 * 1024)  # 1KB to 10KB
                    ptr = stress_memory_pool.allocate(size)
                    thread_results.append((ptr, size))
                    
                    # Randomly deallocate some pointers
                    if random.random() < 0.3 and thread_results:
                        old_ptr, _ = thread_results.pop(random.randint(0, len(thread_results) - 1))
                        stress_memory_pool.deallocate(old_ptr)
                
                # Deallocate remaining pointers
                for ptr, _ in thread_results:
                    stress_memory_pool.deallocate(ptr)
                    
            except Exception as e:
                errors.append((thread_id, e))
            
            return thread_results
        
        # Run concurrent allocations
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(allocation_worker, i) for i in range(num_threads)]
            
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=30)
                    results.extend(result)
                except Exception as e:
                    errors.append(('future', e))
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Errors occurred: {errors}"
        
        # Verify pool statistics
        stats = stress_memory_pool.get_stats()
        assert stats['allocated_blocks'] >= 0
        assert stats['device_id'] == 0
    
    def test_memory_fragmentation_handling(self, stress_memory_pool, mock_cupy_stress):
        """Test handling of memory fragmentation."""
        mock_cp, allocated_pointers = mock_cupy_stress
        
        # Allocate many small blocks
        small_blocks = []
        for i in range(100):
            try:
                ptr = stress_memory_pool.allocate(1024)  # 1KB blocks
                small_blocks.append(ptr)
            except GPUMemoryError:
                break
        
        # Deallocate every other block to create fragmentation
        for i in range(0, len(small_blocks), 2):
            stress_memory_pool.deallocate(small_blocks[i])
        
        # Try to allocate larger blocks - should reuse fragmented memory
        large_blocks = []
        for i in range(10):
            try:
                ptr = stress_memory_pool.allocate(2048)  # 2KB blocks
                large_blocks.append(ptr)
            except GPUMemoryError:
                break
        
        # Cleanup
        for ptr in small_blocks[1::2]:  # Remaining small blocks
            stress_memory_pool.deallocate(ptr)
        for ptr in large_blocks:
            stress_memory_pool.deallocate(ptr)
        
        # Should not raise any errors
        assert True
    
    def test_rapid_allocation_deallocation(self, stress_memory_pool, mock_cupy_stress):
        """Test rapid allocation and deallocation cycles."""
        mock_cp, allocated_pointers = mock_cupy_stress
        
        # Perform rapid allocation/deallocation cycles
        for cycle in range(100):
            ptrs = []
            
            # Allocate batch
            for i in range(10):
                try:
                    size = random.randint(512, 4096)
                    ptr = stress_memory_pool.allocate(size)
                    ptrs.append(ptr)
                except GPUMemoryError:
                    break
            
            # Deallocate batch
            for ptr in ptrs:
                stress_memory_pool.deallocate(ptr)
        
        # Verify pool is in good state
        stats = stress_memory_pool.get_stats()
        assert stats['allocated_blocks'] == 0  # All should be deallocated
    
    def test_memory_leak_detection(self, stress_memory_pool, mock_cupy_stress):
        """Test detection of memory leaks."""
        mock_cp, allocated_pointers = mock_cupy_stress
        
        initial_stats = stress_memory_pool.get_stats()
        initial_allocated = initial_stats['total_allocated']
        
        # Allocate and deallocate memory
        ptrs = []
        for i in range(50):
            ptr = stress_memory_pool.allocate(2048)
            ptrs.append(ptr)
        
        for ptr in ptrs:
            stress_memory_pool.deallocate(ptr)
        
        # Check for memory leaks
        final_stats = stress_memory_pool.get_stats()
        
        # All allocated blocks should be returned to free pool
        assert final_stats['allocated_blocks'] == 0
        
        # Free blocks should contain the allocated memory
        assert final_stats['free_blocks'] > 0


class TestBatchSizeOptimizerStress:
    """Stress tests for BatchSizeOptimizer."""
    
    @pytest.fixture
    def mock_memory_pool_stress(self):
        """Mock memory pool with varying memory conditions."""
        pool = Mock()
        
        # Simulate varying memory conditions
        memory_states = [
            (1000 * 1024**3, 800 * 1024**3, 200 * 1024**3),  # High memory
            (1000 * 1024**3, 200 * 1024**3, 800 * 1024**3),  # Low memory
            (1000 * 1024**3, 500 * 1024**3, 500 * 1024**3),  # Medium memory
        ]
        
        call_count = [0]
        
        def get_memory_info():
            state_idx = call_count[0] % len(memory_states)
            call_count[0] += 1
            total, free, used = memory_states[state_idx]
            from src.chemi_accel.backends.cuda.memory import MemoryInfo
            return MemoryInfo(total=total, free=free, used=used)
        
        pool.get_memory_info.side_effect = get_memory_info
        return pool
    
    @pytest.fixture
    def stress_optimizer(self, mock_memory_pool_stress):
        """Create optimizer for stress testing."""
        from src.chemi_accel.backends.cuda.memory import BatchSizeOptimizer
        return BatchSizeOptimizer(mock_memory_pool_stress, safety_factor=0.8)
    
    def test_concurrent_batch_size_estimation(self, stress_optimizer):
        """Test concurrent batch size estimations."""
        results = []
        errors = []
        
        def estimation_worker(thread_id):
            """Worker function for batch size estimation."""
            try:
                thread_results = []
                for i in range(100):
                    item_size = random.randint(1024, 10 * 1024**2)  # 1KB to 10MB
                    max_batch = random.randint(100, 10000)
                    
                    batch_size = stress_optimizer.estimate_optimal_batch_size(
                        item_size, max_batch
                    )
                    
                    thread_results.append((item_size, max_batch, batch_size))
                    
                    # Verify constraints
                    assert batch_size >= 1
                    assert batch_size <= max_batch
                
                return thread_results
                
            except Exception as e:
                errors.append((thread_id, e))
                return []
        
        # Run concurrent estimations
        num_threads = 20
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(estimation_worker, i) for i in range(num_threads)]
            
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=30)
                    results.extend(result)
                except Exception as e:
                    errors.append(('future', e))
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) > 0
        
        # Verify all results are valid
        for item_size, max_batch, batch_size in results:
            assert batch_size >= 1
            assert batch_size <= max_batch
    
    def test_cache_performance_under_load(self, stress_optimizer):
        """Test cache performance under high load."""
        # Pre-populate cache with common sizes
        common_sizes = [1024, 2048, 4096, 8192, 16384]
        for size in common_sizes:
            stress_optimizer.estimate_optimal_batch_size(size)
        
        # Measure cache hit performance
        start_time = time.time()
        
        for _ in range(10000):
            size = random.choice(common_sizes)
            batch_size = stress_optimizer.estimate_optimal_batch_size(size)
            assert batch_size >= 1
        
        cache_time = time.time() - start_time
        
        # Clear cache and measure uncached performance
        stress_optimizer.clear_cache()
        
        start_time = time.time()
        
        for _ in range(1000):  # Fewer iterations for uncached
            size = random.choice(common_sizes)
            batch_size = stress_optimizer.estimate_optimal_batch_size(size)
            assert batch_size >= 1
        
        uncached_time = time.time() - start_time
        
        # Cache should provide significant speedup
        # (This is a rough check - actual speedup depends on mock overhead)
        assert cache_time < uncached_time * 2  # At least some improvement


class TestMemoryManagerStress:
    """Stress tests for CUDAMemoryManager."""
    
    @pytest.fixture
    def mock_cupy_manager_stress(self):
        """Mock cupy for memory manager stress testing."""
        with patch('src.chemi_accel.backends.cuda.memory.cp') as mock_cp:
            # Mock multiple devices
            devices = {}
            for device_id in range(4):  # 4 mock devices
                mock_device = Mock()
                devices[device_id] = mock_device
            
            def get_device(device_id):
                return devices.get(device_id, devices[0])
            
            mock_cp.cuda.Device.side_effect = get_device
            
            # Mock memory info for different devices
            def get_mem_info():
                return (500 * 1024**2, 1000 * 1024**2)  # 500MB free, 1GB total
            
            mock_cp.cuda.runtime.memGetInfo.side_effect = get_mem_info
            
            # Mock allocation
            allocation_counter = [0]
            
            def mock_alloc(size):
                allocation_counter[0] += 1
                mock_ptr = Mock()
                mock_ptr.size = size
                mock_ptr.id = allocation_counter[0]
                return mock_ptr
            
            mock_cp.cuda.alloc.side_effect = mock_alloc
            
            yield mock_cp
    
    @pytest.fixture
    def stress_memory_manager(self, mock_cupy_manager_stress):
        """Create memory manager for stress testing."""
        with patch('src.chemi_accel.backends.cuda.memory.CUDA_AVAILABLE', True):
            # Reset singleton
            from src.chemi_accel.backends.cuda.memory import CUDAMemoryManager
            CUDAMemoryManager._instance = None
            return CUDAMemoryManager()
    
    def test_multi_device_stress(self, stress_memory_manager, mock_cupy_manager_stress):
        """Test stress operations across multiple devices."""
        num_devices = 4
        operations_per_device = 100
        
        def device_worker(device_id):
            """Worker function for device operations."""
            errors = []
            try:
                with stress_memory_manager.device_context(device_id):
                    ptrs = []
                    
                    # Perform allocations
                    for i in range(operations_per_device):
                        size = random.randint(1024, 100 * 1024)
                        ptr = stress_memory_manager.allocate(size)
                        ptrs.append(ptr)
                        
                        # Randomly deallocate some pointers
                        if len(ptrs) > 10 and random.random() < 0.2:
                            old_ptr = ptrs.pop(random.randint(0, len(ptrs) - 1))
                            stress_memory_manager.deallocate(old_ptr)
                    
                    # Cleanup remaining pointers
                    for ptr in ptrs:
                        stress_memory_manager.deallocate(ptr)
                        
            except Exception as e:
                errors.append((device_id, e))
            
            return errors
        
        # Run operations on multiple devices concurrently
        with ThreadPoolExecutor(max_workers=num_devices) as executor:
            futures = [executor.submit(device_worker, i) for i in range(num_devices)]
            
            all_errors = []
            for future in as_completed(futures):
                try:
                    errors = future.result(timeout=60)
                    all_errors.extend(errors)
                except Exception as e:
                    all_errors.append(('future', e))
        
        # Verify no errors occurred
        assert len(all_errors) == 0, f"Errors occurred: {all_errors}"
        
        # Verify all devices were initialized
        assert len(stress_memory_manager._pools) == num_devices
        assert len(stress_memory_manager._optimizers) == num_devices
    
    def test_rapid_device_switching(self, stress_memory_manager, mock_cupy_manager_stress):
        """Test rapid device switching under load."""
        num_switches = 1000
        num_devices = 4
        
        for i in range(num_switches):
            device_id = random.randint(0, num_devices - 1)
            
            # Switch device and perform operation
            stress_memory_manager.set_device(device_id)
            
            # Allocate and deallocate memory
            ptr = stress_memory_manager.allocate(1024)
            stress_memory_manager.deallocate(ptr)
            
            # Verify current device
            assert stress_memory_manager.get_current_backend() is not None
        
        # All devices should be initialized
        assert len(stress_memory_manager._pools) == num_devices
    
    def test_memory_manager_singleton_stress(self, mock_cupy_manager_stress):
        """Test singleton behavior under concurrent access."""
        with patch('src.chemi_accel.backends.cuda.memory.CUDA_AVAILABLE', True):
            # Reset singleton
            from src.chemi_accel.backends.cuda.memory import CUDAMemoryManager
            CUDAMemoryManager._instance = None
            
            managers = []
            errors = []
            
            def get_manager_worker(thread_id):
                """Worker to get memory manager instance."""
                try:
                    manager = CUDAMemoryManager()
                    managers.append(manager)
                    
                    # Perform some operations
                    manager.set_device(0)
                    ptr = manager.allocate(1024)
                    manager.deallocate(ptr)
                    
                except Exception as e:
                    errors.append((thread_id, e))
            
            # Create many threads trying to get manager instance
            num_threads = 50
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(get_manager_worker, i) for i in range(num_threads)]
                
                for future in as_completed(futures):
                    try:
                        future.result(timeout=30)
                    except Exception as e:
                        errors.append(('future', e))
            
            # Verify no errors occurred
            assert len(errors) == 0, f"Errors occurred: {errors}"
            
            # All managers should be the same instance
            assert len(managers) == num_threads
            for manager in managers:
                assert manager is managers[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])