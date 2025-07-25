"""
CUDA Memory Management System for ChemiAccel.

This module provides unified memory allocation, memory pools, and automatic
batch size optimization for efficient GPU memory utilization.
"""

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import cupy as cp
    import cupy.cuda.memory as cuda_memory

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None

# Type definitions for when CUDA is not available
if not CUDA_AVAILABLE:
    from typing import Any

    class MockMemoryPointer:
        pass

    # Create a mock cp module for type annotations
    class MockCp:
        class cuda:
            MemoryPointer = MockMemoryPointer

    if cp is None:
        cp = MockCp()

from ...core.exceptions import GPUMemoryError
from ...utils.logging import get_logger

logger = get_logger()


@dataclass
class MemoryInfo:
    """GPU memory information."""

    total: int
    free: int
    used: int

    @property
    def utilization(self) -> float:
        """Memory utilization as percentage."""
        return (self.used / self.total) * 100 if self.total > 0 else 0.0


class CUDAMemoryPool:
    """Memory pool for efficient GPU memory reuse."""

    def __init__(self, device_id: int = 0, initial_size: Optional[int] = None):
        """Initialize CUDA memory pool.

        Args:
            device_id: CUDA device ID
            initial_size: Initial pool size in bytes (None for auto)
        """
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA not available")

        self.device_id = device_id
        self._lock = threading.Lock()
        self._allocated_blocks: Dict[int, cp.cuda.MemoryPointer] = {}
        self._free_blocks: Dict[int, List[cp.cuda.MemoryPointer]] = {}
        self._total_allocated = 0

        # Set device context
        with cp.cuda.Device(device_id):
            self._device = cp.cuda.Device(device_id)

            # Initialize memory pool
            if initial_size is None:
                # Use 80% of available memory as initial pool size
                mem_info = self.get_memory_info()
                initial_size = int(mem_info.free * 0.8)

            self._pool_size = initial_size
            logger.info(
                f"Initializing CUDA memory pool on device {device_id} "
                f"with {initial_size / (1024**3):.2f} GB"
            )

    def get_memory_info(self) -> MemoryInfo:
        """Get current GPU memory information."""
        with cp.cuda.Device(self.device_id):
            free, total = cp.cuda.runtime.memGetInfo()
            used = total - free
            return MemoryInfo(total=total, free=free, used=used)

    def allocate(self, size: int, alignment: int = 256) -> cp.cuda.MemoryPointer:
        """Allocate memory from pool.

        Args:
            size: Size in bytes to allocate
            alignment: Memory alignment in bytes

        Returns:
            Memory pointer

        Raises:
            GPUMemoryError: If allocation fails
        """
        # Align size
        aligned_size = ((size + alignment - 1) // alignment) * alignment

        with self._lock:
            # Try to reuse existing block
            if aligned_size in self._free_blocks and self._free_blocks[aligned_size]:
                ptr = self._free_blocks[aligned_size].pop()
                self._allocated_blocks[id(ptr)] = ptr
                logger.debug(f"Reused memory block: {aligned_size} bytes")
                return ptr

            # Allocate new block
            try:
                with cp.cuda.Device(self.device_id):
                    ptr = cp.cuda.alloc(aligned_size)
                    self._allocated_blocks[id(ptr)] = ptr
                    self._total_allocated += aligned_size
                    logger.debug(f"Allocated new memory block: {aligned_size} bytes")
                    return ptr
            except cp.cuda.memory.OutOfMemoryError as e:
                mem_info = self.get_memory_info()
                raise GPUMemoryError(
                    required_memory=aligned_size // (1024**2),
                    available_memory=mem_info.free // (1024**2),
                ) from e

    def deallocate(self, ptr: cp.cuda.MemoryPointer) -> None:
        """Return memory to pool for reuse.

        Args:
            ptr: Memory pointer to deallocate
        """
        with self._lock:
            ptr_id = id(ptr)
            if ptr_id in self._allocated_blocks:
                size = ptr.size
                del self._allocated_blocks[ptr_id]

                # Add to free blocks for reuse
                if size not in self._free_blocks:
                    self._free_blocks[size] = []
                self._free_blocks[size].append(ptr)
                logger.debug(f"Returned memory block to pool: {size} bytes")

    def clear(self) -> None:
        """Clear all memory pools and force garbage collection."""
        with self._lock:
            # Clear allocated blocks
            self._allocated_blocks.clear()

            # Clear free blocks
            for size_blocks in self._free_blocks.values():
                for ptr in size_blocks:
                    ptr.free()
            self._free_blocks.clear()

            self._total_allocated = 0

            # Force CUDA memory pool cleanup
            if hasattr(cp.cuda.memory, "get_default_memory_pool"):
                pool = cp.cuda.memory.get_default_memory_pool()
                pool.free_all_blocks()

            logger.info("Cleared CUDA memory pool")

    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self._lock:
            free_count = sum(len(blocks) for blocks in self._free_blocks.values())
            free_size = sum(
                ptr.size for blocks in self._free_blocks.values() for ptr in blocks
            )

            return {
                "allocated_blocks": len(self._allocated_blocks),
                "free_blocks": free_count,
                "total_allocated": self._total_allocated,
                "free_size": free_size,
                "device_id": self.device_id,
            }


class BatchSizeOptimizer:
    """Automatic batch size optimization based on available GPU memory."""

    def __init__(self, memory_pool: CUDAMemoryPool, safety_factor: float = 0.8):
        """Initialize batch size optimizer.

        Args:
            memory_pool: CUDA memory pool instance
            safety_factor: Safety factor for memory usage (0.0-1.0)
        """
        self.memory_pool = memory_pool
        self.safety_factor = safety_factor
        self._cache: Dict[Tuple[int, int], int] = {}

    def estimate_optimal_batch_size(
        self, single_item_memory: int, max_batch_size: int = 10000
    ) -> int:
        """Estimate optimal batch size for given memory requirements.

        Args:
            single_item_memory: Memory required per item in bytes
            max_batch_size: Maximum allowed batch size

        Returns:
            Optimal batch size
        """
        cache_key = (single_item_memory, max_batch_size)
        if cache_key in self._cache:
            return self._cache[cache_key]

        mem_info = self.memory_pool.get_memory_info()
        available_memory = int(mem_info.free * self.safety_factor)

        # Calculate theoretical maximum batch size
        theoretical_max = available_memory // single_item_memory

        # Apply constraints
        optimal_batch_size = min(theoretical_max, max_batch_size)
        optimal_batch_size = max(1, optimal_batch_size)  # At least 1

        # Cache result
        self._cache[cache_key] = optimal_batch_size

        logger.debug(
            f"Optimal batch size: {optimal_batch_size} "
            f"(memory per item: {single_item_memory / (1024**2):.2f} MB)"
        )

        return optimal_batch_size

    def clear_cache(self) -> None:
        """Clear batch size cache."""
        self._cache.clear()


class CUDAMemoryManager:
    """Unified CUDA memory management system."""

    _instance: Optional["CUDAMemoryManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "CUDAMemoryManager":
        """Singleton pattern for memory manager."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize memory manager."""
        if hasattr(self, "_initialized"):
            return

        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA not available")

        self._initialized = True
        self._pools: Dict[int, CUDAMemoryPool] = {}
        self._optimizers: Dict[int, BatchSizeOptimizer] = {}
        self._current_device = 0

        # Initialize default device
        self._initialize_device(0)

    def _initialize_device(self, device_id: int) -> None:
        """Initialize memory management for a device."""
        if device_id not in self._pools:
            pool = CUDAMemoryPool(device_id)
            optimizer = BatchSizeOptimizer(pool)

            self._pools[device_id] = pool
            self._optimizers[device_id] = optimizer

            logger.info(f"Initialized memory management for device {device_id}")

    def set_device(self, device_id: int) -> None:
        """Set current CUDA device."""
        self._initialize_device(device_id)
        self._current_device = device_id
        cp.cuda.Device(device_id).use()

    def get_pool(self, device_id: Optional[int] = None) -> CUDAMemoryPool:
        """Get memory pool for device."""
        if device_id is None:
            device_id = self._current_device

        self._initialize_device(device_id)
        return self._pools[device_id]

    def get_optimizer(self, device_id: Optional[int] = None) -> BatchSizeOptimizer:
        """Get batch size optimizer for device."""
        if device_id is None:
            device_id = self._current_device

        self._initialize_device(device_id)
        return self._optimizers[device_id]

    @contextmanager
    def device_context(self, device_id: int):
        """Context manager for device switching."""
        old_device = self._current_device
        try:
            self.set_device(device_id)
            yield
        finally:
            self.set_device(old_device)

    def allocate(
        self, size: int, device_id: Optional[int] = None
    ) -> cp.cuda.MemoryPointer:
        """Allocate memory on device."""
        pool = self.get_pool(device_id)
        return pool.allocate(size)

    def deallocate(
        self, ptr: cp.cuda.MemoryPointer, device_id: Optional[int] = None
    ) -> None:
        """Deallocate memory on device."""
        pool = self.get_pool(device_id)
        pool.deallocate(ptr)

    def get_memory_info(self, device_id: Optional[int] = None) -> MemoryInfo:
        """Get memory information for device."""
        pool = self.get_pool(device_id)
        return pool.get_memory_info()

    def optimize_batch_size(
        self,
        single_item_memory: int,
        max_batch_size: int = 10000,
        device_id: Optional[int] = None,
    ) -> int:
        """Optimize batch size for device."""
        optimizer = self.get_optimizer(device_id)
        return optimizer.estimate_optimal_batch_size(single_item_memory, max_batch_size)

    def clear_all(self) -> None:
        """Clear all memory pools."""
        for pool in self._pools.values():
            pool.clear()
        for optimizer in self._optimizers.values():
            optimizer.clear_cache()
        logger.info("Cleared all CUDA memory pools")

    def get_stats(self) -> Dict[int, Dict[str, Any]]:
        """Get statistics for all devices."""
        return {device_id: pool.get_stats() for device_id, pool in self._pools.items()}


# Global memory manager instance
_memory_manager: Optional[CUDAMemoryManager] = None


def get_memory_manager() -> CUDAMemoryManager:
    """Get global CUDA memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = CUDAMemoryManager()
    return _memory_manager


@contextmanager
def cuda_memory_context(device_id: Optional[int] = None):
    """Context manager for CUDA memory operations."""
    manager = get_memory_manager()
    if device_id is not None:
        with manager.device_context(device_id):
            yield manager
    else:
        yield manager
