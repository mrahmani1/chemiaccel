"""
CUDA Backend for ChemiAccel.

This module provides CUDA-accelerated implementations for molecular
descriptor calculations using custom CUDA kernels and RAPIDS integration.
"""

from .graph_kernels import (
    GraphDescriptorCalculator,
    GraphKernelManager,
    get_graph_kernel_manager,
)
from .kernels import BasicDescriptorCalculator, CUDAKernelManager, get_kernel_manager
from .memory import (
    BatchSizeOptimizer,
    CUDAMemoryManager,
    CUDAMemoryPool,
    MemoryInfo,
    cuda_memory_context,
    get_memory_manager,
)
from .optimization_utils import (
    KernelPerformanceMetrics,
    KernelProfiler,
    MemoryAccessAnalyzer,
    OccupancyOptimizer,
    OptimizationResult,
    PerformanceRegressionTester,
    get_kernel_profiler,
)
from .optimized_kernels import KernelProfiler as OptimizedKernelProfiler
from .optimized_kernels import OptimizedCUDAKernelManager, get_optimized_kernel_manager
from .streaming import CUDAStream, DataStreamer, StreamingConfig, StreamingProcessor

__all__ = [
    # Memory management
    "CUDAMemoryManager",
    "CUDAMemoryPool",
    "BatchSizeOptimizer",
    "MemoryInfo",
    "get_memory_manager",
    "cuda_memory_context",
    # Streaming
    "StreamingConfig",
    "CUDAStream",
    "DataStreamer",
    "StreamingProcessor",
    # Kernels
    "CUDAKernelManager",
    "BasicDescriptorCalculator",
    "get_kernel_manager",
    # Graph Kernels
    "GraphKernelManager",
    "GraphDescriptorCalculator",
    "get_graph_kernel_manager",
    # Optimized Kernels
    "OptimizedCUDAKernelManager",
    "OptimizedKernelProfiler",
    "get_optimized_kernel_manager",
    # Optimization Utils
    "MemoryAccessAnalyzer",
    "OccupancyOptimizer",
    "KernelProfiler",
    "PerformanceRegressionTester",
    "KernelPerformanceMetrics",
    "OptimizationResult",
    "get_kernel_profiler",
]
