"""
CUDA Backend for ChemiAccel.

This module provides CUDA-accelerated implementations for molecular
descriptor calculations using custom CUDA kernels and RAPIDS integration.
"""

from .memory import (
    CUDAMemoryManager,
    CUDAMemoryPool,
    BatchSizeOptimizer,
    MemoryInfo,
    get_memory_manager,
    cuda_memory_context
)

from .streaming import (
    StreamingConfig,
    CUDAStream,
    DataStreamer,
    StreamingProcessor
)

from .kernels import (
    CUDAKernelManager,
    BasicDescriptorCalculator,
    get_kernel_manager
)

from .graph_kernels import (
    GraphKernelManager,
    GraphDescriptorCalculator,
    get_graph_kernel_manager
)

from .optimized_kernels import (
    OptimizedCUDAKernelManager,
    KernelProfiler as OptimizedKernelProfiler,
    get_optimized_kernel_manager
)

from .optimization_utils import (
    MemoryAccessAnalyzer,
    OccupancyOptimizer,
    KernelProfiler,
    PerformanceRegressionTester,
    KernelPerformanceMetrics,
    OptimizationResult,
    get_kernel_profiler
)

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