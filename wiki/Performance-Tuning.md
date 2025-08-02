# Performance Tuning

ChemiAccel provides comprehensive performance optimization capabilities to maximize throughput and minimize computation time. This guide covers system-level optimizations, algorithm tuning, and hardware-specific enhancements.

## Overview

Performance tuning features include:
- **Hardware Optimization**: CPU, GPU, and memory tuning for maximum performance
- **Algorithm Optimization**: Batch processing, parallel computation, and caching strategies
- **System-Level Tuning**: Operating system, network, and storage optimizations
- **Profiling and Benchmarking**: Detailed performance analysis and bottleneck identification
- **Auto-Tuning**: Automatic parameter optimization based on workload characteristics
- **Scaling Strategies**: Optimal resource utilization across multiple nodes

## Quick Start

### Basic Performance Analysis

```python
from chemi_accel.performance import PerformanceProfiler, AutoTuner
from chemi_accel import Calculator

# Initialize performance profiler
profiler = PerformanceProfiler(
    metrics=['throughput', 'latency', 'memory_usage', 'gpu_utilization'],
    detailed_logging=True
)

# Create calculator with profiling enabled
calc = Calculator(backend="cuda", enable_profiling=True)

# Profile descriptor calculation
molecules = load_test_dataset()  # 10K molecules for profiling
with profiler.profile("descriptor_calculation"):
    results = calc.pandas(molecules, descriptors="comprehensive")

# Get performance analysis
performance_report = profiler.get_detailed_report()
print(f"Throughput: {performance_report['throughput']:.1f} molecules/second")
print(f"GPU utilization: {performance_report['gpu_utilization']:.1f}%")
print(f"Memory efficiency: {performance_report['memory_efficiency']:.2f}")
```

## Hardware-Specific Optimizations

### GPU Performance Tuning

```python
from chemi_accel.performance.gpu import GPUOptimizer, CUDAProfiler

class GPUPerformanceTuner:
    """Comprehensive GPU performance optimization for ChemiAccel."""
    
    def __init__(self, gpu_config=None):
        self.gpu_optimizer = GPUOptimizer()
        self.cuda_profiler = CUDAProfiler()
        self.gpu_config = gpu_config or self.get_default_gpu_config()
        
    def get_default_gpu_config(self):
        """Get default GPU configuration based on detected hardware."""
        
        gpu_info = self.gpu_optimizer.detect_gpu_capabilities()
        
        default_config = {
            'memory_pool_size': min(gpu_info['total_memory'] * 0.8, 16 * 1024**3),  # 80% or 16GB
            'batch_size': self.calculate_optimal_batch_size(gpu_info),
            'streams': min(gpu_info['multiprocessor_count'] // 4, 8),
            'thread_block_size': 256,
            'shared_memory_config': 'prefer_shared',
            'cache_config': 'prefer_l1',
            'occupancy_target': 0.75
        }
        
        return default_config
    
    def calculate_optimal_batch_size(self, gpu_info):
        """Calculate optimal batch size based on GPU memory and compute capability."""
        
        # Memory-based calculation
        available_memory = gpu_info['available_memory']
        descriptor_memory_per_molecule = 2048  # Estimated bytes per molecule
        
        memory_limited_batch = available_memory // (descriptor_memory_per_molecule * 200)  # 200 descriptors
        
        # Compute capability-based calculation
        compute_capability = gpu_info['compute_capability']
        if compute_capability >= 7.0:  # V100, RTX series
            compute_optimal_batch = 10000
        elif compute_capability >= 6.0:  # GTX 10 series, P100
            compute_optimal_batch = 5000
        else:  # Older GPUs
            compute_optimal_batch = 2000
        
        # Take minimum to avoid memory issues
        optimal_batch = min(memory_limited_batch, compute_optimal_batch)
        
        # Round to nearest power of 2 for better memory alignment
        return 2 ** int(np.log2(optimal_batch))
    
    def optimize_memory_management(self):
        """Optimize GPU memory management for ChemiAccel workloads."""
        
        memory_optimizations = {
            'memory_pool': {
                'enable': True,
                'initial_size': self.gpu_config['memory_pool_size'],
                'growth_factor': 1.5,
                'max_size': None  # No limit
            },
            'unified_memory': {
                'enable': True,
                'prefetch_policy': 'adaptive',
                'hint_preferred_location': True
            },
            'garbage_collection': {
                'frequency': 100,  # Every 100 operations
                'threshold': 0.85,  # When 85% memory used
                'aggressive_mode': False
            },
            'memory_mapping': {
                'large_arrays': 'host_pinned',
                'intermediate_results': 'device',
                'final_results': 'host_pageable'
            }
        }
        
        self.gpu_optimizer.configure_memory_management(memory_optimizations)
        return memory_optimizations
    
    def optimize_kernel_execution(self):
        """Optimize CUDA kernel execution parameters."""
        
        # Profile existing kernels
        kernel_profiles = self.cuda_profiler.profile_kernels()
        
        kernel_optimizations = {}
        
        for kernel_name, profile in kernel_profiles.items():
            # Analyze occupancy
            theoretical_occupancy = profile['theoretical_occupancy']
            achieved_occupancy = profile['achieved_occupancy']
            
            optimizations = {
                'thread_block_size': self.optimize_thread_block_size(profile),
                'shared_memory_usage': self.optimize_shared_memory(profile),
                'register_usage': self.optimize_register_usage(profile),
                'grid_size': self.optimize_grid_size(profile)
            }
            
            kernel_optimizations[kernel_name] = optimizations
        
        # Apply optimizations
        for kernel_name, opts in kernel_optimizations.items():
            self.gpu_optimizer.apply_kernel_optimizations(kernel_name, opts)
        
        return kernel_optimizations
    
    def optimize_thread_block_size(self, kernel_profile):
        """Optimize thread block size for maximum occupancy."""
        
        register_count = kernel_profile['register_count']
        shared_memory_per_block = kernel_profile['shared_memory_per_block']
        
        # Test different block sizes
        candidate_sizes = [64, 128, 256, 512, 1024]
        best_size = 256  # Default
        best_occupancy = 0
        
        for block_size in candidate_sizes:
            occupancy = self.calculate_occupancy(
                block_size=block_size,
                register_count=register_count,
                shared_memory=shared_memory_per_block
            )
            
            if occupancy > best_occupancy:
                best_occupancy = occupancy
                best_size = block_size
        
        return best_size
    
    def optimize_data_layout(self, data_characteristics):
        """Optimize data layout for coalesced memory access."""
        
        optimizations = {
            'array_of_structures_to_structure_of_arrays': True,
            'memory_alignment': 128,  # Align to cache line
            'padding': 'auto',  # Automatic padding for alignment
            'interleaving_pattern': self.determine_optimal_interleaving(data_characteristics)
        }
        
        return optimizations
    
    def create_multi_gpu_strategy(self, num_gpus, workload_size):
        """Create multi-GPU optimization strategy."""
        
        if num_gpus == 1:
            return {'strategy': 'single_gpu', 'optimizations': self.gpu_config}
        
        # Determine optimal data parallelism strategy
        molecules_per_gpu = workload_size // num_gpus
        
        if molecules_per_gpu > 50000:  # Large chunks per GPU
            strategy = 'data_parallel_static'
        elif molecules_per_gpu > 10000:  # Medium chunks
            strategy = 'data_parallel_dynamic'
        else:  # Small chunks
            strategy = 'pipeline_parallel'
        
        multi_gpu_config = {
            'strategy': strategy,
            'communication_backend': 'nccl',
            'topology_aware': True,
            'load_balancing': 'adaptive',
            'synchronization': {
                'frequency': 'batch_end',
                'method': 'allreduce' if strategy == 'data_parallel_static' else 'p2p'
            },
            'memory_management': {
                'peer_access': True,
                'unified_memory': False,  # Better control with explicit management
                'memory_pools_per_gpu': True
            }
        }
        
        return multi_gpu_config
    
    def benchmark_configurations(self, test_molecules, configurations):
        """Benchmark different GPU configurations."""
        
        benchmark_results = {}
        
        for config_name, config in configurations.items():
            print(f"Benchmarking configuration: {config_name}")
            
            # Apply configuration
            self.gpu_optimizer.apply_configuration(config)
            
            # Run benchmark
            calc = Calculator(backend="cuda", **config)
            
            start_time = time.time()
            results = calc.pandas(test_molecules, descriptors="comprehensive")
            end_time = time.time()
            
            # Collect performance metrics
            metrics = self.cuda_profiler.get_performance_metrics()
            
            benchmark_results[config_name] = {
                'execution_time': end_time - start_time,
                'throughput': len(test_molecules) / (end_time - start_time),
                'gpu_utilization': metrics['gpu_utilization'],
                'memory_efficiency': metrics['memory_efficiency'],
                'power_consumption': metrics['power_consumption'],
                'configuration': config
            }
        
        # Rank configurations by performance
        ranked_configs = sorted(
            benchmark_results.items(),
            key=lambda x: x[1]['throughput'],
            reverse=True
        )
        
        return benchmark_results, ranked_configs

# Example: GPU performance optimization
gpu_tuner = GPUPerformanceTuner()

# Optimize memory management
memory_config = gpu_tuner.optimize_memory_management()
print("GPU memory optimizations applied")

# Optimize kernel execution
kernel_optimizations = gpu_tuner.optimize_kernel_execution()
print(f"Optimized {len(kernel_optimizations)} CUDA kernels")

# Test different configurations
test_molecules = load_benchmark_dataset()  # 50K molecules

configurations = {
    'default': gpu_tuner.gpu_config,
    'memory_optimized': {**gpu_tuner.gpu_config, 'batch_size': 5000, 'streams': 4},
    'compute_optimized': {**gpu_tuner.gpu_config, 'batch_size': 15000, 'streams': 8},
    'balanced': {**gpu_tuner.gpu_config, 'batch_size': 8000, 'streams': 6}
}

benchmark_results, ranked_configs = gpu_tuner.benchmark_configurations(
    test_molecules, configurations
)

print("\nConfiguration benchmark results:")
for i, (config_name, results) in enumerate(ranked_configs):
    print(f"{i+1}. {config_name}: {results['throughput']:.1f} mol/s "
          f"({results['gpu_utilization']:.1f}% GPU util)")
```

### CPU Performance Optimization

```python
from chemi_accel.performance.cpu import CPUOptimizer, NUMAAnalyzer

class CPUPerformanceTuner:
    """CPU-specific performance optimization for ChemiAccel."""
    
    def __init__(self):
        self.cpu_optimizer = CPUOptimizer()
        self.numa_analyzer = NUMAAnalyzer()
        
    def analyze_cpu_topology(self):
        """Analyze CPU topology for optimal thread placement."""
        
        topology = self.numa_analyzer.get_topology()
        
        analysis = {
            'numa_nodes': topology['numa_nodes'],
            'cores_per_node': topology['cores_per_node'],
            'threads_per_core': topology['threads_per_core'],
            'cache_hierarchy': topology['cache_levels'],
            'memory_bandwidth': topology['memory_bandwidth_per_node'],
            'inter_node_latency': topology['inter_node_latency']
        }
        
        return analysis
    
    def optimize_thread_affinity(self, num_threads, workload_type="compute_intensive"):
        """Optimize thread affinity for different workload types."""
        
        topology = self.analyze_cpu_topology()
        
        if workload_type == "compute_intensive":
            # Spread threads across physical cores, avoid hyperthreading
            affinity_strategy = self.create_core_spread_affinity(topology, num_threads)
        
        elif workload_type == "memory_intensive":
            # Optimize for memory bandwidth, spread across NUMA nodes
            affinity_strategy = self.create_numa_spread_affinity(topology, num_threads)
        
        elif workload_type == "mixed":
            # Balanced approach
            affinity_strategy = self.create_balanced_affinity(topology, num_threads)
        
        # Apply affinity settings
        self.cpu_optimizer.set_thread_affinity(affinity_strategy)
        
        return affinity_strategy
    
    def optimize_memory_allocation(self, memory_pattern="sequential"):
        """Optimize memory allocation strategies."""
        
        topology = self.analyze_cpu_topology()
        
        memory_config = {
            'numa_policy': self.determine_numa_policy(topology, memory_pattern),
            'huge_pages': self.should_use_huge_pages(memory_pattern),
            'memory_prefetching': True,
            'allocation_strategy': memory_pattern,
            'binding_policy': 'local'  # Bind to local NUMA node
        }
        
        # Configure memory allocation
        self.cpu_optimizer.configure_memory_allocation(memory_config)
        
        return memory_config
    
    def optimize_vectorization(self, instruction_sets=None):
        """Optimize vectorization using available instruction sets."""
        
        available_isa = self.cpu_optimizer.detect_instruction_sets()
        
        if instruction_sets is None:
            # Use best available instruction set
            if 'AVX-512' in available_isa:
                instruction_sets = ['AVX-512']
            elif 'AVX2' in available_isa:
                instruction_sets = ['AVX2', 'FMA']
            elif 'AVX' in available_isa:
                instruction_sets = ['AVX']
            else:
                instruction_sets = ['SSE4.2']
        
        vectorization_config = {
            'instruction_sets': instruction_sets,
            'alignment': 64,  # 64-byte alignment for AVX-512
            'loop_unrolling': 4,
            'prefetch_distance': 8,
            'vectorization_hints': True
        }
        
        self.cpu_optimizer.configure_vectorization(vectorization_config)
        
        return vectorization_config
    
    def optimize_compiler_flags(self):
        """Generate optimized compiler flags for ChemiAccel native extensions."""
        
        cpu_features = self.cpu_optimizer.get_cpu_features()
        
        base_flags = ['-O3', '-march=native', '-mtune=native']
        
        # Add instruction set specific flags
        if 'avx512f' in cpu_features:
            base_flags.extend(['-mavx512f', '-mavx512cd'])
        elif 'avx2' in cpu_features:
            base_flags.extend(['-mavx2', '-mfma'])
        elif 'avx' in cpu_features:
            base_flags.append('-mavx')
        
        # Add optimization flags
        optimization_flags = [
            '-funroll-loops',
            '-fprefetch-loop-arrays',
            '-ffast-math',
            '-flto',  # Link-time optimization
            '-fomit-frame-pointer',
            '-fno-semantic-interposition'
        ]
        
        base_flags.extend(optimization_flags)
        
        return base_flags
    
    def create_numa_aware_calculator(self, molecules_per_node=None):
        """Create NUMA-aware calculator configuration."""
        
        topology = self.analyze_cpu_topology()
        num_numa_nodes = len(topology['numa_nodes'])
        
        if molecules_per_node is None:
            molecules_per_node = 10000  # Default chunk size per NUMA node
        
        numa_config = {
            'num_processes': num_numa_nodes,
            'molecules_per_process': molecules_per_node,
            'numa_binding': True,
            'memory_interleaving': False,  # Prefer local memory
            'process_placement': 'numa_spread',
            'shared_memory': {
                'enable': True,
                'size_mb': 1024,
                'numa_local': True
            }
        }
        
        return numa_config

# Example: CPU optimization
cpu_tuner = CPUPerformanceTuner()

# Analyze CPU topology
topology = cpu_tuner.analyze_cpu_topology()
print(f"Detected {len(topology['numa_nodes'])} NUMA nodes")
print(f"Cores per node: {topology['cores_per_node']}")

# Optimize for compute-intensive workload
affinity_config = cpu_tuner.optimize_thread_affinity(
    num_threads=32,
    workload_type="compute_intensive"
)

# Optimize memory allocation
memory_config = cpu_tuner.optimize_memory_allocation(
    memory_pattern="sequential"
)

# Configure vectorization
vectorization_config = cpu_tuner.optimize_vectorization()
print(f"Using instruction sets: {vectorization_config['instruction_sets']}")

# Generate optimized compiler flags
compiler_flags = cpu_tuner.optimize_compiler_flags()
print(f"Recommended compiler flags: {' '.join(compiler_flags)}")

# Create NUMA-aware configuration
numa_config = cpu_tuner.create_numa_aware_calculator()
```

## Algorithm-Level Optimizations

### Batch Processing Optimization

```python
from chemi_accel.performance import BatchOptimizer, CacheManager

class AlgorithmicOptimizer:
    """Algorithm-level performance optimizations."""
    
    def __init__(self):
        self.batch_optimizer = BatchOptimizer()
        self.cache_manager = CacheManager()
        
    def optimize_batch_processing(self, molecules, descriptors, hardware_config):
        """Optimize batch processing strategy based on workload and hardware."""
        
        workload_analysis = self.analyze_workload_characteristics(molecules, descriptors)
        
        # Calculate optimal batch size
        optimal_batch_size = self.calculate_optimal_batch_size(
            workload_analysis, hardware_config
        )
        
        # Determine batching strategy
        if workload_analysis['molecule_size_variance'] > 0.5:
            # High variance in molecule sizes - use adaptive batching
            batching_strategy = 'adaptive_size'
            batch_config = {
                'strategy': batching_strategy,
                'base_batch_size': optimal_batch_size,
                'size_groups': self.create_size_groups(molecules),
                'rebalancing_frequency': 10
            }
        
        elif workload_analysis['descriptor_complexity_variance'] > 0.3:
            # High variance in descriptor complexity - use complexity-based batching
            batching_strategy = 'complexity_based'
            batch_config = {
                'strategy': batching_strategy,
                'complexity_threshold': workload_analysis['median_complexity'],
                'simple_batch_size': int(optimal_batch_size * 1.5),
                'complex_batch_size': int(optimal_batch_size * 0.7)
            }
        
        else:
            # Uniform workload - use fixed batching
            batching_strategy = 'fixed'
            batch_config = {
                'strategy': batching_strategy,
                'batch_size': optimal_batch_size,
                'padding': True  # Pad last batch to maintain alignment
            }
        
        return batch_config
    
    def implement_intelligent_caching(self, descriptor_types):
        """Implement intelligent caching for descriptor calculations."""
        
        cache_strategies = {}
        
        for descriptor_type in descriptor_types:
            if descriptor_type in ['molecular_weight', 'atom_count', 'bond_count']:
                # Simple descriptors - cache aggressively
                cache_strategies[descriptor_type] = {
                    'cache_level': 'memory',
                    'ttl': 3600,  # 1 hour
                    'max_entries': 100000,
                    'eviction_policy': 'lru'
                }
            
            elif descriptor_type in ['morgan_fingerprint', 'maccs_keys']:
                # Fingerprints - medium caching
                cache_strategies[descriptor_type] = {
                    'cache_level': 'memory',
                    'ttl': 1800,  # 30 minutes
                    'max_entries': 50000,
                    'eviction_policy': 'lfu',
                    'compression': True
                }
            
            elif descriptor_type in ['3d_descriptors', 'conformer_dependent']:
                # Complex 3D descriptors - minimal caching
                cache_strategies[descriptor_type] = {
                    'cache_level': 'disk',
                    'ttl': 300,   # 5 minutes
                    'max_entries': 10000,
                    'eviction_policy': 'ttl',
                    'compression': True
                }
            
            else:
                # Default caching strategy
                cache_strategies[descriptor_type] = {
                    'cache_level': 'memory',
                    'ttl': 900,   # 15 minutes
                    'max_entries': 25000,
                    'eviction_policy': 'lru'
                }
        
        # Configure caching system
        self.cache_manager.configure_multilevel_cache(cache_strategies)
        
        return cache_strategies
    
    def optimize_parallel_execution(self, num_workers, workload_characteristics):
        """Optimize parallel execution strategy."""
        
        if workload_characteristics['io_intensive']:
            # I/O intensive workload - use more workers
            parallel_config = {
                'worker_count': min(num_workers * 2, 64),
                'queue_size': num_workers * 4,
                'worker_type': 'thread',  # Threads better for I/O
                'load_balancing': 'round_robin'
            }
        
        elif workload_characteristics['cpu_intensive']:
            # CPU intensive - match worker count to cores
            parallel_config = {
                'worker_count': num_workers,
                'queue_size': num_workers,
                'worker_type': 'process',  # Processes for CPU work
                'load_balancing': 'work_stealing'
            }
        
        elif workload_characteristics['memory_intensive']:
            # Memory intensive - fewer workers to avoid contention
            parallel_config = {
                'worker_count': max(1, num_workers // 2),
                'queue_size': num_workers,
                'worker_type': 'process',
                'load_balancing': 'capacity_based',
                'memory_sharing': True
            }
        
        else:
            # Balanced workload
            parallel_config = {
                'worker_count': num_workers,
                'queue_size': num_workers * 2,
                'worker_type': 'hybrid',  # Mix of threads and processes
                'load_balancing': 'adaptive'
            }
        
        return parallel_config
    
    def implement_pipeline_optimization(self, processing_stages):
        """Implement pipeline optimization for multi-stage processing."""
        
        pipeline_config = {
            'stages': [],
            'buffer_sizes': {},
            'stage_parallelism': {}
        }
        
        for i, stage in enumerate(processing_stages):
            stage_name = stage['name']
            stage_complexity = stage['complexity']
            
            # Determine optimal parallelism for each stage
            if stage_complexity == 'high':
                parallelism = 4
                buffer_size = 1000
            elif stage_complexity == 'medium':
                parallelism = 2
                buffer_size = 2000
            else:
                parallelism = 1
                buffer_size = 5000
            
            pipeline_config['stages'].append({
                'name': stage_name,
                'parallelism': parallelism,
                'buffer_size': buffer_size,
                'optimization_hints': stage.get('hints', [])
            })
        
        return pipeline_config
    
    def create_adaptive_scheduler(self, system_resources):
        """Create adaptive scheduler for dynamic workload management."""
        
        scheduler_config = {
            'scheduling_algorithm': 'work_stealing',
            'load_balancing': 'adaptive',
            'resource_monitoring': {
                'cpu_threshold': 0.8,
                'memory_threshold': 0.85,
                'gpu_threshold': 0.9
            },
            'adaptation_strategies': {
                'high_cpu_load': 'reduce_parallelism',
                'high_memory_load': 'increase_batch_size',
                'high_gpu_load': 'queue_throttling',
                'low_utilization': 'increase_concurrency'
            },
            'feedback_mechanism': {
                'monitoring_interval': 30,  # seconds
                'adaptation_cooldown': 60,  # seconds
                'performance_history': 100  # Keep last 100 measurements
            }
        }
        
        return scheduler_config

# Example: Comprehensive algorithmic optimization
algo_optimizer = AlgorithmicOptimizer()

# Load workload
molecules = load_large_dataset()  # 1M molecules
descriptors = ["comprehensive"]

# Hardware configuration
hardware_config = {
    'cpu_cores': 32,
    'memory_gb': 128,
    'gpu_count': 4,
    'gpu_memory_gb': 32,
    'storage_type': 'nvme_ssd'
}

# Optimize batch processing
batch_config = algo_optimizer.optimize_batch_processing(
    molecules, descriptors, hardware_config
)
print(f"Optimal batching strategy: {batch_config['strategy']}")

# Implement intelligent caching
cache_config = algo_optimizer.implement_intelligent_caching(descriptors)
print(f"Configured caching for {len(cache_config)} descriptor types")

# Optimize parallel execution
workload_characteristics = {
    'io_intensive': False,
    'cpu_intensive': True,
    'memory_intensive': False
}

parallel_config = algo_optimizer.optimize_parallel_execution(
    hardware_config['cpu_cores'], workload_characteristics
)
print(f"Parallel execution: {parallel_config['worker_count']} workers")

# Create adaptive scheduler
scheduler_config = algo_optimizer.create_adaptive_scheduler(hardware_config)
```

## System-Level Optimizations

### Operating System Tuning

```python
from chemi_accel.performance.system import SystemOptimizer

class SystemLevelTuner:
    """System-level performance optimizations."""
    
    def __init__(self):
        self.system_optimizer = SystemOptimizer()
        
    def optimize_kernel_parameters(self):
        """Optimize Linux kernel parameters for ChemiAccel workloads."""
        
        kernel_params = {
            # Memory management
            'vm.swappiness': 1,              # Minimize swapping
            'vm.dirty_ratio': 15,            # Dirty page writeback threshold
            'vm.dirty_background_ratio': 5,  # Background writeback
            'vm.overcommit_memory': 1,       # Always overcommit
            'vm.zone_reclaim_mode': 0,       # Disable zone reclaim
            
            # Network optimization
            'net.core.rmem_max': 134217728,          # 128MB receive buffer
            'net.core.wmem_max': 134217728,          # 128MB send buffer
            'net.ipv4.tcp_rmem': '4096 87380 134217728',
            'net.ipv4.tcp_wmem': '4096 16384 134217728',
            'net.ipv4.tcp_congestion_control': 'bbr',
            
            # Scheduler optimization
            'kernel.sched_migration_cost_ns': 5000000,  # 5ms
            'kernel.sched_min_granularity_ns': 10000000, # 10ms
            'kernel.sched_wakeup_granularity_ns': 15000000, # 15ms
            
            # I/O optimization
            'kernel.io_delay_type': 0,       # No I/O delay
            'vm.dirty_expire_centisecs': 500, # 5 seconds
            'vm.dirty_writeback_centisecs': 100, # 1 second
        }
        
        return kernel_params
    
    def configure_cpu_governor(self, workload_type="performance"):
        """Configure CPU frequency governor."""
        
        if workload_type == "performance":
            governor_config = {
                'governor': 'performance',
                'min_freq': 'max',
                'max_freq': 'max',
                'boost': True
            }
        elif workload_type == "balanced":
            governor_config = {
                'governor': 'ondemand',
                'up_threshold': 80,
                'sampling_rate': 10000,
                'boost': True
            }
        else:  # power_save
            governor_config = {
                'governor': 'powersave',
                'min_freq': 'min',
                'max_freq': '80%',
                'boost': False
            }
        
        self.system_optimizer.configure_cpu_governor(governor_config)
        return governor_config
    
    def optimize_interrupt_handling(self, num_cpus):
        """Optimize interrupt handling for high-performance computing."""
        
        interrupt_config = {
            'irqbalance': False,  # Disable automatic IRQ balancing
            'rps_cpus': f"0-{num_cpus-1}",  # Use all CPUs for RPS
            'rps_flow_cnt': 32768,  # Flow hash table size
            'netdev_max_backlog': 30000,  # Network device queue size
            'netdev_budget': 600,  # NAPI polling budget
            'gpu_irq_affinity': self.calculate_gpu_irq_affinity(num_cpus)
        }
        
        return interrupt_config
    
    def configure_storage_optimization(self, storage_type):
        """Configure storage optimizations."""
        
        if storage_type == 'nvme_ssd':
            storage_config = {
                'scheduler': 'none',  # No I/O scheduler for NVMe
                'queue_depth': 32,
                'read_ahead_kb': 4096,
                'nr_requests': 256,
                'rotational': 0,
                'add_random': 0
            }
        elif storage_type == 'sata_ssd':
            storage_config = {
                'scheduler': 'noop',
                'queue_depth': 16,
                'read_ahead_kb': 2048,
                'nr_requests': 128,
                'rotational': 0,
                'add_random': 0
            }
        else:  # traditional HDD
            storage_config = {
                'scheduler': 'deadline',
                'queue_depth': 8,
                'read_ahead_kb': 8192,
                'nr_requests': 64,
                'rotational': 1,
                'add_random': 1
            }
        
        return storage_config
    
    def create_performance_profile(self, optimization_level="aggressive"):
        """Create comprehensive system performance profile."""
        
        if optimization_level == "aggressive":
            profile = {
                'cpu_governor': self.configure_cpu_governor("performance"),
                'kernel_params': self.optimize_kernel_parameters(),
                'interrupt_handling': self.optimize_interrupt_handling(32),
                'storage_config': self.configure_storage_optimization('nvme_ssd'),
                'memory_management': {
                    'transparent_hugepages': 'always',
                    'numa_balancing': 0,
                    'swap_tendency': 1
                },
                'process_priority': {
                    'nice_level': -10,
                    'io_priority': 'rt:4',
                    'oom_score_adj': -1000
                }
            }
        elif optimization_level == "balanced":
            profile = {
                'cpu_governor': self.configure_cpu_governor("balanced"),
                'kernel_params': {k: v for k, v in self.optimize_kernel_parameters().items() 
                                if not k.startswith('kernel.sched')},
                'memory_management': {
                    'transparent_hugepages': 'madvise',
                    'numa_balancing': 1,
                    'swap_tendency': 10
                }
            }
        else:  # conservative
            profile = {
                'cpu_governor': self.configure_cpu_governor("power_save"),
                'memory_management': {
                    'transparent_hugepages': 'never',
                    'numa_balancing': 1,
                    'swap_tendency': 60
                }
            }
        
        return profile

# Example: System-level optimization
system_tuner = SystemLevelTuner()

# Create aggressive performance profile
performance_profile = system_tuner.create_performance_profile("aggressive")

print("System optimization profile created:")
print(f"CPU governor: {performance_profile['cpu_governor']['governor']}")
print(f"Kernel parameters: {len(performance_profile['kernel_params'])} optimizations")
print(f"Memory management: {performance_profile['memory_management']['transparent_hugepages']} hugepages")
```

## Auto-Tuning and Profiling

### Comprehensive Performance Profiler

```python
from chemi_accel.performance import ComprehensiveProfiler, AutoTuner

class PerformanceAutoTuner:
    """Comprehensive auto-tuning system for ChemiAccel."""
    
    def __init__(self):
        self.profiler = ComprehensiveProfiler()
        self.auto_tuner = AutoTuner()
        self.optimization_history = []
        
    def profile_comprehensive_workload(self, molecules, descriptors, duration_minutes=30):
        """Profile comprehensive workload across different configurations."""
        
        profiling_results = {}
        
        # Test different batch sizes
        batch_sizes = [1000, 2000, 5000, 10000, 20000]
        
        # Test different backend configurations
        backend_configs = [
            {'backend': 'cpu', 'num_threads': 16},
            {'backend': 'cpu', 'num_threads': 32},
            {'backend': 'cuda', 'batch_size': 5000},
            {'backend': 'cuda', 'batch_size': 10000}
        ]
        
        for config in backend_configs:
            for batch_size in batch_sizes:
                if config['backend'] == 'cpu' and batch_size > 10000:
                    continue  # Skip large batches for CPU
                
                config_key = f"{config['backend']}_batch_{batch_size}"
                
                try:
                    # Create calculator with configuration
                    calc_config = {**config, 'batch_size': batch_size}
                    calc = Calculator(**calc_config)
                    
                    # Profile execution
                    with self.profiler.profile_session(config_key):
                        sample_molecules = molecules[:min(50000, len(molecules))]
                        results = calc.pandas(sample_molecules, descriptors=descriptors)
                    
                    # Get profiling results
                    session_results = self.profiler.get_session_results(config_key)
                    profiling_results[config_key] = session_results
                    
                    print(f"Profiled {config_key}: {session_results['throughput']:.1f} mol/s")
                    
                except Exception as e:
                    print(f"Profiling failed for {config_key}: {e}")
        
        return profiling_results
    
    def analyze_performance_bottlenecks(self, profiling_results):
        """Analyze performance bottlenecks from profiling data."""
        
        bottleneck_analysis = {}
        
        for config_name, results in profiling_results.items():
            bottlenecks = []
            
            # CPU utilization analysis
            if results['cpu_utilization'] < 0.7:
                bottlenecks.append({
                    'type': 'cpu_underutilization',
                    'severity': 'medium',
                    'description': f"CPU utilization only {results['cpu_utilization']:.1%}",
                    'recommendation': 'Increase parallelism or batch size'
                })
            
            # Memory analysis
            if results['memory_efficiency'] < 0.6:
                bottlenecks.append({
                    'type': 'memory_inefficiency',
                    'severity': 'high',
                    'description': f"Memory efficiency {results['memory_efficiency']:.1%}",
                    'recommendation': 'Optimize memory access patterns'
                })
            
            # I/O analysis
            if results.get('io_wait_time', 0) > 0.2:
                bottlenecks.append({
                    'type': 'io_bottleneck',
                    'severity': 'high',
                    'description': f"High I/O wait time: {results['io_wait_time']:.1%}",
                    'recommendation': 'Use faster storage or increase I/O parallelism'
                })
            
            # GPU analysis (if applicable)
            if 'gpu_utilization' in results:
                if results['gpu_utilization'] < 0.8:
                    bottlenecks.append({
                        'type': 'gpu_underutilization',
                        'severity': 'high',
                        'description': f"GPU utilization {results['gpu_utilization']:.1%}",
                        'recommendation': 'Increase batch size or improve memory transfer'
                    })
            
            bottleneck_analysis[config_name] = bottlenecks
        
        return bottleneck_analysis
    
    def generate_optimization_recommendations(self, profiling_results, bottleneck_analysis):
        """Generate optimization recommendations based on analysis."""
        
        recommendations = []
        
        # Find best performing configuration
        best_config = max(profiling_results.items(), key=lambda x: x[1]['throughput'])
        best_config_name, best_results = best_config
        
        recommendations.append({
            'type': 'best_configuration',
            'priority': 'high',
            'description': f"Use {best_config_name} configuration",
            'expected_improvement': f"{best_results['throughput']:.1f} mol/s throughput",
            'implementation': self.extract_config_parameters(best_config_name)
        })
        
        # Analyze common bottlenecks
        all_bottlenecks = []
        for bottlenecks in bottleneck_analysis.values():
            all_bottlenecks.extend(bottlenecks)
        
        # Group bottlenecks by type
        bottleneck_counts = {}
        for bottleneck in all_bottlenecks:
            btype = bottleneck['type']
            if btype not in bottleneck_counts:
                bottleneck_counts[btype] = 0
            bottleneck_counts[btype] += 1
        
        # Generate recommendations for common bottlenecks
        for bottleneck_type, count in bottleneck_counts.items():
            if count >= len(profiling_results) * 0.5:  # Present in >50% of configs
                
                if bottleneck_type == 'cpu_underutilization':
                    recommendations.append({
                        'type': 'system_optimization',
                        'priority': 'medium',
                        'description': 'Optimize CPU utilization',
                        'actions': [
                            'Increase number of worker threads',
                            'Optimize thread affinity',
                            'Reduce thread synchronization overhead'
                        ]
                    })
                
                elif bottleneck_type == 'memory_inefficiency':
                    recommendations.append({
                        'type': 'memory_optimization',
                        'priority': 'high',
                        'description': 'Optimize memory usage patterns',
                        'actions': [
                            'Enable memory prefetching',
                            'Optimize data layout for cache efficiency',
                            'Use memory pools to reduce allocation overhead'
                        ]
                    })
                
                elif bottleneck_type == 'gpu_underutilization':
                    recommendations.append({
                        'type': 'gpu_optimization',
                        'priority': 'high',
                        'description': 'Optimize GPU utilization',
                        'actions': [
                            'Increase batch size for better GPU occupancy',
                            'Optimize memory transfers (use pinned memory)',
                            'Increase number of CUDA streams'
                        ]
                    })
        
        return recommendations
    
    def implement_auto_tuning(self, molecules, descriptors, optimization_target="throughput"):
        """Implement automatic parameter tuning."""
        
        # Initial profiling
        print("Starting comprehensive performance profiling...")
        profiling_results = self.profile_comprehensive_workload(molecules, descriptors)
        
        # Bottleneck analysis
        bottleneck_analysis = self.analyze_performance_bottlenecks(profiling_results)
        
        # Generate recommendations
        recommendations = self.generate_optimization_recommendations(
            profiling_results, bottleneck_analysis
        )
        
        # Implement top recommendations
        implemented_optimizations = []
        
        for rec in recommendations[:3]:  # Implement top 3 recommendations
            if rec['type'] == 'best_configuration':
                config = rec['implementation']
                implemented_optimizations.append(config)
                print(f"Applied best configuration: {config}")
            
            elif rec['type'] in ['system_optimization', 'memory_optimization', 'gpu_optimization']:
                for action in rec['actions']:
                    optimization = self.implement_optimization_action(action)
                    if optimization:
                        implemented_optimizations.append(optimization)
                        print(f"Applied optimization: {action}")
        
        # Validate improvements
        validation_results = self.validate_optimizations(
            molecules, descriptors, implemented_optimizations
        )
        
        optimization_summary = {
            'original_best_throughput': max(r['throughput'] for r in profiling_results.values()),
            'optimized_throughput': validation_results['throughput'],
            'improvement_factor': validation_results['throughput'] / max(r['throughput'] for r in profiling_results.values()),
            'implemented_optimizations': implemented_optimizations,
            'recommendations': recommendations
        }
        
        return optimization_summary

# Example: Comprehensive auto-tuning
auto_tuner = PerformanceAutoTuner()

# Load test workload
molecules = load_performance_test_dataset()  # 100K molecules
descriptors = ["comprehensive"]

# Run auto-tuning
optimization_summary = auto_tuner.implement_auto_tuning(
    molecules=molecules,
    descriptors=descriptors,
    optimization_target="throughput"
)

print(f"\nAuto-tuning Results:")
print(f"Original best throughput: {optimization_summary['original_best_throughput']:.1f} mol/s")
print(f"Optimized throughput: {optimization_summary['optimized_throughput']:.1f} mol/s")
print(f"Improvement factor: {optimization_summary['improvement_factor']:.2f}x")
print(f"Applied {len(optimization_summary['implemented_optimizations'])} optimizations")
```

## Best Practices

### 1. Performance Monitoring
```python
# Set up continuous performance monitoring
def setup_performance_monitoring():
    monitoring_config = {
        'metrics': ['throughput', 'latency', 'resource_utilization'],
        'sampling_interval': 10,  # seconds
        'alert_thresholds': {
            'throughput_drop': 0.2,  # 20% drop
            'memory_usage': 0.9,     # 90% memory usage
            'gpu_utilization': 0.95  # 95% GPU usage
        }
    }
    return monitoring_config
```

### 2. Workload-Specific Optimization
```python
# Optimize based on workload characteristics
def optimize_for_workload(workload_type):
    if workload_type == "large_molecules":
        return {
            'batch_size': 1000,
            'memory_pool_size': '8GB',
            'thread_count': 16
        }
    elif workload_type == "many_descriptors":
        return {
            'batch_size': 5000,
            'caching_enabled': True,
            'parallel_descriptors': True
        }
```

### 3. Hardware-Aware Configuration
```python
# Automatically configure based on detected hardware
def auto_configure_hardware():
    hardware = detect_hardware_capabilities()
    
    if hardware['gpu_count'] > 0:
        return configure_gpu_optimizations(hardware)
    elif hardware['cpu_cores'] > 16:
        return configure_high_core_cpu(hardware)
    else:
        return configure_basic_cpu(hardware)
```

---

*For specific optimization scenarios, see the [Backend Selection](Backend-Selection) and [HPC Integration](HPC-Integration) guides.*
