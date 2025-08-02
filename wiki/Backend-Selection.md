# Backend Selection Guide

ChemiAccel supports multiple computational backends to maximize performance across different hardware configurations. This guide helps you choose the optimal backend for your specific use case and hardware setup.

## Available Backends

### CPU Backend
- **Best for**: Small to medium datasets, development, systems without GPU
- **Hardware**: Any modern CPU with multiple cores
- **Memory**: System RAM only
- **Scalability**: Linear with CPU cores

### CUDA Backend  
- **Best for**: Large datasets, production workloads, high-throughput screening
- **Hardware**: NVIDIA GPUs with CUDA Compute Capability 6.0+
- **Memory**: GPU memory + system RAM
- **Scalability**: Excellent for parallel workloads

### JAX Backend
- **Best for**: Research applications, TPU access, gradient-based optimization
- **Hardware**: GPUs, TPUs, or high-end CPUs
- **Memory**: Device-specific memory management
- **Scalability**: Excellent with automatic optimization

### RAPIDS Backend
- **Best for**: Large-scale data processing, integrated ML pipelines
- **Hardware**: NVIDIA GPUs compatible with RAPIDS
- **Memory**: GPU memory for entire pipeline
- **Scalability**: End-to-end GPU acceleration

## Quick Backend Selection

### Choose CPU Backend If:
- Dataset < 10,000 molecules
- Development/prototyping
- No GPU available
- Memory constraints
- Simple descriptor calculations

### Choose CUDA Backend If:
- Dataset > 10,000 molecules
- NVIDIA GPU available
- Production workloads
- Maximum performance needed
- Established CUDA ecosystem

### Choose JAX Backend If:
- Need automatic differentiation
- TPU access available
- Research applications
- Complex mathematical operations
- XLA optimization beneficial

### Choose RAPIDS Backend If:
- End-to-end GPU pipeline desired
- Large-scale data processing
- Integration with cuDF/cuML
- Memory-efficient operations
- RAPIDS ecosystem integration

## Performance Comparison

### Benchmark Results

| Dataset Size | CPU (16 cores) | CUDA (RTX 3080) | JAX (GPU) | RAPIDS |
|--------------|----------------|-----------------|-----------|---------|
| 1K molecules | 2.3s | 1.8s | 2.1s | 2.5s |
| 10K molecules | 23s | 4.2s | 4.8s | 3.9s |
| 100K molecules | 230s | 18s | 21s | 16s |
| 1M molecules | 2300s | 95s | 110s | 85s |

*Benchmark: Comprehensive descriptor calculation (200+ descriptors)*

### Memory Usage Comparison

| Backend | Peak Memory (1M molecules) | Streaming Support |
|---------|----------------------------|-------------------|
| CPU | 32 GB | ✅ Yes |
| CUDA | 24 GB GPU + 16 GB RAM | ✅ Yes |
| JAX | 20 GB GPU + 12 GB RAM | ✅ Yes |
| RAPIDS | 28 GB GPU + 8 GB RAM | ✅ Yes |

## Backend Configuration

### CPU Backend Configuration

```python
from chemi_accel import Calculator
from chemi_accel.backends.cpu import CpuConfig

# Basic CPU configuration
cpu_calc = Calculator(backend="cpu")

# Advanced CPU configuration
cpu_config = CpuConfig(
    num_threads=16,              # Use all CPU cores
    batch_size=1000,             # Optimize for memory
    enable_mkl=True,             # Intel MKL acceleration
    memory_mapping=True,         # Memory-mapped file I/O
    numa_policy="interleave",    # NUMA optimization
    prefetch_factor=2,           # Data prefetching
    worker_init_fn=None          # Custom worker initialization
)

cpu_calc = Calculator(backend="cpu", config=cpu_config)
```

#### CPU Optimization Tips

```python
# For development (fast startup)
dev_config = CpuConfig(
    num_threads=4,
    batch_size=100,
    enable_mkl=False  # Faster imports
)

# For maximum CPU performance
perf_config = CpuConfig(
    num_threads=32,              # All logical cores
    batch_size=2000,             # Large batches
    enable_mkl=True,             # Math acceleration
    memory_mapping=True,         # Efficient I/O
    cache_size_mb=4096          # Large cache
)
```

### CUDA Backend Configuration

```python
from chemi_accel.backends.cuda import CudaConfig

# Basic CUDA configuration
cuda_calc = Calculator(backend="cuda")

# Advanced CUDA configuration
cuda_config = CudaConfig(
    device_ids=[0, 1],           # Multi-GPU setup
    memory_pool_mb=16384,        # Pre-allocate 16GB
    streams=4,                   # Concurrent streams
    batch_size=5000,             # Large GPU batches
    enable_peer_access=True,     # GPU-to-GPU transfers
    compute_mode="default",      # Compute mode
    memory_growth=True,          # Dynamic memory allocation
    synchronize_streams=False    # Asynchronous execution
)

cuda_calc = Calculator(backend="cuda", config=cuda_config)
```

#### GPU Memory Management

```python
# Monitor GPU memory usage
def monitor_gpu_memory():
    import cupy
    mempool = cupy.get_default_memory_pool()
    pinned_mempool = cupy.get_default_pinned_memory_pool()
    
    print(f"GPU memory used: {mempool.used_bytes() / 1024**3:.2f} GB")
    print(f"GPU memory free: {mempool.free_bytes() / 1024**3:.2f} GB")
    print(f"Pinned memory used: {pinned_mempool.n_free_blocks()}")

# Optimize memory usage
cuda_config = CudaConfig(
    memory_pool_mb=8192,         # Limit memory pool
    memory_fraction=0.8,         # Use 80% of GPU memory
    allow_memory_growth=True,    # Grow as needed
    garbage_collect_frequency=100 # Cleanup frequency
)
```

#### Multi-GPU Configuration

```python
# Use multiple GPUs
multi_gpu_config = CudaConfig(
    device_ids=[0, 1, 2, 3],     # 4 GPUs
    data_parallel=True,          # Data parallelism
    load_balancing="round_robin", # Load balancing strategy
    synchronization="barrier"    # Synchronization method
)

calc = Calculator(backend="cuda", config=multi_gpu_config)

# Manual GPU selection
calc.set_gpu_devices([0, 2])    # Use only GPUs 0 and 2
```

### JAX Backend Configuration

```python
from chemi_accel.backends.jax import JaxConfig

# Basic JAX configuration
jax_calc = Calculator(backend="jax")

# Advanced JAX configuration
jax_config = JaxConfig(
    platform="gpu",              # "gpu", "tpu", or "cpu"
    preallocate=False,           # Dynamic memory allocation
    memory_fraction=0.75,        # Use 75% of device memory
    enable_x64=False,            # Use 32-bit precision
    jit_cache_size=1000,         # JIT compilation cache
    xla_optimization=True,       # Enable XLA optimizations
    device_count=1               # Number of devices to use
)

jax_calc = Calculator(backend="jax", config=jax_config)
```

#### JAX-Specific Features

```python
# Enable automatic differentiation
jax_config = JaxConfig(
    enable_autodiff=True,        # Enable grad() functionality
    gradient_checkpointing=True, # Memory-efficient gradients
    precision="float32"          # Computation precision
)

# TPU configuration
tpu_config = JaxConfig(
    platform="tpu",
    tpu_topology="2x2",          # TPU topology
    precompile_kernels=True      # Pre-compile for TPU
)
```

#### XLA Optimization

```python
# Advanced XLA settings
import jax
jax.config.update("jax_enable_x64", False)  # Use 32-bit for speed
jax.config.update("jax_platform_name", "gpu")
jax.config.update("jax_disable_jit", False)  # Enable JIT

# Custom XLA flags
jax_config = JaxConfig(
    xla_flags={
        "xla_gpu_cuda_data_dir": "/usr/local/cuda",
        "xla_gpu_strict_conv_algorithm_picker": "false"
    }
)
```

### RAPIDS Backend Configuration

```python
from chemi_accel.backends.rapids import RapidsConfig

# Basic RAPIDS configuration
rapids_calc = Calculator(backend="rapids")

# Advanced RAPIDS configuration
rapids_config = RapidsConfig(
    gpu_memory_fraction=0.8,     # Use 80% of GPU memory
    pool_allocator=True,         # Use memory pool
    enable_spilling=True,        # Spill to host memory
    spill_device_limit=0.8,      # Spill threshold
    chunk_size="128MB",          # Processing chunk size
    n_workers=4,                 # Dask workers
    threads_per_worker=2         # Threads per worker
)

rapids_calc = Calculator(backend="rapids", config=rapids_config)
```

#### RAPIDS Integration

```python
# Integration with cuDF
import cudf

# Load data directly into GPU DataFrame
df = cudf.read_csv("molecules.csv")
smiles_series = df["smiles"]

# Calculate descriptors with cuDF output
results_cudf = rapids_calc.cudf(smiles_series)

# Seamless integration with cuML
import cuml
from cuml.ensemble import RandomForestRegressor

# Train model on GPU
X_gpu = results_cudf.drop(["smiles"], axis=1)
y_gpu = cudf.Series(target_values)

model = RandomForestRegressor(n_estimators=100)
model.fit(X_gpu, y_gpu)
```

## Automatic Backend Selection

### Intelligent Backend Selection

```python
from chemi_accel import Calculator
from chemi_accel.core.backend_selector import AutoBackendSelector

# Automatic selection based on system and dataset
selector = AutoBackendSelector()
best_backend = selector.select_backend(
    dataset_size=50000,          # Number of molecules
    descriptor_complexity="high", # Descriptor complexity
    memory_limit_gb=16,          # Memory constraint
    time_constraint_minutes=30   # Time constraint
)

calc = Calculator(backend=best_backend)
```

### Benchmark-Based Selection

```python
# Run micro-benchmarks to select optimal backend
def benchmark_backends(sample_molecules, descriptors):
    """Benchmark all available backends."""
    results = {}
    
    for backend in ["cpu", "cuda", "jax", "rapids"]:
        try:
            calc = Calculator(backend=backend)
            
            start_time = time.time()
            calc.pandas(sample_molecules[:100], descriptors=descriptors)
            end_time = time.time()
            
            results[backend] = {
                "time_per_molecule": (end_time - start_time) / 100,
                "available": True
            }
        except Exception as e:
            results[backend] = {
                "time_per_molecule": float('inf'),
                "available": False,
                "error": str(e)
            }
    
    return results

# Use benchmark results
sample_mols = ["CCO", "c1ccccc1", "CC(=O)O"]
benchmark_results = benchmark_backends(sample_mols, ["molecular_weight", "logp"])

# Select fastest available backend
fastest_backend = min(
    (k for k, v in benchmark_results.items() if v["available"]),
    key=lambda k: benchmark_results[k]["time_per_molecule"]
)

print(f"Optimal backend: {fastest_backend}")
```

## Hardware-Specific Recommendations

### High-End Workstation
```python
# Configuration for workstation with RTX 4090
workstation_config = {
    "backend": "cuda",
    "device_ids": [0],
    "memory_pool_mb": 20480,  # 20GB pre-allocation
    "batch_size": 10000,
    "streams": 6
}

calc = Calculator(**workstation_config)
```

### Cloud GPU Instance
```python
# Configuration for AWS g4dn.xlarge
cloud_config = {
    "backend": "cuda",
    "memory_pool_mb": 12288,  # 12GB for T4
    "batch_size": 8000,
    "memory_growth": True,    # Handle varying workloads
    "enable_monitoring": True # Monitor cloud usage
}

calc = Calculator(**cloud_config)
```

### Multi-Node Cluster
```python
# Configuration for HPC cluster
cluster_config = {
    "backend": "cuda",
    "device_ids": [0, 1, 2, 3],  # 4 GPUs per node
    "distributed": True,
    "node_rank": 0,              # Set per node
    "world_size": 8,             # Total number of GPUs
    "distributed_backend": "nccl"
}

calc = Calculator(**cluster_config)
```

### Laptop/Development
```python
# Configuration for development laptop
laptop_config = {
    "backend": "cpu",
    "num_threads": 4,
    "batch_size": 500,
    "memory_limit_gb": 8,
    "enable_cache": True
}

calc = Calculator(**laptop_config)
```

## Backend Switching and Fallbacks

### Dynamic Backend Switching

```python
class AdaptiveCalculator:
    """Calculator that adapts backend based on workload."""
    
    def __init__(self):
        self.backends = self._initialize_backends()
        self.current_backend = None
    
    def _initialize_backends(self):
        backends = {}
        for name in ["cuda", "jax", "cpu"]:
            try:
                backends[name] = Calculator(backend=name)
            except Exception:
                pass
        return backends
    
    def calculate(self, molecules, descriptors):
        # Choose backend based on dataset size
        if len(molecules) > 10000 and "cuda" in self.backends:
            backend = "cuda"
        elif len(molecules) > 1000 and "jax" in self.backends:
            backend = "jax"
        else:
            backend = "cpu"
        
        return self.backends[backend].pandas(molecules, descriptors=descriptors)

# Use adaptive calculator
adaptive_calc = AdaptiveCalculator()
results = adaptive_calc.calculate(molecules, ["molecular_weight", "logp"])
```

### Graceful Fallbacks

```python
def calculate_with_fallback(molecules, descriptors):
    """Calculate with automatic fallback to CPU."""
    backend_order = ["cuda", "jax", "cpu"]
    
    for backend in backend_order:
        try:
            calc = Calculator(backend=backend)
            return calc.pandas(molecules, descriptors=descriptors)
        except Exception as e:
            print(f"Backend {backend} failed: {e}")
            continue
    
    raise RuntimeError("All backends failed")

# Use with fallback
results = calculate_with_fallback(molecules, ["molecular_weight", "logp"])
```

## Performance Monitoring

### Backend Performance Metrics

```python
from chemi_accel.core.profiler import BackendProfiler

# Profile backend performance
profiler = BackendProfiler()

with profiler:
    results = calc.pandas(molecules, descriptors=["molecular_weight", "logp"])

# Get profiling results
metrics = profiler.get_metrics()
print(f"Total time: {metrics['total_time']:.2f}s")
print(f"GPU utilization: {metrics['gpu_utilization']:.1f}%")
print(f"Memory efficiency: {metrics['memory_efficiency']:.1f}%")
```

### Real-time Monitoring

```python
# Monitor performance in real-time
import threading
import time

class PerformanceMonitor:
    def __init__(self, calc):
        self.calc = calc
        self.monitoring = False
        self.metrics = []
    
    def start_monitoring(self):
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        self.monitoring = False
        self.monitor_thread.join()
    
    def _monitor_loop(self):
        while self.monitoring:
            if hasattr(self.calc, 'get_backend_metrics'):
                metrics = self.calc.get_backend_metrics()
                self.metrics.append({
                    'timestamp': time.time(),
                    'memory_used': metrics.get('memory_used', 0),
                    'utilization': metrics.get('utilization', 0)
                })
            time.sleep(1)

# Use performance monitor
monitor = PerformanceMonitor(calc)
monitor.start_monitoring()

# Run calculations
results = calc.pandas(molecules, descriptors=["all"])

monitor.stop_monitoring()
print(f"Collected {len(monitor.metrics)} performance samples")
```

## Troubleshooting Backend Issues

### Common Issues and Solutions

#### CUDA Backend Issues

```python
# Check CUDA availability
def diagnose_cuda():
    try:
        import cupy as cp
        print(f"CUDA available: {cp.cuda.is_available()}")
        print(f"CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
        print(f"GPU count: {cp.cuda.runtime.getDeviceCount()}")
        
        for i in range(cp.cuda.runtime.getDeviceCount()):
            with cp.cuda.Device(i):
                meminfo = cp.cuda.runtime.memGetInfo()
                print(f"GPU {i}: {meminfo[1] // 1024**3} GB total, "
                      f"{meminfo[0] // 1024**3} GB free")
    except Exception as e:
        print(f"CUDA diagnostics failed: {e}")

diagnose_cuda()
```

#### JAX Backend Issues

```python
# Check JAX configuration
def diagnose_jax():
    try:
        import jax
        print(f"JAX version: {jax.__version__}")
        print(f"Available devices: {jax.devices()}")
        print(f"Default platform: {jax.default_backend()}")
        
        # Test basic operation
        import jax.numpy as jnp
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.dot(x, x)
        print(f"JAX test successful: {y}")
    except Exception as e:
        print(f"JAX diagnostics failed: {e}")

diagnose_jax()
```

#### Memory Issues

```python
# Memory diagnostic function
def diagnose_memory_issues(calc, molecules):
    import psutil
    import gc
    
    print(f"System memory: {psutil.virtual_memory().percent}% used")
    
    try:
        # Test with small batch
        small_batch = molecules[:10]
        results = calc.pandas(small_batch)
        print("Small batch successful")
        
        # Test with larger batch
        batch_size = 100
        while batch_size <= len(molecules):
            try:
                batch = molecules[:batch_size]
                results = calc.pandas(batch)
                print(f"Batch size {batch_size} successful")
                batch_size *= 2
            except MemoryError:
                print(f"Memory limit reached at batch size {batch_size}")
                break
                
    except Exception as e:
        print(f"Memory diagnostic failed: {e}")
    
    finally:
        gc.collect()

# Run memory diagnostics
diagnose_memory_issues(calc, molecules)
```

## Best Practices

### 1. Start with Automatic Selection
```python
# Let ChemiAccel choose the best backend
calc = Calculator()  # Automatic backend selection
```

### 2. Profile Your Workload
```python
# Profile different backends with your specific data
def profile_workload(molecules, descriptors):
    results = {}
    for backend in ["cpu", "cuda", "jax"]:
        try:
            calc = Calculator(backend=backend)
            start = time.time()
            calc.pandas(molecules[:1000], descriptors=descriptors)
            results[backend] = time.time() - start
        except:
            pass
    return results
```

### 3. Configure for Your Hardware
```python
# Optimize configuration for your specific hardware
import psutil

cpu_count = psutil.cpu_count()
memory_gb = psutil.virtual_memory().total // (1024**3)

if cpu_count >= 16 and memory_gb >= 32:
    # High-end system
    calc = Calculator(
        backend="cuda" if cuda_available else "cpu",
        batch_size=10000,
        num_threads=cpu_count
    )
else:
    # Standard system
    calc = Calculator(
        backend="cpu",
        batch_size=1000,
        num_threads=min(8, cpu_count)
    )
```

### 4. Handle Errors Gracefully
```python
# Always have fallback options
def robust_calculation(molecules, descriptors):
    for backend in ["cuda", "jax", "cpu"]:
        try:
            calc = Calculator(backend=backend)
            return calc.pandas(molecules, descriptors=descriptors)
        except Exception as e:
            print(f"Backend {backend} failed: {e}")
            continue
    raise RuntimeError("All backends failed")
```

---

*For specific optimization tips, see the [Performance Tuning](Performance-Tuning) guide.*
