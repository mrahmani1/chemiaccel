# Troubleshooting

This comprehensive troubleshooting guide covers common issues, error messages, and solutions when using ChemiAccel. Use this guide to quickly diagnose and resolve problems across different deployment scenarios.

## Overview

Common troubleshooting categories:
- **Installation Issues**: Dependency conflicts, environment setup problems
- **Runtime Errors**: Backend failures, memory issues, calculation errors
- **Performance Problems**: Slow execution, resource utilization issues
- **Network and Distributed Computing**: Connectivity and scaling problems
- **Cloud and HPC Issues**: Platform-specific deployment problems
- **Data and File Format Issues**: Input validation and format problems

## Installation and Setup Issues

### Python Environment Problems

#### Issue: ImportError - Missing Dependencies

**Error Message:**
```
ImportError: No module named 'cupy'
ModuleNotFoundError: No module named 'chemi_accel'
```

**Solution:**
```bash
# Check Python version (3.8+ required)
python --version

# Create clean virtual environment
python -m venv chemiaccel-env
source chemiaccel-env/bin/activate  # Linux/Mac
# or
chemiaccel-env\Scripts\activate  # Windows

# Update pip and install dependencies
pip install --upgrade pip setuptools wheel

# Install ChemiAccel with appropriate backend
pip install chemi-accel[cuda]  # For NVIDIA GPU support
pip install chemi-accel[jax]   # For JAX/TPU support
pip install chemi-accel[all]   # All optional dependencies

# Verify installation
python -c "import chemi_accel; print(chemi_accel.__version__)"
```

#### Issue: CUDA Installation Problems

**Error Message:**
```
RuntimeError: CUDA driver version is insufficient for CUDA runtime version
ImportError: libcudart.so.11.0: cannot open shared object file
```

**Solution:**
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Install appropriate CUDA toolkit
# For Ubuntu/Debian:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.4.0/local_installers/cuda-repo-ubuntu2004-11-4-local_11.4.0-470.42.01-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-4-local_11.4.0-470.42.01-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-11-4-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda

# Install matching CuPy version
pip install cupy-cuda114  # For CUDA 11.4

# Set environment variables
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

#### Issue: Conflicting Package Versions

**Error Message:**
```
ERROR: pip's dependency resolver does not currently consider all the rules
ERROR: Cannot install chemi-accel and tensorflow==2.10.0 because these have conflicting dependencies
```

**Solution:**
```bash
# Check for conflicts
pip check

# Create requirements file with compatible versions
cat > requirements.txt << EOF
chemi-accel>=1.0.0
tensorflow>=2.11.0
numpy>=1.21.0,<1.25.0
pandas>=1.4.0
rdkit-pypi>=2022.3.1
EOF

# Install with constraints
pip install -r requirements.txt

# Alternative: Use conda for better dependency resolution
conda create -n chemiaccel python=3.9
conda activate chemiaccel
conda install -c conda-forge rdkit
pip install chemi-accel
```

### Backend Configuration Issues

#### Issue: Backend Not Available

**Error Message:**
```
RuntimeError: Backend 'cuda' not available on this system
BackendError: No suitable backend found for current configuration
```

**Diagnostic Commands:**
```python
from chemi_accel import Calculator
from chemi_accel.core.system import system_info

# Check system capabilities
system_info()

# Check available backends
calc = Calculator()
print("Available backends:", calc.available_backends)

# Test each backend individually
for backend in ["cpu", "cuda", "jax"]:
    try:
        test_calc = Calculator(backend=backend)
        print(f"✓ {backend} backend working")
    except Exception as e:
        print(f"✗ {backend} backend failed: {e}")
```

**Solution:**
```python
# Automatic backend selection with fallback
def create_robust_calculator():
    backend_priority = ["cuda", "jax", "cpu"]
    
    for backend in backend_priority:
        try:
            calc = Calculator(backend=backend)
            print(f"Using {backend} backend")
            return calc
        except Exception as e:
            print(f"Backend {backend} unavailable: {e}")
            continue
    
    raise RuntimeError("No working backend found")

# Use with error handling
try:
    calc = create_robust_calculator()
except RuntimeError as e:
    print(f"Fatal error: {e}")
    print("Please check your installation and system configuration")
```

## Runtime Errors

### Memory Issues

#### Issue: Out of Memory Errors

**Error Messages:**
```
CUDA out of memory. Tried to allocate 4.00 GiB
MemoryError: Unable to allocate array with shape (1000000, 200)
RuntimeError: [Errno 12] Cannot allocate memory
```

**Diagnostic Tools:**
```python
import psutil
import gc
from chemi_accel.performance import MemoryProfiler

def diagnose_memory_issue():
    # System memory info
    memory = psutil.virtual_memory()
    print(f"Total RAM: {memory.total / 1024**3:.1f} GB")
    print(f"Available RAM: {memory.available / 1024**3:.1f} GB")
    print(f"Memory usage: {memory.percent:.1f}%")
    
    # GPU memory info (if available)
    try:
        import cupy as cp
        mempool = cp.get_default_memory_pool()
        print(f"GPU memory used: {mempool.used_bytes() / 1024**3:.1f} GB")
        print(f"GPU memory free: {mempool.free_bytes() / 1024**3:.1f} GB")
    except ImportError:
        print("GPU memory info not available")
    
    # Python object memory
    gc.collect()
    print(f"Python objects: {len(gc.get_objects())}")

diagnose_memory_issue()
```

**Solutions:**
```python
# Solution 1: Reduce batch size
calc = Calculator(backend="cuda", batch_size=1000)  # Smaller batches

# Solution 2: Enable streaming processing
calc = Calculator(
    backend="cuda",
    enable_streaming=True,
    max_memory_gb=8  # Limit memory usage
)

# Solution 3: Process in chunks
def process_large_dataset_in_chunks(molecules, chunk_size=5000):
    results = []
    
    for i in range(0, len(molecules), chunk_size):
        chunk = molecules[i:i+chunk_size]
        
        # Force garbage collection
        gc.collect()
        
        # Clear GPU memory cache
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
        except ImportError:
            pass
        
        # Process chunk
        chunk_results = calc.pandas(chunk, descriptors="default")
        results.append(chunk_results)
        
        print(f"Processed chunk {i//chunk_size + 1}/{len(molecules)//chunk_size + 1}")
    
    return pd.concat(results, ignore_index=True)

# Solution 4: Use memory-efficient data types
def optimize_dataframe_memory(df):
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    return df
```

#### Issue: Memory Leaks

**Symptoms:**
- Memory usage increases over time
- Performance degrades with continued use
- System becomes unresponsive

**Detection:**
```python
import tracemalloc
from chemi_accel.performance import MemoryLeakDetector

def detect_memory_leaks():
    # Start memory tracing
    tracemalloc.start()
    
    detector = MemoryLeakDetector()
    detector.start_monitoring()
    
    # Run test workload multiple times
    molecules = ["CCO", "c1ccccc1", "CC(=O)O"] * 1000
    
    for i in range(10):
        calc = Calculator()
        results = calc.pandas(molecules, descriptors=["molecular_weight"])
        
        # Take memory snapshot
        snapshot = tracemalloc.take_snapshot()
        if i > 0:
            detector.compare_snapshots(previous_snapshot, snapshot)
        previous_snapshot = snapshot
        
        print(f"Iteration {i+1}: {detector.get_current_memory_usage():.1f} MB")
    
    # Get leak report
    leak_report = detector.get_leak_report()
    if leak_report['suspected_leaks']:
        print("Memory leaks detected:")
        for leak in leak_report['suspected_leaks']:
            print(f"  {leak['location']}: {leak['size_mb']:.1f} MB")

detect_memory_leaks()
```

**Solutions:**
```python
# Solution 1: Explicit cleanup
class ManagedCalculator:
    def __init__(self, **kwargs):
        self.calc = Calculator(**kwargs)
    
    def __enter__(self):
        return self.calc
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Explicit cleanup
        if hasattr(self.calc, 'cleanup'):
            self.calc.cleanup()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear GPU memory
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
        except ImportError:
            pass

# Usage
with ManagedCalculator(backend="cuda") as calc:
    results = calc.pandas(molecules, descriptors="default")

# Solution 2: Periodic memory cleanup
def process_with_periodic_cleanup(molecules, cleanup_interval=10000):
    results = []
    
    for i in range(0, len(molecules), cleanup_interval):
        chunk = molecules[i:i+cleanup_interval]
        
        with ManagedCalculator() as calc:
            chunk_results = calc.pandas(chunk, descriptors="default")
            results.append(chunk_results)
        
        # Explicit cleanup every iteration
        if i % (cleanup_interval * 5) == 0:
            gc.collect()
    
    return pd.concat(results, ignore_index=True)
```

### Calculation Errors

#### Issue: Invalid Molecule Errors

**Error Messages:**
```
InvalidMoleculeError: Cannot parse SMILES 'InvalidSMILES123'
RDKit ERROR: Could not parse molecule from SMILES
ValueError: Molecule validation failed
```

**Diagnostic and Solution:**
```python
from rdkit import Chem
from chemi_accel.core.exceptions import InvalidMoleculeError

def validate_and_clean_molecules(molecules):
    """Validate and clean molecule list."""
    
    valid_molecules = []
    invalid_molecules = []
    
    for i, smiles in enumerate(molecules):
        try:
            # Basic SMILES validation
            mol = Chem.MolFromSmiles(smiles)
            
            if mol is None:
                invalid_molecules.append((i, smiles, "Invalid SMILES syntax"))
                continue
            
            # Additional validation
            if mol.GetNumAtoms() == 0:
                invalid_molecules.append((i, smiles, "Empty molecule"))
                continue
            
            if mol.GetNumAtoms() > 1000:
                invalid_molecules.append((i, smiles, "Molecule too large"))
                continue
            
            # Sanitization check
            try:
                Chem.SanitizeMol(mol)
                canonical_smiles = Chem.MolToSmiles(mol)
                valid_molecules.append(canonical_smiles)
            except:
                invalid_molecules.append((i, smiles, "Sanitization failed"))
        
        except Exception as e:
            invalid_molecules.append((i, smiles, str(e)))
    
    print(f"Valid molecules: {len(valid_molecules)}")
    print(f"Invalid molecules: {len(invalid_molecules)}")
    
    if invalid_molecules:
        print("First 5 invalid molecules:")
        for i, (idx, smiles, reason) in enumerate(invalid_molecules[:5]):
            print(f"  {idx}: '{smiles}' - {reason}")
    
    return valid_molecules, invalid_molecules

# Usage with error handling
def robust_calculation(molecules, descriptors):
    # Validate molecules first
    valid_molecules, invalid_molecules = validate_and_clean_molecules(molecules)
    
    if not valid_molecules:
        raise ValueError("No valid molecules found in input")
    
    # Calculate descriptors for valid molecules
    calc = Calculator()
    
    try:
        results = calc.pandas(valid_molecules, descriptors=descriptors)
        
        # Add invalid molecule placeholders if needed
        if invalid_molecules:
            results = add_invalid_molecule_placeholders(results, invalid_molecules)
        
        return results
    
    except Exception as e:
        print(f"Calculation failed: {e}")
        # Try with smaller batch size
        print("Retrying with smaller batch size...")
        calc = Calculator(batch_size=500)
        return calc.pandas(valid_molecules, descriptors=descriptors)
```

#### Issue: Descriptor Calculation Failures

**Error Messages:**
```
DescriptorCalculationError: Failed to calculate 'morgan_fingerprint'
RuntimeError: Descriptor 'unknown_descriptor' not found
ValueError: Incompatible descriptor configuration
```

**Diagnostic Tools:**
```python
def diagnose_descriptor_issues(molecules, descriptors):
    """Diagnose descriptor calculation problems."""
    
    calc = Calculator()
    
    # Check descriptor availability
    available_descriptors = calc.available_descriptors
    print(f"Available descriptors: {len(available_descriptors)}")
    
    # Validate requested descriptors
    if isinstance(descriptors, str):
        if descriptors == "all":
            descriptors = available_descriptors
        elif descriptors in calc.descriptor_categories:
            descriptors = calc.descriptor_categories[descriptors]
        else:
            descriptors = [descriptors]
    
    invalid_descriptors = [d for d in descriptors if d not in available_descriptors]
    if invalid_descriptors:
        print(f"Invalid descriptors: {invalid_descriptors}")
        descriptors = [d for d in descriptors if d in available_descriptors]
    
    # Test descriptor calculation on sample
    sample_molecules = molecules[:min(10, len(molecules))]
    
    working_descriptors = []
    failing_descriptors = []
    
    for descriptor in descriptors:
        try:
            test_result = calc.pandas(sample_molecules, descriptors=[descriptor])
            working_descriptors.append(descriptor)
            print(f"✓ {descriptor}")
        except Exception as e:
            failing_descriptors.append((descriptor, str(e)))
            print(f"✗ {descriptor}: {e}")
    
    return working_descriptors, failing_descriptors

# Usage
working_descriptors, failing_descriptors = diagnose_descriptor_issues(
    molecules, ["molecular_weight", "logp", "invalid_descriptor"]
)

# Calculate only working descriptors
if working_descriptors:
    results = calc.pandas(molecules, descriptors=working_descriptors)
```

## Performance Issues

### Slow Execution

#### Issue: Unexpectedly Slow Performance

**Diagnostic Commands:**
```python
from chemi_accel.performance import PerformanceProfiler
import time

def diagnose_performance_issues(molecules, descriptors):
    """Diagnose performance bottlenecks."""
    
    profiler = PerformanceProfiler()
    
    # Baseline performance test
    small_sample = molecules[:100]
    
    with profiler.profile("baseline_test"):
        calc = Calculator()
        baseline_results = calc.pandas(small_sample, descriptors=descriptors)
    
    baseline_metrics = profiler.get_metrics("baseline_test")
    baseline_throughput = len(small_sample) / baseline_metrics['execution_time']
    
    print(f"Baseline throughput: {baseline_throughput:.1f} molecules/second")
    
    # Test different configurations
    test_configs = [
        {"backend": "cpu", "batch_size": 500},
        {"backend": "cpu", "batch_size": 2000},
        {"backend": "cuda", "batch_size": 5000},
        {"backend": "cuda", "batch_size": 10000}
    ]
    
    results = {}
    
    for config in test_configs:
        config_name = f"{config['backend']}_batch_{config['batch_size']}"
        
        try:
            with profiler.profile(config_name):
                test_calc = Calculator(**config)
                test_results = test_calc.pandas(small_sample, descriptors=descriptors)
            
            metrics = profiler.get_metrics(config_name)
            throughput = len(small_sample) / metrics['execution_time']
            
            results[config_name] = {
                'throughput': throughput,
                'cpu_utilization': metrics.get('cpu_utilization', 0),
                'memory_usage': metrics.get('memory_usage', 0),
                'gpu_utilization': metrics.get('gpu_utilization', 0)
            }
            
            print(f"{config_name}: {throughput:.1f} mol/s")
            
        except Exception as e:
            results[config_name] = {'error': str(e)}
            print(f"{config_name}: ERROR - {e}")
    
    # Find best configuration
    best_config = max(
        (k, v for k, v in results.items() if 'throughput' in v),
        key=lambda x: x[1]['throughput']
    )
    
    if best_config:
        print(f"\nBest configuration: {best_config[0]}")
        print(f"Best throughput: {best_config[1]['throughput']:.1f} mol/s")
    
    return results

# Run performance diagnosis
performance_results = diagnose_performance_issues(molecules, descriptors)
```

**Solutions:**
```python
# Solution 1: Optimize batch size
def find_optimal_batch_size(molecules, descriptors, max_batch_size=20000):
    """Find optimal batch size through testing."""
    
    sample_molecules = molecules[:min(5000, len(molecules))]
    batch_sizes = [500, 1000, 2000, 5000, 10000, max_batch_size]
    
    best_batch_size = 1000
    best_throughput = 0
    
    for batch_size in batch_sizes:
        try:
            calc = Calculator(batch_size=batch_size)
            
            start_time = time.time()
            results = calc.pandas(sample_molecules, descriptors=descriptors)
            end_time = time.time()
            
            throughput = len(sample_molecules) / (end_time - start_time)
            print(f"Batch size {batch_size}: {throughput:.1f} mol/s")
            
            if throughput > best_throughput:
                best_throughput = throughput
                best_batch_size = batch_size
        
        except Exception as e:
            print(f"Batch size {batch_size} failed: {e}")
            break
    
    return best_batch_size

optimal_batch = find_optimal_batch_size(molecules, descriptors)
print(f"Optimal batch size: {optimal_batch}")

# Solution 2: Enable caching
calc = Calculator(
    backend="cuda",
    batch_size=optimal_batch,
    enable_cache=True,
    cache_size_mb=2048
)

# Solution 3: Parallel processing
from concurrent.futures import ProcessPoolExecutor
import numpy as np

def parallel_calculation(molecules, descriptors, num_processes=4):
    """Process molecules in parallel."""
    
    # Split molecules into chunks
    chunk_size = len(molecules) // num_processes
    chunks = [molecules[i:i+chunk_size] for i in range(0, len(molecules), chunk_size)]
    
    def process_chunk(chunk):
        calc = Calculator()
        return calc.pandas(chunk, descriptors=descriptors)
    
    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        chunk_results = list(executor.map(process_chunk, chunks))
    
    # Combine results
    return pd.concat(chunk_results, ignore_index=True)

# Usage for large datasets
if len(molecules) > 50000:
    results = parallel_calculation(molecules, descriptors, num_processes=8)
else:
    results = calc.pandas(molecules, descriptors=descriptors)
```

### Resource Utilization Issues

#### Issue: Low GPU Utilization

**Diagnostic Commands:**
```python
def diagnose_gpu_utilization():
    """Diagnose GPU utilization issues."""
    
    try:
        import cupy as cp
        import pynvml
        
        # Initialize NVML
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        
        print(f"Found {device_count} GPU(s)")
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            # GPU info
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            print(f"GPU {i}: {name}")
            print(f"  Memory: {memory_info.used / 1024**3:.1f} / {memory_info.total / 1024**3:.1f} GB")
            print(f"  GPU Utilization: {utilization.gpu}%")
            print(f"  Memory Utilization: {utilization.memory}%")
        
        # CuPy memory pool info
        mempool = cp.get_default_memory_pool()
        print(f"CuPy memory pool: {mempool.used_bytes() / 1024**3:.1f} GB used")
        
    except ImportError:
        print("GPU monitoring tools not available")
    except Exception as e:
        print(f"GPU monitoring failed: {e}")

diagnose_gpu_utilization()
```

**Solutions:**
```python
# Solution 1: Increase batch size for better GPU occupancy
calc = Calculator(
    backend="cuda",
    batch_size=10000,  # Larger batches for GPU
    streams=4          # Multiple CUDA streams
)

# Solution 2: Optimize memory transfers
calc = Calculator(
    backend="cuda",
    enable_pinned_memory=True,  # Faster CPU-GPU transfers
    prefetch_data=True,         # Overlap computation and transfer
    memory_pool_size="8GB"      # Pre-allocate GPU memory
)

# Solution 3: Multi-GPU utilization
calc = Calculator(
    backend="cuda",
    device_ids=[0, 1, 2, 3],    # Use multiple GPUs
    data_parallel=True          # Enable data parallelism
)

# Solution 4: Asynchronous processing
import asyncio

async def async_gpu_processing(molecule_batches):
    """Process multiple batches asynchronously."""
    
    calc = Calculator(backend="cuda", streams=8)
    
    async def process_batch(batch):
        return calc.pandas(batch, descriptors="default")
    
    # Process batches concurrently
    tasks = [process_batch(batch) for batch in molecule_batches]
    results = await asyncio.gather(*tasks)
    
    return pd.concat(results, ignore_index=True)

# Usage
molecule_batches = [molecules[i:i+5000] for i in range(0, len(molecules), 5000)]
results = asyncio.run(async_gpu_processing(molecule_batches))
```

## Network and Distributed Computing Issues

### Connectivity Problems

#### Issue: Node Communication Failures

**Error Messages:**
```
ConnectionError: Failed to connect to worker node 192.168.1.10:8000
TimeoutError: Worker node response timeout
NetworkError: Connection refused by remote host
```

**Diagnostic Commands:**
```bash
# Test network connectivity
ping 192.168.1.10
telnet 192.168.1.10 8000
nmap -p 8000 192.168.1.10

# Check firewall settings
sudo ufw status
sudo iptables -L

# Test port availability
netstat -tulpn | grep 8000
ss -tulpn | grep 8000
```

**Solutions:**
```python
# Solution 1: Robust connection handling
from chemi_accel.distributed import DistributedManager
import time

class RobustDistributedManager:
    def __init__(self, cluster_config):
        self.cluster_config = cluster_config
        self.manager = DistributedManager(cluster_config)
        self.retry_config = {
            'max_retries': 3,
            'retry_delay': 5,
            'backoff_factor': 2
        }
    
    def connect_with_retry(self, node_address):
        """Connect to node with retry logic."""
        
        for attempt in range(self.retry_config['max_retries']):
            try:
                return self.manager.connect_to_node(node_address)
            
            except ConnectionError as e:
                if attempt < self.retry_config['max_retries'] - 1:
                    delay = self.retry_config['retry_delay'] * (self.retry_config['backoff_factor'] ** attempt)
                    print(f"Connection failed to {node_address}, retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    print(f"Failed to connect to {node_address} after {self.retry_config['max_retries']} attempts")
                    raise
    
    def health_check_all_nodes(self):
        """Check health of all nodes."""
        
        healthy_nodes = []
        unhealthy_nodes = []
        
        for node in self.cluster_config.worker_nodes:
            try:
                response = self.manager.ping_node(node, timeout=10)
                if response['status'] == 'healthy':
                    healthy_nodes.append(node)
                else:
                    unhealthy_nodes.append((node, response.get('error', 'Unknown')))
            except Exception as e:
                unhealthy_nodes.append((node, str(e)))
        
        print(f"Healthy nodes: {len(healthy_nodes)}")
        print(f"Unhealthy nodes: {len(unhealthy_nodes)}")
        
        return healthy_nodes, unhealthy_nodes

# Solution 2: Network configuration
def configure_network_settings():
    """Configure network settings for distributed computing."""
    
    network_config = {
        'tcp_keepalive': True,
        'tcp_keepalive_idle': 600,      # 10 minutes
        'tcp_keepalive_interval': 60,   # 1 minute  
        'tcp_keepalive_probes': 3,
        'socket_timeout': 300,          # 5 minutes
        'connection_pool_size': 10,
        'max_retries': 3,
        'retry_backoff': True
    }
    
    return network_config

# Solution 3: Firewall configuration script
def generate_firewall_config(ports):
    """Generate firewall configuration commands."""
    
    commands = [
        "# Configure firewall for ChemiAccel distributed computing",
        "sudo ufw allow ssh",  # Keep SSH access
    ]
    
    for port in ports:
        commands.append(f"sudo ufw allow {port}/tcp")
    
    commands.extend([
        "sudo ufw --force enable",
        "sudo ufw status"
    ])
    
    return "\n".join(commands)

# Generate firewall commands
firewall_script = generate_firewall_config([8000, 8001, 8002])
print("Run these commands on all nodes:")
print(firewall_script)
```

### Load Balancing Issues

#### Issue: Uneven Work Distribution

**Symptoms:**
- Some nodes finish much faster than others
- Overall cluster utilization is low
- Hot spots on specific nodes

**Diagnostic Tools:**
```python
def diagnose_load_balancing(cluster_manager):
    """Diagnose load balancing issues."""
    
    # Get node performance metrics
    node_metrics = {}
    
    for node in cluster_manager.worker_nodes:
        try:
            metrics = cluster_manager.get_node_metrics(node)
            node_metrics[node] = {
                'cpu_utilization': metrics['cpu_utilization'],
                'memory_usage': metrics['memory_usage'],
                'task_queue_size': metrics['task_queue_size'],
                'completed_tasks': metrics['completed_tasks'],
                'average_task_time': metrics['average_task_time']
            }
        except Exception as e:
            node_metrics[node] = {'error': str(e)}
    
    # Analyze load distribution
    utilizations = [m['cpu_utilization'] for m in node_metrics.values() if 'cpu_utilization' in m]
    
    if utilizations:
        avg_utilization = np.mean(utilizations)
        std_utilization = np.std(utilizations)
        
        print(f"Average CPU utilization: {avg_utilization:.1f}%")
        print(f"Utilization std dev: {std_utilization:.1f}%")
        
        if std_utilization > 20:
            print("WARNING: High variance in node utilization detected")
            print("This indicates load balancing issues")
    
    # Print per-node metrics
    for node, metrics in node_metrics.items():
        if 'error' in metrics:
            print(f"{node}: ERROR - {metrics['error']}")
        else:
            print(f"{node}: CPU {metrics['cpu_utilization']:.1f}%, "
                  f"Queue {metrics['task_queue_size']}, "
                  f"Avg task time {metrics['average_task_time']:.2f}s")
    
    return node_metrics
```

**Solutions:**
```python
# Solution 1: Dynamic load balancing
class DynamicLoadBalancer:
    def __init__(self, cluster_manager):
        self.cluster_manager = cluster_manager
        self.node_weights = {}
        self.rebalancing_enabled = True
    
    def update_node_weights(self):
        """Update node weights based on performance."""
        
        for node in self.cluster_manager.worker_nodes:
            try:
                metrics = self.cluster_manager.get_node_metrics(node)
                
                # Calculate weight based on performance
                cpu_factor = max(0.1, 1.0 - metrics['cpu_utilization'] / 100)
                queue_factor = max(0.1, 1.0 / (1 + metrics['task_queue_size']))
                speed_factor = 1.0 / max(0.1, metrics['average_task_time'])
                
                weight = cpu_factor * queue_factor * speed_factor
                self.node_weights[node] = weight
                
            except Exception:
                self.node_weights[node] = 0.1  # Minimum weight for failed nodes
    
    def distribute_tasks(self, tasks):
        """Distribute tasks based on current node weights."""
        
        self.update_node_weights()
        
        # Normalize weights
        total_weight = sum(self.node_weights.values())
        if total_weight == 0:
            return self.distribute_evenly(tasks)
        
        normalized_weights = {node: weight / total_weight 
                            for node, weight in self.node_weights.items()}
        
        # Distribute tasks
        task_distribution = {}
        start_idx = 0
        
        for node, weight in normalized_weights.items():
            num_tasks = int(len(tasks) * weight)
            end_idx = start_idx + num_tasks
            
            task_distribution[node] = tasks[start_idx:end_idx]
            start_idx = end_idx
        
        # Assign remaining tasks to best performing node
        if start_idx < len(tasks):
            best_node = max(self.node_weights.items(), key=lambda x: x[1])[0]
            task_distribution[best_node].extend(tasks[start_idx:])
        
        return task_distribution
    
    def monitor_and_rebalance(self, active_tasks):
        """Monitor task execution and rebalance if needed."""
        
        if not self.rebalancing_enabled:
            return
        
        slow_nodes = []
        fast_nodes = []
        
        for node, tasks in active_tasks.items():
            if not tasks:  # No active tasks
                continue
            
            avg_progress = np.mean([task['progress'] for task in tasks])
            expected_progress = 0.5  # 50% expected at this point
            
            if avg_progress < expected_progress * 0.7:  # More than 30% behind
                slow_nodes.append(node)
            elif avg_progress > expected_progress * 1.3:  # More than 30% ahead
                fast_nodes.append(node)
        
        # Rebalance if needed
        if slow_nodes and fast_nodes:
            self.rebalance_tasks(slow_nodes, fast_nodes, active_tasks)
    
    def rebalance_tasks(self, slow_nodes, fast_nodes, active_tasks):
        """Rebalance tasks from slow to fast nodes."""
        
        print(f"Rebalancing tasks from {len(slow_nodes)} slow nodes to {len(fast_nodes)} fast nodes")
        
        for slow_node in slow_nodes:
            if slow_node not in active_tasks:
                continue
            
            # Find tasks that can be moved
            movable_tasks = [task for task in active_tasks[slow_node] 
                           if task['progress'] < 0.1]  # Less than 10% complete
            
            if not movable_tasks:
                continue
            
            # Move tasks to fastest available node
            best_fast_node = min(fast_nodes, 
                               key=lambda n: len(active_tasks.get(n, [])))
            
            # Move up to 30% of movable tasks
            tasks_to_move = movable_tasks[:max(1, len(movable_tasks) // 3)]
            
            for task in tasks_to_move:
                self.cluster_manager.move_task(task['id'], slow_node, best_fast_node)
                active_tasks[slow_node].remove(task)
                active_tasks[best_fast_node].append(task)
            
            print(f"Moved {len(tasks_to_move)} tasks from {slow_node} to {best_fast_node}")

# Usage
load_balancer = DynamicLoadBalancer(cluster_manager)
task_distribution = load_balancer.distribute_tasks(molecule_tasks)
```

## Cloud and HPC Issues

### Authentication and Access Problems

#### Issue: Cloud Authentication Failures

**Error Messages:**
```
AuthenticationError: Invalid AWS credentials
PermissionError: Access denied to GCP resources
AzureIdentityError: Unable to obtain token
```

**Solutions:**
```python
# Solution 1: Robust credential management
import os
from chemi_accel.cloud import CloudCredentialManager

def setup_cloud_credentials():
    """Set up cloud credentials with fallbacks."""
    
    credential_manager = CloudCredentialManager()
    
    # AWS credentials
    aws_methods = [
        'environment_variables',  # AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
        'aws_profile',           # ~/.aws/credentials
        'iam_role',              # EC2 instance role
        'sts_assume_role'        # Cross-account role
    ]
    
    aws_creds = credential_manager.get_aws_credentials(methods=aws_methods)
    
    # GCP credentials
    gcp_methods = [
        'service_account_key',   # GOOGLE_APPLICATION_CREDENTIALS
        'gcloud_auth',          # gcloud auth
        'compute_metadata'      # GCE metadata service
    ]
    
    gcp_creds = credential_manager.get_gcp_credentials(methods=gcp_methods)
    
    # Azure credentials
    azure_methods = [
        'environment_variables',  # AZURE_CLIENT_ID, etc.
        'azure_cli',             # az login
        'managed_identity'       # Azure managed identity
    ]
    
    azure_creds = credential_manager.get_azure_credentials(methods=azure_methods)
    
    return {
        'aws': aws_creds,
        'gcp': gcp_creds,
        'azure': azure_creds
    }

# Solution 2: Credential validation
def validate_cloud_access():
    """Validate cloud access and permissions."""
    
    validation_results = {}
    
    # Test AWS access
    try:
        import boto3
        
        # Test basic access
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        
        # Test required permissions
        ec2 = boto3.client('ec2')
        ec2.describe_instances(MaxResults=5)  # Test EC2 access
        
        validation_results['aws'] = {
            'status': 'success',
            'account_id': identity['Account'],
            'user_arn': identity['Arn']
        }
        
    except Exception as e:
        validation_results['aws'] = {
            'status': 'failed',
            'error': str(e)
        }
    
    # Test GCP access
    try:
        from google.cloud import compute_v1
        
        compute_client = compute_v1.InstancesClient()
        # This will fail if no access, but validates credentials
        
        validation_results['gcp'] = {'status': 'success'}
        
    except Exception as e:
        validation_results['gcp'] = {
            'status': 'failed',
            'error': str(e)
        }
    
    # Test Azure access
    try:
        from azure.identity import DefaultAzureCredential
        from azure.mgmt.compute import ComputeManagementClient
        
        credential = DefaultAzureCredential()
        # Test credential by creating client
        compute_client = ComputeManagementClient(
            credential, 
            subscription_id=os.environ.get('AZURE_SUBSCRIPTION_ID', 'test')
        )
        
        validation_results['azure'] = {'status': 'success'}
        
    except Exception as e:
        validation_results['azure'] = {
            'status': 'failed',
            'error': str(e)
        }
    
    return validation_results

# Validate access
access_results = validate_cloud_access()
for provider, result in access_results.items():
    status = "✓" if result['status'] == 'success' else "✗"
    print(f"{status} {provider.upper()}: {result['status']}")
    if result['status'] == 'failed':
        print(f"  Error: {result['error']}")
```

## Data and File Format Issues

### Input Validation Problems

#### Issue: Malformed Input Data

**Error Messages:**
```
ValueError: Invalid file format
ParseError: Cannot parse molecule data
FormatError: Unsupported file extension
```

**Solutions:**
```python
def comprehensive_data_validation(input_file):
    """Comprehensive input data validation."""
    
    import os
    from pathlib import Path
    
    validation_results = {
        'file_exists': False,
        'file_readable': False,
        'format_supported': False,
        'content_valid': False,
        'molecule_count': 0,
        'errors': [],
        'warnings': []
    }
    
    # Check file existence
    if not os.path.exists(input_file):
        validation_results['errors'].append(f"File not found: {input_file}")
        return validation_results
    
    validation_results['file_exists'] = True
    
    # Check file readability
    try:
        with open(input_file, 'r') as f:
            f.read(100)  # Test read
        validation_results['file_readable'] = True
    except Exception as e:
        validation_results['errors'].append(f"File not readable: {e}")
        return validation_results
    
    # Check format support
    file_extension = Path(input_file).suffix.lower()
    supported_formats = ['.smi', '.smiles', '.csv', '.sdf', '.mol', '.txt']
    
    if file_extension not in supported_formats:
        validation_results['warnings'].append(
            f"Unsupported extension {file_extension}. Supported: {supported_formats}"
        )
    else:
        validation_results['format_supported'] = True
    
    # Validate content
    try:
        if file_extension in ['.smi', '.smiles', '.txt']:
            molecules = load_smiles_file(input_file)
        elif file_extension == '.csv':
            molecules = load_csv_file(input_file)
        elif file_extension == '.sdf':
            molecules = load_sdf_file(input_file)
        else:
            raise ValueError(f"Cannot parse {file_extension} files")
        
        validation_results['molecule_count'] = len(molecules)
        validation_results['content_valid'] = True
        
        # Validate sample molecules
        sample_size = min(100, len(molecules))
        valid_count = 0
        
        for mol in molecules[:sample_size]:
            if validate_single_molecule(mol):
                valid_count += 1
        
        validity_rate = valid_count / sample_size
        if validity_rate < 0.9:
            validation_results['warnings'].append(
                f"Low molecule validity rate: {validity_rate:.1%}"
            )
        
    except Exception as e:
        validation_results['errors'].append(f"Content validation failed: {e}")
    
    return validation_results

def load_smiles_file(file_path):
    """Load SMILES from various file formats."""
    
    molecules = []
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            if not line or line.startswith('#'):  # Skip empty lines and comments
                continue
            
            # Handle different SMILES file formats
            parts = line.split()
            
            if len(parts) >= 1:
                smiles = parts[0]
                molecules.append(smiles)
            else:
                print(f"Warning: Skipping invalid line {line_num}: '{line}'")
    
    return molecules

def load_csv_file(file_path):
    """Load molecules from CSV file."""
    
    import pandas as pd
    
    try:
        df = pd.read_csv(file_path)
        
        # Try common column names for SMILES
        smiles_columns = ['smiles', 'SMILES', 'mol', 'molecule', 'structure']
        smiles_column = None
        
        for col in smiles_columns:
            if col in df.columns:
                smiles_column = col
                break
        
        if smiles_column is None:
            # Use first column if no standard column found
            smiles_column = df.columns[0]
            print(f"Warning: Using column '{smiles_column}' as SMILES column")
        
        molecules = df[smiles_column].dropna().tolist()
        return molecules
        
    except Exception as e:
        raise ValueError(f"Failed to load CSV file: {e}")

def validate_single_molecule(smiles):
    """Validate a single molecule SMILES."""
    
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            return False
        
        # Additional validation
        if mol.GetNumAtoms() == 0:
            return False
        
        # Try sanitization
        Chem.SanitizeMol(mol)
        return True
        
    except:
        return False

# Usage
validation_result = comprehensive_data_validation("molecules.smi")

if validation_result['errors']:
    print("Validation errors:")
    for error in validation_result['errors']:
        print(f"  ✗ {error}")

if validation_result['warnings']:
    print("Validation warnings:")
    for warning in validation_result['warnings']:
        print(f"  ⚠ {warning}")

if validation_result['content_valid']:
    print(f"✓ Valid input file with {validation_result['molecule_count']} molecules")
```

## Best Practices for Troubleshooting

### 1. Systematic Debugging Approach
```python
def systematic_debug(molecules, descriptors):
    """Systematic debugging approach."""
    
    debug_steps = [
        ("Environment check", check_environment),
        ("Input validation", lambda: validate_molecules(molecules)),
        ("Backend availability", check_backends),
        ("Memory requirements", lambda: estimate_memory_needs(molecules, descriptors)),
        ("Small sample test", lambda: test_small_sample(molecules[:10], descriptors)),
        ("Performance baseline", lambda: benchmark_performance(molecules[:1000], descriptors))
    ]
    
    for step_name, step_function in debug_steps:
        print(f"Running {step_name}...")
        try:
            result = step_function()
            print(f"✓ {step_name}: OK")
        except Exception as e:
            print(f"✗ {step_name}: {e}")
            return False
    
    return True
```

### 2. Comprehensive Logging
```python
import logging

def setup_debug_logging():
    """Set up comprehensive logging for debugging."""
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('chemiaccel_debug.log'),
            logging.StreamHandler()
        ]
    )
    
    # Enable ChemiAccel debug logging
    chemi_logger = logging.getLogger('chemi_accel')
    chemi_logger.setLevel(logging.DEBUG)
    
    return chemi_logger

# Enable debug logging
debug_logger = setup_debug_logging()
```

### 3. Performance Monitoring
```python
def monitor_during_calculation(calc, molecules, descriptors):
    """Monitor system during calculation."""
    
    import threading
    import psutil
    
    monitoring_data = []
    monitoring_active = True
    
    def monitor_resources():
        while monitoring_active:
            data = {
                'timestamp': time.time(),
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_io': psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {},
                'network_io': psutil.net_io_counters()._asdict()
            }
            
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    data['gpu_utilization'] = gpus[0].load * 100
                    data['gpu_memory_percent'] = gpus[0].memoryUtil * 100
            except ImportError:
                pass
            
            monitoring_data.append(data)
            time.sleep(1)
    
    # Start monitoring
    monitor_thread = threading.Thread(target=monitor_resources)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    try:
        # Run calculation
        results = calc.pandas(molecules, descriptors=descriptors)
        return results, monitoring_data
    finally:
        monitoring_active = False
        monitor_thread.join(timeout=5)
```

---

*For specific troubleshooting scenarios, see the individual component guides: [Backend Selection](Backend-Selection), [Cloud Deployment](Cloud-Deployment), [HPC Integration](HPC-Integration), and [Performance Tuning](Performance-Tuning).*
