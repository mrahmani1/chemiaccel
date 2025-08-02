# Configuration Guide

This guide covers how to configure ChemiAccel for optimal performance in your specific environment, including backend selection, resource management, and advanced settings.

## Configuration Overview

ChemiAccel can be configured through multiple methods:
- **Environment variables** - System-wide settings
- **Configuration files** - Project-specific settings  
- **Runtime parameters** - Code-level configuration
- **Cloud platform settings** - Cloud-specific configuration

## Environment Variables

### Core Settings

```bash
# Set the default backend
export CHEMIACCEL_DEFAULT_BACKEND=cuda

# Set maximum memory usage (in GB)
export CHEMIACCEL_MAX_MEMORY_GB=16

# Set number of CPU threads
export CHEMIACCEL_NUM_THREADS=8

# Enable/disable GPU acceleration
export CHEMIACCEL_USE_GPU=true

# Set CUDA device (for multi-GPU systems)
export CUDA_VISIBLE_DEVICES=0,1
```

### Backend-Specific Settings

```bash
# CUDA Backend Configuration
export CHEMIACCEL_CUDA_MEMORY_POOL=16384  # MB
export CHEMIACCEL_CUDA_STREAMS=4

# JAX Backend Configuration  
export JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# CPU Backend Configuration
export CHEMIACCEL_CPU_BATCH_SIZE=1000
export OMP_NUM_THREADS=16
```

### Cloud Configuration

```bash
# AWS Configuration
export AWS_DEFAULT_REGION=us-east-1
export CHEMIACCEL_S3_BUCKET=my-chemiaccel-data

# GCP Configuration
export GOOGLE_CLOUD_PROJECT=my-project-id
export CHEMIACCEL_GCS_BUCKET=my-chemiaccel-data

# Azure Configuration
export AZURE_SUBSCRIPTION_ID=your-subscription-id
export CHEMIACCEL_BLOB_CONTAINER=chemiaccel-data
```

## Configuration Files

### Local Configuration (.chemiaccel.yaml)

Create a `.chemiaccel.yaml` file in your project directory:

```yaml
# ChemiAccel Configuration
version: "1.0"

# Default backend selection
backend:
  default: "cuda"
  fallback: "cpu"
  
# Resource limits
resources:
  max_memory_gb: 16
  num_threads: 8
  gpu_memory_fraction: 0.8
  
# Performance settings  
performance:
  batch_size: 1000
  cache_size_mb: 2048
  enable_jit: true
  
# Logging configuration
logging:
  level: "INFO"
  file: "chemiaccel.log"
  max_size_mb: 100
  
# Backend-specific settings
backends:
  cuda:
    memory_pool_mb: 16384
    streams: 4
    device_ids: [0, 1]
    
  jax:
    platform: "gpu"
    preallocate: false
    memory_fraction: 0.75
    
  cpu:
    num_threads: 16
    batch_size: 500
    
# Cloud settings
cloud:
  aws:
    region: "us-east-1"
    s3_bucket: "my-chemiaccel-data"
    instance_type: "g4dn.xlarge"
    
  gcp:
    project_id: "my-project-id"
    zone: "us-central1-a"
    machine_type: "n1-standard-4"
    
  azure:
    resource_group: "chemiaccel-rg"
    location: "eastus"
    vm_size: "Standard_NC6s_v3"

# Data management
data:
  cache_directory: "./cache"
  temp_directory: "/tmp/chemiaccel"
  max_cache_size_gb: 10
  cleanup_on_exit: true
  
# Security settings
security:
  enable_encryption: false
  key_size: 2048
  hash_algorithm: "sha256"
```

### Global Configuration (~/.chemiaccel/config.yaml)

Create a global configuration file in your home directory:

```yaml
# Global ChemiAccel Configuration
user:
  name: "Your Name"
  email: "your.email@example.com"
  organization: "Your Organization"

# Default preferences
preferences:
  default_backend: "cuda"
  output_format: "pandas"
  precision: "float32"
  
# License and authentication
license:
  key: "your-license-key"
  type: "academic"  # or "commercial"
  
# Cloud credentials (optional - prefer environment variables)
credentials:
  aws_profile: "default"
  gcp_service_account: "path/to/service-account.json"
  azure_tenant_id: "your-tenant-id"
```

## Runtime Configuration

### Basic Calculator Configuration

```python
from chemi_accel import Calculator
from chemi_accel.core.config import ChemiAccelConfig

# Create custom configuration
config = ChemiAccelConfig(
    backend="cuda",
    max_memory_gb=16,
    num_threads=8,
    batch_size=1000,
    enable_cache=True,
    cache_size_mb=2048
)

# Initialize calculator with configuration
calc = Calculator(config=config)
```

### Advanced Configuration

```python
from chemi_accel.core.config import (
    ChemiAccelConfig, 
    BackendConfig, 
    PerformanceConfig,
    CloudConfig
)

# Backend-specific configuration
backend_config = BackendConfig(
    cuda_memory_pool_mb=16384,
    cuda_streams=4,
    jax_preallocate=False,
    cpu_num_threads=16
)

# Performance optimization
perf_config = PerformanceConfig(
    batch_size=1000,
    enable_jit=True,
    memory_mapping=True,
    prefetch_data=True
)

# Cloud configuration
cloud_config = CloudConfig(
    provider="aws",
    region="us-east-1",
    instance_type="g4dn.xlarge",
    auto_scaling=True
)

# Complete configuration
config = ChemiAccelConfig(
    backend="cuda",
    backend_config=backend_config,
    performance_config=perf_config,
    cloud_config=cloud_config
)

calc = Calculator(config=config)
```

## Backend Configuration

### CUDA Backend

```python
from chemi_accel.backends.cuda import CudaConfig

cuda_config = CudaConfig(
    device_ids=[0, 1],          # Use GPUs 0 and 1
    memory_pool_mb=16384,       # Pre-allocate 16GB
    streams=4,                  # Number of CUDA streams
    enable_peer_access=True,    # Enable P2P memory access
    compute_mode="default"      # or "exclusive_thread"
)

calc = Calculator(backend="cuda", backend_config=cuda_config)
```

### JAX Backend

```python
from chemi_accel.backends.jax import JaxConfig

jax_config = JaxConfig(
    platform="gpu",             # "gpu", "tpu", or "cpu"
    preallocate=False,          # Allocate memory on-demand
    memory_fraction=0.75,       # Use 75% of available memory
    enable_x64=False,           # Use 32-bit precision
    jit_cache_size=1000        # JIT compilation cache size
)

calc = Calculator(backend="jax", backend_config=jax_config)
```

### CPU Backend

```python
from chemi_accel.backends.cpu import CpuConfig

cpu_config = CpuConfig(
    num_threads=16,             # Number of CPU threads
    batch_size=500,             # Batch size for processing
    enable_mkl=True,            # Use Intel MKL if available
    memory_mapping=True,        # Use memory mapping for large datasets
    numa_policy="interleave"    # NUMA memory allocation policy
)

calc = Calculator(backend="cpu", backend_config=cpu_config)
```

## Performance Tuning

### Memory Management

```python
# Optimize memory usage
config = ChemiAccelConfig(
    max_memory_gb=32,           # Maximum memory usage
    memory_growth=True,         # Allow memory growth
    memory_fraction=0.8,        # Use 80% of available memory
    enable_memory_pool=True,    # Use memory pooling
    garbage_collection=True     # Enable automatic cleanup
)
```

### Batch Processing

```python
# Optimize batch sizes based on your hardware
config = ChemiAccelConfig(
    batch_size="auto",          # Auto-determine optimal batch size
    adaptive_batching=True,     # Adjust batch size dynamically
    min_batch_size=100,         # Minimum batch size
    max_batch_size=10000        # Maximum batch size
)
```

### Caching Configuration

```python
from chemi_accel.core.cache import CacheConfig

cache_config = CacheConfig(
    enable_cache=True,
    cache_directory="./cache",
    max_cache_size_gb=50,
    cache_compression=True,
    cache_encryption=False,
    ttl_hours=24,               # Time-to-live for cached results
    cleanup_on_startup=True
)

config = ChemiAccelConfig(cache_config=cache_config)
```

## Cloud Configuration

### AWS Configuration

```python
from chemi_accel.cloud.aws import AwsConfig

aws_config = AwsConfig(
    region="us-east-1",
    profile="default",
    s3_bucket="my-chemiaccel-data",
    ec2_instance_type="g4dn.xlarge",
    ecs_cluster="chemiaccel-cluster",
    batch_job_queue="chemiaccel-queue",
    enable_spot_instances=True,
    max_spot_price=0.50
)
```

### Google Cloud Configuration

```python
from chemi_accel.cloud.gcp import GcpConfig

gcp_config = GcpConfig(
    project_id="my-project-id",
    zone="us-central1-a",
    service_account_path="path/to/service-account.json",
    gcs_bucket="my-chemiaccel-data",
    compute_machine_type="n1-standard-4",
    vertex_ai_region="us-central1",
    enable_preemptible=True
)
```

### Azure Configuration

```python
from chemi_accel.cloud.azure import AzureConfig

azure_config = AzureConfig(
    subscription_id="your-subscription-id",
    resource_group="chemiaccel-rg",
    location="eastus",
    storage_account="chemiaccelstorage",
    vm_size="Standard_NC6s_v3",
    batch_pool_id="chemiaccel-pool",
    enable_low_priority=True
)
```

## Logging Configuration

### Basic Logging

```python
import logging
from chemi_accel.core.logging import setup_logging

# Configure logging
setup_logging(
    level=logging.INFO,
    file="chemiaccel.log",
    max_size_mb=100,
    backup_count=5,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
```

### Advanced Logging

```python
from chemi_accel.core.logging import LoggingConfig

logging_config = LoggingConfig(
    level="INFO",
    handlers={
        "console": {
            "level": "INFO",
            "format": "%(levelname)s: %(message)s"
        },
        "file": {
            "level": "DEBUG", 
            "filename": "chemiaccel.log",
            "max_bytes": 100 * 1024 * 1024,  # 100MB
            "backup_count": 5
        },
        "syslog": {
            "level": "WARNING",
            "address": ("localhost", 514)
        }
    },
    loggers={
        "chemi_accel.core": "DEBUG",
        "chemi_accel.backends": "INFO",
        "chemi_accel.cloud": "WARNING"
    }
)
```

## Configuration Validation

### Validate Configuration

```python
from chemi_accel.core.config import validate_config

# Validate configuration before use
config = ChemiAccelConfig(backend="cuda", max_memory_gb=16)

validation_result = validate_config(config)
if not validation_result.is_valid:
    print(f"Configuration errors: {validation_result.errors}")
    exit(1)

# Configuration is valid
calc = Calculator(config=config)
```

### Configuration Templates

```python
from chemi_accel.core.config import ConfigTemplate

# Use predefined templates
config = ConfigTemplate.high_performance_gpu()
# or
config = ConfigTemplate.cost_optimized_cloud()
# or  
config = ConfigTemplate.development_local()

calc = Calculator(config=config)
```

## Best Practices

### Development Environment

```yaml
# .chemiaccel.yaml for development
backend:
  default: "cpu"
  
resources:
  max_memory_gb: 8
  
logging:
  level: "DEBUG"
  
performance:
  batch_size: 100
  enable_jit: false  # Faster startup for development
```

### Production Environment

```yaml
# .chemiaccel.yaml for production
backend:
  default: "cuda"
  fallback: "cpu"
  
resources:
  max_memory_gb: 64
  
logging:
  level: "INFO"
  file: "/var/log/chemiaccel.log"
  
performance:
  batch_size: "auto"
  enable_jit: true
  
data:
  cache_directory: "/data/chemiaccel/cache"
  max_cache_size_gb: 100
```

### Security Considerations

```yaml
security:
  enable_encryption: true
  key_size: 4096
  hash_algorithm: "sha256"
  
# Don't store credentials in config files
# Use environment variables or secure key management
```

## Troubleshooting Configuration

### Common Issues

1. **Backend not available**: Check GPU drivers and CUDA installation
2. **Memory errors**: Reduce `max_memory_gb` and `batch_size`
3. **Performance issues**: Enable JIT compilation and caching
4. **Cloud authentication**: Verify credentials and permissions

### Configuration Debugging

```python
from chemi_accel.core.config import debug_config

# Debug current configuration
debug_config()

# Check backend availability
from chemi_accel import Calculator
calc = Calculator()
print(f"Available backends: {calc.available_backends}")
```

---

*For backend-specific optimization, see the [Backend Selection](Backend-Selection) guide.*
