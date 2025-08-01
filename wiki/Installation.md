# Installation Guide

This guide covers installing ChemiAccel on different platforms and with various computational backends.

## System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher (3.9+ recommended)
- **RAM**: 8 GB minimum, 16 GB recommended
- **Storage**: 2 GB free space
- **OS**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 18.04+)

### For GPU Acceleration
- **NVIDIA GPU**: CUDA-capable with Compute Capability 6.0+
- **CUDA**: Version 11.0 or higher
- **GPU Memory**: 4 GB minimum, 8 GB+ recommended

### For Cloud/HPC Usage
- **Network**: Stable internet connection for cloud deployments
- **Authentication**: Appropriate cloud credentials (AWS, GCP, Azure)

## Quick Installation

For most users, the simplest installation is:

```bash
pip install chemi-accel
```

This installs ChemiAccel with CPU support and basic dependencies.

## Installation Options

### 1. Basic Installation (CPU only)

```bash
pip install chemi-accel
```

**Includes:**
- Core ChemiAccel functionality
- CPU-based descriptor calculation
- Basic visualization tools
- RDKit integration

### 2. GPU-Accelerated Installation

#### CUDA Backend
```bash
pip install chemi-accel[cuda]
```

**Additional components:**
- CuPy for GPU arrays
- RAPIDS (cuDF, cuGraph) for GPU DataFrames  
- Custom CUDA kernels for molecular descriptors
- GPU-accelerated fingerprint calculations

#### JAX Backend (GPU/TPU)
```bash
pip install chemi-accel[jax]
```

**Additional components:**
- JAX with GPU/TPU support
- JIT-compiled descriptor functions
- Automatic differentiation capabilities
- TPU support for Google Cloud

### 3. Full Installation

```bash
pip install chemi-accel[all]
```

**Includes everything:**
- All computational backends (CPU, CUDA, JAX)
- Cloud platform integrations
- Advanced visualization tools
- Development dependencies

### 4. Development Installation

```bash
git clone https://github.com/chemiaccel/chemiaccel.git
cd chemiaccel  
pip install -e .[dev]
```

**Additional development tools:**
- Testing frameworks (pytest, hypothesis)
- Code formatting (black, isort)  
- Type checking (mypy)
- Documentation tools (Sphinx)

## Platform-Specific Instructions

### Windows

#### Prerequisites
```bash
# Install Visual Studio Build Tools (required for some dependencies)
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Install Miniconda (recommended)
# Download from: https://docs.conda.io/en/latest/miniconda.html
```

#### Installation
```bash
# Create conda environment
conda create -n chemiaccel python=3.10
conda activate chemiaccel

# Install ChemiAccel
pip install chemi-accel[cuda]  # For GPU support
```

### macOS

#### Prerequisites
```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and required tools
brew install python@3.10
```

#### Installation
```bash
# Create virtual environment
python3.10 -m venv chemiaccel-env
source chemiaccel-env/bin/activate

# Install ChemiAccel
pip install chemi-accel

# Note: CUDA support not available on macOS
# Use JAX backend for Apple Silicon GPU acceleration:
pip install chemi-accel[jax]
```

### Linux (Ubuntu/Debian)

#### Prerequisites
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and development tools
sudo apt install python3.10 python3.10-venv python3.10-dev
sudo apt install build-essential git curl

# For GPU support, install CUDA Toolkit
# Follow NVIDIA's official installation guide:
# https://developer.nvidia.com/cuda-downloads
```

#### Installation
```bash
# Create virtual environment
python3.10 -m venv chemiaccel-env
source chemiaccel-env/bin/activate

# Install ChemiAccel with GPU support
pip install chemi-accel[cuda]
```

### Linux (CentOS/RHEL)

```bash
# Install Python and development tools
sudo yum groupinstall -y "Development Tools"
sudo yum install -y python3 python3-pip python3-devel

# Create virtual environment and install
python3 -m venv chemiaccel-env
source chemiaccel-env/bin/activate
pip install --upgrade pip
pip install chemi-accel[cuda]
```

## Docker Installation

### Pre-built Images

```bash
# CPU-only image
docker pull chemiaccel/chemiaccel:latest

# GPU-enabled image
docker pull chemiaccel/chemiaccel:gpu

# Run with GPU support
docker run --gpus all -it chemiaccel/chemiaccel:gpu
```

### Build from Source

```bash
git clone https://github.com/chemiaccel/chemiaccel.git
cd chemiaccel

# Build CPU image
docker build -t chemiaccel:cpu .

# Build GPU image  
docker build -f docker/Dockerfile.gpu -t chemiaccel:gpu .
```

## Cloud Platform Setup

### AWS SageMaker

```bash
# Install SageMaker SDK
pip install sagemaker boto3

# Configure AWS credentials
aws configure
```

### Google Cloud Platform

```bash
# Install GCP SDK
pip install google-cloud-aiplatform

# Authenticate
gcloud auth application-default login
```

### Microsoft Azure

```bash  
# Install Azure ML SDK
pip install azureml-sdk

# Configure authentication
az login
```

## Verification

After installation, verify ChemiAccel is working correctly:

```python
import chemi_accel
from chemi_accel import Calculator

# Test basic functionality
calc = Calculator()
print(f"ChemiAccel version: {chemi_accel.__version__}")
print(f"Available backends: {calc.available_backends}")

# Test with sample molecule
smiles = ["CCO", "CC(=O)O", "c1ccccc1"]  # ethanol, acetic acid, benzene
results = calc.pandas(smiles)
print(f"Calculated descriptors for {len(results)} molecules")
print(results.head())
```

### GPU Verification

```python
# Test CUDA backend
if 'cuda' in calc.available_backends:
    calc_cuda = Calculator(backend='cuda')
    results_gpu = calc_cuda.pandas(smiles)
    print("CUDA backend working correctly!")

# Test JAX backend  
if 'jax' in calc.available_backends:
    calc_jax = Calculator(backend='jax')
    results_jax = calc_jax.pandas(smiles)
    print("JAX backend working correctly!")
```

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# If you get import errors, try upgrading pip and reinstalling
pip install --upgrade pip setuptools wheel
pip uninstall chemi-accel
pip install chemi-accel[all]
```

#### CUDA Issues
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Verify CuPy installation
python -c "import cupy; print(cupy.cuda.runtime.runtimeGetVersion())"
```

#### Memory Issues
```bash
# For large datasets, increase swap space or use chunked processing
# Set environment variable to limit memory usage:
export CHEMIACCEL_MAX_MEMORY_GB=8
```

#### Permission Issues (Linux/macOS)
```bash
# If you get permission errors, use virtual environment
python -m venv chemiaccel-env
source chemiaccel-env/bin/activate
pip install chemi-accel
```

### Getting Help

If you encounter issues:

1. **Check the [FAQ](FAQ)** for common solutions
2. **Search existing [issues](https://github.com/chemiaccel/chemiaccel/issues)**
3. **Create a new issue** with your system details and error messages
4. **Join our [community discussions](https://github.com/chemiaccel/community/discussions)**

### System Information

When reporting issues, include this information:

```python
import chemi_accel
import sys
import platform

print(f"ChemiAccel version: {chemi_accel.__version__}")
print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")

# Check backends
calc = Calculator()
print(f"Available backends: {calc.available_backends}")

# Check GPU availability
try:
    import cupy
    print(f"CUDA available: {cupy.cuda.is_available()}")
    if cupy.cuda.is_available():
        print(f"GPU count: {cupy.cuda.runtime.getDeviceCount()}")
except ImportError:
    print("CuPy not installed")

try:
    import jax
    print(f"JAX devices: {jax.devices()}")
except ImportError:
    print("JAX not installed")
```

## Next Steps

After successful installation:

1. **Read the [Basic Usage](Basic-Usage) guide**
2. **Try the [Examples](Examples)**  
3. **Configure your preferred [Backend](Backend-Selection)**
4. **Explore domain-specific guides** for your use case

---

*For development setup, see the [Development Setup](Development-Setup) guide.*
