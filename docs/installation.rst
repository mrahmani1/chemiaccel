Installation Guide
==================

ChemiAccel supports multiple installation options depending on your hardware and use case requirements.

System Requirements
-------------------

**Minimum Requirements:**
- Python 3.8 or higher (3.9, 3.10, 3.11 supported)
- 8 GB RAM
- 10 GB disk space

**Recommended for GPU acceleration:**
- NVIDIA GPU with Compute Capability 7.0+ (RTX 20xx series or newer)
- CUDA Toolkit 11.x
- 16 GB RAM
- 50 GB disk space

**For TPU/JAX backend:**
- Google Cloud TPU access or compatible hardware
- JAX with XLA support

Installation Options
--------------------

Basic Installation (CPU only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For basic molecular descriptor calculations using CPU backend:

.. code-block:: bash

    pip install chemi-accel

This installs core dependencies including RDKit, NumPy, Pandas, and SciPy.

GPU-Accelerated Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For NVIDIA GPU acceleration with CUDA backend:

.. code-block:: bash

    # Install CUDA dependencies
    pip install chemi-accel[cuda]

This includes CuPy, cuDF, cuGraph, and Numba for GPU acceleration.

For JAX/TPU backend:

.. code-block:: bash

    # Install JAX dependencies
    pip install chemi-accel[jax]

This includes JAX, JAXlib, and Optax for TPU/XLA acceleration.

For RAPIDS integration:

.. code-block:: bash

    # Install RAPIDS dependencies
    pip install chemi-accel[rapids]

This includes cuDF, cuPy, and cuGraph for GPU-native DataFrames.

Complete Installation
~~~~~~~~~~~~~~~~~~~~~

For all features including development tools:

.. code-block:: bash

    # Install all dependencies
    pip install chemi-accel[all]

This includes all backends, cloud integrations, and development tools.

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

For contributors and developers:

.. code-block:: bash

    # Clone repository
    git clone https://github.com/chemi-accel/chemi-accel.git
    cd chemi-accel

    # Install in development mode
    pip install -e .[dev]

This includes testing frameworks, linting tools, and documentation generators.

Verification
------------

Test your installation:

.. code-block:: python

    import chemi_accel
    from chemi_accel import Calculator

    # Test basic functionality
    calc = Calculator()
    print(f"ChemiAccel version: {chemi_accel.__version__}")
    print(f"Available backends: {calc.available_backends}")

    # Test with simple molecule
    smiles = ["CCO", "CCN"]
    results = calc.pandas(smiles)
    print(results.head())

GPU Backend Verification
~~~~~~~~~~~~~~~~~~~~~~~~

Test CUDA backend:

.. code-block:: python

    from chemi_accel import Calculator

    # Check CUDA availability
    calc = Calculator(backend="cuda")
    print(f"CUDA backend available: {calc.backend.is_available()}")

    # Run GPU-accelerated calculation
    smiles = ["CCO"] * 1000
    results = calc.pandas(smiles)
    print(f"Processed {len(results)} molecules on GPU")

Test JAX backend:

.. code-block:: python

    from chemi_accel import Calculator

    # Check JAX availability
    calc = Calculator(backend="jax")
    print(f"JAX backend available: {calc.backend.is_available()}")

    # Check available devices
    import jax
    print(f"JAX devices: {jax.devices()}")

Docker Installation
-------------------

For containerized deployment:

.. code-block:: bash

    # Pull latest image
    docker pull chemi-accel/chemi-accel:latest

    # Run with GPU support
    docker run --gpus all -it chemi-accel/chemi-accel:latest

Conda Installation
------------------

For Conda environments:

.. code-block:: bash

    # Create environment
    conda create -n chemi-accel python=3.10
    conda activate chemi-accel

    # Install from conda-forge
    conda install -c conda-forge chemi-accel

    # Or install with pip in conda environment
    pip install chemi-accel[cuda]

Troubleshooting
---------------

**CUDA Installation Issues:**

1. Verify CUDA toolkit installation:

.. code-block:: bash

    nvcc --version
    nvidia-smi

2. Check CuPy installation:

.. code-block:: python

    import cupy
    print(cupy.cuda.runtime.runtimeGetVersion())

**JAX Installation Issues:**

1. Verify JAX installation:

.. code-block:: python

    import jax
    print(jax.lib.xla_bridge.get_backend().platform)

**Memory Issues:**

- Reduce batch size for large datasets
- Use streaming processing for very large datasets
- Monitor GPU memory usage with ``nvidia-smi``

**Import Errors:**

- Ensure all dependencies are installed for your chosen backend
- Check Python version compatibility
- Verify virtual environment activation

For more detailed troubleshooting, see the :doc:`troubleshooting` guide.