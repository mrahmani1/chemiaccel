Backends API Reference
======================

ChemiAccel supports multiple computational backends for optimal performance across different hardware configurations.

CUDA Backend
------------

GPU acceleration for NVIDIA hardware using CUDA kernels and RAPIDS ecosystem.

.. automodule:: chemi_accel.backends.cuda
    :members:
    :undoc-members:
    :show-inheritance:

CUDA Kernels
~~~~~~~~~~~~

.. automodule:: chemi_accel.backends.cuda.kernels
    :members:
    :undoc-members:
    :show-inheritance:

CUDA Memory Management
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: chemi_accel.backends.cuda.memory
    :members:
    :undoc-members:
    :show-inheritance:

CUDA Streaming
~~~~~~~~~~~~~~

.. automodule:: chemi_accel.backends.cuda.streaming
    :members:
    :undoc-members:
    :show-inheritance:

JAX Backend
-----------

TPU and XLA acceleration using JAX for functional programming and automatic differentiation.

.. automodule:: chemi_accel.backends.jax
    :members:
    :undoc-members:
    :show-inheritance:

JAX Molecular Operations
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: chemi_accel.backends.jax.molecular
    :members:
    :undoc-members:
    :show-inheritance:

JAX Batch Processing
~~~~~~~~~~~~~~~~~~~

.. automodule:: chemi_accel.backends.jax.batch_processing
    :members:
    :undoc-members:
    :show-inheritance:

JAX Vectorization
~~~~~~~~~~~~~~~~~

.. automodule:: chemi_accel.backends.jax.vectorization
    :members:
    :undoc-members:
    :show-inheritance:

RAPIDS Backend
--------------

GPU-native DataFrames and processing using cuDF and cuPy.

.. automodule:: chemi_accel.backends.rapids
    :members:
    :undoc-members:
    :show-inheritance:

cuDF Integration
~~~~~~~~~~~~~~~

.. automodule:: chemi_accel.backends.rapids.cudf_integration
    :members:
    :undoc-members:
    :show-inheritance:

cuPy Operations
~~~~~~~~~~~~~~

.. automodule:: chemi_accel.backends.rapids.cupy_operations
    :members:
    :undoc-members:
    :show-inheritance:

Usage Examples
--------------

Backend Selection
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel import Calculator

    # Automatic backend selection
    calc = Calculator()
    print(f"Selected backend: {calc.backend.name}")

    # Manual backend selection
    cuda_calc = Calculator(backend="cuda")
    jax_calc = Calculator(backend="jax")
    rapids_calc = Calculator(backend="rapids")
    cpu_calc = Calculator(backend="cpu")

Backend Availability Check
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.core.backend import BackendManager

    manager = BackendManager()
    
    # Check available backends
    available = manager.available_backends()
    print(f"Available backends: {available}")
    
    # Check specific backend
    if manager.is_backend_available("cuda"):
        print("CUDA backend is available")
    else:
        print("CUDA backend not available")

CUDA Backend Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel import Calculator

    # CUDA-specific configuration
    cuda_config = {
        'device_id': 0,
        'memory_pool_size': '4GB',
        'stream_count': 4,
        'kernel_cache_size': 100
    }
    
    calc = Calculator(backend="cuda", backend_config=cuda_config)
    
    # Check GPU memory usage
    memory_info = calc.backend.get_memory_info()
    print(f"GPU memory: {memory_info}")

JAX Backend Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel import Calculator
    import jax

    # Configure JAX for TPU
    jax_config = {
        'platform': 'tpu',
        'jit_compile': True,
        'precision': 'float32'
    }
    
    calc = Calculator(backend="jax", backend_config=jax_config)
    
    # Check available devices
    devices = jax.devices()
    print(f"JAX devices: {devices}")

RAPIDS Backend Usage
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel import Calculator

    # RAPIDS backend for GPU DataFrames
    calc = Calculator(backend="rapids")
    
    smiles = ["CCO", "CCN", "CC(=O)O"] * 1000
    
    # Get results as cuDF DataFrame
    cudf_results = calc.cudf(smiles)
    print(f"Result type: {type(cudf_results)}")
    print(f"GPU memory usage: {cudf_results.memory_usage().sum()} bytes")

Performance Comparison
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import time
    from chemi_accel import Calculator

    # Test dataset
    large_smiles = ["CCO", "CCN", "CC(=O)O"] * 10000
    
    backends = ["cpu", "cuda", "jax", "rapids"]
    results = {}
    
    for backend in backends:
        try:
            calc = Calculator(backend=backend)
            start_time = time.time()
            calc.pandas(large_smiles)
            elapsed = time.time() - start_time
            results[backend] = elapsed
            print(f"{backend}: {elapsed:.2f} seconds")
        except Exception as e:
            print(f"{backend}: Not available ({e})")

Backend Memory Management
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel import Calculator

    calc = Calculator(backend="cuda")
    
    # Enable memory monitoring
    calc.backend.enable_memory_monitoring()
    
    # Process data
    results = calc.pandas(large_smiles)
    
    # Get memory statistics
    stats = calc.backend.get_memory_stats()
    print(f"Peak memory usage: {stats['peak_memory_mb']} MB")
    print(f"Current memory usage: {stats['current_memory_mb']} MB")
    
    # Clear memory cache
    calc.backend.clear_memory_cache()