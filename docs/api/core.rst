Core API Reference
==================

The core module provides the fundamental classes and functions for ChemiAccel's molecular descriptor calculation system.

Calculator
----------

.. automodule:: chemi_accel.core.calculator
    :members:
    :undoc-members:
    :show-inheritance:

Backend Management
------------------

.. automodule:: chemi_accel.core.backend
    :members:
    :undoc-members:
    :show-inheritance:

Molecular Data Structures
--------------------------

.. automodule:: chemi_accel.core.molecular
    :members:
    :undoc-members:
    :show-inheritance:

Exception Handling
------------------

.. automodule:: chemi_accel.core.exceptions
    :members:
    :undoc-members:
    :show-inheritance:

Usage Examples
--------------

Basic Calculator Usage
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel import Calculator

    # Initialize calculator with auto backend selection
    calc = Calculator()
    
    # Process SMILES list
    smiles = ["CCO", "CCN", "CC(=O)O"]
    results = calc.pandas(smiles)
    print(results.head())

Backend-Specific Usage
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel import Calculator

    # CUDA backend for NVIDIA GPUs
    cuda_calc = Calculator(backend="cuda")
    results_cuda = cuda_calc.pandas(smiles)

    # JAX backend for TPUs
    jax_calc = Calculator(backend="jax")
    results_jax = jax_calc.jax(smiles)

    # RAPIDS backend for GPU DataFrames
    rapids_calc = Calculator(backend="rapids")
    results_cudf = rapids_calc.cudf(smiles)

Error Handling
~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel import Calculator
    from chemi_accel.core.exceptions import ChemiAccelError, BackendError

    calc = Calculator()
    
    try:
        results = calc.pandas(["invalid_smiles"])
    except ChemiAccelError as e:
        print(f"ChemiAccel error: {e}")
    except BackendError as e:
        print(f"Backend error: {e}")

Custom Configuration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel import Calculator

    # Custom configuration
    config = {
        'batch_size': 1000,
        'n_jobs': 4,
        'memory_limit': '8GB',
        'descriptors': ['molecular_weight', 'logp', 'tpsa']
    }
    
    calc = Calculator(config=config)
    results = calc.pandas(smiles)