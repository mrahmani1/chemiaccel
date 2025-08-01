Descriptors API Reference
=========================

The descriptors module provides comprehensive molecular descriptor calculation capabilities across multiple domains.

Basic Descriptors
-----------------

.. automodule:: chemi_accel.descriptors.basic
    :members:
    :undoc-members:
    :show-inheritance:

Fingerprints
------------

.. automodule:: chemi_accel.descriptors.fingerprints
    :members:
    :undoc-members:
    :show-inheritance:

3D Geometrical Descriptors
---------------------------

.. automodule:: chemi_accel.descriptors.geometrical_3d
    :members:
    :undoc-members:
    :show-inheritance:

Constitutional Descriptors
--------------------------

.. automodule:: chemi_accel.descriptors.constitutional
    :members:
    :undoc-members:
    :show-inheritance:

Topological Descriptors
-----------------------

.. automodule:: chemi_accel.descriptors.topological
    :members:
    :undoc-members:
    :show-inheritance:

Advanced Descriptors
--------------------

.. automodule:: chemi_accel.descriptors.advanced
    :members:
    :undoc-members:
    :show-inheritance:

Drug Discovery Descriptors
---------------------------

.. automodule:: chemi_accel.descriptors.drug_discovery
    :members:
    :undoc-members:
    :show-inheritance:

ADMET Descriptors
-----------------

.. automodule:: chemi_accel.descriptors.admet
    :members:
    :undoc-members:
    :show-inheritance:

Graph Neural Network Features
-----------------------------

.. automodule:: chemi_accel.descriptors.gnn_features
    :members:
    :undoc-members:
    :show-inheritance:

Solvation Thermodynamics
------------------------

.. automodule:: chemi_accel.descriptors.solvation_thermodynamics
    :members:
    :undoc-members:
    :show-inheritance:

Pharmacophore Screening
-----------------------

.. automodule:: chemi_accel.descriptors.pharmacophore_screening
    :members:
    :undoc-members:
    :show-inheritance:

Reaction Informatics
--------------------

.. automodule:: chemi_accel.descriptors.reaction_informatics
    :members:
    :undoc-members:
    :show-inheritance:

Usage Examples
--------------

Basic Molecular Properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.descriptors.basic import BasicDescriptors

    basic = BasicDescriptors()
    smiles = ["CCO", "CCN", "CC(=O)O"]
    
    # Calculate basic properties
    results = basic.calculate_batch(smiles)
    print(results[['molecular_weight', 'logp', 'tpsa']].head())

Morgan Fingerprints
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.descriptors.fingerprints import MorganFingerprints

    morgan = MorganFingerprints(radius=2, n_bits=2048)
    fingerprints = morgan.calculate_batch(smiles)
    
    # Convert to dense array
    fp_array = morgan.to_dense(fingerprints)
    print(f"Fingerprint shape: {fp_array.shape}")

3D Descriptors
~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.descriptors.geometrical_3d import Geometrical3D

    geom3d = Geometrical3D(use_conformers=True)
    descriptors_3d = geom3d.calculate_batch(smiles)
    
    print(descriptors_3d[['pmi_x', 'pmi_y', 'pmi_z']].head())

Drug Discovery Pipeline
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.descriptors.drug_discovery import DrugDiscoveryDescriptors

    drug_desc = DrugDiscoveryDescriptors()
    drug_props = drug_desc.calculate_batch(smiles)
    
    # Check drug-likeness
    drug_like = drug_desc.lipinski_filter(drug_props)
    print(f"Drug-like compounds: {drug_like.sum()}")

GPU-Accelerated Calculations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel import Calculator

    # Use CUDA backend for GPU acceleration
    calc = Calculator(backend="cuda")
    
    # Large dataset processing
    large_smiles = ["CCO"] * 100000
    results = calc.pandas(large_smiles, descriptors=[
        'molecular_weight', 'logp', 'tpsa', 'morgan_fingerprint'
    ])
    
    print(f"Processed {len(results)} molecules on GPU")

Custom Descriptor Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel import Calculator

    calc = Calculator()
    
    # Select specific descriptors
    custom_descriptors = [
        'molecular_weight',
        'logp', 
        'tpsa',
        'num_rotatable_bonds',
        'num_hbd',
        'num_hba'
    ]
    
    results = calc.pandas(smiles, descriptors=custom_descriptors)
    print(results.columns.tolist())