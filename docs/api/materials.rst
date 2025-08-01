Materials Science API Reference
=================================

Comprehensive materials analysis including crystal structures, periodic descriptors, and solid-state properties.

Materials Calculator
--------------------

.. automodule:: chemi_accel.materials.materials_calculator
    :members:
    :undoc-members:
    :show-inheritance:

Crystal Structure Parser
------------------------

.. automodule:: chemi_accel.materials.crystal_parser
    :members:
    :undoc-members:
    :show-inheritance:

Crystal Descriptors
-------------------

.. automodule:: chemi_accel.materials.crystal_descriptors
    :members:
    :undoc-members:
    :show-inheritance:

Periodic Descriptors
--------------------

.. automodule:: chemi_accel.materials.periodic_descriptors
    :members:
    :undoc-members:
    :show-inheritance:

Solid State Properties
----------------------

.. automodule:: chemi_accel.materials.solid_state_properties
    :members:
    :undoc-members:
    :show-inheritance:

Intermolecular Interactions
---------------------------

.. automodule:: chemi_accel.materials.intermolecular_interactions
    :members:
    :undoc-members:
    :show-inheritance:

Crystal Structure Prediction
----------------------------

.. automodule:: chemi_accel.materials.crystal_structure_prediction
    :members:
    :undoc-members:
    :show-inheritance:

Polymorph Inverse Design
------------------------

.. automodule:: chemi_accel.materials.polymorph_inverse_design
    :members:
    :undoc-members:
    :show-inheritance:

Experimental Validation
-----------------------

.. automodule:: chemi_accel.materials.experimental_validation
    :members:
    :undoc-members:
    :show-inheritance:

Validation Framework
--------------------

.. automodule:: chemi_accel.materials.validation
    :members:
    :undoc-members:
    :show-inheritance:

Usage Examples
--------------

CIF File Processing
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.materials import MaterialsCalculator

    calc = MaterialsCalculator(use_gpu=True)
    
    # Load crystal structure from CIF file
    structure_data = calc.from_cif_file("quartz.cif")
    
    print(f"Space group: {structure_data['space_group']}")
    print(f"Unit cell volume: {structure_data['unit_cell_volume']:.2f} Å³")
    print(f"Density: {structure_data['density']:.3f} g/cm³")

POSCAR File Analysis
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.materials.crystal_parser import POSCARParser

    parser = POSCARParser()
    
    # Parse VASP POSCAR file
    structure = parser.parse_poscar("POSCAR")
    
    print(f"Chemical formula: {structure.composition}")
    print(f"Lattice parameters: {structure.lattice.abc}")
    print(f"Lattice angles: {structure.lattice.angles}")

Crystal Descriptor Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.materials.crystal_descriptors import CrystalDescriptors

    descriptor_calc = CrystalDescriptors()
    
    # Calculate descriptors from structure
    descriptors = descriptor_calc.calculate_descriptors(structure)
    
    print(f"Packing efficiency: {descriptors['packing_efficiency']:.3f}")
    print(f"Coordination number: {descriptors['avg_coordination_number']:.1f}")
    print(f"Bond angle variance: {descriptors['bond_angle_variance']:.2f}")

Periodic Boundary Conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.materials.periodic_descriptors import PeriodicDescriptors

    periodic_calc = PeriodicDescriptors()
    
    # Calculate periodic descriptors
    periodic_props = periodic_calc.calculate_periodic_properties(structure)
    
    print(f"Radial distribution function peak: {periodic_props['rdf_first_peak']:.2f} Å")
    print(f"Structure factor intensity: {periodic_props['structure_factor_max']:.2f}")
    print(f"Voronoi volume variance: {periodic_props['voronoi_volume_variance']:.3f}")

Batch Crystal Processing
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.materials import MaterialsCalculator

    calc = MaterialsCalculator(backend="cuda")
    
    # Process multiple CIF files
    cif_directory = "crystal_structures/"
    results = calc.batch_process_directory(cif_directory, "*.cif")
    
    # Analyze results
    print(f"Processed {len(results)} crystal structures")
    print(results[['space_group', 'density', 'band_gap']].head())

Solid State Property Prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.materials.solid_state_properties import SolidStateProperties

    prop_calc = SolidStateProperties()
    
    # Predict electronic properties
    electronic_props = prop_calc.predict_electronic_properties(structure)
    
    print(f"Predicted band gap: {electronic_props['band_gap']:.2f} eV")
    print(f"Effective mass (electron): {electronic_props['electron_mass']:.3f} m₀")
    print(f"Dielectric constant: {electronic_props['dielectric_constant']:.2f}")

Crystal Structure Prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.materials.crystal_structure_prediction import CrystalPredictor

    predictor = CrystalPredictor()
    
    # Predict crystal structure from composition
    composition = "SiO2"
    predicted_structures = predictor.predict_structures(
        composition=composition,
        max_structures=10,
        space_groups=['P1', 'P21/c', 'Pnma', 'I41/amd']
    )
    
    for i, structure in enumerate(predicted_structures):
        print(f"Structure {i+1}:")
        print(f"  Space group: {structure['space_group']}")
        print(f"  Energy: {structure['formation_energy']:.3f} eV/atom")
        print(f"  Volume: {structure['unit_cell_volume']:.2f} Å³")

Polymorph Analysis
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.materials.polymorph_inverse_design import PolymorphDesigner

    designer = PolymorphDesigner()
    
    # Design polymorphs with target properties
    target_properties = {
        'density': 2.65,      # g/cm³
        'band_gap': 5.0,      # eV
        'bulk_modulus': 150,  # GPa
        'hardness': 7.0       # Mohs scale
    }
    
    polymorphs = designer.generate_polymorphs(
        base_composition="SiO2",
        target_properties=target_properties,
        max_generations=50
    )
    
    print(f"Generated {len(polymorphs)} polymorph candidates")
    for polymorph in polymorphs[:3]:
        print(f"Space group: {polymorph['space_group']}")
        print(f"Fitness score: {polymorph['fitness']:.3f}")

Intermolecular Interaction Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.materials.intermolecular_interactions import InteractionAnalyzer

    analyzer = InteractionAnalyzer()
    
    # Analyze hydrogen bonding
    h_bonds = analyzer.find_hydrogen_bonds(structure, cutoff=3.5)
    
    print(f"Number of hydrogen bonds: {len(h_bonds)}")
    for bond in h_bonds[:5]:
        print(f"  {bond['donor']} -> {bond['acceptor']}: {bond['distance']:.2f} Å")
    
    # Analyze van der Waals interactions
    vdw_interactions = analyzer.find_vdw_interactions(structure)
    
    print(f"Van der Waals contacts: {len(vdw_interactions)}")

High-Throughput Screening
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.materials import MaterialsCalculator

    calc = MaterialsCalculator(backend="rapids")  # Use GPU DataFrames
    
    # Screen large database
    database_path = "materials_project_database.json"
    screening_criteria = {
        'band_gap': (1.0, 3.0),      # eV
        'formation_energy': (-3.0, 0.0),  # eV/atom
        'stability': 'stable'
    }
    
    candidates = calc.screen_database(
        database_path=database_path,
        criteria=screening_criteria,
        descriptors=['band_gap', 'formation_energy', 'elastic_modulus']
    )
    
    print(f"Found {len(candidates)} candidate materials")
    print(candidates.head())

Experimental Data Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.materials.experimental_validation import ExperimentalValidator

    validator = ExperimentalValidator()
    
    # Compare with experimental data
    experimental_data = {
        'density': 2.648,     # g/cm³
        'band_gap': 8.9,      # eV
        'bulk_modulus': 37.0  # GPa
    }
    
    validation_results = validator.validate_predictions(
        predicted_properties=descriptors,
        experimental_data=experimental_data
    )
    
    print("Validation Results:")
    for prop, result in validation_results.items():
        print(f"  {prop}: MAE = {result['mae']:.3f}, R² = {result['r2']:.3f}")