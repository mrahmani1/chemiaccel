Polymer Science API Reference
==============================

Specialized modules for polymer analysis, copolymer characterization, and polymer property prediction.

Monomer Analysis
----------------

.. automodule:: chemi_accel.polymer.monomer
    :members:
    :undoc-members:
    :show-inheritance:

Copolymer Analysis
------------------

.. automodule:: chemi_accel.polymer.copolymer
    :members:
    :undoc-members:
    :show-inheritance:

Oligomer Characterization
-------------------------

.. automodule:: chemi_accel.polymer.oligomer
    :members:
    :undoc-members:
    :show-inheritance:

Thermoset Analysis
------------------

.. automodule:: chemi_accel.polymer.thermoset
    :members:
    :undoc-members:
    :show-inheritance:

Conformational Analysis
-----------------------

.. automodule:: chemi_accel.polymer.conformers
    :members:
    :undoc-members:
    :show-inheritance:

Repeating Unit Analysis
-----------------------

.. automodule:: chemi_accel.polymer.repeating
    :members:
    :undoc-members:
    :show-inheritance:

Inverse Design
--------------

.. automodule:: chemi_accel.polymer.inverse_design
    :members:
    :undoc-members:
    :show-inheritance:

Predictive Models
-----------------

Property Engine
~~~~~~~~~~~~~~~

.. automodule:: chemi_accel.polymer.predictive.property_engine
    :members:
    :undoc-members:
    :show-inheritance:

Descriptors
~~~~~~~~~~~

.. automodule:: chemi_accel.polymer.predictive.descriptors
    :members:
    :undoc-members:
    :show-inheritance:

Model Training
~~~~~~~~~~~~~~

.. automodule:: chemi_accel.polymer.predictive.model_training
    :members:
    :undoc-members:
    :show-inheritance:

Usage Examples
--------------

Monomer Reactivity Prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.polymer.monomer import MonomerAnalyzer

    analyzer = MonomerAnalyzer()
    
    # Analyze monomer reactivity
    monomer_smiles = "C=CC(=O)OC"  # Methyl acrylate
    reactivity = analyzer.predict_reactivity(monomer_smiles)
    
    print(f"Reactivity ratios: {reactivity['r1']}, {reactivity['r2']}")
    print(f"Polymerization mechanism: {reactivity['mechanism']}")

Copolymer Composition Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.polymer.copolymer import CopolymerAnalyzer

    analyzer = CopolymerAnalyzer()
    
    # Define copolymer system
    monomers = ["C=C", "C=CC#N"]  # Ethylene, Acrylonitrile
    feed_ratios = [0.7, 0.3]
    
    # Analyze composition
    composition = analyzer.analyze_composition(monomers, feed_ratios)
    
    print(f"Instantaneous composition: {composition['instantaneous']}")
    print(f"Cumulative composition: {composition['cumulative']}")
    print(f"Sequence distribution: {composition['sequence_distribution']}")

Polymer Property Prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.polymer.predictive.property_engine import PropertyEngine

    engine = PropertyEngine()
    
    # Predict polymer properties
    polymer_smiles = "[CH2CH(C(=O)OC)]n"  # Poly(methyl acrylate)
    properties = engine.predict_properties(polymer_smiles)
    
    print(f"Glass transition temperature: {properties['tg']:.1f} K")
    print(f"Density: {properties['density']:.3f} g/cm³")
    print(f"Elastic modulus: {properties['elastic_modulus']:.1f} MPa")

Thermoset Network Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.polymer.thermoset import ThermosetAnalyzer

    analyzer = ThermosetAnalyzer()
    
    # Define crosslinking system
    resin = "CC(C)(C1=CC=C(O)C=C1)C2=CC=C(O)C=C2"  # Bisphenol A
    hardener = "C1OC1C2OC2"  # Epichlorohydrin
    
    # Analyze crosslink density
    network = analyzer.analyze_network(resin, hardener, crosslink_ratio=0.8)
    
    print(f"Crosslink density: {network['crosslink_density']:.3f} mol/cm³")
    print(f"Gel fraction: {network['gel_fraction']:.3f}")
    print(f"Network modulus: {network['network_modulus']:.1f} MPa")

Oligomer Chain Analysis
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.polymer.oligomer import OligomerAnalyzer

    analyzer = OligomerAnalyzer()
    
    # Generate oligomer chains
    monomer = "C=CC(=O)OC"  # Methyl acrylate
    oligomers = analyzer.generate_oligomers(monomer, max_length=10)
    
    # Analyze end-to-end distance
    conformational_props = analyzer.analyze_conformations(oligomers)
    
    for length, props in conformational_props.items():
        print(f"Chain length {length}: Rg = {props['radius_of_gyration']:.2f} Å")

Polymer Inverse Design
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.polymer.inverse_design import PolymerInverseDesign

    designer = PolymerInverseDesign()
    
    # Target properties
    target_properties = {
        'tg': 373.15,  # 100°C
        'density': 1.2,  # g/cm³
        'elastic_modulus': 2000,  # MPa
        'solubility_parameter': 19.5  # MPa^0.5
    }
    
    # Generate polymer candidates
    candidates = designer.generate_candidates(
        target_properties=target_properties,
        max_generations=100,
        population_size=50
    )
    
    # Display best candidates
    for i, candidate in enumerate(candidates[:5]):
        print(f"Candidate {i+1}:")
        print(f"  SMILES: {candidate['smiles']}")
        print(f"  Predicted Tg: {candidate['properties']['tg']:.1f} K")
        print(f"  Fitness score: {candidate['fitness']:.3f}")

Conformational Sampling
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.polymer.conformers import ConformerGenerator

    generator = ConformerGenerator()
    
    # Generate conformers for polymer chain
    polymer_smiles = "[CH2CH(C(=O)OC)]10"  # Poly(methyl acrylate) decamer
    conformers = generator.generate_conformers(
        polymer_smiles,
        num_conformers=100,
        method='rdkit'
    )
    
    # Analyze conformational properties
    properties = generator.analyze_conformers(conformers)
    
    print(f"Average radius of gyration: {properties['avg_rg']:.2f} Å")
    print(f"End-to-end distance: {properties['avg_ete']:.2f} Å")
    print(f"Persistence length: {properties['persistence_length']:.2f} Å")

Batch Polymer Processing
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.polymer import PolymerCalculator

    calc = PolymerCalculator(backend="cuda")  # GPU acceleration
    
    # Process multiple polymer systems
    polymer_smiles = [
        "[CH2CH2]n",           # Polyethylene
        "[CH2CH(C)]n",         # Polypropylene  
        "[CH2CH(C6H5)]n",      # Polystyrene
        "[CH2CH(C(=O)OC)]n"    # Poly(methyl acrylate)
    ]
    
    # Calculate polymer descriptors
    results = calc.calculate_batch(polymer_smiles)
    
    print("Polymer Properties:")
    print(results[['tg', 'density', 'elastic_modulus']].head())