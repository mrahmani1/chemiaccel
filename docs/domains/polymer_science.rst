Polymer Science Tutorial
=======================

This comprehensive guide covers ChemiAccel's specialized polymer analysis capabilities, from monomer characterization to complex copolymer design and property prediction.

Overview
--------

ChemiAccel's polymer modules provide:

* **Monomer analysis** - Reactivity ratios and polymerization mechanisms
* **Copolymer characterization** - Composition analysis and sequence distribution
* **Thermoset analysis** - Crosslink density and network properties
* **Conformational sampling** - Chain conformations and dynamics
* **Property prediction** - Glass transition, mechanical properties, solubility
* **Inverse design** - Generate polymers with target properties

Polymer Hierarchy
------------------

ChemiAccel organizes polymer analysis into hierarchical levels:

.. code-block:: text

    Monomer → Oligomer → Polymer Chain → Polymer Network
       ↓         ↓           ↓              ↓
    Reactivity  Sequence   Conformation   Bulk Properties

Basic Polymer Analysis
----------------------

Monomer Characterization
~~~~~~~~~~~~~~~~~~~~~~~~

Analyze individual monomers for polymerization potential:

.. code-block:: python

    from chemi_accel.polymer.monomer import MonomerAnalyzer
    from chemi_accel import Calculator

    # Initialize monomer analyzer
    analyzer = MonomerAnalyzer()

    # Analyze common monomers
    monomers = {
        'styrene': 'C=Cc1ccccc1',
        'methyl_methacrylate': 'C=C(C)C(=O)OC',
        'acrylonitrile': 'C=CC#N',
        'vinyl_acetate': 'C=COC(=O)C',
        'ethylene': 'C=C'
    }

    print("=== MONOMER ANALYSIS ===")
    for name, smiles in monomers.items():
        analysis = analyzer.analyze_monomer(smiles)
        
        print(f"\n{name.upper()}:")
        print(f"  SMILES: {smiles}")
        print(f"  Polymerization mechanism: {analysis['mechanism']}")
        print(f"  Reactivity ratio (r1): {analysis['reactivity_ratio']:.3f}")
        print(f"  Ceiling temperature: {analysis['ceiling_temp']:.1f} K")
        print(f"  Thermodynamic favorability: {analysis['delta_h_poly']:.1f} kJ/mol")

Copolymer Composition Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Analyze copolymer composition and sequence distribution:

.. code-block:: python

    from chemi_accel.polymer.copolymer import CopolymerAnalyzer

    # Initialize copolymer analyzer
    copolymer_analyzer = CopolymerAnalyzer()

    # Define copolymer system
    comonomer_1 = 'C=Cc1ccccc1'  # Styrene
    comonomer_2 = 'C=CC#N'       # Acrylonitrile

    # Feed composition
    feed_ratio_1 = 0.7  # 70% styrene
    feed_ratio_2 = 0.3  # 30% acrylonitrile

    print("=== COPOLYMER ANALYSIS ===")
    print(f"Feed composition: {feed_ratio_1:.1%} styrene, {feed_ratio_2:.1%} acrylonitrile")

    # Analyze composition drift
    composition_data = copolymer_analyzer.analyze_composition_drift(
        monomer_1=comonomer_1,
        monomer_2=comonomer_2,
        feed_ratio_1=feed_ratio_1,
        feed_ratio_2=feed_ratio_2,
        conversion_range=(0, 0.95),
        points=20
    )

    print("\nComposition vs Conversion:")
    for point in composition_data['composition_curve'][:5]:  # Show first 5 points
        conv = point['conversion']
        f1 = point['instantaneous_f1']
        F1 = point['cumulative_F1']
        print(f"  Conversion: {conv:.1%} | Inst. f1: {f1:.3f} | Cum. F1: {F1:.3f}")

    # Sequence distribution analysis
    sequence_analysis = copolymer_analyzer.analyze_sequence_distribution(
        monomer_1=comonomer_1,
        monomer_2=comonomer_2,
        feed_ratio_1=feed_ratio_1,
        feed_ratio_2=feed_ratio_2
    )

    print(f"\nSequence Distribution:")
    print(f"  Number average sequence length (styrene): {sequence_analysis['ln1']:.2f}")
    print(f"  Number average sequence length (acrylonitrile): {sequence_analysis['ln2']:.2f}")
    print(f"  Probability of alternation: {sequence_analysis['prob_alternation']:.3f}")
    print(f"  Randomness parameter: {sequence_analysis['randomness']:.3f}")

Polymer Property Prediction
----------------------------

Glass Transition Temperature Prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Predict thermal properties of polymers:

.. code-block:: python

    from chemi_accel.polymer.predictive.property_engine import PropertyEngine

    # Initialize property prediction engine
    property_engine = PropertyEngine()

    # Define polymer structures
    polymers = {
        'polystyrene': '[CH2CH(c1ccccc1)]n',
        'poly_methyl_methacrylate': '[CH2C(C)(C(=O)OC)]n',
        'poly_ethylene': '[CH2CH2]n',
        'poly_propylene': '[CH2CH(C)]n',
        'poly_vinyl_chloride': '[CH2CHCl]n'
    }

    print("=== POLYMER PROPERTY PREDICTION ===")
    for name, structure in polymers.items():
        properties = property_engine.predict_properties(structure)
        
        print(f"\n{name.upper()}:")
        print(f"  Structure: {structure}")
        print(f"  Glass transition temp: {properties['tg']:.1f} K ({properties['tg']-273.15:.1f} °C)")
        print(f"  Density: {properties['density']:.3f} g/cm³")
        print(f"  Elastic modulus: {properties['elastic_modulus']:.0f} MPa")
        print(f"  CTE: {properties['thermal_expansion']:.1e} K⁻¹")

Mechanical Property Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Predict mechanical behavior of polymers:

.. code-block:: python

    from chemi_accel.polymer.predictive.descriptors import PolymerDescriptors

    # Calculate polymer-specific descriptors
    descriptor_calc = PolymerDescriptors()

    polymer_smiles = '[CH2CH(c1ccccc1)]n'  # Polystyrene
    
    # Calculate comprehensive descriptors
    descriptors = descriptor_calc.calculate_polymer_descriptors(polymer_smiles)

    print("=== POLYMER DESCRIPTORS ===")
    print(f"Polymer: {polymer_smiles}")
    print(f"\nStructural descriptors:")
    print(f"  Repeating unit MW: {descriptors['repeat_unit_mw']:.2f} g/mol")
    print(f"  Backbone flexibility: {descriptors['backbone_flexibility']:.3f}")
    print(f"  Side chain branching: {descriptors['side_chain_branching']:.3f}")
    print(f"  Polar surface area per unit: {descriptors['psa_per_unit']:.2f} Ų")

    print(f"\nThermal descriptors:")
    print(f"  van der Waals volume: {descriptors['vdw_volume']:.2f} ų")
    print(f"  Cohesive energy density: {descriptors['cohesive_energy_density']:.1f} J/cm³")
    print(f"  Solubility parameter: {descriptors['solubility_parameter']:.1f} MPa^0.5")

    print(f"\nMechanical descriptors:")
    print(f"  Chain stiffness: {descriptors['chain_stiffness']:.3f}")
    print(f"  Packing efficiency: {descriptors['packing_efficiency']:.3f}")
    print(f"  Free volume fraction: {descriptors['free_volume_fraction']:.3f}")

Advanced Polymer Analysis
--------------------------

Conformational Analysis
~~~~~~~~~~~~~~~~~~~~~~

Analyze polymer chain conformations and dynamics:

.. code-block:: python

    from chemi_accel.polymer.conformers import ConformerGenerator
    import numpy as np

    # Initialize conformer generator
    conformer_gen = ConformerGenerator()

    # Generate conformers for oligomer
    oligomer_smiles = '[CH2CH(c1ccccc1)]10'  # Polystyrene decamer
    
    print("=== CONFORMATIONAL ANALYSIS ===")
    print(f"Analyzing oligomer: {oligomer_smiles}")

    # Generate multiple conformers
    conformers = conformer_gen.generate_conformers(
        polymer_smiles=oligomer_smiles,
        num_conformers=100,
        method='rdkit',
        optimize_conformers=True
    )

    print(f"Generated {len(conformers)} conformers")

    # Analyze conformational properties
    conformational_props = conformer_gen.analyze_conformers(conformers)

    print(f"\nConformational Properties:")
    print(f"  Average radius of gyration: {conformational_props['avg_rg']:.2f} ± {conformational_props['std_rg']:.2f} Å")
    print(f"  Average end-to-end distance: {conformational_props['avg_ete']:.2f} ± {conformational_props['std_ete']:.2f} Å")
    print(f"  Characteristic ratio: {conformational_props['characteristic_ratio']:.2f}")
    print(f"  Persistence length: {conformational_props['persistence_length']:.2f} Å")

    # Analyze chain flexibility
    flexibility_analysis = conformer_gen.analyze_chain_flexibility(conformers)
    
    print(f"\nChain Flexibility:")
    print(f"  Kuhn length: {flexibility_analysis['kuhn_length']:.2f} Å")
    print(f"  Flexibility parameter: {flexibility_analysis['flexibility_parameter']:.3f}")
    print(f"  Local backbone flexibility: {flexibility_analysis['local_flexibility']:.3f}")

Thermoset Network Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~

Analyze crosslinked polymer networks:

.. code-block:: python

    from chemi_accel.polymer.thermoset import ThermosetAnalyzer

    # Initialize thermoset analyzer
    thermoset_analyzer = ThermosetAnalyzer()

    # Define thermoset system (epoxy resin)
    resin = 'CC(C)(C1=CC=C(O)C=C1)C2=CC=C(O)C=C2'  # Bisphenol A
    hardener = 'C1OC1C2OC2'  # Diglycidyl ether
    
    print("=== THERMOSET ANALYSIS ===")
    print(f"Resin: {resin}")
    print(f"Hardener: {hardener}")

    # Analyze crosslinking reaction
    crosslink_analysis = thermoset_analyzer.analyze_crosslinking_reaction(
        resin=resin,
        hardener=hardener,
        stoichiometric_ratio=1.0,
        temperature=180,  # °C
        time=120  # minutes
    )

    print(f"\nCrosslinking Reaction:")
    print(f"  Gel point: {crosslink_analysis['gel_point']:.3f} conversion")
    print(f"  Final conversion: {crosslink_analysis['final_conversion']:.3f}")
    print(f"  Reaction rate constant: {crosslink_analysis['rate_constant']:.2e} s⁻¹")

    # Network property prediction
    network_props = thermoset_analyzer.predict_network_properties(
        resin=resin,
        hardener=hardener,
        conversion=0.95,
        temperature=25  # °C for property evaluation
    )

    print(f"\nNetwork Properties:")
    print(f"  Crosslink density: {network_props['crosslink_density']:.3f} mol/cm³")
    print(f"  Glass transition temp: {network_props['tg']:.1f} K ({network_props['tg']-273.15:.1f} °C)")
    print(f"  Elastic modulus: {network_props['elastic_modulus']:.0f} MPa")
    print(f"  Coefficient of thermal expansion: {network_props['cte']:.1e} K⁻¹")
    print(f"  Water uptake: {network_props['water_uptake']:.2f} wt%")

Polymer Inverse Design
----------------------

Target Property Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Design polymers with specific target properties:

.. code-block:: python

    from chemi_accel.polymer.inverse_design import PolymerInverseDesign

    # Initialize inverse design engine
    polymer_designer = PolymerInverseDesign(
        algorithm='genetic_algorithm',
        population_size=100,
        generations=50
    )

    # Define target properties
    target_properties = {
        'glass_transition_temp': 350.0,  # K (77°C)
        'density': 1.2,                  # g/cm³
        'elastic_modulus': 2500,         # MPa
        'thermal_conductivity': 0.2,     # W/m·K
        'dielectric_constant': 3.5,      # -
        'water_uptake': 0.5             # wt%
    }

    # Property weights (importance)
    property_weights = {
        'glass_transition_temp': 3.0,   # Very important
        'density': 1.0,
        'elastic_modulus': 2.0,         # Important
        'thermal_conductivity': 2.0,    # Important
        'dielectric_constant': 1.5,
        'water_uptake': 1.0
    }

    print("=== POLYMER INVERSE DESIGN ===")
    print("Target Properties:")
    for prop, value in target_properties.items():
        print(f"  {prop}: {value}")

    # Generate polymer candidates
    candidates = polymer_designer.generate_polymers(
        target_properties=target_properties,
        property_weights=property_weights,
        constraint_functions={
            'synthesizability': lambda x: x < 5.0,  # Synthetic accessibility
            'stability': lambda x: x > 0.8         # Thermal stability
        }
    )

    print(f"\nGenerated {len(candidates)} polymer candidates")
    print("\nTop 5 Candidates:")
    
    for i, candidate in enumerate(candidates[:5]):
        print(f"\n{i+1}. {candidate['structure']}")
        print(f"   Fitness score: {candidate['fitness']:.3f}")
        print(f"   Predicted properties:")
        for prop, value in candidate['predicted_properties'].items():
            target = target_properties[prop]
            error = abs(value - target) / target * 100
            print(f"     {prop}: {value:.2f} (target: {target:.2f}, error: {error:.1f}%)")

Multi-Objective Polymer Design
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Balance competing properties in polymer design:

.. code-block:: python

    from chemi_accel.ai.multi_objective_optimization import MultiObjectiveOptimizer

    # Multi-objective optimization for polymers
    polymer_moo = MultiObjectiveOptimizer(algorithm='nsga3')

    # Define competing objectives
    objectives = [
        {'name': 'toughness', 'direction': 'maximize', 'weight': 2.0},
        {'name': 'stiffness', 'direction': 'maximize', 'weight': 2.0},
        {'name': 'processing_temp', 'direction': 'minimize', 'weight': 1.5},
        {'name': 'cost', 'direction': 'minimize', 'weight': 1.0}
    ]

    # Run multi-objective optimization
    pareto_polymers = polymer_moo.optimize_polymers(
        objectives=objectives,
        population_size=200,
        generations=100,
        mutation_rate=0.15
    )

    print("=== MULTI-OBJECTIVE POLYMER OPTIMIZATION ===")
    print(f"Pareto front contains {len(pareto_polymers)} solutions")

    # Analyze trade-offs
    print("\nTrade-off Analysis:")
    for i, polymer in enumerate(pareto_polymers[:5]):
        print(f"\nSolution {i+1}: {polymer['structure']}")
        print(f"  Toughness: {polymer['properties']['toughness']:.1f} kJ/m²")
        print(f"  Stiffness: {polymer['properties']['stiffness']:.0f} MPa")
        print(f"  Processing temp: {polymer['properties']['processing_temp']:.0f} °C")
        print(f"  Relative cost: {polymer['properties']['cost']:.2f}")

    # Select balanced solution
    balanced_polymer = polymer_moo.select_knee_solution(pareto_polymers)
    print(f"\nBalanced Solution: {balanced_polymer['structure']}")
    print("This polymer offers the best trade-off between all objectives.")

High-Throughput Polymer Screening
----------------------------------

Large-Scale Property Screening
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Screen large libraries of polymers for specific applications:

.. code-block:: python

    from chemi_accel.polymer import PolymerCalculator
    import pandas as pd

    # Initialize GPU-accelerated polymer calculator
    polymer_calc = PolymerCalculator(backend='cuda')

    # Generate polymer library
    def generate_polymer_library():
        """Generate diverse polymer library"""
        
        # Common backbone structures
        backbones = [
            '[CH2CH2]',           # Polyethylene
            '[CH2CH(C)]',         # Polypropylene
            '[CH2CH(c1ccccc1)]',  # Polystyrene
            '[CH2CH(C#N)]',       # Polyacrylonitrile
            '[CH2CH(Cl)]',        # PVC
            '[CH2C(C)(C(=O)OC)]', # PMMA
        ]
        
        # Side chain modifications
        side_chains = ['', 'C', 'CC', 'CCC', 'c1ccccc1', 'C(=O)OC', 'C#N', 'F', 'Cl']
        
        polymer_library = []
        for backbone in backbones:
            for side_chain in side_chains:
                if side_chain:
                    # Create modified backbone
                    modified = backbone.replace(')', f'{side_chain})')
                    polymer_library.append(f'{modified}n')
                else:
                    polymer_library.append(f'{backbone}n')
        
        return polymer_library

    # Generate library
    polymer_library = generate_polymer_library()
    print(f"Generated polymer library with {len(polymer_library)} structures")

    # High-throughput screening
    print("\n=== HIGH-THROUGHPUT SCREENING ===")
    screening_results = polymer_calc.screen_polymers(
        polymer_library=polymer_library,
        properties=['tg', 'density', 'elastic_modulus', 'solubility_parameter'],
        screening_criteria={
            'tg': (300, 400),           # K
            'density': (0.9, 1.5),      # g/cm³
            'elastic_modulus': (1000, 5000),  # MPa
            'solubility_parameter': (15, 25)   # MPa^0.5
        }
    )

    # Filter results
    filtered_polymers = screening_results[
        (screening_results['tg'] >= 300) &
        (screening_results['tg'] <= 400) &
        (screening_results['density'] >= 0.9) &
        (screening_results['density'] <= 1.5) &
        (screening_results['elastic_modulus'] >= 1000) &
        (screening_results['elastic_modulus'] <= 5000)
    ]

    print(f"Found {len(filtered_polymers)} polymers meeting criteria")
    print("\nTop 10 Candidates:")
    top_candidates = filtered_polymers.nlargest(10, 'elastic_modulus')
    
    for i, (idx, polymer) in enumerate(top_candidates.iterrows()):
        print(f"{i+1}. {polymer['structure']}")
        print(f"   Tg: {polymer['tg']:.1f} K")
        print(f"   Density: {polymer['density']:.3f} g/cm³")
        print(f"   E-modulus: {polymer['elastic_modulus']:.0f} MPa")

Structure-Property Relationships
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Analyze relationships between polymer structure and properties:

.. code-block:: python

    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor

    # Analyze structure-property relationships
    print("=== STRUCTURE-PROPERTY ANALYSIS ===")

    # Calculate additional structural descriptors
    from chemi_accel.polymer.predictive.descriptors import PolymerDescriptors
    
    descriptor_calc = PolymerDescriptors()
    
    # Calculate descriptors for all polymers
    structural_descriptors = []
    for polymer in polymer_library[:50]:  # Analyze subset for speed
        try:
            descriptors = descriptor_calc.calculate_polymer_descriptors(polymer)
            descriptors['structure'] = polymer
            structural_descriptors.append(descriptors)
        except:
            continue
    
    descriptor_df = pd.DataFrame(structural_descriptors)
    
    # Merge with property data
    property_df = polymer_calc.calculate_properties_batch(
        descriptor_df['structure'].tolist()
    )
    
    combined_df = pd.merge(descriptor_df, property_df, on='structure')
    
    # Correlation analysis
    numeric_columns = combined_df.select_dtypes(include=[np.number]).columns
    correlation_matrix = combined_df[numeric_columns].corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Polymer Structure-Property Correlations')
    plt.tight_layout()
    plt.savefig('polymer_correlations.png', dpi=300)
    
    # Feature importance analysis
    X = combined_df[['repeat_unit_mw', 'backbone_flexibility', 'side_chain_branching', 
                     'cohesive_energy_density', 'free_volume_fraction']]
    y = combined_df['tg']
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance for Glass Transition Temperature:")
    for _, row in feature_importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")

Specialized Applications
------------------------

Biomedical Polymers
~~~~~~~~~~~~~~~~~~~

Design polymers for biomedical applications:

.. code-block:: python

    # Biocompatible polymer design
    biomedical_targets = {
        'biocompatibility_score': 0.9,     # High biocompatibility
        'degradation_rate': 0.1,           # Slow degradation (1/day)
        'elastic_modulus': 10,             # Soft (MPa) - tissue-like
        'water_uptake': 50,                # High water content (%)
        'protein_adsorption': 0.2          # Low protein adsorption
    }

    biomedical_polymers = polymer_designer.generate_polymers(
        target_properties=biomedical_targets,
        constraint_functions={
            'toxicity': lambda x: x < 0.1,        # Very low toxicity
            'immunogenicity': lambda x: x < 0.15,  # Low immune response
            'sterilizability': lambda x: x > 0.8   # Can be sterilized
        }
    )

    print("=== BIOMEDICAL POLYMER DESIGN ===")
    print(f"Generated {len(biomedical_polymers)} biomedical polymer candidates")

Electronic Materials
~~~~~~~~~~~~~~~~~~~~

Design polymers for electronic applications:

.. code-block:: python

    # Conductive polymer design
    electronic_targets = {
        'electrical_conductivity': 1e-3,   # S/cm
        'dielectric_constant': 5.0,        # High dielectric
        'breakdown_voltage': 300,          # V/μm
        'thermal_stability': 0.95,         # High thermal stability
        'processability': 0.8             # Good processability
    }

    electronic_polymers = polymer_designer.generate_polymers(
        target_properties=electronic_targets,
        constraint_functions={
            'environmental_stability': lambda x: x > 0.85,
            'cost_effectiveness': lambda x: x > 0.7
        }
    )

    print("=== ELECTRONIC POLYMER DESIGN ===")
    print(f"Generated {len(electronic_polymers)} electronic polymer candidates")

Best Practices
--------------

Polymer Structure Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Use proper SMILES notation** for repeat units
2. **Consider stereochemistry** when relevant
3. **Account for chain ends** in oligomer analysis
4. **Specify molecular weight** for property predictions

Experimental Validation
~~~~~~~~~~~~~~~~~~~~~~

1. **Synthesizability assessment** - Check if polymers can be made
2. **Processing conditions** - Consider real-world processing
3. **Property measurement** - Validate predictions with experiments
4. **Failure mode analysis** - Understand property limits

Model Limitations
~~~~~~~~~~~~~~~~~

1. **Molecular weight effects** - Properties depend on MW
2. **Processing history** - Thermal and mechanical history matters
3. **Additive effects** - Consider fillers, plasticizers, etc.
4. **Environmental conditions** - Temperature, humidity, stress

Data Quality
~~~~~~~~~~~~

1. **Curated databases** - Use high-quality polymer property data
2. **Consistent conditions** - Standardize measurement conditions
3. **Uncertainty quantification** - Report prediction uncertainties
4. **Domain applicability** - Stay within model training domain

For more advanced polymer analysis examples, see the :doc:`../examples/polymer_property_prediction` and :doc:`../notebooks/polymer_analysis` sections.