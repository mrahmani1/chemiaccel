Materials Science Tutorial
=========================

This comprehensive guide covers ChemiAccel's materials science capabilities, from crystal structure analysis to solid-state property prediction and materials discovery workflows.

Overview
--------

ChemiAccel's materials modules provide:

* **Crystal structure analysis** - CIF/POSCAR parsing and structural descriptors
* **Periodic descriptors** - Symmetry analysis and distribution functions
* **Property prediction** - Electronic, mechanical, and thermal properties
* **Polymorph analysis** - Crystal form prediction and stability assessment
* **High-throughput screening** - Large-scale materials discovery
* **Experimental integration** - Validation with experimental data

Materials Hierarchy
--------------------

ChemiAccel organizes materials analysis across multiple scales:

.. code-block:: text

    Atoms → Unit Cell → Crystal → Bulk Material
      ↓        ↓         ↓           ↓
   Bonding  Symmetry  Structure  Properties

Basic Crystal Analysis
----------------------

Loading Crystal Structures
~~~~~~~~~~~~~~~~~~~~~~~~~~

Import and analyze crystal structures from various formats:

.. code-block:: python

    from chemi_accel.materials import MaterialsCalculator
    from chemi_accel.materials.crystal_parser import CIFParser, POSCARParser

    # Initialize materials calculator
    calc = MaterialsCalculator(use_gpu=True)

    print("=== CRYSTAL STRUCTURE LOADING ===")

    # Load from CIF file
    cif_parser = CIFParser()
    quartz_structure = cif_parser.parse_cif('quartz.cif')

    print("Quartz Structure (from CIF):")
    print(f"  Chemical formula: {quartz_structure.composition}")
    print(f"  Space group: {quartz_structure.space_group}")
    print(f"  Lattice parameters: a={quartz_structure.lattice.a:.3f} Å")
    print(f"                      b={quartz_structure.lattice.b:.3f} Å")
    print(f"                      c={quartz_structure.lattice.c:.3f} Å")
    print(f"  Unit cell volume: {quartz_structure.lattice.volume:.2f} Ų")
    print(f"  Density: {quartz_structure.density:.3f} g/cm³")

    # Load from POSCAR file
    poscar_parser = POSCARParser()
    vasp_structure = poscar_parser.parse_poscar('POSCAR')

    print(f"\nVASP Structure (from POSCAR):")
    print(f"  Chemical formula: {vasp_structure.composition}")
    print(f"  Number of atoms: {len(vasp_structure.sites)}")
    print(f"  Lattice vectors:")
    for i, vec in enumerate(vasp_structure.lattice.matrix):
        print(f"    a{i+1}: [{vec[0]:8.4f}, {vec[1]:8.4f}, {vec[2]:8.4f}]")

Crystal Descriptor Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calculate comprehensive structural descriptors:

.. code-block:: python

    from chemi_accel.materials.crystal_descriptors import CrystalDescriptors

    # Initialize descriptor calculator
    descriptor_calc = CrystalDescriptors()

    print("=== CRYSTAL DESCRIPTORS ===")

    # Calculate descriptors for quartz
    descriptors = descriptor_calc.calculate_descriptors(quartz_structure)

    print("Structural Descriptors:")
    print(f"  Packing efficiency: {descriptors['packing_efficiency']:.3f}")
    print(f"  Coordination number (avg): {descriptors['avg_coordination_number']:.2f}")
    print(f"  Bond length variance: {descriptors['bond_length_variance']:.4f} Ų")
    print(f"  Bond angle variance: {descriptors['bond_angle_variance']:.2f}°²")
    print(f"  Density: {descriptors['density']:.3f} g/cm³")

    print(f"\nGeometric Descriptors:")
    print(f"  Sphericity: {descriptors['sphericity']:.3f}")
    print(f"  Asphericity: {descriptors['asphericity']:.3f}")
    print(f"  Volume per atom: {descriptors['volume_per_atom']:.2f} ų/atom")
    print(f"  Surface area per volume: {descriptors['surface_area_per_volume']:.3f} Å⁻¹")

    print(f"\nSymmetry Descriptors:")
    print(f"  Space group number: {descriptors['space_group_number']}")
    print(f"  Crystal system: {descriptors['crystal_system']}")
    print(f"  Point group: {descriptors['point_group']}")
    print(f"  Wyckoff positions: {len(descriptors['wyckoff_positions'])}")

Periodic Boundary Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~

Analyze periodic properties and distribution functions:

.. code-block:: python

    from chemi_accel.materials.periodic_descriptors import PeriodicDescriptors

    # Initialize periodic descriptor calculator
    periodic_calc = PeriodicDescriptors()

    print("=== PERIODIC DESCRIPTORS ===")

    # Calculate radial distribution function
    rdf_data = periodic_calc.calculate_rdf(
        structure=quartz_structure,
        r_max=10.0,  # Å
        dr=0.1,      # Å
        elements=['Si', 'O']
    )

    print("Radial Distribution Function:")
    print(f"  First peak position: {rdf_data['first_peak_position']:.2f} Å")
    print(f"  First peak height: {rdf_data['first_peak_height']:.2f}")
    print(f"  Coordination number (first shell): {rdf_data['first_shell_coordination']:.1f}")

    # Structure factor analysis
    structure_factor = periodic_calc.calculate_structure_factor(
        structure=quartz_structure,
        q_max=20.0,  # Å⁻¹
        num_points=1000
    )

    print(f"\nStructure Factor:")
    print(f"  Maximum intensity: {structure_factor['max_intensity']:.2f}")
    print(f"  First diffraction peak: {structure_factor['first_peak_q']:.2f} Å⁻¹")
    print(f"  Peak width (FWHM): {structure_factor['peak_width']:.3f} Å⁻¹")

    # Voronoi analysis
    voronoi_analysis = periodic_calc.calculate_voronoi_descriptors(quartz_structure)

    print(f"\nVoronoi Analysis:")
    print(f"  Average Voronoi volume: {voronoi_analysis['avg_voronoi_volume']:.2f} ų")
    print(f"  Voronoi volume variance: {voronoi_analysis['volume_variance']:.3f}")
    print(f"  Average number of neighbors: {voronoi_analysis['avg_neighbors']:.1f}")
    print(f"  Coordination heterogeneity: {voronoi_analysis['coordination_heterogeneity']:.3f}")

Property Prediction
-------------------

Electronic Properties
~~~~~~~~~~~~~~~~~~~~~

Predict electronic band structure and properties:

.. code-block:: python

    from chemi_accel.materials.solid_state_properties import SolidStateProperties

    # Initialize property predictor
    property_calc = SolidStateProperties()

    print("=== ELECTRONIC PROPERTIES ===")

    # Predict electronic properties
    electronic_props = property_calc.predict_electronic_properties(quartz_structure)

    print("Electronic Structure:")
    print(f"  Band gap: {electronic_props['band_gap']:.2f} eV")
    print(f"  Band gap type: {electronic_props['band_gap_type']}")
    print(f"  Valence band maximum: {electronic_props['vbm']:.2f} eV")
    print(f"  Conduction band minimum: {electronic_props['cbm']:.2f} eV")

    print(f"\nCharge Transport:")
    print(f"  Electron effective mass: {electronic_props['electron_mass']:.3f} m₀")
    print(f"  Hole effective mass: {electronic_props['hole_mass']:.3f} m₀")
    print(f"  Carrier mobility (electron): {electronic_props['electron_mobility']:.1f} cm²/V·s")
    print(f"  Carrier mobility (hole): {electronic_props['hole_mobility']:.1f} cm²/V·s")

    print(f"\nDielectric Properties:")
    print(f"  Static dielectric constant: {electronic_props['dielectric_static']:.2f}")
    print(f"  Optical dielectric constant: {electronic_props['dielectric_optical']:.2f}")
    print(f"  Refractive index: {electronic_props['refractive_index']:.3f}")

Mechanical Properties
~~~~~~~~~~~~~~~~~~~~

Predict mechanical behavior and elastic constants:

.. code-block:: python

    # Predict mechanical properties
    mechanical_props = property_calc.predict_mechanical_properties(quartz_structure)

    print("=== MECHANICAL PROPERTIES ===")

    print("Elastic Constants:")
    print(f"  Bulk modulus: {mechanical_props['bulk_modulus']:.1f} GPa")
    print(f"  Shear modulus: {mechanical_props['shear_modulus']:.1f} GPa")
    print(f"  Young's modulus: {mechanical_props['youngs_modulus']:.1f} GPa")
    print(f"  Poisson's ratio: {mechanical_props['poissons_ratio']:.3f}")

    print(f"\nHardness and Strength:")
    print(f"  Vickers hardness: {mechanical_props['vickers_hardness']:.1f} GPa")
    print(f"  Fracture toughness: {mechanical_props['fracture_toughness']:.2f} MPa·m^0.5")
    print(f"  Tensile strength: {mechanical_props['tensile_strength']:.1f} MPa")

    print(f"\nAnisotropy:")
    print(f"  Universal elastic anisotropy: {mechanical_props['universal_anisotropy']:.3f}")
    print(f"  Elastic anisotropy (Zener): {mechanical_props['zener_anisotropy']:.3f}")

Thermal Properties
~~~~~~~~~~~~~~~~~

Predict thermal behavior and transport:

.. code-block:: python

    # Predict thermal properties
    thermal_props = property_calc.predict_thermal_properties(quartz_structure)

    print("=== THERMAL PROPERTIES ===")

    print("Heat Capacity and Expansion:")
    print(f"  Heat capacity (300 K): {thermal_props['heat_capacity_300k']:.2f} J/mol·K")
    print(f"  Debye temperature: {thermal_props['debye_temperature']:.1f} K")
    print(f"  Linear thermal expansion: {thermal_props['thermal_expansion']:.2e} K⁻¹")

    print(f"\nThermal Transport:")
    print(f"  Thermal conductivity: {thermal_props['thermal_conductivity']:.2f} W/m·K")
    print(f"  Thermal diffusivity: {thermal_props['thermal_diffusivity']:.2e} m²/s")

    print(f"\nPhase Stability:")
    print(f"  Melting point: {thermal_props['melting_point']:.0f} K")
    print(f"  Formation energy: {thermal_props['formation_energy']:.3f} eV/atom")
    print(f"  Decomposition temperature: {thermal_props['decomposition_temp']:.0f} K")

High-Throughput Materials Discovery
-----------------------------------

Database Screening
~~~~~~~~~~~~~~~~~

Screen large materials databases for target properties:

.. code-block:: python

    import pandas as pd

    print("=== HIGH-THROUGHPUT SCREENING ===")

    # Define screening criteria
    screening_criteria = {
        'band_gap': (1.0, 3.0),          # eV - semiconductor range
        'formation_energy': (-3.0, 0.0), # eV/atom - stable compounds
        'bulk_modulus': (50, 300),        # GPa - mechanical stability
        'thermal_conductivity': (1.0, 50) # W/m·K - thermal management
    }

    # Screen materials database
    screening_results = calc.screen_materials_database(
        database_path='materials_project_database.json',
        criteria=screening_criteria,
        max_results=1000
    )

    print(f"Screening Results: {len(screening_results)} materials found")

    # Analyze top candidates
    top_candidates = screening_results.nlargest(10, 'band_gap')

    print("\nTop 10 Semiconductor Candidates:")
    for i, (idx, material) in enumerate(top_candidates.iterrows()):
        print(f"{i+1}. {material['formula']}")
        print(f"   Space group: {material['space_group']}")
        print(f"   Band gap: {material['band_gap']:.2f} eV")
        print(f"   Formation energy: {material['formation_energy']:.3f} eV/atom")
        print(f"   Bulk modulus: {material['bulk_modulus']:.1f} GPa")

    # Export results
    screening_results.to_csv('semiconductor_candidates.csv', index=False)
    print(f"\nResults exported to semiconductor_candidates.csv")

Batch Processing of Crystal Structures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Process multiple crystal structures efficiently:

.. code-block:: python

    from pathlib import Path

    print("=== BATCH CRYSTAL PROCESSING ===")

    # Process directory of CIF files
    cif_directory = Path('crystal_structures/')
    cif_files = list(cif_directory.glob('*.cif'))

    print(f"Found {len(cif_files)} CIF files")

    # Batch process with GPU acceleration
    batch_results = calc.batch_process_crystals(
        structure_files=cif_files,
        descriptors=[
            'packing_efficiency',
            'band_gap',
            'bulk_modulus',
            'formation_energy',
            'thermal_conductivity'
        ],
        chunk_size=100  # Process in batches of 100
    )

    print(f"Processed {len(batch_results)} crystal structures")

    # Statistical analysis
    print("\nDescriptor Statistics:")
    for descriptor in batch_results.columns:
        if descriptor != 'structure_file':
            mean_val = batch_results[descriptor].mean()
            std_val = batch_results[descriptor].std()
            print(f"  {descriptor}: {mean_val:.3f} ± {std_val:.3f}")

    # Identify outliers
    from scipy import stats
    
    outliers = []
    for descriptor in ['band_gap', 'bulk_modulus']:
        z_scores = np.abs(stats.zscore(batch_results[descriptor].dropna()))
        outlier_mask = z_scores > 3
        outlier_structures = batch_results[outlier_mask]['structure_file'].tolist()
        outliers.extend(outlier_structures)

    print(f"\nIdentified {len(set(outliers))} outlier structures")

Crystal Structure Prediction
----------------------------

Polymorph Prediction
~~~~~~~~~~~~~~~~~~~~

Predict alternative crystal forms of known compounds:

.. code-block:: python

    from chemi_accel.materials.crystal_structure_prediction import CrystalPredictor

    # Initialize crystal structure predictor
    predictor = CrystalPredictor()

    print("=== CRYSTAL STRUCTURE PREDICTION ===")

    # Predict polymorphs for SiO2
    composition = "SiO2"
    predicted_structures = predictor.predict_polymorphs(
        composition=composition,
        max_structures=10,
        space_groups=['P1', 'P-1', 'P21/c', 'Pnma', 'I41/amd', 'P3121'],
        energy_cutoff=0.1  # eV/atom above ground state
    )

    print(f"Predicted {len(predicted_structures)} SiO2 polymorphs")

    print("\nPolymorph Analysis:")
    for i, structure in enumerate(predicted_structures):
        print(f"\nPolymorph {i+1}:")
        print(f"  Space group: {structure['space_group']}")
        print(f"  Formation energy: {structure['formation_energy']:.3f} eV/atom")
        print(f"  Relative stability: {structure['relative_energy']:.3f} eV/atom")
        print(f"  Density: {structure['density']:.3f} g/cm³")
        print(f"  Volume per formula unit: {structure['volume_per_fu']:.2f} ų")
        print(f"  Coordination number: {structure['avg_coordination']:.1f}")

    # Stability analysis
    ground_state = min(predicted_structures, key=lambda x: x['formation_energy'])
    metastable = [s for s in predicted_structures if s['relative_energy'] < 0.05]

    print(f"\nStability Analysis:")
    print(f"  Ground state: {ground_state['space_group']}")
    print(f"  Metastable forms: {len(metastable)} (within 0.05 eV/atom)")

Inverse Design
~~~~~~~~~~~~~

Design materials with target properties:

.. code-block:: python

    from chemi_accel.materials.polymorph_inverse_design import MaterialsInverseDesign

    # Initialize inverse design
    materials_designer = MaterialsInverseDesign()

    print("=== MATERIALS INVERSE DESIGN ===")

    # Define target properties
    target_properties = {
        'band_gap': 2.0,              # eV - ideal for solar cells
        'absorption_coefficient': 1e5, # cm⁻¹ - high absorption
        'carrier_mobility': 100,       # cm²/V·s - good transport
        'stability': 0.9,             # thermodynamic stability
        'earth_abundance': 0.8        # abundant elements
    }

    # Generate material candidates
    candidates = materials_designer.generate_materials(
        target_properties=target_properties,
        element_constraints=['light_elements_only'],  # H, C, N, O, F, etc.
        structure_constraints=['cubic', 'tetragonal', 'orthorhombic'],
        num_candidates=50
    )

    print(f"Generated {len(candidates)} material candidates")

    print("\nTop 5 Candidates:")
    for i, candidate in enumerate(candidates[:5]):
        print(f"\n{i+1}. {candidate['composition']}")
        print(f"   Space group: {candidate['space_group']}")
        print(f"   Predicted band gap: {candidate['band_gap']:.2f} eV")
        print(f"   Absorption coefficient: {candidate['absorption_coefficient']:.1e} cm⁻¹")
        print(f"   Carrier mobility: {candidate['carrier_mobility']:.1f} cm²/V·s")
        print(f"   Fitness score: {candidate['fitness']:.3f}")
        print(f"   Synthesizability: {candidate['synthesizability']:.3f}")

Experimental Integration
------------------------

Validation with Experimental Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare predictions with experimental measurements:

.. code-block:: python

    from chemi_accel.materials.experimental_validation import ExperimentalValidator

    # Initialize validator
    validator = ExperimentalValidator()

    print("=== EXPERIMENTAL VALIDATION ===")

    # Load experimental data
    experimental_data = pd.read_csv('experimental_properties.csv')
    
    # Predict properties for experimental structures
    predicted_properties = []
    for idx, row in experimental_data.iterrows():
        structure_file = row['cif_file']
        structure = cif_parser.parse_cif(structure_file)
        
        # Predict properties
        predicted = property_calc.predict_all_properties(structure)
        predicted['material_id'] = row['material_id']
        predicted_properties.append(predicted)

    predicted_df = pd.DataFrame(predicted_properties)

    # Validation analysis
    validation_results = validator.validate_predictions(
        experimental_data=experimental_data,
        predicted_data=predicted_df,
        properties=['band_gap', 'bulk_modulus', 'formation_energy']
    )

    print("Validation Results:")
    for prop, metrics in validation_results.items():
        print(f"\n{prop.upper()}:")
        print(f"  Mean Absolute Error: {metrics['mae']:.3f}")
        print(f"  Root Mean Square Error: {metrics['rmse']:.3f}")
        print(f"  R² Score: {metrics['r2']:.3f}")
        print(f"  Mean Absolute Percentage Error: {metrics['mape']:.1f}%")

    # Error analysis
    large_errors = validation_results['band_gap']['large_errors']
    if large_errors:
        print(f"\nMaterials with large band gap prediction errors:")
        for material_id in large_errors[:5]:
            exp_val = experimental_data[experimental_data['material_id'] == material_id]['band_gap'].iloc[0]
            pred_val = predicted_df[predicted_df['material_id'] == material_id]['band_gap'].iloc[0]
            error = abs(pred_val - exp_val)
            print(f"  {material_id}: Exp={exp_val:.2f} eV, Pred={pred_val:.2f} eV, Error={error:.2f} eV")

X-ray Diffraction Pattern Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate theoretical diffraction patterns:

.. code-block:: python

    from chemi_accel.materials.characterization import XRDSimulator

    # Initialize XRD simulator
    xrd_sim = XRDSimulator()

    print("=== X-RAY DIFFRACTION SIMULATION ===")

    # Simulate XRD pattern
    xrd_pattern = xrd_sim.simulate_powder_pattern(
        structure=quartz_structure,
        wavelength=1.5406,  # Cu Kα radiation (Å)
        two_theta_range=(10, 80),  # degrees
        peak_width=0.1,     # degrees FWHM
        background_noise=0.05
    )

    print("XRD Pattern Simulation:")
    print(f"  Number of reflections: {len(xrd_pattern['peaks'])}")
    print(f"  Strongest peak: {xrd_pattern['strongest_peak']['intensity']:.0f} counts")
    print(f"  Peak position: {xrd_pattern['strongest_peak']['two_theta']:.2f}°")
    print(f"  Miller indices: {xrd_pattern['strongest_peak']['hkl']}")

    # Peak analysis
    print(f"\nTop 5 Reflections:")
    for i, peak in enumerate(xrd_pattern['peaks'][:5]):
        print(f"  {i+1}. 2θ = {peak['two_theta']:.2f}°, I = {peak['intensity']:.0f}, hkl = {peak['hkl']}")

    # Export for comparison with experimental data
    xrd_data = pd.DataFrame({
        'two_theta': [p['two_theta'] for p in xrd_pattern['peaks']],
        'intensity': [p['intensity'] for p in xrd_pattern['peaks']],
        'hkl': [str(p['hkl']) for p in xrd_pattern['peaks']]
    })
    xrd_data.to_csv('simulated_xrd_pattern.csv', index=False)

Advanced Applications
---------------------

Machine Learning Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ML models for enhanced property prediction:

.. code-block:: python

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score

    print("=== MACHINE LEARNING INTEGRATION ===")

    # Prepare training data
    training_data = batch_results.dropna()
    
    # Features: structural descriptors
    feature_columns = ['packing_efficiency', 'avg_coordination_number', 'volume_per_atom']
    X = training_data[feature_columns]
    
    # Target: band gap
    y = training_data['band_gap']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train ML model
    ml_model = RandomForestRegressor(n_estimators=100, random_state=42)
    ml_model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = ml_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"ML Model Performance:")
    print(f"  Mean Absolute Error: {mae:.3f} eV")
    print(f"  R² Score: {r2:.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': ml_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nFeature Importance:")
    for _, row in feature_importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")

Uncertainty Quantification
~~~~~~~~~~~~~~~~~~~~~~~~~~

Assess prediction reliability:

.. code-block:: python

    from chemi_accel.ai.uncertainty_quantification import MaterialsUncertaintyQuantifier

    # Initialize uncertainty quantifier
    uq = MaterialsUncertaintyQuantifier()

    print("=== UNCERTAINTY QUANTIFICATION ===")

    # Make predictions with uncertainty
    uncertain_predictions = []
    for structure in predicted_structures[:5]:
        composition = structure['composition']
        
        # Predict with uncertainty
        prediction = uq.predict_with_uncertainty(
            composition=composition,
            structure_info=structure,
            property='band_gap'
        )
        
        uncertain_predictions.append({
            'composition': composition,
            'predicted_band_gap': prediction['mean'],
            'epistemic_uncertainty': prediction['epistemic_uncertainty'],
            'aleatoric_uncertainty': prediction['aleatoric_uncertainty'],
            'total_uncertainty': prediction['total_uncertainty'],
            'confidence_interval': prediction['confidence_interval']
        })

    print("Predictions with Uncertainty:")
    for pred in uncertain_predictions:
        print(f"\n{pred['composition']}:")
        print(f"  Band gap: {pred['predicted_band_gap']:.2f} ± {pred['total_uncertainty']:.2f} eV")
        print(f"  95% CI: [{pred['confidence_interval'][0]:.2f}, {pred['confidence_interval'][1]:.2f}] eV")
        print(f"  Model uncertainty: {pred['epistemic_uncertainty']:.3f} eV")
        print(f"  Data uncertainty: {pred['aleatoric_uncertainty']:.3f} eV")

Best Practices
--------------

Structure Quality Assessment
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Validate crystal structures** - Check for reasonable bond lengths and angles
2. **Assess symmetry** - Verify space group assignments
3. **Check completeness** - Ensure all atoms are properly specified
4. **Consider disorder** - Account for atomic site disorder when present

Property Prediction Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Domain applicability** - Stay within training data domain
2. **Composition constraints** - Consider elemental limitations
3. **Structure-property relationships** - Understand underlying physics
4. **Uncertainty assessment** - Always report prediction uncertainties

Experimental Validation
~~~~~~~~~~~~~~~~~~~~~~

1. **Synthesis feasibility** - Consider if materials can be made
2. **Measurement conditions** - Account for temperature, pressure effects
3. **Sample quality** - Ensure high-quality experimental samples
4. **Multiple techniques** - Use complementary characterization methods

Computational Efficiency
~~~~~~~~~~~~~~~~~~~~~~~

1. **GPU acceleration** - Use CUDA backend for large datasets
2. **Batch processing** - Process multiple structures together
3. **Memory management** - Monitor GPU memory usage
4. **Parallel processing** - Utilize multiple cores/GPUs

For more advanced materials science examples and workflows, see the :doc:`../examples/materials_science_demo` and :doc:`../notebooks/materials_discovery` sections.