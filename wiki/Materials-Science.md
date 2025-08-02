# Materials Science

ChemiAccel provides comprehensive tools for crystal structure analysis and materials property prediction. This guide covers crystallographic analysis, electronic structure calculations, and materials discovery workflows.

## Overview

The materials science module enables:
- **Crystal Structure Analysis**: CIF/POSCAR file processing with periodic boundary conditions
- **Electronic Property Prediction**: Band structure, density of states, work function calculations
- **Mechanical Property Modeling**: Elastic constants, hardness, and thermal properties
- **Materials Discovery**: High-throughput screening and property optimization
- **Defect Analysis**: Point defects, grain boundaries, and surface properties

## Quick Start

### Basic Crystal Analysis

```python
from chemi_accel.materials import MaterialsCalculator
import pandas as pd

# Initialize materials calculator
calc = MaterialsCalculator(use_gpu=True)

# Load and analyze crystal structure
structure_data = calc.from_cif_file("perovskite.cif")
print(f"Crystal system: {structure_data['crystal_system']}")
print(f"Space group: {structure_data['space_group']}")
print(f"Lattice parameters: {structure_data['lattice_parameters']}")

# Calculate basic properties
properties = calc.calculate_structure_descriptors(structure_data['structure'])
print(f"Density: {properties['density']:.3f} g/cm³")
print(f"Packing efficiency: {properties['packing_efficiency']:.3f}")
```

## Crystal Structure Processing

### Loading Different File Formats

```python
from chemi_accel.materials import CrystalParser
import glob

parser = CrystalParser()

# Load CIF files
cif_files = glob.glob("structures/*.cif")
cif_structures = []

for cif_file in cif_files:
    structure = parser.parse_cif(cif_file)
    cif_structures.append({
        'filename': cif_file,
        'formula': structure.composition.reduced_formula,
        'space_group': structure.get_space_group_info()[1],
        'volume': structure.volume,
        'structure': structure
    })

cif_df = pd.DataFrame(cif_structures)
print(f"Loaded {len(cif_df)} CIF structures")

# Load POSCAR files
poscar_files = glob.glob("structures/POSCAR*")
poscar_structures = []

for poscar_file in poscar_files:
    structure = parser.parse_poscar(poscar_file)
    poscar_structures.append({
        'filename': poscar_file,
        'formula': structure.composition.reduced_formula,
        'lattice_type': structure.lattice_type,
        'volume': structure.volume,
        'structure': structure
    })

poscar_df = pd.DataFrame(poscar_structures)
print(f"Loaded {len(poscar_df)} POSCAR structures")
```

### Crystal Structure Validation

```python
from chemi_accel.materials import StructureValidator

def validate_crystal_structures(structures):
    """Validate crystal structures for common issues."""
    
    validator = StructureValidator()
    validation_results = []
    
    for structure in structures:
        result = validator.validate_structure(structure)
        
        validation_results.append({
            'formula': structure.composition.reduced_formula,
            'is_valid': result['is_valid'],
            'symmetry_issues': result.get('symmetry_issues', []),
            'distance_issues': result.get('distance_issues', []),
            'charge_neutrality': result['charge_neutral'],
            'completeness': result['structure_complete']
        })
    
    return pd.DataFrame(validation_results)

# Validate loaded structures
all_structures = [s['structure'] for s in cif_structures + poscar_structures]
validation_df = validate_crystal_structures(all_structures)

# Filter valid structures
valid_structures = validation_df[validation_df['is_valid'] == True]
print(f"Valid structures: {len(valid_structures)}/{len(validation_df)}")
```

## Electronic Structure Analysis

### Band Structure Calculations

```python
from chemi_accel.materials import ElectronicStructureCalculator
import matplotlib.pyplot as plt

electronic_calc = ElectronicStructureCalculator(method="DFT-PBE")

def calculate_electronic_properties(structure):
    """Calculate electronic properties for a crystal structure."""
    
    # Calculate band structure
    band_structure = electronic_calc.calculate_band_structure(
        structure=structure,
        kpath_density=50,
        energy_range=(-10, 10)  # eV around Fermi level
    )
    
    # Calculate density of states
    dos = electronic_calc.calculate_density_of_states(
        structure=structure,
        energy_range=(-10, 10),
        energy_resolution=0.01  # eV
    )
    
    # Extract key properties
    properties = {
        'band_gap': band_structure['band_gap'],
        'band_gap_type': band_structure['gap_type'],  # 'direct' or 'indirect'
        'fermi_energy': band_structure['fermi_energy'],
        'valence_band_maximum': band_structure['vbm'],
        'conduction_band_minimum': band_structure['cbm'],
        'effective_mass_electron': band_structure['electron_effective_mass'],
        'effective_mass_hole': band_structure['hole_effective_mass'],
        'dos_at_fermi': dos['dos_at_fermi']
    }
    
    return properties, band_structure, dos

# Example: Silicon band structure
si_structure = parser.parse_cif("Si.cif")
si_electronic, si_bands, si_dos = calculate_electronic_properties(si_structure)

print("Silicon Electronic Properties:")
print(f"Band gap: {si_electronic['band_gap']:.3f} eV ({si_electronic['band_gap_type']})")
print(f"Electron effective mass: {si_electronic['effective_mass_electron']:.3f} m₀")
print(f"Hole effective mass: {si_electronic['effective_mass_hole']:.3f} m₀")
```

### Work Function and Surface Properties

```python
from chemi_accel.materials import SurfaceAnalyzer

def analyze_surface_properties(structure, miller_indices):
    """Analyze surface properties for different crystal faces."""
    
    surface_analyzer = SurfaceAnalyzer()
    surface_results = []
    
    for indices in miller_indices:
        # Generate surface slab
        surface_slab = surface_analyzer.create_surface_slab(
            structure=structure,
            miller_index=indices,
            slab_thickness=15,  # Å
            vacuum_thickness=20  # Å
        )
        
        # Calculate surface properties
        surface_props = surface_analyzer.calculate_surface_properties(surface_slab)
        
        surface_results.append({
            'miller_index': indices,
            'surface_energy': surface_props['surface_energy'],  # J/m²
            'work_function': surface_props['work_function'],    # eV
            'surface_roughness': surface_props['roughness'],
            'adsorption_sites': len(surface_props['adsorption_sites']),
            'surface_area': surface_props['surface_area']       # m²/g
        })
    
    return pd.DataFrame(surface_results)

# Analyze common surfaces for metals
metal_surfaces = [(1,0,0), (1,1,0), (1,1,1)]
au_structure = parser.parse_cif("Au.cif")
au_surfaces = analyze_surface_properties(au_structure, metal_surfaces)

print("Gold Surface Properties:")
print(au_surfaces[['miller_index', 'surface_energy', 'work_function']])
```

## Mechanical Properties

### Elastic Constants and Mechanical Properties

```python
from chemi_accel.materials import MechanicalPropertyCalculator

def calculate_mechanical_properties(structure):
    """Calculate comprehensive mechanical properties."""
    
    mech_calc = MechanicalPropertyCalculator()
    
    # Calculate elastic constants tensor
    elastic_constants = mech_calc.calculate_elastic_constants(
        structure=structure,
        strain_magnitude=0.01,
        method="finite_difference"
    )
    
    # Calculate derived mechanical properties
    mechanical_props = mech_calc.calculate_mechanical_properties(elastic_constants)
    
    properties = {
        'bulk_modulus': mechanical_props['bulk_modulus'],        # GPa
        'shear_modulus': mechanical_props['shear_modulus'],      # GPa
        'young_modulus': mechanical_props['young_modulus'],      # GPa
        'poisson_ratio': mechanical_props['poisson_ratio'],
        'hardness_vickers': mechanical_props['vickers_hardness'], # GPa
        'fracture_toughness': mechanical_props['fracture_toughness'], # MPa·m^0.5
        'compressibility': mechanical_props['compressibility'],   # GPa⁻¹
        'sound_velocity': mechanical_props['average_sound_velocity'], # m/s
        'debye_temperature': mechanical_props['debye_temperature'] # K
    }
    
    # Analyze elastic anisotropy
    anisotropy = mech_calc.analyze_elastic_anisotropy(elastic_constants)
    properties.update({
        'zener_anisotropy': anisotropy['zener_ratio'],
        'universal_anisotropy': anisotropy['universal_anisotropy'],
        'shear_anisotropy_percent': anisotropy['shear_anisotropy_percent']
    })
    
    return properties, elastic_constants

# Example: Calculate properties for different materials
materials = [
    ("Diamond", "C.cif"),
    ("Silicon", "Si.cif"),
    ("Iron", "Fe.cif"),
    ("Aluminum", "Al.cif")
]

mechanical_results = []
for name, cif_file in materials:
    structure = parser.parse_cif(cif_file)
    props, elastic_tensor = calculate_mechanical_properties(structure)
    props['material'] = name
    mechanical_results.append(props)

mech_df = pd.DataFrame(mechanical_results)
print("Mechanical Properties Comparison:")
print(mech_df[['material', 'bulk_modulus', 'shear_modulus', 'young_modulus', 'hardness_vickers']])
```

### Thermal Properties

```python
from chemi_accel.materials import ThermalPropertyCalculator

def calculate_thermal_properties(structure, temperature_range=(0, 1000)):
    """Calculate thermal properties over temperature range."""
    
    thermal_calc = ThermalPropertyCalculator()
    
    # Calculate phonon density of states
    phonon_dos = thermal_calc.calculate_phonon_dos(
        structure=structure,
        supercell_size=(2, 2, 2),
        force_calculator="finite_difference"
    )
    
    # Calculate thermal properties
    temperatures = np.linspace(temperature_range[0], temperature_range[1], 100)
    thermal_data = []
    
    for T in temperatures:
        if T == 0:
            continue  # Skip T=0 to avoid division by zero
            
        props = thermal_calc.calculate_thermal_properties_at_temperature(
            phonon_dos=phonon_dos,
            temperature=T
        )
        
        thermal_data.append({
            'temperature': T,
            'heat_capacity': props['heat_capacity'],        # J/mol/K
            'entropy': props['entropy'],                    # J/mol/K
            'thermal_expansion': props['thermal_expansion'], # K⁻¹
            'gruneisen_parameter': props['gruneisen_parameter'],
            'thermal_conductivity': props['thermal_conductivity'] # W/m/K
        })
    
    thermal_df = pd.DataFrame(thermal_data)
    
    # Calculate room temperature values
    rt_props = thermal_df[thermal_df['temperature'].between(295, 305)].mean()
    
    return thermal_df, rt_props

# Example: Silicon thermal properties
si_structure = parser.parse_cif("Si.cif")
si_thermal_df, si_rt = calculate_thermal_properties(si_structure)

print("Silicon Thermal Properties at Room Temperature:")
print(f"Heat capacity: {si_rt['heat_capacity']:.2f} J/mol/K")
print(f"Thermal expansion: {si_rt['thermal_expansion']:.2e} K⁻¹")
print(f"Thermal conductivity: {si_rt['thermal_conductivity']:.1f} W/m/K")

# Plot thermal properties
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0,0].plot(si_thermal_df['temperature'], si_thermal_df['heat_capacity'])
axes[0,0].set_title('Heat Capacity')
axes[0,0].set_xlabel('Temperature (K)')
axes[0,0].set_ylabel('Cp (J/mol/K)')

axes[0,1].plot(si_thermal_df['temperature'], si_thermal_df['thermal_expansion'])
axes[0,1].set_title('Thermal Expansion')
axes[0,1].set_xlabel('Temperature (K)')
axes[0,1].set_ylabel('α (K⁻¹)')

axes[1,0].plot(si_thermal_df['temperature'], si_thermal_df['thermal_conductivity'])
axes[1,0].set_title('Thermal Conductivity')
axes[1,0].set_xlabel('Temperature (K)')
axes[1,0].set_ylabel('κ (W/m/K)')

axes[1,1].plot(si_thermal_df['temperature'], si_thermal_df['entropy'])
axes[1,1].set_title('Entropy')
axes[1,1].set_xlabel('Temperature (K)')
axes[1,1].set_ylabel('S (J/mol/K)')

plt.tight_layout()
plt.show()
```

## Defect Analysis

### Point Defects

```python
from chemi_accel.materials import DefectAnalyzer

def analyze_point_defects(structure, defect_types):
    """Analyze point defects in crystal structures."""
    
    defect_analyzer = DefectAnalyzer()
    defect_results = []
    
    for defect_type in defect_types:
        if defect_type == 'vacancy':
            # Create vacancy defects
            defect_structures = defect_analyzer.create_vacancy_defects(
                structure=structure,
                supercell_size=(2, 2, 2)
            )
        elif defect_type == 'interstitial':
            # Create interstitial defects
            defect_structures = defect_analyzer.create_interstitial_defects(
                structure=structure,
                interstitial_element='H',  # Hydrogen interstitials
                supercell_size=(2, 2, 2)
            )
        elif defect_type == 'substitution':
            # Create substitutional defects
            defect_structures = defect_analyzer.create_substitutional_defects(
                structure=structure,
                original_element='Si',
                substitute_element='Ge',
                supercell_size=(2, 2, 2)
            )
        
        # Calculate defect formation energies
        for defect_struct in defect_structures:
            formation_energy = defect_analyzer.calculate_defect_formation_energy(
                perfect_structure=structure,
                defective_structure=defect_struct,
                chemical_potentials={'Si': -5.43, 'O': -4.95}  # eV
            )
            
            defect_results.append({
                'defect_type': defect_type,
                'defect_site': defect_struct.defect_site,
                'formation_energy': formation_energy,
                'electronic_structure': defect_analyzer.calculate_defect_levels(defect_struct)
            })
    
    return pd.DataFrame(defect_results)

# Example: Silicon point defects
si_structure = parser.parse_cif("Si.cif")
defect_types = ['vacancy', 'interstitial', 'substitution']
si_defects = analyze_point_defects(si_structure, defect_types)

print("Silicon Point Defect Analysis:")
print(si_defects[['defect_type', 'formation_energy']].groupby('defect_type').mean())
```

### Grain Boundaries

```python
from chemi_accel.materials import GrainBoundaryAnalyzer

def analyze_grain_boundaries(structure, gb_types):
    """Analyze grain boundary properties."""
    
    gb_analyzer = GrainBoundaryAnalyzer()
    gb_results = []
    
    for gb_type in gb_types:
        # Create grain boundary structure
        gb_structure = gb_analyzer.create_grain_boundary(
            structure=structure,
            gb_plane=gb_type['plane'],
            misorientation_angle=gb_type['angle'],
            vacuum_thickness=20  # Å
        )
        
        # Calculate grain boundary properties
        gb_props = gb_analyzer.calculate_gb_properties(gb_structure)
        
        gb_results.append({
            'gb_plane': gb_type['plane'],
            'misorientation_angle': gb_type['angle'],
            'gb_energy': gb_props['grain_boundary_energy'],    # J/m²
            'excess_volume': gb_props['excess_volume'],        # Å³/Å²
            'segregation_energy': gb_props['segregation_energy'], # eV
            'cohesive_strength': gb_props['cohesive_strength']  # GPa
        })
    
    return pd.DataFrame(gb_results)

# Example: Common grain boundaries in FCC metals
fcc_structure = parser.parse_cif("Al.cif")
gb_types = [
    {'plane': (1, 0, 0), 'angle': 36.87},  # Σ5 GB
    {'plane': (1, 1, 0), 'angle': 50.77},  # Σ11 GB
    {'plane': (1, 1, 1), 'angle': 60.00}   # Σ3 twin boundary
]

al_gb_analysis = analyze_grain_boundaries(fcc_structure, gb_types)
print("Aluminum Grain Boundary Properties:")
print(al_gb_analysis)
```

## High-Throughput Materials Discovery

### Materials Screening Pipeline

```python
from chemi_accel.materials import HighThroughputScreener
from chemi_accel.materials.databases import MaterialsProjectAPI

def high_throughput_materials_screening():
    """High-throughput screening of materials properties."""
    
    # Initialize screener and database connection
    screener = HighThroughputScreener(backend="cuda")
    mp_api = MaterialsProjectAPI(api_key="your_api_key")
    
    # Define search criteria
    search_criteria = {
        'elements': ['Ti', 'Al', 'V'],  # Titanium alloys
        'formation_energy_per_atom': {'$lt': -0.5},  # Stable compounds
        'band_gap': {'$gt': 0.0, '$lt': 3.0},  # Semiconductors
        'spacegroup.number': {'$in': list(range(195, 231))}  # Cubic systems
    }
    
    # Query materials database
    candidate_structures = mp_api.query(search_criteria, limit=1000)
    print(f"Found {len(candidate_structures)} candidate structures")
    
    # Screen for target properties
    screening_results = []
    
    target_properties = {
        'bulk_modulus': (100, 300),      # GPa - moderate stiffness
        'shear_modulus': (50, 150),      # GPa
        'band_gap': (1.0, 2.5),          # eV - suitable for photovoltaics
        'formation_energy': (-2.0, -0.5) # eV/atom - stable
    }
    
    for structure_data in candidate_structures:
        structure = structure_data['structure']
        
        # Calculate properties
        props = screener.calculate_comprehensive_properties(structure)
        
        # Check if meets criteria
        meets_criteria = True
        for prop_name, (min_val, max_val) in target_properties.items():
            if not (min_val <= props.get(prop_name, 0) <= max_val):
                meets_criteria = False
                break
        
        if meets_criteria:
            screening_results.append({
                'formula': structure.composition.reduced_formula,
                'space_group': structure.get_space_group_info()[1],
                'bulk_modulus': props['bulk_modulus'],
                'shear_modulus': props['shear_modulus'],
                'band_gap': props['band_gap'],
                'formation_energy': props['formation_energy'],
                'density': props['density'],
                'hardness': props['hardness_vickers']
            })
    
    results_df = pd.DataFrame(screening_results)
    
    # Rank by multi-criteria scoring
    def calculate_score(row):
        # Normalized scoring (higher is better)
        modulus_score = (row['bulk_modulus'] - 100) / 200  # 0-1 scale
        gap_score = 1 - abs(row['band_gap'] - 1.75) / 1.5  # Prefer ~1.75 eV
        stability_score = (abs(row['formation_energy']) - 0.5) / 1.5
        
        return 0.4 * modulus_score + 0.3 * gap_score + 0.3 * stability_score
    
    results_df['composite_score'] = results_df.apply(calculate_score, axis=1)
    results_df = results_df.sort_values('composite_score', ascending=False)
    
    print("Top 10 Materials Candidates:")
    print(results_df.head(10)[['formula', 'bulk_modulus', 'band_gap', 'composite_score']])
    
    return results_df

# Run high-throughput screening
screening_results = high_throughput_materials_screening()
```

### Property Prediction Models

```python
from chemi_accel.materials.ml import MaterialPropertyPredictor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def train_materials_property_model():
    """Train ML models for materials property prediction."""
    
    # Load materials database
    materials_db = load_materials_database()  # Your materials database
    
    # Calculate structural descriptors
    calc = MaterialsCalculator(use_gpu=True)
    
    X_features = []
    y_properties = []
    
    for material in materials_db:
        structure = material['structure']
        
        # Calculate structural descriptors
        descriptors = calc.calculate_structure_descriptors(structure)
        
        # Feature vector
        feature_vector = [
            descriptors['density'],
            descriptors['packing_efficiency'],
            descriptors['coordination_number_avg'],
            descriptors['bond_length_avg'],
            descriptors['electronegativity_avg'],
            descriptors['atomic_radius_avg'],
            descriptors['volume_per_atom'],
            len(structure.species)  # Number of unique elements
        ]
        
        X_features.append(feature_vector)
        
        # Target properties
        y_properties.append([
            material['bulk_modulus'],
            material['band_gap'],
            material['formation_energy'],
            material['thermal_conductivity']
        ])
    
    X = np.array(X_features)
    y = np.array(y_properties)
    
    # Train models for each property
    property_names = ['bulk_modulus', 'band_gap', 'formation_energy', 'thermal_conductivity']
    models = {}
    
    for i, prop_name in enumerate(property_names):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y[:, i], test_size=0.2, random_state=42
        )
        
        # Train Random Forest model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        models[prop_name] = model
        
        print(f"{prop_name}:")
        print(f"  Training R² = {train_score:.3f}")
        print(f"  Test R² = {test_score:.3f}")
        
        # Feature importance
        feature_names = ['density', 'packing_eff', 'coord_num', 'bond_length', 
                        'electronegativity', 'atomic_radius', 'vol_per_atom', 'n_elements']
        importance = model.feature_importances_
        
        print(f"  Top features: {', '.join([f'{name}({imp:.3f})' for name, imp in 
                                          sorted(zip(feature_names, importance), 
                                                key=lambda x: x[1], reverse=True)[:3]])}")
    
    return models

# Train materials property models
materials_models = train_materials_property_model()
```

## Specialized Applications

### Photovoltaic Materials

```python
from chemi_accel.materials import PhotovoltaicAnalyzer

def screen_photovoltaic_materials():
    """Screen materials for photovoltaic applications."""
    
    pv_analyzer = PhotovoltaicAnalyzer()
    
    # Load candidate structures (perovskites, chalcopyrites, etc.)
    pv_candidates = load_pv_candidate_structures()
    
    pv_results = []
    
    for structure in pv_candidates:
        # Calculate photovoltaic-relevant properties
        pv_props = pv_analyzer.calculate_pv_properties(structure)
        
        pv_results.append({
            'formula': structure.composition.reduced_formula,
            'band_gap': pv_props['band_gap'],
            'absorption_coefficient': pv_props['absorption_coefficient'],
            'carrier_lifetime': pv_props['carrier_lifetime'],
            'mobility_electron': pv_props['electron_mobility'],
            'mobility_hole': pv_props['hole_mobility'],
            'defect_tolerance': pv_props['defect_tolerance_factor'],
            'stability_score': pv_props['thermodynamic_stability'],
            'efficiency_limit': pv_props['shockley_queisser_limit']
        })
    
    pv_df = pd.DataFrame(pv_results)
    
    # Filter promising candidates
    promising_pv = pv_df[
        (pv_df['band_gap'] > 1.0) & (pv_df['band_gap'] < 1.8) &  # Optimal gap
        (pv_df['absorption_coefficient'] > 1e4) &                  # Strong absorption
        (pv_df['defect_tolerance'] > 0.5) &                       # Defect tolerant
        (pv_df['stability_score'] > 0.7)                          # Stable
    ].sort_values('efficiency_limit', ascending=False)
    
    print("Promising Photovoltaic Materials:")
    print(promising_pv.head(10)[['formula', 'band_gap', 'efficiency_limit', 'stability_score']])
    
    return promising_pv

pv_materials = screen_photovoltaic_materials()
```

### Battery Materials

```python
from chemi_accel.materials import BatteryMaterialAnalyzer

def analyze_battery_materials():
    """Analyze materials for battery applications."""
    
    battery_analyzer = BatteryMaterialAnalyzer()
    
    # Cathode materials analysis
    cathode_candidates = [
        "LiCoO2", "LiFePO4", "LiMn2O4", "LiNiO2", 
        "Li(Ni1/3Mn1/3Co1/3)O2", "LiNi0.8Mn0.1Co0.1O2"
    ]
    
    cathode_results = []
    
    for formula in cathode_candidates:
        structure = get_structure_from_formula(formula)
        
        # Calculate battery-relevant properties
        battery_props = battery_analyzer.calculate_cathode_properties(structure)
        
        cathode_results.append({
            'formula': formula,
            'theoretical_capacity': battery_props['theoretical_capacity'],  # mAh/g
            'average_voltage': battery_props['average_voltage'],            # V vs Li/Li+
            'energy_density': battery_props['energy_density'],             # Wh/kg
            'li_diffusivity': battery_props['li_ion_diffusivity'],         # cm²/s
            'volume_change': battery_props['volume_change_percent'],       # %
            'thermal_stability': battery_props['thermal_stability_temp'],   # °C
            'cycle_life_prediction': battery_props['cycle_life_estimate']   # cycles
        })
    
    cathode_df = pd.DataFrame(cathode_results)
    
    print("Cathode Materials Comparison:")
    print(cathode_df[['formula', 'theoretical_capacity', 'average_voltage', 
                     'energy_density', 'thermal_stability']])
    
    return cathode_df

battery_analysis = analyze_battery_materials()
```

## Best Practices

### 1. Structure Validation
```python
# Always validate crystal structures before analysis
def validate_before_analysis(structure):
    validator = StructureValidator()
    result = validator.validate_structure(structure)
    
    if not result['is_valid']:
        print(f"Structure validation failed: {result['issues']}")
        return False
    return True
```

### 2. Convergence Testing
```python
# Test convergence of calculated properties
def test_convergence(structure, property_name, parameter_range):
    results = []
    for param_value in parameter_range:
        calc_result = calculate_property(structure, **{parameter_name: param_value})
        results.append(calc_result)
    
    # Check convergence
    relative_change = abs(results[-1] - results[-2]) / results[-1]
    return relative_change < 0.01  # 1% convergence criterion
```

### 3. Cross-Validation
```python
# Cross-validate predictions with experimental data
def validate_predictions(calculated_values, experimental_values):
    correlation = np.corrcoef(calculated_values, experimental_values)[0, 1]
    mae = np.mean(np.abs(calculated_values - experimental_values))
    
    print(f"Correlation: {correlation:.3f}")
    print(f"Mean Absolute Error: {mae:.3f}")
    
    return correlation > 0.8  # Acceptable correlation threshold
```

---

*For specialized materials applications, see the [Polymer Science](Polymer-Science) guide and [Examples](Examples) gallery.*
