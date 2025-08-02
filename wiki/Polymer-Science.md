# Polymer Science

ChemiAccel provides specialized tools for polymer analysis, offering comprehensive workflows for monomer analysis, copolymer composition prediction, and polymer property modeling. This guide covers polymer-specific features and best practices.

## Overview

The polymer science module in ChemiAccel enables:
- **Monomer Analysis**: Reactivity prediction and polymerization behavior
- **Copolymer Design**: Composition optimization and property prediction
- **Polymer Property Modeling**: Glass transition temperature, mechanical properties
- **Sequence Analysis**: Chain structure and topology analysis
- **Predictive Modeling**: ML-based property prediction for novel polymers

## Quick Start

### Basic Polymer Analysis

```python
from chemi_accel.polymer import CopolymerAnalyzer, MonomerPredictor
from chemi_accel import Calculator

# Initialize analyzers
copolymer_analyzer = CopolymerAnalyzer()
monomer_predictor = MonomerPredictor()

# Analyze monomer reactivity
styrene_reactivity = monomer_predictor.predict_reactivity("c1ccc(C=C)cc1")
print(f"Styrene reactivity ratios: {styrene_reactivity}")

# Analyze copolymer composition
composition_data = copolymer_analyzer.analyze_copolymer_composition(
    monomers=["c1ccc(C=C)cc1", "C=CC#N"],  # Styrene, Acrylonitrile
    ratios=[0.7, 0.3]
)
print(f"Glass transition temperature: {composition_data['tg']:.1f}°C")
```

## Monomer Analysis

### Reactivity Prediction

```python
from chemi_accel.polymer import MonomerPredictor
import pandas as pd

predictor = MonomerPredictor()

# Common vinyl monomers
monomers = [
    "C=C",                    # Ethylene
    "C=CC",                   # Propylene  
    "c1ccc(C=C)cc1",         # Styrene
    "C=CC#N",                # Acrylonitrile
    "C=CC(=O)OC",            # Methyl acrylate
    "C=CC(=O)O",             # Acrylic acid
    "C=CC(=O)N",             # Acrylamide
    "C=CC(C)(C)C(=O)OC",     # Methyl methacrylate
]

# Predict reactivity parameters for all monomers
reactivity_data = []
for monomer in monomers:
    reactivity = predictor.predict_reactivity(monomer)
    reactivity_data.append({
        'smiles': monomer,
        'r1_styrene': reactivity.get('r1_styrene', None),
        'r2_styrene': reactivity.get('r2_styrene', None),
        'Q': reactivity.get('Q', None),  # Resonance parameter
        'e': reactivity.get('e', None),  # Polarity parameter
        'polymerization_rate': predictor.predict_radical_polymerization_rate(monomer)
    })

reactivity_df = pd.DataFrame(reactivity_data)
print(reactivity_df)
```

### Monomer Screening

```python
def screen_monomers_for_properties(monomers, target_tg_range, target_modulus_range):
    """Screen monomers for desired polymer properties."""
    
    predictor = MonomerPredictor()
    suitable_monomers = []
    
    for monomer in monomers:
        # Predict homopolymer properties
        homo_props = predictor.predict_homopolymer_properties(monomer)
        
        tg = homo_props.get('glass_transition_temp', None)
        modulus = homo_props.get('elastic_modulus', None)
        
        if (target_tg_range[0] <= tg <= target_tg_range[1] and 
            target_modulus_range[0] <= modulus <= target_modulus_range[1]):
            
            suitable_monomers.append({
                'smiles': monomer,
                'tg': tg,
                'modulus': modulus,
                'processability': homo_props.get('processability_index', 0)
            })
    
    return pd.DataFrame(suitable_monomers)

# Screen for flexible polymers
flexible_candidates = screen_monomers_for_properties(
    monomers=monomers,
    target_tg_range=(-50, 50),      # Low Tg for flexibility
    target_modulus_range=(0.1, 10)  # Moderate modulus
)

print("Flexible polymer candidates:")
print(flexible_candidates)
```

## Copolymer Analysis

### Composition-Property Relationships

```python
from chemi_accel.polymer import CopolymerAnalyzer
import numpy as np
import matplotlib.pyplot as plt

analyzer = CopolymerAnalyzer()

# Study styrene-acrylonitrile copolymers
styrene = "c1ccc(C=C)cc1"
acrylonitrile = "C=CC#N"

compositions = np.linspace(0, 1, 21)  # 0 to 100% styrene
results = []

for styrene_fraction in compositions:
    an_fraction = 1 - styrene_fraction
    
    analysis = analyzer.analyze_copolymer_composition(
        monomers=[styrene, acrylonitrile],
        ratios=[styrene_fraction, an_fraction]
    )
    
    results.append({
        'styrene_fraction': styrene_fraction,
        'tg': analysis['glass_transition_temp'],
        'tensile_strength': analysis['mechanical_properties']['tensile_strength'],
        'impact_strength': analysis['mechanical_properties']['impact_strength'],
        'chemical_resistance': analysis['chemical_resistance_index']
    })

copolymer_df = pd.DataFrame(results)

# Plot composition-property relationships
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0,0].plot(copolymer_df['styrene_fraction'], copolymer_df['tg'])
axes[0,0].set_title('Glass Transition Temperature')
axes[0,0].set_xlabel('Styrene Fraction')
axes[0,0].set_ylabel('Tg (°C)')

axes[0,1].plot(copolymer_df['styrene_fraction'], copolymer_df['tensile_strength'])
axes[0,1].set_title('Tensile Strength')
axes[0,1].set_xlabel('Styrene Fraction')
axes[0,1].set_ylabel('Tensile Strength (MPa)')

axes[1,0].plot(copolymer_df['styrene_fraction'], copolymer_df['impact_strength'])
axes[1,0].set_title('Impact Strength')
axes[1,0].set_xlabel('Styrene Fraction')
axes[1,0].set_ylabel('Impact Strength (J/m)')

axes[1,1].plot(copolymer_df['styrene_fraction'], copolymer_df['chemical_resistance'])
axes[1,1].set_title('Chemical Resistance')
axes[1,1].set_xlabel('Styrene Fraction')  
axes[1,1].set_ylabel('Resistance Index')

plt.tight_layout()
plt.show()
```

### Optimal Composition Finding

```python
from scipy.optimize import minimize
from chemi_accel.polymer import MultiObjectivePolymerOptimizer

def optimize_copolymer_composition():
    """Find optimal copolymer composition for multiple objectives."""
    
    optimizer = MultiObjectivePolymerOptimizer()
    
    # Define objectives (minimize negative values to maximize)
    def objective_function(composition):
        styrene_frac, an_frac = composition
        
        analysis = analyzer.analyze_copolymer_composition(
            monomers=[styrene, acrylonitrile],
            ratios=[styrene_frac, an_frac]
        )
        
        # Multi-objective: maximize strength, minimize cost, target Tg
        strength = analysis['mechanical_properties']['tensile_strength']
        cost = estimate_polymer_cost([styrene, acrylonitrile], [styrene_frac, an_frac])
        tg_deviation = abs(analysis['glass_transition_temp'] - 100)  # Target Tg = 100°C
        
        # Weighted objective (customize weights as needed)
        return -(0.4 * strength - 0.3 * cost - 0.3 * tg_deviation)
    
    # Constraint: compositions must sum to 1
    constraints = {'type': 'eq', 'fun': lambda x: x[0] + x[1] - 1}
    bounds = [(0, 1), (0, 1)]  # Each component between 0 and 1
    
    # Optimize
    result = minimize(
        objective_function,
        x0=[0.5, 0.5],  # Initial guess
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    if result.success:
        optimal_composition = result.x
        print(f"Optimal composition: {optimal_composition[0]:.3f} styrene, {optimal_composition[1]:.3f} AN")
        
        # Analyze optimal composition
        optimal_analysis = analyzer.analyze_copolymer_composition(
            monomers=[styrene, acrylonitrile],
            ratios=optimal_composition
        )
        print(f"Properties: Tg = {optimal_analysis['glass_transition_temp']:.1f}°C")
        print(f"Tensile strength = {optimal_analysis['mechanical_properties']['tensile_strength']:.1f} MPa")
    else:
        print("Optimization failed")

optimize_copolymer_composition()
```

## Advanced Polymer Modeling

### Sequence Distribution Analysis

```python
from chemi_accel.polymer import SequenceAnalyzer

def analyze_copolymer_microstructure():
    """Analyze copolymer sequence distribution and microstructure."""
    
    sequence_analyzer = SequenceAnalyzer()
    
    # Calculate reactivity ratios
    r1 = 0.04  # Styrene reactivity ratio toward AN
    r2 = 0.40  # AN reactivity ratio toward styrene
    
    # Analyze sequence distribution
    sequences = sequence_analyzer.calculate_sequence_distribution(
        r1=r1, r2=r2,
        f1=0.5,  # Feed composition (50% styrene)
        chain_length=1000
    )
    
    print("Sequence Analysis:")
    print(f"Average block length S: {sequences['avg_block_length_1']:.2f}")
    print(f"Average block length AN: {sequences['avg_block_length_2']:.2f}")
    print(f"Probability of S-S dyads: {sequences['dyad_prob_11']:.3f}")
    print(f"Probability of S-AN dyads: {sequences['dyad_prob_12']:.3f}")
    print(f"Probability of AN-AN dyads: {sequences['dyad_prob_22']:.3f}")
    
    # Generate representative chain
    chain = sequence_analyzer.generate_representative_chain(
        reactivity_ratios=[r1, r2],
        feed_composition=[0.5, 0.5],
        chain_length=100
    )
    
    print(f"\nRepresentative chain (first 20 units): {chain[:20]}")
    
    return sequences

microstructure = analyze_copolymer_microstructure()
```

### Polymer Conformation Analysis

```python
from chemi_accel.polymer import ConformerGenerator, PolymerConformationAnalyzer

def analyze_polymer_conformation():
    """Analyze polymer chain conformations and properties."""
    
    # Generate polymer conformations
    conformer_gen = ConformerGenerator()
    conformation_analyzer = PolymerConformationAnalyzer()
    
    # Create oligomer model (10-mer of polyethylene)
    pe_oligomer = "C" + "C" * 9  # Simplified representation
    
    # Generate multiple conformations
    conformations = conformer_gen.generate_polymer_conformations(
        polymer_smiles=pe_oligomer,
        num_conformations=100,
        algorithm="molecular_dynamics",
        temperature=300,  # K
        simulation_time=1000  # ps
    )
    
    # Analyze conformational properties
    for i, conf in enumerate(conformations[:5]):  # Analyze first 5 conformations
        properties = conformation_analyzer.analyze_conformation(conf)
        
        print(f"Conformation {i+1}:")
        print(f"  Radius of gyration: {properties['radius_of_gyration']:.2f} Å")
        print(f"  End-to-end distance: {properties['end_to_end_distance']:.2f} Å")
        print(f"  Asphericity: {properties['asphericity']:.3f}")
        print(f"  Persistence length: {properties['persistence_length']:.2f} Å")
    
    # Calculate ensemble averages
    ensemble_props = conformation_analyzer.calculate_ensemble_properties(conformations)
    print(f"\nEnsemble averages:")
    print(f"  <Rg²>: {ensemble_props['mean_rg_squared']:.2f} Å²")
    print(f"  <R²>: {ensemble_props['mean_end_to_end_squared']:.2f} Å²")
    print(f"  Characteristic ratio: {ensemble_props['characteristic_ratio']:.2f}")

analyze_polymer_conformation()
```

## Thermoset Analysis

### Cross-linking Prediction

```python
from chemi_accel.polymer import ThermosetAnalyzer

def analyze_thermoset_properties():
    """Analyze thermoset polymer cross-linking and properties."""
    
    thermoset_analyzer = ThermosetAnalyzer()
    
    # Epoxy resin system
    epoxy_resin = "C1OC1COc2ccc(C(C)(C)c3ccc(OCC4CO4)cc3)cc2"  # DGEBA
    hardener = "NCCCCCCN"  # Hexamethylene diamine
    
    # Predict cross-linking behavior
    crosslink_analysis = thermoset_analyzer.predict_crosslinking(
        resin_smiles=epoxy_resin,
        hardener_smiles=hardener,
        stoichiometric_ratio=1.0,
        cure_temperature=120,  # °C
        cure_time=2  # hours
    )
    
    print("Cross-linking Analysis:")
    print(f"Gel point: {crosslink_analysis['gel_point']:.1f}% conversion")
    print(f"Final conversion: {crosslink_analysis['final_conversion']:.1f}%")
    print(f"Cross-link density: {crosslink_analysis['crosslink_density']:.2e} mol/cm³")
    
    # Predict final properties
    final_props = thermoset_analyzer.predict_final_properties(crosslink_analysis)
    
    print(f"\nFinal Properties:")
    print(f"Glass transition temperature: {final_props['tg']:.1f}°C")
    print(f"Elastic modulus: {final_props['elastic_modulus']:.1f} GPa")
    print(f"Coefficient of thermal expansion: {final_props['cte']:.1e} /°C")
    print(f"Thermal conductivity: {final_props['thermal_conductivity']:.3f} W/m·K")
    
    return crosslink_analysis, final_props

crosslink_data, thermoset_props = analyze_thermoset_properties()
```

### Cure Kinetics Modeling

```python
from chemi_accel.polymer import CureKineticsModeler
import numpy as np

def model_cure_kinetics():
    """Model thermoset cure kinetics and optimize cure cycle."""
    
    kinetics_modeler = CureKineticsModeler()
    
    # Define cure cycle parameters
    temperature_profile = np.array([
        [0, 25],      # Start at room temperature
        [10, 80],     # Heat to 80°C in 10 min
        [30, 120],    # Heat to 120°C over 20 min
        [150, 120],   # Hold at 120°C for 120 min
        [180, 25]     # Cool down over 30 min
    ])
    
    # Model cure kinetics
    cure_model = kinetics_modeler.fit_autocatalytic_model(
        epoxy_system="DGEBA_amine",
        temperature_profile=temperature_profile,
        initial_conditions={'conversion': 0.0, 'catalyst_activity': 1.0}
    )
    
    # Predict conversion vs time
    time_points = np.linspace(0, 180, 100)
    conversion_profile = cure_model.predict_conversion(time_points)
    
    # Optimize cure cycle
    optimal_cycle = kinetics_modeler.optimize_cure_cycle(
        target_conversion=0.95,
        max_temperature=150,
        max_cure_time=180,
        energy_penalty_weight=0.1
    )
    
    print("Optimal Cure Cycle:")
    for stage in optimal_cycle['temperature_profile']:
        print(f"  {stage['time']:.0f} min: {stage['temperature']:.0f}°C")
    
    print(f"Total cure time: {optimal_cycle['total_time']:.0f} min")
    print(f"Energy consumption: {optimal_cycle['energy_consumption']:.1f} kJ")
    print(f"Final conversion: {optimal_cycle['final_conversion']:.1f}%")
    
    return cure_model, optimal_cycle

cure_model, optimized_cycle = model_cure_kinetics()
```

## High-Throughput Polymer Screening

### Automated Property Prediction

```python
from chemi_accel.polymer import HighThroughputPolymerScreener
import itertools

def high_throughput_polymer_screening():
    """Screen large numbers of polymer combinations."""
    
    screener = HighThroughputPolymerScreener(backend="cuda")  # Use GPU acceleration
    
    # Define monomer libraries
    vinyl_monomers = [
        "C=C",                    # Ethylene
        "C=CC",                   # Propylene
        "c1ccc(C=C)cc1",         # Styrene
        "C=CC#N",                # Acrylonitrile
        "C=CC(=O)OC",            # Methyl acrylate
        "C=CC(=O)OCCCC",         # Butyl acrylate
        "C=CC(=O)N",             # Acrylamide
        "C=CC(C)C(=O)OC",        # Methyl methacrylate
        "C=CC(C)C(=O)OCCCC",     # Butyl methacrylate
        "C=CC(=O)OCCO"           # 2-Hydroxyethyl acrylate
    ]
    
    # Generate all binary copolymer combinations
    copolymer_combinations = list(itertools.combinations(vinyl_monomers, 2))
    
    print(f"Screening {len(copolymer_combinations)} copolymer combinations...")
    
    # Screen properties for different compositions
    screening_results = []
    
    for monomer1, monomer2 in copolymer_combinations:
        for ratio1 in [0.1, 0.3, 0.5, 0.7, 0.9]:  # Different compositions
            ratio2 = 1 - ratio1
            
            # Predict properties
            props = screener.predict_copolymer_properties(
                monomers=[monomer1, monomer2],
                ratios=[ratio1, ratio2]
            )
            
            screening_results.append({
                'monomer1': monomer1,
                'monomer2': monomer2,
                'ratio1': ratio1,
                'tg': props['glass_transition_temp'],
                'tensile_strength': props['mechanical_properties']['tensile_strength'],
                'elongation': props['mechanical_properties']['elongation_at_break'],
                'solubility_parameter': props['solubility_parameter'],
                'processability': props['processability_index']
            })
    
    screening_df = pd.DataFrame(screening_results)
    
    # Find best candidates for specific applications
    # Example: Flexible film application
    flexible_film_criteria = (
        (screening_df['tg'] < 25) &         # Low Tg for flexibility
        (screening_df['elongation'] > 100) & # High elongation
        (screening_df['tensile_strength'] > 20) & # Adequate strength
        (screening_df['processability'] > 0.7)    # Good processability
    )
    
    flexible_candidates = screening_df[flexible_film_criteria].sort_values(
        'tensile_strength', ascending=False
    )
    
    print(f"\nTop 5 flexible film candidates:")
    print(flexible_candidates.head()[['monomer1', 'monomer2', 'ratio1', 'tg', 'tensile_strength', 'elongation']])
    
    return screening_df

screening_results = high_throughput_polymer_screening()
```

## Machine Learning for Polymer Properties

### Training Custom Polymer Property Models

```python
from chemi_accel.polymer.ml import PolymerPropertyPredictor
from chemi_accel import Calculator
import numpy as np

def train_polymer_property_model():
    """Train ML models for polymer property prediction."""
    
    # Load polymer dataset
    polymer_database = load_polymer_database()  # Your polymer database
    
    # Calculate molecular descriptors for monomers
    calc = Calculator(backend="cuda")
    
    # Process monomer data
    monomer_descriptors = []
    polymer_properties = []
    
    for entry in polymer_database:
        monomers = entry['monomers']
        ratios = entry['ratios']
        
        # Calculate descriptors for each monomer
        monomer_descs = []
        for monomer in monomers:
            desc = calc.pandas([monomer], descriptors="polymer_relevant")
            monomer_descs.append(desc.iloc[0].values)
        
        # Create composition-weighted descriptor vector
        composition_desc = np.average(monomer_descs, axis=0, weights=ratios)
        monomer_descriptors.append(composition_desc)
        
        # Target properties
        polymer_properties.append([
            entry['glass_transition_temp'],
            entry['tensile_strength'],
            entry['elongation_at_break'],
            entry['elastic_modulus']
        ])
    
    X = np.array(monomer_descriptors)
    y = np.array(polymer_properties)
    
    # Train property prediction models
    property_predictor = PolymerPropertyPredictor()
    
    # Train separate models for each property
    models = {}
    property_names = ['tg', 'tensile_strength', 'elongation', 'modulus']
    
    for i, prop_name in enumerate(property_names):
        model = property_predictor.train_property_model(
            X_train=X,
            y_train=y[:, i],
            model_type="neural_network",
            validation_split=0.2,
            epochs=100
        )
        models[prop_name] = model
        
        print(f"{prop_name} model R² = {model.score:.3f}")
    
    # Test on new polymer compositions
    test_compositions = [
        (['c1ccc(C=C)cc1', 'C=CC#N'], [0.6, 0.4]),  # Styrene-AN
        (['C=CC(=O)OC', 'C=CC(=O)OCCCC'], [0.5, 0.5])  # MA-BA
    ]
    
    for monomers, ratios in test_compositions:
        # Calculate descriptors
        test_descs = []
        for monomer in monomers:
            desc = calc.pandas([monomer], descriptors="polymer_relevant")
            test_descs.append(desc.iloc[0].values)
        
        test_composition_desc = np.average(test_descs, axis=0, weights=ratios)
        
        # Predict properties
        predictions = {}
        for prop_name, model in models.items():
            pred = model.predict([test_composition_desc])[0]
            predictions[prop_name] = pred
        
        print(f"\nPredictions for {monomers} at {ratios}:")
        for prop, value in predictions.items():
            print(f"  {prop}: {value:.2f}")
    
    return models

# Train polymer property models
polymer_models = train_polymer_property_model()
```

## Best Practices

### 1. Validate Predictions Experimentally
```python
# Always validate computational predictions with experimental data
def validate_polymer_predictions(predictions, experimental_data):
    correlation = np.corrcoef(predictions, experimental_data)[0, 1]
    rmse = np.sqrt(np.mean((predictions - experimental_data)**2))
    return correlation, rmse
```

### 2. Consider Processing Conditions
```python
# Account for processing conditions in property predictions
processing_conditions = {
    'temperature': 200,  # °C
    'pressure': 1.0,     # bar
    'shear_rate': 100,   # s⁻¹
    'cooling_rate': 10   # °C/min
}

adjusted_properties = analyzer.adjust_for_processing(
    base_properties=predicted_props,
    conditions=processing_conditions
)
```

### 3. Use Appropriate Model Validation
```python
# Cross-validate polymer property models
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    model, X, y, 
    cv=5, 
    scoring='r2'
)
print(f"Cross-validation R² = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
```

---

*For more information on specific polymer applications, see the [Examples](Examples) gallery and [Materials Science](Materials-Science) guide.*
