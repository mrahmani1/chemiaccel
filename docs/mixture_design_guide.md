# ChemiAccel Mixture & Formulation Design Guide

The ChemiAccel Mixture module provides comprehensive AI-driven tools for complex mixture and formulation design. This guide covers all major capabilities from basic mixture analysis to advanced optimization and visualization.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Core Components](#core-components)
4. [Phase Behavior Prediction](#phase-behavior-prediction)
5. [Interaction Modeling](#interaction-modeling)
6. [Inverse Design](#inverse-design)
7. [Dynamic Processes](#dynamic-processes)
8. [Visualization](#visualization)
9. [Advanced Examples](#advanced-examples)
10. [API Reference](#api-reference)

## Overview

The mixture module implements the following key capabilities:

- **GPU-accelerated phase behavior prediction** using thermodynamic models (NRTL, UNIQUAC, Peng-Robinson) and AI surrogates
- **Inter-component interaction modeling** for hydrogen bonding, π-π stacking, van der Waals, electrostatic, and polymer-solvent interactions
- **Inverse design for optimal mixture composition** with multi-objective optimization and constraint handling
- **Dynamic process integration** for crystallization, mixing, phase separation, and other formulation processes
- **Interactive visualization** of phase diagrams, property landscapes, and molecular interactions

## Quick Start

### Basic Mixture Analysis

```python
from chemi_accel.mixture import MixtureCalculator

# Create calculator with GPU acceleration
calc = MixtureCalculator(backend="cuda", use_gpu=True)

# Define mixture components: ethanol-water mixture
components = [
    ("CCO", 0.6),  # Ethanol - 60%
    ("O", 0.4)     # Water - 40%
]

# Create mixture
mixture = calc.create_mixture(
    components, 
    temperature=298.15,
    pressure=101325.0
)

# Calculate basic properties
properties = calc.calculate_mixture_properties(mixture)
print(f"Average molecular weight: {properties['molecular_weight_avg']:.2f}")
```

### Phase Behavior Prediction

```python
# Predict phase equilibrium
phase_result = calc.predict_phase_behavior(
    mixture,
    temperature_range=(250, 400),
    pressure_range=(1e5, 5e5),
    model='nrtl'
)

print(f"Number of phases: {phase_result.n_phases}")
print(f"System stable: {phase_result.is_stable}")
```

### Interaction Analysis

```python
# Calculate molecular interactions
interactions = calc.model_interactions(
    mixture,
    interaction_types=['hydrogen_bonding', 'van_der_waals']
)

# Predict mixing behavior
mixing_prediction = calc.interaction_modeler.predict_mixing_behavior(
    mixture, interactions
)
print(f"Miscibility score: {mixing_prediction['miscibility_score']:.3f}")
```

## Core Components

### MixtureCalculator

The main interface for all mixture analysis tasks:

```python
calc = MixtureCalculator(
    backend="cuda",        # "cuda", "jax", or "cpu"
    temperature=298.15,    # Default temperature (K)
    pressure=101325.0,     # Default pressure (Pa)
    use_gpu=True          # Enable GPU acceleration
)
```

### MixtureData

Data structure representing mixture systems:

```python
from chemi_accel.mixture import ComponentData, MixtureData

# Manual creation
components = [
    ComponentData(
        smiles="CCO",
        mol_fraction=0.5,
        molecular_weight=46.07,
        density=0.789
    ),
    ComponentData(
        smiles="O", 
        mol_fraction=0.5
    )
]

mixture = MixtureData(
    components=components,
    temperature=298.15,
    pressure=101325.0
)

# Access properties
print(f"Components: {mixture.n_components}")
print(f"SMILES: {mixture.smiles_list}")
print(f"Composition: {mixture.composition_vector}")
```

## Phase Behavior Prediction

### Thermodynamic Models

The module supports multiple thermodynamic models:

#### NRTL Model
```python
# Use NRTL for liquid-liquid equilibria
phase_result = calc.predict_phase_equilibrium(
    mixture, 
    model='nrtl',
    temperature_range=(250, 400)
)
```

#### Peng-Robinson EOS
```python
# Use Peng-Robinson for vapor-liquid equilibria
phase_result = calc.predict_phase_equilibrium(
    mixture,
    model='peng_robinson',
    pressure_range=(1e5, 10e5)
)
```

#### AI Surrogate Models
```python
# Train AI model for fast predictions
import pandas as pd

# Prepare training data
training_data = pd.DataFrame({
    'temperature': [300, 320, 340],
    'pressure': [1e5, 2e5, 3e5],
    'composition_0': [0.5, 0.6, 0.7],
    'composition_1': [0.5, 0.4, 0.3],
    'phase_split': [0.0, 0.2, 0.8],
    'stability': [0.9, 0.7, 0.5]
})

# Train AI predictor
ai_model = calc.phase_predictor.models['ai_surrogate']
ai_model.train_model(training_data)

# Use for predictions
phase_result = calc.predict_phase_behavior(mixture, model='ai_surrogate')
```

### Phase Diagram Generation

```python
# Generate comprehensive phase diagram
phase_result = calc.predict_phase_behavior(
    mixture,
    temperature_range=(250, 400),
    pressure_range=(1e5, 5e5),
    n_points=100  # Resolution
)

# Access phase diagram data
if phase_result.phase_diagram_data:
    data_points = phase_result.phase_diagram_data['data_points']
    print(f"Generated {len(data_points)} phase diagram points")
```

## Interaction Modeling

### Interaction Types

The module quantifies various molecular interactions:

```python
# Calculate all interaction types
interactions = calc.model_interactions(
    mixture,
    interaction_types=[
        'hydrogen_bonding',    # H-bond donors/acceptors
        'pi_stacking',        # Aromatic π-π interactions
        'van_der_waals',      # Lennard-Jones potential
        'electrostatic',      # Coulombic interactions
        'polymer_solvent'     # Flory-Huggins theory
    ]
)

# Analyze results
for interaction in interactions:
    print(f"{interaction.interaction_type}: "
          f"{interaction.energy:.2f} kJ/mol between "
          f"components {interaction.component_i}-{interaction.component_j}")
```

### Interaction Matrix

```python
# Build interaction matrix
n_components = mixture.n_components
interaction_matrix = calc.interaction_modeler.build_interaction_matrix(
    interactions, n_components
)

print("Interaction matrix (kJ/mol):")
print(interaction_matrix)
```

### Mixing Predictions

```python
# Predict mixing behavior
mixing_prediction = calc.interaction_modeler.predict_mixing_behavior(
    mixture, interactions
)

print(f"Total interaction energy: {mixing_prediction['total_interaction_energy']}")
print(f"Miscibility score: {mixing_prediction['miscibility_score']}")
print(f"Dominant interactions: {mixing_prediction['dominant_interactions']}")
```

## Inverse Design

### Single-Objective Optimization

```python
# Define target properties
target_properties = {
    'molecular_weight_avg': 40.0,
    'density_mixture': 0.85,
    'viscosity_mixture': 1.0
}

# Available components
component_library = [
    "O",           # Water
    "CCO",         # Ethanol
    "CC(=O)C",     # Acetone
    "CCCCCC"       # Hexane
]

# Optimize composition
optimal_composition = calc.design_optimal_mixture(
    target_properties=target_properties,
    component_library=component_library
)

print(f"Optimal composition: {optimal_composition.composition}")
print(f"Feasibility score: {optimal_composition.feasibility_score}")
```

### Multi-Objective Optimization

```python
from chemi_accel.mixture.inverse_design import (
    PropertyTargetObjective, StabilityObjective, CostObjective
)

# Define multiple objectives
objectives = [
    PropertyTargetObjective({'molecular_weight_avg': 40.0}),
    StabilityObjective(['thermal', 'mechanical']),
    CostObjective({'O': 0.1, 'CCO': 1.0, 'CC(=O)C': 2.0})
]

# Find Pareto-optimal solutions
pareto_solutions = calc.inverse_designer.design_pareto_optimal_mixtures(
    objectives=objectives,
    component_library=component_library,
    n_solutions=50
)

print(f"Found {len(pareto_solutions)} Pareto-optimal solutions")
```

### Constraint Handling

```python
from chemi_accel.mixture.inverse_design import (
    CompositionConstraint, PropertyConstraint
)

# Define constraints
constraints = {
    'composition_bounds': {
        'O': (0.1, 0.8),      # Water: 10-80%
        'CCO': (0.0, 0.5)     # Ethanol: max 50%
    },
    'property_bounds': {
        'molecular_weight_avg': (20.0, 60.0)
    }
}

# Optimize with constraints
optimal_composition = calc.design_optimal_mixture(
    target_properties=target_properties,
    component_library=component_library,
    constraints=constraints
)
```

### Adaptive Design

```python
# Adaptive design with experimental feedback
experimental_feedback = [
    {'property_deviations': {'molecular_weight_avg': -2.0}},
    {'property_deviations': {'density_mixture': 0.1}}
]

adaptive_suggestions = calc.inverse_designer.adaptive_design(
    initial_targets=target_properties,
    component_library=component_library,
    experimental_feedback=experimental_feedback,
    n_iterations=3
)
```

## Dynamic Processes

### Process Definition

```python
from chemi_accel.mixture.dynamic_processes import ProcessConditions

# Define time-varying conditions
conditions = ProcessConditions()

# Temperature ramp
conditions.temperature_profile = lambda t: 298.15 + 50 * (t / 3600)

# Constant pressure  
conditions.pressure_profile = lambda t: 101325.0

# Increasing shear rate
conditions.shear_rate_profile = lambda t: 10 * (1 + t / 1800)
```

### Process Simulation

```python
# Simulate crystallization and mixing
process_result = calc.integrate_dynamic_processes(
    mixture=mixture,
    process_conditions=conditions,
    process_types=['crystallization', 'mixing'],
    simulation_time=3600.0,  # 1 hour
    dt=60.0                  # 1 minute steps
)

print(f"Process: {process_result.process_type}")
print(f"Time points: {len(process_result.time_points)}")
print(f"Final properties: {process_result.final_properties}")
```

### Process Models

#### Crystallization
```python
# Crystallization process with nucleation and growth
process_result = calc.integrate_dynamic_processes(
    mixture, conditions, ['crystallization']
)

# Access crystal data
final_microstructure = process_result.microstructure_evolution[-1]
crystal_fraction = final_microstructure.get('crystallization_crystal_fraction', 0)
print(f"Final crystal fraction: {crystal_fraction:.3f}")
```

#### Mixing
```python
# Mixing process with homogenization
process_result = calc.integrate_dynamic_processes(
    mixture, conditions, ['mixing']
)

# Check mixing efficiency
final_homogeneity = process_result.final_properties.get('mixing_homogeneity', 0)
print(f"Final homogeneity: {final_homogeneity:.3f}")
```

#### Phase Separation
```python
# Phase separation with domain growth
process_result = calc.integrate_dynamic_processes(
    mixture, conditions, ['phase_separation']
)

# Analyze separation progress
separation_progress = process_result.final_properties.get('phase_separation_separation_progress', 0)
print(f"Separation progress: {separation_progress:.3f}")
```

### Process Optimization

```python
# Optimize process conditions
target_properties = {
    'crystallization_crystal_fraction': 0.3,
    'mixing_homogeneity': 0.9
}

variable_conditions = {
    'temperature': (280.0, 350.0),
    'shear_rate': (0.0, 100.0)
}

optimal_conditions, simulation_result = calc.process_integrator.optimize_process_conditions(
    mixture=mixture,
    target_properties=target_properties,
    process_types=['crystallization', 'mixing'],
    variable_conditions=variable_conditions
)
```

## Visualization

### Interactive Dashboards

```python
from chemi_accel.mixture import MixtureVisualizer

visualizer = MixtureVisualizer()

# Create comprehensive dashboard
dashboard = visualizer.create_dashboard(
    mixture=mixture,
    phase_equilibrium=phase_result,
    interactions=interactions,
    process_results=process_result
)

print(f"Dashboard with {len(dashboard['panels'])} panels created")
```

### Phase Diagrams

```python
# 2D phase diagram
phase_viz = visualizer.create_phase_diagram(
    phase_result,
    diagram_type='2d',
    interactive=True
)

# 3D phase diagram
phase_viz_3d = visualizer.create_phase_diagram(
    phase_result,
    diagram_type='3d',
    interactive=True
)

# Ternary diagram (for 3-component systems)
phase_viz_ternary = visualizer.create_phase_diagram(
    phase_result,
    diagram_type='ternary',
    interactive=True
)
```

### Property Landscapes

```python
# Generate mixture data across composition space
mixture_data = []
for x1 in np.linspace(0, 1, 20):
    x2 = 1 - x1
    mix = calc.create_mixture([("CCO", x1), ("O", x2)])
    # Add calculated properties...
    mixture_data.append(mix)

# Create property landscape
landscape = visualizer.create_property_landscape(
    mixture_data,
    property_name='density',
    resolution=50
)

# Visualize as contour plot
contour_viz = visualizer.visualize_property_landscape(
    landscape,
    plot_type='contour',
    interactive=True
)
```

### Interaction Networks

```python
# Network visualization of molecular interactions
network_viz = visualizer.visualize_interactions(
    interactions,
    mixture,
    visualization_type='network',
    interactive=True
)

# Interaction matrix heatmap
matrix_viz = visualizer.visualize_interactions(
    interactions,
    mixture,
    visualization_type='matrix',
    interactive=True
)
```

### Process Animations

```python
# Time series animation
animation = visualizer.create_process_animation(
    process_result,
    property_names=['temperature', 'crystal_fraction'],
    animation_type='time_series',
    interactive=True
)

# Composition evolution
comp_animation = visualizer.create_process_animation(
    process_result,
    property_names=[],
    animation_type='composition_evolution',
    interactive=True
)
```

## Advanced Examples

### Pharmaceutical Formulation Design

```python
# Design optimal tablet coating formulation
calc = MixtureCalculator(backend="cuda")

# Polymer coating components
components = [
    "CC(C)(C)OC(=O)C(C)(C)C",  # Methylmethacrylate copolymer
    "CCO",                      # Ethanol (solvent)
    "O",                        # Water (anti-solvent)
    "CC(C)CO"                   # Isopropanol (co-solvent)
]

# Target properties for coating
target_properties = {
    'molecular_weight_avg': 80.0,    # Good film formation
    'viscosity_mixture': 50.0,       # Sprayable consistency
    'surface_tension': 25.0          # Good wetting
}

# Pharmaceutical constraints
constraints = {
    'composition_bounds': {
        'CCO': (0.7, 0.9),  # Primary solvent
        'O': (0.0, 0.2),    # Limited water
    },
    'property_bounds': {
        'viscosity_mixture': (20.0, 100.0)
    }
}

optimal_formulation = calc.design_optimal_mixture(
    target_properties, components, constraints
)
```

### Green Solvent Selection

```python
# Multi-objective green solvent optimization
from chemi_accel.mixture.inverse_design import EnvironmentalObjective

# Green chemistry components
green_solvents = [
    "CCO",        # Ethanol
    "CC(C)CO",    # Isopropanol  
    "CC(=O)C",    # Acetone
    "O"           # Water
]

# Environmental impact scores (lower is better)
environmental_scores = {
    "CCO": 0.2,
    "CC(C)CO": 0.3,
    "CC(=O)C": 0.5,
    "O": 0.1
}

objectives = [
    PropertyTargetObjective({'density_mixture': 0.8}),
    EnvironmentalObjective(environmental_scores)
]

green_solutions = calc.inverse_designer.design_pareto_optimal_mixtures(
    objectives, green_solvents, n_solutions=30
)
```

### Battery Electrolyte Optimization

```python
# Design battery electrolyte mixture
electrolyte_components = [
    "CCOC(=O)C",                    # Ethyl acetate
    "C1COC(=O)O1",                  # Ethylene carbonate
    "CC1COC(=O)O1",                 # Propylene carbonate
    "CCOC(=O)OCC"                   # Diethyl carbonate
]

# Battery performance targets
battery_targets = {
    'ionic_conductivity': 10.0,      # mS/cm
    'electrochemical_window': 4.5,   # V
    'viscosity_mixture': 2.0         # cP
}

# Temperature stability constraints
temp_constraints = {
    'property_bounds': {
        'thermal_stability_temp': (150.0, 300.0)  # °C
    }
}

optimal_electrolyte = calc.design_optimal_mixture(
    battery_targets, electrolyte_components, temp_constraints
)
```

## API Reference

### Core Classes

#### MixtureCalculator
Main interface for mixture analysis and design.

**Methods:**
- `create_mixture(components, temperature, pressure, **kwargs)` - Create mixture from components
- `calculate_mixture_properties(mixture, properties)` - Calculate mixture properties
- `predict_phase_behavior(mixture, **kwargs)` - Predict phase equilibrium  
- `model_interactions(mixture, interaction_types, **kwargs)` - Calculate molecular interactions
- `design_optimal_mixture(target_properties, component_library, constraints, **kwargs)` - Inverse design optimization

#### MixtureData
Data structure representing mixture systems.

**Properties:**
- `n_components` - Number of components
- `smiles_list` - List of SMILES strings
- `composition_vector` - Composition as numpy array
- `components` - List of ComponentData objects
- `temperature` - Mixture temperature (K)
- `pressure` - Mixture pressure (Pa)

#### ComponentData
Data structure for individual mixture components.

**Attributes:**
- `smiles` - SMILES string
- `mol_fraction` - Mole fraction
- `mass_fraction` - Mass fraction (optional)
- `molecular_weight` - Molecular weight (auto-calculated)
- `density` - Component density (optional)
- `properties` - Additional properties dict

### Specialized Classes

#### PhaseBehaviorPredictor
GPU-accelerated phase behavior prediction.

**Methods:**
- `predict_phase_equilibrium(mixture, temperature_range, pressure_range, model, **kwargs)`
- `load_binary_parameters(parameter_file)` - Load NRTL/UNIQUAC parameters
- `calibrate_model(experimental_data, model)` - Calibrate model parameters

#### InteractionModeler
Inter-component interaction modeling system.

**Methods:**
- `calculate_interactions(mixture, interaction_types, **kwargs)`
- `build_interaction_matrix(interactions, n_components)`
- `predict_mixing_behavior(mixture, interactions)`
- `visualize_interactions(interactions, mixture)`

#### MixtureInverseDesigner  
AI-driven inverse design for optimal compositions.

**Methods:**
- `optimize_composition(target_properties, component_library, constraints, **kwargs)`
- `design_pareto_optimal_mixtures(objectives, component_library, **kwargs)`
- `adaptive_design(initial_targets, component_library, experimental_feedback, **kwargs)`

#### DynamicProcessIntegrator
Dynamic process integration for formulations.

**Methods:**
- `simulate_process(mixture, process_conditions, process_types, **kwargs)`
- `optimize_process_conditions(mixture, target_properties, process_types, variable_conditions, **kwargs)`
- `analyze_process_sensitivity(mixture, process_types, base_conditions, sensitivity_parameters, **kwargs)`

#### MixtureVisualizer
Interactive visualization system.

**Methods:**
- `create_dashboard(mixture, phase_equilibrium, interactions, process_results, **kwargs)`
- `create_phase_diagram(phase_equilibrium, diagram_type, interactive, **kwargs)`
- `create_property_landscape(mixture_data, property_name, resolution, **kwargs)`
- `visualize_interactions(interactions, mixture, visualization_type, **kwargs)`
- `create_process_animation(process_results, property_names, animation_type, **kwargs)`

### Data Structures

#### PhaseEquilibrium
Results from phase equilibrium calculations.

**Attributes:**
- `temperature` - System temperature
- `pressure` - System pressure  
- `phases` - List of phase data
- `stability` - Stability metrics
- `n_phases` - Number of phases
- `is_stable` - Boolean stability flag

#### MolecularInteraction
Data structure for molecular interactions.

**Attributes:**
- `component_i` - First component index
- `component_j` - Second component index
- `interaction_type` - Type of interaction
- `strength` - Interaction strength
- `energy` - Interaction energy (kJ/mol)
- `distance` - Interaction distance (Å)

#### OptimalComposition
Results from mixture composition optimization.

**Attributes:**
- `composition` - Optimal composition array
- `properties` - Calculated properties
- `objective_values` - Objective function values
- `feasibility_score` - Constraint satisfaction score
- `components` - Component list
- `pareto_front` - Pareto front data (multi-objective)

This comprehensive guide covers all major aspects of the ChemiAccel Mixture module. For more examples and advanced usage, see the example scripts in the `/examples` directory.